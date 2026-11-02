"""Practical TAP optimizer corresponding to Proposed Framework equations.

The persistent budget excludes parameters/gradients/temporary workspaces and
includes bases, projected moments, row/column statistics and dense EF buffers.
No gradient-buffer aliasing, quantization, offload, or convergence claim is made.
"""
from dataclasses import asdict
import torch
from torch.optim import Optimizer
from .config import OptimizerConfig
from .linalg import matrix, restore, orth


def transport(u_new, u_old, m_old, v_old, step_old, beta1, beta2):
    t = u_new.T @ u_old
    m_hat = m_old / (1 - beta1 ** step_old)
    v_hat = v_old / (1 - beta2 ** step_old)
    centered = (v_hat - m_hat.square()).clamp_min(0)
    mean = t @ m_hat
    variance = (t.square() @ centered + mean.square()).clamp_min(0)
    return t @ m_old, (1 - beta2 ** step_old) * variance


class TAPOptimizer(Optimizer):
    def __init__(self, params, config=None):
        self.config = config or OptimizerConfig()
        c = self.config
        super().__init__(params, dict(lr=c.lr))
        self.blocks = [p for group in self.param_groups for p in group["params"] if p.requires_grad]
        if len({id(p) for p in self.blocks}) != len(self.blocks):
            raise ValueError("Tied parameters must be passed only once")
        if any(p.dtype != torch.float32 for p in self.blocks):
            raise ValueError("Keep parameters FP32; use BF16 autocast for model operations")
        self.budget_bytes = int(2 * 4 * sum(p.numel() for p in self.blocks) * c.state_budget_ratio)
        self.planned = {id(p): min(c.rank, matrix(p).shape[0]) for p in self.blocks}
        # Reduce the largest planned allocation until it fits, never below r_min.
        while self.planned_bytes() > self.budget_bytes:
            eligible = [p for p in self.blocks if self.planned[id(p)] > min(c.rank_min, matrix(p).shape[0])]
            if not eligible:
                raise ValueError(f"Minimum TAP states need {self.planned_bytes()} bytes; budget={self.budget_bytes}. "
                                 "Increase state_budget_ratio or lower rank_min; dense EF counts toward budget.")
            p = max(eligible, key=lambda x: self._size(x, self.planned[id(x)]))
            self.planned[id(p)] -= 1

    def _size(self, p, rank):
        n, m = matrix(p).shape
        # Vector U=1 is implicit; no full-rank vector error buffer is necessary.
        return 4 * ((n * rank if n > 1 else 0) + 2 * rank * m + n + m
                    + (n * m if self.config.error_feedback and n > 1 else 0))

    def planned_bytes(self):
        return sum(self._size(p, self.planned[id(p)]) for p in self.blocks)

    def actual_state_bytes(self):
        return sum(v.numel() * v.element_size() for s in self.state.values()
                   for v in s.values() if isinstance(v, torch.Tensor))

    def _init(self, p, g):
        c = self.config
        n, m = g.shape
        r = self.planned[id(p)]
        s = self.state[p]
        s.update(step=0, high_checks=0, b_bar=1.0, f_bar=1.0, nu=0.0, rank_sum=0.0)
        if n > 1:
            s["U"] = torch.linalg.svd(g, full_matrices=False).U[:, :r].contiguous()
        s["m"] = torch.zeros((r, m), device=p.device)
        s["v"] = torch.zeros_like(s["m"])
        s["R"] = torch.zeros(n, device=p.device)
        s["C"] = torch.zeros(m, device=p.device)
        if c.error_feedback and n > 1:
            s["xi"] = torch.zeros_like(g)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        c = self.config
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError("Sparse gradients are unsupported")
                g = matrix(p.grad.detach()).float()
                if not torch.isfinite(g).all():
                    raise FloatingPointError("Non-finite task gradient")
                if not self.state[p]:
                    self._init(p, g)
                s = self.state[p]
                t = s["step"] + 1
                n, m = g.shape
                old_u = s.get("U", torch.ones((1, 1), device=p.device))
                old_m, old_v = s["m"], s["v"]
                a_full = g + s["xi"] if "xi" in s else g
                u = old_u
                refreshed = False
                if t > 1 and n > 1 and c.refresh_basis and t % c.refresh_every == 0:
                    mh = old_m / (1 - c.beta1 ** (t - 1))
                    # B is a linear operator; never materialize another dense B or B B^T.
                    def b_mm(x):
                        return c.rho * (old_u @ (mh @ x)) + (1 - c.rho) * (a_full @ x)
                    def bt_mm(x):
                        return c.rho * (mh.T @ (old_u.T @ x)) + (1 - c.rho) * (a_full.T @ x)
                    old_proj = old_u.T @ a_full
                    denom = (c.rho ** 2 * mh.square().sum()
                             + (1 - c.rho) ** 2 * a_full.square().sum()
                             + 2 * c.rho * (1 - c.rho) * (mh * old_proj).sum()).clamp_min(0)
                    y = c.rho * mh + (1 - c.rho) * old_proj
                    energy = float((y.square().sum() / (denom + c.eps)).clamp(0, 1))
                    r_old = old_u.shape[1]
                    r = r_old
                    if c.adaptive_rank and t % c.check_every == 0:
                        if energy < c.theta_low:
                            r = min(r_old + c.rank_increment, c.rank_max, n)
                            s["high_checks"] = 0
                        elif energy > c.theta_high:
                            s["high_checks"] += 1
                            if s["high_checks"] >= c.hysteresis:
                                r = max(r_old - c.rank_increment, min(c.rank_min, n))
                                s["high_checks"] = 0
                        else:
                            s["high_checks"] = 0
                    while r > r_old and self.planned_bytes() - self._size(p, r_old) + self._size(p, r) > self.budget_bytes:
                        r -= 1
                    if r < r_old:
                        _, eigvec = torch.linalg.eigh(y @ y.T)
                        q = old_u @ eigvec[:, -r:]
                    elif r > r_old:
                        probe = torch.randn((m, r - r_old), device=p.device)
                        v = b_mm(probe)
                        v = v - old_u @ (old_u.T @ v)
                        q = orth(torch.cat([old_u, v], dim=1), r)
                    else:
                        q = old_u
                    for _ in range(c.power_steps):
                        q = orth(b_mm(bt_mm(q)), r)
                    u = q
                    refreshed = True
                    self.planned[id(p)] = r
                    overlap = float(((u.T @ old_u).square().sum() / min(r, r_old)).clamp(0, 1))
                    s["b_bar"] = c.beta_b * s["b_bar"] + (1 - c.beta_b) * overlap
                if refreshed and c.state_transport:
                    half_m, half_v = transport(u, old_u, old_m, old_v, t - 1, c.beta1, c.beta2)
                elif refreshed:
                    # Defined ablation: retain coordinate slots, zero-pad/truncate when ranks change.
                    half_m = torch.zeros((u.shape[1], m), device=p.device)
                    half_v = torch.zeros_like(half_m)
                    overlap_r = min(half_m.shape[0], old_m.shape[0])
                    half_m[:overlap_r] = old_m[:overlap_r]
                    half_v[:overlap_r] = old_v[:overlap_r]
                elif t > 1 and c.state_transport:
                    # T=I still applies centered-proxy clipping when beta1/beta2 differ.
                    # Algebraic identity shortcut avoids needless U^T U multiplications.
                    half_m = old_m
                    old_mean = old_m / (1 - c.beta1 ** (t - 1))
                    half_v = torch.maximum(old_v, (1 - c.beta2 ** (t - 1)) * old_mean.square())
                else:
                    half_m, half_v = old_m, old_v
                a = u.T @ a_full
                # Statistics use CLIPPED TASK gradients, before adding xi.
                g2 = g.square()
                s["R"].mul_(c.beta_c).add_(g2.mean(dim=1), alpha=1 - c.beta_c)
                s["C"].mul_(c.beta_c).add_(g2.mean(dim=0), alpha=1 - c.beta_c)
                fidelity = float((a.square().sum() / (a_full.square().sum() + c.eps)).clamp(0, 1))
                s["f_bar"] = c.beta_f * s["f_bar"] + (1 - c.beta_f) * fidelity
                if "xi" in s:
                    residual = a_full - u @ a
                    if t > 1:
                        residual = residual + c.kappa_ef * (old_u @ old_m - u @ half_m)
                    s["xi"].copy_(residual)
                new_m = c.beta1 * half_m + (1 - c.beta1) * a
                new_v = c.beta2 * half_v + (1 - c.beta2) * a.square()
                mh = new_m / (1 - c.beta1 ** t)
                vh = new_v / (1 - c.beta2 ** t)
                if c.analytical_direction:
                    prev_nu = s["nu"]
                    s["nu"] = max(prev_nu, float(vh.max()))
                    vh = vh.clamp_min(prev_nu)
                direction = u @ (mh / (vh.sqrt() + c.eps))
                p.add_(restore(direction, p.shape), alpha=-group["lr"])
                s.update(m=new_m, v=new_v, step=t, rank_sum=s["rank_sum"] + u.shape[1])
                if n > 1:
                    s["U"] = u
        return loss

    def state_dict(self):
        value = super().state_dict()
        value["tap_config"] = asdict(self.config)
        return value

    def load_state_dict(self, state_dict):
        value = dict(state_dict)
        saved = value.pop("tap_config", None)
        if saved is not None and saved != asdict(self.config):
            raise ValueError("Resume optimizer configuration differs from saved configuration")
        super().load_state_dict(value)
        for p in self.blocks:
            if self.state[p]:
                self.planned[id(p)] = self.state[p]["m"].shape[0]
        if self.planned_bytes() > self.budget_bytes:
            raise ValueError("Loaded states exceed budget")
