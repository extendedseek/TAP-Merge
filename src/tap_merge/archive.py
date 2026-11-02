"""Versioned, block-addressable task packages and precision reconstruction."""
import hashlib
import json
from pathlib import Path
import torch
from .linalg import matrix, restore, normalize, sparse_encode, sparse_decode

SCHEMA = 1


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(v) for v in value]
    return value


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def safe_load(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def fingerprint(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        x = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((list(x.shape), str(x.dtype))).encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def block_id(name):
    return hashlib.sha256(name.encode()).hexdigest()[:24]


def parameter_layout(model):
    canonical, aliases, seen = {}, {}, {}
    for name, p in model.named_parameters(remove_duplicate=False):
        if id(p) in seen:
            aliases[name] = seen[id(p)]
        else:
            seen[id(p)] = name
            canonical[name] = p
    return canonical, aliases


def precision(stats, lambda_rc=0.5, lambda_ld=0.5, eps=1e-8):
    r, c, u, vh = (stats[k].float() for k in ("R", "C", "U", "v_hat"))
    rc = r[:, None] * c[None, :] / ((r.mean() * c.mean()).sqrt() + eps)
    ld = u.square() @ vh
    return lambda_rc * normalize(rc, eps) + lambda_ld * normalize(ld, eps) + eps


def checkpoint_stats(p, optimizer):
    s = optimizer.state[p]
    if not s or not s["step"]:
        raise ValueError("Every trainable block needs a gradient before archiving")
    u = s.get("U", torch.ones((1, 1), device=p.device))
    return cpu_tree(dict(U=u, v_hat=s["v"] / (1 - optimizer.config.beta2 ** s["step"]),
                         R=s["R"], C=s["C"], b_bar=s["b_bar"], f_bar=s["f_bar"], step=s["step"]))


@torch.no_grad()
def save_early(model, optimizer, base_state, directory, p_early=0.01):
    directory = Path(directory)
    canonical, _ = parameter_layout(model)
    for name, p in canonical.items():
        stats = checkpoint_stats(p, optimizer)
        delta = matrix(p.detach().cpu() - base_state[name])
        z = stats["U"].T @ delta
        residual = delta - stats["U"] @ z
        stats.update(Z=z, Q=sparse_encode(residual, p_early),
                     saliency=delta.square() * precision(stats).sqrt())
        atomic_save(stats, directory / "early" / (block_id(name) + ".pt"))


@torch.no_grad()
def save_task(model, optimizer, base_state, directory, metadata):
    directory = Path(directory)
    canonical, aliases = parameter_layout(model)
    blocks = {}
    for name, p in canonical.items():
        key = block_id(name)
        early = directory / "early" / (key + ".pt")
        if not early.exists():
            raise ValueError("Missing early checkpoint; choose early_tokens <= training budget")
        stats = checkpoint_stats(p, optimizer)
        final = p.detach().cpu().clone()
        delta = matrix(final - base_state[name])
        stats.update(final=final, saliency=delta.square() * precision(stats).sqrt())
        atomic_save(stats, directory / "final" / (key + ".pt"))
        blocks[name] = {"id": key, "shape": list(p.shape)}
    # Reject models with changed buffers: this implementation merges parameter deltas only.
    for name, buf in model.named_buffers():
        if name in base_state and not torch.equal(buf.detach().cpu(), base_state[name]):
            raise ValueError(f"Nonparameter buffer {name} changed during adaptation")
    manifest = dict(schema=SCHEMA, base_fingerprint=fingerprint(base_state),
                    blocks=blocks, aliases=aliases, metadata=metadata)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class TaskArchive:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("Unsupported task-package schema")

    def load(self, name):
        key = self.manifest["blocks"][name]["id"]
        return (safe_load(self.directory / "early" / (key + ".pt")),
                safe_load(self.directory / "final" / (key + ".pt")))

    def export(self, base_state):
        if fingerprint(base_state) != self.manifest["base_fingerprint"]:
            raise ValueError("Base checkpoint mismatch")
        state = dict(base_state)
        for name in self.manifest["blocks"]:
            _, f = self.load(name)
            state[name] = f["final"]
        for alias, canonical in self.manifest["aliases"].items():
            state[alias] = state[canonical]
        return state


def reconstructed_early(early):
    return early["U"] @ early["Z"] + sparse_decode(early["Q"])


def restore_delta(delta, shape):
    return restore(delta, shape)
