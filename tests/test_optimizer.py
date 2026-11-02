import copy
import pytest
import torch
from tap_merge.config import OptimizerConfig
from tap_merge.optimizer import TAPOptimizer, transport
from tap_merge.linalg import matrix, restore, orth


def small(**kwargs):
    return OptimizerConfig(rank=2, rank_min=1, rank_max=4, rank_increment=1,
                           refresh_every=2, check_every=2, state_budget_ratio=3, **kwargs)


@pytest.mark.parametrize("shape", [(3,), (8, 3), (3, 8), ()])
def test_orientation_round_trip(shape):
    x = torch.arange(max(1, torch.tensor(shape).prod().item() if shape else 1)).reshape(shape).float()
    assert torch.equal(restore(matrix(x), shape), x)


def test_transport_matches_dense_diagonal_proxy():
    torch.manual_seed(4)
    old = torch.linalg.qr(torch.randn(7, 3)).Q
    new = torch.linalg.qr(torch.randn(7, 4)).Q
    m = torch.randn(3, 5) * 0.1
    v = torch.rand(3, 5) * 0.4
    b1, b2, t = 0.9, 0.99, 5
    mt, vt = transport(new, old, m, v, t, b1, b2)
    mh, vh = m / (1 - b1**t), v / (1 - b2**t)
    expected = []
    for j in range(5):
        covariance = old @ torch.diag((vh[:, j] - mh[:, j].square()).clamp_min(0)) @ old.T
        mean = new.T @ (old @ mh[:, j])
        expected.append(torch.diag(new.T @ covariance @ new) + mean.square())
    assert torch.allclose(mt, new.T @ (old @ m), atol=1e-6)
    assert torch.allclose(vt, torch.stack(expected, 1) * (1 - b2**t), atol=1e-6)
    assert (vt >= 0).all()


def test_first_step_and_error_feedback_are_equation_exact():
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.randn(4, 8))
    initial = p.detach().clone()
    g = torch.randn_like(p)
    opt = TAPOptimizer([p], small())
    p.grad = g.clone()
    opt.step()
    s = opt.state[p]
    a = s["U"].T @ g
    expected = initial - opt.config.lr * s["U"] @ (a / (a.abs() + opt.config.eps))
    assert torch.allclose(p, expected, atol=1e-6)
    assert torch.allclose(s["xi"], g - s["U"] @ a, atol=1e-6)


def test_row_column_statistics_exclude_error_feedback():
    p = torch.nn.Parameter(torch.zeros(4, 8))
    opt = TAPOptimizer([p], small())
    p.grad = torch.ones_like(p)
    opt.step()
    opt.state[p]["xi"].fill_(20)
    p.grad = torch.full_like(p, 2)
    opt.step()
    expected = opt.config.beta_c * (1 - opt.config.beta_c) + (1 - opt.config.beta_c) * 4
    assert torch.allclose(opt.state[p]["R"], torch.full((4,), expected))
    assert torch.allclose(opt.state[p]["C"], torch.full((8,), expected))


def test_identity_transport_still_clips_negative_centered_proxy():
    p = torch.nn.Parameter(torch.zeros(8))
    opt = TAPOptimizer([p], small())
    p.grad = torch.ones_like(p)
    opt.step()
    s = opt.state[p]
    s["m"].fill_(0.5)
    s["v"].fill_(0.00001)
    old_mean = s["m"] / (1 - opt.config.beta1)
    expected_half = (1 - opt.config.beta2) * old_mean.square()
    p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.allclose(s["v"], opt.config.beta2 * expected_half)


def test_rotated_basis_error_feedback_preserves_discarded_momentum():
    torch.manual_seed(33)
    p = torch.nn.Parameter(torch.zeros(6, 12))
    opt = TAPOptimizer([p], small())
    p.grad = torch.randn_like(p)
    opt.step()
    s = opt.state[p]
    old_u, old_m, xi = s["U"].clone(), s["m"].clone(), s["xi"].clone()
    p.grad = torch.randn_like(p)
    signal = p.grad + xi
    opt.step()
    u = s["U"]
    half_m = u.T @ old_u @ old_m
    expected = signal - u @ (u.T @ signal) + opt.config.kappa_ef * (old_u @ old_m - u @ half_m)
    assert torch.allclose(s["xi"], expected, atol=2e-5)
    assert torch.allclose(u.T @ u, torch.eye(u.shape[1]), atol=1e-5)


def test_rank_expansion_and_persistent_budget():
    torch.manual_seed(8)
    p = torch.nn.Parameter(torch.zeros(8, 12))
    cfg = small(theta_low=0.99, theta_high=1.0, rho=0)
    opt = TAPOptimizer([p], cfg)
    for _ in range(8):
        p.grad = torch.randn_like(p)
        opt.step()
        assert opt.actual_state_bytes() <= opt.budget_bytes
    assert opt.state[p]["m"].shape[0] > 2


def test_rank_reduction_uses_hysteresis():
    p = torch.nn.Parameter(torch.zeros(8, 12))
    opt = TAPOptimizer([p], small(theta_low=0.1, theta_high=0.5, hysteresis=2, error_feedback=False))
    for t in range(4):
        p.grad = torch.ones_like(p)
        opt.step()
        if t < 3:
            assert opt.state[p]["m"].shape[0] == 2
    assert opt.state[p]["m"].shape[0] == 1


def test_budget_failure_is_explicit_and_vectors_are_global():
    p = torch.nn.Parameter(torch.ones(100))
    with pytest.raises(ValueError, match="Minimum TAP states"):
        TAPOptimizer([p], OptimizerConfig())
    # A sufficiently large low-rank matrix can share the GLOBAL allocation.
    matrix_p = torch.nn.Parameter(torch.ones(100, 500))
    opt = TAPOptimizer([p, matrix_p], OptimizerConfig(rank=8, rank_min=1))
    assert opt.planned_bytes() <= opt.budget_bytes


def test_resume_matches_uninterrupted_updates():
    torch.manual_seed(22)
    p = torch.nn.Parameter(torch.randn(6, 12))
    opt = TAPOptimizer([p], small())
    p.grad = torch.randn_like(p)
    opt.step()
    saved = copy.deepcopy(opt.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    other = TAPOptimizer([q], small())
    other.load_state_dict(saved)
    gradient = torch.randn_like(p)
    rng = torch.get_rng_state()
    p.grad = gradient.clone()
    opt.step()
    torch.set_rng_state(rng)
    q.grad = gradient.clone()
    other.step()
    assert torch.equal(p, q)
    assert torch.equal(opt.state[p]["xi"], other.state[q]["xi"])


def test_orthogonal_completion_and_nonrefresh_stability():
    u = orth(torch.zeros(7, 3), 3)
    assert torch.allclose(u.T @ u, torch.eye(3))
    p = torch.nn.Parameter(torch.zeros(4, 8))
    opt = TAPOptimizer([p], small())
    for _ in range(2):
        p.grad = torch.randn_like(p)
        opt.step()
    before = opt.state[p]["b_bar"]
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt.state[p]["b_bar"] == before


@pytest.mark.parametrize("transport_on,feedback_on", [(False, True), (True, False), (False, False)])
def test_optimizer_ablation_switches(transport_on, feedback_on):
    p = torch.nn.Parameter(torch.zeros(8, 12))
    opt = TAPOptimizer([p], small(state_transport=transport_on, error_feedback=feedback_on))
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()
    assert ("xi" in opt.state[p]) == feedback_on
