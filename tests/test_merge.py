import pytest
import torch
from tap_merge.config import MergeConfig
from tap_merge.merge import decompose, reliability, aggregate
from tap_merge.linalg import top_mask, sparse_encode, sparse_decode, normalize
from tap_merge.baselines import ties, dare_ties


def test_exact_decomposition_with_bounded_nonorthogonal_residual():
    early = torch.tensor([[1.0, 0.0]])
    final = torch.tensor([[3.0, 2.0]])
    de, dl, q, c = decompose(early, final)
    assert c == 1
    assert torch.equal(de + dl, final)
    assert (de * dl).sum() > 0  # The bounded fit need not be orthogonal.
    assert 0 <= q <= 1


def test_antialignment_and_zero_vectors():
    for early in (torch.ones(3, 4), torch.zeros(3, 4)):
        de, dl, q, c = decompose(early, -torch.ones(3, 4))
        assert c == q == 0
        assert de.count_nonzero() == 0


def test_zero_reliability_fallback_and_zero_exponent():
    cfg = MergeConfig(gamma_q=1, gamma_b=0, gamma_f=0)
    raw, weights = reliability(torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.tensor([0.2, 0.8]), cfg)
    assert torch.equal(raw, torch.zeros(2))
    assert torch.equal(weights, torch.tensor([0.2, 0.8]))
    raw, weights = reliability(torch.ones(2), torch.zeros(2), torch.zeros(2), torch.tensor([0.2, 0.8]), cfg)
    assert torch.equal(raw, torch.ones(2))


def test_deterministic_budget_including_ties_and_zero_scores():
    mask = top_mask(torch.zeros(2, 5), 0.21)
    assert mask.sum() == 3
    assert mask.flatten().tolist() == [True] * 3 + [False] * 7
    assert not top_mask(torch.ones(2, 5), 0).any()
    assert top_mask(torch.ones(2, 5), 1).all()


def test_sparse_residual_reconstructs_exactly_at_full_density():
    value = torch.randn(6, 8)
    assert torch.equal(value, sparse_decode(sparse_encode(value, 1)))


def test_closed_form_is_stationary_and_obeys_shrinkage_bound():
    torch.manual_seed(5)
    d = torch.rand(3, 4, 5) + 0.1
    j, h = torch.ones_like(d), torch.rand_like(d) + 0.1
    weights = torch.tensor([0.2, 0.3, 0.5])
    masks = torch.ones_like(d, dtype=torch.bool)
    delta, total, _ = aggregate(d, j, h, masks, weights, 1, 0.1)
    x = delta.clone().requires_grad_()
    a = weights[:, None, None] * h
    loss = (a * (x - d).square()).sum() + 0.1 * x.square().sum()
    grad = torch.autograd.grad(loss, x)[0]
    assert grad.abs().max() < 1e-6
    assert (delta.abs() <= total / (0.1 + total) * d.abs().amax(0) + 1e-6).all()


def test_balanced_support_and_zero_support():
    d = torch.tensor([[[1.0]], [[-1.0]]])
    ones = torch.ones_like(d)
    delta, _, confidence = aggregate(d, ones, ones, ones, torch.tensor([0.5, 0.5]), 2, 0.1)
    assert delta == confidence == 0
    delta, _, _ = aggregate(d, ones, ones, torch.zeros_like(d), torch.tensor([0.5, 0.5]), 2, 0.1)
    assert delta == 0


def test_confident_conflict_is_attenuated_more_for_late_stage():
    d = torch.tensor([[[2.0]], [[-1.0]]])
    ones = torch.ones_like(d)
    early, _, _ = aggregate(d, ones, ones, ones, torch.tensor([0.5, 0.5]), 1, 0.1)
    late, _, _ = aggregate(d, ones, ones, ones, torch.tensor([0.5, 0.5]), 2, 0.1)
    assert late > early


def test_normalization_median_positive_and_all_zero():
    assert normalize(torch.zeros(4)).count_nonzero() == 0
    assert torch.allclose(normalize(torch.tensor([0.0, 1, 3, 5, 7])), torch.tensor([0.0, 1/3, 1, 5/3, 7/3]))


def test_baseline_election_and_dare_reproducibility():
    ds = [torch.tensor([2.0, -1.0]), torch.tensor([-1.0, -3.0])]
    assert torch.equal(ties(ds, 1), torch.tensor([2.0, -2.0]))
    assert torch.equal(dare_ties(ds, seed=4), dare_ties(ds, seed=4))


@pytest.mark.parametrize("kwargs", [{"c_max": 2}, {"gamma_q": 0.8}, {"lambda_late": 0}, {"late_max": 0.9}])
def test_invalid_merge_settings(kwargs):
    with pytest.raises(ValueError):
        MergeConfig(**kwargs)
