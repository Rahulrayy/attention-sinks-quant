"""Metric definitions are locked before data (plan §6). These tests pin them.

Each metric is checked against a case where the correct answer is known
analytically, so that a later refactor cannot quietly change what "sink mass"
or "entropy" means partway through the project.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from sinks.metrics import (  # noqa: E402
    aggregate_inf_norm,
    excess_kurtosis,
    fraction_heads_sinking,
    head_entropy,
    outlier_channels,
    received_attention,
    residual_inf_norm,
    sink_mass,
)

B, H, T, C = 2, 4, 16, 32


def causal_mask_probs(weights):
    """Row-normalise under a causal mask so rows are valid distributions."""
    mask = torch.tril(torch.ones(T, T))
    w = weights * mask
    return w / w.sum(dim=-1, keepdim=True)


# --- sink mass ---------------------------------------------------------------

def test_total_sink_is_one():
    """Every query q>0 puts all its mass on position 0."""
    w = torch.zeros(B, H, T, T)
    w[..., 0] = 1.0
    probs = causal_mask_probs(w)
    assert torch.allclose(sink_mass(probs), torch.ones(B, H))


def test_no_sink_is_near_zero():
    """Attention on the diagonal only — nothing reaches position 0 from q>0."""
    probs = torch.diag_embed(torch.ones(B, H, T))
    assert sink_mass(probs).max().item() == pytest.approx(0.0)


def test_query_zero_is_excluded():
    """Query 0 can only attend to itself under a causal mask, so its sink mass
    is identically 1 and would bias every head upward by 1/T for reasons that
    have nothing to do with the phenomenon."""
    probs = torch.diag_embed(torch.ones(B, H, T))
    assert sink_mass(probs, exclude_self=True).max().item() == pytest.approx(0.0)
    assert sink_mass(probs, exclude_self=False).max().item() == pytest.approx(1.0 / T)


def test_returns_full_per_head_distribution():
    """Shape (B, H) — never pre-reduced. The distribution is bimodal and a mean
    over heads hides exactly the structure under investigation."""
    w = torch.rand(B, H, T, T)
    assert sink_mass(causal_mask_probs(w)).shape == (B, H)


def test_fraction_heads_sinking():
    masses = torch.tensor([[0.9, 0.8, 0.1, 0.05]])
    assert fraction_heads_sinking(masses, threshold=0.5) == pytest.approx(0.5)


# --- entropy -----------------------------------------------------------------

def test_entropy_of_one_hot_is_zero():
    probs = torch.zeros(B, H, T, T)
    probs[..., 0] = 1.0
    assert head_entropy(probs).abs().max().item() == pytest.approx(0.0, abs=1e-5)


def test_entropy_of_uniform_is_log_t():
    probs = torch.full((B, H, T, T), 1.0 / T)
    assert head_entropy(probs).mean().item() == pytest.approx(math.log(T), abs=1e-5)


def test_entropy_separates_sinking_from_diffuse():
    """The metric exists to tell these two apart; if it cannot, drop it."""
    sinking = torch.zeros(B, H, T, T)
    sinking[..., 0] = 1.0
    diffuse = torch.full((B, H, T, T), 1.0 / T)
    assert head_entropy(sinking).mean() < head_entropy(diffuse).mean()


# --- massive activations -----------------------------------------------------

def test_residual_inf_norm():
    x = torch.zeros(B, T, C)
    x[0, 3, 7] = -42.0
    assert residual_inf_norm(x)[0, 3].item() == pytest.approx(42.0)


def test_aggregate_takes_max_over_layers():
    per_layer = torch.zeros(5, B, T)
    per_layer[2, 0, 1] = 9.0
    per_layer[4, 0, 1] = 3.0
    assert aggregate_inf_norm(per_layer)[0, 1].item() == pytest.approx(9.0)


def test_outlier_channels_flags_only_the_outliers():
    x = torch.ones(B, T, C)
    x[0, 0, 5] = 500.0          # 500x the median channel max
    x[0, 0, 11] = 50.0          # below the 100x threshold
    flagged = outlier_channels(x, threshold=100.0)
    assert flagged[5] and not flagged[11]
    assert flagged.sum().item() == 1


def test_outlier_threshold_is_relative_not_absolute():
    """Scaling the whole tensor must not change which channels are flagged —
    activation scale varies by model and layer."""
    x = torch.ones(B, T, C)
    x[0, 0, 5] = 500.0
    assert torch.equal(outlier_channels(x), outlier_channels(x * 1e6))


# --- kurtosis ----------------------------------------------------------------

def test_gaussian_excess_kurtosis_is_near_zero():
    """Excess (Fisher) kurtosis, so numbers are comparable to Bondarenko et al.
    rather than off by 3."""
    torch.manual_seed(0)
    assert excess_kurtosis(torch.randn(100_000)) == pytest.approx(0.0, abs=0.15)


def test_outliers_raise_kurtosis():
    torch.manual_seed(0)
    x = torch.randn(10_000)
    spiked = x.clone()
    spiked[:5] = 100.0
    assert excess_kurtosis(spiked) > excess_kurtosis(x) + 10


# --- received attention ------------------------------------------------------

def test_received_attention_finds_the_sink():
    w = torch.zeros(B, H, T, T)
    w[..., 0] = 1.0
    r = received_attention(causal_mask_probs(w))
    assert r.shape == (B, T)
    assert r[:, 0].min().item() > 0.9
    assert r[:, 1:].max().item() == pytest.approx(0.0)
