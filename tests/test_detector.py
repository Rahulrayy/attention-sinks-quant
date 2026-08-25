"""Day 2 gate — plan §11. The detector must agree with attention.

These tests build synthetic residual-stream norms with a KNOWN sink structure,
including the multi-level case that the naive position_0 rule misses, and
assert the detector recovers it. The validation tests then check that the gate
actually fires when magnitude and attention disagree — a validator that never
raises is worth nothing.
"""

import pytest

torch = pytest.importorskip("torch")

from sinks.detector import (  # noqa: E402
    DetectorValidationError,
    detect_sinks,
    sweep_tau,
    validate_against_attention,
)


def make_norms(sink_positions, *, batch=2, seq=64, base=1.0, spike=200.0):
    """Flat magnitude everywhere except a few known sink positions."""
    m = torch.full((batch, seq), base)
    for p in sink_positions:
        m[:, p] = spike
    return m


# --- detection ---------------------------------------------------------------

def test_recovers_position_0():
    mask = detect_sinks(make_norms([0]), tau=50.0)
    assert mask[:, 0].all()
    assert mask.sum().item() == mask.shape[0]


def test_recovers_multiple_sink_levels():
    """The case the naive position_0 rule gets wrong — six sinks, as in Qwen3-14B."""
    positions = [0, 5, 17, 31, 48, 60]
    mask = detect_sinks(make_norms(positions), tau=50.0)
    found = torch.nonzero(mask.any(dim=0)).flatten().tolist()
    assert found == positions


def test_flat_input_flags_nothing():
    """No sink structure means no flags. A detector that always fires is useless."""
    assert not detect_sinks(torch.ones(2, 64), tau=10.0).any()


def test_median_is_per_sequence_not_batch_wide():
    """One high-magnitude sequence must not suppress detection in another.

    Sequence 0 sits at 1000x the scale of sequence 1. A batch-wide median would
    put the threshold above sequence 1's spike entirely and miss it.
    """
    m = torch.empty(2, 64)
    m[0] = 1000.0
    m[0, 0] = 200_000.0
    m[1] = 1.0
    m[1, 0] = 200.0
    mask = detect_sinks(m, tau=50.0)
    assert mask[0, 0] and mask[1, 0], "both sequences' sinks must be found"


def test_tau_below_one_is_rejected():
    with pytest.raises(ValueError):
        detect_sinks(make_norms([0]), tau=0.5)


# --- tau sweep ---------------------------------------------------------------

def test_sweep_is_monotone_in_tau():
    """Raising tau can only ever flag fewer tokens."""
    m = torch.rand(2, 64) * 10.0
    m[:, 0] = 5000.0
    m[:, 20] = 300.0
    counts = [r.n_flagged for r in sweep_tau(m, [10, 20, 50, 100]).values()]
    assert counts == sorted(counts, reverse=True), counts


def test_sweep_reports_every_tau():
    taus = [10, 20, 50, 100]
    assert list(sweep_tau(make_norms([0]), taus).keys()) == taus


# --- validation gate ---------------------------------------------------------

def make_received(high_positions, *, batch=2, seq=64, low=0.001, high=0.5):
    r = torch.full((batch, seq), low)
    for p in high_positions:
        r[:, p] = high
    return r


def test_validation_passes_when_magnitude_and_attention_agree():
    positions = [0, 17]
    report = validate_against_attention(
        detect_sinks(make_norms(positions), tau=50.0),
        make_received(positions),
    )
    assert report["position_0_recall"] == 1.0
    assert report["attention_agreement"] == 1.0


def test_validation_fails_when_flagged_token_gets_no_attention():
    """A magnitude outlier nothing attends to is not a sink.

    THIS is the failure the gate exists for: position 17 spikes in the residual
    stream but no head attends to it. Letting it into the fp16 exception list
    would inflate D_sink with a token that has nothing to do with the sink.
    """
    mask = detect_sinks(make_norms([0, 17]), tau=50.0)
    received = make_received([0])          # 17 deliberately left low
    with pytest.raises(DetectorValidationError, match="received-attention threshold"):
        validate_against_attention(mask, received)


def test_validation_fails_when_position_0_is_missed():
    mask = detect_sinks(make_norms([17]), tau=50.0)
    with pytest.raises(DetectorValidationError, match="position 0"):
        validate_against_attention(mask, make_received([17]))


def test_validation_fails_when_nothing_is_flagged():
    with pytest.raises(DetectorValidationError, match="no tokens"):
        validate_against_attention(torch.zeros(2, 64, dtype=torch.bool), make_received([0]))


# --- layer-relative detector (the primary one) -------------------------------

from sinks.detector import (  # noqa: E402
    detect_sinks_layerwise,
    layer_relative_norms,
    sweep_tau_layerwise,
)


def make_per_layer(sink_positions, *, layers=6, batch=2, seq=64, spike=200.0):
    """Layers on wildly different activation scales, each with the same sink.

    The scale spread is the point: it is what makes the aggregate detector fail,
    because one loud layer sets the median for all of them.
    """
    m = torch.empty(layers, batch, seq)
    for l in range(layers):
        base = 10.0 ** l                     # layer scales span 1 .. 1e5
        m[l] = base
        for p in sink_positions:
            m[l, :, p] = base * spike
    return m


def test_layer_relative_recovers_a_sink_the_aggregate_form_hides():
    """The Qwen3-0.6B finding, reduced to its mechanism.

    The aggregate detector fails when a LOUD layer without a sink masks a QUIET
    layer that has one. Maxing over layers before taking the median lets the
    loud layer set both the peak and the median, so the quiet layer's sink
    vanishes into a ratio of ~1. Giving every layer the same relative sink would
    NOT reproduce this — both forms handle that case identically.
    """
    quiet = torch.full((2, 64), 1.0)
    quiet[:, 0] = 1000.0                     # a 1000x sink, on a quiet layer
    loud = torch.full((2, 64), 5000.0)       # louder everywhere, no sink at all
    per_layer = torch.stack([quiet, loud])

    agg = per_layer.amax(dim=0)
    agg_ratio = (agg.max() / agg.median()).item()
    rel_ratio = layer_relative_norms(per_layer).max().item()

    assert agg_ratio < 2.0, f"aggregate should be blind here, got {agg_ratio:.1f}x"
    assert rel_ratio > 500.0, f"layer-relative should see it, got {rel_ratio:.1f}x"

    assert not detect_sinks(agg.unsqueeze(0) if agg.dim() == 1 else agg, tau=5.0).any()
    assert detect_sinks_layerwise(per_layer, tau=5.0)[:, 0].all()


def test_layer_relative_threshold_is_absolute():
    """Input is already a ratio to each layer's median, so tau applies directly."""
    mask = detect_sinks_layerwise(make_per_layer([0, 17], spike=200.0), tau=50.0)
    assert torch.nonzero(mask.any(dim=0)).flatten().tolist() == [0, 17]


def test_layer_relative_stable_across_a_wide_tau_range():
    """Width of the stable range is the criterion for preferring this form."""
    per_layer = make_per_layer([0], spike=1000.0)
    found = {
        tau: r.positions for tau, r in sweep_tau_layerwise(per_layer, [5, 10, 20, 50, 100]).items()
    }
    assert all(p == [0] for p in found.values()), found


def test_layer_relative_flags_nothing_without_a_sink():
    flat = torch.stack([torch.full((2, 64), 10.0**l) for l in range(6)])
    assert not detect_sinks_layerwise(flat, tau=5.0).any()


def test_layer_relative_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"expected \(L, B, T\)"):
        layer_relative_norms(torch.rand(2, 64))


def test_layer_relative_rejects_tau_below_one():
    with pytest.raises(ValueError, match="tau must exceed"):
        detect_sinks_layerwise(make_per_layer([0]), tau=0.5)
