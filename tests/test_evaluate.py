"""The headline metric, and the decomposition that keeps it honest.

D_sink sums two effects: the sink token's own prediction being repaired (near
tautological, present under every granularity) and the contamination of every
other token being lifted (the effect the literature actually claims). These
tests pin the split, using synthetic NLL where the correct answer is known.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from quant.evaluate import (  # noqa: E402
    d_sink,
    d_sink_decomposed,
    delta_nll,
    delta_ppl,
    perplexity,
)

B, T = 2, 16
SINKS = [0]


def nll(base=1.0, *, at=None, value=None):
    x = torch.full((B, T), base)
    if at is not None:
        x[:, at] = value
    return x


def test_perplexity_is_exp_mean_nll():
    assert perplexity(torch.full((2, 8), 2.0)) == pytest.approx(math.exp(2.0))


def test_perplexity_rejects_empty():
    with pytest.raises(ValueError, match="no tokens"):
        perplexity(torch.empty(2, 0))


def test_delta_ppl_is_zero_for_identical_runs():
    x = nll()
    assert delta_ppl(x, x) == pytest.approx(0.0)


def test_d_sink_matches_the_plan_definition():
    assert d_sink(0.5, 0.2) == pytest.approx(0.3)


# --- the decomposition -------------------------------------------------------

def test_pure_self_effect_has_zero_contamination_share():
    """Damage confined to the sink token itself.

    This is the per_token case: exempting position 0 repairs position 0 and
    nothing else. The summed metric is large, but the effect under audit is
    exactly zero — and reporting only the total would hide that.
    """
    ref = nll(1.0)
    none = nll(1.0, at=0, value=9.0)      # only the sink is damaged
    exempt = nll(1.0)                     # exempting it restores the reference

    out = d_sink_decomposed(ref, none, exempt, SINKS)
    assert out["d_sink_total"] > 0
    assert out["d_sink_non_sink_nats"] == pytest.approx(0.0)
    assert out["contamination_share"] == pytest.approx(0.0)


def test_pure_contamination_is_fully_attributed():
    """Damage confined to the NON-sink tokens — the per_tensor mechanism."""
    ref = nll(1.0)
    none = nll(2.0, at=0, value=1.0)      # every token but the sink is damaged
    exempt = nll(1.0)

    out = d_sink_decomposed(ref, none, exempt, SINKS)
    assert out["d_sink_non_sink_nats"] > 0
    assert out["d_sink_at_sink_nats"] == pytest.approx(0.0)
    assert out["contamination_share"] == pytest.approx(1.0)


def test_mixed_case_splits_between_the_two():
    ref = nll(1.0)
    none = nll(2.0, at=0, value=9.0)      # both effects present
    exempt = nll(1.0)

    out = d_sink_decomposed(ref, none, exempt, SINKS)
    assert out["d_sink_non_sink_nats"] > 0
    assert out["d_sink_at_sink_nats"] > 0
    assert 0.0 < out["contamination_share"] < 1.0


def test_no_damage_anywhere_gives_zero():
    ref = nll(1.0)
    out = d_sink_decomposed(ref, ref, ref, SINKS)
    assert out["d_sink_total"] == pytest.approx(0.0)
    assert out["contamination_share"] == pytest.approx(0.0)


def test_multiple_sink_levels_are_all_excluded():
    """Qwen-style multi-level sinks, not just position 0."""
    sinks = [0, 5, 11]
    ref = nll(1.0)
    none = ref.clone()
    none[:, sinks] = 9.0
    out = d_sink_decomposed(ref, none, ref, sinks)
    assert out["d_sink_non_sink_nats"] == pytest.approx(0.0)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shapes must match"):
        d_sink_decomposed(nll(), nll(), torch.ones(B, T + 1), SINKS)


def test_empty_sink_list_is_rejected():
    with pytest.raises(ValueError, match="undefined"):
        d_sink_decomposed(nll(), nll(), nll(), [])


def test_decomposition_is_exactly_additive():
    """The reason the split is done in nats and not in perplexity units.

    Count-weighted parts must reconstruct the total exactly. The perplexity
    version of this identity does not hold — exp(mean(.)) is not additive over
    token subsets — and an earlier draft of this metric reported a
    contamination share of 1.106 as a result.
    """
    ref = nll(1.0)
    none = nll(2.0, at=0, value=9.0)
    exempt = nll(1.2, at=0, value=1.0)

    out = d_sink_decomposed(ref, none, exempt, SINKS)
    n_tot, n_s = out["n_total_positions"], out["n_sink_positions"]
    reconstructed = (
        (n_tot - n_s) / n_tot * out["d_sink_non_sink_nats"]
        + n_s / n_tot * out["d_sink_at_sink_nats"]
    )
    assert reconstructed == pytest.approx(out["d_sink_total_nats"], rel=1e-12)


def test_perplexity_and_nats_agree_on_sign():
    """The plan-§6 number and the log-space number must never disagree in
    direction, or the two reported metrics would tell opposite stories."""
    ref = nll(1.0)
    none = nll(2.0, at=0, value=9.0)
    exempt = nll(1.0)
    out = d_sink_decomposed(ref, none, exempt, SINKS)
    assert (out["d_sink_total"] > 0) == (out["d_sink_total_nats"] > 0)


def test_delta_nll_is_zero_for_identical_runs():
    x = nll()
    assert delta_nll(x, x) == pytest.approx(0.0)


# --- the outlier_channels mask, derived rather than stored --------------------
#
# `sinks.measure` records the raw statistic (per-layer channel_max) and this
# derives the mask from it at read time. The alternative -- writing a mask into
# the measurement JSON -- would put a derived quantity beside the thing it comes
# from, free to disagree with it the moment the threshold moves.

from quant.evaluate import outlier_mask_from_sinks  # noqa: E402


def sinks_json(rows):
    return {"residual": [{"name": f"model.layers.{i}", "channel_max": r}
                         for i, r in enumerate(rows)]}


def test_the_mask_comes_from_the_recorded_per_layer_maxima():
    rows = [[1.0] * 8, [1.0] * 8]
    rows[1][3] = 500.0
    mask = outlier_mask_from_sinks(sinks_json(rows), threshold=100.0)
    assert mask.shape == (8,)
    assert mask[3] and mask.sum().item() == 1


def test_the_threshold_belongs_to_the_arm_not_the_measurement():
    """Same JSON, two thresholds, two masks. This is why the mask is not stored
    next to the maxima: one file would have to pick a threshold forever."""
    rows = [[1.0] * 8, [1.0] * 8]
    rows[1][3] = 500.0
    rows[1][6] = 50.0
    assert outlier_mask_from_sinks(sinks_json(rows), threshold=100.0).sum() == 1
    assert outlier_mask_from_sinks(sinks_json(rows), threshold=20.0).sum() == 2


def test_rows_of_different_widths_are_refused():
    """They index one residual stream. Ragged rows mean the JSON is not what
    this function thinks it is, and silently truncating would build a mask over
    the wrong channels."""
    with pytest.raises(ValueError, match="disagree on width"):
        outlier_mask_from_sinks(sinks_json([[1.0] * 8, [1.0] * 4]))


def test_a_sinks_json_without_channel_max_is_an_error_not_an_empty_mask():
    """An empty mask would raise later, in resolve_fp16_exceptions, with a
    message about the model instead of about the file that was missing."""
    with pytest.raises(ValueError, match="no per-layer channel_max"):
        outlier_mask_from_sinks({"residual": [{"name": "model.layers.0"}]})


def test_an_outlier_free_model_yields_an_empty_mask_rather_than_a_guess():
    """The gated arms are this case at the locked threshold: the gate removed
    the massive activations, so there is no channel to exempt. The mask comes
    back empty and the CALLER decides what that means -- it is a fact about the
    checkpoint, not a failure of the derivation."""
    mask = outlier_mask_from_sinks(sinks_json([[1.0] * 8, [1.2] * 8]), threshold=100.0)
    assert mask.sum().item() == 0


# --- which tau becomes the detected_sinks arm --------------------------------
#
# This selection lived inside `main()`, where no test could reach it, and it
# round-tripped its own dict keys through float: the grid is written "2" ...
# "100", and `str(float("100"))` is "100.0", which is not a key. It raised
# KeyError on every checkpoint in the roster and nothing caught it, because the
# arm had never been executed. Extracted so these can hold it.

from quant.evaluate import detected_sink_positions  # noqa: E402


def detector_json(table, validation=None, kind="layerwise"):
    return {"primary_detector": kind, "detector": {kind: table},
            "detector_validation": {kind: validation or {}}}


def flagged(tau, positions):
    return {"tau": tau, "n_flagged": len(positions), "positions": positions}


def test_an_integer_keyed_tau_grid_is_selectable():
    """The regression. Keys are written as integers; a float round trip loses
    them. If this fails with KeyError, the arm is unrunnable again."""
    table = {"2": flagged(2, [0, 7]), "10": flagged(10, [0]), "100": flagged(100, [0])}
    tau, positions = detected_sink_positions(detector_json(table))
    assert tau == "100" and positions == [0]


def test_the_largest_validated_tau_wins():
    """Largest is the conservative end: it admits only the tokens whose
    magnitude is most clearly anomalous, so the exception list cannot quietly
    absorb borderline positions and inflate D_sink."""
    table = {"5": flagged(5, [0, 3, 9]), "20": flagged(20, [0, 3]), "50": flagged(50, [0])}
    assert detected_sink_positions(detector_json(table))[0] == "50"


def test_a_tau_that_failed_attention_validation_is_skipped():
    """C17's gate. Magnitude flagging a token the heads do not attend to breaks
    the attribution chain, and such a tau must not supply the exception list."""
    table = {"10": flagged(10, [0, 5]), "50": flagged(50, [0, 99])}
    validation = {"50": {"failed": "magnitude and attention disagree"}}
    tau, positions = detected_sink_positions(detector_json(table, validation))
    assert tau == "10" and positions == [0, 5]


def test_a_tau_that_flagged_nothing_is_skipped():
    """Empty is not a candidate: the arm would be identical to `none` and its
    D_sink a spurious zero."""
    table = {"10": flagged(10, [0]), "100": flagged(100, [])}
    assert detected_sink_positions(detector_json(table))[0] == "10"


def test_a_sink_free_checkpoint_raises_rather_than_returning_nothing():
    """Both gated arms are this case: tau=2 over-flags and fails validation,
    every larger tau flags nothing. There is no exception list to build, and
    that is the answer rather than an error (LIMITATIONS 17)."""
    table = {"2": flagged(2, [0, 291]), "5": flagged(5, []), "10": flagged(10, [])}
    validation = {"2": {"failed": "magnitude and attention disagree"}}
    with pytest.raises(ValueError, match="sink-free"):
        detected_sink_positions(detector_json(table, validation))


# --- what a run file records as its corpus ----------------------------------
#
# The path is provenance, not an identifier: runs are compared on
# `corpus_sha256`. Recording it absolutely put a home directory into 246 run
# files, 56 of them committed, which is why this exists.

from quant.evaluate import DEFAULT_CORPUS, REPO_ROOT, provenance_path  # noqa: E402


def test_a_corpus_inside_the_repo_is_recorded_relative_and_posix():
    got = provenance_path(str(DEFAULT_CORPUS))
    assert got == "data/fineweb_edu.txt"
    assert not got.startswith("/") and ":" not in got   # no drive letter, no root


def test_an_already_relative_path_survives_unchanged():
    assert provenance_path("data/code_python.txt") == "data/code_python.txt"


def test_a_corpus_outside_the_repo_is_left_alone():
    """There is no shorter form that still identifies it, so it is recorded as
    given rather than mangled into something that resolves elsewhere."""
    outside = str(REPO_ROOT.parent / "somewhere_else.txt")
    assert provenance_path(outside) == outside


def test_a_missing_corpus_stays_missing():
    """The streamed arm records no path at all; it must not become the string
    'None' or the repo root."""
    assert provenance_path(None) is None
    assert provenance_path("") == ""
