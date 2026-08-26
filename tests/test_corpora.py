"""The cross-corpus join, which decides which claims are retracted.

This module's output retired the ranking half of R6 (C21), so its failure modes
matter more than most. The two that would be silent:

  * `matched_pair` computing a reduction from an interval that crosses zero.
    That is a ratio of noise to noise, and it is exactly the number the README
    quotes as the audit's headline.
  * `r6_markdown` reporting "orders damage at: 90%" for a roster it does not
    order. A monotonicity check that is too permissive would have let the
    retracted claim stand.

Both are pinned below with inputs whose right answer is obvious by inspection.
"""

import json

import pytest

from analysis.corpora import (
    localisation_markdown,
    matched_pair,
    r6_markdown,
    verdict,
)


class Interval:
    """Stand-in for analysis.stats' interval type: point plus a zero-crossing flag."""

    def __init__(self, point, lo, hi):
        self.point, self.lo, self.hi = point, lo, hi

    @property
    def crosses_zero(self):
        return self.lo <= 0.0 <= self.hi


def cells_from(spec, bits=8, seed=0):
    """Build the `load_cells` mapping that analysis.report.rows consumes.

    `spec` is {model: (per_tensor_none_nll, per_tensor_exc_nll, ppl_ref)}. Only
    the fields rows() actually reads are populated -- a fuller fake would hide
    which ones the join depends on.
    """
    cells = {}
    for model, (none_seq, exc_seq, ppl_ref) in spec.items():
        for gran in ("per_tensor", "per_token"):
            for exc, seqs in (("none", none_seq), ("position_0", exc_seq)):
                cells[(model, bits, gran, exc, seed)] = {
                    "quantized_means": {"per_seq_all": list(seqs),
                                        "nll_all": sum(seqs) / len(seqs)},
                    "delta_ppl": 1.0,
                    "ppl_ref": ppl_ref,
                    "ppl_quant": ppl_ref * 1.5,
                }
    return cells


# --- the matched pair --------------------------------------------------------

def test_matched_pair_reports_the_reduction():
    """Baseline damaged, head-wise near-zero: the reduction is the ratio."""
    cells = cells_from({
        "gated_1b_baseline": ([1.0] * 8, [0.5] * 8, 14.7),
        "gated_1b_headwise": ([1.0] * 8, [0.99] * 8, 14.6),
    })
    m = matched_pair(cells)
    assert m["baseline"] == pytest.approx(0.5)
    assert m["headwise"] == pytest.approx(0.01, abs=1e-9)
    assert m["reduction"] == pytest.approx(50.0)


def test_matched_pair_refuses_a_reduction_from_a_zero_crossing_numerator():
    """If the BASELINE's own interval contains zero there is no damage to
    reduce, and a ratio would divide noise by noise. The headline number must
    come back None rather than large."""
    noisy = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
    cells = cells_from({
        "gated_1b_baseline": (noisy, [0.0] * 8, 14.7),
        "gated_1b_headwise": ([1.0] * 8, [0.999] * 8, 14.6),
    })
    assert matched_pair(cells)["reduction"] is None


def test_matched_pair_is_empty_when_an_arm_is_missing():
    """A half-run grid must not silently report a one-sided headline."""
    assert matched_pair(cells_from({"gated_1b_baseline": ([1.0] * 8, [0.5] * 8, 14.7)})) == {}


# --- the ranking check, which retracted a claim ------------------------------

def _dist(model, underflows):
    return {model: {"model": model, "bits": 8, "layers": {
        f"model.layers.{i}.mlp.up_proj": {"underflow_tensor": u, "underflow_token": 0.05,
                                          "dispersion": 2.0, "eff_bits": 7.0}
        for i, u in enumerate(underflows)}, "summary": {}}}


def _diag(model, dppl):
    return {model: {"a_only_dynamic": dppl, "a_only_per_token": 0.5, "ppl_ref": 14.0}}


def test_r6_ranking_reports_no_threshold_when_the_order_inverts():
    """The low-damage model is given MORE annihilated layers than the
    high-damage one at every threshold. There is no cut-off at which the
    statistic ranks this roster, and the table must say so in those words --
    this is the check that retracted the claim on the code corpus."""
    dist = {**_dist("quiet", [0.99, 0.99, 0.99]), **_dist("broken", [0.99])}
    diag = {**_diag("quiet", 1.0), **_diag("broken", 3000.0)}

    text = r6_markdown(dist, diag, dist, diag, label_a="A", label_b="B")
    assert "**no threshold**" in text


def test_r6_ranking_reports_thresholds_when_the_order_holds():
    dist = {**_dist("quiet", [0.1]), **_dist("broken", [0.99, 0.99])}
    diag = {**_diag("quiet", 1.0), **_diag("broken", 3000.0)}

    text = r6_markdown(dist, diag, dist, diag, label_a="A", label_b="B")
    assert "**no threshold**" not in text
    assert "90%" in text


# --- the verdict block -------------------------------------------------------

def test_verdict_counts_negative_point_estimates():
    """Per-token `D_sink` going negative is how the code corpus showed the
    metric had hit its floor. The count has to be reported, not rounded away."""
    cells_a = cells_from({"m": ([1.0] * 8, [0.9] * 8, 14.0)})
    cells_b = cells_from({"m": ([0.9] * 8, [1.0] * 8, 5.0)})   # exemption HURT

    text = verdict(cells_a, cells_b, label_a="A", label_b="B")
    assert "0/1 point estimates are negative" in text
    assert "1/1 point estimates are negative" in text


# --- the causal table --------------------------------------------------------

def test_localisation_distinguishes_a_small_recovery_from_a_large_one(tmp_path):
    """A 1.33x recovery and a 3265x recovery must not both print "1x better"."""
    root = tmp_path / "diag"
    root.mkdir(parents=True)
    for skip, dppl in ((), 3000.0), (("o_proj",), 2250.0), (("layers.0.mlp",), 2.0):
        suffix = ("_skip-" + "-".join(skip)) if skip else ""
        (root / f"gated_1b_elementwise_b8_diag{suffix}.json").write_text(json.dumps({
            "model": "gated_1b_elementwise", "bits": 8, "ppl_ref": 14.0,
            "skip_modules": list(skip),
            "arms": [{"arm": "a_only_dynamic", "delta_ppl": dppl}],
        }), encoding="utf-8")

    text = localisation_markdown(str(root), str(root), label_a="A", label_b="B")
    assert "1.33× better" in text
    assert "1500× better" in text
    assert "1× better" not in text


# --- the bit-width stability report (R5, C22) --------------------------------

def test_bitwidth_survivors_name_the_models_not_a_count():
    """R5's claim is "the single exception is head-wise", so the table has to
    name survivors rather than count them -- a count of 1 on each corpus would
    hide that they can be *different* models, and naming them is what made the
    destroyed-status flip visible."""
    from analysis.corpora import bitwidth_stability

    # ppl_ref 10: dppl 5 -> 1.5x (survives), dppl 200 -> 21x (destroyed).
    alive = cells_from({"gated_1b_headwise": ([1.0] * 8, [0.9] * 8, 10.0)})
    dead = cells_from({"gated_1b_headwise": ([1.0] * 8, [0.9] * 8, 10.0)})
    for k in alive:
        alive[k] = {**alive[k], "ppl_quant": 15.0, "delta_ppl": 5.0}
    for k in dead:
        dead[k] = {**dead[k], "ppl_quant": 210.0, "delta_ppl": 200.0}

    text = bitwidth_stability(alive, dead, label_a="A", label_b="B", widths=(8,))
    assert "headwise" in text
    assert "**none**" in text


def test_bitwidth_growth_is_a_magnitude_not_a_signed_ratio():
    """Per-token `D_sink` changes sign between corpora (§5.7), so an 8->6 growth
    computed on signed values would come back negative and read as shrinkage.
    The report divides magnitudes."""
    from analysis.corpora import bitwidth_stability

    cells = {}
    for bits, (none_seq, exc_seq) in ((8, ([1.0] * 8, [1.01] * 8)),
                                      (6, ([1.0] * 8, [1.1] * 8))):
        for gran in ("per_tensor", "per_token"):
            for exc, seqs in (("none", none_seq), ("position_0", exc_seq)):
                cells[("m", bits, gran, exc, 0)] = {
                    "quantized_means": {"per_seq_all": list(seqs),
                                        "nll_all": sum(seqs) / len(seqs)},
                    "delta_ppl": 1.0, "ppl_ref": 10.0, "ppl_quant": 11.0,
                }

    text = bitwidth_stability(cells, cells, label_a="A", label_b="B", widths=(8, 6))
    # D_sink is -0.01 at 8 bits and -0.10 at 6: a 10x growth in magnitude.
    assert "10.0×" in text
    assert "-10" not in text
