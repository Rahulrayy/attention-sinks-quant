"""Figure 3's geometry, asserted as numbers rather than admired as a PNG.

A figure is the one artefact in this repo that nothing checks by reading it.
`analysis.figures.lambada_rows` exists so the numbers that become ink can be
compared against the numbers in `analysis.lambada`'s tables, which is the
C19 lesson in a different medium: two paths to the same quantity, only one of
them ever verified.

Three properties are worth pinning here and only one of them is obvious.

  * **The interval ends swap.** The bootstrap is on the DROP; the axis is
    ACCURACY. A large drop is a low accuracy, so `ci_lo` must come from the
    interval's `hi`. Getting this backwards produces a plausible-looking bar
    that is wrong in a way no reader could detect.
  * **The floor and the interval are single-sourced.** `analysis.lambada` owns
    both. A second copy in the figure module would be free to drift from the
    table printed beside it.
  * **The import edge stays one-way.** `analysis.lambada` imports from
    `analysis.figures`; the reverse is deferred to call time. If that ever
    became a module-level import the pair would be a cycle.
"""

import json
import subprocess
import sys

import pytest

from analysis.figures import fig_lambada, lambada_rows, ordered_models
from analysis.lambada import SATURATION_FLOOR, drop_interval


def cell(model, gran, ref, quant):
    """A LAMBADA run from two 0/1 vectors, keyed the way `load_runs` keys them."""
    return (model, gran), {
        "model": model, "act_granularity": gran, "bits": 8, "static": True,
        "n_examples": len(ref),
        "accuracy_ref": sum(ref) / len(ref),
        "accuracy_quant": sum(quant) / len(quant),
        "accuracy_drop": (sum(ref) - sum(quant)) / len(ref),
        "n_discordant": sum(1 for a, b in zip(ref, quant) if a != b),
        "per_example_correct_ref": list(ref),
        "per_example_correct_quant": list(quant),
    }


def both_arms(model, *, tensor, token, ref):
    return dict([cell(model, "per_tensor", ref, tensor),
                 cell(model, "per_token", ref, token)])


# --- the axis mapping --------------------------------------------------------

def test_the_drop_interval_ends_swap_onto_the_accuracy_axis():
    """ci_lo is the accuracy implied by the LARGEST drop. Reversing these two
    lines would draw a bar of the right width in the wrong place."""
    ref = [1] * 100
    runs = both_arms("gpt2_small", ref=ref,
                     tensor=[1] * 60 + [0] * 40, token=[1] * 98 + [0, 0])
    row = next(r for r in lambada_rows(runs) if r["gran"] == "per_tensor")
    iv = drop_interval(runs[("gpt2_small", "per_tensor")])

    assert row["ci_lo"] == pytest.approx(row["acc_ref"] - iv.hi)
    assert row["ci_hi"] == pytest.approx(row["acc_ref"] - iv.lo)
    assert row["ci_lo"] < row["ci_hi"]
    # And the point the bar surrounds is the measured accuracy.
    assert row["ci_lo"] <= row["acc"] <= row["ci_hi"]


def test_the_bar_is_centred_on_the_measured_drop():
    """The row carries the same drop the table prints, not a re-derivation."""
    ref = [1] * 200
    runs = both_arms("gpt2_small", ref=ref,
                     tensor=[1] * 150 + [0] * 50, token=[1] * 199 + [0])
    for row in lambada_rows(runs):
        run = runs[(row["model"], row["gran"])]
        assert row["drop"] == pytest.approx(run["accuracy_drop"], abs=1e-12)
        assert row["acc"] == pytest.approx(run["accuracy_quant"])


# --- the two flags -----------------------------------------------------------

def test_the_floor_flag_uses_the_same_threshold_as_the_tables():
    """A cell at or below the floor is flagged; one just above it is not. The
    boundary is `analysis.lambada.SATURATION_FLOOR` and nothing else."""
    n = 1000
    at_floor = [1] * int(SATURATION_FLOOR * n) + [0] * (n - int(SATURATION_FLOOR * n))
    above = [1] * (int(SATURATION_FLOOR * n) + 1) + [0] * (n - int(SATURATION_FLOOR * n) - 1)

    rows = lambada_rows(both_arms("gpt2_small", ref=[1] * n,
                                  tensor=at_floor, token=above))
    by_gran = {r["gran"]: r for r in rows}
    assert by_gran["per_tensor"]["floored"]
    assert not by_gran["per_token"]["floored"]


def test_the_null_flag_agrees_with_the_interval_it_came_from():
    """Hollow markers mean 'this drop is not distinguishable from zero'. That
    verdict must be the interval's, not a second opinion."""
    runs = both_arms("gpt2_small", ref=[1] * 100,
                     tensor=[1] * 60 + [0] * 40, token=[1] * 99 + [0])
    for row in lambada_rows(runs):
        iv = drop_interval(runs[(row["model"], row["gran"])])
        assert row["null"] == iv.crosses_zero
    by_gran = {r["gran"]: r for r in lambada_rows(runs)}
    assert not by_gran["per_tensor"]["null"]   # 40 points is not zero
    assert by_gran["per_token"]["null"]        # one example is


# --- what gets drawn, and in what order --------------------------------------

def test_a_model_missing_an_arm_is_dropped_rather_than_half_drawn():
    """Half a row is worse than no row: the figure's whole subject is the
    comparison between the two granularities."""
    runs = both_arms("gpt2_small", ref=[1] * 50,
                     tensor=[1] * 30 + [0] * 20, token=[1] * 49 + [0])
    runs.update([cell("qwen3_0.6b_base", "per_tensor", [1] * 50, [0] * 50)])

    drawn = {r["model"] for r in lambada_rows(runs)}
    assert drawn == {"gpt2_small"}


def test_no_complete_pair_is_an_error_not_an_empty_figure():
    runs = dict([cell("gpt2_small", "per_tensor", [1] * 10, [0] * 10)])
    with pytest.raises(ValueError, match="per-tensor and a per-token"):
        lambada_rows(runs)


def test_row_order_is_figure_1_order_so_the_two_panels_stack():
    """Both figures put the first model of MODEL_ORDER at the top, which on a
    matplotlib y axis means it is last in the list."""
    models = ["gpt2_small", "qwen3_0.6b_base", "gated_1b_headwise"]
    runs = {}
    for m in models:
        runs.update(both_arms(m, ref=[1] * 40,
                              tensor=[1] * 20 + [0] * 20, token=[1] * 39 + [0]))
    order = list(dict.fromkeys(r["model"] for r in lambada_rows(runs)))
    assert order == ordered_models(models)[::-1]


def test_mixed_bit_widths_are_refused_rather_than_captioned_wrong(tmp_path):
    """The title's bit width is read off the cells. Two widths in one call would
    put a number in the caption that half the data does not have."""
    runs = both_arms("gpt2_small", ref=[1] * 40,
                     tensor=[1] * 20 + [0] * 20, token=[1] * 39 + [0])
    runs[("gpt2_small", "per_token")]["bits"] = 6
    with pytest.raises(ValueError, match="more than one bit width"):
        fig_lambada(runs, out=str(tmp_path / "fig3.png"))


def test_fig_lambada_writes_the_png_it_returns(tmp_path):
    runs = both_arms("gpt2_small", ref=[1] * 40,
                     tensor=[1] * 20 + [0] * 20, token=[1] * 39 + [0])
    out = tmp_path / "fig3.png"
    p = fig_lambada(runs, out=str(out))
    assert p == out and out.exists() and out.stat().st_size > 0


# --- the module graph --------------------------------------------------------

def test_importing_figures_does_not_import_lambada():
    """The deferred import is load-bearing: `analysis.lambada` imports four
    names from `analysis.figures` at module level, so a module-level import back
    would close the cycle."""
    code = ("import analysis.figures, sys; "
            "print(json.dumps('analysis.lambada' in sys.modules))")
    proc = subprocess.run(
        [sys.executable, "-c", "import json; " + code],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(proc.stdout.strip()) is False
