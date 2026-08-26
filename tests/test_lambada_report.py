"""The LAMBADA reporting contract, whose failure modes are all silent.

Accuracy is discrete and bounded, and both properties create ways for a table to
be wrong while looking right:

  * **A zero drop means two different things.** Zero across many discordant
    pairs is cancellation -- the metric moved on both sides and netted out.
    Zero across no discordant pairs is the arms agreeing on every example. The
    first is a measurement, the second is the absence of one, and a table that
    printed them identically would repeat C18 in a new metric.

  * **A bounded metric saturates.** Once a model is at chance, further damage
    cannot lower its score, so the drop stops ordering anything -- the accuracy
    analogue of the destroyed-cell problem LIMITATIONS 19 records for
    perplexity.

  * **The arm key must survive the dynamic control.** `per_tensor` and
    `per_tensor_dynamic` differ only by a boolean in the JSON, and collapsing
    them would silently overwrite one cell with the other.
"""

import json

import pytest

from analysis.lambada import (
    SATURATION_FLOOR,
    drop_interval,
    load_runs,
    markdown,
    power,
)


def write_run(root, model, arm, ref, quant, static=True, bits=8):
    """A LAMBADA cell from two 0/1 vectors, which are the whole of the metric."""
    root.mkdir(parents=True, exist_ok=True)
    gran = "per_tensor" if arm.startswith("per_tensor") else "per_token"
    payload = {
        "model": model, "task": "lambada_openai", "bits": bits,
        "act_granularity": gran, "static": static,
        "n_examples": len(ref),
        "accuracy_ref": sum(ref) / len(ref),
        "accuracy_quant": sum(quant) / len(quant),
        "accuracy_drop": (sum(ref) - sum(quant)) / len(ref),
        "n_discordant": sum(1 for a, b in zip(ref, quant) if a != b),
        "per_example_correct_ref": list(ref),
        "per_example_correct_quant": list(quant),
        "target_ppl_ratio": 1.0,
    }
    (root / f"{model}_b{bits}_{arm}_lambada.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


# --- the paired interval -----------------------------------------------------

def test_drop_interval_is_paired_over_examples():
    """The quantized arm is wrong on 10 of 100 examples the reference got right,
    so the drop is 0.10 and the interval is built from the per-example
    differences rather than from two independent accuracy estimates."""
    ref = [1] * 100
    quant = [1] * 90 + [0] * 10
    iv = drop_interval({"per_example_correct_ref": ref,
                        "per_example_correct_quant": quant})
    assert iv.point == pytest.approx(0.10)
    assert not iv.crosses_zero
    assert iv.n == 100


def test_drop_interval_widens_as_the_sample_shrinks():
    """The same effect on ten examples cannot be called non-zero. This is the
    property `power()` reports, asserted here so a change that smoothed the
    interval would be caught rather than admired."""
    small = drop_interval({"per_example_correct_ref": [1] * 10,
                           "per_example_correct_quant": [1] * 8 + [0, 0]})
    large = drop_interval({"per_example_correct_ref": [1] * 500,
                           "per_example_correct_quant": [1] * 400 + [0] * 100})
    assert small.point == pytest.approx(0.2)
    assert large.point == pytest.approx(0.2)
    assert small.crosses_zero
    assert not large.crosses_zero
    assert small.width > large.width


def test_a_cancelling_drop_is_not_the_same_as_no_change():
    """Both cells have a drop of exactly zero. One reached it by moving on four
    examples and cancelling; the other never moved. The discordant count is what
    separates them, and it is the number the reader needs."""
    cancel_ref = [1, 1, 0, 0] + [1] * 6
    cancel_quant = [0, 0, 1, 1] + [1] * 6
    still_ref = [1] * 10

    iv_cancel = drop_interval({"per_example_correct_ref": cancel_ref,
                               "per_example_correct_quant": cancel_quant})
    iv_still = drop_interval({"per_example_correct_ref": still_ref,
                              "per_example_correct_quant": still_ref})

    assert iv_cancel.point == pytest.approx(0.0)
    assert iv_still.point == pytest.approx(0.0)
    # The cancelling cell has real spread; the static one has none at all.
    assert iv_cancel.width > 0.0
    assert iv_still.width == pytest.approx(0.0)


# --- the tables --------------------------------------------------------------

def test_markdown_reports_the_discordant_count(tmp_path):
    root = tmp_path / "lambada"
    write_run(root, "gpt2_small", "per_tensor", [1, 1, 1, 0], [0, 0, 1, 0])
    write_run(root, "gpt2_small", "per_token", [1, 1, 1, 0], [1, 1, 1, 0])

    text = markdown(load_runs(str(root)), None)
    assert "| 2 |" in text          # per-tensor moved on two examples
    assert "| 0 |" in text          # per-token moved on none


def test_markdown_flags_a_saturated_cell(tmp_path):
    """A model driven to chance cannot be damaged further, so its drop stops
    ranking. The floor marker has to appear."""
    root = tmp_path / "lambada"
    n = 100
    ref = [1] * 50 + [0] * 50
    dead = [0] * n                                  # 0.00 accuracy, at the floor
    write_run(root, "gpt2_small", "per_tensor", ref, dead)
    write_run(root, "gpt2_small", "per_token", ref, ref)

    text = markdown(load_runs(str(root)), None)
    assert "⌊" in text
    assert f"{SATURATION_FLOOR:.0%}" in text


def test_load_runs_keeps_the_dynamic_control_separate(tmp_path):
    """`per_tensor` and `per_tensor_dynamic` share a granularity and differ by
    one boolean. Keying on granularity alone would drop one of them."""
    root = tmp_path / "lambada"
    write_run(root, "m", "per_tensor", [1, 1], [0, 1], static=True)
    write_run(root, "m", "per_tensor_dynamic", [1, 1], [1, 1], static=False)

    runs = load_runs(str(root))
    assert set(runs) == {("m", "per_tensor"), ("m", "per_tensor_dynamic")}
    assert runs[("m", "per_tensor")]["n_discordant"] == 1
    assert runs[("m", "per_tensor_dynamic")]["n_discordant"] == 0


def test_power_reports_a_resolution_wider_than_a_tiny_effect(tmp_path):
    """A cell whose drop is smaller than its own resolution is a bound, not a
    measurement, and the table has to make that checkable."""
    root = tmp_path / "lambada"
    ref = [1] * 200
    quant = [0] + [1] * 199          # a single flip: drop 0.005
    write_run(root, "m", "per_token", ref, quant)

    runs = load_runs(str(root))
    text = power(runs)
    iv = drop_interval(runs[("m", "per_token")])
    assert iv.point == pytest.approx(0.005)
    assert iv.width / 2 >= iv.point      # cannot be called non-zero
    assert "±" in text
