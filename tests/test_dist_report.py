"""The R6 join, which is where a mechanism claim could quietly become circular.

Two of these guard against that specifically. `load_diag` must exclude the
`--skip-modules` runs, because those runs ARE the intervention: letting one of
them stand in as a model's damage would have the statistic explaining its own
treatment. And `sweep` must actually detect a reordering, because its whole job
is to report the thresholds at which the ranking fails -- a monotonicity check
that always says "fine" is worse than none.
"""

import json

import pytest

from analysis.distributions import (
    blocks,
    load_diag,
    load_dist,
    localisation,
    profile,
    sweep,
)


def layer(uf_tensor, uf_token=0.05, dispersion=2.0):
    return {"underflow_tensor": uf_tensor, "underflow_token": uf_token,
            "dispersion": dispersion, "eff_bits": 7.0, "crest": 3.0}


def write_dist(root, model, layers, bits=8):
    root.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "bits": bits, "ppl_ref": 14.0, "layers": layers,
               "summary": {"underflow_tensor_median": 0.2,
                           "underflow_token_median": 0.1}}
    (root / f"{model}_b{bits}_dist.json").write_text(json.dumps(payload), encoding="utf-8")


def write_diag(root, model, dppl, skip=(), bits=8):
    root.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "bits": bits, "ppl_ref": 14.0,
               "skip_modules": list(skip),
               "arms": [{"arm": "a_only_dynamic", "delta_ppl": dppl},
                        {"arm": "a_only_per_token", "delta_ppl": 0.5}]}
    suffix = ("_skip-" + "-".join(skip)) if skip else ""
    (root / f"{model}_b{bits}_diag{suffix}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


# --- block extraction --------------------------------------------------------

def test_blocks_reads_both_naming_schemes():
    """Qwen3 names blocks `model.layers.N`, GPT-2 names them `transformer.h.N`.
    A regex that only knew one would silently report zero annihilated blocks for
    the roster's only pre-QK-Norm checkpoint."""
    layers = {"model.layers.6.mlp.down_proj": layer(0.99),
              "transformer.h.2.mlp.c_proj": layer(0.99)}
    assert blocks(layers) == [2, 6]


def test_blocks_does_not_confuse_block_1_with_block_10():
    """`layers.1` is a prefix of `layers.10`. Substring matching is how
    quant.diagnose selects modules to exempt, so the same hazard exists on the
    reporting side and is pinned here."""
    layers = {"model.layers.10.mlp.up_proj": layer(0.99)}
    assert blocks(layers) == [10]


def test_blocks_respects_the_threshold():
    layers = {"model.layers.3.mlp.up_proj": layer(0.85)}
    assert blocks(layers, 0.9) == []
    assert blocks(layers, 0.8) == [3]


# --- the join ----------------------------------------------------------------

def test_load_diag_excludes_the_intervention_runs(tmp_path):
    """A `--skip-modules` run is a treatment, not a measurement of damage. If it
    were loaded as a model's damage, the statistic would be scored against a
    number the intervention already changed."""
    root = tmp_path / "diag"
    write_diag(root, "m", 3481.0)
    write_diag(root, "m", 6.5, skip=("layers.0.mlp",))

    loaded = load_diag(str(root))
    assert loaded["m"]["a_only_dynamic"] == pytest.approx(3481.0)


def test_profile_sorts_by_the_damage_it_must_explain(tmp_path):
    dist, diag = tmp_path / "dist", tmp_path / "diag"
    for name, dppl in (("a", 100.0), ("b", 1.0), ("c", 10.0)):
        write_dist(dist, name, {"model.layers.0.mlp.up_proj": layer(0.99)})
        write_diag(diag, name, dppl)

    rows = profile(load_dist(str(dist)), load_diag(str(diag)))
    assert [r["model"] for r in rows] == ["b", "c", "a"]


def test_profile_counts_blocks_not_just_layers(tmp_path):
    """Three annihilated projections inside one block is a different finding
    from three spread across three blocks, and the table reports both."""
    dist, diag = tmp_path / "dist", tmp_path / "diag"
    write_dist(dist, "m", {
        "model.layers.0.mlp.up_proj": layer(0.99),
        "model.layers.0.mlp.gate_proj": layer(0.99),
        "model.layers.0.mlp.down_proj": layer(0.99),
        "model.layers.5.mlp.up_proj": layer(0.10),
    })
    write_diag(diag, "m", 42.0)

    r = profile(load_dist(str(dist)), load_diag(str(diag)))[0]
    assert r["n_annihilated"] == 3
    assert r["annihilated_blocks"] == [0]
    assert r["first_block"] == 0
    assert r["n_blocks"] == 6          # highest index seen, 5, plus one


# --- the threshold sweep -----------------------------------------------------

def test_sweep_detects_a_reordering(tmp_path):
    """The low-damage model is given many partially-underflowing layers and no
    annihilated one. At a loose threshold it outranks the damaged model and the
    sweep must say so; at a strict one the ordering is restored."""
    dist, diag = tmp_path / "dist", tmp_path / "diag"
    write_dist(dist, "quiet", {f"model.layers.{i}.mlp.up_proj": layer(0.6)
                               for i in range(20)})
    write_diag(diag, "quiet", 1.0)
    write_dist(dist, "broken", {"model.layers.0.mlp.up_proj": layer(0.999)})
    write_diag(diag, "broken", 3000.0)

    text = sweep(load_dist(str(dist)), load_diag(str(diag)), thresholds=(0.5, 0.9))
    reproduces = text.split("Reproduces the damage ordering at: ")[1].split("\n")[0]
    fails = text.split("Fails to at: ")[1].split("\n")[0]
    assert "90%" in reproduces
    assert "50%" in fails


def test_localisation_marks_the_unexempted_run_as_the_control(tmp_path):
    root = tmp_path / "diag"
    write_diag(root, "m", 3481.0)
    write_diag(root, "m", 6.5, skip=("layers.0.mlp",))

    text = localisation(str(root))
    assert "(control)" in text
    assert "536× better" in text


# --- the axis table (C23) ----------------------------------------------------

def test_axis_table_picks_the_worst_layer_by_per_tensor_underflow(tmp_path):
    """The row has to describe the layer the damage actually lives in, which is
    the one a shared scale annihilates -- not the one with the largest
    dispersion on some other axis."""
    from analysis.distributions import axis_markdown

    dist, diag = tmp_path / "dist", tmp_path / "diag"
    write_dist(dist, "m", {
        "model.layers.1.mlp.up_proj": {**layer(0.20), "col_dispersion": 9000.0,
                                       "underflow_col": 0.01},
        "model.layers.7.mlp.up_proj": {**layer(0.99), "col_dispersion": 12.0,
                                       "underflow_col": 0.02},
    })
    write_diag(diag, "m", 42.0)

    text = axis_markdown(load_dist(str(dist)), load_diag(str(diag)))
    assert "`L7.mlp.up_proj`" in text
    assert "L1.mlp.up_proj" not in text


def test_axis_table_survives_runs_written_before_the_col_fields_existed(tmp_path):
    """`runs/dist*` JSONs predate the feature-axis statistic. A report that
    crashed or printed a bare 0 on them would silently misdescribe the older
    grids rather than saying the measurement is absent."""
    from analysis.distributions import axis_markdown

    dist, diag = tmp_path / "dist", tmp_path / "diag"
    write_dist(dist, "m", {"model.layers.0.mlp.up_proj": layer(0.99)})  # no col_*
    write_diag(diag, "m", 42.0)

    text = axis_markdown(load_dist(str(dist)), load_diag(str(diag)))
    assert "—" in text
    assert "0.0000" not in text
