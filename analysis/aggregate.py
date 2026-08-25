"""runs/**/*.json -> one long-format dataframe.

Design rule §7.2: analysis NEVER touches a model. If something in here needs to
load weights, it belongs upstream in sinks/ or quant/.

Long format, one row per (model, bits, granularity, exception, seed, metric).
Wide format tempts you into unpaired comparisons; long format does not.

D_sink is RECONSTRUCTED here rather than measured in a cell, because it is a
difference between two cells. Each quant cell stores only the aggregate means it
needs (see quant/evaluate.py); the reference term cancels, so joining the `none`
arm against an exception arm on (model, bits, granularity, seed) recovers the
full decomposition without anyone having stored a per-token array.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant.evaluate import d_sink_from_means


def load_quant_runs(root: str = "runs/quant") -> pd.DataFrame:
    """One row per grid cell."""
    rows = []
    for path in sorted(Path(root).glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        rows.append(
            {
                "model": d["model"],
                "seed": d["calib_seed"],
                "bits": d["bits"],
                "granularity": d["act_granularity"],
                "exception": d["fp16_exception"],
                "ppl_ref": d["ppl_ref"],
                "ppl_quant": d["ppl_quant"],
                "delta_ppl": d["delta_ppl"],
                "delta_nll": d["delta_nll"],
                "nll_all": d["quantized_means"]["nll_all"],
                "nll_sink": d["quantized_means"]["nll_sink"],
                "nll_non_sink": d["quantized_means"]["nll_non_sink"],
                "n_sink_cols": d["quantized_means"]["n_sink_cols"],
                "n_total_cols": d["quantized_means"]["n_total_cols"],
                "n_exempt_positions": len(d.get("exempt_positions", [])),
                # Provenance, absent on cells written before 2026-08-25 (C19).
                # Two cells are comparable only if holdout_sha matches; the
                # older cells were confirmed to match by ppl_ref equality.
                "corpus_sha256": d.get("corpus_sha256"),
                "holdout_sha": d.get("holdout_sha"),
                "source": path.name,
            }
        )
    if not rows:
        raise FileNotFoundError(f"no quant run JSONs under {root}")
    return pd.DataFrame(rows)


def assert_comparable(df: pd.DataFrame) -> None:
    """Every cell of a model must share one unquantized reference.

    `ppl_ref` is measured on the held-out slice with quantization disabled, so
    it cannot depend on bits, granularity, exception or seed. If it varies, the
    cells were measured on different held-out slices and D_sink differences
    between them are comparing two experiments.

    This exists because that happened (C19): a corpus reader that tokenized
    line-by-line instead of whole-file shifted `ppl_ref` by 0.06-0.76% — small
    enough to read as GPU noise, large enough to make every cell it produced
    incomparable to the committed ones.
    """
    bad = []
    for model, sub in df.groupby("model"):
        refs = sub.ppl_ref.round(9).unique()
        if len(refs) > 1:
            spread = 100 * (max(refs) / min(refs) - 1)
            bad.append(f"  {model}: {len(refs)} distinct ppl_ref, spread {spread:.2f}% — {refs}")
    if bad:
        raise ValueError(
            "cells of the same model disagree on the unquantized reference, so "
            "they were not measured on the same held-out slice:\n"
            + "\n".join(bad)
            + "\n\nCheck `holdout_sha` on the cells involved. Re-run the odd ones "
            "out rather than aggregating across them (see C19)."
        )


def load_sink_runs(root: str = "runs/sinks") -> pd.DataFrame:
    """One row per measurement run, with the headline sink descriptors."""
    rows = []
    for path in sorted(Path(root).glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        kind = d.get("primary_detector", "layerwise")
        table, val = d["detector"][kind], d["detector_validation"][kind]
        # Keep the ORIGINAL key alongside its numeric value: the JSON stores
        # tau as "100", and round-tripping through float gives "100.0", which
        # is not a key. Sort by the number, index with the string.
        valid = sorted(
            ((float(t), t) for t in table
             if table[t]["n_flagged"] > 0 and "failed" not in val[t]),
            key=lambda pair: pair[0],
        )
        rows.append(
            {
                "model": d["args"]["model"],
                "seed": d["args"]["calib_seed"],
                "prepend_bos": d["args"]["prepend_bos"],
                "mean_sink_mass": d.get("mean_sink_mass"),
                "max_head_sink_mass": d.get("max_head_sink_mass"),
                "frac_heads_sinking": d.get("frac_heads_sinking"),
                "sink_free": d.get("sink_free"),
                "max_layer_relative": max(d["layer_relative_norm"]),
                "n_valid_taus": len(valid),
                "sink_positions": table[valid[-1][1]]["positions"] if valid else [],
                "source": path.name,
            }
        )
    if not rows:
        raise FileNotFoundError(f"no sink run JSONs under {root}")
    return pd.DataFrame(rows)


def compute_d_sink(df: pd.DataFrame, exception: str = "position_0") -> pd.DataFrame:
    """Join the `none` arm against an exception arm to reconstruct D_sink.

    Joined on (model, bits, granularity, seed) — the seed is part of the key, so
    the result is PAIRED by construction and can go straight into
    analysis.stats.paired_bootstrap without a further alignment step.
    """
    none = df[df.exception == "none"]
    exc = df[df.exception == exception]
    if exc.empty:
        raise ValueError(f"no cells with exception={exception!r}")

    key = ["model", "bits", "granularity", "seed"]
    merged = none.merge(exc, on=key, suffixes=("_none", "_exc"))
    if merged.empty:
        raise ValueError(
            f"no (model, bits, granularity, seed) cell has BOTH 'none' and "
            f"{exception!r}. D_sink is a difference between two cells; without "
            "both arms on the same seed it cannot be formed."
        )

    out = []
    for _, r in merged.iterrows():
        dec = d_sink_from_means(
            {
                "nll_all": r.nll_all_none,
                "nll_sink": r.nll_sink_none,
                "nll_non_sink": r.nll_non_sink_none,
                "n_sink_cols": r.n_sink_cols_none,
                "n_total_cols": r.n_total_cols_none,
            },
            {
                "nll_all": r.nll_all_exc,
                "nll_sink": r.nll_sink_exc,
                "nll_non_sink": r.nll_non_sink_exc,
                "n_sink_cols": r.n_sink_cols_exc,
                "n_total_cols": r.n_total_cols_exc,
            },
        )
        out.append(
            {
                "model": r.model,
                "bits": r.bits,
                "granularity": r.granularity,
                "seed": r.seed,
                "exception": exception,
                "d_sink_ppl": r.delta_ppl_none - r.delta_ppl_exc,
                **dec,
            }
        )
    return pd.DataFrame(out)


def to_long(df: pd.DataFrame, id_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    return df.melt(id_vars=id_cols, value_vars=value_cols,
                   var_name="metric", value_name="value")


def write_summary(df: pd.DataFrame, out: str = "runs/results/summary.csv") -> Path:
    """The one file under runs/ that is committed."""
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def main() -> None:
    Path("runs/results").mkdir(parents=True, exist_ok=True)

    sinks = load_sink_runs()
    write_summary(sinks, "runs/results/sinks_summary.csv")
    print(f"sink runs: {len(sinks)} rows -> runs/results/sinks_summary.csv")
    print(sinks[["model", "mean_sink_mass", "frac_heads_sinking",
                 "max_layer_relative", "sink_free"]].to_string(index=False))

    try:
        quant = load_quant_runs()
    except FileNotFoundError:
        print("\nno quant runs yet — run `make quant` before figures")
        return

    assert_comparable(quant)
    write_summary(quant, "runs/results/summary.csv")
    print(f"\nquant cells: {len(quant)} rows -> runs/results/summary.csv")

    ds = compute_d_sink(quant)
    write_summary(ds, "runs/results/d_sink.csv")
    print(f"D_sink rows: {len(ds)} -> runs/results/d_sink.csv")


if __name__ == "__main__":
    main()
