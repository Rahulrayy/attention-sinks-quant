"""The README's result tables, generated from runs/quant rather than typed.

Every `D_sink` number quoted in README §5 and in HANDOFF §3 comes out of here.
Transcribing them by hand is how the README ended up still carrying the
superseded R3 figures for five days after R3-rev replaced them.

Two conventions, both inherited from analysis.figures and kept identical so a
table and a figure can never disagree:

  * The point estimate is calibration draw 0. The other four draws exist and are
    reported by `variance()`, but per-token scaling is dynamic and gives all five
    draws byte-identical results (C18), so a mean over draws would be a mean over
    one number on the arm that matters.
  * The interval is a percentile bootstrap over held-out SEQUENCES, which vary
    for both granularities. See analysis.stats.sequence_bootstrap.

`ppl_ref` and `delta_ppl` are carried in the table on purpose: a `D_sink` is
only interpretable while the quantized model still works, and the 4-bit cells
are the standing proof that the metric outlives its valid range (LIMITATIONS
§19).
"""

from __future__ import annotations

import argparse

from .figures import MODEL_ORDER, is_destroyed, load_cells
from .stats import sequence_bootstrap


def _ordered(models) -> list[str]:
    known = [m for m in MODEL_ORDER if m in models]
    return known + sorted(set(models) - set(MODEL_ORDER))


def rows(cells: dict, *, bits: int, seed: int = 0, exception: str = "position_0") -> list[dict]:
    """One row per model: both granularities, with intervals and damage level."""
    out = []
    for model in _ordered({k[0] for k in cells}):
        row: dict = {"model": model}
        for gran in ("per_tensor", "per_token"):
            none = cells.get((model, bits, gran, "none", seed))
            exc = cells.get((model, bits, gran, exception, seed))
            if not none or not exc:
                continue
            iv = sequence_bootstrap(
                none["quantized_means"]["per_seq_all"], exc["quantized_means"]["per_seq_all"]
            )
            row[gran] = iv
            row[f"{gran}_dppl"] = none["delta_ppl"]
            row[f"{gran}_destroyed"] = is_destroyed(cells, model, bits, gran, seed)
            row["ppl_ref"] = none["ppl_ref"]
        if "per_tensor" in row and "per_token" in row:
            a, b = row["per_tensor"].point, row["per_token"].point
            # A ratio is only meaningful when both ends are real damage measured
            # on models that still work. It is suppressed when the numerator's
            # interval crosses zero (nothing to reduce, so the ratio divides
            # noise by noise) and when either arm is in the destroyed regime,
            # where the numerator is a difference between two broken models.
            usable = (
                b > 0
                and not row["per_tensor"].crosses_zero
                and not row.get("per_tensor_destroyed")
                and not row.get("per_token_destroyed")
            )
            row["ratio"] = a / b if usable else None
        out.append(row)
    return out


def _fmt(iv) -> str:
    if iv is None:
        return "—"
    s = f"{iv.point:+.4f} [{iv.lo:+.4f}, {iv.hi:+.4f}]"
    return s + " *ZERO*" if iv.crosses_zero else s


def markdown(cells: dict, *, bits: int, seed: int = 0,
             exception: str = "position_0") -> str:
    """The README table for one bit width.

    ``exception`` picks which fp16 arm the `none` cell is differenced against.
    The default is the token-axis one the headline metric is defined on;
    ``outlier_channels`` reads the same table on the feature axis, which is the
    only way the two are comparable — same holdout, same draws, same bootstrap.
    """
    axis = ("D_sink" if exception in ("position_0", "detected_sinks")
            else f"`{exception}` damage removed")
    lines = [
        f"{axis} at {bits}-bit activations, nats, 95% sequence-bootstrap CI. "
        "*ZERO* = interval contains zero.",
        "",
        "| model | ppl_ref | Δppl (per-tensor) | per-tensor (2023) | "
        "Δppl (per-token) | per-token (modern) | ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    table = rows(cells, bits=bits, seed=seed, exception=exception)
    undefined = [r["model"] for r in table
                 if "per_tensor" not in r and "per_token" not in r]
    for r in table:
        if r["model"] in undefined:
            continue
        ratio = f"{r['ratio']:.0f}×" if r.get("ratio") else "—"
        dead_a = " **DESTROYED**" if r.get("per_tensor_destroyed") else ""
        dead_b = " **DESTROYED**" if r.get("per_token_destroyed") else ""
        lines.append(
            f"| `{r['model']}` | {r.get('ppl_ref', float('nan')):.1f} | "
            f"{r.get('per_tensor_dppl', float('nan')):+.2f}{dead_a} | {_fmt(r.get('per_tensor'))} | "
            f"{r.get('per_token_dppl', float('nan')):+.4f}{dead_b} | "
            f"{_fmt(r.get('per_token'))} | {ratio} |"
        )
    if undefined:
        lines += [
            "",
            "No `" + exception + "` cell exists for " +
            ", ".join(f"`{m}`" for m in undefined) +
            ". That is not a missing run: the arm cannot be constructed for a "
            "model the detector flags nothing on, so there is nothing to hold "
            "in fp16 and the difference would be a spurious zero "
            "(LIMITATIONS §17, and §23 for this axis).",
        ]
    if any(r.get("per_tensor_destroyed") or r.get("per_token_destroyed") for r in table):
        lines += [
            "",
            "**DESTROYED** = quantized perplexity exceeds 10× the reference. "
            "`D_sink` on such a cell is a difference between two broken models "
            "and does not rank anything (LIMITATIONS §19).",
        ]
    return "\n".join(lines)


def bitwidth_markdown(cells: dict, *, gran: str = "per_token", seed: int = 0) -> str:
    """One granularity across every bit width — README §5.5.

    Defaults to per-token because that is the arm the research question lives
    on; the per-tensor column is destroyed below 8 bits on almost everything,
    so a table of it would be a table of uninterpretable cells.
    """
    widths = sorted({k[1] for k in cells}, reverse=True)
    lines = [
        f"`D_sink` under **{gran.replace('_', '-')}** scaling, nats, "
        "95% sequence-bootstrap CI.",
        "",
        "| model | " + " | ".join(f"{b}-bit" for b in widths) + " |",
        "|---" * (len(widths) + 1) + "|",
    ]
    for model in _ordered({k[0] for k in cells}):
        cols = []
        for bits in widths:
            r = next((x for x in rows(cells, bits=bits, seed=seed) if x["model"] == model), None)
            iv = r.get(gran) if r else None
            cell = _fmt(iv)
            if r and r.get(f"{gran}_destroyed"):
                cell += " ⚠"
            cols.append(cell)
        lines.append(f"| `{model}` | " + " | ".join(cols) + " |")
    lines += ["", "⚠ = quantized perplexity above 10× the reference; the cell is a "
                  "difference between two destroyed models and ranks nothing."]
    return "\n".join(lines)


def threshold_sweep(cells: dict, *, seed: int = 0, thresholds=(2, 3, 5, 10, 20, 50, 100)) -> str:
    """How many cells the destroyed-model threshold flags, as it moves.

    `DESTROYED_PPL_RATIO` is a judgement call and it is load-bearing: it is what
    removed the element-wise 8-bit per-tensor cell from the ranking in README
    §5.3. A judgement call that changes conclusions has to be shown to be robust
    or declared fragile, not asserted to be either — this prints the evidence.

    Read the gaps, not the counts. A threshold sitting inside a wide empty band
    is insensitive there; one sitting next to a cluster is not.
    """
    ratios = []
    for (model, bits, gran, exc, s), c in cells.items():
        if exc != "none" or s != seed or not c.get("ppl_ref"):
            continue
        ratios.append((c["ppl_quant"] / c["ppl_ref"], model, bits, gran))
    ratios.sort()

    lines = ["Destroyed-cell threshold sweep (exception=none, seed 0).", ""]
    lines.append("| threshold | cells flagged | of | cells that change vs 10x |")
    lines.append("|---|---|---|---|")
    base = {(m, b, g) for r, m, b, g in ratios if r > 10.0}
    for t in thresholds:
        flagged = {(m, b, g) for r, m, b, g in ratios if r > t}
        delta = flagged ^ base
        names = ", ".join(f"{m.replace('gated_1b_', '')} {b}b {g.replace('per_', '')}"
                          for m, b, g in sorted(delta)) or "—"
        lines.append(f"| {t}× | {len(flagged)} | {len(ratios)} | {names} |")

    lines += ["", "Sorted ppl_quant / ppl_ref, to show where the gaps are:", ""]
    for r, m, b, g in ratios:
        lines.append(f"  {r:>12.2f}   {m} {b}b {g}")
    return "\n".join(lines)


def variance(cells: dict, *, bits: int, exception: str = "position_0") -> str:
    """Spread of the arm's effect across calibration draws, per model x granularity.

    Exists to keep C18 visible in the output: the per-token rows should read
    exactly 0.000000, and a reader should be able to see that rather than infer
    it from a suspiciously tight interval.
    """
    import statistics

    lines = [f"`{exception}` effect, spread across the 5 calibration draws, "
             f"{bits}-bit:", ""]
    for model in _ordered({k[0] for k in cells}):
        for gran in ("per_tensor", "per_token"):
            vals = []
            for seed in range(5):
                none = cells.get((model, bits, gran, "none", seed))
                exc = cells.get((model, bits, gran, exception, seed))
                if none and exc:
                    vals.append(
                        none["quantized_means"]["nll_all"] - exc["quantized_means"]["nll_all"]
                    )
            if len(vals) < 2:
                continue
            sd = statistics.stdev(vals)
            flag = "" if sd > 0 else "   <- draw is not a variance source (C18)"
            lines.append(f"  {model:<22} {gran:<11} n={len(vals)}  std={sd:.6f}{flag}")
    return "\n".join(lines)


def arm_summary(cells: dict, *, bits: int = 8, exception: str = "outlier_channels") -> str:
    """Total damage with and without one fp16 arm, averaged over the draws.

    The interval tables above are in nats and are the statistical claim. This is
    the same arms in perplexity, which is the unit the rest of the README uses
    for total damage, and averaged over all five calibration draws rather than
    read off seed 0 — for a static per-tensor arm the draw IS a variance source
    (C18 applies only to the dynamic one), and on GPT-2 seed 0 alone points the
    opposite way from the other four.
    """
    import statistics

    lines = [
        f"Total damage with and without the `{exception}` exemption, "
        f"{bits}-bit, mean over 5 calibration draws.",
        "",
        "| model | granularity | Δppl none | Δppl exempt | damage removed |",
        "|---|---|---|---|---|",
    ]
    any_row = False
    for model in _ordered({k[0] for k in cells}):
        for gran in ("per_tensor", "per_token"):
            none = [cells[(model, bits, gran, "none", s)]["delta_ppl"]
                    for s in range(5) if (model, bits, gran, "none", s) in cells]
            arm = [cells[(model, bits, gran, exception, s)]["delta_ppl"]
                   for s in range(5) if (model, bits, gran, exception, s) in cells]
            if not none or not arm:
                continue
            a, b = statistics.mean(none), statistics.mean(arm)
            pct = f"{100 * (a - b) / a:+.1f}%" if a else "—"
            lines.append(f"| `{model}` | {gran.replace('_', '-')} | {a:+.4f} "
                         f"| {b:+.4f} | **{pct}** (n={len(none)},{len(arm)}) |")
            any_row = True
    if not any_row:
        return f"(no `{exception}` cells at {bits} bits)"
    lines += [
        "",
        "*Damage removed* is positive when the exemption helped. A negative "
        "number is not a small effect measured precisely — it is the arm making "
        "the model slightly worse, which a whole-channel exemption can do by "
        "spending range on a channel that was not the problem.",
    ]
    return "\n".join(lines)


def outlier_mask_table(sinks_root: str = "runs/sinks",
                       thresholds=(100.0, 50.0, 20.0, 10.0)) -> str:
    """How many channels the feature-axis detector flags, per model. Generated.

    The first column is the only one that is a result: 100x is the threshold
    `sinks/metrics.py` locked before any data was collected. The rest are a
    sensitivity diagnostic, and they are printed together precisely so nobody
    can quote a looser one as though it were the definition. Choosing the
    threshold that makes an arm runnable, after seeing which models it makes
    runnable, is the failure this project audits others for.
    """
    import json
    from pathlib import Path

    import torch

    from sinks.metrics import aggregate_outlier_channels

    rows_ = []
    for path in sorted(Path(sinks_root).glob("*_calib0_nobos.json")):
        with open(path, encoding="utf-8") as fh:
            sinks = json.load(fh)
        chan = [r["channel_max"] for r in sinks.get("residual", []) if r.get("channel_max")]
        if not chan:
            continue
        t = torch.tensor(chan, dtype=torch.float32)
        model = path.name.replace("_calib0_nobos.json", "")
        rows_.append((model, t.shape[1],
                      [int(aggregate_outlier_channels(t, threshold=th).sum())
                       for th in thresholds]))
    if not rows_:
        return f"(no sink runs with channel_max under {sinks_root})"

    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    rows_.sort(key=lambda r: order.get(r[0], len(MODEL_ORDER)))

    head = " | ".join(f"{'**' if th == 100.0 else ''}{th:g}×"
                      f"{'**' if th == 100.0 else ''}" for th in thresholds)
    lines = [
        "Channels flagged by the residual-stream outlier detector "
        "(`aggregate_outlier_channels`), per model.",
        "",
        f"| model | channels | {head} |",
        "|---|---|" + "---|" * len(thresholds),
    ]
    for model, width, counts in rows_:
        cells_ = " | ".join(f"**{c}**" if i == 0 else str(c)
                            for i, c in enumerate(counts))
        lines.append(f"| `{model}` | {width} | {cells_} |")
    lines += [
        "",
        "**100× is the locked threshold** (`sinks/metrics.py`, fixed before any "
        "data was collected). The looser columns are a sensitivity diagnostic "
        "and are not results: a model that only becomes measurable at 10× was "
        "not measurable.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, action="append",
                        help="repeatable; default: every bit width present")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default="runs/quant")
    parser.add_argument("--by-bitwidth", action="store_true",
                        help="one granularity across every width (README §5.5)")
    parser.add_argument("--granularity", default="per_token")
    parser.add_argument("--exception", default="position_0",
                        help="fp16 arm to difference against: position_0 "
                             "(default), detected_sinks, outlier_channels")
    parser.add_argument("--outlier-mask", action="store_true",
                        help="channels the feature-axis detector flags, per model (C24)")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="sensitivity of the destroyed-cell threshold (HANDOFF §12)")
    ns = parser.parse_args()

    if ns.outlier_mask:
        print(outlier_mask_table())
        return

    cells = load_cells(ns.root)
    if ns.by_bitwidth:
        print(bitwidth_markdown(cells, gran=ns.granularity, seed=ns.seed))
        return
    if ns.threshold_sweep:
        print(threshold_sweep(cells, seed=ns.seed))
        return

    widths = ns.bits or sorted({k[1] for k in cells}, reverse=True)
    for bits in widths:
        print(markdown(cells, bits=bits, seed=ns.seed, exception=ns.exception))
        print()
        # Perplexity, draw-averaged, for whichever arm is being reported. The
        # interval table above is nats at seed 0; this is the same arm in the
        # unit the README uses for total damage, over every draw.
        print(arm_summary(cells, bits=bits, exception=ns.exception))
        print()
        print(variance(cells, bits=bits, exception=ns.exception))
        print()


if __name__ == "__main__":
    main()
