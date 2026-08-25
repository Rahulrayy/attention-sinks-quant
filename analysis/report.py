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


def markdown(cells: dict, *, bits: int, seed: int = 0) -> str:
    """The README table for one bit width."""
    lines = [
        f"`D_sink` at {bits}-bit activations, nats, 95% sequence-bootstrap CI. "
        "*ZERO* = interval contains zero.",
        "",
        "| model | ppl_ref | Δppl (per-tensor) | per-tensor (2023) | "
        "Δppl (per-token) | per-token (modern) | ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    table = rows(cells, bits=bits, seed=seed)
    for r in table:
        ratio = f"{r['ratio']:.0f}×" if r.get("ratio") else "—"
        dead_a = " **DESTROYED**" if r.get("per_tensor_destroyed") else ""
        dead_b = " **DESTROYED**" if r.get("per_token_destroyed") else ""
        lines.append(
            f"| `{r['model']}` | {r.get('ppl_ref', float('nan')):.1f} | "
            f"{r.get('per_tensor_dppl', float('nan')):+.2f}{dead_a} | {_fmt(r.get('per_tensor'))} | "
            f"{r.get('per_token_dppl', float('nan')):+.4f}{dead_b} | "
            f"{_fmt(r.get('per_token'))} | {ratio} |"
        )
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


def variance(cells: dict, *, bits: int) -> str:
    """Spread of D_sink across calibration draws, per model x granularity.

    Exists to keep C18 visible in the output: the per-token rows should read
    exactly 0.000000, and a reader should be able to see that rather than infer
    it from a suspiciously tight interval.
    """
    import statistics

    lines = [f"D_sink spread across the 5 calibration draws, {bits}-bit:", ""]
    for model in _ordered({k[0] for k in cells}):
        for gran in ("per_tensor", "per_token"):
            vals = []
            for seed in range(5):
                none = cells.get((model, bits, gran, "none", seed))
                exc = cells.get((model, bits, gran, "position_0", seed))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, action="append",
                        help="repeatable; default: every bit width present")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default="runs/quant")
    parser.add_argument("--by-bitwidth", action="store_true",
                        help="one granularity across every width (README §5.5)")
    parser.add_argument("--granularity", default="per_token")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="sensitivity of the destroyed-cell threshold (HANDOFF §12)")
    ns = parser.parse_args()

    cells = load_cells(ns.root)
    if ns.by_bitwidth:
        print(bitwidth_markdown(cells, gran=ns.granularity, seed=ns.seed))
        return
    if ns.threshold_sweep:
        print(threshold_sweep(cells, seed=ns.seed))
        return

    widths = ns.bits or sorted({k[1] for k in cells}, reverse=True)
    for bits in widths:
        print(markdown(cells, bits=bits, seed=ns.seed))
        print()
        print(variance(cells, bits=bits))
        print()


if __name__ == "__main__":
    main()
