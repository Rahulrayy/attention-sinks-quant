"""Does the perplexity ordering reach behaviour? The LAMBADA tables.

Reads `runs/lambada` and joins it to `runs/quant`, because the question is not
"what is the accuracy" but "does the thing perplexity ranked show up in
something a user would notice". Every table here puts an accuracy drop next to
the `Δppl` it is supposed to correspond to.

Two reporting rules, both of which exist because accuracy is discrete.

**Report the discordant count, always.** On a paired 0/1 outcome every example
the two arms agree on contributes exactly zero to the difference, so the
discordant pairs are the entire sample. A drop of 0.0000 across 26 discordant
pairs is a real measurement of cancellation; a drop of 0.0000 across 0
discordant pairs is the absence of one. They must not print the same, which is
C18's lesson carried into a second metric.

**Never report an accuracy drop on a destroyed model as if it ranked
something.** `analysis.figures.is_destroyed` governs perplexity cells for the
reason LIMITATIONS §19 gives, and accuracy has its own version of the same
failure: once a model is at chance, further damage cannot lower its score, so
the metric saturates and stops ordering anything. The floor is reported next to
the score so a reader can see how much room was left.

Design rule §7.2: this module loads no weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .figures import is_destroyed, load_cells, ordered_models, short
from .stats import sequence_bootstrap

# LAMBADA's chance rate is not 1/vocab: the answer is a natural English word in
# context, and a model that has learned only unigram frequencies scores a few
# percent. Published near-chance systems sit around 1%, which is the level below
# which "worse" stops meaning anything.
SATURATION_FLOOR = 0.02


def load_runs(root: str = "runs/lambada", bits: int = 8) -> dict[tuple[str, str], dict]:
    """Every LAMBADA cell, keyed by (model, arm)."""
    out = {}
    for path in sorted(Path(root).glob(f"*_b{bits}_*_lambada.json")):
        with open(path, encoding="utf-8") as fh:
            run = json.load(fh)
        arm = run["act_granularity"] + ("" if run.get("static", True) else "_dynamic")
        out[(run["model"], arm)] = run
    return out


def drop_interval(run: dict):
    """Paired bootstrap over examples of (ref correct − quant correct).

    The same examples in the same order for both arms, so the difference is
    paired example-by-example — identical machinery to the sequence bootstrap
    every perplexity interval in this repo uses.
    """
    return sequence_bootstrap(
        run["per_example_correct_ref"], run["per_example_correct_quant"]
    )


def markdown(runs: dict, cells: dict | None = None, *, bits: int = 8,
             seed: int = 0) -> str:
    """The headline: accuracy damage beside the perplexity damage it should track."""
    models = ordered_models({m for m, _ in runs})
    out = [
        f"LAMBADA (`lambada_openai`), {bits}-bit activations. Accuracy is "
        "greedy exact-match on the final word; the interval is a 95% paired "
        "bootstrap over examples.",
        "",
        "| model | fp16 acc | per-tensor acc | drop | discordant | per-token acc "
        "| drop | discordant |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        pt, ptok = runs.get((m, "per_tensor")), runs.get((m, "per_token"))
        if not pt or not ptok:
            continue
        cells_ = []
        for r in (pt, ptok):
            iv = drop_interval(r)
            flag = " *ZERO*" if iv.crosses_zero else ""
            floor = " ⌊" if r["accuracy_quant"] <= SATURATION_FLOOR else ""
            cells_ += [f"{r['accuracy_quant']:.4f}{floor}",
                       f"{iv.point:+.4f} [{iv.lo:+.4f}, {iv.hi:+.4f}]{flag}",
                       str(r["n_discordant"])]
        out.append(f"| {short(m).replace(chr(10), ' ')} | {pt['accuracy_ref']:.4f} | "
                   + " | ".join(cells_) + " |")
    out += [
        "",
        "*Drop* is fp16 accuracy minus quantized accuracy, so positive means "
        "quantization cost accuracy. *Discordant* counts examples the two arms "
        "answered differently — the whole sample of a paired 0/1 comparison. A "
        "drop of zero across many discordant pairs is cancellation, not "
        "agreement.",
        f"⌊ marks a cell at or below {SATURATION_FLOOR:.0%} accuracy, where the "
        "metric has saturated and further damage cannot register.",
    ]
    if cells:
        out += ["", _join_to_perplexity(runs, cells, models, bits=bits, seed=seed)]
    return "\n".join(out)


def _join_to_perplexity(runs, cells, models, *, bits: int, seed: int) -> str:
    """The actual question: do the two metrics agree about what is damaged?"""
    out = [
        "Against the perplexity damage it is supposed to track. `Δppl` is the "
        "fully-quantized `none` cell from the grid, on FineWeb-Edu; the accuracy "
        "columns are LAMBADA.",
        "",
        "| model | Δppl per-tensor | acc drop per-tensor | Δppl per-token | "
        "acc drop per-token |",
        "|---|---|---|---|---|",
    ]
    for m in models:
        pt, ptok = runs.get((m, "per_tensor")), runs.get((m, "per_token"))
        if not pt or not ptok:
            continue
        row = [f"| {short(m).replace(chr(10), ' ')}"]
        for gran, r in (("per_tensor", pt), ("per_token", ptok)):
            cell = cells.get((m, bits, gran, "none", seed)) if cells else None
            dead = " **D**" if cell and is_destroyed(cells, m, bits, gran, seed) else ""
            row.append(f"{cell['delta_ppl']:+.2f}{dead}" if cell else "—")
            row.append(f"{r['accuracy_drop']:+.4f}")
        out.append(" | ".join(row) + " |")
    out += ["", "**D** = the perplexity cell is above 10× its reference and ranks "
                "nothing (LIMITATIONS §19). Its accuracy column still does, until "
                "the accuracy floor is reached."]
    return "\n".join(out)


def rank_agreement(runs: dict, cells: dict, *, bits: int = 8, seed: int = 0,
                   gran: str = "per_tensor") -> str:
    """Does the accuracy ordering reproduce the perplexity ordering?

    This is the question the task was run to answer, so it is computed rather
    than eyeballed. C21 retracted a claim because an ordering held on one corpus
    and not another; the same scepticism applies to an ordering that holds on
    one metric, and the only way to know is to rank the roster twice and compare.

    Saturated cells are named but excluded from the comparison. A model at the
    accuracy floor could be damaged a hundred times further without moving, so
    its rank is a lower bound rather than a position, and letting it vote would
    let the metric's ceiling masquerade as agreement.
    """
    rows = []
    for (m, arm), r in runs.items():
        if arm != gran:
            continue
        cell = cells.get((m, bits, gran, "none", seed))
        if not cell:
            continue
        rows.append({
            "model": m,
            "dppl": cell["delta_ppl"],
            "drop": r["accuracy_drop"],
            "floored": r["accuracy_quant"] <= SATURATION_FLOOR,
            "crosses_zero": drop_interval(r).crosses_zero,
        })
    if len(rows) < 2:
        return "(too few cells to rank)"

    usable = [r for r in rows if not r["floored"]]
    floored = [r["model"] for r in rows if r["floored"]]
    by_ppl = [r["model"] for r in sorted(usable, key=lambda r: r["dppl"])]
    by_acc = [r["model"] for r in sorted(usable, key=lambda r: r["drop"])]

    out = [
        f"Ordering under **{gran.replace('_', '-')}**, least damaged first. "
        "The two metrics are independent; whether they agree is the point.",
        "",
        "| ranked by | order |",
        "|---|---|",
        "| Δppl (perplexity, FineWeb-Edu) | " + " < ".join(
            m.replace("gated_1b_", "") for m in by_ppl) + " |",
        "| accuracy drop (LAMBADA) | " + " < ".join(
            m.replace("gated_1b_", "") for m in by_acc) + " |",
        "",
        f"**{'Identical' if by_ppl == by_acc else 'DIFFERENT'}** on the "
        f"{len(usable)} rankable models.",
    ]

    # An ordering built from cells that are individually indistinguishable from
    # zero is an ordering of noise, and agreement or disagreement between two
    # such orderings means nothing either way. Say so rather than let a verdict
    # of DIFFERENT read as a finding.
    n_null = sum(1 for r in usable if r["crosses_zero"])
    if n_null:
        out.append(
            f"⚠ {n_null} of {len(usable)} accuracy drops here have intervals "
            "containing zero, so this ordering is partly an ordering of "
            "non-effects. Agreement and disagreement are both uninformative to "
            "that extent — read the verdict only where the drops are real."
        )
    if floored:
        out.append(
            "Excluded as saturated: " + ", ".join(
                m.replace("gated_1b_", "") for m in floored)
            + " — at the accuracy floor, so its rank is a lower bound, not a "
              "position. It is the most damaged model on both metrics anyway."
        )
    return "\n".join(out)


def power(runs: dict, *, bits: int = 8) -> str:
    """What difference could this sample have detected, per cell.

    Stated because a null result is only informative alongside the effect it
    could have ruled out. The half-width of the paired interval is that number:
    a drop smaller than it would not have been distinguishable from zero here,
    whatever the true effect.
    """
    out = ["Resolution of each cell — the half-width of its paired interval, "
           "i.e. the smallest accuracy drop this sample could have called "
           "non-zero.", "",
           "| model | arm | n examples | discordant | drop | resolution |",
           "|---|---|---|---|---|---|"]
    for (m, arm), r in sorted(runs.items()):
        iv = drop_interval(r)
        out.append(f"| `{m}` | {arm} | {r['n_examples']} | {r['n_discordant']} | "
                   f"{iv.point:+.4f} | ±{iv.width / 2:.4f} |")
    out += ["", "A cell whose drop is smaller than its resolution is a bound, not "
                "a measurement. Widening the sample narrows this; nothing else "
                "does."]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--root", default="runs/lambada")
    parser.add_argument("--quant-root", default="runs/quant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--power", action="store_true",
                        help="per-cell resolution only")
    parser.add_argument("--ranks", action="store_true",
                        help="does the accuracy ordering match the perplexity one")
    ns = parser.parse_args()

    runs = load_runs(ns.root, ns.bits)
    if not runs:
        raise SystemExit(f"no LAMBADA runs under {ns.root} at {ns.bits} bits")

    if ns.power:
        print(power(runs, bits=ns.bits))
        return

    cells = load_cells(ns.quant_root)
    if ns.ranks:
        print(rank_agreement(runs, cells, bits=ns.bits, seed=ns.seed))
        return

    print(markdown(runs, cells, bits=ns.bits, seed=ns.seed))
    print()
    print(rank_agreement(runs, cells, bits=ns.bits, seed=ns.seed))
    print()
    print(power(runs, bits=ns.bits))


if __name__ == "__main__":
    main()
