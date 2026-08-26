"""The R6 tables: activation shape against the per-tensor damage it has to explain.

Reads `runs/dist` (quant.distributions) and `runs/diag` (quant.diagnose) and
joins them per model. Nothing here loads a checkpoint — design rule §7.2 — and
nothing here is transcribed into the README by hand, for the reason C19 records.

The join is the point. A distributional statistic is only interesting if it
ranks something, so every table in this module puts the statistic next to
`a_only_dynamic`, the arm it is supposed to account for. The dynamic arm is the
right target rather than `a_only_static` because it cannot clip, so a
correlation with it cannot be smuggled in by calibration coverage.

`ANNIHILATED` is the one threshold introduced here, and it is declared rather
than tuned: a layer counts as annihilated when a shared per-tensor scale rounds
more than 90% of its entries to zero. `--sweep` prints the roster ranking at
every threshold from 0.5 to 0.99 so a reader can check that the ordering does
not depend on where the line was drawn.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .figures import ordered_models, short

# A layer whose entries mostly round to zero under one shared scale. See the
# module docstring: the ranking is reported across a sweep of this value.
ANNIHILATED = 0.9

_BLOCK = re.compile(r"(?:layers|h)\.(\d+)\.")


def load_dist(root: str = "runs/dist", bits: int = 8) -> dict[str, dict]:
    """Every distribution run at one bit width, keyed by model."""
    out = {}
    for path in sorted(Path(root).glob(f"*_b{bits}_dist.json")):
        with open(path, encoding="utf-8") as fh:
            run = json.load(fh)
        out[run["model"]] = run
    return out


def load_diag(root: str = "runs/diag", bits: int = 8) -> dict[str, dict]:
    """Arm decompositions with no `--skip-modules`, keyed by model.

    Runs that exempted projections are skipped here on purpose: they are
    interventions, not the damage the distributional statistic has to explain.
    They are read separately by `localisation()`.
    """
    out = {}
    for path in sorted(Path(root).glob(f"*_b{bits}_diag.json")):
        with open(path, encoding="utf-8") as fh:
            run = json.load(fh)
        if run.get("skip_modules"):
            continue
        out[run["model"]] = {a["arm"]: a["delta_ppl"] for a in run["arms"]}
        out[run["model"]]["ppl_ref"] = run["ppl_ref"]
    return out


def blocks(layers: dict, threshold: float = ANNIHILATED) -> list[int]:
    """Sorted transformer block indices holding at least one annihilated layer."""
    idx = set()
    for name, stats in layers.items():
        if stats.get("underflow_tensor", 0.0) <= threshold:
            continue
        m = _BLOCK.search(name)
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)


def profile(dist: dict, diag: dict, threshold: float = ANNIHILATED) -> list[dict]:
    """One row per model, ordered by the damage the statistic has to explain."""
    rows = []
    for model, run in dist.items():
        layers = run["layers"]
        n_blocks = (max(blocks(layers, -1.0)) + 1) if layers else 0
        ann = [v for v in layers.values() if v.get("underflow_tensor", 0) > threshold]
        ann_tok = [v for v in layers.values() if v.get("underflow_token", 0) > threshold]
        bl = blocks(layers, threshold)
        rows.append({
            "model": model,
            "dynamic": diag.get(model, {}).get("a_only_dynamic"),
            "per_token": diag.get(model, {}).get("a_only_per_token"),
            "n_layers": len(layers),
            "n_blocks": n_blocks,
            "summary": run["summary"],
            "n_annihilated": len(ann),
            "n_annihilated_token": len(ann_tok),
            "annihilated_blocks": bl,
            "first_block": bl[0] if bl else None,
            "early_blocks": [b for b in bl if b < n_blocks / 2],
        })
    rows.sort(key=lambda r: (r["dynamic"] is None, r["dynamic"] or 0.0))
    return rows


def markdown(dist: dict, diag: dict, threshold: float = ANNIHILATED) -> str:
    """The roster table: does activation shape rank per-tensor damage?"""
    rows = profile(dist, diag, threshold)
    out = [
        "Activation shape under an 8-bit grid, against the per-tensor damage it "
        "has to explain. Rows sorted by damage.",
        "",
        "| model | Δppl per-tensor (dynamic) | Δppl per-token | median underflow "
        "per-tensor | median underflow per-token | annihilated layers | blocks | "
        "first block |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["summary"]
        out.append(
            f"| `{r['model']}` | {r['dynamic']:+.2f} | {r['per_token']:+.2f} | "
            f"{s['underflow_tensor_median']:.3f} | {s['underflow_token_median']:.3f} | "
            f"{r['n_annihilated']}/{r['n_layers']} | "
            f"{len(r['annihilated_blocks'])}/{r['n_blocks']} | "
            f"{r['first_block'] if r['first_block'] is not None else '—'} |"
        )
    out += [
        "",
        f"*Annihilated* = a shared per-tensor scale rounds more than "
        f"{threshold:.0%} of the layer's input entries to zero. The per-token "
        "column is the control: if it moved with the damage too, the statistic "
        "would be measuring the checkpoint rather than the granularity.",
    ]
    return "\n".join(out)


def token_control(dist: dict, diag: dict, threshold: float = ANNIHILATED) -> str:
    """The falsification check, stated as its own table rather than a footnote.

    Per-token scaling is nearly free on every model in the roster, so a
    statistic that explains the per-tensor spread must NOT also explain a
    per-token spread — there is not one to explain. Annihilation counts under
    per-row scales are printed here so that can be checked rather than asserted.
    """
    rows = profile(dist, diag, threshold)
    out = [
        "Control: the same count under PER-TOKEN scales, which the grid shows to "
        "be nearly free everywhere.",
        "",
        "| model | Δppl per-tensor | Δppl per-token | annihilated (per-tensor) | "
        "annihilated (per-token) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| `{r['model']}` | {r['dynamic']:+.2f} | {r['per_token']:+.2f} | "
            f"{r['n_annihilated']}/{r['n_layers']} | "
            f"{r['n_annihilated_token']}/{r['n_layers']} |"
        )
    return "\n".join(out)


def layer_table(dist: dict, layer: str, models=None) -> str:
    """One tensor across several checkpoints — the matched-pair contrast.

    Used for the layer-0 MLP input, where the three 1B arms differ by one
    boolean in config.json and by nothing else in the training pipeline.
    """
    models = models or ordered_models(dist)
    present = [m for m in models if m in dist and layer in dist[m]["layers"]]
    if not present:
        return f"(no distribution run carries `{layer}`)"

    out = [
        f"`{layer}` across the roster.",
        "",
        "| model | row dispersion | eff. bits (median row) | underflow per-tensor "
        "| underflow per-token |",
        "|---|---|---|---|---|",
    ]
    for m in present:
        v = dist[m]["layers"][layer]
        out.append(
            f"| {short(m).replace(chr(10), ' ')} | {v['dispersion']:.1f}× | "
            f"{v['eff_bits']:.2f} | {v['underflow_tensor']:.4f} | "
            f"{v['underflow_token']:.4f} |"
        )
    out += [
        "",
        "*Row dispersion* = amax(tensor) / median row amax: the factor by which "
        "one shared scale is too coarse for a typical token. It is exactly what "
        "per-token scaling divides out, and 1.0 means per-token buys nothing.",
    ]
    return "\n".join(out)


def localisation(root: str = "runs/diag", bits: int = 8, arm: str = "a_only_dynamic") -> str:
    """Every `--skip-modules` intervention, against its own unexempted control.

    This is the half of R6 that is causal rather than correlational, so the
    baseline row (nothing exempted) is printed alongside rather than left for
    the reader to remember.
    """
    runs = []
    for path in sorted(Path(root).glob(f"*_b{bits}_diag*.json")):
        with open(path, encoding="utf-8") as fh:
            run = json.load(fh)
        d = {a["arm"]: a["delta_ppl"] for a in run["arms"]}
        if arm not in d:
            continue
        runs.append((run["model"], tuple(run.get("skip_modules") or ()), d[arm],
                     run["ppl_ref"]))

    base = {m: v for m, s, v, _ in runs if not s}
    out = [
        f"fp16 exemption experiments, `{arm}` at {bits} bits. "
        "Δppl with nothing exempted is the control for each model.",
        "",
        "| model | exempted | modules left in fp16 | Δppl | ppl / ppl_ref | vs control |",
        "|---|---|---|---|---|---|",
    ]
    for model, skip, dppl, ppl_ref in sorted(runs, key=lambda r: (r[0], r[2])):
        ctrl = base.get(model)
        rel = "—"
        if ctrl and dppl > 0:
            ratio = ctrl / dppl
            # Two decimals below 10x: an exemption that recovers a third of the
            # damage and one that recovers 500x must not both print "1x".
            if ratio > 1.05:
                rel = f"{ratio:.0f}× better" if ratio >= 10 else f"{ratio:.2f}× better"
        if not skip:
            rel = "(control)"
        label = ", ".join(skip) if skip else "nothing"
        out.append(
            f"| `{model}` | {label} | {len(skip)} pattern(s) | {dppl:+.2f} | "
            f"{(ppl_ref + dppl) / ppl_ref:.2f}× | {rel} |"
        )
    return "\n".join(out)


def sweep(dist: dict, diag: dict, thresholds=(0.5, 0.7, 0.9, 0.95, 0.99)) -> str:
    """Does the roster ordering survive moving the annihilation threshold?

    Same discipline as report.threshold_sweep. A ranking that only holds at one
    cut-off is a fitted statistic, not a mechanism, and this is what shows the
    difference.
    """
    rows = profile(dist, diag)
    counts = {
        t: [sum(1 for v in dist[r["model"]]["layers"].values()
                if v.get("underflow_tensor", 0) > t) for r in rows]
        for t in thresholds
    }

    out = ["Annihilated-layer count by threshold. Rows sorted by per-tensor damage.", ""]
    out.append("| model | Δppl per-tensor | " +
               " | ".join(f">{t:.0%}" for t in thresholds) + " |")
    out.append("|---" * (len(thresholds) + 2) + "|")
    for i, r in enumerate(rows):
        cols = [str(counts[t][i]) for t in thresholds]
        out.append(f"| `{r['model']}` | {r['dynamic']:+.2f} | " + " | ".join(cols) + " |")

    # The ordering is the claim, not the counts, so it is checked here rather
    # than left to the reader. Ties count as ordered: two models the statistic
    # cannot separate is a weaker result, not a wrong one.
    def ordered(vals):
        return all(a <= b for a, b in zip(vals, vals[1:]))

    ok = [t for t in thresholds if ordered(counts[t])]
    bad = [t for t in thresholds if not ordered(counts[t])]
    out += [
        "",
        "Reproduces the damage ordering at: "
        + (", ".join(f"{t:.0%}" for t in ok) or "no threshold"),
        "",
        "Fails to at: " + (", ".join(f"{t:.0%}" for t in bad) or "none"),
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--dist-root", default="runs/dist")
    parser.add_argument("--diag-root", default="runs/diag")
    parser.add_argument("--threshold", type=float, default=ANNIHILATED)
    parser.add_argument("--layer", default="model.layers.0.mlp.gate_proj")
    parser.add_argument("--sweep", action="store_true", help="threshold sensitivity only")
    ns = parser.parse_args()

    dist = load_dist(ns.dist_root, ns.bits)
    diag = load_diag(ns.diag_root, ns.bits)
    if not dist:
        raise SystemExit(f"no distribution runs under {ns.dist_root} at {ns.bits} bits")

    if ns.sweep:
        print(sweep(dist, diag))
        return

    print(markdown(dist, diag, ns.threshold))
    print()
    print(token_control(dist, diag, ns.threshold))
    print()
    print(layer_table(dist, ns.layer))
    print()
    print(localisation(ns.diag_root, ns.bits))
    print()
    print(sweep(dist, diag))


if __name__ == "__main__":
    main()
