"""Which conclusions survive a change of corpus, and which were artefacts of one.

LIMITATIONS §18 has said since the beginning that every number in this project
rests on a single corpus, and that the one swap already performed (the repo's
own markdown → FineWeb-Edu) moved the headline matched-pair reduction by ~5×
and flipped two CIs across zero. Two corpora demonstrate that the sensitivity
exists. They cannot bound it, because nothing separates "different sample of
similar text" from "different domain".

`data/code_python.txt` is the third point and deliberately the most distant one
available: Python source against educational web prose. This module joins the
two grids and reports what moved.

The design rule it exists to enforce: **a claim that only holds on one corpus is
a property of that corpus.** So nothing here asserts stability — every verdict
is computed from the two grids and printed with the numbers behind it, including
the ones that come back negative. The ranking half of R6 was retracted on the
strength of this module's output (C21), which is the point of writing it.

Reads `runs/quant*` (grids) and `runs/dist*` / `runs/diag*` (R6). Loads no
weights: design rule §7.2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .distributions import ANNIHILATED, load_diag, load_dist, profile
from .figures import load_cells, ordered_models, short
from .report import rows

# The matched pair the audit turns on: same lab, same data, same tokenizer,
# differing only by `headwise_attn_output_gate` in config.json.
MATCHED_PAIR = ("gated_1b_baseline", "gated_1b_headwise")


def _dsink(table, model, gran):
    r = next((x for x in table if x["model"] == model), None)
    return (r or {}).get(gran)


def dsink_markdown(cells_a, cells_b, *, label_a: str, label_b: str,
                   bits: int = 8, seed: int = 0) -> str:
    """`D_sink` on both corpora, per model × granularity."""
    ta, tb = rows(cells_a, bits=bits, seed=seed), rows(cells_b, bits=bits, seed=seed)
    models = [r["model"] for r in ta]

    def fmt(iv):
        if iv is None:
            return "—"
        s = f"{iv.point:+.4f}"
        return s + " *ZERO*" if iv.crosses_zero else s

    out = [
        f"`D_sink` at {bits} bits, nats, draw {seed}. *ZERO* = 95% "
        "sequence-bootstrap interval contains zero.",
        "",
        f"| model | per-tensor, {label_a} | per-tensor, {label_b} | "
        f"per-token, {label_a} | per-token, {label_b} |",
        "|---|---|---|---|---|",
    ]
    for m in models:
        out.append(
            f"| `{m}` | {fmt(_dsink(ta, m, 'per_tensor'))} | "
            f"{fmt(_dsink(tb, m, 'per_tensor'))} | "
            f"{fmt(_dsink(ta, m, 'per_token'))} | "
            f"{fmt(_dsink(tb, m, 'per_token'))} |"
        )
    return "\n".join(out)


def damage_markdown(cells_a, cells_b, *, label_a: str, label_b: str,
                    bits: int = 8, seed: int = 0) -> str:
    """Damage level and reference perplexity, which `D_sink` alone cannot show.

    Carried because R4 exists: a `D_sink` is only interpretable while the
    quantized model still works, and whether a cell is destroyed can itself be
    corpus-dependent.
    """
    ta, tb = rows(cells_a, bits=bits, seed=seed), rows(cells_b, bits=bits, seed=seed)
    out = [
        f"Damage level, {bits} bits. **D** marks a cell above 10× its own reference.",
        "",
        f"| model | ppl_ref {label_a} | ppl_ref {label_b} | Δppl per-tensor {label_a} "
        f"| Δppl per-tensor {label_b} | Δppl per-token {label_a} | Δppl per-token {label_b} |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in ta:
        m = r["model"]
        s = next((x for x in tb if x["model"] == m), {})
        da = " **D**" if r.get("per_tensor_destroyed") else ""
        db = " **D**" if s.get("per_tensor_destroyed") else ""
        out.append(
            f"| `{m}` | {r.get('ppl_ref', float('nan')):.1f} | "
            f"{s.get('ppl_ref', float('nan')):.1f} | "
            f"{r.get('per_tensor_dppl', float('nan')):+.2f}{da} | "
            f"{s.get('per_tensor_dppl', float('nan')):+.2f}{db} | "
            f"{r.get('per_token_dppl', float('nan')):+.4f} | "
            f"{s.get('per_token_dppl', float('nan')):+.4f} |"
        )
    return "\n".join(out)


def matched_pair(cells, *, bits: int = 8, seed: int = 0) -> dict:
    """The audit's headline: per-tensor `D_sink`, ungated baseline → head-wise gate.

    Returned as a dict rather than printed so the two corpora can be compared on
    it directly. `reduction` is None when the numerator's interval crosses zero,
    because a ratio of noise to noise is not a reduction.
    """
    table = rows(cells, bits=bits, seed=seed)
    base, head = (_dsink(table, m, "per_tensor") for m in MATCHED_PAIR)
    if base is None or head is None:
        return {}
    usable = not base.crosses_zero and head.point != 0
    return {
        "baseline": base.point,
        "headwise": head.point,
        "headwise_crosses_zero": head.crosses_zero,
        "reduction": (base.point / head.point) if usable else None,
    }


def verdict(cells_a, cells_b, *, label_a: str, label_b: str,
            bits: int = 8, seed: int = 0) -> str:
    """Per-claim stability, computed rather than asserted."""
    ma, mb = (matched_pair(c, bits=bits, seed=seed) for c in (cells_a, cells_b))
    ta, tb = rows(cells_a, bits=bits, seed=seed), rows(cells_b, bits=bits, seed=seed)

    lines = [f"Claim stability, {label_a} vs {label_b}, {bits} bits.", ""]

    # 1. The fix works: baseline -> head-wise, per-tensor.
    lines += ["**R3-rev, the fix works.** Per-tensor `D_sink` on the matched pair:", ""]
    for lbl, m in ((label_a, ma), (label_b, mb)):
        # An absent arm is reported, not skipped and not crashed on: a grid
        # that is missing half the matched pair must not silently drop the row
        # the audit's headline is computed from.
        if not m:
            lines.append(f"- {lbl}: matched pair not present in this grid "
                         f"({' and '.join(MATCHED_PAIR)} required)")
            continue
        red = f"{m['reduction']:.0f}×" if m.get("reduction") else "not computable"
        zero = " (interval crosses zero)" if m.get("headwise_crosses_zero") else ""
        lines.append(
            f"- {lbl}: {m['baseline']:+.4f} → {m['headwise']:+.4f}{zero}, "
            f"a **{red}** reduction"
        )

    # 2. Redundancy: per-token damage vs per-tensor damage.
    lines += ["", "**R3-rev, and it is redundant.** Per-token `D_sink` per model:", ""]
    for lbl, t in ((label_a, ta), (label_b, tb)):
        pts = [(r["model"], r["per_token"]) for r in t if r.get("per_token")]
        n_zero = sum(1 for _, iv in pts if iv.crosses_zero)
        n_neg = sum(1 for _, iv in pts if iv.point < 0)
        biggest = max((abs(iv.point) for _, iv in pts), default=0.0)
        lines.append(
            f"- {lbl}: {n_zero}/{len(pts)} intervals contain zero, "
            f"{n_neg}/{len(pts)} point estimates are negative, "
            f"largest |D_sink| = {biggest:.4f} nats"
        )

    # 3. R4: is the element-wise arm destroyed on both?
    lines += ["", "**R4, the element-wise arm is destroyed.**", ""]
    for lbl, t in ((label_a, ta), (label_b, tb)):
        r = next((x for x in t if x["model"] == "gated_1b_elementwise"), {})
        ref, d = r.get("ppl_ref"), r.get("per_tensor_dppl")
        if ref is None or d is None:
            lines.append(f"- {lbl}: `gated_1b_elementwise` not present in this grid")
            continue
        lines.append(
            f"- {lbl}: Δppl {d:+.2f} against ppl_ref {ref:.1f} = "
            f"**{(ref + d) / ref:.0f}×** the reference — "
            f"{'destroyed' if r.get('per_tensor_destroyed') else 'NOT destroyed'}"
        )

    # 4. Does anything change destroyed-status between corpora?
    flipped = []
    for r in ta:
        s = next((x for x in tb if x["model"] == r["model"]), {})
        for g in ("per_tensor", "per_token"):
            if bool(r.get(f"{g}_destroyed")) != bool(s.get(f"{g}_destroyed")):
                flipped.append(f"{r['model']} {g}")
    lines += ["", "**Destroyed-status flips between corpora:** "
                  + (", ".join(flipped) if flipped else "none"), ""]
    return "\n".join(lines)


def bitwidth_stability(cells_a, cells_b, *, label_a: str, label_b: str,
                       widths=(8, 6, 4), seed: int = 0) -> str:
    """R5's bit-width claims, one printed number per sentence, on both corpora.

    R5 was written as prose against a single corpus, and C21 established that
    exactly this kind of claim — an ordering read off a grid, with no
    intervention behind it — is the kind that does not travel. Each block below
    corresponds to a sentence in README §5.5 so the sentence can be checked
    rather than believed.

    The per-tensor block reports **survivors** rather than damage, because that
    is the form R5's claim takes ("per-tensor does not survive to 6 bits, the
    single exception being head-wise"). A survivor is a cell at or below 10× its
    own reference — `analysis.figures.DESTROYED_PPL_RATIO`, whose sensitivity
    HANDOFF §12 already documents, and whose borderline cell is precisely the
    head-wise one this table is about.
    """
    def table(cells, bits):
        return rows(cells, bits=bits, seed=seed)

    out = [f"R5's bit-width claims on both corpora, draw {seed}.", ""]

    # 1. "The redundancy weakens": per-token |D_sink| and its 8 -> 6 growth.
    # The growth column needs two widths to exist; with one, the table is still
    # worth printing and the column is simply absent rather than a crash.
    pair = widths[:2]
    wide, narrow = (pair[0], pair[1]) if len(pair) > 1 else (pair[0], None)
    growth_hdr = f" | {wide}→{narrow} {label_a} | {wide}→{narrow} {label_b}" if narrow else ""
    out += [f"**Per-token `D_sink`"
            + (f", and how it grows from {wide} to {narrow} bits.**" if narrow else ".**")
            + " A magnitude, because the sign is not stable (§5.7).", "",
            "| model | " + " | ".join(f"{b}b {lbl}" for b in pair
                                      for lbl in (label_a, label_b))
            + growth_hdr + " |",
            "|---" * (1 + 2 * len(pair) + (2 if narrow else 0)) + "|"]
    for r in table(cells_a, wide):
        m = r["model"]
        vals = {(lbl, b): _dsink(table(cells, b), m, "per_token")
                for lbl, cells in ((label_a, cells_a), (label_b, cells_b))
                for b in pair}
        cols = []
        for b in pair:
            for lbl in (label_a, label_b):
                iv = vals[(lbl, b)]
                cols.append("—" if iv is None else
                            f"{abs(iv.point):.4f}" + ("*" if iv.crosses_zero else ""))
        if narrow:
            for lbl in (label_a, label_b):
                hi, lo = vals[(lbl, narrow)], vals[(lbl, wide)]
                # Magnitudes, not signed values: per-token D_sink changes sign
                # between corpora (§5.7), and a signed ratio would report a
                # doubling of damage as a negative "shrinkage".
                cols.append("—" if not hi or not lo or lo.point == 0
                            else f"{abs(hi.point) / abs(lo.point):.1f}×")
        out.append(f"| `{m}` | " + " | ".join(cols) + " |")
    out += ["", "`*` = the interval contains zero, so the magnitude is an upper "
                "bound on something indistinguishable from no effect."]

    # 2. "N of five exclude zero at 6 bits against M of five at 8."
    out += ["", "**How many per-token intervals exclude zero**, which is the "
                "form R5's weakening claim takes.", "",
            "| corpus | " + " | ".join(f"{b}-bit" for b in widths) + " |",
            "|---" * (len(widths) + 1) + "|"]
    for lbl, cells in ((label_a, cells_a), (label_b, cells_b)):
        cols = []
        for b in widths:
            t = table(cells, b)
            ivs = [r["per_token"] for r in t if r.get("per_token")]
            cols.append(f"{sum(1 for iv in ivs if not iv.crosses_zero)}/{len(ivs)}")
        out.append(f"| {lbl} | " + " | ".join(cols) + " |")

    # 3. "Per-tensor does not survive to 6 bits, the exception being head-wise."
    out += ["", "**Which models survive per-tensor**, i.e. sit at or below 10× "
                "their own reference. This is the sentence C21's lesson applies "
                "to most directly.", "",
            "| corpus | " + " | ".join(f"{b}-bit" for b in widths) + " |",
            "|---" * (len(widths) + 1) + "|"]
    for lbl, cells in ((label_a, cells_a), (label_b, cells_b)):
        cols = []
        for b in widths:
            alive = [r["model"] for r in table(cells, b)
                     if r.get("per_tensor_dppl") is not None
                     and not r.get("per_tensor_destroyed")]
            cols.append(", ".join(x.replace("gated_1b_", "") for x in alive) or "**none**")
        out.append(f"| {lbl} | " + " | ".join(cols) + " |")

    # 4. The direction, which is a different claim from the survivor count.
    out += ["", "**Least-damaged model under per-tensor**, per width. The "
                "*direction* is a separate claim from the survivor count above, "
                "and the two do not have to travel together.", "",
            "| corpus | " + " | ".join(f"{b}-bit" for b in widths) + " |",
            "|---" * (len(widths) + 1) + "|"]
    for lbl, cells in ((label_a, cells_a), (label_b, cells_b)):
        cols = []
        for b in widths:
            t = [r for r in table(cells, b) if r.get("per_tensor_dppl") is not None]
            if not t:
                cols.append("—")
                continue
            best = min(t, key=lambda r: r["per_tensor_dppl"])
            others = [r["per_tensor_dppl"] for r in t if r is not best]
            margin = (min(others) / best["per_tensor_dppl"]) if others and best[
                "per_tensor_dppl"] > 0 else float("nan")
            cols.append(f"{best['model'].replace('gated_1b_', '')} "
                        f"(+{best['per_tensor_dppl']:.2f}, {margin:.0f}× better than next)")
        out.append(f"| {lbl} | " + " | ".join(cols) + " |")
    return "\n".join(out)


def r6_markdown(dist_a, diag_a, dist_b, diag_b, *, label_a: str, label_b: str,
                layer: str = "model.layers.0.mlp.gate_proj",
                threshold: float = ANNIHILATED) -> str:
    """R6 on both corpora: the layer-0 tensor, and the cross-model ranking.

    Split deliberately. The localisation is a property of one tensor on one
    checkpoint and is checked directly; the ranking is a claim about five points
    and is checked by whether the ordering survives at all. They came back
    differently, which is why they are no longer reported as one finding.
    """
    out = [f"**The layer-0 tensor** — `{layer}`.", "",
           f"| model | dispersion {label_a} | dispersion {label_b} | "
           f"underflow/tensor {label_a} | underflow/tensor {label_b} |",
           "|---|---|---|---|---|"]
    for m in ordered_models(set(dist_a) & set(dist_b)):
        la, lb = dist_a[m]["layers"], dist_b[m]["layers"]
        if layer not in la or layer not in lb:
            continue
        out.append(
            f"| {short(m).replace(chr(10), ' ')} | {la[layer]['dispersion']:.1f}× | "
            f"{lb[layer]['dispersion']:.1f}× | {la[layer]['underflow_tensor']:.4f} | "
            f"{lb[layer]['underflow_tensor']:.4f} |"
        )

    def ordered(vals):
        """Non-decreasing AND actually separating.

        Ties are allowed — two models the statistic cannot separate is a weaker
        result, not a wrong one. A column where EVERY count is identical is
        different: it is trivially non-decreasing while ranking nothing, which
        happens at thresholds strict enough that no layer is flagged anywhere.
        Crediting it would let "orders the roster" mean "flagged nothing".
        """
        return len(set(vals)) > 1 and all(x <= y for x, y in zip(vals, vals[1:]))

    out += ["", "**The cross-model ranking** — annihilated-layer count against "
                "per-tensor damage, both corpora, at every threshold.", "",
            "| corpus | " + " | ".join(f">{t:.0%}" for t in (0.5, 0.7, 0.9, 0.95, 0.99))
            + " | orders damage at |", "|---" * 7 + "|"]
    for lbl, dist, diag in ((label_a, dist_a, diag_a), (label_b, dist_b, diag_b)):
        rs = profile(dist, diag, threshold)
        cols, ok = [], []
        for t in (0.5, 0.7, 0.9, 0.95, 0.99):
            counts = [sum(1 for v in dist[r["model"]]["layers"].values()
                          if v.get("underflow_tensor", 0) > t) for r in rs]
            cols.append(",".join(str(c) for c in counts))
            if ordered(counts):
                ok.append(f"{t:.0%}")
        out.append(f"| {lbl} | " + " | ".join(cols) + " | "
                   + (", ".join(ok) if ok else "**no threshold**") + " |")
    out += ["", "Counts are listed in increasing order of per-tensor damage, so a "
                "row that is not non-decreasing is a threshold at which the "
                "statistic does not rank the roster."]
    return "\n".join(out)


def localisation_markdown(diag_root_a: str, diag_root_b: str, *, label_a: str,
                          label_b: str, model: str = "gated_1b_elementwise",
                          bits: int = 8, arm: str = "a_only_dynamic") -> str:
    """The causal test on both corpora, exemption by exemption."""
    def runs(root):
        out = {}
        for path in sorted(Path(root).glob(f"{model}_b{bits}_diag*.json")):
            with open(path, encoding="utf-8") as fh:
                run = json.load(fh)
            d = {a["arm"]: a["delta_ppl"] for a in run["arms"]}
            if arm in d:
                out[tuple(run.get("skip_modules") or ())] = (d[arm], run["ppl_ref"])
        return out

    ra, rb = runs(diag_root_a), runs(diag_root_b)
    ca, cb = ra.get(()), rb.get(())

    out = [f"**The causal test** — `{model}`, `{arm}` at {bits} bits, both corpora.", "",
           f"| left in fp16 | Δppl {label_a} | vs control | Δppl {label_b} | vs control |",
           "|---|---|---|---|---|"]
    for skip in sorted(set(ra) | set(rb), key=lambda s: (len(s), s)):
        cells = []
        for r, ctrl in ((ra, ca), (rb, cb)):
            v = r.get(skip)
            if v is None:
                cells += ["—", "—"]
                continue
            d = v[0]
            rel = "—"
            if ctrl and d > 0 and ctrl[0] / d > 1.05:
                # Two decimals below 10x, as in analysis.distributions: an
                # exemption that recovers a third of the damage and one that
                # recovers 3000x must not both print "1x".
                r = ctrl[0] / d
                rel = f"{r:.0f}× better" if r >= 10 else f"{r:.2f}× better"
            if not skip:
                rel = "(control)"
            cells += [f"{d:+.2f}", rel]
        label = ", ".join(skip) if skip else "nothing *(control)*"
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-a", default="runs/quant")
    parser.add_argument("--root-b", default="runs/quant_code")
    parser.add_argument("--label-a", default="FineWeb-Edu")
    parser.add_argument("--label-b", default="code")
    parser.add_argument("--dist-a", default="runs/dist")
    parser.add_argument("--dist-b", default="runs/dist_code")
    parser.add_argument("--diag-a", default="runs/diag")
    parser.add_argument("--diag-b", default="runs/diag_code")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--bitwidth", action="store_true",
                        help="R5's bit-width claims on both corpora (README §5.5)")
    parser.add_argument("--seed", type=int, default=0)
    ns = parser.parse_args()

    ca, cb = load_cells(ns.root_a), load_cells(ns.root_b)
    kw = dict(label_a=ns.label_a, label_b=ns.label_b, bits=ns.bits, seed=ns.seed)

    if ns.bitwidth:
        print(bitwidth_stability(ca, cb, label_a=ns.label_a, label_b=ns.label_b,
                                 seed=ns.seed))
        return

    print(dsink_markdown(ca, cb, **kw)); print()
    print(damage_markdown(ca, cb, **kw)); print()
    print(verdict(ca, cb, **kw)); print()
    print(r6_markdown(load_dist(ns.dist_a, ns.bits), load_diag(ns.diag_a, ns.bits),
                      load_dist(ns.dist_b, ns.bits), load_diag(ns.diag_b, ns.bits),
                      label_a=ns.label_a, label_b=ns.label_b)); print()
    print(localisation_markdown(ns.diag_a, ns.diag_b, label_a=ns.label_a,
                                label_b=ns.label_b, bits=ns.bits))


if __name__ == "__main__":
    main()
