"""Every figure in the README is produced here. No figure is made by hand.

Figure 1 is D_sink per model x activation granularity with bootstrap CIs. It
goes directly under the finding in the README, before any architecture prose.
Figure 2 is the same quantity against bit width. Figure 3 leaves perplexity
entirely and plots LAMBADA accuracy, which is the only panel here a reader can
interpret without knowing what a nat is.

Two conventions this module enforces rather than leaves to the caller:

  * CIs come from the SEQUENCE bootstrap, not the calibration-draw bootstrap.
    Dynamic per-token quantization never reads the calibration set, so draws
    give it a zero-width interval; sequences vary for every arm. See
    analysis.stats.sequence_bootstrap.
  * An interval that crosses zero is drawn hollow and annotated. A reader
    should not have to squint at whether an error bar touches the axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .stats import sequence_bootstrap  # noqa: E402

GRAN_COLOUR = {"per_tensor": "#b45309", "per_token": "#0f766e"}
GRAN_LABEL = {"per_tensor": "per-tensor (2023 setting)", "per_token": "per-token (modern)"}

# Sink-bearing models first, then the two gated arms, so a reader meets the
# controlled pair (baseline vs head-wise) adjacent to each other.
MODEL_ORDER = [
    "gpt2_small",
    "qwen3_0.6b_base",
    "gated_1b_baseline",
    "gated_1b_headwise",
    "gated_1b_elementwise",
]

# Every gated arm starts with "gated_1b_", so splitting on "_" labels all three
# of them "gated" and the figure silently loses the comparison it exists to show.
SHORT = {
    "gpt2_small": "GPT-2 small",
    "qwen3_0.6b_base": "Qwen3-0.6B",
    "gated_1b_baseline": "1B baseline\n(ungated)",
    "gated_1b_headwise": "1B head-wise\n(+0.1%)",
    "gated_1b_elementwise": "1B element-wise\n(+12%)",
}


def ordered_models(models) -> list[str]:
    known = [m for m in MODEL_ORDER if m in models]
    return known + sorted(set(models) - set(MODEL_ORDER))


def short(name: str) -> str:
    return SHORT.get(name, name)


def load_cells(root: str = "runs/quant") -> dict:
    cells = {}
    for p in Path(root).glob("*.json"):
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        cells[(d["model"], d["bits"], d["act_granularity"], d["fp16_exception"], d["calib_seed"])] = d
    if not cells:
        raise FileNotFoundError(f"no quant cells under {root}")
    return cells


# A quantized model whose perplexity is more than this multiple of its own
# reference is treated as destroyed: D_sink is then a difference between two
# broken models and carries no information about which degrades more gracefully
# (LIMITATIONS §19). The threshold is a judgement call and is stated rather than
# tuned — at 10x, an unquantized ppl of 14.5 has to reach 145 to be flagged,
# which no working language model does.
DESTROYED_PPL_RATIO = 10.0


def is_destroyed(cells: dict, model: str, bits: int, gran: str, seed: int = 0) -> bool:
    """Has the fully-quantized arm left the regime where perplexity means anything?

    Read off the `none` arm, which is the undefended cell: if that survives, the
    exception arm it is compared against survives too.
    """
    cell = cells.get((model, bits, gran, "none", seed))
    if not cell or not cell["ppl_ref"]:
        return False
    return cell["ppl_quant"] / cell["ppl_ref"] > DESTROYED_PPL_RATIO


def d_sink_intervals(cells: dict, *, bits: int, seed: int = 0, key: str = "per_seq_all"):
    """(model, granularity) -> Interval, from the stored per-sequence arrays."""
    out = {}
    models = sorted({k[0] for k in cells})
    for m in models:
        for gran in ("per_tensor", "per_token"):
            a = cells.get((m, bits, gran, "none", seed))
            b = cells.get((m, bits, gran, "position_0", seed))
            if not a or not b:
                continue
            out[(m, gran)] = sequence_bootstrap(
                a["quantized_means"][key], b["quantized_means"][key]
            )
    return out


def fig1_d_sink(cells: dict, *, bits: int = 8, out: str = "runs/results/fig1_d_sink.png"):
    """THE figure: sink-attributable damage, by model and activation granularity."""
    intervals = d_sink_intervals(cells, bits=bits)
    models = ordered_models({m for m, _ in intervals})[::-1]  # first model on top
    if not models:
        raise ValueError(f"no complete none/position_0 pairs at {bits} bits")

    fig, ax = plt.subplots(figsize=(9, 0.9 * len(models) + 2.6))
    h = 0.34

    for i, m in enumerate(models):
        for j, gran in enumerate(("per_tensor", "per_token")):
            iv = intervals.get((m, gran))
            if iv is None:
                continue
            y = i + (h / 2 if j == 0 else -h / 2)
            hollow = iv.crosses_zero
            dead = is_destroyed(cells, m, bits, gran)
            ax.plot(
                [iv.lo, iv.hi], [y, y],
                color=GRAN_COLOUR[gran], lw=2.4, solid_capstyle="round",
                alpha=0.45 if (hollow or dead) else 1.0, zorder=2,
            )
            ax.scatter(
                [iv.point], [y], s=62, zorder=3,
                marker="X" if dead else "o",
                facecolor="white" if (hollow and not dead) else GRAN_COLOUR[gran],
                edgecolor=GRAN_COLOUR[gran], linewidth=1.8,
            )
            # A destroyed cell must be annotated even when its interval looks
            # tidy: the number is a difference between two broken models, and
            # nothing about the error bar reveals that.
            note = ("model destroyed — not interpretable" if dead
                    else "CI crosses zero" if hollow else None)
            if note:
                ax.annotate(
                    note, (iv.hi, y), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color="#b91c1c" if dead else GRAN_COLOUR[gran], style="italic",
                )

    ax.axvline(0, color="#111", lw=1.0, zorder=1)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short(m) for m in models], fontsize=9)
    ax.set_ylim(-0.7, len(models) - 0.3)
    ax.set_xlabel(r"$D_{sink}$  (nats of perplexity damage removed by holding position 0 in fp16)")
    ax.set_title(
        f"Sink-attributable quantization damage at {bits}-bit activations\n"
        "higher = the sink matters more; zero = holding it in fp16 buys nothing",
        fontsize=11, loc="left",
    )
    handles = [
        plt.Line2D([], [], color=GRAN_COLOUR[g], lw=2.4, marker="o",
                   markerfacecolor=GRAN_COLOUR[g], label=GRAN_LABEL[g])
        for g in ("per_tensor", "per_token")
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_bitwidth(cells: dict, out: str = "runs/results/fig2_bitwidth.png", widths=None):
    """Does the per-token arm's advantage survive as the bit width falls?

    Reported separately from Figure 1 because the 4-bit cells sit in a regime
    where the model is largely destroyed and perplexity differences stop being
    interpretable. Kept visible rather than dropped, with the caveat drawn on —
    and 6-bit exists precisely to put an interpretable point between the two.

    Widths default to every one present in `cells`, descending, so adding a bit
    width to the grid adds it to the figure rather than silently not appearing.
    """
    widths = widths or sorted({k[1] for k in cells}, reverse=True)
    intervals = {b: d_sink_intervals(cells, bits=b) for b in widths}

    models = ordered_models({k[0] for k in cells})
    fig, ax = plt.subplots(figsize=(0.95 * len(widths) * len(models) + 3.0, 5.0))
    bw, span = 0.34, 1.05 * len(widths) + 0.9
    labelled = set()

    for i, m in enumerate(models):
        for bi, bits in enumerate(widths):
            for j, gran in enumerate(("per_tensor", "per_token")):
                iv = intervals[bits].get((m, gran))
                if iv is None:
                    continue
                x = i * span + bi * 1.05 + (j - 0.5) * bw
                dead = is_destroyed(cells, m, bits, gran)
                label = None
                if not dead and gran not in labelled:
                    label, _ = GRAN_LABEL[gran], labelled.add(gran)
                ax.bar(
                    [x], [iv.point], width=bw,
                    color="white" if dead else GRAN_COLOUR[gran],
                    edgecolor=GRAN_COLOUR[gran], linewidth=1.1,
                    hatch="////" if dead else None,
                    label=label,
                )

    # Symlog: D_sink spans four orders of magnitude across bit widths, and a
    # linear axis would render every 8-bit bar as a flat line at zero.
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylabel(r"$D_{sink}$ (nats, symlog)")
    ax.set_xticks([i * span + b * 1.05 for i in range(len(models)) for b in range(len(widths))])
    ax.set_xticklabels([f"{b}b" for _ in models for b in widths], fontsize=8)
    # Model name once per group, under the bit-width ticks. Labelling every bar
    # with the model name renders all three gated arms as the same word.
    centre = (len(widths) - 1) * 1.05 / 2
    for i, m in enumerate(models):
        ax.text(i * span + centre, -0.13, short(m).replace("\n", " "),
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5)
    ax.set_title(
        "Bit-width dependence of sink-attributable damage\n"
        "hatched: quantized perplexity > 10x the reference — the model is "
        "destroyed and the difference is not interpretable",
        fontsize=10, loc="left",
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, lw=0.6)
    fig.tight_layout()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def lambada_rows(runs: dict) -> list[dict]:
    """Everything Figure 3 draws, as numbers, before any of it becomes ink.

    Split out from `fig_lambada` so the figure's geometry can be asserted
    against the tables in `analysis.lambada` rather than eyeballed in a PNG. A
    figure that disagreed with the table beside it is the failure mode C19 had
    in a different form: two paths to the same number, only one of them checked.

    Draw order, bottom row first, so the caller's y index is the list index.
    """
    from .lambada import SATURATION_FLOOR, drop_interval

    models = ordered_models({m for m, _ in runs})[::-1]  # first model on top
    models = [m for m in models
              if (m, "per_tensor") in runs and (m, "per_token") in runs]
    if not models:
        raise ValueError("no model has both a per-tensor and a per-token LAMBADA cell")

    rows = []
    for m in models:
        for gran in ("per_tensor", "per_token"):
            r = runs[(m, gran)]
            iv = drop_interval(r)
            rows.append({
                "model": m,
                "gran": gran,
                "acc_ref": r["accuracy_ref"],
                "acc": r["accuracy_quant"],
                "drop": iv.point,
                # The drop's interval, mapped onto the accuracy axis. hi drop =
                # lo accuracy, so the ends swap.
                "ci_lo": r["accuracy_ref"] - iv.hi,
                "ci_hi": r["accuracy_ref"] - iv.lo,
                "null": iv.crosses_zero,
                "floored": r["accuracy_quant"] <= SATURATION_FLOOR,
            })
    return rows


def fig_lambada(runs: dict, *, out: str = "runs/results/fig3_lambada.png"):
    """The only figure here that is not perplexity: LAMBADA accuracy, by arm.

    Reads the same cells `analysis.lambada` tabulates, and shows the one thing a
    table of five-decimal drops does not: how far each model falls, on a scale
    where the reader already knows what the numbers mean.

    Three conventions, each inherited from a mistake this project already made.

      * The fp16 score is drawn as a grey rule across the row, not as a third
        bar. The quantity of interest is the FALL from it, and a chart of three
        adjacent bars invites reading the two quantized arms against each other
        while the reference floats somewhere behind them.
      * The interval belongs to the DROP, not to the accuracy. It is the paired
        bootstrap over examples that `analysis.lambada.drop_interval` computes,
        mapped back onto the accuracy axis as `acc_ref - [hi, lo]`. Bootstrapping
        the two accuracies separately would throw the pairing away, which is
        trap 9.8 in a second metric. One visible consequence, left in rather
        than clipped: every resample varies the REFERENCE too, so a bar drawn
        against the full-sample `acc_ref` can extend slightly past 0 on a cell
        at the floor. That is the interval the pairing actually produces.
      * A cell at the accuracy floor is annotated, not merely plotted low. Once a
        model is at chance the metric has stopped ordering anything, and nothing
        about a short bar says so.

    `analysis.lambada` imports from this module, so the reverse edge is deferred
    to call time. The floor and the interval are single-sourced there on purpose:
    a second definition of either would be free to drift from the tables.

    The bit width is read off the cells rather than passed in. A `bits=`
    argument that only reached the title would let a caller label an 8-bit
    figure as 6-bit, which is a caption that lies about its own data.
    """
    from .lambada import SATURATION_FLOOR

    rows = lambada_rows(runs)
    models = list(dict.fromkeys(r["model"] for r in rows))

    widths = {runs[(r["model"], r["gran"])].get("bits") for r in rows}
    if len(widths) != 1:
        raise ValueError(f"cells span more than one bit width: {sorted(widths)}")
    bits = widths.pop()

    fig, ax = plt.subplots(figsize=(10.0, 0.95 * len(models) + 3.1))
    h = 0.34

    for i, m in enumerate(models):
        acc_ref = runs[(m, "per_tensor")]["accuracy_ref"]
        # One grey rule per model: the score every arm in the row falls from.
        ax.plot([acc_ref, acc_ref], [i - h, i + h],
                color="#404040", lw=2.0, solid_capstyle="butt", zorder=4)
        for j, gran in enumerate(("per_tensor", "per_token")):
            row = next(r for r in rows if r["model"] == m and r["gran"] == gran)
            y = i + (h / 2 if j == 0 else -h / 2)
            acc, null, floored = row["acc"], row["null"], row["floored"]
            ci_lo, ci_hi = row["ci_lo"], row["ci_hi"]

            # The fall, drawn from the reference rule to the quantized score.
            ax.plot([acc, acc_ref], [y, y], color=GRAN_COLOUR[gran],
                    lw=1.2, alpha=0.55, zorder=2)
            # The CI is on the drop; on this axis that is acc_ref - [hi, lo].
            ax.plot([ci_lo, ci_hi], [y, y],
                    color=GRAN_COLOUR[gran], lw=2.6, solid_capstyle="round",
                    alpha=0.45 if null else 1.0, zorder=3)
            ax.scatter([acc], [y], s=62, zorder=5,
                       marker="X" if floored else "o",
                       facecolor="white" if (null and not floored) else GRAN_COLOUR[gran],
                       edgecolor=GRAN_COLOUR[gran], linewidth=1.8)

            # The floor note is lifted clear of the row: it is long, and at the
            # floor the drop bar runs the full width of the axes underneath it.
            note = ("at the accuracy floor — the metric has stopped ordering" if floored
                    else "CI crosses zero" if null else None)
            if note:
                ax.annotate(note, (ci_hi, y), xytext=(10, 11 if floored else 0),
                            textcoords="offset points", va="center", fontsize=8,
                            color="#b91c1c" if floored else GRAN_COLOUR[gran],
                            style="italic")

    ax.axvline(SATURATION_FLOOR, color="#b91c1c", lw=0.9, ls=":", zorder=1)
    ax.annotate(f"accuracy floor ({SATURATION_FLOOR:.0%})", (SATURATION_FLOOR, len(models) - 0.45),
                xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                color="#b91c1c", va="center")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short(m) for m in models], fontsize=9)
    # The extra room below the bottom row is where the legend goes, so it never
    # lands on top of the element-wise per-token marker at 0.59.
    ax.set_ylim(-1.25, len(models) - 0.3)
    ax.set_xlim(-0.02, 0.80)
    ax.set_xlabel("LAMBADA accuracy (greedy exact-match on the final word)")
    ax.set_title(
        f"What {bits}-bit activation quantization costs in behaviour\n"
        "grey rule = fp16; marker = quantized; bar = 95% paired bootstrap on the drop",
        fontsize=11, loc="left",
    )
    handles = [plt.Line2D([], [], color="#404040", lw=2.0, label="fp16 reference")]
    handles += [
        plt.Line2D([], [], color=GRAN_COLOUR[g], lw=2.6, marker="o",
                   markerfacecolor=GRAN_COLOUR[g], label=GRAN_LABEL[g])
        for g in ("per_tensor", "per_token")
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main() -> None:
    from .lambada import load_runs

    cells = load_cells()
    print("wrote", fig1_d_sink(cells, bits=8))
    print("wrote", fig_bitwidth(cells))

    # Figure 3 needs `runs/lambada`, which the grid does not produce. Say so
    # rather than exiting 0 with two figures where the README expects three.
    runs = load_runs()
    if runs:
        print("wrote", fig_lambada(runs))
    else:
        print("skipped fig3: no LAMBADA runs under runs/lambada "
              "(python -m quant.lambada --model <m> ...)")


if __name__ == "__main__":
    main()
