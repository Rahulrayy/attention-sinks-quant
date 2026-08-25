"""Perplexity + LAMBADA under a quantization config.

One JSON per config to runs/quant/<config>.json (design rule §7.1).

WHY D_sink IS DECOMPOSED HERE
-----------------------------
Plan §6 defines the headline metric as

    D_sink = dppl(everything quantized) - dppl(sink tokens kept fp16)

That number sums two effects which point the same way but mean very different
things:

  (a) SELF: the sink token's own prediction stops being corrupted. Holding a
      token in fp16 trivially removes that token's own quantization error, so
      (a) is large under EVERY granularity and is close to tautological.

  (b) CONTAMINATION: every OTHER token gets a tighter shared scale, because the
      outlier no longer drags the range out. This is the effect the mitigation
      literature actually claims, and it is the one that should collapse under
      per-token scaling.

Measured on a linear toy model with no cross-token mixing, per_token still
showed ~99% of per_tensor's D_sink at 4 bits — entirely effect (a), since (b)
is structurally impossible there. Reporting only the summed metric would have
made per-token scaling look almost as sink-sensitive as per-tensor and pointed
the audit at the wrong conclusion.

So: report the total for comparability with the plan, and report the non-sink
restriction as the headline. In a real decoder the two are not fully separable
(a corrupted sink propagates through attention), which is exactly why the
decomposition has to be stated rather than assumed away.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch

# The corpus the committed results were measured on. Preferred over streaming a
# fresh sample, because the corpus swap moved the headline by ~5x (data/README.md).
DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "fineweb_edu.txt"


def _int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.replace(" ", "").split(",") if x]


def _str_list(spec: str) -> list[str]:
    return [x for x in spec.replace(" ", "").split(",") if x]


def _sha256_prefix(path: str, n: int = 16) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:n]


def load_corpus(path: str) -> list[str]:
    """Read a corpus file as ONE string, for tokenize_stream.

    Not a stylistic choice. Feeding the file line-by-line tokenizes each line
    independently, which moves BPE boundaries at every newline and drops the
    blank lines separating documents — the two streams diverge at token 846 on
    `data/fineweb_edu.txt` and produce different held-out slices. Cells measured
    the two ways are not comparable, and the difference is small enough
    (ppl_ref 14.708 vs 14.654 on `1B_baseline`) to look like noise rather than a
    changed experiment.

    Whole-file is the path the committed 8-bit and 4-bit cells were measured on,
    so it is the one that has to be preserved. `holdout_sha` in each cell's
    provenance makes any future divergence visible instead of subtle.

    Text mode is deliberate: the committed corpora carry CRLF and universal
    newlines must normalise them before tokenization (data/README.md).
    """
    return [open(path, encoding="utf-8", errors="ignore").read()]


def holdout_sha(ids: list[int], n: int = 16) -> str:
    """Fingerprint of the held-out token stream.

    Two cells are comparable only if this matches: same corpus bytes, same
    tokenizer, same slicing, same reader. The corpus hash alone does not catch a
    changed reader, because the bytes on disk are identical either way.
    """
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:n]


def perplexity(token_nll: torch.Tensor) -> float:
    """exp of mean per-token negative log-likelihood, in nats."""
    if token_nll.numel() == 0:
        raise ValueError("no tokens to score")
    return math.exp(token_nll.detach().to(torch.float64).mean().item())


def _restrict(token_nll: torch.Tensor, positions: list[int] | None, *, keep: bool) -> torch.Tensor:
    """Select or drop the given sequence positions from a (B, T) NLL tensor."""
    if not positions:
        return token_nll if not keep else token_nll[:, :0]
    idx = torch.as_tensor(positions, device=token_nll.device)
    idx = idx[idx < token_nll.shape[-1]]
    if keep:
        return token_nll[:, idx]
    mask = torch.ones(token_nll.shape[-1], dtype=torch.bool, device=token_nll.device)
    mask[idx] = False
    return token_nll[:, mask]


def delta_ppl(nll_quant: torch.Tensor, nll_ref: torch.Tensor) -> float:
    """Perplexity damage from quantization, against the unquantized reference."""
    return perplexity(nll_quant) - perplexity(nll_ref)


def delta_nll(nll_quant: torch.Tensor, nll_ref: torch.Tensor) -> float:
    """Quantization damage in nats — the same quantity as delta_ppl, in log space.

    Perplexity is exp(mean(nll)), which is NOT additive over token subsets: the
    damage measured on all tokens is not a weighted average of the damage on
    disjoint subsets of them. Any attempt to attribute a share of a perplexity
    delta to a group of tokens is therefore ill-defined, and in practice
    overshoots — an early version of the decomposition below reported a
    contamination share of 1.106.

    Mean NLL is additive by construction, so the decomposition is done here and
    perplexity is reported alongside for comparability with the plan and the
    prior literature.
    """
    return (
        nll_quant.detach().to(torch.float64).mean().item()
        - nll_ref.detach().to(torch.float64).mean().item()
    )


def d_sink(dppl_all: float, dppl_sinks_fp16: float) -> float:
    """The plan §6 definition, kept verbatim for comparability."""
    return dppl_all - dppl_sinks_fp16


def d_sink_decomposed(
    nll_ref: torch.Tensor,
    nll_none: torch.Tensor,
    nll_exempt: torch.Tensor,
    sink_positions: list[int],
) -> dict[str, float]:
    """Split D_sink into the tautological part and the part under audit.

    All three tensors are per-token NLL of shape (B, T) over the same held-out
    slice: the unquantized reference, the fully-quantized run, and the run with
    ``sink_positions`` held in fp16.

    Returns:
      d_sink_total          the plan §6 metric, in perplexity units
      d_sink_total_nats     the same damage in log space, where it decomposes
      d_sink_non_sink_nats  THE HEADLINE — damage removed from tokens that were
                            not themselves exempted. Contamination, effect (b).
      d_sink_at_sink_nats   damage removed at the exempted positions. Effect
                            (a), reported so the total stays auditable rather
                            than as a finding in its own right.
      contamination_share   the fraction of the total attributable to (b),
                            weighted by token counts. Near zero means the
                            "mitigation" does nothing but protect the sink
                            token's own prediction.

    The count-weighted parts sum to the total exactly; ``test_evaluate.py``
    asserts that identity.
    """
    if not (nll_ref.shape == nll_none.shape == nll_exempt.shape):
        raise ValueError(
            f"NLL shapes must match: ref {tuple(nll_ref.shape)}, "
            f"none {tuple(nll_none.shape)}, exempt {tuple(nll_exempt.shape)}"
        )
    if not sink_positions:
        raise ValueError("no sink positions given; D_sink is undefined")

    def split(keep: bool) -> float:
        return d_sink(
            delta_nll(_restrict(nll_none, sink_positions, keep=keep),
                      _restrict(nll_ref, sink_positions, keep=keep)),
            delta_nll(_restrict(nll_exempt, sink_positions, keep=keep),
                      _restrict(nll_ref, sink_positions, keep=keep)),
        )

    total_nats = d_sink(delta_nll(nll_none, nll_ref), delta_nll(nll_exempt, nll_ref))
    non_sink = split(keep=False)
    at_sink = split(keep=True)

    n_total = nll_ref.shape[-1]
    n_sink = _restrict(nll_ref, sink_positions, keep=True).shape[-1]
    w_non_sink = (n_total - n_sink) / n_total

    return {
        "d_sink_total": d_sink(delta_ppl(nll_none, nll_ref), delta_ppl(nll_exempt, nll_ref)),
        "d_sink_total_nats": total_nats,
        "d_sink_non_sink_nats": non_sink,
        "d_sink_at_sink_nats": at_sink,
        "contamination_share": (w_non_sink * non_sink / total_nats) if total_nats else 0.0,
        "n_sink_positions": n_sink,
        "n_total_positions": n_total,
    }


def sink_positions_to_columns(positions: list[int]) -> list[int]:
    """Map absolute sequence positions to per-token-NLL column indices.

    A causal LM scores the prediction of token t from the context ending at
    t-1, so a length-T sequence yields T-1 scored columns and column j
    corresponds to absolute position j+1.

    Position 0 is therefore dropped: it is never a prediction target. That has a
    substantive consequence for the decomposition in this module. When the only
    sink is at position 0 — the usual case outside the Qwen family — there is no
    "own prediction" for it to repair, so effect (a) is empty by construction and
    every nat of D_sink is contamination. The self-effect only becomes real for
    the multi-level sinks at positions 5, 17, ... that Qwen models show, which is
    precisely where the decomposition earns its keep.
    """
    return [p - 1 for p in positions if p >= 1]


@torch.no_grad()
def token_nll(model, batches, *, device: str = "cuda") -> torch.Tensor:
    """Per-token negative log-likelihood over a batch stream.

    Returns (n_batches, T-1) in float64. Column j scores absolute position j+1.
    """
    rows = []
    for batch in batches:
        ids = batch.to(device)
        logits = model(ids, use_cache=False).logits
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        target = ids[:, 1:]
        rows.append(-logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).to(torch.float64).cpu())
    if not rows:
        raise RuntimeError("token_nll saw zero batches; check seq_len against the slice size")
    return torch.cat(rows, dim=0)


def _means(nll: torch.Tensor, sink_columns: list[int]) -> dict:
    """Mean NLL over all / sink / non-sink columns, plus per-sequence means.

    The per-sequence arrays exist because the calibration draw is NOT a usable
    randomness source for every arm. Per-token activation scaling is dynamic: it
    derives its scale from the tensor in front of it and never reads the
    calibration set, so five "disjoint calibration draws" give five byte-identical
    results and a confidence interval of width exactly zero. Measured on GPT-2:
    per_tensor had std 10.2 (8-bit) and 422 (4-bit) across draws, while per_token
    had std 0.000000 in both.

    A zero-width CI is not a precise measurement, it is an absent one — and it
    would land on the per-token arm, which is the arm the whole project is about.
    Bootstrapping over held-out SEQUENCES gives a randomness source that applies
    to both arms, so the two are comparable. These arrays are small (one float per
    sequence) and cost nothing to store.
    """
    total = nll.shape[-1]
    keep = _restrict(nll, sink_columns, keep=True)
    drop = _restrict(nll, sink_columns, keep=False)
    nan = float("nan")
    return {
        "nll_all": nll.mean().item(),
        "nll_sink": keep.mean().item() if keep.numel() else nan,
        "nll_non_sink": drop.mean().item() if drop.numel() else nan,
        "n_sink_cols": keep.shape[-1],
        "n_total_cols": total,
        "per_seq_all": nll.mean(dim=-1).tolist(),
        "per_seq_sink": keep.mean(dim=-1).tolist() if keep.numel() else [],
        "per_seq_non_sink": drop.mean(dim=-1).tolist() if drop.numel() else [],
    }


def d_sink_from_means(none: dict, exempt: dict) -> dict[str, float]:
    """Reconstruct the decomposition from two cells' aggregate means.

    The reference term cancels: D_sink is a difference of two damages measured
    against the SAME reference, so only the two quantized runs are needed. That
    is why a grid cell can store six floats instead of a per-token array.
    """
    n_total = none["n_total_cols"]
    n_sink = none["n_sink_cols"]
    if (n_total, n_sink) != (exempt["n_total_cols"], exempt["n_sink_cols"]):
        raise ValueError("cells disagree on token counts; they scored different slices")

    total = none["nll_all"] - exempt["nll_all"]
    non_sink = none["nll_non_sink"] - exempt["nll_non_sink"]
    at_sink = none["nll_sink"] - exempt["nll_sink"]
    w_non_sink = (n_total - n_sink) / n_total

    return {
        "d_sink_total_nats": total,
        "d_sink_non_sink_nats": non_sink,
        "d_sink_at_sink_nats": at_sink,
        "contamination_share": (w_non_sink * non_sink / total) if total else 0.0,
        "n_sink_positions": n_sink,
        "n_total_positions": n_total,
    }


def evaluate_cell(
    model,
    *,
    calib_batches_fn,
    holdout_batches_fn,
    bits: int,
    act_granularity: str,
    exception_kind: str,
    sink_mask=None,
    outlier_mask=None,
    device: str = "cuda",
) -> dict:
    """Run one cell of the configs/quant.yaml grid.

    Order matters: the reference pass runs with quantization DISABLED on the
    already-patched model rather than on the unpatched one, so the reference and
    quantized runs go through byte-identical code paths and any difference is
    quantization rather than a change of execution path.
    """
    from .calibrate import collect_ranges, reset_ranges
    from .patch import patch_model, resolve_fp16_exceptions, set_quant_enabled

    exceptions = resolve_fp16_exceptions(
        exception_kind, sink_mask=sink_mask, outlier_mask=outlier_mask
    )
    sink_columns = sink_positions_to_columns(exceptions.token_positions)

    restore, patched = patch_model(
        model,
        w_bits=bits,
        a_bits=bits,
        w_granularity="per_channel",
        a_granularity=act_granularity,
        exceptions=exceptions,
    )

    try:
        set_quant_enabled(model, False)
        nll_ref = token_nll(model, holdout_batches_fn(), device=device)

        set_quant_enabled(model, True)
        reset_ranges(model)
        ranges = {}
        if act_granularity == "per_tensor":
            ranges = collect_ranges(model, calib_batches_fn(), device=device)

        nll_quant = token_nll(model, holdout_batches_fn(), device=device)
    finally:
        restore()

    ref_means = _means(nll_ref, sink_columns)
    quant_means = _means(nll_quant, sink_columns)

    return {
        "bits": bits,
        "act_granularity": act_granularity,
        "fp16_exception": exception_kind,
        "exempt_positions": exceptions.token_positions,
        "exempt_channels": exceptions.channel_indices,
        "n_patched_layers": len(patched),
        "n_calibrated_layers": len(ranges),
        "ppl_ref": perplexity(nll_ref),
        "ppl_quant": perplexity(nll_quant),
        "delta_ppl": delta_ppl(nll_quant, nll_ref),
        "delta_nll": delta_nll(nll_quant, nll_ref),
        "reference_means": ref_means,
        "quantized_means": quant_means,
    }


def main() -> None:
    import json

    import yaml

    from .calibrate import build_slices, to_batches, tokenize_stream

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, help="single-cell mode: the bit width to run")
    parser.add_argument("--act-granularity", choices=["per_tensor", "per_token"])
    parser.add_argument("--fp16-exception")
    parser.add_argument("--calib-seed", type=int)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="runs/quant")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--quant-config", default="configs/quant.yaml")
    parser.add_argument("--sinks-json", help="runs/sinks/*.json supplying detected sink positions")
    parser.add_argument(
        "--text-file",
        help=f"local corpus; defaults to {DEFAULT_CORPUS} when present, else streams FineWeb-Edu",
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="walk the whole grid for this model in ONE process, loading the "
             "checkpoint once. Resumable: finished cells are skipped.",
    )
    parser.add_argument("--bits-list", help="grid mode: comma list, default from quant.yaml")
    parser.add_argument("--granularities", help="grid mode: comma list, default from quant.yaml")
    parser.add_argument("--exceptions", help="grid mode: comma list, default from quant.yaml")
    parser.add_argument("--seeds", help="grid mode: comma list, default from quant.yaml")
    parser.add_argument(
        "--calib-tokens", type=int,
        help="tokens per calibration draw; default from quant.yaml. The published "
             "grid used 2048 — the config value needs a corpus this one does not have.",
    )
    parser.add_argument("--eval-tokens", type=int, help="held-out tokens; default from quant.yaml")
    ns = parser.parse_args()

    single = [ns.bits, ns.act_granularity, ns.fp16_exception, ns.calib_seed]
    if not ns.grid and any(v is None for v in single):
        parser.error(
            "single-cell mode needs --bits, --act-granularity, --fp16-exception "
            "and --calib-seed. Pass --grid to walk the whole grid instead."
        )

    from sinks.measure import find_model_spec, load_config, load_model

    cfg = load_config(ns.models_config)
    spec = find_model_spec(cfg, ns.model)
    with open(ns.quant_config, encoding="utf-8") as fh:
        qcfg = yaml.safe_load(fh)

    model, tokenizer = load_model(spec, cfg.get("defaults", {}))
    model.to(ns.device)

    corpus_path = ns.text_file or (str(DEFAULT_CORPUS) if DEFAULT_CORPUS.exists() else None)
    if corpus_path:
        text_source = load_corpus(corpus_path)
        provenance = {"corpus": corpus_path, "corpus_sha256": _sha256_prefix(corpus_path)}
        print(f"corpus: {corpus_path}  sha256:{provenance['corpus_sha256']}")
    else:
        from datasets import load_dataset

        stream = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
        )
        text_source = (rec["text"] for rec in stream)
        provenance = {"corpus": "stream:fineweb-edu/sample-10BT", "corpus_sha256": None}
        print("corpus: STREAMED — this is a different document set than the "
              "committed results were measured on (LIMITATIONS.md §18)")

    seeds = _int_list(ns.seeds) if ns.seeds else list(qcfg["calibration_seeds"])
    n_draws = len(seeds)
    per_draw = ns.calib_tokens or qcfg["calibration_tokens"]
    eval_tokens = ns.eval_tokens or qcfg["eval_tokens"]
    provenance.update({"seq_len": ns.seq_len, "calib_tokens": per_draw, "eval_tokens": eval_tokens})

    ids = tokenize_stream(tokenizer, text_source, eval_tokens + n_draws * per_draw)
    slices = build_slices(ids, n_draws=n_draws, tokens_per_draw=per_draw, eval_tokens=eval_tokens)
    provenance["holdout_sha"] = holdout_sha(slices.holdout)
    print(f"corpus={len(ids)} tok  holdout={len(slices.holdout)}  draws={slices.n_draws}"
          f"x{per_draw}  seq={ns.seq_len}  holdout_sha={provenance['holdout_sha']}", flush=True)

    sink_mask = None
    if ns.fp16_exception == "detected_sinks" or (
        ns.grid and "detected_sinks" in (_str_list(ns.exceptions) if ns.exceptions
                                         else qcfg["fp16_exceptions"])
    ):
        if not ns.sinks_json:
            raise SystemExit("--fp16-exception detected_sinks requires --sinks-json")
        with open(ns.sinks_json, encoding="utf-8") as fh:
            sinks = json.load(fh)
        kind = sinks.get("primary_detector", "layerwise")
        table = sinks["detector"][kind]
        validation = sinks.get("detector_validation", {}).get(kind, {})
        # Take the LARGEST tau that both flags something and validates against
        # attention. Largest is the conservative end: it admits only the tokens
        # whose magnitude is most clearly anomalous, so the exception list does
        # not quietly absorb borderline positions and inflate D_sink.
        candidates = [
            t for t in sorted((float(x) for x in table), reverse=True)
            if table[str(t)]["n_flagged"] > 0 and "failed" not in validation.get(str(t), {})
        ]
        if not candidates:
            raise SystemExit(
                f"{ns.sinks_json}: no tau both flagged sinks and passed attention "
                "validation. Re-run sinks.measure before building an exception list."
            )
        positions = table[str(candidates[0])]["positions"]
        sink_mask = torch.zeros(1, ns.seq_len, dtype=torch.bool)
        sink_mask[0, [p for p in positions if p < ns.seq_len]] = True

    bos = tokenizer.bos_token_id

    if ns.grid:
        written = run_grid(
            model, tokenizer,
            model_id=ns.model,
            slices=slices,
            seq_len=ns.seq_len,
            bits_list=_int_list(ns.bits_list) if ns.bits_list else list(qcfg["bits"]),
            granularities=(_str_list(ns.granularities) if ns.granularities
                           else list(qcfg["act_granularity"])),
            exceptions=(_str_list(ns.exceptions) if ns.exceptions
                        else list(qcfg["fp16_exceptions"])),
            seeds=seeds,
            sink_mask=sink_mask,
            device=ns.device,
            out=ns.out,
            provenance=provenance,
        )
        print(f"[{ns.model}] wrote {len(written)} new cells to {ns.out}")
        return

    def calib_batches():
        return to_batches(slices.draw(ns.calib_seed), ns.seq_len, bos_token_id=bos)

    def holdout_batches():
        return to_batches(slices.holdout, ns.seq_len, bos_token_id=bos)

    result = evaluate_cell(
        model,
        calib_batches_fn=calib_batches,
        holdout_batches_fn=holdout_batches,
        bits=ns.bits,
        act_granularity=ns.act_granularity,
        exception_kind=ns.fp16_exception,
        sink_mask=sink_mask,
        device=ns.device,
    )

    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{ns.model}_b{ns.bits}_{ns.act_granularity}_{ns.fp16_exception}_calib{ns.calib_seed}.json"
    with open(out_dir / name, "w", encoding="utf-8") as fh:
        json.dump(
            {"model": ns.model, "calib_seed": ns.calib_seed, **provenance, **result},
            fh, indent=2,
        )

    print(f"wrote {out_dir / name}  dppl={result['delta_ppl']:+.4f}")


def run_grid(
    model,
    tokenizer,
    *,
    model_id: str,
    slices,
    seq_len: int,
    bits_list: list[int],
    granularities: list[str],
    exceptions: list[str],
    seeds: list[int],
    sink_mask=None,
    device: str = "cuda",
    out: str = "runs/quant",
    skip_existing: bool = True,
    provenance: dict | None = None,
) -> list[str]:
    """Walk the grid for ONE model, loading it exactly once.

    The Makefile spawns a separate process per cell, which is the right default
    for crash isolation but reloads a 3.4 GB checkpoint every time. This does the
    same work in one process while preserving the property that matters: one JSON
    per cell, written as soon as that cell finishes (design rule §7.1). A crash
    at cell 37 still leaves cells 1-36 on disk.

    ``skip_existing`` makes the whole grid resumable — rerun after a crash and it
    picks up where it stopped.
    """
    import json
    from pathlib import Path

    from .calibrate import to_batches

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bos = tokenizer.bos_token_id
    written = []

    for bits in bits_list:
        for gran in granularities:
            for exc in exceptions:
                for seed in seeds:
                    name = f"{model_id}_b{bits}_{gran}_{exc}_calib{seed}.json"
                    path = out_dir / name
                    if skip_existing and path.exists():
                        continue

                    def calib_fn(_s=seed):
                        return to_batches(slices.draw(_s), seq_len, bos_token_id=bos)

                    def hold_fn():
                        return to_batches(slices.holdout, seq_len, bos_token_id=bos)

                    result = evaluate_cell(
                        model,
                        calib_batches_fn=calib_fn,
                        holdout_batches_fn=hold_fn,
                        bits=bits,
                        act_granularity=gran,
                        exception_kind=exc,
                        sink_mask=sink_mask,
                        device=device,
                    )
                    with open(path, "w", encoding="utf-8") as fh:
                        # Provenance first: a measured field must always win a
                        # key collision, or a number gets silently replaced by
                        # a label. Pinned in tests/test_grid.py.
                        json.dump(
                            {"model": model_id, "calib_seed": seed,
                             **(provenance or {}), **result},
                            fh, indent=2,
                        )
                    written.append(name)
                    print(
                        f"  {model_id} b{bits} {gran:<10} {exc:<11} seed{seed}  "
                        f"dppl={result['delta_ppl']:+9.4f}",
                        flush=True,
                    )
    return written


if __name__ == "__main__":
    main()
