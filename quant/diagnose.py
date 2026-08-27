"""Why does one arm break under per-tensor activation quant when its neighbours don't?

A grid cell reports one number, `delta_ppl`, for weights and activations
quantized together under one granularity. That is the right unit for the audit
and the wrong unit for asking *what broke*. This module splits it:

  w_only            weights per-channel, activations untouched — the control
                    that should be near-lossless at 8 bits
  a_only_static     activations per-tensor from a CALIBRATED range, weights
                    untouched — the 2023 deployed scheme
  a_only_dynamic    activations per-tensor from the range of the tensor in front
                    of it, weights untouched
  a_only_per_token  activations per-token, weights untouched — the modern scheme
  full_static       both, per-tensor static — reproduces the grid cell

The static/dynamic pair is the load-bearing one. Static per-tensor quantization
CLAMPS anything the calibration pass did not see, so a model whose eval-time
activations run hotter than its calibration draw is destroyed by clipping rather
than by rounding. Those are different failures with different fixes, and
`delta_ppl` alone cannot tell them apart: comparing the two arms does.

The coverage table answers the same question from the other side, by measuring
the range each layer actually sees on the held-out slice against the range it
was calibrated with. A ratio above 1 is a layer being clipped.

Design rule §7.2 keeps analysis/ away from model weights, so this lives in
quant/ with the rest of the code that loads checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .evaluate import (
    DEFAULT_CORPUS,
    provenance_path,
    _sha256_prefix,
    delta_ppl,
    holdout_sha,
    load_corpus,
    perplexity,
    token_nll,
)

ARMS = ("w_only", "a_only_static", "a_only_dynamic", "a_only_per_token", "full_static")


def _spec(arm: str, bits: int) -> dict:
    """patch_model kwargs for one arm. None disables that side entirely."""
    if arm == "w_only":
        return {"w_bits": bits, "a_bits": None, "a_granularity": "per_tensor"}
    if arm in ("a_only_static", "a_only_dynamic"):
        return {"w_bits": None, "a_bits": bits, "a_granularity": "per_tensor"}
    if arm == "a_only_per_token":
        return {"w_bits": None, "a_bits": bits, "a_granularity": "per_token"}
    if arm == "full_static":
        return {"w_bits": bits, "a_bits": bits, "a_granularity": "per_tensor"}
    raise ValueError(f"unknown arm: {arm!r}")


def measure_arm(
    model, *, arm: str, bits: int, calib_batches_fn, holdout_batches_fn, nll_ref,
    device="cuda", skip_modules: tuple[str, ...] = (),
) -> dict:
    """Damage from one arm, against a reference measured outside this function.

    The reference is passed in rather than recomputed per arm: it is the same
    unquantized model every time, and recomputing it five times would cost five
    holdout passes to produce five identical numbers.

    ``skip_modules`` leaves matching layers in fp16. It exists to localise a
    failure: if excluding one projection removes the damage, that projection is
    where the damage lives. `q_proj` is the interesting one on the gated arms,
    because the attention gate is fused into it as extra output width (C13) —
    so `--skip-modules q_proj` asks whether the gate is the fragile part.
    """
    from .calibrate import collect_ranges
    from .patch import DEFAULT_SKIP, patch_model, set_quant_enabled

    restore, patched = patch_model(
        model, w_granularity="per_channel",
        skip=DEFAULT_SKIP + tuple(skip_modules), **_spec(arm, bits),
    )
    try:
        set_quant_enabled(model, True)
        ranges = {}
        if arm in ("a_only_static", "full_static"):
            ranges = collect_ranges(model, calib_batches_fn(), device=device)
        nll_q = token_nll(model, holdout_batches_fn(), device=device)
    finally:
        restore()
        torch.cuda.empty_cache()

    return {
        "arm": arm,
        "n_patched": len(patched),
        "n_calibrated": len(ranges),
        "ppl_quant": perplexity(nll_q),
        "delta_ppl": delta_ppl(nll_q, nll_ref),
    }


def coverage(model, *, bits: int, calib_batches_fn, holdout_batches_fn, device="cuda") -> dict:
    """Per-layer calibrated range vs the range the held-out slice actually needs.

    Both passes run in observation mode, which is pass-through: nothing is
    quantized, so this measures the model's own activations rather than the
    effect of quantizing them.
    """
    from .calibrate import collect_ranges
    from .patch import QuantLinear, patch_model, set_observing

    restore, _ = patch_model(
        model, w_bits=None, a_bits=bits, a_granularity="per_tensor", w_granularity="per_channel"
    )
    try:
        calib = collect_ranges(model, calib_batches_fn(), device=device)

        for module in model.modules():
            if isinstance(module, QuantLinear):
                module.observed_amax = 0.0
        set_observing(model, True)
        with torch.no_grad():
            for batch in holdout_batches_fn():
                model(batch.to(device), use_cache=False)
        set_observing(model, False)
        hold = {
            name: mod.observed_amax
            for name, mod in model.named_modules()
            if isinstance(mod, QuantLinear) and mod.observed_amax > 0.0
        }
    finally:
        restore()
        torch.cuda.empty_cache()

    ratios = {n: hold[n] / calib[n] for n in sorted(set(calib) & set(hold)) if calib[n] > 0}
    worst = sorted(ratios.items(), key=lambda kv: -kv[1])[:8]
    return {
        "n_layers": len(ratios),
        "n_clipped": sum(1 for r in ratios.values() if r > 1.0),
        "max_ratio": max(ratios.values()) if ratios else None,
        "median_ratio": sorted(ratios.values())[len(ratios) // 2] if ratios else None,
        "worst_layers": [{"layer": n, "holdout_over_calib": r} for n, r in worst],
    }


def main() -> None:
    from .calibrate import build_slices, to_batches, tokenize_stream

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--calib-tokens", type=int, default=2048)
    parser.add_argument("--eval-tokens", type=int, default=8192)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="runs/diag")
    parser.add_argument("--text-file")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--arms", help=f"comma list, default all of: {','.join(ARMS)}")
    parser.add_argument(
        "--skip-modules",
        help="comma list of name substrings left in fp16, e.g. q_proj. Localises "
             "damage to a projection; recorded in the output filename.",
    )
    ns = parser.parse_args()

    arms = tuple(ns.arms.replace(" ", "").split(",")) if ns.arms else ARMS
    skip_modules = tuple(ns.skip_modules.replace(" ", "").split(",")) if ns.skip_modules else ()

    from sinks.measure import find_model_spec, load_config, load_model

    cfg = load_config(ns.models_config)
    spec = find_model_spec(cfg, ns.model)
    model, tokenizer = load_model(spec, cfg.get("defaults", {}))
    model.to(ns.device)

    corpus = ns.text_file or str(DEFAULT_CORPUS)
    ids = tokenize_stream(tokenizer, load_corpus(corpus), ns.eval_tokens + 5 * ns.calib_tokens)
    slices = build_slices(
        ids, n_draws=5, tokens_per_draw=ns.calib_tokens, eval_tokens=ns.eval_tokens
    )
    bos = tokenizer.bos_token_id

    def calib_fn():
        return to_batches(slices.draw(ns.calib_seed), ns.seq_len, bos_token_id=bos)

    def hold_fn():
        return to_batches(slices.holdout, ns.seq_len, bos_token_id=bos)

    with torch.no_grad():
        nll_ref = token_nll(model, hold_fn(), device=ns.device)
    ppl_ref = perplexity(nll_ref)
    print(f"{ns.model}  ppl_ref={ppl_ref:.3f}  bits={ns.bits}", flush=True)

    results = []
    for arm in arms:
        r = measure_arm(
            model, arm=arm, bits=ns.bits, calib_batches_fn=calib_fn,
            holdout_batches_fn=hold_fn, nll_ref=nll_ref, device=ns.device,
            skip_modules=skip_modules,
        )
        r["skip_modules"] = list(skip_modules)
        results.append(r)
        print(f"  {arm:<18} dppl={r['delta_ppl']:+12.4f}   ppl={r['ppl_quant']:.2f}"
              f"   (n_patched={r['n_patched']})", flush=True)

    cov = None
    if not ns.skip_coverage:
        cov = coverage(
            model, bits=ns.bits, calib_batches_fn=calib_fn,
            holdout_batches_fn=hold_fn, device=ns.device,
        )
        print(f"  coverage: {cov['n_clipped']}/{cov['n_layers']} layers clipped, "
              f"median ratio {cov['median_ratio']:.3f}, max {cov['max_ratio']:.3f}", flush=True)
        for w in cov["worst_layers"][:5]:
            print(f"      {w['layer']:<44} x{w['holdout_over_calib']:.3f}", flush=True)

    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("_skip-" + "-".join(skip_modules)) if skip_modules else ""
    path = out_dir / f"{ns.model}_b{ns.bits}_diag{suffix}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": ns.model, "bits": ns.bits, "ppl_ref": ppl_ref,
                "seq_len": ns.seq_len, "calib_tokens": ns.calib_tokens,
                "eval_tokens": ns.eval_tokens, "calib_seed": ns.calib_seed,
                "corpus": provenance_path(corpus),
                "corpus_sha256": _sha256_prefix(corpus),
                "holdout_sha": holdout_sha(slices.holdout),
                "skip_modules": list(skip_modules),
                "arms": results, "coverage": cov,
            },
            fh, indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
