"""Per-layer activation shape: what per-tensor scaling has to absorb and per-token does not.

R4 left a mechanism open. The element-wise gated checkpoint is destroyed by
per-tensor 8-bit activation quantization (+3388 dppl against a reference of
14.5) while its head-wise sibling — one boolean away in the same config — takes
+2.29, and per-token scaling is nearly free on both (+0.68 / +0.40). Four
candidate mechanisms are already dead (README §5.4): it is not the weights, not
the model, not calibration clipping, and not the fused gate projection.

What is left is the granularity itself, so this module measures the one thing
the two granularities actually differ in.

    per_tensor   one scale for the whole activation, s = amax(X) / qmax
    per_token    one scale per row,                  s_i = amax(X_i) / qmax

Everything else about them is identical — same rounding, same clamp, same bit
width. So the entire gap between the two arms has to be expressible as a
property of how magnitude is distributed ACROSS ROWS. This module reports that
property directly:

  dispersion       amax(X) / median_i amax(X_i)
                   The factor by which the shared scale is too coarse for a
                   typical row. This is exactly the quantity per-token scaling
                   divides out, and it is 1.0 for a tensor whose rows all reach
                   the same magnitude.

  eff_bits         log2(qmax / dispersion) + 1
                   The same number in the units the audit is denominated in:
                   how many of the b bits the median row actually gets once the
                   scale has been set by the largest row. At dispersion = qmax
                   the median row is left with the sign bit alone.

  underflow_tensor fraction of entries with |x| < s/2, i.e. entries that round
                   to zero. The consequence, rather than the cause.

  underflow_token  the same fraction under per-row scales.

  col_dispersion   amax(X) / median_c amax_r |X[:,c]| — the same statistic
                   transposed onto the feature axis. This is the axis check.
                   One huge ROW gives dispersion >> 1 and col_dispersion ~ 1;
                   one huge FEATURE CHANNEL gives the reverse; a single huge
                   entry inflates both. Per-token scaling divides out row
                   dispersion and nothing else, so this is what decides whether
                   per-token is the right fix or merely a fix that happened to
                   work.

  underflow_col    entries rounding to zero under one scale per feature channel
                   — the LLM.int8 axis.

**Naming, because this repo already uses the word.** `col_*` here means the
FEATURE axis. It is *not* `quant.fakequant`'s `per_channel` granularity, which
reduces over every axis but the first and therefore gives one scale per TOKEN
when applied to an activation. That collision is real and is why these fields
are called `col_` rather than `channel_`.

The last pair is the load-bearing one, and it is what makes this a test rather
than an illustration. `underflow_tensor` blowing up on the element-wise arm is
only a mechanism if `underflow_token` does NOT — otherwise the finding is
merely "this checkpoint has strange activations", which explains nothing about
granularity and would predict per-token damage that the grid does not show.
That is an outcome this module can come back with, and HANDOFF §11.2 names the
honest response to it: report the effect without a mechanism.

Ranges are measured per forward call, which is the unit a DYNAMIC per-tensor
quantizer actually sees. That is deliberate: the dynamic arm cannot clip, and it
is the *worse* of the two per-tensor arms (+3482 vs +3388), so explaining it
explains the failure without the calibration-coverage confound that
quant.diagnose's coverage table already ruled out.

Nothing here is quantized. Every pass runs through pass-through QuantLinear
wrappers, so the module set is byte-identical to the one the grid quantizes
(including GPT-2's Conv1D and the lm_head exclusion) while the numbers describe
the model's own activations. Design rule §7.2 keeps analysis/ away from model
weights, so the measurement lives here and only the cross-model table lives in
analysis/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .evaluate import (
    DEFAULT_CORPUS,
    provenance_path,
    _sha256_prefix,
    holdout_sha,
    load_corpus,
    perplexity,
    token_nll,
)
from .fakequant import qrange

# Per-call scalars, in the order they are packed into the accumulator row.
# `col_*` fields are appended rather than interleaved so an older run JSON stays
# readable: the earlier fields keep their meaning and their position.
FIELDS = ("amax", "row_amax_median", "row_amax_p99", "rms",
          "underflow_tensor", "underflow_token",
          "col_amax_median", "underflow_col")


def _call_stats(x: torch.Tensor, qmax: int) -> torch.Tensor:
    """The per-call scalars for one activation tensor, packed into a row.

    Kept on the GPU and stacked at the end rather than ``.item()``-ed here: one
    sync per layer per forward call is thousands of stalls over the holdout, for
    numbers that are not read until the pass is over.

    float32 throughout. Deriving a range in bf16 quantizes the measurement of
    the quantizer, which is the same silent no-op that tests/test_fakequant.py
    exists to catch one level down.
    """
    xf = x.detach().to(torch.float32).reshape(-1, x.shape[-1]).abs()

    amax = xf.amax()
    row_amax = xf.amax(dim=-1)
    col_amax = xf.amax(dim=0)
    rms = xf.pow(2).mean().sqrt()

    # An entry rounds to zero iff |x| < s/2, for s the scale it is quantized at.
    half_tensor = amax / (2 * qmax)
    half_token = (row_amax / (2 * qmax)).unsqueeze(-1)
    half_col = (col_amax / (2 * qmax)).unsqueeze(0)

    return torch.stack([
        amax,
        row_amax.median(),
        torch.quantile(row_amax, 0.99),
        rms,
        (xf < half_tensor).to(torch.float32).mean(),
        (xf < half_token).to(torch.float32).mean(),
        col_amax.median(),
        (xf < half_col).to(torch.float32).mean(),
    ])


def collect(model, *, bits: int, batches_fn, device="cuda") -> dict[str, dict]:
    """Per-layer activation shape over one pass of ``batches_fn``.

    Returns ``{layer_name: {stat: value}}``, aggregated across forward calls:
    medians for the range statistics (a mean over a heavy-tailed per-call ratio
    reports the worst sequence, not the typical one) and means for the underflow
    fractions (every call contributes the same number of entries, so the mean is
    the true fraction over the pass).
    """
    from .patch import QuantLinear, patch_model

    _, qmax = qrange(bits, symmetric=True)
    acc: dict[str, list[torch.Tensor]] = {}
    handles = []

    # w_bits=None, a_bits=None makes every wrapper a pass-through, so this
    # measures the model rather than the effect of quantizing it.
    restore, _ = patch_model(model, w_bits=None, a_bits=None)
    try:
        for name, module in model.named_modules():
            if not isinstance(module, QuantLinear):
                continue
            acc[name] = []

            def hook(mod, args, _name=name):
                if args and isinstance(args[0], torch.Tensor) and args[0].numel():
                    acc[_name].append(_call_stats(args[0], qmax))

            handles.append(module.register_forward_pre_hook(hook))

        with torch.no_grad():
            for batch in batches_fn():
                model(batch.to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
        restore()
        torch.cuda.empty_cache()

    out: dict[str, dict] = {}
    for name, rows in acc.items():
        if not rows:
            continue
        table = torch.stack(rows).cpu()
        med = table.median(dim=0).values.tolist()
        mean = table.mean(dim=0).tolist()
        s = dict(zip(FIELDS, med))
        for k in ("underflow_tensor", "underflow_token", "underflow_col"):
            s[k] = mean[FIELDS.index(k)]
        s["n_calls"] = len(rows)

        s["dispersion"] = s["amax"] / s["row_amax_median"] if s["row_amax_median"] > 0 else None
        # The same statistic transposed. Together the two say WHICH AXIS carries
        # the tensor's extremes, which is the whole of R6's mechanism claim:
        # per-token scaling divides out row dispersion and nothing else, so a
        # tensor that is peaked across rows is one per-token can rescue and a
        # tensor that is peaked across features is not. One huge row gives
        # dispersion >> 1 with col_dispersion ~ 1; one huge feature channel
        # gives the reverse. This is the axis check, not a second opinion.
        s["col_dispersion"] = (
            s["amax"] / s["col_amax_median"] if s["col_amax_median"] > 0 else None
        )
        s["crest"] = s["amax"] / s["rms"] if s["rms"] > 0 else None
        s["eff_bits"] = (
            max(0.0, math.log2(qmax / s["dispersion"]) + 1.0)
            if s["dispersion"] and s["dispersion"] > 0
            else None
        )
        out[name] = s
    return out


def summarize(layers: dict[str, dict], *, top: int = 8) -> dict:
    """Model-level roll-up. Medians and maxima across layers, plus the worst few.

    The maximum matters as much as the median here and is reported alongside it:
    a network is destroyed by its worst layer, not by its typical one, and a
    median that looks ordinary next to a max three orders of magnitude larger is
    a different finding from a median that is uniformly bad.
    """
    def col(key):
        return sorted(v[key] for v in layers.values() if v.get(key) is not None)

    def med(key):
        c = col(key)
        return c[len(c) // 2] if c else None

    def mx(key):
        c = col(key)
        return c[-1] if c else None

    def mn(key):
        c = col(key)
        return c[0] if c else None

    worst = sorted(
        ((k, v) for k, v in layers.items() if v.get("dispersion") is not None),
        key=lambda kv: -kv[1]["dispersion"],
    )

    return {
        "n_layers": len(layers),
        "dispersion_median": med("dispersion"),
        "dispersion_max": mx("dispersion"),
        "eff_bits_median": med("eff_bits"),
        "eff_bits_min": mn("eff_bits"),
        "col_dispersion_median": med("col_dispersion"),
        "col_dispersion_max": mx("col_dispersion"),
        "underflow_col_median": med("underflow_col"),
        "crest_median": med("crest"),
        "crest_max": mx("crest"),
        "underflow_tensor_median": med("underflow_tensor"),
        "underflow_tensor_max": mx("underflow_tensor"),
        "underflow_token_median": med("underflow_token"),
        "underflow_token_max": mx("underflow_token"),
        "worst_layers": [
            {"layer": k, "dispersion": v["dispersion"],
             "col_dispersion": v.get("col_dispersion"), "eff_bits": v["eff_bits"],
             "underflow_tensor": v["underflow_tensor"],
             "underflow_token": v["underflow_token"],
             "underflow_col": v.get("underflow_col")}
            for k, v in worst[:top]
        ],
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
    parser.add_argument("--out", default="runs/dist")
    parser.add_argument("--text-file")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--top", type=int, default=8)
    ns = parser.parse_args()

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

    def hold_fn():
        return to_batches(slices.holdout, ns.seq_len, bos_token_id=bos)

    # The reference perplexity is carried so a distribution run can be checked
    # against the grid cell it is supposed to explain: same corpus, same holdout
    # slice, same ppl_ref, or the two are not describing the same experiment.
    with torch.no_grad():
        nll_ref = token_nll(model, hold_fn(), device=ns.device)
    ppl_ref = perplexity(nll_ref)
    print(f"{ns.model}  ppl_ref={ppl_ref:.3f}  bits={ns.bits}", flush=True)

    layers = collect(model, bits=ns.bits, batches_fn=hold_fn, device=ns.device)
    summary = summarize(layers, top=ns.top)

    print(f"  {summary['n_layers']} layers, {ns.bits}-bit grid", flush=True)
    print(f"  dispersion   median {summary['dispersion_median']:>10.2f}"
          f"   max {summary['dispersion_max']:>12.2f}", flush=True)
    print(f"  eff_bits     median {summary['eff_bits_median']:>10.2f}"
          f"   min {summary['eff_bits_min']:>12.2f}", flush=True)
    print(f"  underflow    per-tensor {summary['underflow_tensor_median']:>7.4f}"
          f"   per-token {summary['underflow_token_median']:>10.4f}"
          f"   per-feature {summary['underflow_col_median']:>8.4f}   (median layer)",
          flush=True)
    print(f"  col disp     median {summary['col_dispersion_median']:>10.2f}"
          f"   max {summary['col_dispersion_max']:>12.2f}   (the axis check)",
          flush=True)
    for w in summary["worst_layers"][:5]:
        print(f"      {w['layer']:<40} disp x{w['dispersion']:>9.1f}"
              f"  col x{(w['col_dispersion'] or float('nan')):>7.1f}"
              f"  uf_T {w['underflow_tensor']:.4f}"
              f"  uf_tok {w['underflow_token']:.4f}"
              f"  uf_col {(w['underflow_col'] or float('nan')):.4f}", flush=True)

    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ns.model}_b{ns.bits}_dist.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": ns.model, "bits": ns.bits, "ppl_ref": ppl_ref,
                "seq_len": ns.seq_len, "eval_tokens": ns.eval_tokens,
                "calib_seed": ns.calib_seed, "corpus": provenance_path(corpus),
                "corpus_sha256": _sha256_prefix(corpus),
                "holdout_sha": holdout_sha(slices.holdout),
                "summary": summary, "layers": layers,
            },
            fh, indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
