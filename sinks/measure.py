"""Track-A measurement entrypoint.

Writes exactly one JSON per (model, calibration draw, BOS policy) to
runs/sinks/<model>_<calib>_<bos>.json and nothing else. A crashed sweep must
never cost a figure (design rule §7.1).

Batch size 1 for all Track-A measurement (trap §9.2).

Trap §9.4 — every model runs under every BOS policy it actually supports.
"The sink" is whatever occupies position 0, so a cross-family comparison under
mismatched BOS policies compares two different things and calls the difference
architectural.

The plan states the premise as "GPT-2 does not prepend BOS; Qwen and Llama do".
Measured on 2026-08-20 that is false for Qwen3-0.6B-Base, whose tokenizer has
bos_token_id = None: it has no BOS arm to run. configs/models.yaml records the
supported policies per checkpoint, and to_batches refuses to substitute some
other token rather than quietly changing what occupies position 0.

The Day 2 gate lives at the end of ``measure_model``: the magnitude-based
detector is validated against the attention it never looked at. If they
disagree the run aborts rather than writing a JSON, because every downstream
attribution number would be computed against a wrong sink set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from .detector import (
    DetectorValidationError,
    layer_relative_norms,
    sweep_tau,
    sweep_tau_layerwise,
    validate_against_attention,
)
from .hooks import (
    AttentionRecord,
    ResidualRecord,
    attach_attention_hooks,
    attach_residual_hooks,
)
from .metrics import outlier_channels


@dataclass
class MeasureArgs:
    model: str
    calib_seed: int
    prepend_bos: bool
    seq_len: int = 1024
    n_batches: int = 64                # 64 x 1024 = 65536 calibration tokens
    out: str = "runs/sinks"


def load_config(path: str = "configs/models.yaml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def find_model_spec(cfg: dict, model_id: str) -> dict:
    for spec in cfg["models"]:
        if spec["id"] == model_id:
            return spec
    known = ", ".join(s["id"] for s in cfg["models"])
    raise KeyError(f"unknown model id {model_id!r}. Known: {known}")


def resolve_model_path(spec: dict) -> str:
    """Return something ``from_pretrained`` can actually load.

    ``trust_remote_code`` and ``subfolder`` DO NOT COMPOSE. A config's
    ``auto_map`` names bare module files (``configuration_qwen3.Qwen3Config``),
    and transformers resolves those against the repo ROOT, ignoring subfolder.
    QwQZh/gated_attention vendors a separate copy of the modelling code inside
    each arm's subfolder and nothing at the root, so

        from_pretrained("QwQZh/gated_attention", subfolder="1B_baseline",
                        trust_remote_code=True)

    fails with "does not appear to have a file named configuration_qwen3.py".

    Fix: materialise the subfolder into the local cache and load from that
    directory, where the vendored .py files sit beside config.json.
    """
    if spec.get("subfolder") and spec.get("trust_remote_code"):
        import os

        from huggingface_hub import snapshot_download

        root = snapshot_download(
            spec["hf_repo"], allow_patterns=[f"{spec['subfolder']}/*"]
        )
        return os.path.join(root, spec["subfolder"])
    return spec["hf_repo"]


def load_model(spec: dict, defaults: dict):
    """Load with eager attention — SDPA never materialises the probabilities.

    ``trust_remote_code`` is required for the gated_attention checkpoints: they
    vendor their own modeling_qwen3.py rather than using stock transformers.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_model_path(spec)
    trust = spec.get("trust_remote_code", False)

    kwargs = {
        "trust_remote_code": trust,
        "attn_implementation": "eager",
        "dtype": getattr(torch, defaults.get("dtype", "bfloat16")),
    }
    # Only pass subfolder when the path is still a bare repo id; a resolved
    # local path already points inside the arm.
    if spec.get("subfolder") and path == spec["hf_repo"]:
        kwargs["subfolder"] = spec["subfolder"]

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    tok_kwargs = {"trust_remote_code": trust}
    if spec.get("subfolder") and path == spec["hf_repo"]:
        tok_kwargs["subfolder"] = spec["subfolder"]
    tokenizer = AutoTokenizer.from_pretrained(path, **tok_kwargs)
    model.eval()
    return model, tokenizer


def calibration_batches(
    tokenizer,
    args: MeasureArgs,
    text_source,
    *,
    n_draws: int = 5,
    eval_tokens: int = 0,
):
    """Yield (1, T) input_id batches from a disjoint slice of the corpus.

    The slicing itself lives in quant.calibrate and is deliberately NOT
    reimplemented here. Measurement and evaluation must see byte-identical
    draws: the sink positions detected on draw k are later used to build the
    fp16 exception list for the cell calibrated on draw k, and if the two
    modules sliced the corpus even slightly differently that correspondence
    would break silently — the exceptions would protect positions that were
    never the ones measured.

    Pass ``eval_tokens`` matching configs/quant.yaml so the holdout is reserved
    identically on both sides.
    """
    from quant.calibrate import build_slices, to_batches, tokenize_stream

    per_draw = args.n_batches * args.seq_len
    ids = tokenize_stream(tokenizer, text_source, eval_tokens + n_draws * per_draw)
    slices = build_slices(
        ids, n_draws=n_draws, tokens_per_draw=per_draw, eval_tokens=eval_tokens
    )
    return to_batches(
        slices.draw(args.calib_seed),
        args.seq_len,
        prepend_bos=args.prepend_bos,
        bos_token_id=tokenizer.bos_token_id,
    )


def measure_model(model, batches, taus: list[float], device: str = "cuda") -> dict:
    """Run the calibration pass and return every metric in one dict."""
    attn: dict[str, AttentionRecord] = {}
    resid: dict[str, ResidualRecord] = {}

    model.to(device)
    with torch.no_grad(), attach_attention_hooks(model, attn), attach_residual_hooks(model, resid):
        for batch in batches:
            model(batch.to(device), output_attentions=True, use_cache=False)

    if not any(r.batches for r in attn.values()):
        raise RuntimeError(
            "attention hooks fired but recorded nothing. The model is probably "
            "not returning probabilities: check attn_implementation='eager' and "
            "output_attentions=True."
        )

    # ResidualRecord.max_norm is already reduced over the batch, so the stack is
    # (L, T). Both detectors expect a batch axis, so reinstate it as (L, 1, T).
    per_layer = torch.stack([r.max_norm for r in resid.values() if r.max_norm is not None])
    per_layer = per_layer.unsqueeze(1)
    aggregate = per_layer.amax(dim=0)

    received = torch.stack(
        [torch.tensor(r.as_dict()["received_attention"]) for r in attn.values() if r.batches]
    ).mean(dim=0).unsqueeze(0)

    # Both detectors are run and both are reported. layerwise is PRIMARY (see
    # sinks.detector.layer_relative_norms and LIMITATIONS.md §11); aggregate is
    # kept because it is the plan-§6 definition and dropping it silently would
    # make the change to the metric invisible in the output.
    sweeps = {
        "layerwise": sweep_tau_layerwise(per_layer, taus),
        "aggregate": sweep_tau(aggregate, taus),
    }

    # The Day 2 gate. Validate at every tau; a detector that only agrees with
    # attention at one hand-picked tau has not been validated, it has been fit.
    validation: dict[str, dict[str, dict]] = {}
    for kind, sweep in sweeps.items():
        validation[kind] = {}
        for tau, result in sweep.items():
            try:
                validation[kind][str(tau)] = validate_against_attention(result.mask, received)
            except DetectorValidationError as exc:
                validation[kind][str(tau)] = {"failed": str(exc)}

    # Fraction of (layer, head) pairs that sink, using the plan-§6 threshold of
    # S > 0.5. Not a new constant — the same one §6 already reports against.
    sink_masses = torch.tensor(
        [a.as_dict()["sink_mass_per_head"] for a in attn.values() if a.batches]
    )
    frac_sinking = float((sink_masses > 0.5).to(torch.float64).mean().item())
    sink_free = frac_sinking < 0.01

    # The Day 2 gate, corrected. "Detector found nothing" has TWO causes and
    # they must not be conflated:
    #
    #   (a) the detector is broken — attention plainly shows a sink and the
    #       magnitude detector missed it. Abort; every attribution number
    #       downstream would be computed against a wrong sink set.
    #   (b) the model genuinely has no sink. That is a RESULT, not a failure,
    #       and it is the result the gated-attention paper predicts for its own
    #       checkpoints. Aborting here would turn the project's most important
    #       positive finding into a crash.
    #
    # Attention discriminates: if essentially no head concentrates on position 0
    # there is nothing for the detector to find, and finding nothing is correct.
    if all("failed" in v for v in validation["layerwise"].values()) and not sink_free:
        raise DetectorValidationError(
            "the primary (layer-relative) detector disagreed with attention at "
            f"EVERY tau in {taus}, yet {frac_sinking:.1%} of heads have sink mass "
            "> 0.5. Attention says there IS a sink and the detector cannot find "
            "it. Do not proceed to attribution — the sink set is wrong.\n"
            + "\n".join(f"  tau={k}: {v['failed']}" for k, v in validation["layerwise"].items())
        )

    return {
        "attention": [r.as_dict() for r in attn.values()],
        "residual": [r.as_dict() for r in resid.values()],
        "aggregate_inf_norm": aggregate.squeeze(0).tolist(),
        "layer_relative_norm": layer_relative_norms(per_layer).squeeze(0).tolist(),
        "received_attention": received.squeeze(0).tolist(),
        "detector": {
            kind: {str(t): r.as_dict() for t, r in sweep.items()}
            for kind, sweep in sweeps.items()
        },
        "detector_validation": validation,
        "primary_detector": "layerwise",
        "frac_heads_sinking": frac_sinking,
        "sink_free": sink_free,
        "mean_sink_mass": float(sink_masses.mean().item()),
        "max_head_sink_mass": float(sink_masses.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="id from configs/models.yaml")
    parser.add_argument("--calib-seed", type=int, required=True)
    parser.add_argument("--prepend-bos", action="store_true")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-batches", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="runs/sinks")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument(
        "--text-file",
        help="local corpus; defaults to data/fineweb_edu.txt when present, else "
             "streams FineWeb-Edu. Streaming gives a DIFFERENT document set from "
             "the one the quantization grid was measured on (C19).",
    )
    ns = parser.parse_args()

    args = MeasureArgs(
        model=ns.model,
        calib_seed=ns.calib_seed,
        prepend_bos=ns.prepend_bos,
        seq_len=ns.seq_len,
        n_batches=ns.n_batches,
        out=ns.out,
    )

    cfg = load_config(ns.models_config)
    spec = find_model_spec(cfg, ns.model)
    with open(ns.detector_config, encoding="utf-8") as fh:
        taus = yaml.safe_load(fh)["tau_sweep"]

    model, tokenizer = load_model(spec, cfg.get("defaults", {}))

    # Same corpus and same reader as the quantization grid. Sink mass and
    # quantization damage are compared against each other in README §5.3, so
    # measuring them on different document sets would make that comparison
    # meaningless — and the earlier sink runs did exactly that, recording no
    # corpus at all (C19).
    from quant.evaluate import DEFAULT_CORPUS, _sha256_prefix, load_corpus

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
        print("corpus: STREAMED — a different document set from the committed grid")

    results = measure_model(
        model, calibration_batches(tokenizer, args, text_source), taus, device=ns.device
    )

    bos_tag = "bos" if args.prepend_bos else "nobos"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_calib{args.calib_seed}_{bos_tag}.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"args": asdict(args), "spec": spec, **provenance, **results}, fh, indent=2
        )

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
