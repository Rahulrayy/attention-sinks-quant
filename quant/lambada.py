"""LAMBADA: the one downstream task, and the only metric here that is not perplexity.

Every claim this project makes is an ordering of perplexity damage, and four of
its corrections were about orderings of perplexity damage that did not survive a
second corpus, a second threshold, or a second axis (C20–C23). A differently
shaped metric is the cheapest independent check on whether any of those
orderings mean anything, and the plan has asked for one since it was written.

The task is last-word prediction: given a passage whose final word is
determined by long-range context, does the model produce that word.

    accuracy   1 if greedy decoding reproduces EVERY target token, else 0.
               This is lm-eval-harness's `acc` for lambada_openai, kept
               identical so the fp16 numbers can be sanity-checked against
               published ones rather than trusted.
    target_nll mean negative log-likelihood of the target tokens alone.

Accuracy is the point. It is bounded, discrete, and insensitive to the kind of
small smooth degradation perplexity reports, so it answers a question perplexity
structurally cannot: **does the damage reach behaviour?** A model can lose 0.02
nats and get every answer right; that is a real finding about whether the
mitigation debate matters, not a null result.

**Why the comparison is paired, and what to read when it is not.** Both arms
score the SAME examples, so the difference is paired example-by-example and goes
through `analysis.stats.sequence_bootstrap` like every other interval in the
repo. On a 0/1 outcome the paired difference is in {-1, 0, +1}, and every
example the two arms agree on contributes exactly zero — so all of the
information lives in the **discordant** pairs, and `n_discordant` is recorded in
every run. A comparison with three discordant pairs out of a thousand is
powerless no matter how large the sample, and it should be visible that it is
powerless rather than reported as a tight interval around zero. This is C18's
lesson in a different metric: an absent measurement must not look like a precise
one.

**Calibration.** Static per-tensor scales come from the same FineWeb-Edu draw
the grid uses, not from LAMBADA. That is the deployed setting — calibrate once
on general text, then run whatever arrives — and it keeps these cells directly
comparable to the grid's. The consequence is that a per-tensor cell here carries
both quantization damage and a calibration/eval distribution shift; the dynamic
arm, which never reads the calibration set, is the control for that and is
recorded alongside.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .evaluate import DEFAULT_CORPUS, _sha256_prefix, load_corpus, provenance_path

# The lm-eval-harness split. Detokenized (properly cased and punctuated), which
# matters because a BPE tokenizer scores "n't" and " not" very differently.
DATASET = "EleutherAI/lambada_openai"
CONFIG = "en"


def load_texts(n_examples: int, dataset: str = DATASET, config: str = CONFIG) -> list[str]:
    """First ``n_examples`` of the test split, in dataset order.

    Order is not shuffled, so the selection is a deterministic function of
    ``n_examples`` and needs no shuffle seed to reproduce. The exact texts are
    fingerprinted by ``examples_sha`` in every run rather than committed: unlike
    `data/fineweb_edu.txt`, which was streamed and could not be re-obtained,
    this is a fixed versioned split. If the hash ever changes, the split moved
    and the cells are not comparable — same contract as `holdout_sha`.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, name=config, split="test", streaming=True)
    out = []
    for rec in ds:
        text = (rec.get("text") or "").strip()
        if text:
            out.append(text)
        if len(out) >= n_examples:
            break
    if len(out) < n_examples:
        raise RuntimeError(
            f"{dataset} yielded {len(out)} usable examples, fewer than the "
            f"{n_examples} requested"
        )
    return out


def examples_sha(texts: list[str], n: int = 16) -> str:
    """Fingerprint of the exact examples scored. The `holdout_sha` of this task."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:n]


def build_examples(tokenizer, texts: list[str], max_context: int = 512) -> list[dict]:
    """Split each passage into context and final word, and tokenize both.

    The target keeps its leading space (`" word"`), which is how it appears in
    the passage and therefore how a BPE tokenizer would have produced it. Split
    it as a bare `"word"` and the model is asked to predict a token sequence
    that never occurs mid-sentence, which depresses accuracy on every model
    equally and silently.
    """
    out = []
    for text in texts:
        context, sep, target = text.rpartition(" ")
        if not sep or not context or not target:
            continue
        ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        tgt_ids = tokenizer(" " + target, add_special_tokens=False)["input_ids"]
        if not ctx_ids or not tgt_ids:
            continue
        # Truncate from the LEFT: LAMBADA's whole design is that the answer
        # depends on the far context, but the tokens nearest the target are the
        # ones the model conditions on most directly, and dropping those would
        # measure something else entirely.
        ctx_ids = ctx_ids[-max_context:]
        out.append({"context": ctx_ids, "target": tgt_ids, "text": text})
    if not out:
        raise RuntimeError("no usable LAMBADA examples after tokenization")
    return out


@torch.no_grad()
def score(model, examples: list[dict], *, device: str = "cuda") -> dict:
    """Per-example greedy-match and target NLL.

    Batch size 1, as everywhere in Track A (trap §9.2), and here it is also
    forced by the data: contexts vary in length and left-padding a decoder
    changes the positions the model sees.
    """
    correct, nlls, n_tokens = [], [], []

    for ex in examples:
        ids = torch.tensor([ex["context"] + ex["target"]], dtype=torch.long, device=device)
        logits = model(ids, use_cache=False).logits
        n_tgt = len(ex["target"])

        # Position j predicts token j+1, so the target tokens are predicted by
        # the logits at [n_ctx-1 : -1].
        tgt_logits = logits[0, -n_tgt - 1 : -1].float()
        tgt_ids = ids[0, -n_tgt:]

        logp = torch.log_softmax(tgt_logits, dim=-1)
        tok_nll = -logp.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)

        correct.append(bool((tgt_logits.argmax(dim=-1) == tgt_ids).all().item()))
        nlls.append(float(tok_nll.sum().item()))
        n_tokens.append(n_tgt)

    n = len(correct)
    total_tokens = sum(n_tokens)
    return {
        "n_examples": n,
        "accuracy": sum(correct) / n,
        "target_nll": sum(nlls) / total_tokens,
        "target_ppl": float(torch.tensor(sum(nlls) / total_tokens).exp().item()),
        "per_example_correct": [int(c) for c in correct],
        "per_example_nll": nlls,
        "mean_target_tokens": total_tokens / n,
    }


def evaluate_lambada_cell(
    model,
    *,
    calib_batches_fn,
    examples: list[dict],
    bits: int,
    act_granularity: str,
    static: bool = True,
    device: str = "cuda",
) -> dict:
    """One quantization configuration, scored against its own fp16 reference.

    Mirrors `quant.evaluate.evaluate_cell`: the reference runs with quantization
    DISABLED on the already-patched model, so reference and quantized passes go
    through byte-identical code paths and any difference is quantization rather
    than a change of execution path.

    ``static=False`` skips calibration and lets per-tensor derive its range from
    each tensor it sees. That is the control for the calibrate-on-FineWeb /
    evaluate-on-LAMBADA distribution shift described in the module docstring.
    """
    from .calibrate import collect_ranges, reset_ranges
    from .patch import patch_model, set_quant_enabled

    restore, patched = patch_model(
        model, w_bits=bits, a_bits=bits,
        w_granularity="per_channel", a_granularity=act_granularity,
    )
    try:
        set_quant_enabled(model, False)
        ref = score(model, examples, device=device)

        set_quant_enabled(model, True)
        reset_ranges(model)
        ranges = {}
        if act_granularity == "per_tensor" and static:
            ranges = collect_ranges(model, calib_batches_fn(), device=device)
        quant = score(model, examples, device=device)
    finally:
        restore()
        torch.cuda.empty_cache()

    # The paired difference. Every example the two arms agree on contributes
    # exactly zero, so the discordant count is the real sample size and is
    # reported rather than left to be inferred from the interval.
    pairs = list(zip(ref["per_example_correct"], quant["per_example_correct"]))
    n_discordant = sum(1 for a, b in pairs if a != b)

    return {
        "bits": bits,
        "act_granularity": act_granularity,
        "static": static,
        "n_patched": len(patched),
        "n_calibrated": len(ranges),
        "accuracy_ref": ref["accuracy"],
        "accuracy_quant": quant["accuracy"],
        "accuracy_drop": ref["accuracy"] - quant["accuracy"],
        "target_ppl_ref": ref["target_ppl"],
        "target_ppl_quant": quant["target_ppl"],
        "target_ppl_ratio": quant["target_ppl"] / ref["target_ppl"],
        "n_discordant": n_discordant,
        "n_examples": ref["n_examples"],
        "mean_target_tokens": ref["mean_target_tokens"],
        "per_example_correct_ref": ref["per_example_correct"],
        "per_example_correct_quant": quant["per_example_correct"],
        "per_example_nll_ref": ref["per_example_nll"],
        "per_example_nll_quant": quant["per_example_nll"],
    }


def main() -> None:
    from .calibrate import build_slices, to_batches, tokenize_stream

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument(
        "--granularities", default="per_tensor,per_token",
        help="comma list. `per_tensor_dynamic` selects the uncalibrated "
             "per-tensor control on its own, so a run that already has the "
             "static arm need not repeat it -- each arm costs two full passes "
             "over the task and the arms are independent.",
    )
    parser.add_argument("--dynamic", action="store_true",
                        help="append per_tensor_dynamic to whatever "
                             "--granularities selected")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--calib-tokens", type=int, default=2048)
    parser.add_argument("--eval-tokens", type=int, default=8192)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="runs/lambada")
    parser.add_argument("--text-file", help="calibration corpus; defaults to the grid's")
    parser.add_argument("--models-config", default="configs/models.yaml")
    ns = parser.parse_args()

    from sinks.measure import find_model_spec, load_config, load_model

    cfg = load_config(ns.models_config)
    spec = find_model_spec(cfg, ns.model)
    model, tokenizer = load_model(spec, cfg.get("defaults", {}))
    model.to(ns.device)

    texts = load_texts(ns.n_examples)
    examples = build_examples(tokenizer, texts, max_context=ns.max_context)
    sha = examples_sha(texts)

    corpus = ns.text_file or str(DEFAULT_CORPUS)
    ids = tokenize_stream(tokenizer, load_corpus(corpus), ns.eval_tokens + 5 * ns.calib_tokens)
    slices = build_slices(
        ids, n_draws=5, tokens_per_draw=ns.calib_tokens, eval_tokens=ns.eval_tokens
    )
    bos = tokenizer.bos_token_id

    def calib_fn():
        return to_batches(slices.draw(ns.calib_seed), ns.seq_len, bos_token_id=bos)

    print(f"{ns.model}  {len(examples)} examples  sha={sha}  bits={ns.bits}", flush=True)

    # `per_tensor_dynamic` is spelled as a granularity so a single arm can be
    # named, then split back into (granularity, static) for evaluate_lambada_cell.
    arms = []
    for g in ns.granularities.replace(" ", "").split(","):
        if not g:
            continue
        arms.append((g[: -len("_dynamic")], False) if g.endswith("_dynamic")
                    else (g, True))
    if ns.dynamic and ("per_tensor", False) not in arms:
        arms.append(("per_tensor", False))
    if not arms:
        raise SystemExit("--granularities selected no arms")

    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for gran, static in arms:
        r = evaluate_lambada_cell(
            model, calib_batches_fn=calib_fn, examples=examples, bits=ns.bits,
            act_granularity=gran, static=static, device=ns.device,
        )
        tag = gran if static else f"{gran}_dynamic"
        print(f"  {tag:<20} acc {r['accuracy_ref']:.4f} -> {r['accuracy_quant']:.4f}"
              f"  (drop {r['accuracy_drop']:+.4f}, {r['n_discordant']} discordant)"
              f"   target_ppl x{r['target_ppl_ratio']:.2f}", flush=True)

        payload = {
            "model": ns.model, "task": "lambada_openai",
            "dataset": DATASET, "config": CONFIG,
            "n_examples_requested": ns.n_examples, "examples_sha": sha,
            "max_context": ns.max_context,
            "calib_corpus": provenance_path(corpus),
            "calib_corpus_sha256": _sha256_prefix(corpus),
            "calib_seed": ns.calib_seed, "seq_len": ns.seq_len,
            "calib_tokens": ns.calib_tokens,
            **r,
        }
        path = out_dir / f"{ns.model}_b{ns.bits}_{tag}_lambada.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    main()
