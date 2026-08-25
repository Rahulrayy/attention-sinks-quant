# Limitations

Written before the results exist, which is the only way this document stays
honest. Items 7-9 were added on 2026-08-20 after the Day-0 checkpoint audit;
items 18-20 on 2026-08-25, after the corpus swap and the 6-bit run.

### 1. No rotation-based quantization
QuaRot / SpinQuant-style Hadamard rotation is arguably the strongest modern
outlier defence and it is **not** tested here. Implementing it properly is its
own project. "Does sink mitigation survive rotation?" is the obvious next
question and is stated here as future work rather than left for a reader to
notice. *(Note for whoever picks it up: Qwen's non-power-of-2 embedding
dimensions need the Paley construction for Hadamard matrices.)*

### 2. Scale
Track B models are ~17M parameters; Track A tops out at 1B. Nothing here
establishes how sink-mitigation trade-offs behave at 7B+.

### 3. Fake quant is not deployed quant
Rounding error is simulated in fp16. No claims are made about latency,
throughput, or memory in a real serving stack.

### 4. Single data distribution for Track B
The Anatomy paper argues short-context training is causally implicated in sink
formation. Track B trains at ctx 512 throughout and therefore cannot separate
that factor from the architectural one.

### 5. The controlled comparison is one lab's training run
Even with all three arms released, they are a single matched set from one
group. Qwen3-0.6B-Base and GPT-2 provide external validity, but weakly.

### 6. KV-cache quantization is not separated from activation quantization
Unless time allows a follow-up axis, the two are confounded.

### 7. All three arms of the controlled set already use QK-Norm
Verified from `config.json`: `use_qk_norm: true` in `1B_baseline`,
`1B_gate_headwise`, and `1B_gate_elementwise` alike. The set therefore isolates
**gating on top of QK-Norm** — which is the right test for this project's
question — but it says nothing about whether gating and QK-Norm are substitutes
for one another. GPT-2 small is the only checkpoint here without QK-Norm, and
it differs in tokenizer, data, scale, and position encoding simultaneously, so
it cannot carry that comparison on its own.

### 8. The gated_attention weights carry no declared licence
The GitHub implementation is MIT. The HF repo `QwQZh/gated_attention` has **no
licence field and no licence tag** (checked 2026-08-20). Absent a declaration,
reuse terms are unclear. This repo therefore redistributes no weights and only
references the upstream path. Resolving this upstream is an open action item.

### 9. Sink measurement depends on vendored modelling code
The checkpoints ship their own `modeling_qwen3.py` / `configuration_qwen3.py`
pinned at `transformers==4.46.0` and require `trust_remote_code=True`. Hooks
target module names resolved at runtime from that vendored code, not stock
`transformers`. A future transformers release that breaks the vendored file
would change what is being measured; the pin in `requirements.txt` guards this.

### 10. The headline metric was refined before any data was collected
Plan §6 locks metric definitions before results, and requires any change to be
recorded here rather than made quietly. One change was made, on 2026-08-20,
before a single model had been loaded.

`D_sink` as originally defined sums two effects that point the same way but mean
different things: **(a)** the sink token's own prediction stops being corrupted,
and **(b)** every other token gets a tighter shared scale because the outlier no
longer drags the range out. Only (b) is the claim under audit. Effect (a) is
close to tautological — holding any token in fp16 removes that token's own
quantization error — and it is present under *every* granularity.

On a linear toy model with no cross-token mixing, where (b) is structurally
impossible, per-token scaling still produced ~99% of per-tensor's `D_sink` at
4 bits. All of it was effect (a). Reporting only the summed metric would have
made per-token scaling look nearly as sink-sensitive as per-tensor and pointed
the audit at the wrong conclusion.

`D_sink` is therefore reported decomposed, with the non-sink restriction as the
headline. The decomposition is computed in **nats, not perplexity**: perplexity
is `exp(mean(nll))` and is not additive over token subsets, so attributing a
share of a perplexity delta to a group of tokens is ill-defined — a first draft
of the metric returned a contamination share of 1.106. The plan-§6 perplexity
number is still reported alongside for comparability with the prior literature.

Caveat that cannot be engineered away: in a real decoder the two effects are not
fully separable, because a corrupted sink propagates to other tokens through
attention. The decomposition bounds the confusion; it does not eliminate it.

### 11. The sink detector was changed from the plan-§6 definition
Plan §6 defines the detector as `sink(t) = M(t) > τ · median_t(M(t))` with
`M(t) = max_ℓ m(t,ℓ)`. That form maxes over layers **before** the median is
taken, so the denominator is set by whichever layer has the largest typical
activation, compressing every ratio toward 1.

Measured, the compression is severe enough to break the detector outright:

| model | sink ratio, aggregate | sink ratio, layer-relative |
|---|---|---|
| GPT-2 small | 14.2× | — (single sink) |
| Qwen3-0.6B-Base | 19.3× | **1153.9×** |

Against the plan's original grid (τ from 10 to 100) the aggregate detector
flagged **nothing at all** above τ=10 on either model, including on GPT-2 where
the sink is plainly visible in the attention maps (mean sink mass 0.67 by layer
10, every head above 0.5). The τ grid was first extended downward to
`{2,5,10,20,50,100}`; that helped but left only a narrow τ ∈ {5,10} window in
which the detector both fired and agreed with attention.

Normalising within each layer before maxing over layers gives a detector that
validates across the full τ range from 5 to 100 on Qwen3-0.6B-Base. Width of
the stable range is the criterion: a detector that agrees with attention only
inside a narrow band of τ has been fitted to one model, not validated.

Both detectors are computed and both are written to every run JSON;
`primary_detector` names the layer-relative one. The plan-§6 form is retained so
that changing the metric stays visible in the output rather than becoming
invisible in a refactor.

Not claimed: that the layer-relative form is correct in some absolute sense.
It is better-conditioned on the two checkpoints measured so far. Whether it
recovers the multi-level sinks that CushionCache reports for larger Qwen models
is untested — Qwen3-0.6B-Base showed only a single sink at position 0 under
both detectors, and positions 30/31 were flagged at τ=2 by both but **failed**
attention validation, i.e. they are magnitude outliers that no head attends to.

### 12. Qwen3-0.6B-Base has no BOS token, so it has no BOS arm
Plan trap §9.4 requires every model to run both with and without an explicit
BOS. Verified 2026-08-20: `AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")`
returns `bos_token_id = None`. The plan's premise that "Qwen and Llama do"
prepend BOS does not hold for this checkpoint.

The BOS axis is therefore absent for this model rather than merely unbalanced.
Substituting `eos_token_id` would change what occupies position 0, which is the
precise confound §9.4 exists to prevent, so the code refuses. Cross-model
comparisons involving Qwen3-0.6B-Base must be read against the no-BOS arm of
the other checkpoints only.

### 13. The element-wise gated arm is not parameter-matched to the baseline
The gate in `QwQZh/gated_attention` is implemented as extra output width on
`q_proj`, not as a separate module. Computed from the released configs
(H=2048, L=28, 16 heads, head_dim 128):

| arm | `q_proj` out | total `q_proj` params | Δ vs baseline |
|---|---|---|---|
| `1B_baseline` | 2048 | 117.4M | — |
| `1B_gate_headwise` | 2064 | 118.4M | +0.9M (+0.1%) |
| `1B_gate_elementwise` | 4096 | 234.9M | **+117.4M (≈ +12%)** |

The project's plan describes this triple as differing in "one architectural
component". That holds for the head-wise arm. For the element-wise arm — the
variant the paper reports as best, and therefore the one a reader will care
most about — the component costs roughly 12% more parameters on a ~1B model.

Any quantization-robustness difference between `baseline` and `elementwise` is
therefore **confounded between the gate and the added capacity**, and this
project cannot separate them: doing so would need a width-matched ungated
control, which is not released.

Mitigation, not a fix: `baseline` vs `headwise` is genuinely matched at +0.1%
and is reported as the primary controlled comparison. The element-wise contrast
is reported alongside with the confound stated explicitly. Figure 1 leads with
the matched pair.

### 14. No Qwen checkpoint in the roster has a BOS token
Extends §12 from Qwen3-0.6B-Base to all three `gated_attention` arms: their
tokenizers report `bos_token_id = None` (the *config* carries 151643, but the
tokenizer, which builds the batches, does not).

Four of the five checkpoints therefore have no BOS arm; only GPT-2 does. Plan
trap §9.4 requires running every model both with and without an explicit BOS.
That is **unexecutable as written**.

What survives: every model runs no-BOS, so all cross-model comparisons are made
under a matched policy. What is lost: the ability to ask whether BOS presence is
what drives the position-0 sink. GPT-2 alone can answer that within-model, and a
single checkpoint cannot separate a BOS effect from an architecture effect. The
question is out of scope for this project rather than answered by it.

### 15. The released arms are 1.72B parameters, not the "1B" their names suggest
Measured after loading. Consequence for anyone reproducing this: budget ~3.3 GB
of disk per arm (~10 GB for the trio, ~12 GB with the other two checkpoints) and
~3.44 GB of VRAM for weights in bf16, against the ~5 GiB free on a 6 GB card
that is also driving a display.

### 16. `trust_remote_code` and `subfolder` do not compose in transformers
`from_pretrained(repo, subfolder=..., trust_remote_code=True)` fails for this
repo: a config's `auto_map` names bare module files and transformers resolves
them against the repo root, ignoring `subfolder`. The arms vendor their code
inside each subfolder with nothing at the root.

Worked around in `sinks.measure.resolve_model_path` by materialising the
subfolder locally and loading from that directory. Anyone reproducing on a
different transformers version should check this path first — it is the most
version-fragile part of the loading code.

### 17. The Day-2 detector gate was redesigned mid-project
As originally implemented, the gate aborted whenever the magnitude detector
flagged no tokens at any τ. On `1B_gate_headwise` it did exactly that — and the
abort was wrong: that model genuinely has no attention sink, which is the
result its authors claim for it.

A gate phrased as "X must be present" fails whenever the absence of X is itself
a legitimate answer. The gate now distinguishes the two cases using attention:
it aborts only when the detector finds nothing *while* a non-trivial fraction of
heads have sink mass > 0.5. Runs record `sink_free` explicitly.

Consequence for interpretation: `D_sink` is undefined for a sink-free model —
there are no sink positions to exempt, so the `detected_sinks` arm of the
quantization grid cannot be constructed for it. That is not a gap in the data;
it is what "sink-free" means. Comparisons involving such a model must use the
`none` and `position_0` arms only, and the README must say so rather than
leaving an empty cell for a reader to misread as a missing run.

### 18. Every number rests on a single corpus, and that sensitivity is large
The grid has been run twice: once on a small in-repo corpus, once on 281
FineWeb-Edu documents. The second is a better evaluation set — `ppl_ref` fell
136.6 → 29.7 on GPT-2 and 54.9 → 17.9 on Qwen3-0.6B, which is what identified
the first as not being a language-model evaluation set at all.

It is a *better* corpus, not a *sample* of corpora, and the swap was not
cosmetic:

| Claim | On the in-repo corpus (R3) | On FineWeb-Edu (R3-rev) |
|---|---|---|
| Matched-pair reduction | 269× | **55×** |
| Per-token `D_sink` excludes zero | on 1 of 5 models | on **2 of 5** |
| Head-wise / element-wise inversion | 34× gap | **14.5× gap**, held |

One swap moved the headline by ~5× and moved two CIs across zero. Two corpora
cannot bound a sensitivity that large; they only demonstrate it exists. Nothing
here licenses a claim of the form "`D_sink` under per-token scaling is X nats"
without naming the corpus. The findings that survived the swap — the ranks, and
the inversion — are the ones to weight, and they are weighted that way in the
README for exactly this reason.

Both corpora are committed under `data/` with hashes, so the two grids remain
distinguishable and re-checkable. Re-streaming FineWeb-Edu gives a different
document set and is a new experiment, not a reproduction.

### 19. The 4-bit cells are uninterpretable, and 6-bit is what answers that
At 4-bit activations, Δppl runs into the millions. Both arms are destroyed
models, and a perplexity difference between two destroyed models carries no
information about which one degrades more gracefully — the metric has left its
valid range, not merely become noisy.

This matters because the 4-bit cells appear to show a large per-token `D_sink`,
which would suggest the redundancy result is bit-width dependent. That reading
cannot be supported by cells in a destroyed-model regime, and `fig2_bitwidth`
carries the caveat on the figure itself rather than in a caption.

The 6-bit grid was run to put a width in between, where the model still works
(§5.5 of the README). Its cells are interpretable and they do change the answer
— which is the reason the gap was worth closing rather than declaring open.
What remains unbounded is everything below 6 bits: nothing here establishes
where the redundancy stops, only that it is intact at 8, weakening at 6, and
unmeasurable by this metric at 4.

### 20. The held-out slice is 32 sequences, not the budget `configs/quant.yaml` states
`configs/quant.yaml` declares `eval_tokens: 262144` and
`calibration_tokens: 65536`. Every cell actually run used **8192 held-out
tokens and 2048 tokens per calibration draw** — the committed corpus is ~300k
tokens and the configured layout needs ~590k for five disjoint draws, so the
config's budget was never achievable against it. `build_slices` refuses the
short layout rather than silently recycling tokens across draws, so this shows
up as an error, not as a quiet overlap.

The consequence is on the intervals, not the point estimates: the held-out
slice is **32 sequences of 256 tokens**, and every CI in this repo is a
percentile bootstrap over those 32 sequences. That is a small resample base.
Intervals should be read as indicative of sign and order of magnitude, not as
precise widths, and two intervals that merely fail to overlap on 32 sequences
are weaker evidence than the same result on a larger slice.

The actual budget is now recorded per-cell (`seq_len`, `calib_tokens`,
`eval_tokens`, `corpus_sha256`, `holdout_sha`) in every run JSON written from
2026-08-25 onward, so a cell carries its own provenance instead of inheriting a
config value it did not use. Cells written before that date do not carry these
fields; they were all run at 256 / 2048 / 8192 on `data/fineweb_edu.txt`, and
their `ppl_ref` matches the re-run cells exactly, which is how the older cells
were confirmed to be on the same held-out slice (C19).

`holdout_sha` is the one that earns its place: it fingerprints the held-out
*token stream*, and two cells are comparable only if it matches. The corpus
hash cannot do that job — a changed reader produces a different token stream
from byte-identical input.
