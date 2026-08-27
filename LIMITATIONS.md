# Limitations

Written before the results exist, which is the only way this document stays
honest. Items 7-9 were added on 2026-08-20 after the Day-0 checkpoint audit;
items 18-20 on 2026-08-25, after the corpus swap and the 6-bit run; items 21-22
on 2026-08-26, with the layer-0 localisation and the downstream task.

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

One swap moved the headline by ~5× and moved two CIs across zero. Nothing here
licenses a claim of the form "`D_sink` under per-token scaling is X nats"
without naming the corpus.

**A third corpus has since been added** (`data/code_python.txt`, 1.2M characters
of Python source) and 8-bit cells re-run on it. This bounds the sensitivity for
the first time, and the bound is uneven — the point of §5.7 and of
`analysis/corpora.py`:

| Claim | Survived the domain shift? |
|---|---|
| The fix works (matched pair, per-tensor) | **yes** — 55× web, 46× code |
| The redundancy (per-token absorbs it) | **yes, more strongly** — max \|`D_sink`\| 0.0117 nats, 3/5 point estimates negative |
| R4, element-wise destroyed | **yes, worse** — 1003× its reference vs 257× |
| R6 localisation (layer-0 MLP, causal) | **yes, stronger** — 3265× vs 530× from three modules |
| R6 axis attribution (row dispersion) | **n/a — wrong on both corpora** (**C23**) |
| R6 cross-model ranking (correlational) | **no** — orders the roster at no threshold (**C21**) |
| R5 direction (head-wise least damaged, 8 and 6 bits) | **yes** — clear margin on both |
| R5 "per-tensor survives at 6 bits on head-wise alone" | **no** — nothing survives on code (**C22**) |
| R5 8→6 per-token growth rates | **no** — reproduce on GPT-2 only |
| R5 "4-bit answers nothing" | **yes** — everything destroyed on both |

The pattern is not that "everything held" or "everything moved". Interventions
with controls held; a rank correlation over five models did not. That is the
distinction to carry forward, and it is why the retraction is filed as a
correction rather than a caveat.

**What is still unbounded.** Three corpora, all English, all 1.2M characters,
all read whole-file at seq_len 256. Nothing here speaks to other languages,
much longer contexts, or genuinely adversarial input. The code arm now covers
8, 6 and 4 bits, so §5.5 is corpus-checked; the R6 *statistic* is still 8-bit
only on both corpora (§21).

All three corpora are committed under `data/` with hashes, so the grids remain
distinguishable and re-checkable. Re-streaming any of them gives a different
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

**6 bits is narrower than it looked, and on the second corpus it is narrower
still.** C22: on Python source, per-tensor at 6 bits destroys **all five**
models, head-wise included — the FineWeb-Edu grid's one survivor sits at 8.7×
its reference and the same cell on code sits at 12.4×. So the interpretable
window for per-tensor is 8 bits on both corpora and nothing below it on either.
Per-token stays interpretable at 6 on both. The 8→6 *growth rates* quoted in
§5.5 reproduce on GPT-2 alone; treat them as one corpus's numbers, not as a
measured rate of decay.

### 20. The held-out slice is 32 sequences
Every cell was measured with **8192 held-out tokens and 2048 tokens per
calibration draw**. The plan asked for 262144 and 65536, which the committed
corpus cannot supply: five *disjoint* draws plus the holdout need ~590k tokens
against its ~264k. `build_slices` refuses the short layout rather than silently
recycling tokens across draws, so this shows up as an error, not as a quiet
overlap.

`configs/quant.yaml` now states the budget that was actually used, so the
defaults reproduce the published grid; the plan's original figures are kept
there in comments. Until 2026-08-25 it declared the unachievable ones, which
meant the config and the results had never agreed.

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

### 21. The layer-0 localisation is causal; the ranking and the axis story were not
README §5.4 makes two claims of different strength and they should not be read
as one.

The **causal** claim is strong. Holding the layer-0 MLP in fp16 moves the
element-wise arm from +3481.54 to +6.57 (530×) while the same exemption on both
sibling checkpoints does nothing and a 24-module exemption of other blocks
changes almost nothing. That is an intervention with its own controls, on one
model, and it stands on its own.

The **ranking** claim was weak and is now **retracted (C21)**. "Annihilated-layer
count orders the roster by per-tensor damage" rested on **five models**, one of
which (element-wise) is 258× more damaged than the next and therefore carried
most of the apparent correlation on its own. This section originally said that
five points cannot distinguish a mechanism from a coincidence that happens to
sort correctly. The second corpus showed it was the coincidence: on Python
source the count orders the roster at **no threshold**, against three thresholds
on FineWeb-Edu (README §5.7).

What remains true of the statistic is narrower and threshold-shaped. Partial
underflow is survivable and near-total underflow is not — head-wise carries many
layers at 50–70% underflow and is the least damaged model on both corpora — so
the count detects the *kind* of failure without measuring its size or ranking
models against each other.

**The axis attribution was wrong and is corrected (C23).** This section, and
README §5.4, originally described the failure as *row* dispersion — reasoning
that row structure is the only thing per-token scaling can divide out, and
therefore the only thing that could explain a gap between the two
granularities. The premise is sound and the description was never checked
against the transposed statistic. It does not hold: the tensor is extreme on
**both** axes and more so on the feature axis (304× against 28.4×), and the
feature axis is the more dispersed one at every model's worst layer in the
roster — including head-wise, the least damaged model here, at 248.9×.
Per-feature underflow sits at 0.01–0.06 on damaged and undamaged models alike.

So per-tensor underflow remains the only statistic that tracks the damage, and
the causal result is untouched. What is gone is the tidy account of *why*
per-token suffices. It takes that tensor from 99.3% underflow to 46%, not to
zero; 46% at one layer is survivable and 99.3% is not. That is the honest
version and it explains less.

What is not explained at all is **why** an element-wise output gate reshapes
that particular tensor during training. The measurement locates the failure and
says nothing about its origin. Establishing that would need training-time
evidence — checkpoints over the course of a run — which this project does not
have and which its de-scoping section (Track B) already ruled out.

Finally, all of it is one bit width. The dispersion statistic is
bit-width-dependent by construction: `eff_bits` and both underflow fractions are
computed against a specific integer grid, so the layer-0 tensor's ~99.2%
underflow is an **8-bit** number and the 6- and 4-bit equivalents have not been
measured on either corpus.

### 22. LAMBADA is one task, in one language, at one sample size
The downstream check (README §5.8) is a real second metric and it is not a
general claim about downstream behaviour.

**One task.** LAMBADA is last-word prediction with a long-range dependency. It
was chosen because that is the hardest thing a short-context perplexity number
can hide, and because a published fp16 accuracy exists to validate the protocol
against — GPT-2 small scores 0.3070 here on 1000 examples against ~0.326
published on the full 5153, which is the check that the implementation is the
standard one. Nothing here speaks to summarisation, code, instruction
following, or any generative task. A quantized model that answers LAMBADA
correctly can still be degraded in ways this does not touch.

**One language.** English, like all three corpora (§18).

**1000 of 5153 examples**, taken in dataset order and fingerprinted by
`examples_sha`. `python -m analysis.lambada --power` prints what that buys: the
half-width of each cell's paired interval, which is the smallest accuracy drop
the sample could have called non-zero. It runs ±1–3 points. **A cell whose drop
is smaller than its own resolution is a bound, not a measurement**, and several
per-token cells are exactly that. They are reported as intervals crossing zero
rather than as zeroes.

**Accuracy saturates, and perplexity does not.** Once a model is at chance,
further damage cannot lower its score. This is the accuracy analogue of §19's
destroyed-cell problem and it bites on the same cells: a model at 241× its
reference perplexity is at the floor, and the distance between it and a model
at 8000× is unmeasurable by this metric. Cells at or below 2% accuracy are
marked `⌊` in the generated table. Read a floored cell as "broken", never as a
rank.

**The calibration corpus is not the evaluation corpus.** Static per-tensor
scales come from the same FineWeb-Edu draw the grid uses, not from LAMBADA.
That is deliberate — it is the deployed setting, calibrate once on general text
and then run whatever arrives — but it means a static per-tensor cell here
carries both quantization damage and a calibration/evaluation distribution
shift. The `per_tensor_dynamic` arm, which never reads the calibration set, is
the control for exactly that, and it is run for every model.

**Greedy exact-match is one of several LAMBADA protocols.** It is
lm-eval-harness's `acc`, kept identical so the fp16 numbers are checkable
against published ones. A ranked-candidate or stop-word-filtered variant would
give different absolute numbers; whether it would give different *orderings* is
untested here.
