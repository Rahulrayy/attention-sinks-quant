# Handoff — Attention Sinks and Quantization: An Audit

**Date:** 26 August 2026 (fourth session)
**State:** Track A complete and stress-tested. Every finding checked on a second
corpus at all three bit widths, and R6's mechanism checked on the axis it was
attributed to. Four corrections came out of that (C20–C23), all of the same
kind: the measurements held, the explanations wrapped around them did not.
Track B not started.
**Authoritative sources:** `attention-sinks-quant-plan.md` §0 (corrections log), `LIMITATIONS.md`, `README.md`, `runs/results/`
**Numbers:** every result table in the README is generated — `analysis.report` (§5.1–5.5), `analysis.distributions` (§5.4), `analysis.corpora` (§5.7). Do not transcribe them by hand — that is how the README came to carry superseded figures for five days (C19 has the story).

---

## 1. Status at a glance

| | |
|---|---|
| Tests | **163 passing**, 14 files |
| Track A code (`sinks/`, `quant/`, `analysis/`) | **complete** — 0 stubs |
| Track B code (`train/`) | **~15%** — 9 stubs, unchanged |
| Quantization grid | **300 cells** at 8/6/4-bit on FineWeb-Edu, **+60** at 8/6/4-bit on the code corpus (+200 archived on the old corpus) |
| Corpora | **3** committed — FineWeb-Edu (current), Python source (second-corpus arm), in-repo markdown (archived) |
| Diagnostic runs | **23** — 14 web + 9 code, arm decomposition plus localisation tests |
| Distribution runs | **10** — 5 per corpus, re-measured with the feature-axis statistic (C23) |
| Sink measurement runs | **5** — one per checkpoint, draw 0, no-BOS |
| Corrections to the original plan | **23** |
| Limitations recorded | **21** |
| Git | initialised 2026-08-25; the project was never a repository before |

### Done since the previous handoff

- **C23 — R6's mechanism was attributed to the wrong axis** (§3). The
  element-wise layer-0 tensor is extreme on *both* axes and **more** so on the
  feature axis (304× against 28.4×). The feature axis is the more dispersed one
  on every model in the roster, so it separates nothing. The localisation and
  its controls are untouched; the explanation around them is corrected.
- **The axis statistic** — `col_dispersion` and `underflow_col` in
  `quant/distributions.py`, `analysis.distributions --axis` to report them.
  All ten distribution runs re-measured; every previously-published number
  reproduced exactly.
- **The `outlier_channels` prediction from §11 was tested and failed** — before
  the intervention was run, on the measurement. That is what the prediction was
  for.

### Done in the session before that

- **R5 corpus-checked at 6 and 4 bits** (§3), the previous handoff's priority 1.
  40 more cells. Its *direction* claims travel; its thresholds and growth rates
  do not — **C22**.
- **`analysis.corpora --bitwidth`** — one printed number per sentence of README
  §5.5, so each can be checked rather than believed.

### Done earlier

- **R7 — the second corpus** (§3), which was the previous handoff's own
  priority 1 and the project's largest unbounded sensitivity. 1.2M characters of
  Python source, 20 grid cells at 8 bits, plus diagnose and distribution runs
  on all five checkpoints. It bounds the sensitivity for the first time and the
  bound is **uneven**: every intervention-with-controls held or strengthened,
  and the one rank correlation died.
- **C21** — R6's cross-model ranking is **retracted**. It orders the roster at
  three thresholds on FineWeb-Edu and at **none** on code. The causal half of R6
  reproduced and got stronger (3265× vs 530×). The lesson is in §10.
- **`analysis/corpora.py`** (317 lines) — joins two grids and computes a
  per-claim stability verdict rather than asserting one. This is the module
  whose output retracted C21.
- **`data/fetch_code.py`** — the second-corpus fetch, every parameter held
  identical to `fetch_fineweb.py` so domain is the only thing that differs.

### Done earlier still

- **R6 — the layer-0 localisation** (mechanism claim later corrected by C23) (§3). The element-wise arm's
  per-tensor catastrophe is **one tensor**: the layer-0 MLP input, at 28.4× row
  dispersion against 1.6× on both siblings. Holding three modules in fp16 takes
  it from 241× its reference perplexity to 1.45×, with controls.
- **C20** — the "damage is distributed across the network" claim was an
  inference from two failed searches, not a measurement. Retracted; §10.
- **`quant/distributions.py`** (304 lines) — per-layer row dispersion, effective
  bits, and per-granularity underflow. Quantizes nothing; runs through
  pass-through wrappers so the module set matches the grid exactly.
- **`analysis/distributions.py`** (308 lines) — the R6 tables, joined against
  the damage they must explain.

### Done earlier

- **6-bit grid** — 100 cells, the width that handoff named as the single
  genuinely open question. It has an answer (§3).
- **The fragility finding (R4)** — the element-wise arm is destroyed by
  per-tensor 8-bit quantization, which the `D_sink`-only reporting had hidden.
- **`quant/diagnose.py`** — splits a cell into weights / activations,
  static / dynamic, per-tensor / per-token, and can exempt named projections.
- **C19** — the repo CLI never reproduced the committed cells. Found, fixed,
  guarded, and the affected grid re-run (§7).
- **Corpora committed** under `data/` with hashes and a `.gitattributes` pin.
- **`quant.evaluate --grid`** — the scratchpad driver is now in the repo.
- **`analysis/report.py`** — the README's tables, generated not typed.

### Not done

- **Track B entirely.** `train/attention.py` has working `softmax1` and
  `OutputGate`; `CausalSelfAttention`, `model.py`, `data.py`, `train.py` are
  contract-only stubs. No pretraining run has happened.
- **LAMBADA.** The plan calls for one cheap downstream task. Only perplexity.
- **The `detected_sinks` and `outlier_channels` exception arms.** Never run in
  the grid. `detected_sinks` is one command; **`outlier_channels` has no
  end-to-end path at all** — the CLI never builds an `outlier_mask`, so
  selecting it raises. Found this session; §11.3.
- **The code corpus at 6 and 4 bits.** The second-corpus arm is **8-bit only**,
  so §5.5's bit-width conclusions have never been corpus-checked (LIMITATIONS
  §18). This is now the largest open sensitivity.
- **R6 at 6 and 4 bits.** The dispersion statistic is bit-width-dependent by
  construction and has only been run at 8, on either corpus (LIMITATIONS §21).
- **A fourth corpus.** Three corpora, all English, all 1.2M characters, all at
  seq_len 256. Nothing here speaks to other languages or long context.

---

## 2. The question

Three claims circulate in this literature. The first two are settled and are
*not* relitigated: a handful of token positions absorb disproportionate
attention mass (the sink), and those positions carry extreme residual-stream
outliers. The third is under audit:

> That these outliers are what breaks low-bit activation quantization, and that
> mitigating them architecturally buys measurable robustness.

That literature was built against **per-tensor** activation quantization and
**pre-QK-Norm** architectures. Both have moved on. The question is whether the
architectural fix still pays against per-token activation scaling on models that
already use QK-Norm — with confidence intervals, which almost nothing in the
prior work reports.

A negative result was always acceptable. It came back partly negative and partly
something nobody was looking for, which is the interesting case.

---

## 3. Findings

Six findings. R1, R3-rev, R4, R5 and R6 came earlier; **R7 is new — the second
corpus — and it retracts half of R6** (C21) while strengthening the other half.
Read R6 and R7 together; neither is complete alone.

`README.md` §5 carries the same numbers with fuller argument. Everything here is
draw 0, 95% sequence-bootstrap CIs over 32 held-out sequences, and on
`data/fineweb_edu.txt` unless a row says otherwise — R7 is the corpus arm and
names both.

### R1 — the gated checkpoints are genuinely sink-free

| Model | Mean sink mass | Heads > 0.5 | Entropy L0 → Lₙ | Magnitude | Sink-free |
|---|---|---|---|---|---|
| GPT-2 small (no QK-Norm) | 0.383 | 0.396 | 3.04 → 2.94 | 83.7× | no |
| Qwen3-0.6B-Base | 0.523 | 0.565 | 2.09 → 0.79 | 1124× | no |
| `1B_baseline` (control) | 0.439 | 0.464 | 2.58 → 1.32 | 370× | no |
| `1B_headwise` (+0.1% params) | **0.057** | **0.004** | 3.04 → 2.76 | **4.1×** | **yes** |
| `1B_elementwise` (+12% params) | 0.019 | 0.000 | 2.96 → 1.53 | 2.2× | **yes** |

**Re-measured this session on the committed corpus.** The previous numbers were
taken on a streamed sample that the run JSONs did not record, so sink mass and
quantization damage were being compared across two different document sets — the
comparison R4 turns on. Values moved by up to 13% (GPT-2 sink mass 0.340 →
0.383); every ordering and every conclusion held.

At **+0.1% parameters** the head-wise gate collapses the sink by **90×** in
residual magnitude, so sink elimination is not a capacity effect.

### R3-rev — the audit answer at 8 bits

| Model | Sink? | per-tensor (2023) | per-token (modern) | Ratio |
|---|---|---|---|---|
| GPT-2 small | yes | **+0.2097** [+0.188, +0.232] | +0.0047 [−0.002, +0.011] *ZERO* | 44× |
| Qwen3-0.6B-Base | yes | **+0.4378** [+0.396, +0.479] | +0.0096 [+0.003, +0.016] | 46× |
| `1B_baseline` | yes | **+0.6450** [+0.601, +0.690] | +0.0203 [+0.014, +0.026] | 32× |
| `1B_headwise` | **no** | +0.0117 [−0.0003, +0.024] *ZERO* | +0.0012 [−0.003, +0.005] *ZERO* | — |
| `1B_elementwise` | **no** | +0.1702 [+0.104, +0.236] **⚠ destroyed** | +0.0009 [−0.006, +0.007] *ZERO* | — |

**The answer has two halves and neither works alone.** *The fix works*: on the
matched pair, per-tensor `D_sink` falls +0.6450 → +0.0117, a **55×** reduction
with the CI crossing zero. *And it is redundant*: per-token scaling gets
**32–46×** on sink-bearing models without touching the architecture.

### R4 — the element-wise arm is destroyed by the quantization it was designed for

**This corrects the previous handoff.** That handoff's "finding I did not
expect" was that element-wise carries *more* sink-attributable damage than
head-wise (+0.1702 vs +0.0117, 14.5× gap) despite *less* sink mass. The
comparison was invalid: element-wise's per-tensor 8-bit cell has **Δppl +3707
against a reference of 14.5**, so both arms of that `D_sink` are destroyed
models. It is exactly the failure LIMITATIONS §19 flags at 4 bits, unnoticed at
8 because only `D_sink` was being reported and never the damage level.

The claim survives on total damage, where it is far stronger:

| by sink mass (least first) | | by per-tensor 8-bit Δppl (least first) | |
|---|---|---|---|
| `1B_elementwise` | 0.019 | `1B_headwise` | +2.2 |
| `1B_headwise` | 0.057 | GPT-2 small | +8.1 |
| GPT-2 small | 0.383 | Qwen3-0.6B | +14.3 |
| `1B_baseline` | 0.439 | `1B_baseline` | +18.5 |
| Qwen3-0.6B | 0.523 | `1B_elementwise` | **+3600** |

Sink mass gets four of five roughly right and puts the roster's **worst** model
**first**. Element-wise takes **195×** the ungated baseline's damage and
**1600×** its head-wise sibling's, on a model that differs by one boolean.

`quant/diagnose.py` says where it is not:

| model | weights only | acts per-tensor static | per-tensor dynamic | per-token |
|---|---|---|---|---|
| GPT-2 small | +0.16 | +11.82 | +6.35 | +0.27 |
| Qwen3-0.6B-Base | +0.23 | +14.13 | +9.67 | +0.57 |
| `1B_baseline` | +0.04 | +17.88 | +13.47 | +0.68 |
| `1B_headwise` | +0.008 | **+2.29** | +1.30 | +0.40 |
| `1B_elementwise` | +0.01 | **+3388** | **+3482** | +0.68 |

- **Not the weights** — lossless at 8-bit per-channel everywhere.
- **Not the model** — per-token is nearly free on element-wise too (+0.68, the
  same as the baseline). It is a property of sharing one scale across a tensor.
- **Not calibration clipping** — the *dynamic* arm, which cannot clip, is
  **worse** (+3482 vs +3388). Range coverage is unremarkable (median 1.056,
  max 2.42, against 1.65–1.68 elsewhere).
- **Not the gate.** Leaving `q_proj`, which carries the fused gate, entirely in
  fp16 changes nothing (+3468). Exempting `o_proj` recovers ~10% of a model at
  235× its reference. The obvious mechanism was tested and is dead.

**Mechanism found — see R6.** The previous handoff read those two failed
exemptions as evidence that the damage was *distributed across the network*.
That inference is retracted (**C20**): it is localised to three modules, which
is why searching `q_proj` and `o_proj` found nothing.

### R5 — the bit-width question, which the last handoff called the one open one

`D_sink` under **per-token** scaling — the arm that is the research question:

| model | 8-bit | 6-bit | 4-bit |
|---|---|---|---|
| GPT-2 small | +0.0047 *ZERO* | **+0.0160** [+0.001, +0.030] | +0.0253 *ZERO* ⚠ |
| Qwen3-0.6B-Base | +0.0096 | **+0.0379** [+0.015, +0.061] | +3.7120 ⚠ |
| `1B_baseline` | +0.0203 | +0.0102 *ZERO* | +0.3155 ⚠ |
| `1B_headwise` | +0.0012 *ZERO* | +0.0021 *ZERO* | +0.4581 ⚠ |
| `1B_elementwise` | +0.0009 *ZERO* | **+0.1036** [+0.046, +0.168] | +0.0062 *ZERO* ⚠ |

**The redundancy weakens, unevenly, and the answer is qualified.** From 8 to 6
bits the damage grows 3–4× on GPT-2 and Qwen3 — crossing into
distinguishable-from-zero on GPT-2 — and 115× on element-wise. It does *not*
grow on the controlled pair: baseline falls back across zero, head-wise stays
there. Three of five exclude zero at 6 bits against two of five at 8.

So the redundancy result is **an 8-bit result** with a thinner margin at 6, and
nothing here rescues the mitigation claim — the arms carrying the controlled
comparison are the ones that do not show the effect.

**Per-tensor does not survive to 6 bits at all.** Four of five destroyed (Δppl
+1200 to +128000). **On FineWeb-Edu** the exception is `1B_headwise` at
**+112** — the +0.1% variant, while the +12% variant sits at +127610. On code
that cell is destroyed too and nothing survives (see below). The architectural
fix buys something real and large in the low-bit per-tensor regime; it buys it
in a setting nobody deploys.

**4-bit answers nothing.** Every cell is destroyed under *both* granularities on
*both* corpora, so the apparent per-token `D_sink` there ranks nothing.

#### R5 on the second corpus — C22

Run because C21 established that this project's ordering claims may not travel
and R5 is entirely ordering claims. `python -m analysis.corpora --bitwidth`.

| R5's sentence | web | code | |
|---|---|---|---|
| 8→6 per-token growth, GPT-2 | 3.4× | 3.8× | **holds** |
| 8→6 per-token growth, Qwen3 | 4.0× | 17.1× | **no** |
| 8→6 per-token growth, element-wise | 111× | 13.0× | **no** |
| "does not grow on the controlled pair" | baseline 0.5× | baseline **1.6×** | **half** |
| per-token intervals excluding zero, 8→6 | 2/5 → 3/5 | 3/5 → **3/5** | **no** |
| per-tensor survivors at 6 bits | head-wise, alone | **none** | **no** |
| per-tensor survivors at 4 bits | none | none | **holds** |
| least-damaged per-tensor, 8 and 6 bits | head-wise, clear | head-wise, clear | **holds** |

**The direction travels; the thresholds and growth rates do not.** Head-wise is
the least-damaged model under per-tensor at 8 and 6 bits on both corpora, by a
clear margin every time — R5's substantive claim, intact. The "single exception"
sentence is not: head-wise's 6-bit per-tensor cell sits at **12.4×** its
reference on code against **8.7×** on web, crossing the destroyed line. It is
the **only** destroyed-status flip between the two grids at any width.

§12 predicted exactly this cell and said a stricter threshold would force the
sentence to change while leaving the direction alone. A corpus change did what a
threshold change would have. A result sitting 13% from a threshold is not robust
to anything that moves it 13%.

C22 also fixes a smaller error the same check caught: R5's **115×** element-wise
growth is really **111.4×**. It was computed by dividing 0.1036 by 0.0009 — two
values already rounded to four decimals for display. C19's lesson at small
scale: arithmetic on a table cell is not arithmetic on data.

### R6 — the element-wise catastrophe is one tensor, at layer 0

**This closes the mechanism R4 left open and retracts C20.** Per-tensor and
per-token scaling differ in exactly one thing — whether the scale is shared
across token rows — so the gap between them has to be a property of how
magnitude is distributed across rows. `quant/distributions.py` measures that
directly, quantizing nothing.

The layer-0 MLP input, the tensor feeding both branches of the SwiGLU:

| model | row disp | col disp | uf per-tensor | per-token | per-feature |
|---|---|---|---|---|---|
| Qwen3-0.6B | 1.4× | 10.6× | 0.1181 | 0.0851 | 0.0107 |
| `1B_baseline` | 1.6× | 8.5× | 0.0957 | 0.0681 | 0.0109 |
| `1B_headwise` | 1.6× | 5.9× | 0.0705 | 0.0343 | 0.0113 |
| **`1B_elementwise`** | **28.4×** | **304.0×** | **0.9926** | 0.4620 | 0.0107 |

One boolean separates the last row from the two above it. Under a shared 8-bit
scale that tensor loses **99.26%** of its entries to rounding against the
baseline's 9.6%; under per-row scales it loses 46%, which is survivable and is
why per-token stays at +0.68 on the same checkpoint.

**The axis column is C23 and it corrects this section.** R6 originally reported
only row dispersion and called the failure row-dispersed, reasoning that row
dispersion is what per-token divides out. The gate in fact raises **both** axes
and the feature axis **more** (36–52× against 18×), and the feature axis is the
more dispersed one at every model's worst layer — head-wise, the least damaged
model here, sits at 248.9×. Per-feature underflow is 0.01–0.06 everywhere
regardless of damage. So the feature axis separates nothing and the per-tensor
column is the only one that tracks the damage. Per-token is *sufficient*, which
the grid shows directly, but not because it addresses the dominant axis — it
takes this tensor from 99.3% underflow to 46%, and 46% is survivable where
99.3% is not. `python -m analysis.distributions --axis`.

**The causal test** (`a_only_dynamic`, the arm that cannot clip):

| model | left in fp16 | modules | Δppl | ppl / ppl_ref |
|---|---|---|---|---|
| `1B_elementwise` | nothing *(control)* | 0/196 | +3481.54 | 241× |
| `1B_elementwise` | blocks **8–15** MLP | 24/196 | +3342.39 | 232× |
| `1B_elementwise` | block **0** MLP | **3/196** | **+6.57** | **1.45×** |
| `1B_elementwise` | blocks 0–7 MLP | 24/196 | +1.45 | 1.10× |
| `1B_baseline` | block 0 MLP | 3/196 | +12.96 | 1.88× *(ctrl +13.47)* |
| `1B_headwise` | block 0 MLP | 3/196 | +1.36 | 1.09× *(ctrl +1.30)* |

**Three modules — 1.5% of the quantized layers — take the arm from 241× its
reference to 1.45×, a 530× reduction.** Twenty-four *other* modules do nothing;
the same three on either sibling do nothing. Because the tensor sits at block 0,
all 28 blocks process the wreckage, which is how one tensor makes +3482.

**The falsification control held.** Counted under per-token scales the roster
gives 1, 0, 0, 0, 0 annihilated layers — flat, as it must be, since per-token
damage is nearly free everywhere. A statistic that moved with both would be
measuring the checkpoint, not the granularity.

**Two things that used to sit here are gone.** The cross-model ranking is
retracted (C21, R7): it ordered the roster on FineWeb-Edu and orders nothing on
code. The axis attribution is corrected (C23): the tensor is extreme on both
axes, more so on the feature axis. What survives both is the same thing — the
layer-0 tensor and the intervention with its controls, on two corpora. *Why*
the gate reshapes that tensor in training is still unexplained.
LIMITATIONS §21.

### R7 — the second corpus, and what it retracted

**The previous handoff's priority 1.** `data/code_python.txt` — 1.2M characters
of Python source against educational web prose, every fetch parameter held
identical so domain is the only difference. 8-bit cells re-run on all five
checkpoints. `python -m analysis.corpora` generates all of the below.

Code is a much easier corpus (`ppl_ref` 5.3–11.2 against 14.5–29.6) and
quantizes 3–4× better — on every model except the one that breaks, which
breaks harder.

| claim | web | code | verdict |
|---|---|---|---|
| Matched pair, per-tensor `D_sink` | +0.6450 → +0.0117 (**55×**) | +0.5196 → +0.0113 (**46×**) | **held** |
| Redundancy: max per-token \|`D_sink`\| | 0.0203 nats | **0.0117**, 3/5 estimates negative | **held, stronger** |
| R4: element-wise vs its reference | 257× | **1003×** | **held, worse** |
| R6 localisation: layer-0 dispersion | 28.4× vs 1.6× siblings | 25.9× vs 1.5–2.2× | **held** |
| R6 localisation: 3-module exemption | +3481.54 → +6.57 (530×) | +6929.49 → +2.12 (**3265×**) | **held, stronger** |
| R6 ranking: orders the roster at | 90%, 95%, 99% | **no threshold** | **RETRACTED (C21)** |

Destroyed-status flips between corpora: **none**. Every model's single worst
layer is the **same layer** on both corpora.

**Three of five per-token `D_sink` point estimates go negative on code** —
holding position 0 in fp16 made the model very slightly worse. That is not a
physical quantity; it is the metric at its noise floor. The honest reading is
stronger than "the redundancy held": under per-token scaling on code, `D_sink`
has no measurable sign.

**The pattern is the finding.** Interventions with controls held or
strengthened; the one rank correlation over five models did not. That is a
statement about kinds of evidence, not about this statistic — §10.

### What R3-rev corrected in R3 (unchanged, kept for the trail)

| Claim in R3 | Fate |
|---|---|
| Matched-pair reduction 269× | **corrected** — 55×, R3 inflated ~5× by the bad corpus |
| Per-token indistinguishable from zero for *every* model | **corrected** — excludes zero for Qwen3 and `1B_baseline` |
| The head-wise / element-wise inversion | **held**, and R4 now restates it on a valid metric |
| Per-tensor column ordering | **held** — ranks stable, magnitudes shift 0.7–1.5× |

---

## 4. Repo map

Root is the project directory itself (`.venv` and `.idea` live there — PyCharm
project root, not a subfolder layout).

| File | Lines | State | Holds |
|---|---|---|---|
| `sinks/hooks.py` | 304 | done | Forward hooks reducing to scalars in-hook; the `output_attentions` null-out fix; Pébay online moments |
| `sinks/metrics.py` | 125 | done | Sink mass, head entropy, ∞-norms, outlier channels, excess kurtosis, received attention |
| `sinks/detector.py` | 196 | done | Layer-relative and aggregate detectors, τ sweep, attention validation gate |
| `sinks/measure.py` | 339 | done | Track-A CLI; `resolve_model_path`; the Day-2 gate; corpus provenance |
| `quant/fakequant.py` | 169 | done | Quant/dequant, scale derivation, `scale_source` exclusion, static scale from amax |
| `quant/patch.py` | 361 | done | `QuantLinear`, Conv1D support, exception specs, observation mode, patch/restore |
| `quant/calibrate.py` | 188 | done | Disjoint corpus slicing, batching, BOS policy, static range collection |
| `quant/evaluate.py` | 619 | done | Per-token NLL, `D_sink` decomposition, `evaluate_cell`, `run_grid`, `--grid` CLI, corpus loading |
| `quant/diagnose.py` | 246 | done | Arm decomposition, range-coverage table, projection exemption |
| `quant/distributions.py` | 330 | done | Per-layer dispersion on **both** axes, effective bits, per-granularity underflow. Quantizes nothing (R6, C23) |
| `analysis/aggregate.py` | 211 | done | runs/ → dataframes; reconstructs `D_sink`; the comparability guard |
| `analysis/stats.py` | 165 | done | Paired bootstrap, sequence bootstrap, variance-source reporting |
| `analysis/figures.py` | 251 | done | Figure 1, bit-width figure; zero-crossing hollow, destroyed cells hatched |
| `analysis/report.py` | 192 | done | The README's tables; per-width and per-bit-width views |
| `analysis/distributions.py` | 380 | done | R6 tables joined to the damage they explain; `--sweep` finds the thresholds where the ranking fails, `--axis` reports both dispersion axes (C23) |
| `analysis/corpora.py` | 396 | done | Cross-corpus join; per-claim stability verdicts computed rather than asserted, and `--bitwidth` maps every sentence of §5.5 to a number (R7, C21, C22) |
| `train/attention.py` | 82 | **partial** | `softmax1` and `OutputGate` work; `CausalSelfAttention` stubbed |
| `train/model.py` | 28 | **stub** | nanoGPT-ish decoder — contract only |
| `train/data.py` | 18 | **stub** | 16k BPE + streaming uint16 memmap — contract only |
| `train/train.py` | 26 | **stub** | Resumable trainer — contract only |

### Tests

| File | Guards |
|---|---|
| `test_hooks.py` | Pébay merge = single pass; probability null-out actually happens; structural module discovery |
| `test_detector.py` | Multi-level recovery; per-sequence median; layer-relative vs aggregate failure mode; validation raises |
| `test_calibrate.py` | Draw disjointness; holdout never seen; BOS replaces not extends; exempt entries excluded from range |
| `test_patch.py` | `D_sink` mechanism: per-tensor benefits, per-token provably cannot (atol=0) |
| `test_metrics.py` | Every metric against an analytically-known answer |
| `test_evaluate.py` | Decomposition exactly additive in nats (rel=1e-12) |
| `test_grid.py` | Sweep resumability; provenance cannot shadow a measurement; **the tokenization contract (C19)**; the comparability guard, on the numbers that actually broke it |
| `test_stats.py` | Degenerate-CI failure mode; mismatched seeds rejected |
| `test_fakequant.py` | **HARD GATE** — the no-op quantizer check |
| `test_softmax1.py` | The scaling identity `softmax1(x) = s·softmax(x)` |
| `test_distributions.py` | Dispersion against hand-derived answers; the falsification case (within-row peaking must NOT disperse); the axis pair discriminates a hot row from a hot feature channel and admits a hot *entry* disperses both; `collect` restores the model and leaves no hooks |
| `test_dist_report.py` | `--skip-modules` runs excluded from the damage join (a treatment cannot stand in for what it treats); the sweep actually detects a reordering |
| `test_corpora.py` | The headline reduction refuses a zero-crossing numerator; the ranking check reports "no threshold" when the order inverts, and does not credit an all-zero column |

---

## 5. Environment

| | |
|---|---|
| Python | 3.11.9, venv at `.venv/` |
| torch | 2.13.0+cu126, `cuda.is_available() == True` |
| transformers | 4.57.6 (vendored model code is pinned at 4.46 — see C2 / C14) |
| GPU | RTX 3060 Laptop, sm_86, 6144 MiB total / ~4.3 GiB free with the display attached |
| Disk | HF cache ~12 GB |

**Two install traps that cost real time:**

1. **PyPI `torch` gives a CPU build on Windows.** Everything imports, every test
   passes, and `cuda.is_available()` is silently `False`. The CUDA build must
   come from the PyTorch index explicitly.
2. **pip cannot resume a partial download** and the cu126 wheel is 2.6 GB. On a
   flaky connection it fails repeatedly with nothing to show. Fetch the wheel
   with a resumable transfer and install from disk.

---

## 6. Reproducing from scratch

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# THE HARD GATE. Nothing downstream is valid until this passes. Expect an
# ASYMMETRIC signature, not uniform success:
#   W8 per-channel ~0.0075   A8 per-tensor ~0.235   A4 per-tensor ~0.885
python -m pytest tests/test_fakequant.py -v

python -m sinks.measure --model gpt2_small --calib-seed 0 --seq-len 512 --n-batches 4

# The grid. Loads each checkpoint ONCE and skips finished cells, so it is
# resumable after a crash or a tool timeout.
python -m quant.evaluate --model gpt2_small --grid --bits-list 8,6,4 \
    --seq-len 256 --calib-tokens 2048 --eval-tokens 8192

python -m analysis.aggregate && python -m analysis.figures && python -m analysis.report

# R6: per-layer activation shape, and the tables that join it to the damage.
# Quantizes nothing, so it is cheap -- one holdout pass per checkpoint.
python -m quant.distributions --model gated_1b_elementwise --bits 8
python -m analysis.distributions

# R7: the second-corpus arm. --text-file is the only thing that changes; the
# provenance fields (corpus_sha256, holdout_sha) keep the two grids apart.
python -m data.fetch_code                      # provenance only -- the file is committed
python -m quant.evaluate --model <m> --grid --bits-list 8     --granularities per_tensor,per_token --exceptions none,position_0 --seeds 0     --seq-len 256 --calib-tokens 2048 --eval-tokens 8192     --text-file data/code_python.txt --out runs/quant_code
python -m analysis.corpora
```

The corpus is committed, and both `evaluate` and `measure` default to it. Do not
re-stream FineWeb-Edu to reproduce these numbers: a fresh stream is a different
document set and therefore a different experiment (LIMITATIONS §18).

---

## 7. Corrections ledger

Twenty-three corrections to the original plan, every one established by running code
rather than by rethinking. All annotated in place in
`attention-sinks-quant-plan.md` §0 with the original text preserved.

Severity: **HIGH** = changed a conclusion or blocked work · **MED** = changed
method or budget · **LOW** = confirmed or improved.

| # | Sev | Correction |
|---|---|---|
| C1 | LOW | Day-0 gate passed — three arms, not two. Turns a binary contrast into a dose–response axis |
| C2 | MED | No declared licence; vendored modelling code pinned at transformers 4.46 |
| C3 | HIGH | Qwen3-0.6B-Base has no BOS token. Trap §9.4's premise is false for it |
| C4 | HIGH | `D_sink` conflates a near-tautological self-effect with the contamination effect under audit. Now decomposed |
| C5 | HIGH | The decomposition must be in nats, not perplexity — `exp(mean(·))` is not additive over token subsets |
| C6 | HIGH | The aggregate detector is blind above τ=10. Replaced with a layer-relative form |
| C7 | HIGH | GPT-2 uses `Conv1D`, not `nn.Linear`, weight stored transposed. Neither failure crashes |
| C8 | HIGH | fp16 exceptions must be excluded from scale derivation, not merely pasted back |
| C9 | MED | Per-tensor must be statically calibrated, per exception arm |
| C10 | MED | Hardware measured, not estimated |
| C11 | LOW | Days 0–3 gates all passed, including the Day-3 hard gate |
| C13 | HIGH | The gate is fused into `q_proj` — the element-wise arm is +12% params, so that comparison confounds gating with capacity |
| C14 | HIGH | `trust_remote_code` and `subfolder` do not compose |
| C15 | MED | The "1B" arms are 1.72B parameters |
| C16 | HIGH | No Qwen checkpoint has a BOS token — four of five |
| C17 | HIGH | The Day-2 gate could not tell "no sink" from "broken detector" |
| C18 | HIGH | The calibration draw is not a variance source for the per-token arm — five seeds, std exactly 0 |
| **C19** | **HIGH** | **The repo CLI never reproduced the committed cells.** `quant/evaluate.py` tokenized the corpus line-by-line; the scratchpad driver that produced all 200 cells read it whole-file. Same bytes, same hash, different held-out slice — streams diverge at token 846. Surfaced only as a 0.06–0.76% `ppl_ref` shift |

| **C20** | **MED** | **"Not in the two projections I tested" was written up as "distributed across the network."** The damage is localised to three modules — the layer-0 MLP — and exempting them buys 530×. The measurements were right; the inference was not |

| **C21** | **HIGH** | **R6's cross-model ranking did not survive a second corpus.** Orders the roster at three thresholds on FineWeb-Edu and at **none** on Python source. The causal half of R6 reproduced and strengthened. Five models with one 258× outlier were never enough |

| **C22** | **MED** | **R5's "single exception" was corpus-dependent, and its 115× was a rounding artefact.** Head-wise's 6-bit per-tensor cell survives on FineWeb-Edu (8.7×) and is destroyed on code (12.4×) — the only destroyed-status flip between the two grids. The 8→6 element-wise growth was 111×, not 115×: two 4-decimal display values divided |

| **C23** | **MED** | **R6's mechanism was attributed to the wrong axis.** The element-wise layer-0 tensor is extreme on *both* axes and more so on the feature axis (304× vs 28.4×); the feature axis is the more dispersed one on every model in the roster and separates nothing. The localisation and its controls are unaffected — what is corrected is the explanation around them |

C12 is filed as a *result*, not a correction.

### Why C19 is the one to read

It is the only correction that would have silently corrupted future work rather
than the work that had already been done. The committed cells were fine; every
*new* cell would have been measured on a different held-out slice while
appearing, by corpus hash, to be on the same corpus.

It surfaced because the 6-bit grid was run through the repo CLI and came back
with `ppl_ref` 14.6539 against the 8-bit cells' 14.7077 on the same model — a
**0.37%** difference, systematic across all five checkpoints. Large enough to
notice only if you look; small enough to attribute to the GPU if you don't.

Three guards now exist, and they are ordered from most to least likely to fire:

1. `analysis.aggregate.assert_comparable` refuses to aggregate cells of one
   model whose `ppl_ref` disagree. This is the one that catches the general
   case, including causes nobody has thought of yet.
2. Every cell records `holdout_sha`, a fingerprint of the held-out token
   stream. The corpus hash cannot do this job: the bytes on disk are identical
   either way.
3. `tests/test_grid.py` pins the whole-file contract and the failure mode
   itself, so "why does this read the file this way" has an answer in the
   test suite rather than only in a comment.

**The general lesson:** a number that moves by less than a percent is the
dangerous kind. A 5× discrepancy gets investigated; a 0.4% one gets attributed
to hardware noise.

---

## 8. Limitations

Twenty entries in `LIMITATIONS.md`, which is authoritative. Items 1–6 were
written before any code ran; 7–17 came from experiment; 18–20 from this session.

| # | Limitation | Origin |
|---|---|---|
| 1 | No rotation-based quantization (QuaRot / SpinQuant) | a priori |
| 2 | Scale — Track A tops out at 1.72B; nothing established at 7B+ | a priori |
| 3 | Fake quant ≠ deployed quant. No latency/throughput/memory claims | a priori |
| 4 | Single data distribution for Track B pretraining | a priori |
| 5 | The controlled comparison is one lab's training run | a priori |
| 6 | KV-cache quantization not separated from activation quantization | a priori |
| 7 | All three arms already use QK-Norm — isolates gating *on top of* QK-Norm | experiment |
| 8 | The gated_attention weights carry no declared licence | experiment |
| 9 | Sink measurement depends on vendored modelling code | experiment |
| 10 | The headline metric was refined before data collection (C4, C5) | experiment |
| 11 | The sink detector was changed from the §6 definition (C6) | experiment |
| 12 | Qwen3-0.6B-Base has no BOS arm | experiment |
| 13 | The element-wise arm is not parameter-matched (+12%) | experiment |
| 14 | No Qwen checkpoint has a BOS token — §9.4 unexecutable | experiment |
| 15 | The released arms are 1.72B, not 1B | experiment |
| 16 | `trust_remote_code` + `subfolder` incompatibility | experiment |
| 17 | The Day-2 gate was redesigned mid-project; `D_sink` undefined for sink-free models | experiment |
| 18 | **Every number rests on a single corpus**, and one swap moved the headline ~5× | experiment |
| 19 | **The 4-bit cells are uninterpretable** in both granularities; 6-bit is what answers the bit-width question | experiment |
| 20 | **The held-out slice is 32 sequences**; every CI is a bootstrap over those | experiment |

---

## 9. Data on disk

| Path | Count | Contents |
|---|---|---|
| `data/` | 3 | **Committed corpora with sha256s.** `fineweb_edu.txt` (281 docs) carries every current number; `code_python.txt` (137 docs) is the second-corpus arm; the third is archived |
| `runs/sinks/` | 5 | One per model, draw 0, no-BOS. Per-head sink mass and entropy, per-layer ∞-norms, both detectors × 6 τ with validation |
| `runs/quant/` | 300 | **Current.** FineWeb-Edu grid at 8/6/4-bit |
| `runs/quant_repo_corpus/` | 200 | **Archived.** Same grid on the in-repo corpus. Kept deliberately — R3 vs R3-rev rests on it |
| `runs/diag/` | 14 | Arm decomposition per model, plus the `q_proj` / `o_proj` and layer-0 MLP localisation runs (R6) |
| `runs/dist/` | 5 | Per-layer activation shape at 8 bits, one per checkpoint. What R6 rests on |
| `runs/quant_code/` | 60 | **The second-corpus arm.** 8/6/4-bit, both granularities, `none`/`position_0`, draw 0 |
| `runs/diag_code/` | 9 | Arm decomposition and the layer-0 localisation, re-run on code |
| `runs/dist_code/` | 5 | Activation shape on code. R7's stability check |
| `runs/results/` | 5 | `summary.csv`, `d_sink.csv`, `sinks_summary.csv`, and the two figures |

Committed to git: everything except `runs/quant/`, `runs/quant_repo_corpus/`
and `runs/sinks/` — see `.gitignore`. The figures are committed because a
gitignored figure is a broken image in the README, and `runs/diag/` and
`runs/dist/` because headline claims in §5.4 rest on them.

### Checkpoints in the HF cache (~12 GB)

- `QwQZh/gated_attention` — three arms at ~3.3 GB each. **Load via local
  snapshot path, not repo+subfolder** (C14).
- `Qwen/Qwen3-0.6B-Base` — ~1.2 GB, safetensors, stock transformers.
- `gpt2` — ~0.5 GB.

---

## 10. Traps for whoever continues

### Operational

- **Never run two GPU processes at once.** A background grid surviving past its
  notification window while foreground runs were launched against the same 6 GB
  card cut throughput from ~7 cells per window to ~2. Check for live python
  processes before starting GPU work. No data was corrupted — one-JSON-per-cell
  and `skip_existing` meant the processes duplicated effort rather than
  clobbering each other.
- Tool invocations cap at 10 minutes. The 1.72B models need several resumes per
  grid; `skip_existing` handles it. Prefer `--grid` (one load, many cells) over
  per-cell processes: loading a 3.4 GB checkpoint costs ~1–2 min per resume.

### Methodological

- **The Day-3 hard gate is not optional.** If W8 per-channel *and* A8 per-tensor
  both come out lossless, the quantizer is a no-op and every downstream number
  is rounding error in the harness. The expected signature is asymmetric.
- **Any gate phrased "X must be present" is suspect** when absence of X is a
  possible answer (C17). This nearly suppressed the project's best finding.
- **Check that your randomness source applies to every arm** before quoting a CI
  (C18). A zero-width interval is an absent measurement, not a precise one.
- **Check that a metric is still in its valid range before ranking with it.**
  `D_sink` is a difference between two damaged models; once both are destroyed
  it ranks nothing. This was flagged at 4 bits and missed at 8, where it had
  already invalidated a headline comparison. `is_destroyed` now enforces it.
- **A sub-1% discrepancy is the dangerous kind** (C19). Investigate it.
- **A premise about what a fix *can* address is not a measurement of what the
  data *is*** (C23). R6 argued correctly that per-token scaling can only
  divide out row structure, then described the tensor as row-structured
  without measuring the other axis. The transposed statistic took ten lines
  and contradicted it. When a mechanism is stated as "X is what matters",
  measure not-X before writing the sentence.
- **A result sitting close to a threshold is not robust to anything that moves
  it by that much** (C22). §12 flagged head-wise's 6-bit cell at 8.7× against a
  10× line and predicted a threshold change would break the sentence. A
  *corpus* change broke it instead. When you document a result as
  threshold-fragile, treat it as fragile to everything, not just to the
  threshold.
- **Do arithmetic on the data, not on the rendered table** (C22, C19). R5's
  "115×" was 0.1036/0.0009 read off a 4-decimal table; the real ratio is
  111.4×. If a ratio is worth quoting it is worth computing in the generator.
- **An intervention with controls and a rank correlation are not the same
  claim, and must not share a paragraph** (C21). R6 reported both together;
  the second corpus killed one and strengthened the other. Before writing a
  finding, ask which kind of evidence each sentence rests on — and expect the
  correlational half to be the one that does not travel.
- **A null result is a fact about the search, not about the thing** (C20).
  Two projections were exempted, neither helped, and that was written up as
  "the damage is distributed across the network" — when it sat in three
  modules nobody had looked at. Name the search space whenever you report an
  absence, so the next reader can see what was never checked.
- **τ=2 over-flags** on every model measured, and the attention validation gate
  correctly rejects it. Do not widen the τ grid downward to "find more sinks".
- **`D_sink` is undefined for sink-free models** in the `detected_sinks` arm.
  Use `position_0`. An empty cell is not a missing run.

### Mechanical

- The vendored code emits *"You are using a model of type qwen2 to instantiate a
  model of type qwen3"* on every load. Benign.
- HF cache symlink warnings on Windows are benign, but the first
  `snapshot_download` may fail with `WinError 1314`; retry succeeds.
- Console output needs `PYTHONIOENCODING=utf-8` on Windows or `Δ` and `×` crash
  the print.
- `data/*.txt` are pinned `-text` in `.gitattributes`. Do not remove that:
  autocrlf would rewrite the bytes their sha256s cover.

---

## 11. Next steps, in priority order

### 1. R6 at 6 and 4 bits

**The code corpus at 6 and 4 bits is done** (§3, C22) — that was the previous
priority 1 and it is answered. What is left of the bit-width axis is the R6
statistic itself: `dispersion`, `eff_bits` and both underflow fractions are
computed against a specific integer grid, so every R6 number is an **8-bit**
number on **both** corpora.

```bash
python -m quant.distributions --model <m> --bits 6                       # web
python -m quant.distributions --model <m> --bits 6     --text-file data/code_python.txt --out runs/dist_code                # code
```

The question it answers: is the layer-0 tensor still where element-wise dies at
6 bits, where per-tensor destroys **all five** models on code? The causal test
(`quant.diagnose --skip-modules layers.0.mlp --bits 6`) is the half worth
running — C21 and C22 between them establish that on this project the
interventional half is what travels.

Cheap: distributions quantize nothing, so it is one holdout pass per checkpoint
per corpus.

### 2. LAMBADA, or one downstream task

Promoted from 4. The plan calls for one cheap downstream task and perplexity is
still the only measurement here. It matters more after C21/C22 than it did
before: three of the five things those corrections touched were *orderings of
perplexity damage*, and a second, differently-shaped metric is the cheapest
independent check on whether those orderings mean anything. "Which arm is
broken" is a live question, not a formality.

### 3. The `detected_sinks` and `outlier_channels` arms

`detected_sinks` is one command:
`--grid --exceptions detected_sinks --sinks-json runs/sinks/<model>_calib0_nobos.json`.

**`outlier_channels` is not, and this is a live finding.** It is implemented in
`quant/patch.py` and unit-tested, but the `quant.evaluate` CLI **never builds an
`outlier_mask`** — it wires `sink_mask` from `--sinks-json` and nothing else, so
the arm raises `ValueError` if selected. `sinks/measure.py` imports
`outlier_channels` from `sinks.metrics` at line 49 and never calls it. The arm
has no end-to-end path. Wiring it means recording the outlier mask in the sinks
JSON (where the design clearly intended it) and reading it back in `evaluate`;
that implies re-running `sinks.measure` for all five models, which should
reproduce R1 exactly and must be checked to.

**The prediction that used to be here failed — see C23.** It read: the
element-wise failure is row dispersion, so a channel-wise exception should not
rescue the layer-0 tensor. The transposed statistic contradicted it before any
intervention was run: that tensor is *more* dispersed on the feature axis
(304×) than on the row axis (28.4×), and per-feature scaling would take its
underflow to ~1%.

So the live question is now the opposite one, and it is sharper. Per-feature
underflow is flat and near zero across the **whole roster**, damaged and
undamaged models alike — which is exactly the shape of a statistic that
discriminates nothing. If `outlier_channels` nonetheless rescues element-wise
and only element-wise, something separates that model on the feature axis which
`underflow_col` does not capture, and the axis story is unfinished. If it
rescues everything or nothing, the feature axis is confirmed as uninformative
here and the arm can be reported as a null and closed.

### 4. Track B, or cut it

The plan's own de-scoping section says cut Track B first and ship the
inference-only audit. Track A now has six findings, two corpora and its own
retractions on file, so that option
is live and legitimate. If Track B does happen, `softmax1` is the arm worth the
compute — widely cited, never evaluated with seeds — and the scaling identity is
already implemented and tested.

---

## 12. Open judgment calls

Decisions a reasonable person might make differently.

**The destroyed-cell threshold — the most consequential unforced choice here.**

`analysis.figures.DESTROYED_PPL_RATIO = 10.0`. A cell whose quantized
perplexity exceeds 10× its own unquantized reference is treated as a destroyed
model: `D_sink` on it is a difference between two broken models, so it is
hatched in Figure 2, marked in Figure 1, flagged in every generated table, and
excluded from ratio columns.

It is load-bearing. It is what moved the element-wise 8-bit per-tensor cell out
of the ranking, which is the whole of R4.

Check it rather than trusting it — `python -m analysis.report --threshold-sweep`
prints this table and the sorted ratios behind it:

| threshold | cells flagged (of 30) | changes vs 10× |
|---|---|---|
| 2× | 18 | +baseline 8b tensor, +elementwise 6b token, +headwise 6b tensor |
| 3× | 17 | +elementwise 6b token, +headwise 6b tensor |
| 5× | 17 | +elementwise 6b token, +headwise 6b tensor |
| **10×** | **15** | — |
| 20× | 15 | — |
| 50× | 14 | −gpt2 6b tensor |
| 100× | 12 | −gpt2 6b tensor, −qwen3 6b tensor, −gpt2 4b token |

The ratios are not smoothly distributed; they cluster with wide empty bands, and
that is what makes 10× defensible:

- 13 cells sit at **1.01–2.22** — plainly working models.
- Two sit at **8.70** (`headwise` 6b per-tensor) and **8.86** (`elementwise` 6b
  per-token) — genuinely borderline, and the only cells near the line.
- 15 sit at **42.68 and above**, up to 156807 — plainly destroyed.

10× sits inside the empty band between 8.86 and 42.68, and the whole band
10×–20× behaves identically. That is the argument for it.

**What does not depend on it.** R4. The element-wise 8-bit per-tensor cell is at
**257×** and is flagged at every threshold from 3× to 250×. The finding that
sink mass ranks the roster's worst model first is threshold-independent, and so
is the R4 ordering table, which is built from Δppl directly rather than from
flags.

**What does depend on it, and is stated here because it is not obvious.** The
two borderline cells are both quoted in README §5.5:

- At **5× or lower**, `1B_headwise` 6b per-tensor flags as destroyed, and §5.5's
  "per-tensor does not survive to 6 bits, the single exception being head-wise
  at +112" becomes "no model survives". The *direction* — head-wise being the
  least damaged by two orders of magnitude — is unaffected, but the sentence
  would have to change.
- At **5× or lower**, `1B_elementwise` 6b per-token also flags, which would
  remove its +0.1036 from the R5 table — the largest 6-bit per-token value and
  part of the evidence that the redundancy weakens.
- At **50× or higher**, GPT-2 and Qwen3's 6b per-tensor cells stop flagging, and
  "four of five destroyed at 6-bit per-tensor" shrinks.

An earlier version of this section claimed the flagging was identical at 5× and
at 100×. It is not, on either side; the claim was written from intuition and the
sweep contradicts it. That is the same failure this project audits others for,
which is why the sweep is now a command rather than a sentence.

**The detector definition.** The plan's aggregate detector was replaced with a
layer-relative one because it validates across a 20× wider τ range. Defensible,
but not the only possible criterion. Both are computed and written to every run
JSON, so switching back costs nothing.

**Leading Figure 1 with the matched pair.** Element-wise is the paper's headline
variant and shows a larger sink-mass reduction. Head-wise leads because it is
parameter-matched and therefore attributable. A reader primarily interested in
the published method might want the reverse emphasis — though after §3 the case
for leading with head-wise is much stronger than it was.

**Reporting `position_0` rather than `detected_sinks`.** Forced for the
sink-free arms, chosen for consistency on the others. Every model measured has
exactly one sink at position 0, so the two coincide here — but that would not
hold on a larger Qwen model with multi-level sinks, and the multi-level
machinery is consequently **untested against a model that actually has them**.

**How hard to push the element-wise finding.** It is one arm of one lab's
training run, and it is confounded with +12% parameters (LIMITATIONS §13). What
the data carries is that the standard metric ranks it first and the damage ranks
it last. Resist upgrading that to a claim about gating in general.
