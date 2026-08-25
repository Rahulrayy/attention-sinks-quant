# Handoff — Attention Sinks and Quantization: An Audit

**Date:** 21 August 2026
**State:** Track A complete and has an answer. Track B not started.
**Authoritative sources:** `attention-sinks-quant-plan.md` §0 (corrections log), `LIMITATIONS.md`, `README.md`, `runs/results/`

---

## 1. Status at a glance

| | |
|---|---|
| Tests | **120 passing**, 9 files |
| Track A code (`sinks/`, `quant/`, `analysis/`) | **complete** — 0 stubs |
| Track B code (`train/`) | **~15%** — 9 stubs |
| Quantization grid | **200 cells** (+200 archived on the old corpus) |
| Sink measurement runs | **5** — one per checkpoint, draw 0, no-BOS |
| Checkpoints on disk | **~12 GB** HF cache, 5 models |
| Corrections to the original plan | **18**, all annotated in place |
| Limitations recorded | **17** (+2 identified but not yet written — see §8) |

### Done

- **Sink measurement** on all five checkpoints — sink mass, head entropy, residual ∞-norms, outlier channels, kurtosis, multi-level detector with τ sweep and attention validation.
- **Fake-quant harness** — per-tensor / per-token / per-channel, 8/6/4-bit, static calibration, fp16 exception lists, four exception arms.
- **Full quantization grid** — 5 models × {8,4} bits × {per_tensor, per_token} × {none, position_0} × 5 calibration draws, on FineWeb-Edu.
- **Analysis** — long-format aggregation, paired + sequence bootstrap, Figure 1 and a bit-width figure, CSV summaries.
- **Documentation** — plan annotated with 18 corrections, `LIMITATIONS.md` at 17 entries, README leading with the finding.

### Not done

- **Track B entirely.** `train/attention.py` has working `softmax1` and `OutputGate`; `CausalSelfAttention`, `model.py`, `data.py`, `train.py` are stubs. No pretraining run has happened.
- **LAMBADA.** The plan calls for one cheap downstream task. Only perplexity is measured.
- **The `detected_sinks` and `outlier_channels` exception arms.** Implemented and unit-tested, never run in the grid.
- **6-bit.** Configured in `quant.yaml`, never run — and it is the width that would settle the one genuinely open question.
- **Multiple calibration draws in the figures.** All five are on disk; the figure uses draw 0. See C18 for why that matters less than it looks.

---

## 2. The question

Three claims circulate in this literature. The first two are settled and are *not* relitigated: a handful of token positions absorb disproportionate attention mass (the sink), and those positions carry extreme residual-stream outliers. The third is under audit:

> That these outliers are what breaks low-bit activation quantization, and that mitigating them architecturally buys measurable robustness.

That literature was built against **per-tensor** activation quantization and **pre-QK-Norm** architectures. Both have moved on. The question is whether the architectural fix still pays against per-token activation scaling on models that already use QK-Norm — with confidence intervals, which almost nothing in the prior work reports.

A negative result was always acceptable. It came back partly negative, which is the interesting case.

---

## 3. Findings

Labelled R1 / R2 / R3-rev in the plan. **R3 exists but is marked SUPERSEDED** — measured on a bad corpus, two of its claims did not survive. Its numbers are kept in place for the audit trail.

### R1 — the gated checkpoints are genuinely sink-free

| Model | Mean sink mass | Heads > 0.5 | Entropy L0 → Lₙ | Magnitude | Sink-free |
|---|---|---|---|---|---|
| GPT-2 small (no QK-Norm) | 0.340 | 0.243 | 3.03 → 3.36 | 87.8× | no |
| Qwen3-0.6B-Base | 0.473 | 0.475 | 2.24 → 0.89 | 1198× | no |
| `1B_baseline` (control) | 0.385 | 0.362 | 2.71 → 1.85 | 352× | no |
| `1B_headwise` (+0.1% params) | **0.054** | **0.002** | 3.01 → 2.91 | **3.8×** | **yes** |
| `1B_elementwise` (+12% params) | 0.021 | 0.000 | 2.92 → 1.42 | 4.4× | **yes** |

Magnitude = max layer-relative residual ∞-norm. One draw, no-BOS, seq 256.

The head-wise arm is what matters: at **+0.1% parameters** it collapses the sink by 91× in residual magnitude, so sink elimination is **not a capacity effect**. Element-wise goes further but carries +12% parameters and that increment is confounded (C13).

### R3-rev — the audit answer, on FineWeb-Edu

`D_sink` at 8-bit activations, nats, 95% sequence-bootstrap CI. *ZERO* = interval contains zero.

| Model | Sink? | per-tensor (2023) | per-token (modern) | Ratio |
|---|---|---|---|---|
| GPT-2 small | yes | **+0.2097** [+0.188, +0.232] | +0.0047 [−0.002, +0.011] *ZERO* | 45× |
| Qwen3-0.6B-Base | yes | **+0.4378** [+0.396, +0.479] | +0.0096 [+0.003, +0.016] | 46× |
| `1B_baseline` | yes | **+0.6450** [+0.601, +0.690] | +0.0203 [+0.014, +0.026] | 32× |
| `1B_headwise` | **no** | +0.0117 [−0.0003, +0.024] *ZERO* | +0.0012 [−0.003, +0.005] *ZERO* | — |
| `1B_elementwise` | **no** | +0.1702 [+0.104, +0.236] | +0.0009 [−0.006, +0.007] *ZERO* | 183× |

200 cells. FineWeb-Edu, 281 docs / 1.2 MB, stubs filtered. Held-out slice 8192 tokens = 32 sequences at seq 256.

**The answer has two halves and neither works alone.**

*The architectural fix works.* On the matched pair — baseline vs head-wise, +0.1% parameters — per-tensor `D_sink` falls from +0.6450 to +0.0117, a **55× reduction** with the CI crossing zero. First check of this claim on a controlled architectural comparison rather than across models differing in several ways at once.

*And it is redundant.* Per-token scaling reduces sink-attributable damage by **32–46×** on sink-bearing models, to a level that is zero or negligible — without touching the architecture. A paper reporting only the first column would be correct and misleading; only the second would conclude the mitigation does not work.

### The finding I did not expect: sink mass ranks two models backwards

Element-wise carries **more** sink-attributable damage than head-wise (+0.1702 [+0.104, +0.236] vs +0.0117 [−0.0003, +0.024], non-overlapping, a 14.5× gap) despite having **less** sink mass (0.021 vs 0.054). The orderings invert.

Entropy explains it: head-wise keeps attention diffuse across depth (3.01 → 2.91) while element-wise still concentrates (2.92 → 1.42), close to the baseline's 2.71 → 1.85. Element-wise strips attention mass off position 0 but keeps concentration structure that per-tensor quantization trips over.

**Sink mass is therefore not a sufficient predictor of quantization sensitivity.** Anything selecting or evaluating an architecture on sink mass alone — most of this literature — uses a proxy that can rank two models the wrong way round.

This survived the corpus swap that broke two neighbouring claims, which is the main reason to trust it. It only surfaced because the metric plan carried entropy *alongside* sink mass, and because the release shipped two gate variants. Neither was foresight about this effect.

### What R3-rev corrected in R3

| Claim in R3 | Fate | Corrected |
|---|---|---|
| Matched-pair reduction 269× | **corrected** | 55× — R3 inflated ~5× by the bad corpus |
| Per-token indistinguishable from zero *for every model* | **corrected** | False. Excludes zero for Qwen3-0.6B and `1B_baseline` |
| The head-wise / element-wise inversion | **held** | 14.5× gap, non-overlapping, on both corpora |
| Per-tensor column ordering | **held** | Ranks stable; magnitudes shift 0.7–1.5× |

`ppl_ref` fell 136.6 → 29.7 (GPT-2) and 54.9 → 17.9 (Qwen3) on the corpus swap — the old in-repo corpus was not a language-model evaluation set.

---

## 4. Repo map

Root is the project directory itself (`.venv` and `.idea` live there — PyCharm project root, not a subfolder layout). **Not a git repository yet** — `git init` has never been run.

| File | Lines | State | Holds |
|---|---|---|---|
| `sinks/hooks.py` | 304 | done | Forward hooks reducing to scalars in-hook; the `output_attentions` null-out fix; Pébay online moments |
| `sinks/metrics.py` | 125 | done | Sink mass, head entropy, ∞-norms, outlier channels, excess kurtosis, received attention |
| `sinks/detector.py` | 196 | done | Layer-relative and aggregate detectors, τ sweep, attention validation gate |
| `sinks/measure.py` | 325 | done | Track-A CLI; `resolve_model_path`; the Day-2 gate; writes `runs/sinks/*.json` |
| `quant/fakequant.py` | 169 | done | Quant/dequant, scale derivation, `scale_source` exclusion, static scale from amax |
| `quant/patch.py` | 361 | done | `QuantLinear`, Conv1D support, exception specs, observation mode, patch/restore |
| `quant/calibrate.py` | 188 | done | Disjoint corpus slicing, batching, BOS policy, static range collection |
| `quant/evaluate.py` | 502 | done | Per-token NLL, `D_sink` decomposition, `evaluate_cell`, `run_grid`, CLI |
| `analysis/aggregate.py` | 183 | done | runs/ → dataframes; reconstructs `D_sink` by joining cells |
| `analysis/stats.py` | 165 | done | Paired bootstrap, sequence bootstrap, variance-source reporting |
| `analysis/figures.py` | 164 | done | Figure 1, bit-width figure; zero-crossing intervals drawn hollow |
| `train/attention.py` | 82 | **partial** | `softmax1` and `OutputGate` work; `CausalSelfAttention` stubbed |
| `train/model.py` | 28 | **stub** | nanoGPT-ish decoder — contract only |
| `train/data.py` | 18 | **stub** | 16k BPE + streaming uint16 memmap — contract only |
| `train/train.py` | 26 | **stub** | Resumable trainer — contract only |

### Tests

| File | Tests | Guards |
|---|---|---|
| `test_hooks.py` | 18 | Pébay merge = single pass; probability null-out actually happens; structural module discovery |
| `test_detector.py` | 17 | Multi-level recovery; per-sequence median; layer-relative vs aggregate failure mode; validation raises |
| `test_calibrate.py` | 16 | Draw disjointness; holdout never seen; BOS replaces not extends; exempt entries excluded from range |
| `test_patch.py` | 16 | `D_sink` mechanism: per-tensor benefits, per-token provably cannot (atol=0) |
| `test_metrics.py` | 15 | Every metric against an analytically-known answer |
| `test_evaluate.py` | 14 | Decomposition exactly additive in nats (rel=1e-12) |
| `test_stats.py` | 12 | Degenerate-CI failure mode; mismatched seeds rejected |
| `test_fakequant.py` | 7 | **HARD GATE** — the no-op quantizer check |
| `test_softmax1.py` | 5 | The scaling identity `softmax1(x) = s·softmax(x)` |

---

## 5. Environment

| | |
|---|---|
| Python | 3.11.9, venv at `.venv/` |
| torch | 2.13.0+cu126, `cuda.is_available() == True` |
| transformers | 4.57.6 (vendored model code is pinned at 4.46 — see C2 / C14) |
| GPU | RTX 3060 Laptop, sm_86, driver 595.71, 6144 MiB total / **~5.01 GiB free** |
| Disk | ~144 GB free; HF cache ~12 GB |

**Two install traps that cost real time:**

1. **PyPI `torch` gives a CPU build on Windows.** Everything imports, every test passes, and `cuda.is_available()` is silently `False`. The CUDA build must come from the PyTorch index explicitly.
2. **pip cannot resume a partial download** and the cu126 wheel is 2.6 GB. On a flaky connection it fails repeatedly with nothing to show. Fetch the wheel with a resumable transfer and install from disk.

---

## 6. Reproducing from scratch

```bash
# 1. torch, CUDA build — do NOT use plain `pip install torch` on Windows
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. THE HARD GATE. Nothing downstream is valid until this passes.
#    Expect an ASYMMETRIC signature, not uniform success:
#      W8 per-channel ~0.0075   A8 per-tensor ~0.235   A4 per-tensor ~0.885
python -m pytest tests/test_fakequant.py -v

# 3. Sink measurement — writes runs/sinks/<model>_calib0_nobos.json
python -m sinks.measure --model gpt2_small --calib-seed 0 \
    --seq-len 512 --n-batches 4 --text-file <corpus.txt>

# 4. Quantization grid. Use run_grid() — it loads the model ONCE.
#    The Makefile spawns a process per cell, reloading 3.4 GB each time.
#    skip_existing=True makes the whole grid resumable after a crash.

# 5. Aggregate + figures
python -m analysis.aggregate
python -m analysis.figures
```

The grid driver used for the current results (`run_grid.py`, `run_all.py`) lives in the session scratchpad and is **not in the repo**. Folding it into `quant/evaluate.py` as a `--grid` flag is small, worthwhile tidying.

---

## 7. Corrections ledger

Eighteen corrections to the original plan, every one established by running code rather than by rethinking. All annotated in place in `attention-sinks-quant-plan.md` §0 with the original text preserved.

Severity: **HIGH** = changed a conclusion or blocked work · **MED** = changed method or budget · **LOW** = confirmed or improved.

| # | Sev | Correction | Where |
|---|---|---|---|
| C1 | LOW | **Day-0 gate passed — three arms, not two.** All Qwen3-arch, identical config but one boolean. Turns a binary contrast into a dose–response axis. | §3 |
| C2 | MED | **No declared licence; vendored modelling code.** GitHub is MIT; the HF repo has no licence field. Each arm vendors `modeling_qwen3.py` pinned at transformers 4.46, ships `pytorch_model.bin`, requires `trust_remote_code`. | §3, §4 |
| C3 | HIGH | **Qwen3-0.6B-Base has no BOS token.** `bos_token_id = None`. Trap §9.4's premise is false for it. | §4 |
| C4 | HIGH | **`D_sink` conflates two effects** — a near-tautological self-effect plus the contamination effect actually under audit. Now decomposed. | §6 |
| C5 | HIGH | **The decomposition must be in nats, not perplexity.** `exp(mean(·))` is not additive over token subsets; the first draft returned a contamination share of 1.106. | §6 |
| C6 | HIGH | **The aggregate detector is blind above τ=10.** Maxing over layers before the median compresses ratios (GPT-2 14.2×, Qwen3 19.8×). Replaced with a layer-relative form (1198× on Qwen3); τ grid extended to {2,5,10,20,50,100}. | §6, §8 |
| C7 | HIGH | **GPT-2 uses `Conv1D`, not `nn.Linear`**, weight stored transposed. A patcher matching only `nn.Linear` leaves GPT-2 unquantized; per-channel quant on a transposed weight gives one scale per *input* channel. Neither crashes. | §9 |
| C8 | HIGH | **fp16 exceptions must be excluded from scale derivation**, not merely pasted back. Otherwise `D_sink` reads ≈0 under per-tensor — a false negative that looks like a finding. | §9 |
| C9 | MED | **Per-tensor must be statically calibrated, per exception arm.** Deriving the range per evaluation batch hands per-tensor adaptivity a deployed scheme lacks. | §9 |
| C10 | MED | **Hardware measured, not estimated.** ~1 GiB of 6 GiB VRAM drives the display; PyPI torch defaults to CPU. | §10 |
| C11 | LOW | **Days 0–3 gates all passed**, including the Day-3 hard gate with the correct asymmetric signature. | §11 |
| C13 | HIGH | **The gate is fused into `q_proj` — arms are not parameter-matched.** Element-wise +117.4M (≈+12%); head-wise +0.9M (+0.1%). Baseline-vs-element-wise confounds gating with capacity. Cross-checked against the paper's own 201M / 1.6M figures. | §3, §4 |
| C14 | HIGH | **`trust_remote_code` and `subfolder` do not compose.** `auto_map` resolves against the repo root, ignoring subfolder. `from_pretrained(repo, subfolder=…)` fails outright. Worked around in `resolve_model_path`. | §4, §7 |
| C15 | MED | **The "1B" arms are 1.72B parameters.** 3.3 GB each on disk, ~3.44 GB VRAM in bf16 against ~5 GiB free. | §3, §4, §10 |
| C16 | HIGH | **No Qwen checkpoint has a BOS token** — four of five. Trap §9.4's cross-model BOS protocol is unexecutable as written. All models run no-BOS, so comparisons stay matched; the BOS question is out of scope rather than answered. | §9 |
| C17 | HIGH | **The Day-2 gate could not tell "no sink" from "broken detector".** It aborted on the head-wise arm — a model that genuinely has no sink. A gate phrased "X must be present" fails when absence of X is a legitimate answer. Now discriminates using attention. | §6, §11 |
| C18 | HIGH | **The calibration draw is not a variance source for the per-token arm.** Dynamic scaling never reads the calibration set, so five "seeds" give byte-identical results (std = 0.000000) and a CI of width exactly zero — on the arm that *is* the research question. CIs now bootstrap over held-out sequences. | §6 |

C12 is filed as a *result*, not a correction — an early preview on GPT-2, superseded by R2 and R3-rev.

---

## 8. Limitations

Seventeen entries in `LIMITATIONS.md`, which is authoritative. Items 1–6 were written before any code ran; 7–17 came from experiment.

| # | Limitation | Origin |
|---|---|---|
| 1 | No rotation-based quantization (QuaRot / SpinQuant) | a priori |
| 2 | Scale — Track A tops out at 1.72B; nothing established at 7B+ | a priori |
| 3 | Fake quant ≠ deployed quant. No latency/throughput/memory claims | a priori |
| 4 | Single data distribution for Track B pretraining | a priori |
| 5 | The controlled comparison is one lab's training run (a triple, not a pair) | a priori |
| 6 | KV-cache quantization not separated from activation quantization | a priori |
| 7 | All three arms already use QK-Norm — isolates gating *on top of* QK-Norm | experiment |
| 8 | The gated_attention weights carry no declared licence | experiment |
| 9 | Sink measurement depends on vendored modelling code pinned below installed transformers | experiment |
| 10 | The headline metric was refined before data collection (C4, C5) | experiment |
| 11 | The sink detector was changed from the §6 definition (C6) | experiment |
| 12 | Qwen3-0.6B-Base has no BOS arm | experiment |
| 13 | The element-wise arm is not parameter-matched (+12%) | experiment |
| 14 | No Qwen checkpoint has a BOS token — §9.4 unexecutable | experiment |
| 15 | The released arms are 1.72B, not 1B | experiment |
| 16 | `trust_remote_code` + `subfolder` incompatibility | experiment |
| 17 | The Day-2 gate was redesigned mid-project; `D_sink` undefined for sink-free models | experiment |

**Not yet written into `LIMITATIONS.md` — add these:**

- **(18)** The results rest on a **single corpus**. FineWeb-Edu is a better one, not a sample of many. The corpus swap changed the matched-pair reduction by 5× and flipped two CIs across zero, so this sensitivity is large and unbounded.
- **(19)** The **4-bit cells are uninterpretable** — Δppl in the millions, the model is destroyed, and perplexity differences between two destroyed models mean nothing. The bit-width question is open, not answered.

---

## 9. Data on disk

| Path | Count | Contents |
|---|---|---|
| `runs/sinks/` | 5 | One per model, draw 0, no-BOS. Per-head sink mass and entropy, per-layer ∞-norms, both detectors × 6 τ with validation. ~700 KB each |
| `runs/quant/` | 200 | **Current.** FineWeb-Edu grid |
| `runs/quant_repo_corpus/` | 200 | **Archived.** Same grid on the in-repo corpus. Kept deliberately — R3 vs R3-rev rests on it |
| `runs/results/` | 5 | `summary.csv` (200 cells), `d_sink.csv` (100 rows), `sinks_summary.csv`, `fig1_d_sink.png`, `fig2_bitwidth.png` |

### Checkpoints in the HF cache (~12 GB)

- `QwQZh/gated_attention` — three arms at ~3.3 GB each. **Load via local snapshot path, not repo+subfolder** (C14).
- `Qwen/Qwen3-0.6B-Base` — ~1.2 GB, safetensors, stock transformers, no `trust_remote_code`.
- `gpt2` — ~0.5 GB.

The FineWeb-Edu corpus (1.2 MB, 281 docs) lives in the session scratchpad as `fineweb.txt` and **is not in the repo**. Regenerating is a ~1 minute stream but the exact document set will differ — commit the file or a hash if the numbers need to be exactly reproducible.

---

## 10. Traps for whoever continues

### Operational — cost ~40 minutes

**Never run two GPU processes at once.** A background grid task survived past its notification window while foreground runs were launched against the same 6 GB card. Throughput fell from ~7 cells per window to ~2. Check for live python processes before starting GPU work. No data was corrupted — one-JSON-per-cell and `skip_existing` meant the processes duplicated effort rather than clobbering each other.

### Methodological

- **The Day-3 hard gate is not optional.** If W8 per-channel *and* A8 per-tensor both come out lossless, the quantizer is a no-op and every downstream number is rounding error in the harness. The expected signature is asymmetric.
- **Any gate phrased "X must be present" is suspect** when absence of X is a possible answer (C17). This nearly suppressed the project's best positive finding.
- **Check that your randomness source applies to every arm** before quoting a CI (C18). A zero-width interval is an absent measurement, not a precise one.
- **τ=2 over-flags** on every model measured, and the attention validation gate correctly rejects it. Those extra positions are magnitude outliers no head attends to. Do not widen the τ grid downward to "find more sinks".
- **`D_sink` is undefined for sink-free models** in the `detected_sinks` arm. Use `position_0`, which is always constructible. An empty cell is not a missing run.

### Mechanical

- Tool invocations cap at 10 minutes. The 1.72B models need several resumes per grid; `skip_existing` handles it.
- Loading a 3.4 GB checkpoint costs ~1–2 min of every resume window. Prefer `run_grid` (one load, many cells) over per-cell processes.
- The vendored code emits *"You are using a model of type qwen2 to instantiate a model of type qwen3"* on every load. Benign.
- HF cache symlink warnings on Windows are benign, but the first `snapshot_download` may fail with `WinError 1314`; retry succeeds.
- Console output needs `PYTHONIOENCODING=utf-8` on Windows or `Δ` and `×` crash the print.

---

## 11. Next steps, in priority order

### 1. Run 6-bit, or W4A8

The single genuinely open empirical question. 4-bit is uninterpretable (models destroyed); 8-bit shows the effect fully absorbed by per-token scaling. A width in between would establish whether the redundancy is universal or holds only where quantization is gentle. `configs/quant.yaml` already lists `bits: [8, 6, 4]` — only 8 and 4 have been run.

### 2. A second corpus

The corpus swap changed the matched-pair reduction by 5× and flipped two CIs across zero. That is a large sensitivity and one corpus cannot bound it. A second distribution — C4, code, or a domain set — would say whether the FineWeb numbers are the stable ones.

### 3. The `detected_sinks` and `outlier_channels` arms

Both implemented and unit-tested but never run in the grid. `outlier_channels` tests a different mechanism (channel-wise, LLM.int8-style) and would say whether the redundancy finding generalises past token-wise exceptions.

### 4. LAMBADA

The plan calls for one cheap downstream task. Perplexity alone cannot distinguish "slightly worse everywhere" from "broken on the cases that matter".

### 5. Track B, or cut it

The plan's own de-scoping section says cut Track B first and ship the inference-only audit. Track A now has a real, defensible finding, so that option is live and legitimate. If Track B does happen, `softmax1` is the arm worth the compute — widely cited, never evaluated with seeds — and the scaling identity is already implemented and tested.

### 6. Tidying

- `git init` — this has never been a git repository.
- Fold the scratchpad grid driver into `quant/evaluate.py` as a `--grid` flag.
- Add limitations 18 and 19.
- Commit the corpus or its hash for exact reproducibility.

---

## 12. Open judgment calls

Decisions a reasonable person might make differently. Each is recorded in the plan or `LIMITATIONS.md`, but these are where the project is least settled.

**The detector definition.** The plan's aggregate detector was replaced with a layer-relative one because it validates across a 20× wider τ range. Defensible, but not the only possible criterion. The layer-relative form is **not** claimed correct in any absolute sense — only better-conditioned on the checkpoints measured. Both are computed and written to every run JSON, so switching back costs nothing.

**Leading Figure 1 with the matched pair.** Element-wise is the paper's headline variant and shows a larger sink-mass reduction. Head-wise leads because it is parameter-matched and therefore attributable. A reader primarily interested in the published method might want the reverse emphasis.

**Reporting `position_0` rather than `detected_sinks`.** Forced for the sink-free arms, chosen for consistency on the others. Every model measured has exactly one sink at position 0, so the two coincide in practice here — but that would not hold on a larger Qwen model with multi-level sinks, and the multi-level machinery is consequently **untested against a model that actually has them**.

**The inversion's interpretation.** The claim made is that sink mass is *not a sufficient predictor* of quantization sensitivity. The stronger claim — that sink mass is the wrong metric — is not supported by two arms of one lab's training run. What the data carries is an existence proof that the standard metric ranks two real models backwards. Resist upgrading it.
