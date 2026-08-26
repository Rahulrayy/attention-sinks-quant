# Attention Sinks and Quantization: An Audit

**Project plan — portfolio project, ~3–4 weeks part-time**
Last updated: 20 August 2026

---

## 0. Corrections log

Everything below marked **✎ CORRECTION** was established by running code, not by
rethinking the plan. Original text is left in place throughout — struck through
or explicitly quoted — so the audit trail stays legible. That is the same
standard §6 imposes on metric changes, applied to the plan itself.

| # | Section | What changed | Severity |
|---|---|---|---|
| C1 | §3 | Day-0 gate **passed**. Three arms released, not two. All Qwen3-arch, `use_qk_norm: true`, gate is a config flag (size: see C15) | Improves the design |
| C2 | §3, §4 | Weights carry **no declared licence**; vendored `modeling_qwen3.py` pinned at transformers 4.46.0, needs `trust_remote_code` | Open risk |
| C3 | §4 | `Qwen3-0.6B-Base` has **`bos_token_id = None`** — trap §9.4's premise is false, it has no BOS arm | Kills an axis |
| C4 | §6 | `D_sink` conflates a near-tautological self-effect with the contamination effect actually under audit. Now decomposed | Metric change |
| C5 | §6 | The decomposition must be computed in **nats, not perplexity** — `exp(mean(·))` is not additive over token subsets | Metric change |
| C6 | §6, §8 | The magnitude detector's aggregate form is **blind above τ=10** on both models measured. Replaced by a layer-relative form; τ grid extended | Metric change |
| C7 | §9 | New trap: **GPT-2 uses `Conv1D`, not `nn.Linear`**, with a transposed weight | Would have been silent |
| C8 | §9 | New trap: fp16 exceptions must be excluded from **scale derivation**, not just pasted back | Would have faked a null |
| C9 | §9 | New trap: per-tensor activation quant must be **statically calibrated**, per exception arm | Flatters the audited arm |
| C10 | §10 | ~1 GiB of the 6 GiB VRAM drives the display; HF cache is ~7.5 GB not ~5 GB; PyPI `torch` defaults to a **CPU build** | Budget |
| C11 | §11 | Days 0–3 gates all **passed**, including the Day-3 hard gate | Status |
| C13 | §3, §4 | The gate is **fused into `q_proj`**. The element-wise arm carries **+117M params (≈+12%)**, so that comparison confounds gating with capacity. Head-wise is the matched pair | **Undercuts §3's core claim** |
| C14 | §4, §7 | **`trust_remote_code` and `subfolder` do not compose** — the gated arms cannot be loaded via `from_pretrained(repo, subfolder=…)` at all | Blocks loading |
| C15 | §3, §4, §10 | The "1B" arms are **1.72B parameters**. Disk for the trio is **~10 GB, not ~6 GB** | Budget |
| C18 | §6 | **The calibration draw is not a variance source for the per-token arm.** Five "seeds" give byte-identical results (std = 0.000000) because dynamic scaling never reads the calibration set. CIs now bootstrap over held-out sequences | **Breaks the stats plan** |
| C17 | §6, §11 | The Day-2 gate **cannot tell a broken detector from a genuinely sink-free model** — it aborted on the gated arm, turning the project's best positive finding into a crash | **Would have hidden a result** |
| C16 | §9 | **No Qwen checkpoint in the roster has a BOS token.** Trap §9.4's cross-model BOS protocol is unexecutable as written | Kills an axis |
| C19 | §6, §7 | **The repo CLI never reproduced the committed cells.** `quant/evaluate.py` tokenized the corpus line-by-line while the scratchpad driver that produced all 200 cells read it as one string. Same bytes, same tokenizer, different held-out slice — streams diverge at token 846. Surfaced as a 0.06–0.76% `ppl_ref` shift that looked like noise | **Silently incomparable cells** |
| C20 | §7 | **"Not in the projections I tested" was written up as "distributed across the network."** Two failed localisation attempts (`q_proj`, `o_proj`) were read as evidence of a diffuse cause. The damage is in fact localised to **three modules** — the layer-0 MLP — and exempting them recovers a 530× improvement | **Retracted an inference, not a measurement** |
| C21 | §6, §7 | **R6's cross-model ranking did not survive a second corpus.** "Annihilated-layer count orders the roster by per-tensor damage" holds on FineWeb-Edu at three thresholds and on Python source at **none**. Five models with one 258× outlier were never enough to tell a mechanism from a lucky sort. The *causal* half of R6 reproduced and strengthened | **Retracts half a finding** |

Findings that are *results* rather than corrections are marked **✎ RESULT**:
[C12](#c12) (early preview on GPT-2), [R1](#r1) (the gated trio is sink-free),
[R2](#r2) (Figure 1), [R3](#r3) (first full grid, **superseded**) and
**[R3-rev](#r3rev) — the audit answer on FineWeb-Edu, where two R3 claims
were corrected and the inversion finding held**.

---

## 1. What I'm actually asking

The literature says three things that everyone now repeats:

1. A handful of tokens (usually position 0) soak up disproportionate attention mass across most heads and layers — the **attention sink**.
2. Those same token positions carry **massive activations**: extreme outliers in a few residual-stream channels.
3. These outliers are what breaks low-bit **activation quantization**, and mitigating them — architecturally or at inference time — buys you robustness.

Claims 1 and 2 are well established and I'm not going to relitigate them. Claim 3 is where I want to poke, because the mitigation literature was mostly built against per-tensor activation quantization and pre-QK-Norm architectures. Both of those have moved on.

**The question this project answers:**

> Does architectural sink-mitigation still buy measurable quantization robustness once you compare it against a properly-configured quantization baseline — per-token activation scaling, on models that already use QK-Norm — with confidence intervals across seeds?

That's a replication-and-audit question in the Ferrari Dacrema / Jannach mould, transplanted from recsys to quantization. It has a real chance of coming back **negative** — "the architectural fix is largely redundant under modern quantization practice" — and that's a perfectly good result. Much of this literature reports single-run numbers with no seed variance at all, so even a clean confirmation with error bars is worth having.

**What this project is not:** it is not "I discovered attention sinks." It is not a new mitigation method. Framing it as an audit from the first line of the README is what makes the crowded prior work a resource instead of a problem.

---

## 2. Prior work — what's already taken

Read these before writing code. Several of them cover things I might otherwise think were my idea.

### Core phenomenon

| Work | What it established | Relevance |
|---|---|---|
| Xiao et al., **StreamingLLM** (ICLR 2024, arXiv 2309.17453) | Named the attention sink; showed keeping the first few tokens rescues sliding-window generation | Origin of the concept |
| Sun et al., **Massive Activations in LLMs** (ICML 2024, arXiv 2402.17762) | Extreme residual-stream outliers at specific tokens; they *cause* the attention concentration | Defines my outlier metric |
| Gu et al., **When Attention Sink Emerges** (ICLR 2025, arXiv 2410.10781) | Training-dynamics view: when and why the sink forms; distinguishes token-wise from channel-wise outliers | Owns the "emergence" angle — I'm not competing here |
| **The Spike, the Sparse and the Sink** (arXiv 2603.05498, Mar 2026) | Argues sinks are a product of pre-norm architecture + short-context training, not inherent. Traces a Rise–Plateau–Fall trajectory across layers in Llama-2 and Qwen3; identifies late "step-down" blocks that inject an equal-and-opposite spike to cancel the outlier before the head. Notes QK-Norm as a preventative that eases 4/8-bit quant | **Biggest overlap with my Days 1–2.** Read first. My measurement phase is a replication of theirs, and I should say so |
| **Attention Sink in Transformers: A Survey** (arXiv 2604.10098, Apr 2026) | Full survey of utilization / interpretation / mitigation. Also covers the MoE case (Super Experts producing the outliers) | Use as the map of occupied territory |

### Mitigation

| Work | Method | Relevance |
|---|---|---|
| Bondarenko et al., **Quantizable Transformers** (NeurIPS 2023) | Clipped softmax, gated attention; "helping attention heads do nothing". Evaluated on BERT/OPT | Direct ancestor. My extension is modern decoder arch + error bars |
| Miller, **Attention Is Off By One** (blog, 2023) | softmax1 / off-by-one softmax | Widely cited, **never properly evaluated with seeds**. This is the one arm worth my own compute |
| Darcet et al., **ViTs Need Registers** (ICLR 2024) | Dedicated register tokens absorb the artifact | Vision-side analogue; cut from scope |
| Qiu et al., **Gated Attention for LLMs: Non-linearity, Sparsity, and Attention-Sink-Free** (NeurIPS 2025 Oral, arXiv 2505.06708) | Query-dependent sigmoid gate after SDPA output (their "G1"). Demonstrably mitigates massive activations and sinks, improves long-context extrapolation. **Code + checkpoints released.** Shipped in Qwen3-Next | **The single most important paper for this project.** See §3 |

### Quantization side

| Work | Method | Why it matters to me |
|---|---|---|
| KVQuant (Hooper et al., 2024) | Retains the first token in fp16 | **My "keep position 0 in fp16" attribution knob is already a published method.** It's a control, not a contribution |
| CushionCache / Prefixing Attention Sinks (Son et al., EMNLP 2024) | Inserts sink tokens into the prefix to absorb mid-sequence massive activations. Reports **three sink levels in QwQ-32B and six in Qwen3-14B** | Forces a redesign of my sink detector — see §7 |
| RotateKV (arXiv 2501.16383) | Uses massive activations to *find* sinks without needing attention scores, since FlashAttention hides them | Solves my `output_attentions` memory problem too |
| QuaRot / SpinQuant family | Rotation-based outlier suppression | **Not in scope.** Declared as a limitation up front — see §11 |

### Honest positioning

After all that, what's left that I can defend:

- **Reproducibility.** Almost nothing above reports seed variance or confidence intervals. That gap is real and it's the one I'm best placed to fill.
- **The redundancy question.** Nobody has asked whether architectural sink-mitigation still pays once activation quant is per-token rather than per-tensor.
- **Matched-scale cross-architecture comparison.** The gated-attention checkpoints make one controlled comparison possible that didn't exist before.

---

## 3. Why Qwen changes the experiment

The Qwen team's gated attention work matters here for two separate reasons.

**First, I don't have to implement gated attention.** The reference implementation (`github.com/qiuzh20/gated_attention`) is built on the Qwen3 architecture, with checkpoints on HF (`QwQZh/gated_attention`). Reimplementing it myself is now a source of bugs, not a contribution.

**Second — and this is the real payoff — it gives me a natural experiment.** A standard pre-norm model with sinks versus a deliberately sink-free one, trained by the same lab, on the same data pipeline, with the same tokenizer, differing in one architectural component. That's about as close to a controlled architectural comparison as anything available in public weights.

The question then sharpens to: *the mechanism is demonstrably suppressed — does the promised downstream quantization benefit actually materialize, at matched quality, with error bars?*

> **✅ RESOLVED 2026-08-20 — see correction C1 immediately below.** Original text kept for the record:
>
> **⚠ OPEN ITEM — verify before planning around this.**
> I have not confirmed what the public `gated_attention` checkpoints actually are: parameter count, and critically whether **both** the gated and ungated arms are released. If only the gated model is up, the controlled pair collapses and I'd have to fall back to comparing Qwen3-0.6B against a Qwen3-Next-family model — which differ in size, training data, *and* layer topology simultaneously. That's a much weaker comparison and would need reframing. **Ten minutes on the HF repo before anything else.**

> ### ✎ CORRECTION C1 — 2026-08-20 · the Day-0 gate passed, and the design got better
>
> **`QwQZh/gated_attention` ships three arms, not two:**
>
> | Folder | `elementwise_attn_output_gate` | `headwise_attn_output_gate` |
> |---|---|---|
> | `1B_baseline` | false | false |
> | `1B_gate_headwise` | false | **true** |
> | `1B_gate_elementwise` | **true** | false |
>
> All three are Qwen3 architecture with **identical** config otherwise: hidden
> 2048, 28 layers, 16 Q heads / 8 KV heads, vocab 152064, tied embeddings,
> bf16, `use_qk_norm: true`. They share one modelling file and **differ by a
> single boolean in `config.json`**.
>
> That is a tighter natural experiment than this section assumed. The binary
> gated-vs-ungated contrast becomes a **dose–response axis**: no gate → coarse
> (head-wise) gate → fine (element-wise) gate. Element-wise is G1 as published
> and the paper's best variant.
>
> Two consequences the plan did not anticipate:
>
> 1. **Scale is 1B**, not the sub-1B this section implicitly assumed. Three
>    checkpoints at ~2 GB each — see C10 for the disk revision.
> 2. **All three arms already use QK-Norm.** The triple therefore isolates
>    *gating on top of QK-Norm*, which is exactly the right test for this
>    project's question — but it cannot say whether gating and QK-Norm are
>    substitutes for one another. GPT-2 remains the only checkpoint without
>    QK-Norm, and it differs in four other ways at once. Recorded as
>    `LIMITATIONS.md` §7.

> ### ✎ CORRECTION C13 — 2026-08-20 · the gate is fused into `q_proj`, and the arms are **not** parameter-matched
>
> Read from the vendored `modeling_qwen3.py`. The gate is not a separate module
> hanging off the attention block — it is **extra output width on `q_proj`**,
> sliced off and passed through a sigmoid:
>
> ```python
> if   self.headwise_attn_output_gate:     q_proj = Linear(H, nh*hd + nh)
> elif self.elementwise_attn_output_gate:  q_proj = Linear(H, nh*hd * 2)
> else:                                    q_proj = Linear(H, nh*hd)
> ...
> attn_output = attn_output * torch.sigmoid(gate_score)   # G1, on the SDPA output
> ```
>
> Computed from the three configs (H=2048, L=28, 16 heads, head_dim 128):
>
> | arm | `q_proj` out | total `q_proj` params | Δ vs baseline |
> |---|---|---|---|
> | `1B_baseline` | 2048 | 117.4M | — |
> | `1B_gate_headwise` | 2064 | 118.4M | **+0.9M** (+0.1%) |
> | `1B_gate_elementwise` | 4096 | 234.9M | **+117.4M (≈ +12%)** |
>
> **Cross-check against the paper.** Qiu et al. report the element-wise G1 gate
> adding **201M** parameters on MoE-15A2B against **1.6M** for head-wise — the
> same ratio found here. So the capacity difference is known to the authors and
> is a property of the method, not a packaging error in the release. What needs
> correcting is *this plan's* framing of the arms as a clean controlled pair.
>
> **This partly undercuts §3's central claim.** That section calls the pair "as
> close to a controlled architectural comparison as anything available in public
> weights" — differing in *one architectural component*. True, but the
> element-wise variant's component costs ~12% more parameters on a ~1B model, so
> baseline-vs-elementwise confounds **gating with capacity**. Any quantization
> difference between them could be a wider `q_proj` rather than the gate.
>
> **Consequence — the arm ordering inverts.** §4 above frames head-wise as the
> "middle of the dose–response axis" and element-wise as the interesting one
> because it is the paper's best variant. For an *audit*, the priority is
> reversed:
>
> - **`baseline` vs `headwise` is the primary controlled comparison.** +0.1%
>   parameters is genuinely matched, so a difference is attributable to the gate.
> - **`baseline` vs `elementwise` is the headline-variant comparison**, and must
>   be reported with the capacity confound stated, not as a clean contrast.
>
> Keep all three: the dose–response reading survives, and the two comparisons
> answer different questions. But Figure 1 should lead with the matched pair.
>
> Secondary note for the quantization side: the gated arms' `q_proj` output
> carries two quantities of potentially different scale (query vectors and gate
> logits) in one matrix. Per-channel weight quantization gives each output
> channel its own scale so this is handled, but per-*tensor* activation stats on
> anything downstream of that split should not be assumed comparable across arms.

> ### ✎ CORRECTION C14 — 2026-08-20 · `trust_remote_code` + `subfolder` do not compose
>
> The obvious call **fails outright**:
>
> ```
> from_pretrained("QwQZh/gated_attention", subfolder="1B_baseline",
>                 trust_remote_code=True)
> → OSError: does not appear to have a file named configuration_qwen3.py
> ```
>
> A config's `auto_map` names bare module files (`configuration_qwen3.Qwen3Config`)
> and transformers resolves them against the **repo root**, ignoring `subfolder`.
> This repo vendors a separate copy of the modelling code inside each arm and
> puts nothing at the root, so the remote code is unreachable by that route.
>
> Fix, now in `sinks.measure.resolve_model_path`: `snapshot_download` the arm's
> subfolder, then load from that **local directory**, where the vendored `.py`
> files sit beside `config.json`. Confirmed working — the model loads, selects
> `Qwen3Attention` under `attn_implementation="eager"`, and measures end-to-end.
>
> Benign but alarming warning on every load, from `model_type: "qwen2"` in a
> config whose vendored class is `Qwen3Config`:
> *"You are using a model of type qwen2 to instantiate a model of type qwen3."*

> ### ✎ CORRECTION C15 — 2026-08-20 · the "1B" arms are 1.72B parameters
>
> Measured after loading: **1720.8M parameters**, not ~1B. The folder prefix is
> the release's own naming and does not match the parameter count. Composition:
> ~311M tied embeddings (152064 × 2048) + ~1.3B across 28 layers.
>
> Downstream corrections:
> - **Disk**: one arm is 3.3 GB on disk. The trio is **~10.1 GB**, plus
>   Qwen3-0.6B (1.2 GB) and GPT-2 (0.5 GB) — call it **~12 GB**, against the
>   ~7.5 GB estimated in C10 and the ~5 GB in §10. Still inside the 30 GB
>   budget, but it is now the dominant term.
> - **VRAM**: 1.72B in bf16 is ~3.44 GB of weights against 5.01 GiB free
>   (C10). It fits — one arm at a time, B=1 — but the headroom is ~1.5 GB, not
>   the comfortable margin §10's table implies for a "1B" model.

> ### ✎ CORRECTION C2 — 2026-08-20 · two practical risks in those checkpoints
>
> **No declared licence.** The GitHub implementation is MIT. The HF repo
> `QwQZh/gated_attention` has **no licence field and no licence tag**. Absent a
> declaration, reuse terms are unclear. This repo redistributes no weights and
> references the upstream path only. Resolving it upstream is an open action
> item. (`LIMITATIONS.md` §8.)
>
> **Vendored modelling code.** Each folder ships its own `modeling_qwen3.py`
> and `configuration_qwen3.py` pinned at `transformers==4.46.0`, and
> `pytorch_model.bin` rather than safetensors. `trust_remote_code=True` is
> required. The environment currently runs transformers **4.57.6**, so the
> vendored code and the installed library do not agree on version. This is
> untested and is the single largest remaining unknown in Track A.
> (`LIMITATIONS.md` §9.)

---

## 4. Model roster

Four checkpoints for Track A. Each has one job; nothing is in the list "for completeness."

| # | Model | Architecture notes | Licence | Job |
|---|---|---|---|---|
| 1 | `gated_attention` **ungated** baseline | Qwen3-arch, standard softmax attention | Check repo (expect Apache 2.0) | Control arm of the pair |
| 2 | `gated_attention` **G1-gated** | Same, + sigmoid gate on SDPA output | Check repo | Treatment arm — the only true controlled comparison in the project |
| 3 | **Qwen3-0.6B-Base** | RoPE, RMSNorm, **QK-Norm**, GQA, SwiGLU, 151k vocab, tied embeddings | Apache 2.0 | Does the pattern hold on a real production model |
| 4 | **GPT-2 small** (124M) | Pre-norm, learned positions, **no QK-Norm**, MHA | MIT | Old-architecture contrast — makes #3 mean something |

> ### ✎ CORRECTION — 2026-08-20 · the roster as actually configured
>
> Supersedes the table above. Live version is `configs/models.yaml`.
>
> | # | Model | Architecture | Licence | Job |
> |---|---|---|---|---|
> | 1 | `1B_baseline` | Qwen3, 28L, 16Q/8KV, QK-Norm, **no gate** | **undeclared** (C2) | Control arm |
> | 2 | `1B_gate_headwise` | ↑ + head-wise gate | **undeclared** | Treatment, coarse — middle of the dose–response axis |
> | 3 | `1B_gate_elementwise` | ↑ + element-wise gate (G1 as published) | **undeclared** | Treatment, fine — the paper's best variant |
> | 4 | **Qwen3-0.6B-Base** | Qwen3, 28L, 16Q/8KV, QK-Norm | Apache 2.0 | External validity: a *differently trained* pipeline, **not** "does the pattern hold" — arm 1 already answers that for Qwen3-arch |
> | 5 | **GPT-2 small** (124M) | Pre-norm, learned positions, **no QK-Norm**, MHA, `Conv1D` | MIT | The only no-QK-Norm checkpoint — see trap C7 |
>
> **C3 — `Qwen3-0.6B-Base` has `bos_token_id = None`.** Verified against the
> tokenizer. Trap §9.4 states "GPT-2 doesn't prepend BOS; Qwen and Llama do".
> That is **false** for this checkpoint: it has no BOS token, so it has no BOS
> arm — the axis is absent, not merely unbalanced. Substituting `eos` would
> change what occupies position 0, which is the precise confound §9.4 exists to
> prevent, so the code refuses rather than silently substituting. Cross-model
> comparisons involving this checkpoint must read against the other models'
> no-BOS arms only. (`LIMITATIONS.md` §12.)
>
> ### ✎ CORRECTION C16 — 2026-08-20 · no Qwen checkpoint here has a BOS token
>
> C3 found `bos_token_id = None` on Qwen3-0.6B-Base. Verified on the gated arms
> too: the tokenizer reports `bos_token_id = None`, `eos_token_id = 151645`.
> (The *config* carries `bos_token_id: 151643`, but the tokenizer — which is
> what builds the batches — does not.)
>
> So **four of the five checkpoints have no BOS arm**. Only GPT-2 does. Trap
> §9.4's protocol — "run every model both with and without an explicit BOS, or
> the cross-model comparison is meaningless" — is therefore **unexecutable as
> written**: there is no BOS arm to compare against for any Qwen model.
>
> What survives, and it is enough: every model runs no-BOS, so all cross-model
> comparisons are made under a *matched* policy. What is lost is the ability to
> ask whether BOS presence is what drives the position-0 sink. GPT-2 alone can
> answer that within-model, and one checkpoint cannot separate a BOS effect from
> an architecture effect. This should be stated in the README rather than left
> as an unrun row in a table.

> **GQA reporting confirmed.** `configs/models.yaml` commits to
> `per_query_head`. Verified: HF materialises probabilities *after* `repeat_kv`,
> so hooks record 16 heads matching `num_attention_heads`. The commitment holds
> and needs no adjustment.

**Dropped, deliberately:**

- *Pythia* — its 154 intermediate checkpoints only matter for the emergence-during-training question, which the Anatomy paper now owns.
- *TinyLlama* — adds a third tokenizer and no new architectural variable.
- *Llama-2-7B scale appendix* — gated download, restrictive licence, needs a Kaggle T4, and only answers "does it also happen at 7B" (answer: yes, published repeatedly).
- *Qwen3.5 / 3.6 / 3.8* — Qwen3.5 carries Qwen3-Next's 3:1 hybrid layout (its 397B-A17B has 45 Gated DeltaNet layers to 15 attention layers), and the small dense ones are natively multimodal. "Attention sink" isn't well-defined inside a linear-attention layer, and vision tokens confound residual-stream measurement. **Qwen3 dense is the last clean generation for this question.**

### Licensing summary

- Qwen3 dense (0.6B–32B): **Apache 2.0**, all sizes, no gating or signup. Same for Qwen3.5/3.6/3.8-27B; only top-end Max variants use a bespoke licence.
- GPT-2: **MIT**, all sizes, since the staged 2019 release.
- DeepSeek V4: **MIT**, but the smallest variant is 284B total / 13B active. Unusable on consumer hardware. Older DeepSeek releases (V3 base, Coder-V2, VL2) split permissive code from a more restrictive OpenRAIL-derived weights licence — check per repo.
- For a public academic repo, Qwen weights are unremarkable and standard in this literature. Some employers have internal policy on model provenance; irrelevant to the repo itself.

Training-data licences are a separate question from weights licences. For my own pretraining runs: FineWeb-Edu (ODC-By) and TinyStories (CDLA-Sharing) are both clean. Don't redistribute Pile shards.

---

## 5. Two tracks, and why the split is non-negotiable

### Track A — pretrained models, inference only

Measure and attribute. Legal interventions: KV eviction policies, keeping/dropping specific token positions, mixed-precision quantization, prefix insertion.

### Track B — tiny models I pretrain myself

The **only** place where softmax1, clipped softmax, or gating can be tested honestly.

**Why they cannot be mixed:** you cannot retrofit softmax1 onto pretrained weights.

```
softmax1(x)_i = exp(x_i) / (1 + Σ_j exp(x_j))
```

Let `s = Σexp / (1 + Σexp) < 1`. Then `softmax1(x) = s · softmax(x)`. The output isn't "the same attention with less sink" — it's the original convex combination uniformly scaled down. Of course the model degrades, and that tells you nothing about the mechanism. Several blog posts got this wrong; it belongs in my README as a methodological note.

**Track B scope (deliberately small):** three arms × five seeds = 15 runs.

| Arm | What it tests |
|---|---|
| `baseline` | Standard softmax attention, naive path |
| `softmax1` | Off-by-one softmax — the most-cited untested fix |
| `gated` | G1 sigmoid gate on SDPA output, ported from the reference repo |

Cut: register tokens, clipped softmax. Three arms is enough to make a point; five is enough to make a grid I can't finish.

---

## 6. Metrics — defined before anything runs

Locking these down before seeing data is half the point of the project.

### Sink measurement

- **Sink mass** `S(ℓ,h)` = mean over query positions `q > 0` of attention weight assigned to position 0, in layer ℓ head h. Report the full per-head distribution, plus the fraction of heads with `S > 0.5`. Never report only the mean — the distribution is bimodal and the mean hides it.
- **Head entropy** `H(ℓ,h)` = mean Shannon entropy of the attention distribution over queries. Separates genuine sinking from merely diffuse attention.
- **GQA note:** Qwen3 uses grouped-query attention (16 Q heads, 8 KV heads in 0.6B). Decide up front whether sink mass is reported per query head or per KV group, state it, and stay consistent.

### Massive-activation measurement

- Per-token residual-stream ∞-norm at layer ℓ: `m(t,ℓ) = max_c |x⁽ℓ⁾_{t,c}|`
- Aggregate: `M(t) = max_ℓ m(t,ℓ)`
- **Outlier channel:** channel c is flagged if `max_t |x_{t,c}| > 100 × median_c(max_t |x_{t,c}|)`
- **Per-layer activation kurtosis** — Bondarenko's outlier proxy, included so numbers are directly comparable to that paper.

### Sink detector (the redesign)

The naive `position_0` rule is **insufficient for Qwen models**, which show multiple sink levels (three reported in QwQ-32B, six in Qwen3-14B). Replace it with a magnitude-based detector:

```
sink(t) = True  iff  M(t) > τ · median_t(M(t))
```

- Sweep `τ ∈ {10, 20, 50, 100}` and report sensitivity. Do not hard-code one value.
- **Validate the detector against attention:** confirm that flagged tokens do receive elevated attention mass. If they don't, the detector is wrong and the whole attribution chain is broken.
- Bonus: this detector needs only residual-stream norms, no attention probabilities — which also sidesteps the `output_attentions` memory blowup.

> ### ✎ CORRECTION C6 — 2026-08-20 · the aggregate detector is blind, and the τ grid was miscalibrated
>
> The definition above maxes over layers **before** the median is taken, so the
> denominator is set by whichever layer has the largest typical activation.
> That compresses every ratio toward 1. Measured:
>
> | Model | sink ratio, aggregate | sink ratio, layer-relative |
> |---|---|---|
> | GPT-2 small | **14.2×** | — (single sink) |
> | Qwen3-0.6B-Base | **19.8×** | **1198×** |
>
> Against the original τ grid the aggregate detector flagged **nothing at all**
> above τ=10 — on GPT-2, where the sink is unmistakable in the attention maps
> (mean sink mass 0.67 by layer 10, *every* head above 0.5). A grid whose lowest
> rung is 10 would have reported "no sinks detected" for the checkpoint that
> shows the effect most clearly.
>
> Two changes, in order:
>
> 1. **τ grid extended** to `{2, 5, 10, 20, 50, 100}`. Necessary but not
>    sufficient — it left only a narrow τ ∈ {5,10} window where the detector
>    both fired and agreed with attention.
> 2. **Detector replaced** with a layer-relative form: normalise each layer's
>    norms by that layer's own median, *then* max over layers, and compare
>    against τ directly (the input is already a ratio).
>
> ```
>   sink(t) = True  iff  max_ℓ [ m(t,ℓ) / median_t m(t,ℓ) ] > τ
> ```
>
> On Qwen3-0.6B-Base this validates across the **full τ range 5→100** where the
> aggregate form works only at 5–10. Width of the stable range is the criterion:
> a detector that agrees with attention only inside a narrow band of τ has been
> *fitted* to one model, not validated.
>
> Both detectors are computed and written to every run JSON, with
> `primary_detector` naming the layer-relative one — so a change to a §6-locked
> metric stays visible in the output instead of vanishing into a refactor.
> (`LIMITATIONS.md` §11.)
>
> **Not claimed:** that the layer-relative form is correct in any absolute
> sense. It is better-conditioned on the two checkpoints measured so far.
>
> **Still untested:** multi-level sinks. Qwen3-0.6B-Base showed exactly one
> sink, at position 0, under *both* detectors. CushionCache's 3-in-QwQ-32B and
> 6-in-Qwen3-14B are from far larger models, and nothing in this roster is that
> size — so the multi-level machinery may go unexercised for the whole project.
>
> **The validation gate is doing real work.** At τ=2 both detectors flag extra
> positions (65, 218 on Qwen; 30, 31 seen elsewhere) and both then **fail**
> attention validation. Those are magnitude outliers that no head attends to.
> Without the gate they would enter the fp16 exception list and inflate
> `D_sink` with tokens that have nothing to do with the sink.

### Quantization damage

- **Δppl** on a held-out slice (fixed, never touched during calibration).
- One cheap downstream task — LAMBADA. Nothing that takes more than ~10 minutes per config.
- **Headline metric — sink-attributable damage:**

```
D_sink = Δppl(everything quantized) − Δppl(sink tokens kept fp16)
```

Reported per model × per quantization granularity, with bootstrap CIs. This single number is Figure 1.

> ### ✎ CORRECTION C4 — 2026-08-20 · `D_sink` sums two effects that mean different things
>
> As defined it adds together:
>
> - **(a) SELF** — the sink token's own prediction stops being corrupted. Close
>   to tautological: holding *any* token in fp16 removes that token's own
>   quantization error, under *every* granularity.
> - **(b) CONTAMINATION** — every *other* token gets a tighter shared scale
>   because the outlier no longer drags the range out. **This is the effect the
>   mitigation literature actually claims**, and the one that should collapse
>   under per-token scaling.
>
> Caught on a linear toy model with no cross-token mixing, where (b) is
> structurally impossible: per-token scaling still showed **~99% of per-tensor's
> `D_sink` at 4 bits** — all of it effect (a). Reporting only the summed metric
> would have made per-token look nearly as sink-sensitive as per-tensor and
> pointed the audit at the wrong conclusion.
>
> `D_sink` is now reported **decomposed**, with the non-sink restriction as the
> headline and the total retained for comparability with this plan.
>
> **A subtlety that cuts the other way.** A causal LM scores the prediction of
> token *t* from context ending at *t−1*, so position 0 is never a prediction
> target. When the only sink is at position 0 — the case for both models
> measured so far — effect (a) is **empty by construction** and every nat of
> `D_sink` is contamination. Measured `contamination_share` = 1.000 throughout,
> exactly as that reasoning predicts. The decomposition only starts to earn its
> keep on multi-level sinks at positions 5, 17, … which this roster may never
> reach (see C6).

> ### ✎ CORRECTION C5 — 2026-08-20 · the decomposition must be in nats, not perplexity
>
> Perplexity is `exp(mean(nll))`, which is **not additive over token subsets**:
> damage measured on all tokens is not a weighted average of damage on disjoint
> subsets of them. Attributing a *share* of a perplexity delta to a group of
> tokens is therefore ill-defined, and overshoots in practice — the first draft
> of the decomposition returned a contamination share of **1.106**.
>
> Mean NLL is additive by construction, so the split is computed there and
> perplexity is reported alongside for comparability with the prior literature.
> The count-weighted parts now reconstruct the total exactly, and
> `tests/test_evaluate.py` asserts that identity to `rel=1e-12`.
>
> Consequence for the grid: each cell needs to store only six aggregate floats
> (mean NLL over all / sink / non-sink columns, plus counts) rather than
> per-token arrays. The reference term cancels in `D_sink` — both damages are
> measured against the same reference — so `analysis/` reconstructs the metric
> by joining cells. 262k floats × 120 cells would have been ~600 MB of JSON.

### Statistics

- 5 seeds minimum for Track B; for Track A the "seed" is the calibration set draw (5 disjoint calibration sets).
- **Paired bootstrap** over per-seed deltas, never unpaired comparison of means. Seed variance at 17M params is ugly and unpaired tests will drown the effect.
- Report CI width alongside every point estimate. If the CI crosses zero, say so in the text, not just the figure.

---

## 7. Repo layout

```
attention-sinks-quant/
├── README.md                  # leads with the finding + Fig 1, not the architecture
├── LIMITATIONS.md             # written early, not as an afterthought
├── Makefile                   # make measure / quant / train / figs / repro
├── requirements.txt
├── configs/
│   ├── models.yaml            # the 4 Track-A checkpoints
│   ├── quant.yaml             # bit widths × granularity × fp16 exceptions
│   ├── detector.yaml          # τ sweep for the sink detector
│   └── train/
│       ├── baseline.yaml
│       ├── softmax1.yaml
│       └── gated.yaml
├── sinks/
│   ├── hooks.py               # streaming stats in-hook; NEVER stores tensors
│   ├── metrics.py             # sink mass, entropy, ∞-norm, kurtosis, outlier channels
│   ├── detector.py            # magnitude-based multi-level sink detection + validation
│   └── measure.py             # → runs/sinks/<model>_<calib>.json
├── data/                      # COMMITTED corpora + hashes (added 2026-08-25, C19)
│   ├── fineweb_edu.txt        # the 281 docs every current number was measured on
│   └── repo_corpus_archived.txt
├── quant/
│   ├── fakequant.py           # per-tensor | per-token | per-channel, n-bit
│   ├── patch.py               # wraps nn.Linear; toggles W/A quant; fp16 exception list
│   ├── calibrate.py           # activation range collection
│   ├── diagnose.py            # splits a cell into W / A / static / dynamic / per-token
│   ├── distributions.py       # per-layer row dispersion + underflow → runs/dist/ (R6)
│   └── evaluate.py            # ppl → runs/quant/<config>.json; --grid walks the sweep
├── train/
│   ├── attention.py           # ALL arm-specific code lives here, nowhere else
│   ├── model.py               # nanoGPT-ish
│   ├── data.py                # streaming tokenization → uint16 memmap
│   └── train.py               # resumable, checkpoints every 500 steps
├── analysis/
│   ├── aggregate.py           # runs/**/*.json → one long-format dataframe
│   ├── stats.py               # paired bootstrap CIs
│   ├── report.py              # the README's result tables, generated not typed
│   ├── distributions.py       # the R6 tables, joined to the damage they explain
│   └── figures.py             # every figure in the README
├── tests/
│   ├── test_fakequant.py      # ← see §9, this one is load-bearing
│   ├── test_grid.py           # sweep resumability + the tokenization contract (C19)
│   └── test_softmax1.py       # verifies the scaling identity from §5
└── runs/                      # gitignored except results/, diag/ and dist/
```

**Design rules:**

1. Every script writes one JSON per run and nothing else. A crashed sweep never costs a figure.
2. All analysis reads from `runs/`. `analysis/` never touches a model.
3. All arm-specific logic in `train/attention.py`. If arm code leaks into `model.py`, the ablation is no longer clean.
4. No Hydra, no W&B, no experiment tracker. A YAML plus argparse overrides is enough at this scale, and it keeps the repo readable for someone skimming it in two minutes.

> ### ✎ CORRECTION C19 — 2026-08-25 · the repo CLI never reproduced the committed cells
>
> Design rule 1 says every script writes one JSON per run. It does not say the
> script that ran the sweep has to be *in the repo* — and for the whole of Track
> A, it wasn't. The 200 committed cells were produced by a `run_grid.py` living
> in a session scratchpad; `quant/evaluate.py`'s own CLI was never the thing that
> generated a result.
>
> They did not agree. The driver read the corpus as **one string**; the CLI read
> it **line by line**, which moves BPE boundaries at every newline and drops the
> blank lines between documents. Same file, same hash, same tokenizer,
> **different held-out slice** — the streams diverge at token 846.
>
> This surfaced only because the 6-bit grid was run through the CLI and its
> `ppl_ref` came back at 14.6539 against the 8-bit cells' 14.7077 on the same
> model. A **0.37%** shift, systematic across all five checkpoints, and small
> enough to read as noise. Had the two paths differed by less, or had nothing
> new been run through the CLI, the mismatch would have stayed invisible while
> making every future cell quietly incomparable to the committed ones.
>
> Fixed by `quant.evaluate.load_corpus`, which reads whole-file — the path the
> committed cells were measured on — used by both `evaluate` and `diagnose`, and
> pinned in `tests/test_grid.py`. Every cell now records `holdout_sha`, a
> fingerprint of the held-out token stream: the corpus hash cannot catch a
> changed reader, because the bytes on disk are identical either way. The 6-bit
> grid was re-run under the corrected reader.
>
> The general lesson is the one this project keeps relearning: **a number that
> moves by less than a percent is the dangerous kind.** A 5× discrepancy gets
> investigated; a 0.4% one gets attributed to the GPU.

> ### ✎ CORRECTION C20 — 2026-08-26 · two failed searches were written up as an absence
>
> R4 established that the element-wise arm is destroyed by per-tensor 8-bit
> activation quantization, and tried to localise the cause. Two exemptions were
> run: `q_proj`, which carries the fused gate, and `o_proj`, which consumes the
> gated output. Neither rescued the model. README §5.4 concluded that the damage
> was therefore **"distributed across the network, not concentrated in the gate
> path"**, and HANDOFF §3 carried the same sentence.
>
> The measurements were right. The inference was not. Two failed localisation
> attempts establish that the damage is not in *those two projections* — they
> say nothing whatever about whether it is localised somewhere else, and both
> candidates had been chosen from one hypothesis about the gate. When that
> hypothesis died, its search space was mistaken for the whole space.
>
> It is localised, and severely. `quant/distributions.py` measures per-layer row
> dispersion — the one property per-tensor and per-token scaling differ over —
> and flags the layer-0 MLP input at **28.4×** against **1.6×** on both sibling
> checkpoints, with 99.26% of its entries rounding to zero under a shared 8-bit
> scale. Holding that one MLP in fp16 — **three modules of 196** — moves the arm
> from +3481.54 to +6.57, a **530×** reduction, while the same exemption on the
> two siblings does nothing and exempting eight *other* blocks' MLPs (24 modules)
> changes almost nothing. Full writeup in README §5.4.
>
> This one cost nothing but a wrong sentence in two documents, because the
> conclusion it guarded — that the fragility is real, activation-side, and not a
> calibration artifact — never depended on it. The lesson is narrower and worth
> stating anyway: **"I looked in two places and it wasn't there" is a fact about
> the search, not about the thing.** The write-up rule that follows is to name
> the search space whenever a null result is reported, so that the next reader
> can see what was never looked at.

> ### ✎ CORRECTION C21 — 2026-08-26 · the ranking half of R6 was a property of one corpus
>
> R6 made two claims of different kinds and the write-up presented them together.
> One was an intervention with controls: holding the layer-0 MLP in fp16 rescues
> the element-wise arm, exempting other blocks does not, and the same exemption
> on either sibling does nothing. The other was a rank correlation: across five
> models, the count of annihilated layers orders the roster by per-tensor damage.
>
> LIMITATIONS §21 flagged the second as weak at the time it was written — five
> points, one of them 258× more damaged than the next and therefore supplying
> most of the apparent correlation. That was the right worry and it was not
> enough, because the claim still went into §5.4 alongside the causal result.
>
> `data/code_python.txt` settled it. On FineWeb-Edu the count reproduces the
> damage ordering at 90%, 95% and 99%. On Python source it reproduces it at **no
> threshold at all**: head-wise picks up annihilated layers (1 → 3) while
> remaining the least damaged model by a factor of five, and GPT-2 and Qwen3
> swap. `python -m analysis.corpora` prints both rows side by side.
>
> The localisation went the other way and got stronger — 25.9× row dispersion at
> the layer-0 tensor against 1.5–2.2× on both siblings, and a **3265×** damage
> reduction from exempting three modules against 530× on web text.
>
> The lesson is about kinds of evidence rather than about this statistic. **An
> intervention with controls on one model and a rank correlation over five
> models are not the same claim and should not share a paragraph.** The first
> survived a domain shift; the second was never tested against one until now.
> Related: C20, where a null result over two projections was written up as a
> statement about the whole network. Both are the same error — letting the shape
> of the evidence be flattened by the confidence of the prose.

---

## 8. Configs

`configs/quant.yaml`:

```yaml
bits: [8, 6, 4]

weight_granularity: [per_channel]          # control: ~lossless at 8b
act_granularity:
  - per_tensor                             # where the classic effect lives
  - per_token                              # the modern baseline — MUST be run

fp16_exceptions:
  - none                                   # full damage
  - position_0                             # the KVQuant-style control
  - detected_sinks                         # magnitude-based, multi-level
  - outlier_channels                       # LLM.int8-style, top-k by max|x|

calibration_tokens: 65536
calibration_seeds: [0, 1, 2, 3, 4]         # disjoint calibration draws
eval_tokens: 262144
```

> ### ✎ CORRECTION — 2026-08-20 · config drift since this section was written
>
> Live files are authoritative; this section is the original intent.
>
> - **`detector.yaml`**: `tau_sweep` is now `[2, 5, 10, 20, 50, 100]`, and the
>   primary detector is the layer-relative form — see C6.
> - **`quant.yaml`**: unchanged, and the `per_token` line's "MUST be run" has
>   been vindicated — it is where the whole result lives (C12).
> - **`models.yaml`**: five checkpoints, not four; `prepend_bos: [false]` for
>   `qwen3_0.6b_base` (C3); every gated arm carries `trust_remote_code: true`
>   and a `subfolder` (C2).

`configs/train/baseline.yaml`:

```yaml
arch:
  n_layer: 6
  n_head: 6
  d_model: 384
  block_size: 512
  vocab_size: 16384                        # own BPE — see §9
  attention: softmax                       # {softmax, softmax1, gated}
  use_naive_attention: true                # ALL arms, including baseline

optim:
  batch_size: 16
  grad_accum: 2
  lr: 6.0e-4
  warmup_steps: 200
  max_steps: 12000
  weight_decay: 0.1
  grad_clip: 1.0

data:
  dataset: fineweb-edu
  tokens: 100_000_000

run:
  seeds: [0, 1, 2, 3, 4]
  ckpt_every: 500
  log_gpu_telemetry: true                  # temp + clock, see §10
```

---

## 9. Traps, and how each one gets caught

These are ordered by how much time they'd cost me if missed.

**1. A no-op quantizer.** If W8 per-channel *and* A8 per-tensor both come out lossless, the fake-quant is silently doing nothing. This is the most common failure in this kind of project and it invalidates everything downstream. `tests/test_fakequant.py` asserts: W8 per-channel is near-lossless, A8 per-tensor visibly degrades, A4 per-tensor degrades badly. **Do not proceed past Day 3 until this test passes.**

**2. `output_attentions=True` returns probs for every layer at once.** Peak memory is `L × B × H × T × T × 2` bytes — 5.9 GB for a 22-layer/32-head model at T=2048. It also silently disables SDPA/FlashAttention (2–3× slowdown). Fix: per-module forward hooks that compute statistics *inside* the hook and return floats. Peak becomes one attention tensor instead of L. Batch size 1 for all Track-A measurement.

**3. Never stash raw activation tensors.** Running max / mean / kurtosis accumulators in the hook, floats out, tensors freed immediately.

**4. BOS handling confounds everything.** GPT-2 doesn't prepend BOS; Qwen and Llama do. The "sink" is whatever occupies position 0. Run every model both with and without an explicit BOS, or the cross-model comparison is meaningless.

**5. The baseline must use the naive attention path too.** softmax1 and gating can't use SDPA — you're modifying the softmax, so you materialize `B × H × T × T`. If the baseline uses fused kernels and the variants don't, I'm benchmarking kernel implementations. Consequence: **wall-clock comparisons between arms are meaningless.** Report steps and tokens, never seconds.

**6. Embedding parameters dominate at this scale.** At `d_model=384` with GPT-2's 50k vocab, that's 19.3M embedding params against 10.6M in the actual transformer — the part I care about becomes a minority of the model, and the softmax dominates compute. Fix: train a 16k BPE on my own corpus. Total drops to ~17M and the layers under test become the majority.

**7. Per-token activation quant may already absorb the effect.** Treat this as the null hypothesis, not a disappointment. "The 2023 architectural fix is largely redundant under 2025 quantization practice" is a publishable-shaped finding and exactly the kind of thing an audit exists to establish.

**8. Seed variance will be large.** Paired bootstrap on per-seed deltas. Never compare unpaired means.

**9. Sinks are not only at position 0 in Qwen models.** Covered in §6 — magnitude-based detector, τ sweep, validated against attention.

> ### ✎ CORRECTION — 2026-08-20 · four traps this list did not have
>
> Ordered, as above, by how much time they would have cost. All four were found
> by running code, and **none of them crash** — each produces plausible-looking
> wrong numbers, which is why they belong in this section rather than in a
> debugging note.
>
> **C8. The fp16 exception must be excluded from *scale derivation*, not merely
> pasted back afterwards.** *(Worst of the four.)* The obvious implementation
> quantizes the whole tensor, then restores the exempt entries. But the entire
> reason to hold a sink token in fp16 is that it drops out of the observed
> range, letting every remaining token share a tight scale. Derive the scale
> from the full tensor and the surviving tokens still carry the dragged-out
> scale — so `D_sink` would come out ≈ 0 **under `per_tensor`, the exact arm
> where the classic effect is supposed to live**. That is a false negative that
> reads as a finding. The fix is a `scale_source` parameter carrying a masked
> copy; `tests/test_patch.py` pins the mechanism, asserting that `per_token`
> provably *cannot* benefit (to `atol=0`) while `per_tensor` does.
>
> **C9. Per-tensor activation quant must be *statically* calibrated, and
> calibrated per exception arm.** Deriving the per-tensor range from each
> evaluation batch hands it per-batch adaptivity that a deployed per-tensor
> scheme does not have — flattering precisely the arm under audit and biasing
> the project toward its own hypothesis. `quant/calibrate.py` now runs a real
> calibration pass. It must run **once per exception arm**, because the range a
> per-tensor scale should use genuinely differs depending on which tokens are
> held out of it; `reset_ranges` between cells stops one arm's calibration
> leaking into the next.
>
> **C7. GPT-2 does not use `nn.Linear`.** Its `c_attn` / `c_proj` / `c_fc` are
> HF `Conv1D`, which stores weight **transposed** — `(in, out)` — and computes
> `x @ W + b`. Two failures, neither loud:
> - a patcher matching only `nn.Linear` leaves GPT-2 **entirely unquantized**
>   and reports zero damage everywhere;
> - `per_channel` weight quant reduces over every axis but the first, which on a
>   transposed weight gives one scale per **input** channel. No crash — just
>   wrong numbers for the one checkpoint whose job is the pre-QK-Norm contrast,
>   and the W8 control would still have looked near-lossless enough to pass.
>
> The §9.1 no-op guard caught this on first contact with a real model, which is
> the strongest argument for having written it.
>
> **C-extra. `output_attentions` can be defused rather than avoided.** §9.2
> prescribes hooks that compute statistics in-hook. It omits the mechanism that
> makes this actually bounded: a forward hook may **replace** a module's output,
> so the hook reads the probabilities, reduces them to floats, and returns the
> tuple with the probability tensor swapped for `None`. HF then stores `None` in
> its `all_attentions` tuple and peak memory holds **one** attention tensor
> instead of L. A hook that records correctly but forgets to null out will only
> fail on the largest model at the longest context — i.e. late —
> so `tests/test_hooks.py` asserts the replacement explicitly.

---

## 10. Hardware

Target: RTX 3060 mobile, 6 GB VRAM, 16 GB system RAM.

> ### ✎ CORRECTION C10 — 2026-08-20 · measured, not estimated
>
> Verified: `NVIDIA GeForce RTX 3060 Laptop GPU`, sm_86, driver 595.71,
> torch 2.13.0+cu126.
>
> - **VRAM is 5.01 GiB free of 6.00 GiB** — roughly 1 GiB is driving the
>   display. Every budget in this section should be read against ~5 GiB, not 6.
>   The three 1B checkpoints at ~2 GB each still fit comfortably one at a time.
> - **HF cache is ~7.5 GB, not ~5 GB**: three 1B arms (~2 GB each) +
>   Qwen3-0.6B (~1.2 GB) + GPT-2 (~0.5 GB). Still well inside the 30 GB budget;
>   144 GB free at time of writing.
> - **`pip install torch` gives a CPU-only build on Windows.** The PyPI default
>   resolved to `2.13.0+cpu` — everything imports, every test passes, and
>   `torch.cuda.is_available()` is silently `False`. The CUDA build must come
>   from the PyTorch index explicitly (`--index-url .../cu126`).
> - **Practical:** the cu126 wheel is 2.6 GB and pip cannot resume a partial
>   download. On an unreliable connection, fetch the wheel with a resumable
>   transfer and `pip install` it from disk instead.



### VRAM is not the binding constraint

| Model | fp16 weights | Attention peak @ T=1024, B=1 (hooked) | Fits? |
|---|---|---|---|
| GPT-2 small | 0.25 GB | ~25 MB | Trivially |
| Qwen3-0.6B | 1.2 GB | ~34 MB | Yes |
| Qwen3-1.7B | 3.4 GB | ~34 MB | Yes, comfortably |
| Track B model (17M) | 0.07 GB | ~1.8 GB (naive, B=16) | Yes at B=16 |

Track B optimizer state at 17M params is ~290 MB with AdamW in mixed precision — negligible. The naive attention path at B=16, T=512, 6 heads is roughly 100 MB of probs per layer, ~3× that with scores and softmax intermediates, ~1.8 GB across six layers. B=16 is the safe default on a card that's also driving a display; B=32 will work on a headless setup.

### Wall-clock is the binding constraint

Non-embedding params ≈ 10.6M. Over 100M tokens: `6 × 1.06e7 × 1e8 ≈ 6.4e15` FLOPs. At a realistic 15–25% MFU for a small model on the naive attention path: **~30–50 min per run**.

- 3 arms × 5 seeds = 15 runs ≈ **8–12 hours**, one overnight.
- First pass at 3 seeds = 9 runs ≈ 6 hours if I want a same-day signal.

**Measure actual tokens/sec on Day 5 and re-extrapolate.** Don't trust the estimate above.

### Thermals — the real overnight risk

Mobile 3060 TGP varies 80–115 W by chassis, and sustained load throttles. Mitigations:

- Fix a power/clock target rather than letting it boost and sag.
- Log GPU temperature and clock alongside loss (`log_gpu_telemetry: true`).
- Hard surface, not a bed or a lap.
- `train.py` checkpoints every 500 steps and resumes cleanly. Thermal degradation doesn't corrupt results here (I'm comparing final loss, not throughput), but a shutdown at hour 6 of a 10-hour sweep does.

### Disk and RAM

- HF cache for the 4 Track-A checkpoints: ~5 GB.
- Tokenized corpus: 100M tokens as uint16 = 200 MB. **Stream and tokenize incrementally** (`load_dataset(..., streaming=True)`). A non-streaming FineWeb load will eat 16 GB of RAM and swap.
- Checkpoints: final weights only (~70 MB each), no optimizer state. 15 runs = ~1 GB.
- Budget **30 GB free**.

### What's explicitly not needed

This is **fake quantization** — simulating rounding error in fp16 — not real int8 inference. No bitsandbytes kernels, no CUTLASS, no Ada-or-newer requirement. Measuring actual quantized *speedup* would be a different project with genuinely different hardware needs.

---

## 11. Schedule

### Week 1 — Track A measurement

| Day | Work | Gate |
|---|---|---|
| 0 | **Verify the `gated_attention` checkpoints exist in both arms** (§3). Read the Anatomy paper and skim the survey | If the pair doesn't exist, reframe before writing code |
| 1 | Hooks + sink mass on GPT-2 small and Qwen3-0.6B. Reproduce the canonical per-layer/per-head sink figure | If no sink is visible, the hooks are wrong. Stop and fix |
| 2 | Massive activations: ∞-norms, outlier channels, kurtosis. Confirm co-location with sink positions. Build and validate the magnitude detector | Detector must agree with attention on ≥ the position-0 case |
| 3 | Fake-quant harness + `test_fakequant.py` | **Hard gate.** No further work until the test passes |
| 4 | Run the `fp16_exceptions` grid across all 4 checkpoints × 2 granularities. Compute `D_sink` | First real result |
| 5 | Track B: one full baseline training run end-to-end. Measure tokens/sec. Confirm my own model forms a sink | If TinyStories gives no sink (short contexts, easy data), switch to FineWeb-Edu at ctx 512 |
| 6 | Implement softmax1 + port G1 gating. 200-step smoke test each. Run `test_softmax1.py` | Smoke tests only, no conclusions |
| 7 | Launch the 15-run grid overnight. Draft README skeleton around the two figures already banked | — |

### Week 2 — analysis and writing

- Aggregate, bootstrap, figures.
- Write `LIMITATIONS.md` properly.
- README: finding first, method second, architecture last.

> ### ✎ CORRECTION C11 — 2026-08-20 · gate status
>
> | Day | Gate | Status |
> |---|---|---|
> | 0 | Both gated arms exist | ✅ **PASSED** — three arms, not two (C1) |
> | 1 | Sink visible, else hooks are wrong | ✅ **PASSED** — GPT-2 reproduces the canonical figure: mean sink mass 0.02 → 0.67 across depth, 100% of heads > 0.5 by layer 10, entropy falling 3.04 → 1.44 |
> | 2 | Detector agrees with attention | ✅ **PASSED** — but only after replacing the detector (C6) |
> | 3 | **HARD GATE** — fake-quant is not a no-op | ✅ **PASSED** — W8/per-channel 0.0075, A8/per-tensor 0.235, A4/per-tensor 0.885; per-token beats per-tensor 10.9× on outlier activations |
> | 4 | First real result | ⏳ pending the gated trio |
>
> Not on the original schedule but done: Qwen3-0.6B-Base measured end-to-end
> through the real `sinks.measure` CLI, confirming GQA head accounting, QK-Norm
> architecture handling, and the non-monotonic sink trajectory (flat for 3
> layers → 0.62 at layer 3 → dip to 0.32 at layer 13 → 0.76 by layer 27, the
> Rise–Plateau–Fall shape the Anatomy paper describes).
>
> **108 tests passing.** Track A (`sinks/`, `quant/`) is code-complete with no
> remaining stubs. `train/` and `analysis/` are still stubs.

> ### ✎ CORRECTION C17 — 2026-08-20 · the Day-2 gate conflated "no sink" with "broken detector"
>
> §11 Day 2 says: *"Detector must agree with attention on ≥ the position-0
> case"*, and the implementation aborted whenever the detector flagged nothing
> at any τ. Running the head-wise gated arm, it **aborted** — and it was wrong
> to.
>
> "The detector found nothing" has two causes that must not be conflated:
>
> - **(a) the detector is broken** — attention plainly shows a sink and the
>   magnitude detector missed it. Abort: every attribution number downstream
>   would be computed against a wrong sink set.
> - **(b) the model genuinely has no sink.** That is a **result**, and it is
>   precisely the result the gated-attention paper predicts for its own
>   checkpoints. Aborting here turns the project's most important positive
>   finding into a crash.
>
> Attention discriminates between them, and the discriminator reuses §6's own
> `S > 0.5` threshold rather than inventing a constant: if essentially no head
> concentrates on position 0 there is nothing to find, and finding nothing is
> correct. `measure_model` now reports `sink_free`, `frac_heads_sinking`,
> `mean_sink_mass` and `max_head_sink_mass`, and raises only when the detector
> comes up empty **while attention says a sink is there**.
>
> Worth stating plainly: a gate designed to catch a broken detector nearly
> suppressed a genuine finding. Any hard gate phrased as "X must be present"
> has this failure mode when absence of X is itself a possible answer.

<a id="c12"></a>
> ### ✎ C12 — 2026-08-20 · preview of the null hypothesis (a *result*, not a correction)
>
> Measured on GPT-2 small through the real `evaluate_cell` path. **These are not
> results**: one calibration draw, no seeds, no CIs, and a 4096-token corpus of
> this repo's own markdown (hence `ppl_ref` = 148). Reported because the *shape*
> is informative and it validates the plumbing.
>
> | bits | granularity | Δppl (`none`) | Δppl (`position_0`) | **D_sink (nats)** |
> |---|---|---|---|---|
> | 8 | per_tensor | +59.16 | +15.85 | **0.234** |
> | 8 | per_token | +0.75 | −0.20 | **0.0064** |
> | 4 | per_tensor | +11728 | +6965 | 0.513 |
> | 4 | per_token | +4042 | +3586 | 0.115 |
>
> At 8 bits, **per-token scaling shrinks `D_sink` by 36×**, and its baseline
> damage is +0.75 ppl against per-tensor's +59.16. Per-token quantization alone
> removes ~99% of the damage that makes the sink look important. That is trap
> §9.7's null hypothesis — *"the 2023 architectural fix is largely redundant
> under modern quantization practice"* — showing up on the first real
> measurement.
>
> **A sharper hypothesis than the plan currently states:** at 4 bits per-token
> still shows 0.115 nats. The redundancy may be **bit-width dependent** — real
> at 8 bits, weakening at 4. If that survives the seeds and CIs it is a more
> interesting finding than a flat yes/no, and it is worth stating as a
> hypothesis *now*, before the grid runs, rather than discovering it in the
> results and rationalising it afterwards.

> ### ✎ CORRECTION C18 — 2026-08-20 · the plan's Track-A "seed" does not apply to the arm that matters
>
> §6 says: *"for Track A the 'seed' is the calibration set draw (5 disjoint
> calibration sets)"*. That works for statically-calibrated quantization. It
> does **not** work for per-token scaling, which derives its scale from the
> tensor in front of it and never reads the calibration set at all.
>
> Measured across five disjoint draws on GPT-2:
>
> | arm | std of Δppl across draws |
> |---|---|
> | `per_tensor` 8-bit | 10.23 |
> | `per_tensor` 4-bit | 422.46 |
> | **`per_token` 8-bit** | **0.000000** |
> | **`per_token` 4-bit** | **0.000000** |
>
> Five identical numbers. A percentile CI over them has width exactly zero — and
> it would land on the **per-token arm, which is the whole research question**.
> Reported as-is, Figure 1 would show a wide interval next to an infinitely
> precise one, and the precise one would be an artefact of a randomness source
> that does not exist for that arm.
>
> **Fix:** confidence intervals bootstrap over held-out **sequences**, which vary
> for every arm and keep the two granularities comparable. Each grid cell now
> stores per-sequence mean NLL (one float per sequence — negligible) alongside
> the aggregate. The calibration-draw spread is still computed and reported via
> `analysis.stats.describe_variance_sources`, so the zero-variance property is
> visible in the output rather than inferred from suspiciously tight intervals.
>
> This is worth stating in the README: the project set out to audit papers that
> report single-run numbers without variance, and its own pre-registered
> variance source turned out not to apply to half the grid.

<a id="r1"></a>
> ### ✎ RESULT R1 — 2026-08-20 · the gated arms are sink-free, and the matched pair proves it is the gate
>
> All five Track-A checkpoints measured, one calibration draw, no-BOS, seq 512.
>
> | model | mean sink mass | heads > 0.5 | max head | magnitude | sink-free |
> |---|---|---|---|---|---|
> | GPT-2 small (no QK-Norm) | 0.340 | 0.243 | 0.882 | 87.8× | no |
> | Qwen3-0.6B-Base | 0.473 | 0.475 | 0.971 | 1198× | no |
> | **`1B_baseline`** (control) | **0.385** | **0.362** | 0.938 | **352×** | no |
> | **`1B_headwise`** (+0.1% params) | **0.054** | **0.002** | 0.552 | **3.8×** | **yes** |
> | `1B_elementwise` (+12% params) | 0.021 | 0.000 | 0.319 | 4.4× | **yes** |
>
> | contrast | sink mass | heads > 0.5 | magnitude |
> |---|---|---|---|
> | baseline → head-wise **(MATCHED)** | 7.1× ↓ | 0.362 → 0.002 | **91× ↓** |
> | baseline → element-wise *(confounded)* | 18.6× ↓ | 0.362 → 0.000 | 79× ↓ |
>
> **This replicates the gated-attention paper's central claim** — its
> checkpoints are attention-sink-free — on a controlled pair, which is the part
> nobody had done.
>
> **C13's confound turns out not to threaten the headline.** The worry was that
> element-wise gating buys its effect with +12% parameters. But head-wise, at
> **+0.1% parameters**, already collapses the sink: 91× reduction in
> layer-relative magnitude and essentially no sinking heads. Sink elimination is
> therefore **not a capacity effect**. What remains confounded is only the
> *increment* from head-wise to element-wise (7.1× → 18.6×), which is a much
> narrower claim than the one C13 put at risk.
>
> **A wrinkle worth keeping.** Head-wise keeps entropy flat across depth
> (3.01 → 2.91) while element-wise still falls (2.92 → 1.42), close to the
> baseline's fall (2.71 → 1.85). So element-wise removes the *position-0* sink
> but attention still concentrates somewhere; head-wise leaves attention broadly
> diffuse. The two gates are not doing the same thing, and "sink-free" does not
> imply "unconcentrated". Pairing sink mass with entropy is what makes this
> visible — §6's reason for keeping both metrics, vindicated.
>
> **Consequence for the quantization grid:** `D_sink` is **undefined** for both
> gated arms. There are no sink positions to hold in fp16, so the
> `detected_sinks` cell cannot be built for them. That is not missing data — it
> is what sink-free means. Those models get the `none` and `position_0` arms
> only. (`LIMITATIONS.md` §17.)
>
> Caveats, unchanged: one draw, no seeds, no CIs, and a small in-repo corpus.
> This is the measurement phase (§11 Days 1–2), not the audit result.

<a id="r2"></a>
> ### ✎ RESULT R2 — 2026-08-20 · Figure 1 — the null hypothesis, confirmed with CIs
>
> `D_sink` at **8-bit** activations, sequence-bootstrap 95% CIs, in nats:
>
> | model | per-tensor (2023) | per-token (modern) |
> |---|---|---|
> | GPT-2 small | **+0.2219** [+0.183, +0.260] | +0.0004 [−0.013, +0.012] **CI crosses zero** |
> | Qwen3-0.6B-Base | **+0.5287** [+0.487, +0.572] | −0.0068 [−0.017, +0.004] **CI crosses zero** |
> | `1B_baseline` | **+0.8763** [+0.824, +0.932] | +0.0082 [+0.0003, +0.0157] |
>
> **Under per-token activation scaling the sink-attributable damage is
> statistically indistinguishable from zero on two of three models, and ~100×
> smaller on the third.** Under per-tensor it is large and unambiguous on all
> three.
>
> This is trap §9.7's null hypothesis, confirmed rather than merely suspected:
> *the 2023 architectural fix is largely redundant under modern quantization
> practice* — not because the sink stopped existing (R1 shows it plainly there,
> 352× magnitude on `1B_baseline`), but because per-token scaling already
> handles it.
>
> **The 4-bit picture is different, and the caveat is load-bearing.** At 4 bits
> Qwen3-0.6B shows per-token `D_sink` = +2.98 nats — the effect does *not*
> vanish. But those cells sit at Δppl in the millions: the model is destroyed,
> and perplexity differences between two destroyed models are not interpretable.
> The bit-width dependence hypothesised in C12 is **not confirmed** by these
> numbers; it is merely not refuted, and answering it properly needs a bit width
> where the model still works (6-bit, or 4-bit weights with 8-bit activations).
>
> Caveats: one calibration draw per cell for the CI shown, a 232 KB in-repo
> corpus, seq 256, and `1B_baseline` still mid-grid. Not the final numbers.

<a id="r3"></a>
> ### ⚠ R3 (ORIGINAL) — SUPERSEDED by R3-rev below
>
> Measured on a 232 KB in-repo corpus of this project's own markdown and Python.
> `ppl_ref` was 136.6 (GPT-2) and 54.9 (Qwen3-0.6B) — those models find code and
> documentation genuinely hard, and the numbers below are inflated accordingly.
> Kept in full for the audit trail; **two of its claims did not survive** the
> corpus swap. See R3-rev.
>
> ### ✎ RESULT R3 — 2026-08-20 · the complete audit answer (200 grid cells, 5 checkpoints)
>
> `D_sink` at **8-bit** activations, nats, 95% sequence-bootstrap CI.
> *ZERO* marks an interval containing zero.
>
> | model | sink? | per-tensor (2023 setting) | per-token (modern baseline) |
> |---|---|---|---|
> | GPT-2 small | yes | +0.2218 [+0.183, +0.260] | +0.0004 [−0.013, +0.012] *ZERO* |
> | Qwen3-0.6B-Base | yes | +0.5287 [+0.487, +0.572] | −0.0068 [−0.017, +0.004] *ZERO* |
> | `1B_baseline` | yes | **+0.8763** [+0.824, +0.932] | +0.0082 [+0.0003, +0.0157] |
> | `1B_headwise` (+0.1% params) | **no** | **+0.0033** [−0.015, +0.024] *ZERO* | +0.0026 [−0.004, +0.009] *ZERO* |
> | `1B_elementwise` (+12% params) | **no** | +0.1113 [+0.049, +0.175] | +0.0028 [−0.007, +0.013] *ZERO* |
>
> ### The answer has two halves, and each is meaningless without the other
>
> **1. The architectural fix works.** On the matched pair — baseline vs
> head-wise, differing by **+0.1% parameters** — per-tensor `D_sink` falls from
> +0.8763 to +0.0033, a **269× reduction**, with the CI crossing zero. The 2023
> mitigation literature's claim is correct, and this is the first time it has
> been checked on a controlled architectural comparison rather than across
> models that differ in several ways at once.
>
> **2. The architectural fix is redundant.** The per-token column is
> statistically indistinguishable from zero for **every** model, including the
> ungated baseline. Switching the quantizer from per-tensor to per-token
> delivers what the architecture change delivers, at no training cost.
>
> A paper reporting only column one would be correct and misleading. A paper
> reporting only column two would conclude the mitigation does not work. This is
> precisely the gap the project was built to close, and it needed both the
> controlled pair and the modern baseline to close it.
>
> ### The finding I did not expect: sink mass ranks two models backwards
>
> Element-wise has **more** sink-attributable damage than head-wise (+0.1113 vs
> +0.0033, a 34× gap, non-overlapping CIs) despite having **less** sink mass
> (0.021 vs 0.054) — the two orderings invert.
>
> R1 shows why: head-wise keeps attention entropy flat across depth
> (3.01 → 2.91) while element-wise still concentrates (2.92 → 1.42), close to the
> baseline's 2.71 → 1.85. Element-wise strips attention mass off position 0 but
> retains concentration structure that per-tensor quantization still trips over.
>
> **Implication, stated carefully: sink mass is not a sufficient predictor of
> quantization sensitivity.** The standard metric calls element-wise the more
> sink-free model; its quantization behaviour says the opposite. Any method that
> selects or evaluates an architecture on sink mass alone — which is most of this
> literature — is using a proxy that can rank two models the wrong way round.
>
> This only surfaced because §6 insisted on carrying entropy *alongside* sink
> mass rather than reporting a single sink statistic, and because the release
> happened to contain two gate variants rather than one. Neither was foresight
> about this specific effect; both were general caution that paid off.
>
> ### What R3 does **not** establish
>
> - **4-bit is uninterpretable.** Those cells sit at Δppl in the millions. The
>   bit-width dependence floated in C12 is neither confirmed nor refuted, and
>   answering it needs a width where the model still works (6-bit, or W4A8).
> - **One corpus, and a poor one.** 232 KB of this repo's own markdown and
>   Python at seq 256. Code and prose are not a language-model evaluation set;
>   `ppl_ref` is correspondingly high. FineWeb-Edu is the intended corpus.
> - **The element-wise contrast stays confounded** by +12% parameters (C13).
>   The inversion above is measured against head-wise, which is matched — so the
>   confound does not touch that comparison, but it does limit what can be said
>   about element-wise in isolation.
> - **`D_sink` is undefined for the gated arms in the `detected_sinks` arm.**
>   The numbers above use `position_0`, which is always constructible. (§17)
> - No LAMBADA yet; perplexity only.

<a id="r3rev"></a>
> ### ✎ RESULT R3-rev — 2026-08-20 · the audit answer on FineWeb-Edu (200 cells, 5 checkpoints)
>
> Re-run of the entire grid on **FineWeb-Edu** (281 documents, 1.2 MB, stubs
> filtered) instead of the in-repo corpus. Held-out slice 8192 tokens = 32
> sequences at seq 256. This supersedes R3.
>
> The corpus swap was worth doing on its own evidence: `ppl_ref` fell from 136.6
> to **29.7** (GPT-2) and 54.9 to **17.9** (Qwen3-0.6B). The old corpus was not a
> language-model evaluation set.
>
> `D_sink` at 8-bit, nats, 95% sequence-bootstrap CI. *ZERO* = interval contains zero.
>
> | model | sink? | per-tensor (2023) | per-token (modern) | ratio |
> |---|---|---|---|---|
> | GPT-2 small | yes | +0.2097 [+0.188, +0.232] | +0.0047 [−0.002, +0.011] *ZERO* | 45× |
> | Qwen3-0.6B-Base | yes | +0.4378 [+0.396, +0.479] | +0.0096 [+0.003, +0.016] | 46× |
> | `1B_baseline` | yes | **+0.6450** [+0.601, +0.690] | +0.0203 [+0.014, +0.026] | 32× |
> | `1B_headwise` (+0.1%) | **no** | **+0.0117** [−0.0003, +0.024] *ZERO* | +0.0012 [−0.003, +0.005] *ZERO* | — |
> | `1B_elementwise` (+12%) | **no** | +0.1702 [+0.104, +0.236] | +0.0009 [−0.006, +0.007] *ZERO* | 183× |
>
> ### What changed from R3, and what did not
>
> **CORRECTED — the matched-pair reduction is 55×, not 269×.** Baseline
> +0.6450 → head-wise +0.0117. Still a large effect with a barely-zero-crossing
> CI, but R3's headline number was inflated ~5× by the bad corpus. The claim
> "the architectural fix works" stands; the magnitude does not.
>
> **CORRECTED — "statistically indistinguishable from zero for every model" is
> wrong.** On FineWeb-Edu the per-token interval excludes zero for
> Qwen3-0.6B (+0.0096) and for `1B_baseline` (+0.0203). The defensible statement
> is the weaker one:
>
> > per-token activation scaling reduces sink-attributable damage by **32–46×**
> > on sink-bearing models, to a level that is either zero or negligibly small.
>
> That still supports the redundancy conclusion — a 32× reduction from changing
> a quantizer setting is the same order as what the architecture change buys —
> but R3 overclaimed by resting on CIs that a better corpus resolved.
>
> **HELD, and strengthened — the inversion.** Element-wise still carries far more
> sink-attributable damage than head-wise despite less sink mass:
> **+0.1702 [+0.104, +0.236] vs +0.0117 [−0.0003, +0.024]**, non-overlapping, a
> **14.5× gap** (was 34× on the old corpus — smaller, but now on trustworthy
> data). Sink mass ranks these two models backwards on both corpora. This is the
> project's most novel finding and it survived the robustness check that broke
> two of its neighbours.
>
> **HELD — the per-tensor column is corpus-stable in ordering.** Every model
> keeps its rank; magnitudes shift by 0.7–1.5×. The effect is a property of the
> models, not of the evaluation text.
>
> ### Still not established
>
> Unchanged from R3: 4-bit remains uninterpretable (Δppl in the millions), no
> LAMBADA, the element-wise arm stays confounded by +12% parameters (C13), and
> `D_sink` uses `position_0` because `detected_sinks` is undefined for the
> sink-free arms (§17). One corpus is still one corpus — FineWeb-Edu is a better
> one, not a sample of many.

### De-scoping plan (if week 1 overruns)

**Cut Track B entirely and ship the inference-only audit.** Four checkpoints, the full quantization grid, the sink detector, `D_sink` with CIs. That's a complete week of work, has a real finding, and needs zero training runs. Add the pretraining arm later as v2. This is a legitimate outcome, not a failure — and it's better than a half-finished grid.

---

## 12. Limitations (write these before the results, not after)

1. **No rotation-based quantization.** QuaRot/SpinQuant-style Hadamard rotation is arguably the strongest modern outlier defence, and it isn't tested here. Implementing it properly is its own project. "Does sink mitigation survive rotation?" is the obvious next question and should be stated as future work, not left for a reader to notice. (Note for whoever picks it up: Qwen's non-power-of-2 embedding dimensions need the Paley construction for Hadamard matrices.)
2. **Scale.** Track B models are ~17M params; Track A tops out below 2B. Findings about sink-mitigation trade-offs at 7B+ are not established here.
3. **Fake quant ≠ deployed quant.** No claims about latency, throughput, or memory savings in a real serving stack.
4. **Single data distribution** for Track B pretraining. The Anatomy paper argues short-context training is causally implicated in sink formation; I train at ctx 512 throughout and therefore cannot separate that factor.
5. **The controlled pair is one lab's training run.** Even if both `gated_attention` arms exist, they're a single matched pair from one group. Checkpoints 3 and 4 provide external validity, but weakly.
6. **KV-cache quantization not separated from activation quantization** unless time allows a follow-up axis.

> ### ✎ CORRECTION — 2026-08-20 · six more, now written into `LIMITATIONS.md`
>
> The list above was written before any code ran. Items 7–12 were added from
> experiment and live in `LIMITATIONS.md`, which is authoritative:
>
> 7. **All three arms of the controlled triple already use QK-Norm** — the
>    comparison isolates gating *on top of* QK-Norm and cannot say whether the
>    two are substitutes (C1).
> 8. **The gated_attention weights carry no declared licence** (C2).
> 9. **Sink measurement depends on vendored modelling code** pinned to a
>    transformers version older than the one installed (C2).
> 10. **The headline metric was refined before data collection** — `D_sink`
>     decomposed, and computed in nats (C4, C5).
> 11. **The sink detector was changed from the §6 definition** (C6).
> 12. **Qwen3-0.6B-Base has no BOS token, so it has no BOS arm** (C3).
>
> Point 5 above — "the controlled pair is one lab's training run" — is
> unchanged in force but should now read **triple** rather than pair.

---

## 13. README structure (for when I get there)

1. **The finding**, one paragraph, with Figure 1 immediately below.
2. **What this is**: an audit of published claims, with confidence intervals, because the originals mostly don't report them.
3. **What was replicated and what's new** — explicit table, no ambiguity about credit.
4. Method: models, metrics, quantization setup.
5. Results, including negative ones.
6. Limitations (link to `LIMITATIONS.md`).
7. Reproduction: `make repro`, expected runtime, expected hardware.
8. References.

The differentiator here isn't the model or the mechanism — both are well-trodden. It's that the conclusions come with error bars and the prior work is credited precisely. That's the same methodological identity as the thesis, which is the point: two repos that visibly belong to the same person.
