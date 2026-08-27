"""The pretraining arm. Specified, partly built, and deliberately cut.

READ THIS BEFORE ASSUMING ANYTHING HERE RUNS. Most of this package raises
NotImplementedError by design. It is kept rather than deleted because two of its
pieces are real and load-bearing elsewhere, and because deleting it would erase
the record of a scope decision that the audit's conclusions depend on.

WHAT THIS ARM WAS FOR
---------------------
Three of the mitigations in the literature can be measured on released
checkpoints, which is what the rest of this repository does. One cannot:
`softmax1` changes the attention normaliser, so it has to be present during
training. It is widely cited and, as far as we found, has never been evaluated
with seeds. Testing it needs ~17M-parameter models trained from scratch across
three arms and five seeds.

WHY IT WAS CUT
--------------
The project's own de-scoping plan named this arm as the first thing to drop if
the schedule slipped, and shipping the inference-side audit alone as the
fallback. That fallback was taken. The inference-side work answered the question
it was built to ask, across three corpora, three bit widths, two metrics and
five calibration draws, and adding a from-scratch training study would have been
a second project rather than a completion of this one.

The cost of the cut is stated plainly in README section 6 rather than hidden:
this repository says nothing about softmax1, and the table in section 3 keeps
the row marked "never evaluated with seeds, and still is not here".

WHAT ACTUALLY WORKS
-------------------
`attention.softmax1`
    Complete and tested. Implements exp(x_i) / (1 + sum_j exp(x_j)) by
    appending a fixed zero logit and dropping its output slot, which is
    numerically stable via the usual max-shift. Only dim=-1 is supported; any
    other axis raises rather than silently doing something else.

    It also pins the identity that makes this arm necessary in the first place.
    With s = sum(exp) / (1 + sum(exp)) < 1, softmax1(x) = s * softmax(x), so the
    output is the original convex combination scaled down uniformly, not "the
    same attention with less sink". A model retrofitted with softmax1 therefore
    degrades for reasons that say nothing about the sink mechanism. Several
    published blog posts get this wrong. `tests/test_softmax1.py` holds it.

`attention.OutputGate`
    Complete. The G1 query-dependent sigmoid gate on the attention output, in
    both the elementwise and headwise variants, mirroring the config flags on
    the released QwQZh/gated_attention checkpoints. Not used by the audit, which
    measures those checkpoints as released rather than reimplementing them; kept
    because it documents what the released flag actually does.

WHAT IS A CONTRACT ONLY
-----------------------
`attention.CausalSelfAttention`, `model.Block`, `model.GPT`, `data`, `train`.
Nine stubs. Their docstrings carry the design constraints that were worked out
before the cut, and those constraints are the useful part:

  * Every arm, including the baseline, must take the naive materialised
    attention path. softmax1 and gating cannot use fused SDPA, and if the
    baseline used fused kernels while the variants did not, the study would
    benchmark kernel implementations rather than architectures. A consequence
    is that wall-clock comparisons between arms are meaningless: report steps
    and tokens, never seconds.
  * All arm-specific logic stays in `attention.py`. If it leaks into `model.py`
    the ablation stops being clean and the three arms stop being comparable.
  * Embedding parameters dominate at this scale. At d_model=384 with GPT-2's
    50k vocab that is 19.3M embedding parameters against 10.6M in the
    transformer proper, so the part under test becomes a minority of the model.
    A 16k BPE trained on the project's own corpus brings the total to ~17M and
    puts the layers under test back in the majority.

IF ANYONE PICKS THIS UP
-----------------------
`softmax1` is the arm worth the compute, for the reason above: it is the one
claim in the roster that cannot be checked any other way. The scaling identity
is already implemented and tested, so the work is the trainer, not the method.
"""
