"""The last-word protocol, where an off-by-one is silent and survivable.

Two failure modes motivate most of this file.

  * **The alignment.** Position j's logits predict token j+1. Score the target
    with `logits[-n_tgt:]` instead of `logits[-n_tgt-1:-1]` and every model is
    graded on predicting the token AFTER the answer. Accuracy drops to roughly
    zero on everything, uniformly, so the arms still rank plausibly and nothing
    looks wrong. `test_score_reads_the_predicting_positions` pins it with a
    model whose correct answer is known by construction.

  * **The leading space.** `" word"` and `"word"` are different BPE sequences.
    Splitting the target without its space asks every model to predict a token
    sequence that never occurs mid-sentence, which depresses accuracy equally
    everywhere and is invisible in a comparison.

The rest guard the reporting contract: accuracy is all-or-nothing across the
target's tokens, and `n_discordant` -- the real sample size of a paired 0/1
comparison -- is counted rather than inferred.
"""

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from quant.lambada import (  # noqa: E402
    build_examples,
    evaluate_lambada_cell,
    examples_sha,
    score,
)


class FakeTokenizer:
    """Whitespace tokenizer with a stable vocabulary and a real leading-space rule.

    `" cat"` and `"cat"` map to DIFFERENT ids, exactly as they would under BPE,
    so the leading-space test is testing the thing it claims to.
    """

    bos_token_id = None

    def __init__(self):
        self.vocab, self.inv = {}, {}

    def _id(self, tok):
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab) + 1
            self.inv[self.vocab[tok]] = tok
        return self.vocab[tok]

    def __call__(self, text, add_special_tokens=False):
        toks = ([" " + w for w in text[1:].split(" ")] if text.startswith(" ")
                else text.split(" "))
        return {"input_ids": [self._id(t) for t in toks if t not in ("", " ")]}


class Oracle(nn.Module):
    """Emits a chosen next-token id at every position; everything else is 0.

    Lets a test state "the model predicts X here" without training anything, so
    the expected accuracy is known by construction rather than by running it.
    """

    def __init__(self, vocab=64, plan=None):
        super().__init__()
        self.vocab, self.plan = vocab, plan or {}

    def forward(self, ids, use_cache=False):
        b, t = ids.shape
        logits = torch.zeros(b, t, self.vocab)
        for pos in range(t):
            logits[:, pos, self.plan.get(pos, 0)] = 10.0
        return type("O", (), {"logits": logits})()


# --- splitting ---------------------------------------------------------------

def test_target_keeps_its_leading_space():
    """The final word is tokenized as it appears in the passage."""
    tk = FakeTokenizer()
    ex = build_examples(tk, ["the cat sat on the mat"])[0]
    assert tk.inv[ex["target"][0]] == " mat"
    assert " mat" in tk.vocab and "mat" not in tk.vocab


def test_context_is_truncated_from_the_left():
    """LAMBADA's answer depends on far context, but the tokens NEAREST the
    target are what the model conditions on most directly. Truncating from the
    right would drop those and measure a different task."""
    tk = FakeTokenizer()
    text = " ".join(str(i) for i in range(50)) + " end"
    ex = build_examples(tk, [text], max_context=5)[0]
    assert len(ex["context"]) == 5
    assert [tk.inv[i] for i in ex["context"]] == ["45", "46", "47", "48", "49"]


def test_single_word_passages_are_dropped_not_scored_as_empty():
    """A passage with no context cannot pose the task. It must be discarded
    rather than scored against an empty prompt, which would count as a free
    wrong answer for every model."""
    tk = FakeTokenizer()
    with pytest.raises(RuntimeError):
        build_examples(tk, ["mat"])


# --- scoring -----------------------------------------------------------------

def test_score_reads_the_predicting_positions():
    """Context of 3 tokens, target of 1. The token is predicted by the logits at
    index 2 (the last context position), NOT index 3. An oracle that emits the
    right answer at position 2 must score 1.0; one that emits it at position 3
    must score 0.0."""
    ex = [{"context": [11, 12, 13], "target": [42], "text": ""}]

    right = score(Oracle(plan={2: 42}), ex, device="cpu")
    wrong = score(Oracle(plan={3: 42}), ex, device="cpu")

    assert right["accuracy"] == pytest.approx(1.0)
    assert wrong["accuracy"] == pytest.approx(0.0)


def test_accuracy_is_all_or_nothing_across_target_tokens():
    """A two-token answer with only the first token right is WRONG. lm-eval's
    lambada `acc` requires greedy decoding to reproduce the whole word, and a
    per-token average would quietly report partial credit."""
    ex = [{"context": [11, 12], "target": [42, 43], "text": ""}]

    both = score(Oracle(plan={1: 42, 2: 43}), ex, device="cpu")
    half = score(Oracle(plan={1: 42, 2: 44}), ex, device="cpu")

    assert both["accuracy"] == pytest.approx(1.0)
    assert half["accuracy"] == pytest.approx(0.0)


def test_target_nll_is_per_token_not_per_example():
    """Answers differ in token length, so summing NLL per example and averaging
    over examples would weight long answers more heavily than short ones. The
    reported figure divides by total target tokens."""
    ex = [{"context": [11], "target": [42], "text": ""},
          {"context": [11], "target": [42, 42], "text": ""}]
    r = score(Oracle(plan={}), ex, device="cpu")
    assert r["mean_target_tokens"] == pytest.approx(1.5)
    per_tok = [n / t for n, t in zip(r["per_example_nll"], (1, 2))]
    assert r["target_nll"] == pytest.approx(sum(r["per_example_nll"]) / 3)
    assert per_tok[0] == pytest.approx(per_tok[1])


# --- the paired contract -----------------------------------------------------

class TinyLM(nn.Module):
    """Embedding -> Linear -> projection, with `.logits` and `use_cache`.

    A real (if tiny) model rather than an oracle, because the thing under test
    is what QUANTIZATION does to the predictions, and an oracle has no weights
    to quantize -- `patch_model` refuses it outright, which is that guard
    working correctly.
    """

    def __init__(self, vocab=64, d=32):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, d)
        self.mix = nn.Linear(d, d)
        self.proj = nn.Linear(d, vocab)

    def forward(self, ids, use_cache=False):
        h = torch.tanh(self.mix(self.embed(ids)))
        return type("O", (), {"logits": self.proj(h)})()


def solved_examples(n=40, ctx_len=12):
    """Examples the fp16 model answers correctly BY CONSTRUCTION.

    Random targets would be wrong under both arms, the arms would agree on every
    example, and the discordant count would be zero for a reason that has
    nothing to do with the counter. Taking each target from the model's own
    greedy prediction puts reference accuracy at 1.0, so any prediction that
    quantization moves shows up as a discordant pair -- which is the only way to
    tell a working counter from one wired to zero.
    """
    torch.manual_seed(0)
    model = TinyLM()
    out = []
    for _ in range(n):
        ctx = torch.randint(1, 60, (ctx_len,)).tolist()
        with torch.no_grad():
            logits = model(torch.tensor([ctx])).logits
        out.append({"context": ctx, "target": [int(logits[0, -1].argmax())], "text": ""})
    return out


def _cell(bits, examples):
    return evaluate_lambada_cell(
        TinyLM(), calib_batches_fn=lambda: iter([]), examples=examples,
        bits=bits, act_granularity="per_token", device="cpu",
    )


def test_discordant_pairs_are_counted_not_inferred():
    """All the information in a paired 0/1 comparison lives in the examples the
    two arms disagree on, so the count has to be real and has to track damage.
    A counter wired to zero, or one that returned the example count, would fail
    the ordering below.

    8-bit is asserted as *few* flips rather than none. This toy is untrained, so
    many of its argmaxes are near-ties that any perturbation can tip -- which is
    a real property of accuracy as a metric, not a fixture artefact: a discrete
    score moves when a tie moves, however small the perturbation. It is also why
    the run JSON reports `n_discordant` next to the accuracy drop."""
    ex = solved_examples()

    gentle = _cell(8, ex)
    brutal = _cell(2, ex)

    assert gentle["n_examples"] == 40
    assert gentle["accuracy_ref"] == pytest.approx(1.0)   # correct by construction
    assert gentle["n_discordant"] <= 2
    assert brutal["n_discordant"] > gentle["n_discordant"]
    assert brutal["accuracy_drop"] > gentle["accuracy_drop"]


def test_discordant_count_matches_the_per_example_arrays():
    """The count and the arrays it summarises are both written to the run JSON,
    so a reader can recompute it. They must not be able to disagree."""
    r = _cell(2, solved_examples())
    recomputed = sum(1 for a, b in zip(r["per_example_correct_ref"],
                                       r["per_example_correct_quant"]) if a != b)
    assert r["n_discordant"] == recomputed


def test_examples_sha_is_order_sensitive():
    """Two runs that scored the same passages in a different order did not score
    the same first-N slice, and are not comparable."""
    assert examples_sha(["a", "b"]) != examples_sha(["b", "a"])
    assert examples_sha(["a", "b"]) == examples_sha(["a", "b"])


def test_examples_sha_separates_a_join_from_two_examples():
    """Concatenation must not collide: ["ab"] and ["a", "b"] are different
    selections and hashing without a separator would call them equal."""
    assert examples_sha(["ab"]) != examples_sha(["a", "b"])
