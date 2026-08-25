"""Hook mechanics, including the memory fix that trap §9.2 turns on.

Two things here are load-bearing:

1. The online moment merge must equal a single-pass computation. Track-A runs
   fold thousands of batches into one accumulator; a subtly wrong merge would
   produce plausible-looking kurtosis numbers that are simply wrong, with
   nothing to compare them against.

2. The attention hook must actually null out the probability tensor it read.
   That replacement is the entire reason peak memory stays at one attention
   tensor instead of L. A hook that records correctly but forgets to null is
   the 5.9 GB failure described in the plan, and it will only show up on the
   largest model at the longest context — i.e. late.
"""

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from sinks.hooks import (  # noqa: E402
    AttentionRecord,
    ResidualRecord,
    StreamingStats,
    _find_attention_probs,
    _find_hidden_states,
    _looks_like_attention,
    _looks_like_block,
    attach_attention_hooks,
    attach_residual_hooks,
)
from sinks.metrics import excess_kurtosis  # noqa: E402

B, H, T, D = 2, 4, 8, 16


# --- streaming moments -------------------------------------------------------

def test_kurtosis_matches_single_pass():
    torch.manual_seed(0)
    x = torch.randn(10_000)
    s = StreamingStats()
    s.update(x)
    assert s.kurtosis() == pytest.approx(excess_kurtosis(x), abs=1e-6)


def test_batched_merge_equals_single_pass():
    """The Pebay merge is the whole reason nothing is stored. Verify it."""
    torch.manual_seed(0)
    x = torch.randn(9_000) * 3.0 + 1.5
    x[:50] = 400.0                                   # massive activations

    single = StreamingStats()
    single.update(x)

    chunked = StreamingStats()
    for chunk in x.split(700):                       # uneven final chunk on purpose
        chunked.update(chunk)

    assert chunked.count == single.count
    assert chunked.mean == pytest.approx(single.mean, rel=1e-9)
    assert chunked.variance() == pytest.approx(single.variance(), rel=1e-7)
    assert chunked.kurtosis() == pytest.approx(single.kurtosis(), rel=1e-5)


def test_gaussian_kurtosis_is_near_zero():
    torch.manual_seed(0)
    s = StreamingStats()
    for _ in range(10):
        s.update(torch.randn(10_000))
    assert s.kurtosis() == pytest.approx(0.0, abs=0.15)


def test_running_max_is_absolute():
    s = StreamingStats()
    s.update(torch.tensor([1.0, -50.0, 3.0]))
    assert s.running_max == pytest.approx(50.0)


def test_empty_update_is_a_noop():
    s = StreamingStats()
    s.update(torch.empty(0))
    assert s.count == 0 and s.kurtosis() == 0.0


# --- structural discovery ----------------------------------------------------

class ToyAttention(nn.Module):
    """Qwen-style: separate q/k/v projections, returns (out, probs)."""

    def __init__(self, d=D, h=H):
        super().__init__()
        self.h = h
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

    def forward(self, x):
        b, t, d = x.shape
        scores = torch.randn(b, self.h, t, t)
        scores = scores.masked_fill(torch.triu(torch.ones(t, t), 1).bool(), float("-inf"))
        probs = scores.softmax(dim=-1)
        return (self.v_proj(x), probs)


class ToyBlock(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.attn = ToyAttention(d)
        self.mlp = nn.Linear(d, d)

    def forward(self, x):
        h = self.attn(x)[0]
        return (self.mlp(h + x),)


class ToyModel(nn.Module):
    def __init__(self, n_layer=3, d=D):
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(d) for _ in range(n_layer)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


def test_finds_probs_by_shape_not_position():
    probs = torch.rand(B, H, T, T)
    assert _find_attention_probs((torch.rand(B, T, D), probs))[0] == 1
    assert _find_attention_probs((probs, torch.rand(B, T, D)))[0] == 0


def test_ignores_non_square_four_d_tensors():
    """A (B, H, T, head_dim) tensor is not an attention map."""
    assert _find_attention_probs((torch.rand(B, H, T, D),)) is None


def test_finds_hidden_states():
    assert _find_hidden_states((torch.rand(B, T, D),)).shape == (B, T, D)


def test_recognises_both_projection_conventions():
    assert _looks_like_attention(ToyAttention())
    gpt2_style = nn.Module()
    gpt2_style.add_module("c_attn", nn.Linear(D, 3 * D))
    assert _looks_like_attention(gpt2_style)
    assert not _looks_like_attention(nn.Linear(D, D))


def test_recognises_blocks():
    assert _looks_like_block(ToyBlock())
    assert not _looks_like_block(ToyAttention())


# --- hook behaviour ----------------------------------------------------------

def test_attention_hook_records_every_layer():
    model, records = ToyModel(), {}
    with attach_attention_hooks(model, records):
        model(torch.randn(B, T, D))
    assert len(records) == 3
    for rec in records.values():
        assert rec.batches == 1
        assert len(rec.as_dict()["sink_mass_per_head"]) == H


def test_attention_hook_nulls_out_the_probabilities():
    """THE memory fix. Without this, peak is L attention tensors, not one."""
    captured = []

    class Capturing(ToyBlock):
        def forward(self, x):
            out = self.attn(x)
            captured.append(out[1])
            return (self.mlp(out[0] + x),)

    model = ToyModel()
    model.layers[0] = Capturing()
    with attach_attention_hooks(model, {}):
        model(torch.randn(B, T, D))

    assert captured[0] is None, "hook did not replace the probability tensor"


def test_null_out_can_be_disabled_for_debugging():
    captured = []

    class Capturing(ToyBlock):
        def forward(self, x):
            out = self.attn(x)
            captured.append(out[1])
            return (self.mlp(out[0] + x),)

    model = ToyModel()
    model.layers[0] = Capturing()
    with attach_attention_hooks(model, {}, null_out_probs=False):
        model(torch.randn(B, T, D))

    assert isinstance(captured[0], torch.Tensor)


def test_hooks_are_removed_on_context_exit():
    model, records = ToyModel(), {}
    with attach_attention_hooks(model, records):
        model(torch.randn(B, T, D))
    model(torch.randn(B, T, D))
    assert all(r.batches == 1 for r in records.values()), "hooks still firing after exit"


def test_residual_hook_accumulates_per_token_norms():
    model, records = ToyModel(), {}
    x = torch.randn(B, T, D)
    x[:, 0, :] *= 500.0                              # a sink at position 0
    with attach_residual_hooks(model, records):
        model(x)

    assert len(records) == 3
    norms = records["layers.0"].as_dict()["per_token_inf_norm"]
    assert len(norms) == T
    assert norms[0] == max(norms), "the injected sink is not the largest token"


def test_missing_attention_modules_raises():
    with pytest.raises(RuntimeError, match="no attention modules"):
        attach_attention_hooks(nn.Sequential(nn.Linear(D, D)), {})


def test_missing_blocks_raises():
    with pytest.raises(RuntimeError, match="no decoder blocks"):
        attach_residual_hooks(nn.Sequential(nn.Linear(D, D)), {})


def test_records_survive_multiple_batches():
    model, records = ToyModel(), {}
    with attach_attention_hooks(model, records):
        for _ in range(4):
            model(torch.randn(B, T, D))
    assert all(r.batches == 4 for r in records.values())
