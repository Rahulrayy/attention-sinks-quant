"""Corpus slicing, disjointness, and static range collection.

The disjointness tests are not ceremony. Overlapping calibration draws
understate between-seed variance, which makes bootstrap CIs narrower than they
should be — the precise failure mode this project exists to audit in other
people's work. A layout invariant that is never tested is a comment.
"""

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from quant.calibrate import (  # noqa: E402
    CorpusSlices,
    assert_disjoint,
    build_slices,
    collect_ranges,
    reset_ranges,
    to_batches,
    tokenize_stream,
)
from quant.patch import QuantLinear, patch_model, resolve_fp16_exceptions  # noqa: E402

IDS = list(range(10_000))


# --- slicing -----------------------------------------------------------------

def test_holdout_is_carved_off_the_front():
    s = build_slices(IDS, n_draws=5, tokens_per_draw=1000, eval_tokens=2000)
    assert s.holdout == IDS[:2000]
    assert s.n_draws == 5


def test_draws_are_contiguous_and_disjoint():
    s = build_slices(IDS, n_draws=5, tokens_per_draw=1000, eval_tokens=2000)
    seen = set()
    for i in range(5):
        d = s.draw(i)
        assert len(d) == 1000
        assert not seen & set(d), f"draw {i} overlaps an earlier draw"
        seen |= set(d)


def test_no_draw_touches_the_holdout():
    """The eval slice is fixed and must never be seen during calibration."""
    s = build_slices(IDS, n_draws=5, tokens_per_draw=1000, eval_tokens=2000)
    holdout = set(s.holdout)
    for i in range(5):
        assert not holdout & set(s.draw(i))


def test_short_corpus_is_rejected_rather_than_recycled():
    with pytest.raises(ValueError, match="disjoint"):
        build_slices(IDS[:500], n_draws=5, tokens_per_draw=1000, eval_tokens=2000)


def test_out_of_range_seed_is_rejected():
    s = build_slices(IDS, n_draws=3, tokens_per_draw=100, eval_tokens=100)
    with pytest.raises(IndexError, match="out of range"):
        s.draw(3)


def test_assert_disjoint_catches_a_bad_layout():
    """Checked by index span, not token value: ids repeat all over a corpus, so
    a value-based check would flag any two slices sharing vocabulary."""
    bad = CorpusSlices(holdout=[1, 2, 3], draws=[[1, 2, 3]])
    assert_disjoint(bad)          # spans [0,3) and [3,6) do not overlap
    assert_disjoint(CorpusSlices(holdout=[], draws=[[9] * 5, [9] * 5]))


# --- batching ----------------------------------------------------------------

def test_batches_are_size_one():
    batches = list(to_batches(IDS[:1000], 250))
    assert len(batches) == 4
    assert all(b.shape == (1, 250) for b in batches)


def test_ragged_tail_is_dropped():
    assert len(list(to_batches(IDS[:1010], 250))) == 4


def test_bos_replaces_rather_than_extends():
    """Both BOS policies must score the same number of positions."""
    plain = next(to_batches(IDS[:100], 10))
    withbos = next(to_batches(IDS[:100], 10, prepend_bos=True, bos_token_id=999))
    assert plain.shape == withbos.shape
    assert withbos[0, 0].item() == 999
    assert withbos[0, 1].item() == plain[0, 0].item()


def test_bos_without_a_bos_token_is_refused():
    with pytest.raises(ValueError, match="no bos_token_id"):
        next(to_batches(IDS[:100], 10, prepend_bos=True, bos_token_id=None))


def test_tokenize_stream_truncates_to_request():
    class FakeTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1] * 100}

    assert len(tokenize_stream(FakeTok(), ["a"] * 10, 250)) == 250


# --- static range collection -------------------------------------------------

def toy():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))


class Wrapper(nn.Module):
    """collect_ranges calls model(batch, use_cache=False), as HF models accept."""

    def __init__(self):
        super().__init__()
        self.net = toy()
        self.emb = nn.Embedding(64, 8)

    def forward(self, ids, use_cache=False):
        return self.net(self.emb(ids))


def test_collect_ranges_records_every_layer():
    model = Wrapper()
    patch_model(model, a_granularity="per_tensor")
    ranges = collect_ranges(model, [torch.randint(0, 64, (1, 16))], device="cpu")
    assert len(ranges) == 2
    assert all(v > 0 for v in ranges.values())


def test_collect_ranges_commits_static_amax():
    model = Wrapper()
    patch_model(model, a_granularity="per_tensor")
    collect_ranges(model, [torch.randint(0, 64, (1, 16))], device="cpu")
    assert all(
        m.static_amax is not None for m in model.modules() if isinstance(m, QuantLinear)
    )


def test_unpatched_model_is_refused():
    with pytest.raises(RuntimeError, match="no QuantLinear"):
        collect_ranges(Wrapper(), [torch.randint(0, 64, (1, 16))], device="cpu")


def test_reset_clears_ranges_between_grid_cells():
    """Leaking one exception arm's calibration into the next produces a small
    consistent bias rather than an obvious failure — the worst kind."""
    model = Wrapper()
    patch_model(model, a_granularity="per_tensor")
    collect_ranges(model, [torch.randint(0, 64, (1, 16))], device="cpu")
    reset_ranges(model)
    for m in model.modules():
        if isinstance(m, QuantLinear):
            assert m.static_amax is None and m.observed_amax == 0.0


def test_observed_range_excludes_exempt_positions():
    """A range calibrated with the outlier still in it would hand the quantized
    tokens the dragged-out scale the exception exists to avoid."""
    # Distinct ids: a random draw could repeat position 0's token elsewhere in
    # the sequence, and the outlier would then survive the exemption.
    ids = torch.arange(16).unsqueeze(0)

    plain = Wrapper()
    patch_model(plain, a_granularity="per_tensor",
                exceptions=resolve_fp16_exceptions("none"))
    with torch.no_grad():
        plain.emb.weight[ids[0, 0]] *= 500.0
    r_plain = collect_ranges(plain, [ids], device="cpu")

    exempt = Wrapper()
    patch_model(exempt, a_granularity="per_tensor",
                exceptions=resolve_fp16_exceptions("position_0"))
    with torch.no_grad():
        exempt.emb.weight.copy_(plain.emb.weight)
    r_exempt = collect_ranges(exempt, [ids], device="cpu")

    first = next(iter(r_plain))
    assert r_exempt[first] < r_plain[first], (
        "exempting position 0 should shrink the observed range on the first layer"
    )
