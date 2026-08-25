"""The grid driver: one JSON per cell, resumable, and carrying its provenance.

`run_grid` used to live in a session scratchpad, which is how the corpus that
produced 200 committed cells ended up outside the repo. Now that it is in
quant/evaluate.py the two properties the sweep actually depends on are pinned
here: a crash at cell 37 must leave cells 1-36 on disk and be resumable, and a
cell must record the corpus and token budget it was measured with rather than
inheriting a config value it did not use (LIMITATIONS §20).
"""

import json

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from analysis.aggregate import load_quant_runs  # noqa: E402
from quant.calibrate import CorpusSlices, tokenize_stream  # noqa: E402
from quant.evaluate import holdout_sha, load_corpus, run_grid  # noqa: E402

VOCAB, D = 32, 16


class ToyLM(nn.Module):
    """Smallest thing that satisfies the contract run_grid calls through.

    Needs an nn.Linear for patch_model to find, and a `.logits` attribute on the
    output, which is what token_nll reads.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(VOCAB, D)
        self.proj = nn.Linear(D, D)
        self.head = nn.Linear(D, VOCAB)

    def forward(self, ids, use_cache=False):
        h = self.proj(self.embed(ids))
        return type("Out", (), {"logits": self.head(h)})()


class ToyTokenizer:
    bos_token_id = None


def slices(n_draws=2, per_draw=32, eval_tokens=64):
    ids = [i % VOCAB for i in range(eval_tokens + n_draws * per_draw)]
    return CorpusSlices(
        holdout=ids[:eval_tokens],
        draws=[
            ids[eval_tokens + i * per_draw : eval_tokens + (i + 1) * per_draw]
            for i in range(n_draws)
        ],
    )


def grid(tmp_path, **kw):
    kw.setdefault("bits_list", [8])
    kw.setdefault("granularities", ["per_token"])
    kw.setdefault("exceptions", ["none", "position_0"])
    kw.setdefault("seeds", [0, 1])
    return run_grid(
        ToyLM(), ToyTokenizer(), model_id="toy", slices=slices(), seq_len=16,
        device="cpu", out=str(tmp_path), **kw,
    )


def test_writes_one_json_per_cell_with_the_expected_name(tmp_path):
    written = grid(tmp_path)
    assert len(written) == 4
    assert "toy_b8_per_token_none_calib0.json" in written
    assert len(list(tmp_path.glob("*.json"))) == 4


def test_skip_existing_makes_the_sweep_resumable(tmp_path):
    grid(tmp_path, seeds=[0])
    again = grid(tmp_path, seeds=[0, 1])
    # Only the cells that were missing get recomputed. A resume that redid
    # finished cells would still be correct, and would also mean a sweep that
    # crashes late never finishes.
    assert all("calib0" not in name for name in again)
    assert len(again) == 2


def test_skip_existing_false_overwrites(tmp_path):
    grid(tmp_path, seeds=[0])
    assert len(grid(tmp_path, seeds=[0], skip_existing=False)) == 2


def test_provenance_is_recorded_in_every_cell(tmp_path):
    prov = {"corpus": "data/fineweb_edu.txt", "corpus_sha256": "deadbeef",
            "seq_len": 16, "calib_tokens": 32, "eval_tokens": 64}
    grid(tmp_path, provenance=prov)
    for path in tmp_path.glob("*.json"):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        assert d["corpus_sha256"] == "deadbeef"
        assert d["eval_tokens"] == 64


def test_provenance_never_shadows_a_measured_field(tmp_path):
    """A stray provenance key must not overwrite the cell's own results.

    Provenance is merged into the same dict as the measurements, so a key
    collision would silently replace a number with a label.
    """
    grid(tmp_path, provenance={"bits": 999, "delta_ppl": -1.0})
    with open(tmp_path / "toy_b8_per_token_none_calib0.json", encoding="utf-8") as fh:
        d = json.load(fh)
    assert d["bits"] == 8
    assert d["delta_ppl"] != -1.0


def test_cells_are_readable_by_the_aggregator(tmp_path):
    """The grid and the aggregator must agree on the cell schema."""
    grid(tmp_path)
    df = load_quant_runs(str(tmp_path))
    assert len(df) == 4
    assert set(df.granularity) == {"per_token"}
    assert set(df.exception) == {"none", "position_0"}


# --- the tokenization contract -----------------------------------------------

class ChunkTokenizer:
    """Ids are two-character chunks of the raw text.

    Whitespace-splitting would not do: it is newline-agnostic and would show no
    difference. What matters is that WHERE the text is cut changes the ids —
    the property BPE has, in miniature.
    """

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [hash(text[i : i + 2]) for i in range(0, len(text), 2)]}


def test_corpus_is_read_as_one_string(tmp_path):
    """A corpus must reach the tokenizer whole, not line by line.

    Line-by-line tokenization moves BPE boundaries at every newline and drops
    the blank lines between documents, so it produces a different held-out
    slice from the same bytes on disk. The committed 8-bit and 4-bit cells were
    measured whole-file; a reader that quietly changed would make new cells
    incomparable to them while the corpus hash still matched.
    """
    p = tmp_path / "c.txt"
    p.write_text("alpha beta\n\ngamma delta\n", encoding="utf-8")
    assert load_corpus(str(p)) == ["alpha beta\n\ngamma delta\n"]


def test_the_two_readers_really_do_produce_different_streams(tmp_path):
    """Pins the failure mode itself, so the guard above cannot be argued away."""
    p = tmp_path / "c.txt"
    p.write_text("alpha beta\n\ngamma delta\n", encoding="utf-8")
    tok = ChunkTokenizer()

    whole = tokenize_stream(tok, load_corpus(str(p)), 99)
    by_line = tokenize_stream(
        tok, (line for line in open(p, encoding="utf-8") if line.strip()), 99
    )
    # Same bytes, same tokenizer, different token streams: "beta\n\ngamma" is
    # one span whole-file and two spans line-by-line.
    assert whole != by_line


def test_aggregator_refuses_cells_measured_on_different_slices():
    """The C19 guard, on the numbers C19 actually produced.

    0.37% is the real shift between the line-by-line and whole-file readers on
    `1B_baseline`. The check has to fire on a difference that small, because a
    large one would have been noticed without it.
    """
    import pandas as pd

    from analysis.aggregate import assert_comparable

    ok = pd.DataFrame({"model": ["m", "m"], "ppl_ref": [14.707710, 14.707710]})
    assert_comparable(ok)  # identical references: fine

    mixed = pd.DataFrame({"model": ["m", "m"], "ppl_ref": [14.707710, 14.653925]})
    with pytest.raises(ValueError, match="same held-out slice"):
        assert_comparable(mixed)


def test_destroyed_threshold_reads_the_none_arm_and_needs_a_reference():
    """The flag that decides whether a cell may be ranked at all.

    HANDOFF §12 argues for 10x from where the measured ratios cluster. What is
    pinned here is narrower and permanent: the predicate reads the undefended
    `none` arm, compares against each cell's OWN reference, and refuses to guess
    when there is no reference to compare against.
    """
    from analysis.figures import DESTROYED_PPL_RATIO, is_destroyed

    def cell(ppl_ref, ppl_quant):
        return {"ppl_ref": ppl_ref, "ppl_quant": ppl_quant}

    cells = {
        ("m", 8, "per_tensor", "none", 0): cell(14.5, 3721.9),   # 257x
        ("m", 8, "per_token", "none", 0): cell(14.5, 15.2),      # 1.05x
        ("n", 8, "per_tensor", "none", 0): cell(0.0, 100.0),     # no reference
    }
    assert is_destroyed(cells, "m", 8, "per_tensor")
    assert not is_destroyed(cells, "m", 8, "per_token")

    # A ratio is only meaningful against the cell's own reference: 3721.9 is
    # destruction at ppl_ref 14.5 and untouched at ppl_ref 3700.
    near = {("m", 8, "per_tensor", "none", 0): cell(3700.0, 3721.9)}
    assert not is_destroyed(near, "m", 8, "per_tensor")

    assert not is_destroyed(cells, "n", 8, "per_tensor")
    assert not is_destroyed(cells, "absent", 8, "per_tensor")
    assert DESTROYED_PPL_RATIO == 10.0


def test_holdout_sha_separates_streams_the_corpus_hash_cannot():
    assert holdout_sha([1, 2, 3]) == holdout_sha([1, 2, 3])
    assert holdout_sha([1, 2, 3]) != holdout_sha([1, 2, 4])
    # Not a hash of the concatenated digits: [1,23] and [12,3] must differ.
    assert holdout_sha([1, 23]) != holdout_sha([12, 3])
