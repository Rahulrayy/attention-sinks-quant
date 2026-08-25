"""Corpus slicing and activation-range collection.

Five DISJOINT calibration sets are the Track-A analogue of seeds (plan §6).
Disjointness is load-bearing: overlapping draws understate between-seed variance
and the bootstrap CIs come out fraudulently tight, which would let this project
commit the exact sin it exists to audit.

The held-out evaluation slice is fixed and must never appear in any calibration
draw. It is carved off the FRONT of the corpus before any draw is allocated, so
overlap is structurally impossible rather than merely checked for — and
``assert_disjoint`` checks anyway, because a layout invariant that is never
tested is a comment.

This module owns the slicing for BOTH tracks of Track-A work: sinks.measure
imports it rather than reimplementing the draw arithmetic, so the disjointness
guarantee exists in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import torch


@dataclass
class CorpusSlices:
    """A held-out evaluation slice plus N disjoint calibration draws."""

    holdout: list[int]
    draws: list[list[int]]

    @property
    def n_draws(self) -> int:
        return len(self.draws)

    def draw(self, seed: int) -> list[int]:
        if not 0 <= seed < len(self.draws):
            raise IndexError(
                f"calibration seed {seed} out of range; {len(self.draws)} draws were built"
            )
        return self.draws[seed]


def tokenize_stream(tokenizer, text_source: Iterable[str], n_tokens: int) -> list[int]:
    """Accumulate ``n_tokens`` ids from a text iterable.

    Special tokens are not added here. BOS handling is a per-batch decision
    (trap §9.4) and must stay visible at the point where batches are built, not
    be baked invisibly into the token stream.
    """
    ids: list[int] = []
    for text in text_source:
        if not text:
            continue
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(ids) >= n_tokens:
            break
    return ids[:n_tokens]


def build_slices(
    ids: list[int],
    *,
    n_draws: int,
    tokens_per_draw: int,
    eval_tokens: int,
) -> CorpusSlices:
    """Carve the holdout off the front, then allocate contiguous disjoint draws."""
    needed = eval_tokens + n_draws * tokens_per_draw
    if len(ids) < needed:
        raise ValueError(
            f"corpus has {len(ids)} tokens but the layout needs {needed} "
            f"({eval_tokens} held out + {n_draws} x {tokens_per_draw}). Draws must "
            "be disjoint, so a short corpus cannot be recycled across seeds."
        )

    holdout = ids[:eval_tokens]
    draws = [
        ids[eval_tokens + i * tokens_per_draw : eval_tokens + (i + 1) * tokens_per_draw]
        for i in range(n_draws)
    ]
    slices = CorpusSlices(holdout=holdout, draws=draws)
    assert_disjoint(slices)
    return slices


def assert_disjoint(slices: CorpusSlices) -> None:
    """Verify no calibration draw overlaps the holdout or another draw.

    Checked by index range rather than by token value: token ids repeat all over
    a corpus, so a value-based check would report overlap on any two slices that
    merely share vocabulary.
    """
    spans = []
    offset = 0
    for name, chunk in [("holdout", slices.holdout)] + [
        (f"draw{i}", d) for i, d in enumerate(slices.draws)
    ]:
        spans.append((name, offset, offset + len(chunk)))
        offset += len(chunk)

    for i, (name_a, start_a, end_a) in enumerate(spans):
        for name_b, start_b, end_b in spans[i + 1 :]:
            if start_a < end_b and start_b < end_a:
                raise ValueError(f"{name_a} overlaps {name_b}: [{start_a},{end_a}) vs [{start_b},{end_b})")


def to_batches(
    ids: list[int],
    seq_len: int,
    *,
    prepend_bos: bool = False,
    bos_token_id: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield (1, seq_len) batches. Batch size 1 for all Track-A work (trap §9.2).

    With ``prepend_bos`` the BOS replaces the last token rather than extending
    the sequence, so both BOS policies score the same number of positions and
    the two runs stay comparable.
    """
    if prepend_bos and bos_token_id is None:
        raise ValueError(
            "prepend_bos requested but the tokenizer has no bos_token_id. Run "
            "this model without BOS and say so in the results rather than "
            "substituting some other token."
        )

    for i in range(0, len(ids) - seq_len + 1, seq_len):
        chunk = ids[i : i + seq_len]
        if prepend_bos:
            chunk = [bos_token_id] + chunk[:-1]
        yield torch.tensor([chunk], dtype=torch.long)


def collect_ranges(model, batches: Iterable[torch.Tensor], *, device: str = "cuda") -> dict[str, float]:
    """Run a calibration pass and freeze each QuantLinear's observed range.

    The model must already be patched. Ranges are observed with the fp16
    exception mask applied, so a calibration run belongs to ONE cell of the
    exception grid — the range that a per-tensor scale should use genuinely
    differs depending on which tokens are being held out of it.
    """
    from .patch import QuantLinear, commit_observed_ranges, set_observing

    n = set_observing(model, True)
    if n == 0:
        raise RuntimeError(
            "collect_ranges found no QuantLinear layers. Patch the model with "
            "patch_model before calibrating, or the returned ranges are empty "
            "and every per_tensor cell silently falls back to dynamic scaling."
        )

    seen = 0
    with torch.no_grad():
        for batch in batches:
            model(batch.to(device), use_cache=False)
            seen += 1

    set_observing(model, False)

    if seen == 0:
        raise RuntimeError("calibration saw zero batches; check the draw size against seq_len")

    ranges = commit_observed_ranges(model)
    if not ranges:
        raise RuntimeError(
            "calibration observed only zero-magnitude activations across "
            f"{seen} batches, which cannot be right. Check that the batches "
            "reach the device and that the model is not returning early."
        )
    return ranges


def reset_ranges(model) -> None:
    """Clear observed and committed ranges so the next grid cell starts clean.

    Forgetting this between cells would leak one exception arm's calibration
    into the next, which is the kind of bug that produces a small consistent
    bias rather than an obvious failure.
    """
    from .patch import QuantLinear

    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.observed_amax = 0.0
            module.static_amax = None
