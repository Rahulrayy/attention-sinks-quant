"""Streaming tokenization to a uint16 memmap.

100M tokens as uint16 is 200 MB on disk. Stream it — a non-streaming FineWeb
load will eat 16 GB of RAM and swap the machine (plan §10).

Also trains the 16k BPE on the project's own corpus (trap §9.6).
"""

from __future__ import annotations


def train_tokenizer(corpus_iter, vocab_size: int = 16384, out_path: str = "runs/tokenizer.json"):
    raise NotImplementedError


def stream_tokenize(dataset: str, tokenizer, n_tokens: int, out_path: str):
    """load_dataset(..., streaming=True) -> incremental uint16 memmap writes."""
    raise NotImplementedError
