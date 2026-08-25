"""Stream FineWeb-Edu into `data/fineweb_edu.txt`.

Kept for provenance, not for reproduction: the committed file is the artefact
the results were measured on, and re-streaming gives a *different* document
set. See `data/README.md` and `LIMITATIONS.md` §18.

The stub filter is not cosmetic. Documents under 500 characters are shorter
than one 256-token sequence, so they pad the stream with sequence boundaries
rather than text and skew every short-context statistic in the grid.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fineweb_edu.txt"
TARGET_CHARS = 1_200_000
MIN_DOC_CHARS = 500


def main(out: Path = OUT, target_chars: int = TARGET_CHARS) -> None:
    from datasets import load_dataset

    for attempt in range(1, 9):
        try:
            ds = load_dataset(
                "HuggingFaceFW/fineweb-edu", name="sample-10BT",
                split="train", streaming=True,
            )
            buf, n = [], 0
            for rec in ds:
                text = (rec.get("text") or "").strip()
                if len(text) < MIN_DOC_CHARS:
                    continue
                buf.append(text)
                n += len(text)
                if n >= target_chars:
                    break
            io.open(out, "w", encoding="utf-8").write("\n\n".join(buf))
            print(f"OK  {len(buf)} docs  {n} chars -> {out}")
            return
        except Exception as exc:  # streaming datasets fail transiently
            print(f"attempt {attempt}: {type(exc).__name__}: {str(exc)[:140]}", flush=True)
            time.sleep(3)
    raise SystemExit("fineweb fetch failed after 8 attempts")


if __name__ == "__main__":
    main()
