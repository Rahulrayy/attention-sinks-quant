"""Stream Python source into `data/code_python.txt` — the second-corpus arm.

Kept for provenance, not for reproduction, on exactly the same terms as
`fetch_fineweb.py`: the committed file is the artefact the results were measured
on, and re-streaming gives a different document set (`LIMITATIONS.md` §18).

**Why code, and why this dataset.** The roster already has two corpora and both
are prose — educational web text (`fineweb_edu.txt`, current) and the project's
own markdown (`repo_corpus_archived.txt`, archived). Two points establish that
the corpus matters; they cannot bound how much, because nothing separates
"different sample" from "different domain". Source code is the furthest
available point from both: a different token distribution, a different
whitespace regime, and the setting where quantized inference is most actually
deployed. If the audit's conclusions survive the move to code they are not
artefacts of prose; if they do not, this says precisely which ones were.

`codeparrot/codeparrot-clean-valid` is used rather than the more obvious
`bigcode/the-stack-smol` (gated, needs credentials this project does not carry)
or `codeparrot/github-code` (a script-based dataset, and `datasets` 5.x dropped
script support). It is the validation split — already deduplicated and filtered
upstream, and never a training target for any checkpoint in the roster, which
matters because three of them are Qwen3 models whose pretraining mix is
undisclosed.

Parameters are held identical to `fetch_fineweb.py` — same character budget,
same stub filter, same joiner — so that corpus is the only thing that differs
between the two grids. That is the whole point of the arm.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent / "code_python.txt"
TARGET_CHARS = 1_200_000
MIN_DOC_CHARS = 500

# Held identical to fetch_fineweb.py: documents shorter than one 256-token
# sequence pad the stream with sequence boundaries rather than text. On code
# this also drops `__init__.py` stubs and licence-only files, which are
# numerous in any GitHub sample and are not language-model evaluation material.


def main(out: Path = OUT, target_chars: int = TARGET_CHARS) -> None:
    from datasets import load_dataset

    for attempt in range(1, 9):
        try:
            ds = load_dataset(
                "codeparrot/codeparrot-clean-valid", split="train", streaming=True
            )
            buf, n = [], 0
            for rec in ds:
                text = (rec.get("content") or "").strip()
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
    raise SystemExit("code fetch failed after 8 attempts")


if __name__ == "__main__":
    main()
