# Corpora

These files are committed rather than regenerated. The corpus swap changed the
headline matched-pair reduction by about 5x and flipped two confidence
intervals across zero, so re-streaming a fresh sample is not a reproduction of
these numbers. It is a new experiment. See Section 6 of the main README.

| File | Bytes | sha256 (first 32) | Docs | Used by |
|---|---|---|---|---|
| `fineweb_edu.txt` | 1,213,115 | `9c8501c90509271634a9dc64c86458c5` | 281 | `runs/quant/`, the current grid |
| `code_python.txt` | 1,247,005 | `21f9dee1f17187c2d88c1690faef1206` | 137 | `runs/quant_code/`, the second-corpus arm |
| `repo_corpus_archived.txt` | 232,058 | `c467928c3f1c6a393583686f9b15a5bd` | 907 | `runs/quant_repo_corpus/`, archived |

Files are stored with CRLF line endings, which is how they were written. Every
reader in this repo opens them in **text mode** (`open(path, encoding="utf-8")`),
so Python's universal-newline translation yields LF before tokenization. The
byte-level line endings do not enter the token stream. Do not read them in
binary mode and decode manually; that would change the tokenization.

Character counts as seen by a text-mode reader: 1,202,577, 228,362 and 1,213,872.

## Provenance

`fineweb_edu.txt` comes from `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, split
`train`, streamed in order, documents shorter than 500 characters dropped
(stubs skew short-context statistics), joined with a blank line, stopping at
1.2M characters. Regenerate with `python -m data.fetch_fineweb`, but note that
the exact document set depends on the stream and will differ.

`code_python.txt` comes from `codeparrot/codeparrot-clean-valid`, split `train`,
streamed in order, documents shorter than 500 characters dropped, joined with a
blank line, stopping at 1.2M characters, with every parameter identical to
`fineweb_edu.txt` so that domain is the only thing that differs between the two
grids. Regenerate with `python -m data.fetch_code`, same caveat as above.
Chosen as the furthest available point from the other two, which are both
prose; it is also a *validation* split, so it was never a training target for
any checkpoint in the set. See Section 5.7 of the main README and
`analysis/corpora.py`.

## LAMBADA is fetched, not committed

`quant/lambada.py` reads `EleutherAI/lambada_openai` (config `en`, split
`test`) from the Hub at run time and takes the first N examples in dataset
order. Nothing is written into `data/`.

That is a deliberate difference from the corpora above, and the reason is the
one those files exist for. `fineweb_edu.txt` and `code_python.txt` were
**streamed**: re-running the fetch gives a different document set, so the exact
bytes are part of the result and had to be pinned. LAMBADA's test split is a
fixed, versioned artefact and taking the first N of it in order is
deterministic, so there is nothing a committed copy would protect against that
the hash does not.

Every LAMBADA run records `examples_sha`, a fingerprint of the exact passages
scored, on the same contract as `holdout_sha`: two cells are comparable only if
it matches. If the split is ever revised upstream, the hash changes and the
mismatch is visible rather than silent.

`repo_corpus_archived.txt` is the small in-repo corpus used before the swap.
It is kept deliberately, because the correction that the swap forced rests on
it and deleting it would make that correction unverifiable. Its `ppl_ref` values (136.6 on GPT-2, 54.9 on Qwen3)
are what identified it as not being a language-model evaluation set.

## Licensing

These files are redistributed, not original to this repository, and the MIT
License in `LICENSE` does not extend to them.

- `fineweb_edu.txt`: from `HuggingFaceFW/fineweb-edu`, published under the Open
  Data Commons Attribution License (ODC-By 1.0). Attribution required.
- `code_python.txt`: from `codeparrot/codeparrot-clean-valid`, derived from
  public GitHub repositories. Individual files retain the licences of the
  projects they came from.
- `repo_corpus_archived.txt`: this repository's own markdown, MIT along with
  the rest of the original content here.
