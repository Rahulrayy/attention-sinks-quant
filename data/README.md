# Corpora

Both files are committed rather than regenerated. The corpus swap changed the
headline matched-pair reduction by ~5× and flipped two CIs across zero
(R3 → R3-rev), so "re-stream a fresh sample" is *not* a reproduction of these
numbers — it is a new experiment. See `LIMITATIONS.md` §18.

| File | Bytes | sha256 (first 32) | Docs | Used by |
|---|---|---|---|---|
| `fineweb_edu.txt` | 1,213,115 | `9c8501c90509271634a9dc64c86458c5` | 281 | `runs/quant/` — **current** grid, R3-rev |
| `repo_corpus_archived.txt` | 232,058 | `c467928c3f1c6a393583686f9b15a5bd` | 907 | `runs/quant_repo_corpus/` — **archived**, R3 |

Files are stored with CRLF line endings, which is how they were written. Every
reader in this repo opens them in **text mode** (`open(path, encoding="utf-8")`),
so Python's universal-newline translation yields LF before tokenization — the
byte-level line endings do not enter the token stream. Do not read them in
binary mode and decode manually; that would change the tokenization.

Character counts as seen by a text-mode reader: 1,202,577 and 228,362.

## Provenance

`fineweb_edu.txt` — `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, split
`train`, streamed in order, documents shorter than 500 characters dropped
(stubs skew short-context statistics), joined with a blank line, stopping at
1.2M characters. Regenerate with `python -m data.fetch_fineweb`, but note that
the exact document set depends on the stream and will differ.

`repo_corpus_archived.txt` — the small in-repo corpus used before the swap.
Kept deliberately: R3-vs-R3-rev rests on it, and deleting it would make the
correction unverifiable. Its `ppl_ref` values (136.6 on GPT-2, 54.9 on Qwen3)
are what identified it as not being a language-model evaluation set.
