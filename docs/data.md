# Datasets, partitions and token streams

## Corpus contracts

| Corpus | Source requested by manuscript | Role and partition rule |
|---|---|---|
| C4 English | `allenai/c4`, `en` | Train on official train. Split official validation documents equally into development/test with a fixed normalized-text hash. |
| FineWeb | `HuggingFaceFW/fineweb`, `sample-10BT` | Hash-partition documents into 80/10/10 train/dev/test before tokenization. Never use nested 10/100/350BT samples as independent splits. |
| SlimPajama | `cerebras/SlimPajama-627B` | Preserve official train/validation/test as train/dev/test. Four merging tasks are GitHub, arXiv, Wikipedia and Stack Exchange. |

The manuscript describes full corpus sizes; these are not the training token budget.
Each adapted expert uses 100M nonpadding target tokens under the general protocol.
The exact dataset commits are not supplied by the paper. `fetch --revision` resolves
a supplied branch/tag/commit to its actual commit and writes `source.json`.

**Source availability:** the original SlimPajama Hub endpoint returned an access
error during verification. If unavailable in your environment, obtain authorized
local shards or explicitly select a verified mirror with `--dataset-id`. Preserve
its identity and revision in your results; a mirror is not assumed byte-identical.
The exporter checks `meta.redpajama_set_name` (or `redpajama_set`) against the source
labels RedPajamaGithub/RedPajamaArXiv/RedPajamaWikipedia/RedPajamaStackExchange. Inspect
a mirror's schema before use. Never silently relabel a different corpus.

## Preparation commands

Use the common base's saved tokenizer for every task:

```bash
python -m tap_merge fetch --corpus fineweb --revision DATA_COMMIT \
  --max-documents 1000000 --output data/fineweb_source
python -m tap_merge clean --corpus fineweb \
  --input data/fineweb_source/documents.jsonl --output data/fineweb_clean
python -m tap_merge pack --input data/fineweb_clean --output data/fineweb_tokens \
  --tokenizer runs/base_gpt2/tokenizer

python -m tap_merge fetch --corpus slimpajama --domain github \
  --revision DATA_COMMIT --max-documents 1000000 --output data/github_source
python -m tap_merge clean --corpus slimpajama \
  --input data/github_source/documents.jsonl --output data/github_clean
python -m tap_merge pack --input data/github_clean --output data/github_tokens \
  --tokenizer runs/base_gpt2/tokenizer
```

Repeat the SlimPajama path with `arxiv`, `wikipedia`, `stackexchange`. `max-documents`
applies to each *matched official split* after domain filtering. For FineWeb its
single train split is partitioned afterwards. The complete validation partitions
may need many documents to supply a million scored targets after filtering.

For local shards, produce JSONL rows containing `text` and an official `split`
(`train`, `validation`, or `test`), then run `clean` and `pack`. Keep source revision
metadata beside the cleaned files; `pack` copies available `source.json` and
`cleaning.json` into `tokens.json`. The exported `source.json` initially lives
beside `documents.jsonl`; copy it into the clean directory if cleaning a custom
source outside the provided workflow. `clean` automatically carries a sibling
`source.json` forward when present.

## Decontamination

The implementation hashes NFKC-normalized whitespace-collapsed document text.
Hashing normalizes identity only; the stored training text remains original.
Salt `tap-merge-v1` fixes the split. Identical texts therefore receive the same
hash split in C4/FineWeb. After partitioning, heldout priority is test > dev > train:
training documents overlapping either heldout split are dropped, and development
documents overlapping test are dropped. Exact duplicates are removed everywhere.

For near-duplicates, a 128-permutation MinHash LSH index (seed 42) proposes matches
at threshold 0.8. Each cross-partition candidate is verified against the actual
word-5-gram set Jaccard similarity; short documents use their entire word sequence
as a single shingle. Matches at similarity >= 0.8 are dropped according to heldout
priority. LSH is probabilistic candidate retrieval, not an exhaustive proof of
absence of overlap. This routine screens selected documents, not the entire web.
`--exact-only` deliberately disables near-duplicate screening for debugging.

SQLite stores document text on disk; the exact hash set and LSH index occupy host
RAM. For complete trillion-token corpora a distributed deduplication tool is needed.
The included path targets the bounded 100M-token experiments. Cleaning one corpus
at a time does not remove cross-corpus overlap between C4/FineWeb task streams;
if globally isolated multitask evaluation is required, add a combined heldout audit
before training and report the stricter protocol. Use the provided command:

```bash
python scripts/audit_cross_task.py --tasks c4=data/c4_clean fineweb=data/fineweb_clean \
  --output data/audited
```

Pack `data/audited/c4` and `data/audited/fineweb` afterwards. Add all four domain
directories to the same command for a global domain audit. Task markers are
preserved while heldout priority is enforced globally. The manuscript does not fully
specify the scope of its cross-corpus duplicate audit.

## Causal packing and target counts

The tokenizer appends an end-of-text token to each document, then concatenates
tokens into a little-endian/native-little-endian uint32 stream (`*.bin`). Intended
platforms are x86/ARM little-endian. Each block reads S+1 raw tokens, supplies the
first S as inputs and the next S as explicit targets. Its successor starts S
tokens later, so the boundary token supplies context and targets are not repeated.
Padding uses input token 0 and target -100. Causal attention prevents padded suffixes
from affecting scored earlier positions. There is no pad-token-ID-based label mask
that could inadvertently discard real end-of-text tokens.

This implementation passes no labels into Hugging Face's model loss; it computes
cross entropy directly from logits and explicitly shifted targets. Hence targets
are shifted exactly once. Cross-document transitions after EOS are scored, as in
ordinary concatenated causal LM packing. Evaluation resets context at block boundaries.

`TokenStream` shuffles block order deterministically for training, traverses blocks
once, and stores permutation/cursor in resume files. It refuses insufficient data
instead of silently repeating it. The final update masks extra labels to reach
exactly the requested target budget. Processed input positions (including masked
suffix/padded positions) are counted separately for throughput.

Development/test evaluation scores up to the configured one million targets. The
actual count is always saved; a smaller available stream is never represented as
a full one-million-token evaluation.
