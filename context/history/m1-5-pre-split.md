# M1.5 — Regex-style pre-split

**Date:** 2026-09-13
**Crate:** `crates/tokenizer`
**Status:** done — `cargo test` 7 passed, `cargo clippy` clean

## What changed

`train` and `encode` no longer see the text as one long byte sequence. Both
first cut it into chunks, and BPE merges only *within* a chunk. A token can
therefore never span a word boundary.

- `split_chunks(&str) -> Vec<&str>` — new, public (the CLI will want to show
  it, and it is the thing worth looking at by hand)
- `train` works on `Vec<Vec<u32>>`, one sequence per chunk
- `encode` splits, then calls the new private `encode_chunk` per chunk
- `count_pairs` became `count_pairs_into`, accumulating across chunks into one
  shared `stats` map
- merge selection now breaks ties deterministically

## Why no `regex` crate

`tokenizer` has no dependencies and keeps none (intent.md §6). GPT-2's
pre-tokenizer regex is an enumeration of character classes and nothing more,
so it is written out as a state machine over `char`s. Four classes, each
allowed to keep one leading space:

1. optional space + run of letters
2. optional space + a single digit
3. optional space + run of punctuation/symbols
4. run of whitespace

One rule was not obvious and cost a failing test: in a run of spaces, the
*last* space belongs to the word that follows. `"  x"` splits into `" "` and
`" x"`, not `"  "` and `"x"`. GPT-2 encodes this as the lookahead
`\s+(?!\S)`. Without it, a word after double spacing gets a token shape the
model has rarely seen.

## The determinism bug this surfaced

Merge selection was `max_by_key(|(_, &c)| c)` over a `HashMap`. `HashMap`
iteration order is randomised per process, so whenever two pairs tied on
count, two runs on the same corpus learned different merge tables. Chunking
made ties the common case rather than the exception, since chunks are short.

Fixed by ordering on `(count, Reverse(pair))`: highest count first, smallest
pair id on a tie. `training_is_deterministic` trains twice and compares the
serialised tables.

## Measurements

Corpus: one German sentence set repeated 200×, encoding the single sentence
(119 bytes).

| merges | tokens | bytes/token | vocab |
|--------|--------|-------------|-------|
| 50 | 48 | 2.48 | 306 |
| 200 | 26 | 4.58 | 329 |
| 500 | 26 | 4.58 | 329 |

Two things to read off this:

**Training saturates.** 200 and 500 merges give the identical table — the
loop stops once no pair occurs twice. With chunking, an exhausted corpus is
reached far sooner: 500 requested merges yielded 73 learned ones. Before the
split, merges across word boundaries kept the counts alive artificially, so
the vocabulary looked like it was still learning when it was really only
memorising phrases.

**Compression is bought, not given.** `bytes/token` rises steeply while whole
words are still being assembled, then stops dead. Any number past the
saturation point is a lie about vocabulary size.

## Observed

The split is visible in the chunk list:

```
["Der", " Donaudampfschifffahrtskapitaen", " faehrt", " 1", "2", "3", "4",
 " Kilometer", ".", " Die"]
```

- The compound is one chunk, so BPE can merge it internally as deep as the
  corpus allows — this is where German gets expensive: one chunk, many
  merges needed, and each unseen compound falls back to fragments.
- `1234` is four chunks. The leading space rides on the first digit only.
  Whatever the corpus frequency, the number encodes the same way in every
  context — the prerequisite for arithmetic to be learnable at all, though
  not a guarantee of it.
- Punctuation separates cleanly, so `"Welt"` and `"Welt!"` share a token.

## Tests added

- `chunks_keep_the_leading_space` — the three splitter cases above
- `no_merge_crosses_a_word_boundary` — trains on a corpus where `"der Hund"`
  is the most frequent sequence and asserts no token carries a space anywhere
  but its first byte
- `digits_stay_single_tokens` — `"1234"` encodes to 4 tokens after training
  on a corpus full of `"1234"`
- `training_is_deterministic` — two trainings, identical merge tables

## Open after this step

- Vocabulary inspection (last M1 checkbox): print learned merges readably.
- Corpus choice, German vs English (intent.md §9) — now measurable, since
  `bytes/token` is the number that decides it.
