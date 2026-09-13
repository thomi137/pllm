# M1 — Vocabulary inspection

**Date:** 2026-09-13
**Crate:** `crates/tokenizer`
**Status:** done — `cargo test` 10 passed, `cargo clippy` clean. M1 complete.

## What changed

Three methods on `Bpe`, and the example now uses them instead of formatting
tokens by hand.

- `token_repr(id) -> String` — one token for human eyes
- `merge_table() -> String` — the learned merges in rank order
- `longest_tokens(n) -> Vec<(u32, String)>` — learned tokens, longest first

`examples/train_bpe.rs` dropped its inline `.replace(' ', "␣")`, prints the
longest tokens, and writes a second file `<out>.readable.txt` next to the
machine table.

## Why this is not `to_string_repr`

`to_string_repr` is the machine format: `"101 114"`, ids only, nothing but
what `from_string_repr` needs to rebuild the state exactly. Three things it
cannot show, all of which are the reason to look at a merge table at all:

- **Spaces.** `" der"` and `"der"` are different tokens, and in a terminal
  that difference is invisible. `token_repr` renders them `␣der` / `der`.
- **The pair as text.** `256 = 101 + 114` says nothing; `"e" + "r" -> "er"`
  says everything.
- **Broken UTF-8.** `decode()` goes through `from_utf8_lossy`, so a token
  holding only the first byte of `"ü"` prints as `�` — you lose which byte it
  was. `token_repr` prints `<C3>`.

That last case is not an error: byte-level BPE means half a character *is* a
legal token until a merge joins it back up.

## Run on the real corpus

`deu_news_2025_100K` (Leipzig), 11 MB, 500 merges, release build:

```
training:   173.4s
vocab:      756 tokens
compression: 2.35 bytes/token
round-trip: ok
```

### What the merge table shows

The first merges are pure German letter-pair statistics — no words yet:

```
    0     256  "e" + "n" -> "en"
    1     257  "e" + "r" -> "er"
    2     258  "c" + "h" -> "ch"
    3     259  "␣" + "d" -> "␣d"
    4     260  "e" + "i" -> "ei"
```

`en`, `er`, `ch`, `ei` — the German bigram frequency list, learned from
nothing but counting. Rank 3 is already a space attaching itself to `d`,
which becomes `␣der`, `␣die`, `␣das` further down.

By rank 200 the useful pairs are exhausted and it is merging single letters
onto spaces:

```
  199     455  "␣" + "t" -> "␣t"
  201     457  "␣" + "O" -> "␣O"
  202     458  "␣" + "(" -> "␣("
```

That is the vocabulary-size decision made visible: past a few hundred merges
on this corpus, each new token buys almost nothing.

### The longest tokens are all function words

```
␣Prozent  ␣können  ␣werden  ␣weiter  ␣wieder  ␣nicht  ␣über
ischen  ␣einem  ␣einer  ␣einen  ␣unter  ␣haben  ␣wurde  ␣gegen
```

Eight characters is the ceiling at 500 merges. Not one compound noun — they
are individually too rare, even though compounds as a *class* are everywhere.
`␣Prozent` only makes it because news text repeats it constantly.

### And so the compound shatters

```
Der Tokenizer zerlegt Donaudampfschifffahrt in Zürich.
[Der][␣T][ok][en][iz][er][␣z][er][le][gt][␣D][on][au][d][am][pf][sch][i]
[ff][fahr][t][␣in][␣Z][ür][ich][.]
26 tokens for 55 bytes
```

Two things to read off this:

**German is expensive.** 2.35 bytes/token here. English on a comparable
vocabulary lands near 4 — the direct reason non-English costs more tokens,
and more money, on commercial models. `Donaudampfschifffahrt` alone takes 11
tokens.

**`Zürich` splits at `ür`.** The umlaut is two bytes, and the pair survived
long enough to merge — but only because news German repeats it. A rarer
umlaut word would still be sitting as raw bytes.

## The cost side

173 seconds for 500 merges on 11 MB. The loop is O(num_merges × corpus_len)
and rebuilds every chunk on every merge — the naive version, kept on purpose
(intent.md §7: legibility over speed). A real implementation keeps an index
of pair positions and touches only affected chunks. Worth knowing where the
173 seconds went; not worth optimising while it still fits in a coffee break.

## Tests added

- `token_repr_makes_whitespace_visible` — space, newline, lone `0xC3` as
  `<C3>`, unknown id
- `merge_table_is_readable` — one line per merge, ranks start at 0, `"␣Hund"`
  appears with its space marked
- `longest_tokens_are_sorted_long_first` — descending length, compound on top

## M1 is done

All five DoD boxes ticked. Round-trip, compression, serialisation,
pre-split, inspection.

## Open

- Corpus choice (intent.md §9) is now measurable but not measured: the
  English run has not happened. `2.35 bytes/token` is the German number to
  beat.
- Next per plan: M2, scalar autograd.
