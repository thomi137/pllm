# Plan — M1 to M6

> **Layer:** The ordered work queue. The *why* lives in [intent.md](intent.md),
> design decisions in `design.md`, agent rules in `../CLAUDE.md`. If this file
> and intent.md disagree, intent.md wins — or intent.md gets updated first,
> deliberately.

Milestones are sequential. Each is independently runnable and independently
instructive. Do not start the next before the previous passes its tests.

State on 2026-09-13: the M1 core is in place, tests green (`cargo test`, 3
tests). What remains in M1 is the regex pre-split.

---

## M1 — Byte-level BPE tokenizer

**Crate:** `crates/tokenizer` — no dependencies, pure std.

**Goal:** A tokenizer that round-trips arbitrary UTF-8 losslessly, compresses
measurably, and persists its merge table.

### Definition of done

- [x] Round-trip without training (raw bytes)
- [x] Round-trip after training, with measurable compression
- [x] Merge table serialises and reloads stably
- [x] Regex-style pre-split: `train` and `encode` work chunk by chunk
      ([history/m1-5-pre-split.md](history/m1-5-pre-split.md))
- [ ] Vocabulary inspection: the learned merges can be printed readably

### Current state

`Bpe` in [../crates/tokenizer/src/lib.rs](../crates/tokenizer/src/lib.rs):
IDs 0..=255 are the raw bytes, every merge allocates a new ID from 256 up.
Byte-level means no `<UNK>` — any UTF-8 is encodable. Umlauts start as two
bytes and grow back together through merges.

Only the merges are serialised; `ranks` and `vocab` are reconstructible from
them. Training is naive O(num_merges × corpus_len): legibility over speed,
good enough for a few MB.

### What you will observe in practice

**Whitespace is part of the token.** In GPT-style tokenizers `" der"` (with a
leading space) is a different token from `"der"`. Sounds like a detail, but it
is the reason a trailing space in a prompt noticeably degrades output quality
— you force the model into a token distribution it has barely seen.

**The corpus determines everything.** Train on English and `"Zürich"` falls
apart into four or five tokens, because `ü` is already two bytes and was never
merged. German compounds explode. This is also why non-English costs more
tokens, and therefore more money, on commercial models.

**Numbers are a disaster.** Without special handling, `"1234"` becomes
`"12"+"34"` or `"1"+"234"` depending on corpus frequency — inconsistent. That
is why real tokenizers split digits individually up front via regex. It is
also part of the answer to why LLMs are bad at arithmetic.

**The regex pre-split.** Production tokenizers (GPT-2 onwards) first cut the
text into chunks via regex — words, numbers, punctuation — and let BPE merge
only *within* a chunk. That way no token can ever form across a word boundary
(`"der Hund"` never becomes one token, however frequent). The current
tokenizer deliberately does not do this. When you look at the verbose output
and see merges that swallow a space in the middle, you now know why the
professionals prevent it.

### M1.5 — Retrofit the pre-split (before M2)

Run `train` and `encode` chunk by chunk.

Constraint: `tokenizer` has no dependencies, so **no `regex` crate**. The
splitter is hand-written — a small state machine over `char` classes. That is
no loss but the more instructive route: the GPT-2 regex is nothing but an
enumeration of character classes anyway.

Chunk classes, following GPT-2:

1. optional leading space + run of letters (`" der"`)
2. optional leading space + run of digits, **digits split individually**
3. optional leading space + run of punctuation
4. remaining whitespace (newlines)

Code changes:

- `count_pairs` counts per chunk, not across the whole sequence
- `merge` runs per chunk
- `encode` splits first, then encodes each chunk separately and concatenates
- the serialisation format stays (`mini-bpe v1`), but the merges change in
  content — throw old tables away

Tests for it:

- no learned merge crosses a word boundary, even on a corpus where
  `"der Hund"` is highly frequent
- `"1234"` yields four digit tokens, independent of the corpus
- round-trip stays lossless (the existing tests must stay green)
- compression gets *worse* from the split — that is expected, and belongs in
  the notes as an observation, not something to optimise away

### Open

- Corpus: German or English (intent.md §9). Measure both before M3 fixes one.
  File goes to `data/corpus.txt`.

---

## M2 — Scalar autograd

**Crate:** `crates/autograd` — `ndarray` only.

**Goal:** Have built backpropagation by hand once. Deliberately a dead end:
M2 is not a dependency of M3–M6.

### Definition of done

- [ ] `Value` type with `+`, `*`, `tanh`/`relu`, `pow`
- [ ] Topological sort; `backward()` fills every gradient
- [ ] Gradients match numerical differentiation within 1e-5
- [ ] XOR network (2-4-1) converges to a loss near zero

### Steps

1. `Value` holding data, grad, children, and a local backward closure
2. Operators one at a time, each with its own gradient test
3. Topo-sort, then `backward()` walking it in reverse
4. Hand-rolled mini MLP, SGD loop, XOR

### What you will observe

The chain rule is trivial; the bookkeeping is the problem. Gradients
*accumulate* (`+=`, not `=`), otherwise every fan-out in the graph silently
loses a path. That bug is quiet — the loss still drops, only slower. Hence
the numerical check: it is the only honest control.

### Open

- Does M2 stay in the repo as documentation after M3, or get deleted
  (intent.md §9)?

---

## M3 — Bigram model

**Crate:** `crates/model` — `candle-core`, `candle-nn`.

**Goal:** The full training loop on tensors, with the dumbest possible model.
Architecture only comes after the loop stands.

### Definition of done

- [ ] Load corpus, tokenize, split train/val (e.g. 90/10)
- [ ] Batch sampler: random windows, `[B, T]` input and `[B, T]` target
      shifted by one
- [ ] Embedding table `[vocab, vocab]` as logits, cross-entropy loss
- [ ] Loss drops visibly below the unigram baseline loss
- [ ] `generate()` samples autoregressively and produces (bad) text

### What you will observe

The baseline loss is computed, not guessed: `ln(vocab_size)` for a uniform
distribution, and the entropy of the unigram distribution as the sharper bar.
A bigram model must beat that second bar or it has learned nothing.

The output is gibberish with correct letter statistics — and that is exactly
the lesson: the model sees one token back.

---

## M4 — Transformer

**Crate:** `crates/model`, building on M3.

**Goal:** A decoder-only transformer that clearly beats the bigram.

### Definition of done

- [ ] Token embedding + positional encoding
- [ ] Single-head self-attention, causally masked
- [ ] Multi-head by splitting `C` across `n_head`
- [ ] MLP block (4× expansion), residuals, LayerNorm (pre-norm)
- [ ] Blocks stacked, dropout
- [ ] AdamW, learning-rate warmup
- [ ] Validation loss clearly below M3
- [ ] Shape comments `[B, T, C]` at every change of shape

### Order that surfaces mistakes early

1. Attention for one head, no mask — shape first, physics second
2. Add the causal mask; test: position `t` does not change when you change
   position `t+1` in the input
3. Multi-head, MLP, residual, LayerNorm — one at a time, each with a shape
   test
4. Stack, then train

### What you will observe

The mask is the whole of causality. Without it the loss drops beautifully
fast — because the model is allowed to see the answer. A suspiciously good
loss is a bug signal, not a success.

Pre-norm instead of post-norm is why deep stacks converge at all without
careful warmup.

---

## M5 — CLI

**Crate:** `crates/cli` — `clap`, `indicatif`. The binary is `pllm`.

**Goal:** The three verbs, with no logic in the CLI.

### Definition of done

- [ ] `pllm tokenize` — train on a corpus, write the merge table, print stats
- [ ] `pllm train` — write checkpoints, progress bar, val loss
- [ ] `pllm generate --prompt "..."` — load a checkpoint, print text
- [ ] Checkpoints save and reload; a freshly loaded one generates identically
      at the same seed
- [ ] No decision in the CLI that would have deserved a test

### What you will observe

Wherever you are tempted to write an `if` in the CLI, a function is missing
in `model` or `tokenizer`. The CLI is the litmus test for the seams below it.

---

## M6 — Sampling and experiments

**Crates:** `model`, `cli`.

> Not in intent.md §5, where the milestones end at M5. This one should either
> be added to intent.md or dropped — decide deliberately.

**Goal:** Understand how much of perceived model behaviour is the *sampler*,
and answer the open questions in §9 empirically.

### Definition of done

- [ ] Temperature, top-k and top-p (nucleus) as sampling options
- [ ] Greedy vs sampling comparable on the same checkpoint
- [ ] Corpus comparison German/English: tokens per word, compression, val
      loss side by side
- [ ] A short note per experiment in `design.md` — the number plus the
      conclusion

### What you will observe

Temperature 0 loops on itself, temperature 1.4 dissolves into noise. The same
model looks like two different ones depending on the sampler — which is why
model comparisons without stated sampling parameters are worthless.

The corpus comparison answers §9 with numbers instead of taste.
