# Intent — `pllm`

> **Layer:** Intent (the *why* and *what*). Design decisions live in `design.md`,
> the ordered work queue lives in `tasks.md`, agent operating rules live in
> `CLAUDE.md`. This file is the source of truth for scope. If code and this file
> disagree, this file wins — or this file gets updated first, deliberately.

---

## 1. Purpose

Build a small, working GPT-style language model **from scratch in Rust** —
tokenizer, training loop, inference, and a CLI — for the sole purpose of
understanding how LLMs work end to end.

The deliverable is understanding. The artifact is the proof.

## 2. Success criteria

This project is done when all of the following hold:

1. A byte-level BPE tokenizer trains on a local corpus, round-trips arbitrary
   UTF-8 losslessly, and persists/loads its merge table.
2. A hand-written scalar autograd engine trains an XOR network to convergence,
   with gradients verified against numerical differentiation.
3. A decoder-only transformer trains on a few MB of text on CPU and reaches a
   validation loss clearly below the unigram baseline.
4. `pllm generate --prompt "..."` produces text with recognisable word
   structure of the training language.
5. Every component above is explainable by the author without reading the code.

Criterion 5 is the real one. The rest are instrumentation.

## 3. Non-goals

Explicitly out of scope. Do not propose, scaffold, or "helpfully add" these:

- Competitive quality, benchmark scores, or any comparison to real models.
- Distributed or multi-GPU training.
- Fine-tuning, RLHF, instruction tuning, quantization, LoRA.
- A web UI, HTTP server, or REST API.
- Loading pretrained weights from HuggingFace or elsewhere.
- Production concerns: auth, telemetry, containers, deployment, CI beyond
  `cargo test` and `cargo clippy`.
- Premature performance optimisation. Correctness and legibility outrank speed
  everywhere except where training becomes practically impossible.

## 4. Users

A single user: the author, learning. There are no other stakeholders, no
backwards-compatibility obligations, and no external API surface to keep stable.

Consequence: breaking changes are free. Prefer rewriting a module cleanly over
maintaining a migration path.

## 5. Scope — milestones

Each milestone is independently runnable and independently instructive. Do not
start the next before the previous passes its tests.

| # | Milestone | Definition of done |
|---|-----------|--------------------|
| M1 | Byte-level BPE tokenizer | Round-trip tests pass; measurable compression; merge table serialises |
| M2 | Scalar autograd engine | XOR network converges; gradients match numerical check within 1e-5 |
| M3 | Bigram model | Full training loop on tensors; loss decreases; generates (bad) text |
| M4 | Transformer | Multi-head causal attention, MLP, LayerNorm, residuals, AdamW; val loss beats M3 |
| M5 | CLI | `tokenize`, `train`, `generate` subcommands; checkpoints save and reload |

M2 is deliberately thrown away after it teaches its lesson. It is not a
dependency of M3–M5.

## 6. Architecture intent

A Cargo workspace of four crates, in dependency order:

```
crates/tokenizer   no dependencies at all — pure std
crates/autograd    ndarray only — standalone learning artifact, dead end by design
crates/model       candle-core, candle-nn — transformer + training loop
crates/cli         clap, indicatif — thin orchestration over the above
```

The rule that matters: **`cli` contains no logic worth testing.** If a decision
is being made in `cli`, it belongs in `model` or `tokenizer`.

Rationale for using `candle` in M4 rather than continuing hand-written autograd
is recorded in `design.md`. Do not revisit it here.

## 7. Constraints

- **Rust stable.** No nightly features.
- **CPU-first.** GPU via candle's optional backends is allowed but never
  required; the project must remain runnable on a laptop.
- **Small data.** Corpora of a few MB. If something needs more data to be
  demonstrable, it is out of scope.
- **Legibility over cleverness.** Code is read far more than run here. Name
  tensor dimensions in comments (`[B, T, C]`) at every shape change.
- **No hidden magic.** Prefer an explicit loop over a clever iterator chain when
  the loop shows the maths more plainly.

## 8. Working agreement with the agent

- Explore and plan before writing code. State the plan; wait for confirmation on
  anything touching more than one file.
- Deliver **complete files**, not fragments or diffs-in-prose.
- Write the test before or alongside the implementation. Untested numerical code
  is assumed wrong.
- When a maths step is non-obvious, add a comment explaining the *shape* and the
  *why*, not the *what*.
- Do not add dependencies not listed in §6 without asking first.
- Flag disagreement with anything in this file rather than silently working
  around it.

## 9. Open questions

- Which corpus? (German vs English materially changes tokenizer behaviour —
  worth trying both.)
- Character-level positional encoding: learned or sinusoidal? Learned is
  simpler; sinusoidal is more instructive. Decide in `design.md`.
- Does M2 stay in the repo as documentation, or get deleted after M3?

## 10. Glossary

- **Token** — an integer index into the vocabulary. Carries no numeric meaning.
- **Embedding** — the learned lookup table mapping token IDs to vectors. It is
  the first layer of the network, not a preprocessing step.
- **Logits** — unnormalised scores over the vocabulary, shape `[B, T, vocab]`.
- **Merge** — one BPE rule joining two adjacent IDs into a new one. Rank = the
  order it was learned; lower rank applies first.