# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Document layering

This repo separates four layers. Respect the boundaries — put changes in the right file:

| File | Holds | Status |
|------|-------|--------|
| [context/intent.md](context/intent.md) | The *why* and *what*: purpose, success criteria, non-goals, milestones, constraints | Exists — **source of truth for scope** |
| `context/design.md` | The *how*: design decisions and their rationale | Referenced by intent.md, not yet written |
| `context/tasks.md` | The ordered work queue | Referenced by intent.md, not yet written |
| `CLAUDE.md` (this file) | Agent operating rules | — |

**Read [context/intent.md](context/intent.md) in full before proposing work, adding a dependency, or starting a milestone.** If code and intent.md disagree, intent.md wins — or intent.md gets updated first, deliberately. The rules below are the subset that must apply on every turn without re-reading it.

## Project

`pllm` — a GPT-style LLM built from scratch in Rust (tokenizer, autograd, transformer, CLI) for the sole purpose of understanding how LLMs work. **The deliverable is understanding; the artifact is the proof.** The binding success criterion is that the author can explain every component without reading the code — which means legibility outranks correctness-of-abstraction, and both outrank speed.

Single user (the author, learning). No backwards-compatibility obligations, no external API surface. Breaking changes are free — prefer rewriting a module cleanly over maintaining a migration path.

## Operating rules

- **Explore and plan before writing code.** State the plan; wait for confirmation on anything touching more than one file.
- **Deliver complete files**, not fragments or diffs-in-prose.
- **Write the test before or alongside the implementation.** Untested numerical code is assumed wrong.
- **Do not add dependencies** beyond the per-crate lists below without asking first.
- **Name tensor dimensions in comments (`[B, T, C]`) at every shape change.** Where a maths step is non-obvious, comment the *shape* and the *why*, never the *what*.
- **No hidden magic.** Prefer an explicit loop over a clever iterator chain when the loop shows the maths more plainly.
- **Flag disagreement** with intent.md rather than silently working around it.
- **Do not scaffold non-goals.** See intent.md §3 — no web UI/server/API, no pretrained weights, no fine-tuning/RLHF/quantization/LoRA, no distributed training, no deployment or telemetry, no premature optimisation. These are listed because they are the tempting "helpful additions"; adding one is a regression.
- Milestones are sequential and each is independently runnable. Do not start the next before the previous passes its tests.

## Architecture

A Cargo workspace of four crates, in dependency order. The dependency lists are the allowlist:

```
crates/tokenizer   no dependencies at all — pure std
crates/autograd    ndarray only — standalone learning artifact, a dead end by design
crates/model       candle-core, candle-nn — transformer + training loop
crates/cli         clap, indicatif — thin orchestration over the above
```

Two rules that shape where code goes:

- **`cli` contains no logic worth testing.** If a decision is being made in `cli`, it belongs in `model` or `tokenizer`.
- **`autograd` is thrown away after it teaches its lesson.** It is not a dependency of the transformer work. Do not wire it into `model`.

The choice to switch to `candle` rather than continue the hand-written autograd is settled; its rationale belongs in `design.md`. Do not reopen it.

## Constraints

- **Rust stable.** No nightly features.
- **CPU-first.** GPU via candle's optional backends is allowed but never required — it must stay runnable on a laptop.
- **Small data.** Corpora of a few MB. If something needs more data to be demonstrable, it is out of scope.

## Commands

The workspace does not exist yet — no `Cargo.toml` has been committed. Once it does:

```sh
cargo build
cargo test                      # workspace-wide
cargo test -p tokenizer         # one crate
cargo test <substring>          # one test by name
cargo clippy
cargo fmt
cargo mutants                   # .gitignore anticipates mutation testing
```

CI is scoped to `cargo test` and `cargo clippy` only (intent.md §3).

## Open questions

Unresolved in intent.md §9 — do not settle these unilaterally: choice of corpus (German vs English materially changes tokenizer behaviour); learned vs sinusoidal positional encoding; whether `autograd` stays in the repo as documentation after the bigram milestone.
