# pllm
Homegrown LLM to learn how these work.

[![pllm](https://github.com/thomi137/pllm/actions/workflows/rust.yml/badge.svg)](https://github.com/thomi137/pllm/actions/workflows/rust.yml)


## Purpose
This repo illustrates the basic buildup of a LLM. It is currently a work in progress and intended for learning purposes, not for production use (if anybody has access to a large enough machine, feel free to oblige 😀)

Roughly, it is inspired by a youtube video by 
[Syntax](https://github.com/w3cj/how-llms-work/tree/main) and the corresponding [Repository](https://github.com/w3cj/how-llms-work/tree/main)

Along these lines, I structured the repo but using [Rust](https://rust-lang.org/) and because this is a learning Repo, I also tried out the [AI-Native SDLC Playbook](https://claude.com/blog/the-ai-native-sdlc-playbook) by Anthropic to avoid simple vibe-coding and dive into AI-supported development.

## Crates
### Tokenizer
The tokenizer is basically a BPE Algorithm which chunks bytes for better performance and a small state engine instead of using full blown regex. It was designed to stay dependency-free and as performant as possible. Feel free to submit PRs with suggestions for optimizations.

For information on BPE, see [[1]](#1)

### Corpora
I used a file from the [Leipzig Corpora Portal](https://downloads.wortschatz-leipzig.de/corpora/deu_news_2025_100K.tar.gz). I did not check it in since I do not want to blow up the repo.



Feel free to use your own. But you'll see that even your superfast gaming engine is on its knees very fast 😅.


## References
<a id=1>[1]</a>
[Sennrich, Haddow & Birch (2016) — "Neural Machine Translation of Rare Words with Subword Units"](arxiv.org/abs/1508.07909 )
