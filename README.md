# pllm
Homegrown LLM to learn how these work.

[![pllm](https://github.com/thomi137/pllm/actions/workflows/rust.yml/badge.svg)](https://github.com/thomi137/pllm/actions/workflows/rust.yml)
[![Rust 1.90.0+](https://img.shields.io/badge/rust-1.87.0+-orange.svg)](https://www.rust-lang.org)

## Purpose
This repo illustrates the basic buildup of a LLM. It is currently a work in progress and intended for learning purposes, not for production use (if anybody has access to a large enough machine, feel free to oblige 😀)

Roughly, it is inspired by a youtube video by 
[Syntax](https://github.com/w3cj/how-llms-work/tree/main) and the corresponding [Repository](https://github.com/w3cj/how-llms-work/tree/main)

Along these lines, I structured the repo but using [Rust](https://rust-lang.org/) and because this is a learning Repo, I also tried out the [AI-Native SDLC Playbook](https://claude.com/blog/the-ai-native-sdlc-playbook) by Anthropic to avoid simple vibe-coding and dive into AI-supported development.

## Crates
### Tokenizer
The tokenizer is basically a BPE Algorithm which chunks bytes for better performance and a small state engine instead of using full blown regex. It was designed to stay dependency-free and as performant as possible. Feel free to submit PRs with suggestions for optimizations.

For information on BPE, see [[1]](#1)

### Autograd
This crate is merely for illustrational purposes to show how backpropagation on a simple neural network can be implemented it serves as a learning example and is not fit for production use. It solve a simple problem:
given $`x`$ and $`y`$, predict $`x \oplus y`$. We chan easily check the correctnes of the neural network because we know the truth table for XOR:

 |$`x`$|$`y`$|$`x \oplus y`$|
 |:---:|:---:|:---:|
 |$`1`$|$`0`$|$`1`$|
 |$`0`$|$`1`$|$`1`$|
 |$`0`$|$`0`$|$`0`$|
 |$`1`$|$`1`$|$`0`$|

We chose this problem because it is the simplest one which is not linearly separatable (i.e. there is no mapping $`f(x, y) = w_1x + w_2y + b`$ which reproduces the truth table above). So we can use this 
to predict the results using a multilayered neural network (a multilayered perceptron) to illustrate backpropagation [[2]](#2) and training of a neural network.

It also documents common pitfalls like ignoring the chain rule from elementary calculus: $`\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)`$ when summing up gradients. 
For more pitfalls and how we prevented them, see the [implementation explanation](context/history/m2-autograd.md).

The seminal paper and a link to the original python library that implements automatic gradients (autograd) can be found here: [[3]](#3)

### Model (Bigram Model)
For training, we use a Bigram Model. As the [plan](context/plan.md) suggests, the dumbest model. however, we will build on this when reaching the transformer stabe. See [[4]](#4) for a brief intro and [[5]](#5) for a more formal treatment.

## Programming Language
- Rust seemed an unusual candidate for AI programming, so we used that over Python.
- We like rust and want low level memory safety ensured at compile time except where we don't want it (`Rc<RefCell<T>>`)
- We love Rust.

Long story short: It is written in Rust, deal with it 😎.

## Corpora
I used a file from the [Leipzig Corpora Portal](https://downloads.wortschatz-leipzig.de/corpora/deu_news_2025_100K.tar.gz). I did not check it in since I do not want to blow up the repo.

Feel free to use your own. But you'll see that even your superfast gaming engine is on its knees very fast 😅.

## References
<a id=1>[1]</a>
[Sennrich, Haddow & Birch (2016) — "Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909)

<a id=2>[2]</a>
[Rumelhart, Hinton & Williams (1986) — "Learning Representations by Back-Propagating Errors"](https://nature.com/articles/323533a0)

<a id=3>[3]</a>
[Maclaurin, Duvenaud & Adams (2015) - "Autograd: Effortless Gradients in Numpy"](https://indico.ijclab.in2p3.fr/event/2914/contributions/6483/subcontributions/180/attachments/6060/7185/automl-short.pdf)

<a id=4>[4]</a>
[Vasvani et al. (2017) - "Attention Is All You Need"](https://arxiv.org/pdf/1706.03762)

<a id=5>[5]</a>
[Phuong & Hutter (2022) - "Formal Algorithms for Transformers"](https://arxiv.org/pdf/2207.09238)