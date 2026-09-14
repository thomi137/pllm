# M2 — Scalar autograd

**Date:** 2026-09-14
**Crate:** `crates/autograd` (`pllm-autograd`)
**Status:** done — `cargo test -p pllm-autograd` 11 passed, clippy clean

## What was built

- `Value` — one scalar node: data, grad, parents, op label
- ops `+ * - / neg`, `powf`, `tanh`, `relu`, `exp`
- `backward()` — iterative topological sort, then reverse accumulation
- `nn` — `Rng` (xorshift64), `Activation`, `Neuron`, `Layer`, `Mlp`, `mse`
- `examples/xor.rs` — the 2-4-1 network, loss printed as it falls; takes
  `[tanh|relu] [learning_rate] [seed]`

No dependencies. `ndarray` is allowed here by intent.md §6 but a scalar
engine has nothing to put in an array; pure std keeps every number visible.

## The one design decision

Each operation stores, for every parent, the **local derivative computed
during the forward pass** — not a closure to be called later:

```rust
// d(a*b)/da = b, d(a*b)/db = a
Value::from_op(a.data() * b.data(), vec![(a, b.data()), (b, a.data())], "*")
```

At forward time both numbers are already in hand, so the derivative is a
plain `f64`. `backward()` then needs no knowledge of which operation produced
a node — it multiplies and adds, nothing else. That is the whole chain rule,
visible in five lines:

```rust
for v in order.iter().rev() {
    let g = v.grad();
    for (parent, local) in &v.0.borrow().parents {
        parent.0.borrow_mut().grad += g * local;
    }
}
```

The textbook version stores a backward closure per node. It handles more
cases (ops whose derivative depends on values not captured), but it hides the
arithmetic behind a `dyn Fn`. Nothing needed here goes beyond a stored
number.

## Where the difficulty actually is

Not the calculus. Two pieces of bookkeeping:

**`+=`, never `=`.** A value used in two places collects a gradient along
each path, and the paths must sum. `d(x*x)/dx = 2x`; with `=` you get `x`.
This bug is silent — training still improves, just along a wrong gradient.
`gradients_accumulate_when_a_value_is_reused` is the guard.

**Topological order.** A node's own gradient must be complete before it
passes anything further back, so the graph is sorted parents-first and walked
in reverse. The sort is iterative rather than recursive: a few hundred
weights nest deep enough that recursion is a real stack risk.

## Verification

Central differences, `(f(x+h) - f(x-h)) / 2h` with `h = 1e-6`, tolerance
1e-5. The numeric check knows nothing about the graph, so it can only be
fooled by arithmetic that is genuinely right.

Checked per-op (`tanh`, `exp`, `powf` at four points each) and on a deep
expression where one input feeds two paths:

```
f(a, b) = tanh(a*b + a) / (b² + 1)
```

Both `df/da` and `df/db` match. The two-path case is the one that catches a
missing accumulation.

## XOR

2-4-1, tanh hidden, linear output, MSE against ±1, SGD at lr 0.1, seed 1234,
17 parameters.

```
epoch     0  loss 1.029e0
epoch   200  loss 1.466e-3
epoch   400  loss 1.684e-9
epoch   600  loss 1.788e-15
epoch  1000  loss 2.014e-27
epoch  1200  loss 2.632e-30
epoch  2000  loss 2.715e-30
```

Three things to read off this:

**XOR needs the hidden layer.** It is not linearly separable — a single
neuron cannot fit it at any learning rate. Convergence here is the proof that
gradients cross a layer boundary correctly.

**The decay is geometric, then stops.** Each 200 epochs cuts the loss by
roughly six orders of magnitude, until it floors at 2.7e-30 around epoch
1200. That floor is f64 saturation: `tanh` is asymptotic to ±1, so the
prediction closes on the target but never arrives, and the squared remainder
falls below what a double can represent.

**A tiny problem converges absurdly hard.** Loss 1e-30 is not a good sign in
general; it means the network has memorised four points with 17 parameters.
On real data that is exactly what overfitting looks like.

## tanh vs ReLU

The activation started as a `bool` on `Neuron` with `tanh` hardcoded, which
made `relu` unreachable from `Mlp` even though it was implemented and
gradient-tested. It is now an `Activation` enum — `Tanh`, `Relu`, `Linear` —
chosen per network, output layer always `Linear`.

Same problem, same seed, same lr 0.1, 2000 epochs:

| activation | epoch 200 | epoch 2000 |
|------------|-----------|------------|
| tanh | 1.466e-3 | 2.715e-30 |
| relu | 1.240e-1 | 2.081e-14 |

A prediction made beforehand, and wrong: that ReLU would need a smaller step
than tanh because its output is unbounded. It does not — at lr 0.02 it is
simply slower (1.9e-4 by epoch 1000). What differs is the **floor**. ReLU is
piecewise linear, so the fit is exact only where the pieces land; tanh is
asymptotic to ±1, so the output keeps crawling closer until f64 runs out of
mantissa. Six orders of magnitude of "better" that mean nothing in practice.

### Dead ReLU

The reason the choice deserves to be visible rather than a `bool`:

```rust
let neuron = Neuron::new(2, &mut Rng::new(7), Activation::Relu);
for w in &neuron.weights { w.set_data(-1.0); }
// forward on [1, 1] -> 0.0, and every weight gradient is exactly 0.0
```

`relu` has derivative exactly 0 below zero. A neuron whose weighted sum is
negative for *every* input in the data receives no gradient on any of them,
so nothing will ever move it back — it is dead for the rest of training, and
the loss curve gives no hint. tanh cannot do this: it saturates, so the
gradient shrinks towards 0 but never reaches it.

That trade is why deep networks use ReLU anyway: no saturation means no
vanishing gradient through many layers, and a few dead units are an
acceptable price. It is also why initialisation and learning rate stop being
details in M4. `relu_can_die` pins the behaviour.

## Cost

Every epoch rebuilds the whole graph: 17 parameters, 4 samples, several
hundred `Rc<RefCell<…>>` allocations, all for four numbers. 2000 epochs still
run instantly, but this is the lesson M2 exists to deliver — scalar autograd
does not scale, one `Value` per number is the wrong granularity, and M3
switches to tensors on candle.

## Tests

`lib.rs`: addition passes gradient through, multiplication swaps factors,
accumulation on reuse, unary ops vs numeric, deep expression vs numeric,
relu flat below zero.
`nn.rs`: rng deterministic and in range, parameter count matches shape, xor
converges with tanh (loss < 0.01 *and* all four signs correct), xor converges
with relu, relu can die.

## Open

- Does M2 stay in the repo after M3, or get deleted (intent.md §9)? The
  argument for keeping it: this crate is the only place the chain rule is
  visible. Against: it is dead code the moment candle arrives.
- Next per plan: M3, bigram model on tensors.
