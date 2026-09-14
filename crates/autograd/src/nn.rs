//! A multilayer perceptron built from `Value`s, and nothing else.
//!
//! Every weight is one `Value`; a forward pass builds a fresh graph each time.
//! Slow and completely transparent — you can print any intermediate number.

use crate::Value;

/// xorshift64 (Marsaglia), so weight init is reproducible without a
/// dependency. Not the `*` variant — that one multiplies the output to fix up
/// its weak low bits; here the low bits are simply thrown away instead.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1) // a zero state would stay zero forever
    }

    /// Uniform in [-1, 1).
    ///
    /// Unscaled — no fan-in term. At 2-4-1 that happens to match Xavier
    /// (±sqrt(6/(fan_in+fan_out)) ≈ ±1 here), so XOR converges. In a deep
    /// stack it would not: activations drift layer over layer until tanh
    /// saturates and the gradients vanish.
    pub fn next_weight(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // Top 53 bits — the ones a f64 mantissa can hold exactly.
        let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        unit * 2.0 - 1.0
    }
}

/// What a neuron puts its weighted sum through.
///
/// The choice is not cosmetic. `Tanh` is smooth and bounded, so a gradient
/// arrives everywhere, but it saturates: far from zero its derivative
/// `1 - tanh²` goes to 0 and learning stalls. `Relu` has derivative exactly 1
/// above zero — no saturation, which is why deep networks use it — but
/// exactly 0 below, so a neuron pushed negative for every input receives no
/// gradient ever again and is dead for good.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Tanh,
    Relu,
    Linear,
}

/// One neuron: w·x + b, put through an activation.
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    activation: Activation,
}

impl Neuron {
    pub fn new(inputs: usize, rng: &mut Rng, activation: Activation) -> Self {
        Neuron {
            weights: (0..inputs).map(|_| Value::new(rng.next_weight())).collect(),
            bias: Value::new(0.0),
            activation,
        }
    }

    pub fn forward(&self, x: &[Value]) -> Value {
        assert_eq!(x.len(), self.weights.len(), "wrong input width");

        // Explicit loop, not a zip/fold chain: this sum is the neuron.
        let mut sum = self.bias.clone();
        for (w, xi) in self.weights.iter().zip(x) {
            sum = &sum + &(w * xi);
        }

        match self.activation {
            Activation::Tanh => sum.tanh(),
            Activation::Relu => sum.relu(),
            Activation::Linear => sum,
        }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.weights.clone();
        p.push(self.bias.clone());
        p
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize, rng: &mut Rng, activation: Activation) -> Self {
        Layer {
            neurons: (0..outputs).map(|_| Neuron::new(inputs, rng, activation)).collect(),
        }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

/// `sizes` is the full width list, input first: `[2, 4, 1]` is 2 inputs, one
/// hidden layer of 4, one output.
pub struct Mlp {
    layers: Vec<Layer>,
}

impl Mlp {
    /// `hidden` is the activation of every layer but the last.
    pub fn new(sizes: &[usize], hidden: Activation, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let last = sizes.len() - 2;
        let layers = (0..sizes.len() - 1)
            .map(|i| {
                // The output layer stays linear — squashing it would cap the
                // range the loss can pull the prediction into.
                let activation = if i == last { Activation::Linear } else { hidden };
                Layer::new(sizes[i], sizes[i + 1], &mut rng, activation)
            })
            .collect();
        Mlp { layers }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut out = x.to_vec();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    /// One SGD step: p <- p - lr * dL/dp.
    ///
    /// `loss.backward()` must have run first; it zeroes the graph itself, so
    /// there is no separate zero_grad() to forget.
    pub fn step(&self, learning_rate: f64) {
        for p in self.parameters() {
            p.set_data(p.data() - learning_rate * p.grad());
        }
    }
}

/// Mean squared error over a batch.
pub fn mse(predictions: &[Value], targets: &[f64]) -> Value {
    assert_eq!(predictions.len(), targets.len());

    let mut sum = Value::new(0.0);
    for (p, &t) in predictions.iter().zip(targets) {
        let diff = p - &Value::new(t);
        sum = &sum + &diff.powf(2.0);
    }
    &sum * &Value::new(1.0 / predictions.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            let x = a.next_weight();
            assert_eq!(x, b.next_weight());
            assert!((-1.0..1.0).contains(&x), "{x} out of range");
        }
    }

    #[test]
    fn parameter_count_matches_the_shape() {
        // 2->4: 4 neurons x (2 weights + 1 bias) = 12. 4->1: 1 x (4 + 1) = 5.
        let net = Mlp::new(&[2, 4, 1], Activation::Tanh, 1);
        assert_eq!(net.parameters().len(), 17);
    }

    /// Trains XOR and returns the final loss.
    fn train_xor(hidden: Activation, seed: u64, epochs: usize, lr: f64) -> (Mlp, f64) {
        let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let targets = [-1.0, 1.0, 1.0, -1.0]; // tanh-friendly, so ±1 not 0/1

        let net = Mlp::new(&[2, 4, 1], hidden, seed);
        let mut last = f64::MAX;

        for _ in 0..epochs {
            // Fresh graph each epoch — the weights persist, the graph does not.
            let predictions: Vec<Value> = inputs
                .iter()
                .map(|row| {
                    let x: Vec<Value> = row.iter().map(|&v| Value::new(v)).collect();
                    net.forward(&x).remove(0)
                })
                .collect();

            let loss = mse(&predictions, &targets);
            loss.backward();
            net.step(lr);
            last = loss.data();
        }
        (net, last)
    }

    #[test]
    fn relu_can_die() {
        // A ReLU neuron whose sum is negative for every input has gradient 0
        // on every input, so it never updates again. Here it is in isolation:
        // one neuron, weights forced negative, inputs non-negative.
        let neuron = Neuron::new(2, &mut Rng::new(7), Activation::Relu);
        for w in &neuron.weights {
            w.set_data(-1.0);
        }

        let x = vec![Value::new(1.0), Value::new(1.0)];
        let y = neuron.forward(&x);
        y.backward();

        assert_eq!(y.data(), 0.0);
        // No gradient reaches the weights — nothing will ever move them back.
        for w in &neuron.weights {
            assert_eq!(w.grad(), 0.0);
        }
    }

    #[test]
    fn relu_also_fits_xor() {
        // Same problem, same step size, different activation. It converges,
        // but only to 1e-14 where tanh reaches 1e-30: ReLU is piecewise
        // linear, so the fit is exact only where the pieces happen to land,
        // while tanh's asymptote lets the output crawl arbitrarily close.
        let (_, loss) = train_xor(Activation::Relu, 1234, 2000, 0.1);
        assert!(loss < 0.01, "relu stalled at {loss}");
    }

    #[test]
    fn xor_converges() {
        // XOR is the classic: not linearly separable, so it cannot be solved
        // without the hidden layer. If this converges, backprop works through
        // more than one layer.
        let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let targets: [f64; 4] = [-1.0, 1.0, 1.0, -1.0];

        let (net, loss_value) = train_xor(Activation::Tanh, 1234, 2000, 0.1);
        assert!(loss_value < 0.01, "loss stalled at {loss_value}");

        // And the signs must actually be right, not just the loss small.
        for (row, &t) in inputs.iter().zip(&targets) {
            let x: Vec<Value> = row.iter().map(|&v| Value::new(v)).collect();
            let y = net.forward(&x).remove(0).data();
            assert_eq!(y.signum(), t.signum(), "{row:?} -> {y}");
        }
    }
}
