//! Reverse-mode automatic differentiation on single scalars.
//!
//! One `Value` is one number in the computation graph. Every operation builds
//! a new `Value` that remembers its parents and, for each parent, the local
//! derivative d(result)/d(parent) — computed right there in the forward pass,
//! where the numbers are already at hand.
//!
//! `backward()` then walks the graph in reverse and multiplies those local
//! derivatives along the way. That multiply-and-add *is* the chain rule; there
//! is nothing else to it. The work is the bookkeeping: visiting nodes in the
//! right order, and accumulating instead of overwriting when a value was used
//! more than once.
//!
//! Deliberately a dead end — M3 onwards uses candle. See intent.md §5.

use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

pub mod nn;

#[derive(Debug)]
struct Node {
    data: f64,
    /// d(final output)/d(this value). Filled by `backward()`.
    grad: f64,
    /// Each parent, paired with d(this)/d(parent) from the forward pass.
    parents: Vec<(Value, f64)>,
    /// Which operation produced this value. For printing only.
    op: &'static str,
}

/// A scalar in the graph. Cloning shares the same node — that sharing is what
/// makes a value reusable in several places, and why gradients must add up.
#[derive(Clone, Debug)]
pub struct Value(Rc<RefCell<Node>>);

impl Value {
    pub fn new(data: f64) -> Self {
        Value::from_op(data, Vec::new(), "leaf")
    }

    fn from_op(data: f64, parents: Vec<(Value, f64)>, op: &'static str) -> Self {
        Value(Rc::new(RefCell::new(Node { data, grad: 0.0, parents, op })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn op(&self) -> &'static str {
        self.0.borrow().op
    }

    /// Overwrites the number in place, leaving the graph alone. This is how a
    /// weight takes a gradient step without rebuilding anything.
    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    /// d(self)/d(parent) = n * x^(n-1).
    pub fn powf(&self, n: f64) -> Value {
        let x = self.data();
        Value::from_op(x.powf(n), vec![(self.clone(), n * x.powf(n - 1.0))], "powf")
    }

    /// d(tanh)/dx = 1 - tanh(x)^2 — cheap, because tanh(x) is already computed.
    pub fn tanh(&self) -> Value {
        let t = self.data().tanh();
        Value::from_op(t, vec![(self.clone(), 1.0 - t * t)], "tanh")
    }

    /// The derivative at exactly 0 does not exist; 0 is the usual convention.
    pub fn relu(&self) -> Value {
        let x = self.data();
        let local = if x > 0.0 { 1.0 } else { 0.0 };
        Value::from_op(x.max(0.0), vec![(self.clone(), local)], "relu")
    }

    pub fn exp(&self) -> Value {
        let e = self.data().exp();
        Value::from_op(e, vec![(self.clone(), e)], "exp")
    }

    /// Fills `grad` on every value this one depends on.
    ///
    /// Zeroes the whole reachable graph first, so a second `backward()` on a
    /// freshly built graph does not read gradients from the previous step.
    pub fn backward(&self) {
        let order = self.topological_order();

        for v in &order {
            v.zero_grad();
        }
        // d(self)/d(self) = 1 — the seed the chain rule unrolls from.
        self.0.borrow_mut().grad = 1.0;

        // Reverse topological order guarantees a node's own gradient is
        // complete before it is passed further back.
        for v in order.iter().rev() {
            let g = v.grad();
            let node = v.0.borrow();
            for (parent, local) in &node.parents {
                // `+=`, never `=`. A value used in two places receives a
                // gradient along each path, and the paths must add. Getting
                // this wrong is silent: the loss still falls, just wrongly.
                parent.0.borrow_mut().grad += g * local;
            }
        }
    }

    /// Parents before children, so reversing it gives a valid backward order.
    ///
    /// The graph is a DAG, not a tree: a value used twice (`c * c`, a residual
    /// connection) has two children pointing at it. Its gradient is the sum of
    /// both contributions, so it must not be processed until every child has
    /// added its share — which is exactly what reverse topological order
    /// guarantees. `visited` keeps a shared node from being emitted twice,
    /// which would double-count it.
    ///
    /// Iterative rather than recursive: depth follows the longest operation
    /// chain, not the parameter count, and a summed loss over many terms
    /// nests deep enough to blow the stack.
    fn topological_order(&self) -> Vec<Value> {
        let mut order = Vec::new();
        let mut visited: HashSet<*const RefCell<Node>> = HashSet::new();
        // (value, are its parents already on the stack?)
        let mut stack = vec![(self.clone(), false)];

        while let Some((v, expanded)) = stack.pop() {
            if expanded {
                order.push(v);
                continue;
            }
            if !visited.insert(Rc::as_ptr(&v.0)) {
                continue;
            }
            stack.push((v.clone(), true));
            for (parent, _) in &v.0.borrow().parents {
                stack.push((parent.clone(), false));
            }
        }
        order
    }
}

impl From<f64> for Value {
    fn from(x: f64) -> Self {
        Value::new(x)
    }
}

// Operators are implemented on references: `&a + &b` leaves both operands
// usable afterwards, which is what building a graph needs.

impl Add for &Value {
    type Output = Value;
    /// d(a+b)/da = 1, d(a+b)/db = 1 — addition just passes the gradient on.
    fn add(self, rhs: &Value) -> Value {
        Value::from_op(
            self.data() + rhs.data(),
            vec![(self.clone(), 1.0), (rhs.clone(), 1.0)],
            "+",
        )
    }
}

impl Mul for &Value {
    type Output = Value;
    /// d(a*b)/da = b, d(a*b)/db = a — each factor scales the other's gradient.
    fn mul(self, rhs: &Value) -> Value {
        Value::from_op(
            self.data() * rhs.data(),
            vec![(self.clone(), rhs.data()), (rhs.clone(), self.data())],
            "*",
        )
    }
}

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value {
        Value::from_op(-self.data(), vec![(self.clone(), -1.0)], "neg")
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: &Value) -> Value {
        &self.clone() + &(-rhs)
    }
}

impl Div for &Value {
    type Output = Value;
    fn div(self, rhs: &Value) -> Value {
        self * &rhs.powf(-1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central difference: (f(x+h) - f(x-h)) / 2h. The honest check — it knows
    /// nothing about the graph and can only be fooled by bad arithmetic.
    fn numeric_grad(f: impl Fn(f64) -> f64, x: f64) -> f64 {
        let h = 1e-6;
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn addition_passes_the_gradient_through() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let c = &a + &b;
        c.backward();

        assert_eq!(c.data(), -1.0);
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
    }

    #[test]
    fn multiplication_swaps_the_factors() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let c = &a * &b;
        c.backward();

        assert_eq!(a.grad(), -3.0);
        assert_eq!(b.grad(), 2.0);
    }

    #[test]
    fn gradients_accumulate_when_a_value_is_reused() {
        // d(x*x)/dx = 2x. With `=` instead of `+=` this returns x, not 2x —
        // the bug that halves every gradient in a network with shared inputs.
        let x = Value::new(3.0);
        let y = &x * &x;
        y.backward();

        assert_eq!(y.data(), 9.0);
        assert_eq!(x.grad(), 6.0);
    }

    #[test]
    fn unary_ops_match_numeric_differentiation() {
        for &x0 in &[-2.0, -0.5, 0.7, 1.3] {
            let x = Value::new(x0);
            let y = x.tanh();
            y.backward();
            assert!(
                (x.grad() - numeric_grad(f64::tanh, x0)).abs() < 1e-5,
                "tanh at {x0}: {} vs {}",
                x.grad(),
                numeric_grad(f64::tanh, x0)
            );

            let x = Value::new(x0);
            let y = x.exp();
            y.backward();
            assert!((x.grad() - numeric_grad(f64::exp, x0)).abs() < 1e-5);

            let x = Value::new(x0);
            let y = x.powf(3.0);
            y.backward();
            assert!((x.grad() - numeric_grad(|v| v.powf(3.0), x0)).abs() < 1e-5);
        }
    }

    #[test]
    fn a_deep_expression_matches_numeric_differentiation() {
        // f(a, b) = tanh(a*b + a) / (b^2 + 1), with `a` used on two paths.
        let f = |a: f64, b: f64| (a * b + a).tanh() / (b * b + 1.0);

        let a0 = 0.8;
        let b0 = -1.7;
        let a = Value::new(a0);
        let b = Value::new(b0);

        let numerator = (&(&a * &b) + &a).tanh();
        let denominator = &(&b * &b) + &Value::new(1.0);
        let out = &numerator / &denominator;
        out.backward();

        assert!((out.data() - f(a0, b0)).abs() < 1e-12);
        assert!((a.grad() - numeric_grad(|v| f(v, b0), a0)).abs() < 1e-5, "da: {}", a.grad());
        assert!((b.grad() - numeric_grad(|v| f(a0, v), b0)).abs() < 1e-5, "db: {}", b.grad());
    }

    #[test]
    fn relu_is_flat_below_zero() {
        let x = Value::new(-1.5);
        let y = x.relu();
        y.backward();
        assert_eq!(y.data(), 0.0);
        assert_eq!(x.grad(), 0.0); // a dead ReLU learns nothing — by design
    }
}
