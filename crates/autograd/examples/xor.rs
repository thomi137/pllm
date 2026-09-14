// Trains a 2-4-1 network on XOR and prints the loss as it falls.
//
//   cargo run --release -p pllm-autograd --example xor
use pllm_autograd::nn::{Activation, Mlp, mse};
use pllm_autograd::Value;

fn main() {
    // Arguments: [tanh|relu] [learning_rate] [seed]
    let args: Vec<String> = std::env::args().collect();
    let hidden = match args.get(1).map(|s| s.as_str()) {
        Some("relu") => Activation::Relu,
        _ => Activation::Tanh,
    };
    let lr: f64 = args.get(2).map_or(0.1, |s| s.parse().expect("learning rate"));
    let seed: u64 = args.get(3).map_or(1234, |s| s.parse().expect("seed"));

    let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let targets = [-1.0, 1.0, 1.0, -1.0];

    let net = Mlp::new(&[2, 4, 1], hidden, seed);
    println!("{hidden:?}, lr {lr}, seed {seed}");
    println!("parameters: {}", net.parameters().len());

    for epoch in 0..=2000 {
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

        if epoch % 200 == 0 {
            println!("epoch {epoch:>5}  loss {:.3e}", loss.data());
        }
    }

    println!("\nfinal:");
    for (row, &t) in inputs.iter().zip(&targets) {
        let x: Vec<Value> = row.iter().map(|&v| Value::new(v)).collect();
        println!("  {row:?} -> {:+.4}  (target {t:+.0})", net.forward(&x).remove(0).data());
    }
}
