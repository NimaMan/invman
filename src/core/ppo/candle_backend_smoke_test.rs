//! # candle backend smoke test
//!
//! ## Purpose
//! De-risk the candle autodiff backend BEFORE the full PPO trainer is built.
//! This module exercises the exact differentiable-training path PPO depends on:
//! trainable parameters in a `VarMap`, a small MLP built from `candle_nn::Linear`
//! layers, a forward pass (matmul + tanh), a scalar loss, reverse-mode autodiff
//! via `Tensor::backward()` (driven by `Optimizer::backward_step`), and an
//! `AdamW` parameter update. If candle cannot compile in this pyo3 cdylib crate,
//! or its autodiff/optimizer cannot drive a loss down, we learn it here in
//! seconds instead of after writing the whole trainer.
//!
//! ## Algorithm (memorization regression)
//! 1. Build a fixed, deterministic input batch `X` (shape `[batch, in_dim]`) and
//!    a fixed target `Y` (shape `[batch, out_dim]`), both from a closed-form
//!    `sin`-based generator so the test is reproducible with no RNG.
//! 2. Build a 2-layer MLP `f(X) = L2(tanh(L1(X)))` whose weights/biases are
//!    `candle` `Var`s registered in a `VarMap`.
//! 3. Loss = mean-squared error `mean((f(X) - Y)^2)`.
//! 4. Record `loss_before`, run `steps` AdamW updates, record `loss_after`.
//! 5. The test asserts the loss strictly decreases and drops by >10x — i.e. the
//!    gradients flow through `Linear`/`tanh`/`sqr`/`mean` and Adam updates the
//!    `Var`s. This is precisely the machinery the PPO actor-critic update needs.
//!
//! Run: `cargo test --features ppo candle_autodiff` (see crate test-link notes
//! for the libpython link flags required by the pyo3 cdylib at test time).

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

/// Deterministic `rows x cols` matrix as a flat row-major `Vec<f32>`.
/// Closed-form (no RNG) so the smoke test is fully reproducible.
fn deterministic_matrix(rows: usize, cols: usize, salt: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let x = ((i as f32) * 0.37 + (j as f32) * 0.11 + salt).sin();
            values.push(x);
        }
    }
    values
}

/// Fit a 2-layer tanh MLP to a fixed target by `steps` AdamW updates on MSE.
/// Returns `(loss_before, loss_after)`. Exercises the full candle autodiff +
/// optimizer path used by the PPO actor-critic update.
pub fn run_candle_backend_smoke(steps: usize) -> Result<(f32, f32)> {
    let device = Device::Cpu;
    let (batch, in_dim, hidden, out_dim) = (8usize, 4usize, 16usize, 3usize);

    let inputs = Tensor::from_vec(
        deterministic_matrix(batch, in_dim, 0.0),
        (batch, in_dim),
        &device,
    )?;
    let target = Tensor::from_vec(
        deterministic_matrix(batch, out_dim, 1.5),
        (batch, out_dim),
        &device,
    )?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let l1: Linear = linear(in_dim, hidden, vb.pp("l1"))?;
    let l2: Linear = linear(hidden, out_dim, vb.pp("l2"))?;

    let forward = |x: &Tensor| -> Result<Tensor> {
        let hidden_act = l1.forward(x)?.tanh()?;
        l2.forward(&hidden_act)
    };
    let mse = |pred: &Tensor| -> Result<Tensor> { (pred - &target)?.sqr()?.mean_all() };

    let loss_before = mse(&forward(&inputs)?)?.to_scalar::<f32>()?;

    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        },
    )?;
    for _ in 0..steps {
        let loss = mse(&forward(&inputs)?)?;
        optimizer.backward_step(&loss)?;
    }

    let loss_after = mse(&forward(&inputs)?)?.to_scalar::<f32>()?;
    Ok((loss_before, loss_after))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_autodiff_and_adam_reduce_loss() {
        let (before, after) = run_candle_backend_smoke(300).expect("candle smoke run failed");
        assert!(
            after < before,
            "loss must decrease (autodiff+Adam): before={before} after={after}"
        );
        assert!(
            after < before * 0.1,
            "Adam should drive MSE down >10x: before={before} after={after}"
        );
    }
}
