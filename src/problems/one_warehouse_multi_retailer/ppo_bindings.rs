//! # `ppo_bindings` — Python entry point for the OWMR PPO trainer
//!
//! ## Objective
//! Expose the reusable Rust PPO trainer (`core::ppo`) applied to the OWMR
//! instance_14 environment as a single pyo3 function, so Python can train a PPO
//! policy with `training_method="ppo"` the same way it drives CMA-ES — without
//! any PyTorch. The function runs the whole training loop in Rust (releasing the
//! GIL) and returns the held-out result plus the gate anchor and learning curve.
//!
//! ## Returned dict
//! - `gate_holdout_cost`: echelon base-stock gate cost on the holdout (the
//!   in-protocol anchor PPO is compared against; ~50,445).
//! - `best_holdout_cost`, `final_holdout_cost_mean`, `final_holdout_cost_std`:
//!   the best-checkpoint greedy PPO cost on the holdout.
//! - `curve`: per-eval `{iter, phase, train_cost, holdout_greedy_cost}` points.
//!
//! Hyperparameter defaults are the validated 5-seed config from
//! `ppo_baseline/train_ppo_5seed.py`. Report mean±std over >=5 `seed` values.
//!
//! Feature-gated behind `ppo` (the candle backend); only registered when built
//! with `--features ppo`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;

use crate::core::ppo::environment::PpoVecEnv;
use crate::core::ppo::ppo_trainer::{train_ppo, PpoConfig};
use crate::problems::one_warehouse_multi_retailer::ppo_environment::{
    gate_holdout_mean_cost, OwmrPpoEnv,
};

#[pyfunction]
#[pyo3(signature = (
    seed = 0,
    iters = 60,
    train_paths = 384,
    eval_paths = 1024,
    hidden = 128,
    lr = 1.2e-4,
    gamma = 1.0,
    lam = 0.95,
    clip = 0.15,
    ppo_epochs = 5,
    minibatch = 4096,
    vf_coef = 0.5,
    ent_coef = 0.001,
    max_grad_norm = 0.5,
    reward_scale = 1000.0,
    bc_epochs = 120,
    bc_paths = 512,
    bc_lr = 1e-3,
    bc_batch = 2048,
    eval_every = 5,
    train_seed_start = 600_000,
    holdout_seed_start = 900_000,
    search_seed_start = 500_000,
    verbose = false
))]
#[allow(clippy::too_many_arguments)]
fn one_warehouse_multi_retailer_train_ppo(
    py: Python<'_>,
    seed: u64,
    iters: usize,
    train_paths: usize,
    eval_paths: usize,
    hidden: usize,
    lr: f64,
    gamma: f32,
    lam: f32,
    clip: f32,
    ppo_epochs: usize,
    minibatch: usize,
    vf_coef: f32,
    ent_coef: f32,
    max_grad_norm: f32,
    reward_scale: f32,
    bc_epochs: usize,
    bc_paths: usize,
    bc_lr: f64,
    bc_batch: usize,
    eval_every: usize,
    train_seed_start: u64,
    holdout_seed_start: u64,
    search_seed_start: u64,
    verbose: bool,
) -> PyResult<PyObject> {
    let cfg = PpoConfig {
        iters,
        train_paths,
        eval_paths,
        hidden,
        lr,
        gamma,
        lam,
        clip,
        ppo_epochs,
        minibatch,
        vf_coef,
        ent_coef,
        max_grad_norm,
        reward_scale,
        bc_epochs,
        bc_paths,
        bc_lr,
        bc_batch,
        eval_every,
        seed,
        train_seed_start,
        holdout_seed_start,
        search_seed_start,
        verbose,
    };

    // Anchor: the echelon base-stock gate on the same holdout block.
    let gate = gate_holdout_mean_cost(eval_paths, holdout_seed_start);

    // Run the whole training loop in Rust without holding the GIL.
    let outcome = py
        .allow_threads(|| {
            let make_env = |n: usize, eval: bool| {
                Box::new(OwmrPpoEnv::instance_14(n, eval)) as Box<dyn PpoVecEnv>
            };
            train_ppo(make_env, &cfg)
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("PPO training failed: {e}")))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("gate_holdout_cost", gate)?;
    dict.set_item("best_holdout_cost", outcome.best_holdout_cost)?;
    dict.set_item("final_holdout_cost_mean", outcome.final_holdout_cost_mean)?;
    dict.set_item("final_holdout_cost_std", outcome.final_holdout_cost_std)?;
    let curve = PyList::empty_bound(py);
    for point in &outcome.curve {
        let pt = PyDict::new_bound(py);
        pt.set_item("iter", point.iter)?;
        pt.set_item("phase", &point.phase)?;
        pt.set_item("train_cost", point.train_cost)?;
        pt.set_item("holdout_greedy_cost", point.holdout_greedy_cost)?;
        curve.append(pt)?;
    }
    dict.set_item("curve", curve)?;
    Ok(dict.into_any().unbind().into())
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(one_warehouse_multi_retailer_train_ppo, m)?)?;
    Ok(())
}
