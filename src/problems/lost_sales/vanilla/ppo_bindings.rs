//! # `ppo_bindings` — Python entry point for the lost_sales PPO trainer
//!
//! Exposes the reusable Rust PPO trainer (`core::ppo`) applied to the canonical
//! lost_sales instance (`vanilla_l4_p4_poisson5`) via a pyo3 function, mirroring
//! the OWMR binding. This is the second problem wired to PPO — demonstrating that
//! "add a problem" = a `PpoVecEnv` + a `<problem>_train_ppo` binding + one row in
//! `invman.ppo_trainer._PPO_BINDINGS`. Continuous (diagonal-Gaussian) action.
//!
//! Returns per-period costs (total / horizon) since lost_sales is reported as an
//! average per-period cost (published optimal 4.73, capped base-stock 4.80).
//! Feature-gated behind `ppo`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;

use crate::core::ppo::environment::PpoVecEnv;
use crate::core::ppo::ppo_trainer::{train_ppo, PpoConfig};
use crate::problems::lost_sales::vanilla::ppo_environment::{
    gate_holdout_mean_cost_per_period, LostSalesPpoEnv,
};

#[pyfunction]
#[pyo3(signature = (
    seed = 0,
    iters = 80,
    train_paths = 128,
    eval_paths = 512,
    hidden = 64,
    lr = 1e-3,
    gamma = 1.0,
    lam = 0.95,
    clip = 0.2,
    ppo_epochs = 4,
    minibatch = 4096,
    vf_coef = 0.5,
    ent_coef = 0.0,
    max_grad_norm = 0.5,
    bc_epochs = 40,
    bc_paths = 256,
    bc_lr = 1e-3,
    bc_batch = 4096,
    eval_every = 10,
    holdout_seed_start = 900_000,
    verbose = false
))]
#[allow(clippy::too_many_arguments)]
fn lost_sales_train_ppo(
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
    bc_epochs: usize,
    bc_paths: usize,
    bc_lr: f64,
    bc_batch: usize,
    eval_every: usize,
    holdout_seed_start: u64,
    verbose: bool,
) -> PyResult<PyObject> {
    let horizon = LostSalesPpoEnv::vanilla_poisson5_l4(1).horizon();
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
        reward_scale: 1.0, // lost_sales per-period costs are O(1)
        bc_epochs,
        bc_paths,
        bc_lr,
        bc_batch,
        eval_every,
        seed,
        train_seed_start: 600_000,
        holdout_seed_start,
        search_seed_start: 500_000,
        verbose,
    };
    let gate_avg = gate_holdout_mean_cost_per_period(eval_paths, holdout_seed_start);

    let outcome = py
        .allow_threads(|| {
            let make_env = |n: usize, _eval: bool| {
                Box::new(LostSalesPpoEnv::vanilla_poisson5_l4(n)) as Box<dyn PpoVecEnv>
            };
            train_ppo(make_env, &cfg)
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("PPO training failed: {e}")))?;

    let h = horizon as f64;
    let dict = PyDict::new_bound(py);
    dict.set_item("gate_avg_cost_per_period", gate_avg)?;
    dict.set_item("best_avg_cost_per_period", outcome.best_holdout_cost / h)?;
    dict.set_item("final_avg_cost_per_period", outcome.final_holdout_cost_mean / h)?;
    dict.set_item("best_holdout_total", outcome.best_holdout_cost)?;
    dict.set_item("horizon", horizon)?;
    let curve = PyList::empty_bound(py);
    for point in &outcome.curve {
        let pt = PyDict::new_bound(py);
        pt.set_item("iter", point.iter)?;
        pt.set_item("phase", &point.phase)?;
        pt.set_item("avg_cost_per_period", point.holdout_greedy_cost / h)?;
        curve.append(pt)?;
    }
    dict.set_item("curve", curve)?;
    Ok(dict.into_any().unbind().into())
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lost_sales_train_ppo, m)?)?;
    Ok(())
}
