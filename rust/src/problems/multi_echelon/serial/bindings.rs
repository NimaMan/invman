//! pyo3 bindings for the serial Clark-Scarf env soft-tree rollout.
//!
//! Exposes `multi_echelon_serial_soft_tree_population_rollout` (the CMA-ES scoring
//! bridge) plus a single-rollout convenience and an exact-solver helper so the
//! Python autoresearch runner can warm-start at the Clark-Scarf optimum and report
//! the learned cost against it. The rollout itself lives in Rust (`rollout.rs`);
//! these functions only marshal Python arguments into `SerialRolloutConfig`.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{parse_leaf_type, parse_split_type};
use crate::problems::multi_echelon::serial::env::SerialConfig;
use crate::problems::multi_echelon::serial::exact::{
    solve_serial_clark_scarf, GridParams, SerialDemand, SerialStage,
};
use crate::problems::multi_echelon::serial::rollout::{
    population_rollout, rollout, SerialRolloutConfig,
};

fn build_rollout_config(
    holding_cost: Vec<f64>,
    lead_time: Vec<usize>,
    penalty: f64,
    demand_mean: f64,
    demand_std: f64,
    warm_start_levels: Vec<f64>,
    level_min: Vec<f64>,
    level_max: Vec<f64>,
    depth: usize,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    periods: usize,
    warm_up: usize,
) -> PyResult<SerialRolloutConfig> {
    Ok(SerialRolloutConfig {
        config: SerialConfig {
            holding_cost,
            lead_time,
            penalty,
        },
        demand_mean,
        demand_std,
        warm_start_levels,
        depth,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        level_min: level_min.into_iter().map(|v| v as f32).collect(),
        level_max: level_max.into_iter().map(|v| v as f32).collect(),
        periods,
        warm_up,
    })
}

/// Single soft-tree rollout: mean per-period cost after warm-up on one seed.
#[pyfunction]
#[pyo3(signature = (
    flat_params,
    holding_cost,
    lead_time,
    penalty,
    demand_mean,
    demand_std,
    warm_start_levels,
    level_min,
    level_max,
    depth,
    periods,
    warm_up,
    seed=1234,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant"
))]
#[allow(clippy::too_many_arguments)]
fn multi_echelon_serial_soft_tree_rollout(
    flat_params: Vec<f32>,
    holding_cost: Vec<f64>,
    lead_time: Vec<usize>,
    penalty: f64,
    demand_mean: f64,
    demand_std: f64,
    warm_start_levels: Vec<f64>,
    level_min: Vec<f64>,
    level_max: Vec<f64>,
    depth: usize,
    periods: usize,
    warm_up: usize,
    seed: u64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
) -> PyResult<f64> {
    let config = build_rollout_config(
        holding_cost,
        lead_time,
        penalty,
        demand_mean,
        demand_std,
        warm_start_levels,
        level_min,
        level_max,
        depth,
        temperature,
        split_type,
        leaf_type,
        periods,
        warm_up,
    )?;
    rollout(&flat_params, &config, seed)
}

/// Paired population rollout: one mean cost per (params, seed) pair, fanned out via
/// rayon. This is the CMA-ES scoring binding.
#[pyfunction]
#[pyo3(signature = (
    params_batch,
    holding_cost,
    lead_time,
    penalty,
    demand_mean,
    demand_std,
    warm_start_levels,
    level_min,
    level_max,
    depth,
    periods,
    warm_up,
    seeds,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant"
))]
#[allow(clippy::too_many_arguments)]
fn multi_echelon_serial_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    holding_cost: Vec<f64>,
    lead_time: Vec<usize>,
    penalty: f64,
    demand_mean: f64,
    demand_std: f64,
    warm_start_levels: Vec<f64>,
    level_min: Vec<f64>,
    level_max: Vec<f64>,
    depth: usize,
    periods: usize,
    warm_up: usize,
    seeds: Vec<u64>,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
) -> PyResult<Vec<f64>> {
    let config = build_rollout_config(
        holding_cost,
        lead_time,
        penalty,
        demand_mean,
        demand_std,
        warm_start_levels,
        level_min,
        level_max,
        depth,
        temperature,
        split_type,
        leaf_type,
        periods,
        warm_up,
    )?;
    population_rollout(&params_batch, &config, &seeds)
}

/// Exact Clark-Scarf solver helper: returns the optimal echelon base-stock levels
/// (downstream -> upstream) and optimal cost for a Normal-demand serial instance.
/// `echelon_holding` and `lead_time` are downstream -> upstream. Used by the Python
/// runner to warm-start the policy at the optimum and report the MATCH baseline.
#[pyfunction]
#[pyo3(signature = (echelon_holding, lead_time, penalty, demand_mean, demand_std))]
fn multi_echelon_serial_exact_normal_solution<'py>(
    py: Python<'py>,
    echelon_holding: Vec<f64>,
    lead_time: Vec<usize>,
    penalty: f64,
    demand_mean: f64,
    demand_std: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if echelon_holding.len() != lead_time.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "echelon_holding and lead_time must have the same length",
        ));
    }
    let stages: Vec<SerialStage> = echelon_holding
        .iter()
        .zip(lead_time.iter())
        .map(|(h, l)| SerialStage {
            echelon_holding_cost: *h,
            lead_time: *l,
        })
        .collect();
    let solution = solve_serial_clark_scarf(
        &stages,
        penalty,
        SerialDemand::Normal {
            mean: demand_mean,
            std: demand_std,
        },
        GridParams::default(),
    );
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "echelon_base_stock_levels",
        solution.echelon_base_stock_levels,
    )?;
    dict.set_item(
        "local_base_stock_levels_upstream_to_downstream",
        solution.local_base_stock_levels_upstream_to_downstream,
    )?;
    dict.set_item("optimal_cost", solution.optimal_cost)?;
    Ok(dict)
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(multi_echelon_serial_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_serial_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_serial_exact_normal_solution,
        m
    )?)?;
    Ok(())
}
