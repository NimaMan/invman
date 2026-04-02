use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::multi_echelon::heuristics::search_constant_base_stock_from_demands;
use crate::problems::multi_echelon::rollout::{
    population_rollout as multi_echelon_population_rollout, rollout as multi_echelon_rollout,
    MultiEchelonRolloutConfig,
};

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    warehouse_lead_time,
    retailer_lead_time,
    num_retailers,
    warehouse_holding_cost,
    retailer_holding_cost,
    warehouse_expedited_cost,
    warehouse_lost_sale_cost,
    expedited_service_prob,
    warehouse_capacity,
    warehouse_inventory_cap,
    retailer_inventory_cap,
    demand_mean,
    demand_std,
    horizon=2000,
    seed=1234,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn multi_echelon_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    warehouse_lead_time: usize,
    retailer_lead_time: usize,
    num_retailers: usize,
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    expedited_service_prob: f64,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    demand_mean: f64,
    demand_std: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let warehouse_levels = allowed_values
        .as_ref()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("multi-echelon rollouts require allowed_values")
        })?
        .get(0)
        .cloned()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("missing warehouse allowed_values")
        })?;
    let retailer_levels = allowed_values
        .as_ref()
        .unwrap()
        .get(1)
        .cloned()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("missing retailer allowed_values")
        })?;
    let config = MultiEchelonRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        warehouse_lead_time,
        retailer_lead_time,
        num_retailers,
        warehouse_holding_cost,
        retailer_holding_cost,
        warehouse_expedited_cost,
        warehouse_lost_sale_cost,
        expedited_service_prob,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        demand_mean,
        demand_std,
        horizon,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    multi_echelon_rollout(
        &flat_params,
        &config,
        seed,
        &warehouse_levels,
        &retailer_levels,
    )
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    warehouse_lead_time,
    retailer_lead_time,
    num_retailers,
    warehouse_holding_cost,
    retailer_holding_cost,
    warehouse_expedited_cost,
    warehouse_lost_sale_cost,
    expedited_service_prob,
    warehouse_capacity,
    warehouse_inventory_cap,
    retailer_inventory_cap,
    demand_mean,
    demand_std,
    seeds,
    horizon=2000,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn multi_echelon_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    warehouse_lead_time: usize,
    retailer_lead_time: usize,
    num_retailers: usize,
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    expedited_service_prob: f64,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    demand_mean: f64,
    demand_std: f64,
    seeds: Vec<u64>,
    horizon: usize,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let warehouse_levels = allowed_values
        .as_ref()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("multi-echelon rollouts require allowed_values")
        })?
        .get(0)
        .cloned()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("missing warehouse allowed_values")
        })?;
    let retailer_levels = allowed_values
        .as_ref()
        .unwrap()
        .get(1)
        .cloned()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("missing retailer allowed_values")
        })?;
    let config = MultiEchelonRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        warehouse_lead_time,
        retailer_lead_time,
        num_retailers,
        warehouse_holding_cost,
        retailer_holding_cost,
        warehouse_expedited_cost,
        warehouse_lost_sale_cost,
        expedited_service_prob,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        demand_mean,
        demand_std,
        horizon,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    multi_echelon_population_rollout(
        &params_batch,
        &config,
        &seeds,
        &warehouse_levels,
        &retailer_levels,
    )
}

#[pyfunction]
#[pyo3(signature = (
    warehouse_inventory,
    warehouse_pipeline,
    retailer_inventory,
    retailer_pipeline,
    demands,
    expedite_uniforms,
    warehouse_levels,
    retailer_levels,
    warehouse_holding_cost,
    retailer_holding_cost,
    warehouse_expedited_cost,
    warehouse_lost_sale_cost,
    expedited_service_prob,
    warehouse_capacity,
    warehouse_inventory_cap,
    retailer_inventory_cap,
    warm_up_periods_ratio=0.2,
    top_k=10
))]
fn multi_echelon_constant_base_stock_search_from_demands(
    warehouse_inventory: i64,
    warehouse_pipeline: Vec<usize>,
    retailer_inventory: Vec<i64>,
    retailer_pipeline: Vec<Vec<usize>>,
    demands: Vec<Vec<usize>>,
    expedite_uniforms: Vec<Vec<Vec<f64>>>,
    warehouse_levels: Vec<usize>,
    retailer_levels: Vec<usize>,
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    expedited_service_prob: f64,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    warm_up_periods_ratio: f64,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    search_constant_base_stock_from_demands(
        warehouse_inventory,
        &warehouse_pipeline,
        &retailer_inventory,
        &retailer_pipeline,
        &demands,
        &expedite_uniforms,
        &warehouse_levels,
        &retailer_levels,
        warehouse_holding_cost,
        retailer_holding_cost,
        warehouse_expedited_cost,
        warehouse_lost_sale_cost,
        expedited_service_prob,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        warm_up_periods_ratio,
        top_k,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(multi_echelon_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(multi_echelon_soft_tree_population_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_constant_base_stock_search_from_demands,
        m
    )?)?;
    Ok(())
}
