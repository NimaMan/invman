use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::perishable_inventory::env::{parse_issuing_policy, PerishableState};
use crate::problems::perishable_inventory::heuristics::{
    policy_rollout_from_demands, search_base_stock_from_demands, search_bsp_low_ew_from_demands,
};
use crate::problems::perishable_inventory::rollout::{
    population_rollout as perishable_population_rollout, rollout as perishable_rollout,
    rollout_from_demands as perishable_rollout_from_demands, PerishableInventoryRolloutConfig,
};

fn empirical_mean_demand(demands: &[usize]) -> f64 {
    if demands.is_empty() {
        0.0
    } else {
        demands.iter().copied().sum::<usize>() as f64 / demands.len() as f64
    }
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    demand_mean,
    demand_cov,
    shelf_life,
    lead_time,
    holding_cost,
    shortage_cost,
    waste_cost,
    procurement_cost=0.0,
    horizon=2000,
    seed=1234,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    issuing_policy="fifo",
    allowed_values=None
))]
fn perishable_inventory_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    demand_mean: f64,
    demand_cov: f64,
    shelf_life: usize,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    issuing_policy: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let config = PerishableInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        demand_mean,
        demand_cov,
        shelf_life,
        lead_time,
        holding_cost,
        shortage_cost,
        waste_cost,
        procurement_cost,
        horizon,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        issuing_policy: parse_issuing_policy(issuing_policy)?,
    };
    perishable_rollout(&flat_params, &config, seed)
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    demand_mean,
    demand_cov,
    shelf_life,
    lead_time,
    holding_cost,
    shortage_cost,
    waste_cost,
    seeds,
    procurement_cost=0.0,
    horizon=2000,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    issuing_policy="fifo",
    allowed_values=None
))]
fn perishable_inventory_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    demand_mean: f64,
    demand_cov: f64,
    shelf_life: usize,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    seeds: Vec<u64>,
    procurement_cost: f64,
    horizon: usize,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    issuing_policy: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let config = PerishableInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        demand_mean,
        demand_cov,
        shelf_life,
        lead_time,
        holding_cost,
        shortage_cost,
        waste_cost,
        procurement_cost,
        horizon,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        issuing_policy: parse_issuing_policy(issuing_policy)?,
    };
    perishable_population_rollout(&params_batch, &config, &seeds)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    on_hand,
    pipeline_orders,
    demands,
    holding_cost,
    shortage_cost,
    waste_cost,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    issuing_policy="fifo",
    allowed_values=None,
    demand_mean=None
))]
fn perishable_inventory_soft_tree_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    demands: Vec<usize>,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    issuing_policy: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
    demand_mean: Option<f64>,
) -> PyResult<f64> {
    let config = PerishableInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        demand_mean: demand_mean.unwrap_or_else(|| empirical_mean_demand(&demands)),
        demand_cov: 1.0,
        shelf_life: on_hand.len(),
        lead_time: pipeline_orders.len() + 1,
        holding_cost,
        shortage_cost,
        waste_cost,
        procurement_cost,
        horizon: demands.len(),
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        issuing_policy: parse_issuing_policy(issuing_policy)?,
    };
    let state = PerishableState {
        on_hand,
        pipeline_orders,
    };
    perishable_rollout_from_demands(&flat_params, &config, state, &demands)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    on_hand,
    pipeline_orders,
    demands,
    lead_time,
    max_order_size,
    demand_mean,
    holding_cost,
    shortage_cost,
    waste_cost,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    issuing_policy="fifo"
))]
fn perishable_inventory_policy_rollout_from_demands(
    policy_name: &str,
    params: Vec<usize>,
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    demands: Vec<usize>,
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: &str,
) -> PyResult<f64> {
    let state = PerishableState {
        on_hand,
        pipeline_orders,
    };
    policy_rollout_from_demands(
        policy_name,
        &params,
        &state,
        &demands,
        lead_time,
        max_order_size,
        demand_mean,
        holding_cost,
        shortage_cost,
        waste_cost,
        procurement_cost,
        warm_up_periods_ratio,
        parse_issuing_policy(issuing_policy)?,
    )
}

#[pyfunction]
#[pyo3(signature = (
    on_hand,
    pipeline_orders,
    demands,
    lead_time,
    max_order_size,
    demand_mean,
    holding_cost,
    shortage_cost,
    waste_cost,
    position_upper_bound,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    issuing_policy="fifo",
    top_k=12
))]
fn perishable_inventory_base_stock_search_from_demands(
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    demands: Vec<usize>,
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    position_upper_bound: usize,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: &str,
    top_k: usize,
) -> PyResult<((usize, f64), Vec<(usize, f64)>)> {
    let state = PerishableState {
        on_hand,
        pipeline_orders,
    };
    search_base_stock_from_demands(
        &state,
        &demands,
        lead_time,
        max_order_size,
        demand_mean,
        holding_cost,
        shortage_cost,
        waste_cost,
        procurement_cost,
        warm_up_periods_ratio,
        parse_issuing_policy(issuing_policy)?,
        position_upper_bound,
        top_k,
    )
}

#[pyfunction]
#[pyo3(signature = (
    on_hand,
    pipeline_orders,
    demands,
    lead_time,
    max_order_size,
    demand_mean,
    holding_cost,
    shortage_cost,
    waste_cost,
    position_upper_bound,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    issuing_policy="fifo",
    top_k=12
))]
fn perishable_inventory_bsp_low_ew_search_from_demands(
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    demands: Vec<usize>,
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    position_upper_bound: usize,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: &str,
    top_k: usize,
) -> PyResult<((usize, usize, usize, f64), Vec<(usize, usize, usize, f64)>)> {
    let state = PerishableState {
        on_hand,
        pipeline_orders,
    };
    search_bsp_low_ew_from_demands(
        &state,
        &demands,
        lead_time,
        max_order_size,
        demand_mean,
        holding_cost,
        shortage_cost,
        waste_cost,
        procurement_cost,
        warm_up_periods_ratio,
        parse_issuing_policy(issuing_policy)?,
        position_upper_bound,
        top_k,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(perishable_inventory_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_soft_tree_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_policy_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_base_stock_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_bsp_low_ew_search_from_demands,
        m
    )?)?;
    Ok(())
}
