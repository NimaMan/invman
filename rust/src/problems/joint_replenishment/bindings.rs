use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::joint_replenishment::demand::DemandRange;
use crate::problems::joint_replenishment::env::initialize_state;
use crate::problems::joint_replenishment::heuristics::{
    dynamic_order_up_to_order_quantities, minimum_order_quantity_order_quantities,
    policy_rollout_from_paths, simulate_policy,
};
use crate::problems::joint_replenishment::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_paths,
    JointReplenishmentRolloutConfig,
};

fn build_demand_ranges(demand_lows: Vec<usize>, demand_highs: Vec<usize>) -> PyResult<Vec<DemandRange>> {
    if demand_lows.len() != demand_highs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "demand_lows and demand_highs must have the same length",
        ));
    }
    Ok(demand_lows
        .into_iter()
        .zip(demand_highs.into_iter())
        .map(|(low, high)| DemandRange { low, high })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    initial_inventory_levels,
    demand_lows,
    demand_highs,
    truck_capacity,
    minor_order_costs,
    major_order_cost,
    holding_costs,
    shortage_costs,
    periods,
    seed=1234,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn joint_replenishment_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_inventory_levels: Vec<i32>,
    demand_lows: Vec<usize>,
    demand_highs: Vec<usize>,
    truck_capacity: usize,
    minor_order_costs: Vec<f64>,
    major_order_cost: f64,
    holding_costs: Vec<f64>,
    shortage_costs: Vec<f64>,
    periods: usize,
    seed: u64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let demand_ranges = build_demand_ranges(demand_lows, demand_highs)?;
    let initial_state = build_initial_state(&initial_inventory_levels)?;
    let config = JointReplenishmentRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_ranges,
        truck_capacity,
        minor_order_costs,
        major_order_cost,
        holding_costs,
        shortage_costs,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    rollout(&flat_params, &config, &initial_state, seed)
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    initial_inventory_levels,
    demand_lows,
    demand_highs,
    truck_capacity,
    minor_order_costs,
    major_order_cost,
    holding_costs,
    shortage_costs,
    periods,
    seeds,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn joint_replenishment_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_inventory_levels: Vec<i32>,
    demand_lows: Vec<usize>,
    demand_highs: Vec<usize>,
    truck_capacity: usize,
    minor_order_costs: Vec<f64>,
    major_order_cost: f64,
    holding_costs: Vec<f64>,
    shortage_costs: Vec<f64>,
    periods: usize,
    seeds: Vec<u64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let demand_ranges = build_demand_ranges(demand_lows, demand_highs)?;
    let initial_state = build_initial_state(&initial_inventory_levels)?;
    let config = JointReplenishmentRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_ranges,
        truck_capacity,
        minor_order_costs,
        major_order_cost,
        holding_costs,
        shortage_costs,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    population_rollout(&params_batch, &config, &initial_state, &seeds)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    initial_inventory_levels,
    demands,
    demand_lows,
    demand_highs,
    truck_capacity,
    minor_order_costs,
    major_order_cost,
    holding_costs,
    shortage_costs,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn joint_replenishment_soft_tree_rollout_from_paths(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_inventory_levels: Vec<i32>,
    demands: Vec<Vec<usize>>,
    demand_lows: Vec<usize>,
    demand_highs: Vec<usize>,
    truck_capacity: usize,
    minor_order_costs: Vec<f64>,
    major_order_cost: f64,
    holding_costs: Vec<f64>,
    shortage_costs: Vec<f64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let demand_ranges = build_demand_ranges(demand_lows, demand_highs)?;
    let initial_state = build_initial_state(&initial_inventory_levels)?;
    let config = JointReplenishmentRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods: demands.len(),
        demand_ranges,
        truck_capacity,
        minor_order_costs,
        major_order_cost,
        holding_costs,
        shortage_costs,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    rollout_from_paths(&flat_params, &config, &initial_state, &demands)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    initial_inventory_levels,
    demands,
    demand_lows,
    demand_highs,
    truck_capacity,
    minor_order_costs,
    major_order_cost,
    holding_costs,
    shortage_costs,
    discount_factor=0.99
))]
fn joint_replenishment_policy_rollout_from_paths(
    policy_name: &str,
    params: Vec<f64>,
    initial_inventory_levels: Vec<i32>,
    demands: Vec<Vec<usize>>,
    demand_lows: Vec<usize>,
    demand_highs: Vec<usize>,
    truck_capacity: usize,
    minor_order_costs: Vec<f64>,
    major_order_cost: f64,
    holding_costs: Vec<f64>,
    shortage_costs: Vec<f64>,
    discount_factor: f64,
) -> PyResult<f64> {
    let initial_state = initialize_state(&initial_inventory_levels)?;
    let demand_ranges = build_demand_ranges(demand_lows, demand_highs)?;
    policy_rollout_from_paths(
        policy_name,
        &params,
        &initial_state,
        &demands,
        truck_capacity,
        &demand_ranges,
        &minor_order_costs,
        major_order_cost,
        &holding_costs,
        &shortage_costs,
        discount_factor,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    initial_inventory_levels,
    periods,
    replications,
    seed,
    demand_lows,
    demand_highs,
    truck_capacity,
    minor_order_costs,
    major_order_cost,
    holding_costs,
    shortage_costs,
    discount_factor=0.99
))]
fn joint_replenishment_simulate_policy(
    policy_name: &str,
    params: Vec<f64>,
    initial_inventory_levels: Vec<i32>,
    periods: usize,
    replications: usize,
    seed: u64,
    demand_lows: Vec<usize>,
    demand_highs: Vec<usize>,
    truck_capacity: usize,
    minor_order_costs: Vec<f64>,
    major_order_cost: f64,
    holding_costs: Vec<f64>,
    shortage_costs: Vec<f64>,
    discount_factor: f64,
) -> PyResult<(f64, f64)> {
    let demand_ranges = build_demand_ranges(demand_lows, demand_highs)?;
    let summary = simulate_policy(
        policy_name,
        &params,
        &initial_inventory_levels,
        periods,
        replications,
        seed,
        &demand_ranges,
        truck_capacity,
        &minor_order_costs,
        major_order_cost,
        &holding_costs,
        &shortage_costs,
        discount_factor,
    )?;
    Ok((summary.mean_cost, summary.cost_std))
}

#[pyfunction]
#[pyo3(signature = (
    inventory_levels,
    item_targets,
    review_period,
    rounding_threshold,
    truck_capacity,
    period=0
))]
fn joint_replenishment_moq_order_quantities(
    inventory_levels: Vec<i32>,
    item_targets: Vec<usize>,
    review_period: usize,
    rounding_threshold: f64,
    truck_capacity: usize,
    period: usize,
) -> PyResult<Vec<usize>> {
    let mut state = initialize_state(&inventory_levels)?;
    state.period = period;
    minimum_order_quantity_order_quantities(
        &state,
        &item_targets,
        review_period,
        rounding_threshold,
        truck_capacity,
    )
}

#[pyfunction]
#[pyo3(signature = (
    inventory_levels,
    item_targets,
    demand_lows,
    demand_highs,
    truck_capacity,
    holding_costs,
    shortage_costs,
    period=0
))]
fn joint_replenishment_dynout_order_quantities(
    inventory_levels: Vec<i32>,
    item_targets: Vec<usize>,
    demand_lows: Vec<usize>,
    demand_highs: Vec<usize>,
    truck_capacity: usize,
    holding_costs: Vec<f64>,
    shortage_costs: Vec<f64>,
    period: usize,
) -> PyResult<Vec<usize>> {
    let mut state = initialize_state(&inventory_levels)?;
    state.period = period;
    let demand_ranges = build_demand_ranges(demand_lows, demand_highs)?;
    dynamic_order_up_to_order_quantities(
        &state,
        &item_targets,
        truck_capacity,
        &demand_ranges,
        &holding_costs,
        &shortage_costs,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(joint_replenishment_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        joint_replenishment_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        joint_replenishment_soft_tree_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        joint_replenishment_policy_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(joint_replenishment_simulate_policy, m)?)?;
    m.add_function(wrap_pyfunction!(
        joint_replenishment_moq_order_quantities,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        joint_replenishment_dynout_order_quantities,
        m
    )?)?;
    Ok(())
}
