use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::ameliorating_inventory::demand::{
    parse_demand_distribution_kind, DemandModel,
};
use crate::problems::ameliorating_inventory::heuristics::{
    newsvendor_purchase_order_quantity, policy_rollout_from_paths, simulate_policy,
    two_dimensional_order_up_to_order_quantity,
};
use crate::problems::ameliorating_inventory::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_paths,
    AmelioratingInventoryRolloutConfig,
};

fn build_demand_models(
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
) -> PyResult<Vec<DemandModel>> {
    if demand_kinds.len() != demand_means.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "demand_kinds and demand_means must have the same length",
        ));
    }
    demand_kinds
        .iter()
        .zip(demand_means.iter())
        .map(|(kind, mean)| {
            Ok(DemandModel {
                kind: parse_demand_distribution_kind(kind)?,
                param1: *mean,
            })
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    inventory_by_age,
    periods,
    demand_kinds,
    demand_means,
    target_ages,
    product_prices,
    age_retention,
    purchase_cost_per_unit,
    holding_cost_per_unit,
    decay_salvage_values,
    seed=1234,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn ameliorating_inventory_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_by_age: Vec<usize>,
    periods: usize,
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
    target_ages: Vec<usize>,
    product_prices: Vec<f64>,
    age_retention: Vec<f64>,
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: Vec<f64>,
    seed: u64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(&inventory_by_age)?;
    let config = AmelioratingInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_models: build_demand_models(demand_kinds, demand_means)?,
        target_ages,
        product_prices,
        age_retention,
        purchase_cost_per_unit,
        holding_cost_per_unit,
        decay_salvage_values,
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
    inventory_by_age,
    periods,
    demand_kinds,
    demand_means,
    target_ages,
    product_prices,
    age_retention,
    purchase_cost_per_unit,
    holding_cost_per_unit,
    decay_salvage_values,
    seeds,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn ameliorating_inventory_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_by_age: Vec<usize>,
    periods: usize,
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
    target_ages: Vec<usize>,
    product_prices: Vec<f64>,
    age_retention: Vec<f64>,
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: Vec<f64>,
    seeds: Vec<u64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let initial_state = build_initial_state(&inventory_by_age)?;
    let config = AmelioratingInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_models: build_demand_models(demand_kinds, demand_means)?,
        target_ages,
        product_prices,
        age_retention,
        purchase_cost_per_unit,
        holding_cost_per_unit,
        decay_salvage_values,
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
    inventory_by_age,
    realized_demands,
    demand_kinds,
    demand_means,
    target_ages,
    product_prices,
    age_retention,
    purchase_cost_per_unit,
    holding_cost_per_unit,
    decay_salvage_values,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn ameliorating_inventory_soft_tree_rollout_from_paths(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_by_age: Vec<usize>,
    realized_demands: Vec<Vec<usize>>,
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
    target_ages: Vec<usize>,
    product_prices: Vec<f64>,
    age_retention: Vec<f64>,
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: Vec<f64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(&inventory_by_age)?;
    let config = AmelioratingInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods: realized_demands.len(),
        demand_models: build_demand_models(demand_kinds, demand_means)?,
        target_ages,
        product_prices,
        age_retention,
        purchase_cost_per_unit,
        holding_cost_per_unit,
        decay_salvage_values,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    rollout_from_paths(&flat_params, &config, &initial_state, &realized_demands)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    inventory_by_age,
    realized_demands,
    target_ages,
    product_prices,
    age_retention,
    purchase_cost_per_unit,
    holding_cost_per_unit,
    decay_salvage_values,
    discount_factor=0.99
))]
fn ameliorating_inventory_policy_rollout_from_paths(
    policy_name: &str,
    params: Vec<f64>,
    inventory_by_age: Vec<usize>,
    realized_demands: Vec<Vec<usize>>,
    target_ages: Vec<usize>,
    product_prices: Vec<f64>,
    age_retention: Vec<f64>,
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: Vec<f64>,
    discount_factor: f64,
) -> PyResult<f64> {
    let initial_state = build_initial_state(&inventory_by_age)?;
    policy_rollout_from_paths(
        policy_name,
        &params,
        &initial_state,
        &realized_demands,
        &target_ages,
        &product_prices,
        &age_retention,
        purchase_cost_per_unit,
        holding_cost_per_unit,
        &decay_salvage_values,
        discount_factor,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    inventory_by_age,
    periods,
    demand_kinds,
    demand_means,
    target_ages,
    product_prices,
    age_retention,
    purchase_cost_per_unit,
    holding_cost_per_unit,
    decay_salvage_values,
    replications=1000,
    seed=1234,
    discount_factor=0.99
))]
fn ameliorating_inventory_simulate_policy(
    policy_name: &str,
    params: Vec<f64>,
    inventory_by_age: Vec<usize>,
    periods: usize,
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
    target_ages: Vec<usize>,
    product_prices: Vec<f64>,
    age_retention: Vec<f64>,
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: Vec<f64>,
    replications: usize,
    seed: u64,
    discount_factor: f64,
) -> PyResult<(f64, f64)> {
    let initial_state = build_initial_state(&inventory_by_age)?;
    let summary = simulate_policy(
        policy_name,
        &params,
        &initial_state,
        periods,
        replications,
        seed,
        &build_demand_models(demand_kinds, demand_means)?,
        &target_ages,
        &product_prices,
        &age_retention,
        purchase_cost_per_unit,
        holding_cost_per_unit,
        &decay_salvage_values,
        discount_factor,
    )?;
    Ok((summary.mean_cost, summary.cost_std))
}

#[pyfunction]
#[pyo3(signature = (inventory_by_age, total_target_inventory))]
fn ameliorating_inventory_newsvendor_purchase_order(
    inventory_by_age: Vec<usize>,
    total_target_inventory: usize,
) -> PyResult<usize> {
    let state = build_initial_state(&inventory_by_age)?;
    newsvendor_purchase_order_quantity(&state, total_target_inventory)
}

#[pyfunction]
#[pyo3(signature = (inventory_by_age, total_target_inventory, young_target_inventory, young_age_cutoff))]
fn ameliorating_inventory_two_dimensional_order_up_to_order(
    inventory_by_age: Vec<usize>,
    total_target_inventory: usize,
    young_target_inventory: usize,
    young_age_cutoff: usize,
) -> PyResult<usize> {
    let state = build_initial_state(&inventory_by_age)?;
    two_dimensional_order_up_to_order_quantity(
        &state,
        total_target_inventory,
        young_target_inventory,
        young_age_cutoff,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ameliorating_inventory_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_soft_tree_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_policy_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ameliorating_inventory_simulate_policy, m)?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_newsvendor_purchase_order,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_two_dimensional_order_up_to_order,
        m
    )?)?;
    Ok(())
}
