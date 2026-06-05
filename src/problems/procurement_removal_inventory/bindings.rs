use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::procurement_removal_inventory::demand::parse_demand_distribution_kind;
use crate::problems::procurement_removal_inventory::env::{
    build_raw_state, initialize_state, step_state,
};
use crate::problems::procurement_removal_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::procurement_removal_inventory::heuristics::{
    interval_stock_action, policy_rollout, policy_rollout_from_demands,
    returnability_buffer_interval_stock_action, simulate_policy, PolicySimulationSummary,
};
use crate::problems::procurement_removal_inventory::literature::references::{
    ExactVerificationReference, ProcurementRemovalReferenceInstance, PRIMARY_REFERENCE_INSTANCE,
    VERIFICATION_PROBLEM_INSTANCE,
};
use crate::problems::procurement_removal_inventory::rollout::{
    population_rollout, rollout, rollout_from_demands, ProcurementRemovalRolloutConfig,
};

fn primary_reference_to_py(
    py: Python<'_>,
    reference: &ProcurementRemovalReferenceInstance,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", reference.name)?;
    dict.set_item("source", reference.source)?;
    dict.set_item("url", reference.url)?;
    dict.set_item("periods", reference.periods)?;
    dict.set_item(
        "demand_distribution_kind",
        reference.demand_distribution_kind,
    )?;
    dict.set_item("demand_mean", reference.demand_mean)?;
    dict.set_item("initial_inventory_level", reference.initial_inventory_level)?;
    dict.set_item(
        "initial_returnable_inventory",
        reference.initial_returnable_inventory,
    )?;
    dict.set_item("returnable_purchase_cap", reference.returnable_purchase_cap)?;
    dict.set_item("purchase_cost_per_unit", reference.purchase_cost_per_unit)?;
    dict.set_item("return_value_per_unit", reference.return_value_per_unit)?;
    dict.set_item(
        "liquidation_value_per_unit",
        reference.liquidation_value_per_unit,
    )?;
    dict.set_item("holding_cost_per_unit", reference.holding_cost_per_unit)?;
    dict.set_item("shortage_cost_per_unit", reference.shortage_cost_per_unit)?;
    dict.set_item("max_purchase_quantity", reference.max_purchase_quantity)?;
    dict.set_item("max_removal_quantity", reference.max_removal_quantity)?;
    dict.set_item("benchmark_order_up_to", reference.benchmark_order_up_to)?;
    dict.set_item(
        "benchmark_remove_down_to",
        reference.benchmark_remove_down_to,
    )?;
    dict.set_item(
        "benchmark_returnable_buffer",
        reference.benchmark_returnable_buffer,
    )?;
    dict.set_item("literature_verified", false)?;
    dict.set_item(
        "verification_source",
        "repo_native_instance_not_verified_against_literature",
    )?;
    dict.set_item("notes", reference.notes)?;
    Ok(dict.into_any().unbind().into())
}

fn verification_reference_to_py(
    py: Python<'_>,
    reference: &ExactVerificationReference,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("source", reference.source)?;
    dict.set_item("url", reference.url)?;
    dict.set_item("literature_verified", reference.literature_verified)?;
    dict.set_item("verification_source", reference.verification_source)?;
    dict.set_item("periods", reference.periods)?;
    dict.set_item("discount_factor", reference.discount_factor)?;
    dict.set_item("initial_inventory_level", reference.initial_inventory_level)?;
    dict.set_item(
        "initial_returnable_inventory",
        reference.initial_returnable_inventory,
    )?;
    dict.set_item("returnable_purchase_cap", reference.returnable_purchase_cap)?;
    dict.set_item("purchase_cost_per_unit", reference.purchase_cost_per_unit)?;
    dict.set_item("return_value_per_unit", reference.return_value_per_unit)?;
    dict.set_item(
        "liquidation_value_per_unit",
        reference.liquidation_value_per_unit,
    )?;
    dict.set_item("holding_cost_per_unit", reference.holding_cost_per_unit)?;
    dict.set_item("shortage_cost_per_unit", reference.shortage_cost_per_unit)?;
    dict.set_item("demand_support", reference.demand_support.to_vec())?;
    dict.set_item(
        "demand_probabilities",
        reference.demand_probabilities.to_vec(),
    )?;
    dict.set_item("max_purchase_quantity", reference.max_purchase_quantity)?;
    dict.set_item("max_removal_quantity", reference.max_removal_quantity)?;
    dict.set_item(
        "interval_stock_order_up_to",
        reference.interval_stock_order_up_to,
    )?;
    dict.set_item(
        "interval_stock_remove_down_to",
        reference.interval_stock_remove_down_to,
    )?;
    dict.set_item(
        "returnability_buffer_order_up_to",
        reference.returnability_buffer_order_up_to,
    )?;
    dict.set_item(
        "returnability_buffer_remove_down_to",
        reference.returnability_buffer_remove_down_to,
    )?;
    dict.set_item("returnability_buffer", reference.returnability_buffer)?;
    dict.set_item("notes", reference.notes)?;
    Ok(dict.into_any().unbind().into())
}

fn build_rollout_config(
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    periods: usize,
    demand_kind: &str,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<ProcurementRemovalRolloutConfig> {
    Ok(ProcurementRemovalRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_kind: parse_demand_distribution_kind(demand_kind)?,
        demand_mean,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    })
}

fn simulation_summary_to_py(
    py: Python<'_>,
    summary: &PolicySimulationSummary,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("mean_discounted_cost", summary.mean_discounted_cost)?;
    dict.set_item("std_discounted_cost", summary.std_discounted_cost)?;
    dict.set_item("min_discounted_cost", summary.min_discounted_cost)?;
    dict.set_item("max_discounted_cost", summary.max_discounted_cost)?;
    dict.set_item("num_seeds", summary.num_seeds)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn procurement_removal_inventory_primary_reference_instance(py: Python<'_>) -> PyResult<PyObject> {
    primary_reference_to_py(py, &PRIMARY_REFERENCE_INSTANCE)
}

#[pyfunction]
fn procurement_removal_inventory_exact_verification_instance(py: Python<'_>) -> PyResult<PyObject> {
    verification_reference_to_py(py, &VERIFICATION_PROBLEM_INSTANCE)
}

#[pyfunction]
fn procurement_removal_inventory_exact_dp_summary(py: Python<'_>) -> PyResult<PyObject> {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)?;
    let interval = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "interval_stock")?;
    let buffer = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "returnability_buffer_interval_stock",
    )?;

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "verification_reference",
        verification_reference_to_py(py, &VERIFICATION_PROBLEM_INSTANCE)?,
    )?;
    dict.set_item("optimal_discounted_cost", optimal.discounted_cost)?;
    dict.set_item(
        "optimal_first_action",
        (optimal.first_action.0, optimal.first_action.1),
    )?;
    dict.set_item("interval_stock_discounted_cost", interval.discounted_cost)?;
    dict.set_item(
        "interval_stock_first_action",
        (interval.first_action.0, interval.first_action.1),
    )?;
    dict.set_item(
        "returnability_buffer_discounted_cost",
        buffer.discounted_cost,
    )?;
    dict.set_item(
        "returnability_buffer_first_action",
        (buffer.first_action.0, buffer.first_action.1),
    )?;
    dict.set_item(
        "interval_stock_gap_to_optimal",
        interval.discounted_cost - optimal.discounted_cost,
    )?;
    dict.set_item(
        "returnability_buffer_gap_to_optimal",
        buffer.discounted_cost - optimal.discounted_cost,
    )?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn procurement_removal_inventory_build_raw_state(
    inventory_level: usize,
    returnable_inventory: usize,
) -> PyResult<Vec<f32>> {
    let state = initialize_state(inventory_level, returnable_inventory)?;
    build_raw_state(&state)
}

#[pyfunction]
#[pyo3(signature = (
    inventory_level,
    returnable_inventory,
    purchase_quantity,
    removal_quantity,
    realized_demand,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit
))]
fn procurement_removal_inventory_step(
    py: Python<'_>,
    inventory_level: usize,
    returnable_inventory: usize,
    purchase_quantity: usize,
    removal_quantity: usize,
    realized_demand: usize,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
) -> PyResult<PyObject> {
    let state = initialize_state(inventory_level, returnable_inventory)?;
    let outcome = step_state(
        &state,
        purchase_quantity,
        removal_quantity,
        realized_demand,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
    )?;
    let dict = PyDict::new_bound(py);
    dict.set_item("next_period", outcome.next_state.period)?;
    dict.set_item("next_inventory_level", outcome.next_state.inventory_level)?;
    dict.set_item(
        "next_returnable_inventory",
        outcome.next_state.returnable_inventory,
    )?;
    dict.set_item("returned_units", outcome.returned_units)?;
    dict.set_item("liquidated_units", outcome.liquidated_units)?;
    dict.set_item("sales", outcome.sales)?;
    dict.set_item("shortage", outcome.shortage)?;
    dict.set_item("purchase_cost", outcome.purchase_cost)?;
    dict.set_item("removal_credit", outcome.removal_credit)?;
    dict.set_item("holding_cost", outcome.holding_cost)?;
    dict.set_item("shortage_cost", outcome.shortage_cost)?;
    dict.set_item("period_cost", outcome.period_cost)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn procurement_removal_inventory_interval_stock_action(
    inventory_level: usize,
    returnable_inventory: usize,
    order_up_to: usize,
    remove_down_to: usize,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
) -> PyResult<(usize, usize)> {
    let state = initialize_state(inventory_level, returnable_inventory)?;
    interval_stock_action(
        &state,
        order_up_to,
        remove_down_to,
        max_purchase_quantity,
        max_removal_quantity,
    )
}

#[pyfunction]
fn procurement_removal_inventory_returnability_buffer_interval_stock_action(
    inventory_level: usize,
    returnable_inventory: usize,
    order_up_to: usize,
    remove_down_to: usize,
    returnable_buffer: usize,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
) -> PyResult<(usize, usize)> {
    let state = initialize_state(inventory_level, returnable_inventory)?;
    returnability_buffer_interval_stock_action(
        &state,
        order_up_to,
        remove_down_to,
        returnable_buffer,
        max_purchase_quantity,
        max_removal_quantity,
    )
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    inventory_level,
    returnable_inventory,
    periods,
    demand_kind,
    demand_mean,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit,
    max_purchase_quantity,
    max_removal_quantity,
    seed=1234,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None
))]
fn procurement_removal_inventory_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_level: usize,
    returnable_inventory: usize,
    periods: usize,
    demand_kind: &str,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    seed: u64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = initialize_state(inventory_level, returnable_inventory)?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        periods,
        demand_kind,
        demand_mean,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
        discount_factor,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
    )?;
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
    inventory_level,
    returnable_inventory,
    periods,
    demand_kind,
    demand_mean,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit,
    max_purchase_quantity,
    max_removal_quantity,
    seeds,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None
))]
fn procurement_removal_inventory_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_level: usize,
    returnable_inventory: usize,
    periods: usize,
    demand_kind: &str,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    seeds: Vec<u64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let initial_state = initialize_state(inventory_level, returnable_inventory)?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        periods,
        demand_kind,
        demand_mean,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
        discount_factor,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
    )?;
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
    inventory_level,
    returnable_inventory,
    demands,
    demand_mean,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit,
    max_purchase_quantity,
    max_removal_quantity,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None
))]
fn procurement_removal_inventory_soft_tree_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_level: usize,
    returnable_inventory: usize,
    demands: Vec<usize>,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = initialize_state(inventory_level, returnable_inventory)?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        demands.len(),
        "deterministic",
        demand_mean,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
        discount_factor,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
    )?;
    rollout_from_demands(&flat_params, &config, &initial_state, &demands)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    inventory_level,
    returnable_inventory,
    periods,
    seed,
    demand_kind,
    demand_mean,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit,
    max_purchase_quantity,
    max_removal_quantity,
    discount_factor=0.99
))]
fn procurement_removal_inventory_policy_rollout(
    policy_name: &str,
    params: Vec<usize>,
    inventory_level: usize,
    returnable_inventory: usize,
    periods: usize,
    seed: u64,
    demand_kind: &str,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
) -> PyResult<f64> {
    let initial_state = initialize_state(inventory_level, returnable_inventory)?;
    policy_rollout(
        policy_name,
        &params,
        &initial_state,
        periods,
        seed,
        parse_demand_distribution_kind(demand_kind)?,
        demand_mean,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
        discount_factor,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    inventory_level,
    returnable_inventory,
    periods,
    seeds,
    demand_kind,
    demand_mean,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit,
    max_purchase_quantity,
    max_removal_quantity,
    discount_factor=0.99
))]
fn procurement_removal_inventory_simulate_policy(
    py: Python<'_>,
    policy_name: &str,
    params: Vec<usize>,
    inventory_level: usize,
    returnable_inventory: usize,
    periods: usize,
    seeds: Vec<u64>,
    demand_kind: &str,
    demand_mean: f64,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
) -> PyResult<PyObject> {
    let initial_state = initialize_state(inventory_level, returnable_inventory)?;
    let summary = simulate_policy(
        policy_name,
        &params,
        &initial_state,
        periods,
        &seeds,
        parse_demand_distribution_kind(demand_kind)?,
        demand_mean,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
        discount_factor,
    )?;
    simulation_summary_to_py(py, &summary)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    inventory_level,
    returnable_inventory,
    demands,
    returnable_purchase_cap,
    purchase_cost_per_unit,
    return_value_per_unit,
    liquidation_value_per_unit,
    holding_cost_per_unit,
    shortage_cost_per_unit,
    max_purchase_quantity,
    max_removal_quantity,
    discount_factor=0.99
))]
fn procurement_removal_inventory_policy_rollout_from_demands(
    policy_name: &str,
    params: Vec<usize>,
    inventory_level: usize,
    returnable_inventory: usize,
    demands: Vec<usize>,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
    discount_factor: f64,
) -> PyResult<f64> {
    let initial_state = initialize_state(inventory_level, returnable_inventory)?;
    policy_rollout_from_demands(
        policy_name,
        &params,
        &initial_state,
        &demands,
        returnable_purchase_cap,
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
        max_purchase_quantity,
        max_removal_quantity,
        discount_factor,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_primary_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_exact_verification_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_exact_dp_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_build_raw_state,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(procurement_removal_inventory_step, m)?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_interval_stock_action,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_returnability_buffer_interval_stock_action,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_soft_tree_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_soft_tree_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_policy_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_simulate_policy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        procurement_removal_inventory_policy_rollout_from_demands,
        m
    )?)?;
    Ok(())
}
