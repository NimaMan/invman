use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::joint_replenishment::demand::DemandRange;
use crate::problems::joint_replenishment::env::initialize_state;
use crate::problems::joint_replenishment::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::joint_replenishment::heuristics::{
    dynamic_order_up_to_order_quantities, minimum_order_quantity_order_quantities,
    policy_rollout_from_paths, simulate_policy,
};
use crate::problems::joint_replenishment::references::{
    ExactVerificationReference, JointReplenishmentReferenceInstance, PRIMARY_REFERENCE_INSTANCE,
    SMALL_SCALE_SETTINGS, VERIFICATION_PROBLEM_INSTANCE,
};
use crate::problems::joint_replenishment::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_paths,
    JointReplenishmentRolloutConfig,
};

fn build_demand_ranges(
    demand_lows: Vec<usize>,
    demand_highs: Vec<usize>,
) -> PyResult<Vec<DemandRange>> {
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

fn reference_instance_to_py(
    py: Python<'_>,
    reference: &JointReplenishmentReferenceInstance,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", reference.name)?;
    dict.set_item("source", reference.source)?;
    dict.set_item("url", reference.url)?;
    dict.set_item("num_items", reference.num_items)?;
    dict.set_item("truck_capacity", reference.truck_capacity)?;
    dict.set_item("major_order_cost", reference.major_order_cost)?;
    dict.set_item("minor_order_costs", reference.minor_order_costs.to_vec())?;
    dict.set_item("holding_costs", reference.holding_costs.to_vec())?;
    dict.set_item("shortage_costs", reference.shortage_costs.to_vec())?;
    dict.set_item(
        "demand_lows",
        reference
            .demand_ranges
            .iter()
            .map(|range| range.low)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "demand_highs",
        reference
            .demand_ranges
            .iter()
            .map(|range| range.high)
            .collect::<Vec<_>>(),
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
    dict.set_item("truck_capacity", reference.truck_capacity)?;
    dict.set_item("max_order_quantities", reference.max_order_quantities.to_vec())?;
    dict.set_item(
        "initial_inventory_levels",
        reference.initial_inventory_levels.to_vec(),
    )?;
    dict.set_item("major_order_cost", reference.major_order_cost)?;
    dict.set_item("minor_order_costs", reference.minor_order_costs.to_vec())?;
    dict.set_item("holding_costs", reference.holding_costs.to_vec())?;
    dict.set_item("shortage_costs", reference.shortage_costs.to_vec())?;
    dict.set_item(
        "demand_lows",
        reference
            .demand_ranges
            .iter()
            .map(|range| range.low)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "demand_highs",
        reference
            .demand_ranges
            .iter()
            .map(|range| range.high)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("moq_item_targets", reference.moq_item_targets.to_vec())?;
    dict.set_item("moq_review_period", reference.moq_review_period)?;
    dict.set_item("moq_rounding_threshold", reference.moq_rounding_threshold)?;
    dict.set_item("dynout_item_targets", reference.dynout_item_targets.to_vec())?;
    dict.set_item(
        "expected_optimal_discounted_cost",
        reference.expected_optimal_discounted_cost,
    )?;
    dict.set_item(
        "expected_optimal_first_action",
        reference.expected_optimal_first_action.to_vec(),
    )?;
    dict.set_item(
        "expected_moq_discounted_cost",
        reference.expected_moq_discounted_cost,
    )?;
    dict.set_item(
        "expected_moq_first_action",
        reference.expected_moq_first_action.to_vec(),
    )?;
    dict.set_item(
        "expected_dynout_discounted_cost",
        reference.expected_dynout_discounted_cost,
    )?;
    dict.set_item(
        "expected_dynout_first_action",
        reference.expected_dynout_first_action.to_vec(),
    )?;
    dict.set_item("notes", reference.notes)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn joint_replenishment_primary_reference_instance(py: Python<'_>) -> PyResult<PyObject> {
    reference_instance_to_py(py, &PRIMARY_REFERENCE_INSTANCE)
}

#[pyfunction]
fn joint_replenishment_list_reference_instances(py: Python<'_>) -> PyResult<Vec<PyObject>> {
    SMALL_SCALE_SETTINGS
        .iter()
        .map(|reference| reference_instance_to_py(py, reference))
        .collect()
}

#[pyfunction]
fn joint_replenishment_get_reference_instance(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    let reference = SMALL_SCALE_SETTINGS
        .iter()
        .find(|reference| reference.name == name)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "unknown joint_replenishment reference instance '{name}'",
            ))
        })?;
    reference_instance_to_py(py, reference)
}

#[pyfunction]
fn joint_replenishment_exact_verification_instance(py: Python<'_>) -> PyResult<PyObject> {
    verification_reference_to_py(py, &VERIFICATION_PROBLEM_INSTANCE)
}

#[pyfunction]
fn joint_replenishment_exact_dp_summary(py: Python<'_>) -> PyResult<PyObject> {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)?;
    let moq = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "minimum_order_quantity")?;
    let dynout = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "dynamic_order_up_to")?;

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "verification_reference",
        verification_reference_to_py(py, &VERIFICATION_PROBLEM_INSTANCE)?,
    )?;
    dict.set_item("optimal_discounted_cost", optimal.discounted_cost)?;
    dict.set_item("optimal_first_action", optimal.first_action.to_vec())?;
    dict.set_item(
        "matches_expected_optimal_discounted_cost",
        (optimal.discounted_cost - VERIFICATION_PROBLEM_INSTANCE.expected_optimal_discounted_cost)
            .abs()
            < 1e-9,
    )?;
    dict.set_item(
        "matches_expected_optimal_first_action",
        optimal.first_action == [
            VERIFICATION_PROBLEM_INSTANCE.expected_optimal_first_action[0],
            VERIFICATION_PROBLEM_INSTANCE.expected_optimal_first_action[1],
        ],
    )?;
    dict.set_item("moq_discounted_cost", moq.discounted_cost)?;
    dict.set_item("moq_first_action", moq.first_action.to_vec())?;
    dict.set_item("dynout_discounted_cost", dynout.discounted_cost)?;
    dict.set_item("dynout_first_action", dynout.first_action.to_vec())?;
    dict.set_item("moq_gap_to_optimal", moq.discounted_cost - optimal.discounted_cost)?;
    dict.set_item(
        "dynout_gap_to_optimal",
        dynout.discounted_cost - optimal.discounted_cost,
    )?;
    Ok(dict.into_any().unbind().into())
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
    leaf_type="linear",
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
    leaf_type="linear",
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
    leaf_type="linear",
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
    m.add_function(wrap_pyfunction!(
        joint_replenishment_primary_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        joint_replenishment_list_reference_instances,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        joint_replenishment_get_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        joint_replenishment_exact_verification_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(joint_replenishment_exact_dp_summary, m)?)?;
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
