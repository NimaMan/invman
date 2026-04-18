use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::random_yield_inventory::demand::parse_demand_distribution_kind;
use crate::problems::random_yield_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::random_yield_inventory::heuristics::{
    policy_discounted_cost_summary, policy_rollout_from_paths, simulate_policy,
    weighted_newsvendor_order_quantity, yield_inflated_base_stock_order_quantity,
    yield_inflated_base_stock_parameters, DiscountedCostSummary,
};
use crate::problems::random_yield_inventory::literature::{
    ExactVerificationReference, LiteratureBenchmarkFamily, RandomYieldReferenceInstance,
    LITERATURE_BENCHMARK_FAMILIES, PRIMARY_REFERENCE_INSTANCE, VERIFICATION_PROBLEM_INSTANCE,
};
use crate::problems::random_yield_inventory::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_paths,
    RandomYieldInventoryRolloutConfig,
};

fn primary_reference_to_py(
    py: Python<'_>,
    reference: &RandomYieldReferenceInstance,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", reference.name)?;
    dict.set_item("source", reference.source)?;
    dict.set_item("url", reference.url)?;
    dict.set_item("literature_verified", reference.literature_verified)?;
    dict.set_item("verification_source", reference.verification_source)?;
    dict.set_item("periods", reference.periods)?;
    dict.set_item("lead_time", reference.lead_time)?;
    dict.set_item("demand_mean", reference.demand_mean)?;
    dict.set_item("success_probability", reference.success_probability)?;
    dict.set_item("holding_cost", reference.holding_cost)?;
    dict.set_item("shortage_cost", reference.shortage_cost)?;
    dict.set_item("procurement_cost", reference.procurement_cost)?;
    dict.set_item("discount_factor", reference.discount_factor)?;
    dict.set_item("initial_inventory_level", reference.initial_inventory_level)?;
    dict.set_item(
        "initial_pipeline_orders",
        reference.initial_pipeline_orders.to_vec(),
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
    dict.set_item("lead_time", reference.lead_time)?;
    dict.set_item("success_probability", reference.success_probability)?;
    dict.set_item("holding_cost", reference.holding_cost)?;
    dict.set_item("shortage_cost", reference.shortage_cost)?;
    dict.set_item("procurement_cost", reference.procurement_cost)?;
    dict.set_item("discount_factor", reference.discount_factor)?;
    dict.set_item("initial_inventory_level", reference.initial_inventory_level)?;
    dict.set_item(
        "initial_pipeline_orders",
        reference.initial_pipeline_orders.to_vec(),
    )?;
    dict.set_item("demand_support", reference.demand_support.to_vec())?;
    dict.set_item(
        "demand_probabilities",
        reference.demand_probabilities.to_vec(),
    )?;
    dict.set_item("max_order_quantity", reference.max_order_quantity)?;
    Ok(dict.into_any().unbind().into())
}

fn discounted_cost_summary_to_py(
    py: Python<'_>,
    summary: &DiscountedCostSummary,
    params: &[f64],
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("params", params.to_vec())?;
    dict.set_item("mean_cost", summary.mean_cost)?;
    dict.set_item("cost_std", summary.cost_std)?;
    dict.set_item("min_cost", summary.min_cost)?;
    dict.set_item("max_cost", summary.max_cost)?;
    dict.set_item("num_samples", summary.num_samples)?;
    Ok(dict.into_any().unbind().into())
}

fn literature_family_to_py(
    py: Python<'_>,
    family: &LiteratureBenchmarkFamily,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", family.name)?;
    dict.set_item("source", family.source)?;
    dict.set_item("url", family.url)?;
    dict.set_item("horizon_type", family.horizon_type)?;
    dict.set_item("demand_family", family.demand_family)?;
    dict.set_item("yield_model", family.yield_model)?;
    dict.set_item("model_match", family.model_match)?;
    dict.set_item("access_level", family.access_level)?;
    dict.set_item(
        "reported_numbers_available",
        family.reported_numbers_available,
    )?;
    dict.set_item("repo_assertion_basis", family.repo_assertion_basis)?;
    dict.set_item("benchmark_policies", family.benchmark_policies.to_vec())?;
    dict.set_item("lead_times", family.lead_times.to_vec())?;
    dict.set_item("demand_means", family.demand_means.to_vec())?;
    dict.set_item("demand_cvs", family.demand_cvs.to_vec())?;
    dict.set_item(
        "success_probabilities",
        family.success_probabilities.to_vec(),
    )?;
    dict.set_item("critical_ratios", family.critical_ratios.to_vec())?;
    dict.set_item(
        "yield_rate_mean_cv_pairs",
        family.yield_rate_mean_cv_pairs.to_vec(),
    )?;
    dict.set_item("notes", family.notes)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn random_yield_inventory_primary_reference_instance(py: Python<'_>) -> PyResult<PyObject> {
    primary_reference_to_py(py, &PRIMARY_REFERENCE_INSTANCE)
}

#[pyfunction]
fn random_yield_inventory_literature_benchmark_families(py: Python<'_>) -> PyResult<Vec<PyObject>> {
    LITERATURE_BENCHMARK_FAMILIES
        .iter()
        .map(|family| literature_family_to_py(py, family))
        .collect()
}

#[pyfunction]
fn random_yield_inventory_exact_verification_instance(py: Python<'_>) -> PyResult<PyObject> {
    verification_reference_to_py(py, &VERIFICATION_PROBLEM_INSTANCE)
}

#[pyfunction]
fn random_yield_inventory_exact_dp_summary(py: Python<'_>) -> PyResult<PyObject> {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)?;
    let linear_inflation =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "linear_inflation")?;
    let weighted_newsvendor =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "weighted_newsvendor")?;

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "verification_reference",
        verification_reference_to_py(py, &VERIFICATION_PROBLEM_INSTANCE)?,
    )?;
    dict.set_item("optimal_discounted_cost", optimal.discounted_cost)?;
    dict.set_item("optimal_first_action", optimal.first_action)?;
    dict.set_item(
        "linear_inflation_discounted_cost",
        linear_inflation.discounted_cost,
    )?;
    dict.set_item(
        "linear_inflation_first_action",
        linear_inflation.first_action,
    )?;
    dict.set_item(
        "weighted_newsvendor_discounted_cost",
        weighted_newsvendor.discounted_cost,
    )?;
    dict.set_item(
        "weighted_newsvendor_first_action",
        weighted_newsvendor.first_action,
    )?;
    dict.set_item(
        "linear_inflation_gap_to_optimal",
        linear_inflation.discounted_cost - optimal.discounted_cost,
    )?;
    dict.set_item(
        "weighted_newsvendor_gap_to_optimal",
        weighted_newsvendor.discounted_cost - optimal.discounted_cost,
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
    initial_inventory_level,
    pipeline_orders,
    periods,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost,
    procurement_cost,
    seed=1234,
    discount_factor=0.99,
    demand_distribution="poisson",
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None
))]
fn random_yield_inventory_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_inventory_level: f64,
    pipeline_orders: Vec<f64>,
    periods: usize,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    seed: u64,
    discount_factor: f64,
    demand_distribution: &str,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(initial_inventory_level, &pipeline_orders)?;
    let config = RandomYieldInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_mean,
        demand_kind: parse_demand_distribution_kind(demand_distribution)?,
        success_probability,
        holding_cost,
        shortage_cost,
        procurement_cost,
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
    initial_inventory_level,
    pipeline_orders,
    periods,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost,
    procurement_cost,
    seeds,
    discount_factor=0.99,
    demand_distribution="poisson",
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None
))]
fn random_yield_inventory_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_inventory_level: f64,
    pipeline_orders: Vec<f64>,
    periods: usize,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    seeds: Vec<u64>,
    discount_factor: f64,
    demand_distribution: &str,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let initial_state = build_initial_state(initial_inventory_level, &pipeline_orders)?;
    let config = RandomYieldInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_mean,
        demand_kind: parse_demand_distribution_kind(demand_distribution)?,
        success_probability,
        holding_cost,
        shortage_cost,
        procurement_cost,
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
    initial_inventory_level,
    pipeline_orders,
    demands,
    arrival_outcomes,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost,
    procurement_cost,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None
))]
fn random_yield_inventory_soft_tree_rollout_from_paths(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_inventory_level: f64,
    pipeline_orders: Vec<f64>,
    demands: Vec<f64>,
    arrival_outcomes: Vec<bool>,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(initial_inventory_level, &pipeline_orders)?;
    let config = RandomYieldInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods: demands.len(),
        demand_mean,
        demand_kind:
            crate::problems::random_yield_inventory::demand::DemandDistributionKind::Deterministic,
        success_probability,
        holding_cost,
        shortage_cost,
        procurement_cost,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    rollout_from_paths(
        &flat_params,
        &config,
        &initial_state,
        &demands,
        &arrival_outcomes,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    initial_inventory_level,
    pipeline_orders,
    demands,
    arrival_outcomes,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost,
    procurement_cost,
    discount_factor=0.99
))]
fn random_yield_inventory_policy_rollout_from_paths(
    policy_name: &str,
    params: Vec<f64>,
    initial_inventory_level: f64,
    pipeline_orders: Vec<f64>,
    demands: Vec<f64>,
    arrival_outcomes: Vec<bool>,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
) -> PyResult<f64> {
    let initial_state = build_initial_state(initial_inventory_level, &pipeline_orders)?;
    policy_rollout_from_paths(
        policy_name,
        &params,
        &initial_state,
        demand_mean,
        &demands,
        &arrival_outcomes,
        holding_cost,
        shortage_cost,
        procurement_cost,
        success_probability,
        discount_factor,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    initial_inventory_level,
    pipeline_orders,
    periods,
    seeds,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost,
    procurement_cost,
    discount_factor=0.99,
    demand_distribution="poisson"
))]
fn random_yield_inventory_policy_discounted_cost_summary(
    py: Python<'_>,
    policy_name: &str,
    params: Vec<f64>,
    initial_inventory_level: f64,
    pipeline_orders: Vec<f64>,
    periods: usize,
    seeds: Vec<u64>,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
    demand_distribution: &str,
) -> PyResult<PyObject> {
    let summary = policy_discounted_cost_summary(
        policy_name,
        &params,
        initial_inventory_level,
        &pipeline_orders,
        periods,
        &seeds,
        demand_mean,
        parse_demand_distribution_kind(demand_distribution)?,
        success_probability,
        holding_cost,
        shortage_cost,
        procurement_cost,
        discount_factor,
    )?;
    discounted_cost_summary_to_py(py, &summary, &params)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    initial_inventory_level,
    pipeline_orders,
    periods,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost,
    procurement_cost,
    replications=1000,
    seed=1234,
    discount_factor=0.99,
    demand_distribution="poisson"
))]
fn random_yield_inventory_simulate_policy(
    policy_name: &str,
    params: Vec<f64>,
    initial_inventory_level: f64,
    pipeline_orders: Vec<f64>,
    periods: usize,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    replications: usize,
    seed: u64,
    discount_factor: f64,
    demand_distribution: &str,
) -> PyResult<(f64, f64)> {
    let summary = simulate_policy(
        policy_name,
        &params,
        initial_inventory_level,
        &pipeline_orders,
        periods,
        replications,
        seed,
        demand_mean,
        parse_demand_distribution_kind(demand_distribution)?,
        success_probability,
        holding_cost,
        shortage_cost,
        procurement_cost,
        discount_factor,
    )?;
    Ok((summary.mean_cost, summary.cost_std))
}

#[pyfunction]
#[pyo3(signature = (
    demand_mean,
    success_probability,
    lead_time,
    holding_cost,
    shortage_cost
))]
fn random_yield_inventory_linear_inflation_parameters(
    demand_mean: f64,
    success_probability: f64,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
) -> PyResult<(f64, f64)> {
    yield_inflated_base_stock_parameters(
        demand_mean,
        success_probability,
        lead_time,
        holding_cost,
        shortage_cost,
    )
}

#[pyfunction]
#[pyo3(signature = (
    inventory_level,
    pipeline_orders,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost
))]
fn random_yield_inventory_weighted_newsvendor_order(
    inventory_level: f64,
    pipeline_orders: Vec<f64>,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
) -> PyResult<f64> {
    let state = build_initial_state(inventory_level, &pipeline_orders)?;
    weighted_newsvendor_order_quantity(
        &state,
        demand_mean,
        success_probability,
        holding_cost,
        shortage_cost,
    )
}

#[pyfunction]
#[pyo3(signature = (
    inventory_level,
    pipeline_orders,
    demand_mean,
    success_probability,
    holding_cost,
    shortage_cost
))]
fn random_yield_inventory_yield_inflated_base_stock_order(
    inventory_level: f64,
    pipeline_orders: Vec<f64>,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
) -> PyResult<f64> {
    let state = build_initial_state(inventory_level, &pipeline_orders)?;
    yield_inflated_base_stock_order_quantity(
        &state,
        demand_mean,
        success_probability,
        holding_cost,
        shortage_cost,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_primary_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_literature_benchmark_families,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_exact_verification_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_exact_dp_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_soft_tree_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_soft_tree_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_policy_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_policy_discounted_cost_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(random_yield_inventory_simulate_policy, m)?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_linear_inflation_parameters,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_weighted_newsvendor_order,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        random_yield_inventory_yield_inflated_base_stock_order,
        m
    )?)?;
    Ok(())
}
