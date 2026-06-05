use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::one_warehouse_multi_retailer::allocation::parse_allocation_policy;
use crate::problems::one_warehouse_multi_retailer::demand::{
    parse_demand_distribution_kind, DemandDistributionKind, DemandModel,
};
use crate::problems::one_warehouse_multi_retailer::env::{
    build_raw_state, parse_customer_behavior_model, OneWarehouseMultiRetailerState,
};
use crate::problems::one_warehouse_multi_retailer::finite_horizon_dp::{
    evaluate_echelon_base_stock_policy, evaluate_named_heuristic, evaluate_soft_tree_policy,
    solve_optimal_policy, ExactSoftTreeConfig,
};
use crate::problems::one_warehouse_multi_retailer::heuristics::{
    echelon_base_stock_orders, policy_rollout_from_paths, simulate_policy,
};
use crate::problems::one_warehouse_multi_retailer::references::{
    ExactVerificationReference, OneWarehouseMultiRetailerReferenceInstance,
    PublishedBenchmarkReference, KAYNOV_2024_REFERENCE, PRIMARY_REFERENCE_INSTANCE,
    TABLE_A3_INSTANCES, VERIFICATION_PROBLEM_INSTANCE,
};
use crate::problems::one_warehouse_multi_retailer::rollout::{
    build_initial_state, parse_policy_action_mode, parse_policy_state_mode, population_rollout,
    rollout, rollout_from_paths, OneWarehouseMultiRetailerRolloutConfig,
};

fn allocation_policy_to_str(
    policy: crate::problems::one_warehouse_multi_retailer::allocation::AllocationPolicy,
) -> &'static str {
    match policy {
        crate::problems::one_warehouse_multi_retailer::allocation::AllocationPolicy::Proportional => "proportional",
        crate::problems::one_warehouse_multi_retailer::allocation::AllocationPolicy::RandomSequential => "random_sequential",
        crate::problems::one_warehouse_multi_retailer::allocation::AllocationPolicy::MinShortage => "min_shortage",
    }
}

fn customer_behavior_to_str(
    model: crate::problems::one_warehouse_multi_retailer::env::CustomerBehaviorModel,
) -> &'static str {
    match model {
        crate::problems::one_warehouse_multi_retailer::env::CustomerBehaviorModel::LostSales => "lost_sales",
        crate::problems::one_warehouse_multi_retailer::env::CustomerBehaviorModel::Backorder => "backorder",
        crate::problems::one_warehouse_multi_retailer::env::CustomerBehaviorModel::PartialBackorder => "partial_backorder",
    }
}

fn demand_kind_to_str(kind: DemandDistributionKind) -> &'static str {
    match kind {
        DemandDistributionKind::Poisson => "poisson",
        DemandDistributionKind::RoundedNormal => "rounded_normal",
        DemandDistributionKind::DiscreteUniform => "discrete_uniform",
        DemandDistributionKind::Deterministic => "deterministic",
    }
}

fn build_demand_models(
    demand_kinds: Vec<String>,
    demand_param1: Vec<f64>,
    demand_param2: Vec<f64>,
) -> PyResult<Vec<DemandModel>> {
    if demand_kinds.len() != demand_param1.len() || demand_kinds.len() != demand_param2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "demand_kinds, demand_param1, and demand_param2 must have the same length",
        ));
    }
    demand_kinds
        .iter()
        .zip(demand_param1.iter())
        .zip(demand_param2.iter())
        .map(|((kind, param1), param2)| {
            Ok(DemandModel {
                kind: parse_demand_distribution_kind(kind)?,
                param1: *param1,
                param2: *param2,
            })
        })
        .collect()
}

fn benchmark_reference_to_py(
    py: Python<'_>,
    reference: &PublishedBenchmarkReference,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("source", reference.source)?;
    dict.set_item("url", reference.url)?;
    dict.set_item("benchmark_policies", reference.benchmark_policies.to_vec())?;
    dict.set_item("notes", reference.notes)?;
    Ok(dict.into_any().unbind().into())
}

fn reference_instance_to_py(
    py: Python<'_>,
    reference: &OneWarehouseMultiRetailerReferenceInstance,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", reference.name)?;
    dict.set_item("source", reference.source)?;
    dict.set_item("url", reference.url)?;
    dict.set_item("literature_verified", reference.literature_verified)?;
    dict.set_item("benchmark_periods", reference.benchmark_periods)?;
    dict.set_item("benchmark_replications", reference.benchmark_replications)?;
    dict.set_item(
        "customer_behavior",
        customer_behavior_to_str(reference.customer_behavior),
    )?;
    dict.set_item("warehouse_lead_time", reference.warehouse_lead_time)?;
    dict.set_item(
        "retailer_lead_times",
        reference.retailer_lead_times.to_vec(),
    )?;
    dict.set_item(
        "demand_kinds",
        reference
            .demand_models
            .iter()
            .map(|model| demand_kind_to_str(model.kind).to_string())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "demand_param1",
        reference
            .demand_models
            .iter()
            .map(|model| model.param1)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "demand_param2",
        reference
            .demand_models
            .iter()
            .map(|model| model.param2)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("holding_cost_warehouse", reference.holding_cost_warehouse)?;
    dict.set_item(
        "holding_cost_retailers",
        reference.holding_cost_retailers.to_vec(),
    )?;
    dict.set_item(
        "penalty_costs_retailers",
        reference.penalty_costs_retailers.to_vec(),
    )?;
    dict.set_item(
        "emergency_shipment_probability",
        reference.emergency_shipment_probability,
    )?;
    if let Some(row) = reference.published_min_shortage_benchmark {
        let row_dict = PyDict::new_bound(py);
        row_dict.set_item("policy_name", row.policy_name)?;
        row_dict.set_item(
            "allocation_policy",
            row.allocation_policy.map(allocation_policy_to_str),
        )?;
        row_dict.set_item("mean_cost", row.mean_cost)?;
        row_dict.set_item("standard_error", row.standard_error)?;
        row_dict.set_item("relative_gap_percent", row.relative_gap_percent)?;
        dict.set_item("published_min_shortage_benchmark", row_dict)?;
    } else {
        dict.set_item("published_min_shortage_benchmark", py.None())?;
    }
    if let Some(row) = reference.published_proportional_benchmark {
        let row_dict = PyDict::new_bound(py);
        row_dict.set_item("policy_name", row.policy_name)?;
        row_dict.set_item(
            "allocation_policy",
            row.allocation_policy.map(allocation_policy_to_str),
        )?;
        row_dict.set_item("mean_cost", row.mean_cost)?;
        row_dict.set_item("standard_error", row.standard_error)?;
        row_dict.set_item("relative_gap_percent", row.relative_gap_percent)?;
        dict.set_item("published_proportional_benchmark", row_dict)?;
    } else {
        dict.set_item("published_proportional_benchmark", py.None())?;
    }
    if let Some(row) = reference.published_ppo_benchmark {
        let row_dict = PyDict::new_bound(py);
        row_dict.set_item("policy_name", row.policy_name)?;
        row_dict.set_item(
            "allocation_policy",
            row.allocation_policy.map(allocation_policy_to_str),
        )?;
        row_dict.set_item("mean_cost", row.mean_cost)?;
        row_dict.set_item("standard_error", row.standard_error)?;
        row_dict.set_item("relative_gap_percent", row.relative_gap_percent)?;
        dict.set_item("published_ppo_benchmark", row_dict)?;
    } else {
        dict.set_item("published_ppo_benchmark", py.None())?;
    }
    dict.set_item("notes", reference.notes)?;
    Ok(dict.into_any().unbind().into())
}

fn exact_reference_to_py(
    py: Python<'_>,
    reference: &ExactVerificationReference,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("source", reference.source)?;
    dict.set_item("url", reference.url)?;
    dict.set_item("literature_verified", reference.literature_verified)?;
    dict.set_item(
        "customer_behavior",
        customer_behavior_to_str(reference.customer_behavior),
    )?;
    dict.set_item("periods", reference.periods)?;
    dict.set_item("discount_factor", reference.discount_factor)?;
    dict.set_item("warehouse_lead_time", reference.warehouse_lead_time)?;
    dict.set_item(
        "retailer_lead_times",
        reference.retailer_lead_times.to_vec(),
    )?;
    dict.set_item(
        "initial_warehouse_inventory",
        reference.initial_warehouse_inventory,
    )?;
    dict.set_item(
        "initial_warehouse_pipeline",
        reference.initial_warehouse_pipeline.to_vec(),
    )?;
    dict.set_item(
        "initial_retailer_inventory",
        reference.initial_retailer_inventory.to_vec(),
    )?;
    dict.set_item(
        "initial_retailer_pipeline",
        reference
            .initial_retailer_pipeline
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("holding_cost_warehouse", reference.holding_cost_warehouse)?;
    dict.set_item(
        "holding_cost_retailers",
        reference.holding_cost_retailers.to_vec(),
    )?;
    dict.set_item(
        "penalty_costs_retailers",
        reference.penalty_costs_retailers.to_vec(),
    )?;
    dict.set_item(
        "emergency_shipment_probability",
        reference.emergency_shipment_probability,
    )?;
    dict.set_item(
        "optimal_allocation_policy",
        allocation_policy_to_str(reference.optimal_allocation_policy),
    )?;
    dict.set_item(
        "heuristic_warehouse_base_stock_level",
        reference.heuristic_warehouse_base_stock_level,
    )?;
    dict.set_item(
        "heuristic_retailer_base_stock_levels",
        reference.heuristic_retailer_base_stock_levels.to_vec(),
    )?;
    dict.set_item(
        "demand_supports",
        reference
            .demand_supports
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "demand_probabilities",
        reference
            .demand_probabilities
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("max_action_levels", reference.max_action_levels.to_vec())?;
    dict.set_item("notes", reference.notes)?;
    Ok(dict.into_any().unbind().into())
}

fn build_rollout_config(
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    demand_models: Vec<DemandModel>,
    allocation_policy: &str,
    retailer_target_inventory_positions: Option<Vec<usize>>,
    holding_cost_warehouse: f64,
    holding_cost_retailers: Vec<f64>,
    penalty_costs_retailers: Vec<f64>,
    customer_behavior: &str,
    periods: usize,
    emergency_shipment_probability: f64,
    discount_factor: f64,
    policy_action_mode: &str,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
    policy_state_mode: &str,
) -> PyResult<OneWarehouseMultiRetailerRolloutConfig> {
    Ok(OneWarehouseMultiRetailerRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_models,
        allocation_policy: parse_allocation_policy(allocation_policy)?,
        retailer_target_inventory_positions,
        holding_cost_warehouse,
        holding_cost_retailers,
        penalty_costs_retailers,
        customer_behavior: parse_customer_behavior_model(customer_behavior)?,
        emergency_shipment_probability,
        discount_factor,
        policy_action_mode: parse_policy_action_mode(policy_action_mode)?,
        policy_state_mode: parse_policy_state_mode(policy_state_mode)?,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    })
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    initial_warehouse_inventory,
    initial_warehouse_pipeline,
    initial_retailer_inventory,
    initial_retailer_pipeline,
    demand_kinds,
    demand_param1,
    demand_param2,
    holding_cost_warehouse,
    holding_cost_retailers,
    penalty_costs_retailers,
    customer_behavior,
    periods,
    seed=1234,
    emergency_shipment_probability=0.8,
    discount_factor=0.99,
    allocation_policy="proportional",
    policy_action_mode="direct_orders",
    retailer_target_inventory_positions=None,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None,
    policy_state_mode="normalized"
))]
fn one_warehouse_multi_retailer_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_warehouse_inventory: i32,
    initial_warehouse_pipeline: Vec<usize>,
    initial_retailer_inventory: Vec<i32>,
    initial_retailer_pipeline: Vec<Vec<usize>>,
    demand_kinds: Vec<String>,
    demand_param1: Vec<f64>,
    demand_param2: Vec<f64>,
    holding_cost_warehouse: f64,
    holding_cost_retailers: Vec<f64>,
    penalty_costs_retailers: Vec<f64>,
    customer_behavior: &str,
    periods: usize,
    seed: u64,
    emergency_shipment_probability: f64,
    discount_factor: f64,
    allocation_policy: &str,
    policy_action_mode: &str,
    retailer_target_inventory_positions: Option<Vec<usize>>,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
    policy_state_mode: &str,
) -> PyResult<f64> {
    let demand_models = build_demand_models(demand_kinds, demand_param1, demand_param2)?;
    let initial_state = build_initial_state(
        initial_warehouse_inventory,
        &initial_warehouse_pipeline,
        &initial_retailer_inventory,
        &initial_retailer_pipeline,
    )?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        demand_models,
        allocation_policy,
        retailer_target_inventory_positions,
        holding_cost_warehouse,
        holding_cost_retailers,
        penalty_costs_retailers,
        customer_behavior,
        periods,
        emergency_shipment_probability,
        discount_factor,
        policy_action_mode,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
        policy_state_mode,
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
    initial_warehouse_inventory,
    initial_warehouse_pipeline,
    initial_retailer_inventory,
    initial_retailer_pipeline,
    demand_kinds,
    demand_param1,
    demand_param2,
    holding_cost_warehouse,
    holding_cost_retailers,
    penalty_costs_retailers,
    customer_behavior,
    periods,
    seeds,
    emergency_shipment_probability=0.8,
    discount_factor=0.99,
    allocation_policy="proportional",
    policy_action_mode="direct_orders",
    retailer_target_inventory_positions=None,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None,
    policy_state_mode="normalized"
))]
fn one_warehouse_multi_retailer_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_warehouse_inventory: i32,
    initial_warehouse_pipeline: Vec<usize>,
    initial_retailer_inventory: Vec<i32>,
    initial_retailer_pipeline: Vec<Vec<usize>>,
    demand_kinds: Vec<String>,
    demand_param1: Vec<f64>,
    demand_param2: Vec<f64>,
    holding_cost_warehouse: f64,
    holding_cost_retailers: Vec<f64>,
    penalty_costs_retailers: Vec<f64>,
    customer_behavior: &str,
    periods: usize,
    seeds: Vec<u64>,
    emergency_shipment_probability: f64,
    discount_factor: f64,
    allocation_policy: &str,
    policy_action_mode: &str,
    retailer_target_inventory_positions: Option<Vec<usize>>,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
    policy_state_mode: &str,
) -> PyResult<Vec<f64>> {
    let demand_models = build_demand_models(demand_kinds, demand_param1, demand_param2)?;
    let initial_state = build_initial_state(
        initial_warehouse_inventory,
        &initial_warehouse_pipeline,
        &initial_retailer_inventory,
        &initial_retailer_pipeline,
    )?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        demand_models,
        allocation_policy,
        retailer_target_inventory_positions,
        holding_cost_warehouse,
        holding_cost_retailers,
        penalty_costs_retailers,
        customer_behavior,
        periods,
        emergency_shipment_probability,
        discount_factor,
        policy_action_mode,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
        policy_state_mode,
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
    initial_warehouse_inventory,
    initial_warehouse_pipeline,
    initial_retailer_inventory,
    initial_retailer_pipeline,
    demands,
    holding_cost_warehouse,
    holding_cost_retailers,
    penalty_costs_retailers,
    customer_behavior,
    seed=1234,
    emergency_shipment_probability=0.8,
    discount_factor=0.99,
    allocation_policy="proportional",
    policy_action_mode="direct_orders",
    retailer_target_inventory_positions=None,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None,
    policy_state_mode="normalized"
))]
fn one_warehouse_multi_retailer_soft_tree_rollout_from_paths(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_warehouse_inventory: i32,
    initial_warehouse_pipeline: Vec<usize>,
    initial_retailer_inventory: Vec<i32>,
    initial_retailer_pipeline: Vec<Vec<usize>>,
    demands: Vec<Vec<usize>>,
    holding_cost_warehouse: f64,
    holding_cost_retailers: Vec<f64>,
    penalty_costs_retailers: Vec<f64>,
    customer_behavior: &str,
    seed: u64,
    emergency_shipment_probability: f64,
    discount_factor: f64,
    allocation_policy: &str,
    policy_action_mode: &str,
    retailer_target_inventory_positions: Option<Vec<usize>>,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
    policy_state_mode: &str,
) -> PyResult<f64> {
    let demand_models = vec![
        DemandModel {
            kind: parse_demand_distribution_kind("deterministic")?,
            param1: 0.0,
            param2: 0.0,
        };
        initial_retailer_inventory.len()
    ];
    let initial_state = build_initial_state(
        initial_warehouse_inventory,
        &initial_warehouse_pipeline,
        &initial_retailer_inventory,
        &initial_retailer_pipeline,
    )?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        demand_models,
        allocation_policy,
        retailer_target_inventory_positions,
        holding_cost_warehouse,
        holding_cost_retailers,
        penalty_costs_retailers,
        customer_behavior,
        demands.len(),
        emergency_shipment_probability,
        discount_factor,
        policy_action_mode,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
        policy_state_mode,
    )?;
    rollout_from_paths(&flat_params, &config, &initial_state, &demands, seed)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    initial_warehouse_inventory,
    initial_warehouse_pipeline,
    initial_retailer_inventory,
    initial_retailer_pipeline,
    demands,
    holding_cost_warehouse,
    holding_cost_retailers,
    penalty_costs_retailers,
    customer_behavior,
    seed=1234,
    emergency_shipment_probability=0.8,
    discount_factor=0.99,
    allocation_policy="proportional"
))]
fn one_warehouse_multi_retailer_policy_rollout_from_paths(
    policy_name: &str,
    params: Vec<f64>,
    initial_warehouse_inventory: i32,
    initial_warehouse_pipeline: Vec<usize>,
    initial_retailer_inventory: Vec<i32>,
    initial_retailer_pipeline: Vec<Vec<usize>>,
    demands: Vec<Vec<usize>>,
    holding_cost_warehouse: f64,
    holding_cost_retailers: Vec<f64>,
    penalty_costs_retailers: Vec<f64>,
    customer_behavior: &str,
    seed: u64,
    emergency_shipment_probability: f64,
    discount_factor: f64,
    allocation_policy: &str,
) -> PyResult<f64> {
    let initial_state: OneWarehouseMultiRetailerState = build_initial_state(
        initial_warehouse_inventory,
        &initial_warehouse_pipeline,
        &initial_retailer_inventory,
        &initial_retailer_pipeline,
    )?;
    policy_rollout_from_paths(
        policy_name,
        &params,
        &initial_state,
        &demands,
        parse_allocation_policy(allocation_policy)?,
        holding_cost_warehouse,
        &holding_cost_retailers,
        &penalty_costs_retailers,
        parse_customer_behavior_model(customer_behavior)?,
        emergency_shipment_probability,
        discount_factor,
        seed,
    )
}

#[pyfunction]
fn one_warehouse_multi_retailer_echelon_base_stock_orders(
    warehouse_inventory: i32,
    warehouse_pipeline: Vec<usize>,
    retailer_inventory: Vec<i32>,
    retailer_pipeline: Vec<Vec<usize>>,
    warehouse_base_stock_level: usize,
    retailer_base_stock_levels: Vec<usize>,
) -> PyResult<Vec<usize>> {
    let state = build_initial_state(
        warehouse_inventory,
        &warehouse_pipeline,
        &retailer_inventory,
        &retailer_pipeline,
    )?;
    echelon_base_stock_orders(
        &state,
        warehouse_base_stock_level,
        &retailer_base_stock_levels,
    )
}

#[pyfunction]
fn one_warehouse_multi_retailer_benchmark_reference(py: Python<'_>) -> PyResult<PyObject> {
    benchmark_reference_to_py(py, &KAYNOV_2024_REFERENCE)
}

#[pyfunction]
fn one_warehouse_multi_retailer_list_reference_instances(
    py: Python<'_>,
) -> PyResult<Vec<PyObject>> {
    TABLE_A3_INSTANCES
        .iter()
        .map(|reference| reference_instance_to_py(py, reference))
        .collect()
}

#[pyfunction]
fn one_warehouse_multi_retailer_get_reference_instance(
    py: Python<'_>,
    name: &str,
) -> PyResult<PyObject> {
    let reference = TABLE_A3_INSTANCES
        .iter()
        .find(|reference| reference.name == name)
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "unknown reference instance '{name}'"
            ))
        })?;
    reference_instance_to_py(py, reference)
}

#[pyfunction]
fn one_warehouse_multi_retailer_primary_reference_instance(py: Python<'_>) -> PyResult<PyObject> {
    reference_instance_to_py(py, &PRIMARY_REFERENCE_INSTANCE)
}

#[pyfunction]
fn one_warehouse_multi_retailer_exact_verification_instance(py: Python<'_>) -> PyResult<PyObject> {
    exact_reference_to_py(py, &VERIFICATION_PROBLEM_INSTANCE)
}

#[pyfunction]
fn one_warehouse_multi_retailer_exact_dp_summary(py: Python<'_>) -> PyResult<PyObject> {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)?;
    let proportional = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "echelon_base_stock_proportional",
    )?;
    let min_shortage = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "echelon_base_stock_min_shortage",
    )?;
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "verification_reference",
        exact_reference_to_py(py, &VERIFICATION_PROBLEM_INSTANCE)?,
    )?;
    dict.set_item("optimal_discounted_cost", optimal.discounted_cost)?;
    dict.set_item("optimal_first_action", optimal.first_action)?;
    dict.set_item("proportional_discounted_cost", proportional.discounted_cost)?;
    dict.set_item("proportional_first_action", proportional.first_action)?;
    dict.set_item("min_shortage_discounted_cost", min_shortage.discounted_cost)?;
    dict.set_item("min_shortage_first_action", min_shortage.first_action)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
#[pyo3(signature = (
    warehouse_base_stock_level,
    retailer_base_stock_levels,
    allocation_policy="proportional"
))]
fn one_warehouse_multi_retailer_exact_evaluate_echelon_base_stock(
    py: Python<'_>,
    warehouse_base_stock_level: usize,
    retailer_base_stock_levels: Vec<usize>,
    allocation_policy: &str,
) -> PyResult<PyObject> {
    let evaluation = evaluate_echelon_base_stock_policy(
        &VERIFICATION_PROBLEM_INSTANCE,
        warehouse_base_stock_level,
        &retailer_base_stock_levels,
        parse_allocation_policy(allocation_policy)?,
    )?;
    let dict = PyDict::new_bound(py);
    dict.set_item("warehouse_base_stock_level", warehouse_base_stock_level)?;
    dict.set_item("retailer_base_stock_levels", retailer_base_stock_levels)?;
    dict.set_item("allocation_policy", allocation_policy)?;
    dict.set_item("discounted_cost", evaluation.discounted_cost)?;
    dict.set_item("first_action", evaluation.first_action)?;
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
    allocation_policy="proportional",
    policy_action_mode="direct_orders",
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None,
    policy_state_mode="normalized"
))]
fn one_warehouse_multi_retailer_exact_evaluate_soft_tree(
    py: Python<'_>,
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    allocation_policy: &str,
    policy_action_mode: &str,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
    policy_state_mode: &str,
) -> PyResult<PyObject> {
    let evaluation = evaluate_soft_tree_policy(
        &VERIFICATION_PROBLEM_INSTANCE,
        &ExactSoftTreeConfig {
            flat_params,
            input_dim,
            depth,
            action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
            allocation_policy: parse_allocation_policy(allocation_policy)?,
            policy_action_mode: parse_policy_action_mode(policy_action_mode)?,
            policy_state_mode: parse_policy_state_mode(policy_state_mode)?,
            temperature,
            split_type: parse_split_type(split_type)?,
            leaf_type: parse_leaf_type(leaf_type)?,
        },
    )?;
    let dict = PyDict::new_bound(py);
    dict.set_item("allocation_policy", allocation_policy)?;
    dict.set_item("policy_action_mode", policy_action_mode)?;
    dict.set_item("policy_state_mode", policy_state_mode)?;
    dict.set_item("discounted_cost", evaluation.discounted_cost)?;
    dict.set_item("first_action", evaluation.first_action)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    initial_warehouse_inventory,
    initial_warehouse_pipeline,
    initial_retailer_inventory,
    initial_retailer_pipeline,
    periods,
    replications,
    seed,
    demand_kinds,
    demand_param1,
    demand_param2,
    holding_cost_warehouse,
    holding_cost_retailers,
    penalty_costs_retailers,
    customer_behavior,
    emergency_shipment_probability=0.8,
    discount_factor=0.99,
    allocation_policy="proportional"
))]
fn one_warehouse_multi_retailer_simulate_policy(
    policy_name: &str,
    params: Vec<f64>,
    initial_warehouse_inventory: i32,
    initial_warehouse_pipeline: Vec<usize>,
    initial_retailer_inventory: Vec<i32>,
    initial_retailer_pipeline: Vec<Vec<usize>>,
    periods: usize,
    replications: usize,
    seed: u64,
    demand_kinds: Vec<String>,
    demand_param1: Vec<f64>,
    demand_param2: Vec<f64>,
    holding_cost_warehouse: f64,
    holding_cost_retailers: Vec<f64>,
    penalty_costs_retailers: Vec<f64>,
    customer_behavior: &str,
    emergency_shipment_probability: f64,
    discount_factor: f64,
    allocation_policy: &str,
) -> PyResult<(f64, f64)> {
    let initial_state = build_initial_state(
        initial_warehouse_inventory,
        &initial_warehouse_pipeline,
        &initial_retailer_inventory,
        &initial_retailer_pipeline,
    )?;
    let demand_models = build_demand_models(demand_kinds, demand_param1, demand_param2)?;
    let summary = simulate_policy(
        policy_name,
        &params,
        &initial_state,
        periods,
        replications,
        seed,
        &demand_models,
        parse_allocation_policy(allocation_policy)?,
        holding_cost_warehouse,
        &holding_cost_retailers,
        &penalty_costs_retailers,
        parse_customer_behavior_model(customer_behavior)?,
        emergency_shipment_probability,
        discount_factor,
    )?;
    Ok((summary.mean_cost, summary.cost_std))
}

#[pyfunction]
fn one_warehouse_multi_retailer_build_raw_state(
    warehouse_inventory: i32,
    warehouse_pipeline: Vec<usize>,
    retailer_inventory: Vec<i32>,
    retailer_pipeline: Vec<Vec<usize>>,
) -> PyResult<Vec<f32>> {
    let state = build_initial_state(
        warehouse_inventory,
        &warehouse_pipeline,
        &retailer_inventory,
        &retailer_pipeline,
    )?;
    build_raw_state(&state)
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_benchmark_reference,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_list_reference_instances,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_get_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_primary_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_exact_verification_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_exact_dp_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_exact_evaluate_echelon_base_stock,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_exact_evaluate_soft_tree,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_soft_tree_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_soft_tree_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_policy_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_echelon_base_stock_orders,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_simulate_policy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        one_warehouse_multi_retailer_build_raw_state,
        m
    )?)?;
    Ok(())
}
