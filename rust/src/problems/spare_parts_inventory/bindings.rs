use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::spare_parts_inventory::env::{build_raw_state, initialize_state};
use crate::problems::spare_parts_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::spare_parts_inventory::heuristics::{
    base_stock_order_quantity, lead_time_mean_cover_order_quantity, lead_time_mean_cover_target,
    policy_rollout_from_paths, simulate_policy, PolicySimulationSummary,
};
use crate::problems::spare_parts_inventory::literature::kranenburg_lateral_transshipment::{
    compare_to_published_table, evaluate_reference_instance, KranenburgBenchmarkEvaluation,
    KranenburgPublishedComparison, KranenburgSituationSummary, KRANENBURG_TABLE_ROUNDING_TOLERANCE,
};
use crate::problems::spare_parts_inventory::references::{
    ExactVerificationReference, KranenburgLateralTransshipmentReferenceInstance,
    LiteratureBenchmarkPolicyResult, LiteratureBenchmarkScenario, SparePartsReferenceInstance,
    KRANENBURG_2006_TABLE_5_2_BASE_CASE, KRANENBURG_2006_TABLE_5_2_ROWS,
    PRIMARY_REFERENCE_INSTANCE, VAN_OERS_2024_TABLE_1_SCENARIOS, VERIFICATION_PROBLEM_INSTANCE,
};
use crate::problems::spare_parts_inventory::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_paths,
    SparePartsInventoryRolloutConfig,
};

fn primary_reference_to_py(
    py: Python<'_>,
    reference: &SparePartsReferenceInstance,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", reference.name)?;
    dict.set_item("source", reference.source)?;
    dict.set_item("url", reference.url)?;
    dict.set_item("periods", reference.periods)?;
    dict.set_item("installed_base", reference.installed_base)?;
    dict.set_item("procurement_lead_time", reference.procurement_lead_time)?;
    dict.set_item("repair_lead_time", reference.repair_lead_time)?;
    dict.set_item(
        "initial_on_hand_inventory",
        reference.initial_on_hand_inventory,
    )?;
    dict.set_item("initial_backlog", reference.initial_backlog)?;
    dict.set_item(
        "initial_procurement_pipeline",
        reference.initial_procurement_pipeline.to_vec(),
    )?;
    dict.set_item(
        "initial_repair_pipeline",
        reference.initial_repair_pipeline.to_vec(),
    )?;
    dict.set_item("failure_probability", reference.failure_probability)?;
    dict.set_item("holding_cost", reference.holding_cost)?;
    dict.set_item("downtime_cost", reference.downtime_cost)?;
    dict.set_item("procurement_cost", reference.procurement_cost)?;
    dict.set_item(
        "benchmark_base_stock_level",
        reference.benchmark_base_stock_level,
    )?;
    dict.set_item(
        "benchmark_lead_time_mean_cover_safety_buffer",
        reference.benchmark_lead_time_mean_cover_safety_buffer,
    )?;
    dict.set_item("literature_verified", reference.literature_verified)?;
    dict.set_item("verification_source", reference.verification_source)?;
    dict.set_item("notes", reference.notes)?;
    Ok(dict.into_any().unbind().into())
}

fn literature_policy_result_to_py(
    py: Python<'_>,
    row: &LiteratureBenchmarkPolicyResult,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("policy_name", row.policy_name)?;
    dict.set_item("base_stock_levels", row.base_stock_levels.to_vec())?;
    dict.set_item("reported_cost_value", row.reported_cost_value)?;
    dict.set_item("reported_cost_half_width", row.reported_cost_half_width)?;
    dict.set_item("reported_readiness_percent", row.reported_readiness_percent)?;
    dict.set_item(
        "reported_readiness_half_width",
        row.reported_readiness_half_width,
    )?;
    Ok(dict.into_any().unbind().into())
}

fn literature_benchmark_scenario_to_py(
    py: Python<'_>,
    scenario: &LiteratureBenchmarkScenario,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", scenario.name)?;
    dict.set_item("source", scenario.source)?;
    dict.set_item("url", scenario.url)?;
    dict.set_item("literature_verified", scenario.literature_verified)?;
    dict.set_item("verification_source", scenario.verification_source)?;
    dict.set_item("model_family", scenario.model_family)?;
    dict.set_item("am_location", scenario.am_location)?;
    dict.set_item("echelons", scenario.echelons)?;
    dict.set_item("simulation_horizon_days", scenario.simulation_horizon_days)?;
    dict.set_item("table_replications", scenario.table_replications)?;
    dict.set_item("demand_rate_per_hour", scenario.demand_rate_per_hour)?;
    dict.set_item(
        "review_intervals_hours",
        scenario.review_intervals_hours.to_vec(),
    )?;
    dict.set_item(
        "transport_lead_times_hours",
        scenario.transport_lead_times_hours.to_vec(),
    )?;
    dict.set_item("am_lead_time_hours", scenario.am_lead_time_hours)?;
    dict.set_item("regular_sourcing_cost", scenario.regular_sourcing_cost)?;
    dict.set_item("am_sourcing_cost", scenario.am_sourcing_cost)?;
    dict.set_item(
        "holding_costs_as_reported",
        scenario.holding_costs_as_reported.to_vec(),
    )?;
    dict.set_item(
        "downtime_cost_as_reported",
        scenario.downtime_cost_as_reported,
    )?;
    let policy_rows: PyResult<Vec<PyObject>> = scenario
        .published_policy_results
        .iter()
        .map(|row| literature_policy_result_to_py(py, row))
        .collect();
    dict.set_item("published_policy_results", policy_rows?)?;
    dict.set_item("notes", scenario.notes)?;
    Ok(dict.into_any().unbind().into())
}

fn kranenburg_reference_to_py(
    py: Python<'_>,
    reference: &KranenburgLateralTransshipmentReferenceInstance,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", reference.name)?;
    dict.set_item("source", reference.source)?;
    dict.set_item("url", reference.url)?;
    dict.set_item("literature_verified", reference.literature_verified)?;
    dict.set_item("verification_source", reference.verification_source)?;
    dict.set_item("table", reference.table)?;
    dict.set_item("varied_parameter", reference.varied_parameter)?;
    dict.set_item("varied_value_label", reference.varied_value_label)?;
    dict.set_item(
        "demand_rate_per_local_warehouse",
        reference.demand_rate_per_local_warehouse,
    )?;
    dict.set_item("num_local_warehouses", reference.num_local_warehouses)?;
    dict.set_item("holding_cost", reference.holding_cost)?;
    dict.set_item("emergency_cost", reference.emergency_cost)?;
    dict.set_item(
        "lateral_transshipment_cost",
        reference.lateral_transshipment_cost,
    )?;
    dict.set_item("joint_warehouse_cost", reference.joint_warehouse_cost)?;
    dict.set_item("waiting_time_target", reference.waiting_time_target)?;
    dict.set_item("emergency_time", reference.emergency_time)?;
    dict.set_item(
        "lateral_transshipment_time",
        reference.lateral_transshipment_time,
    )?;
    dict.set_item("joint_warehouse_time", reference.joint_warehouse_time)?;
    dict.set_item(
        "regular_replenishment_time",
        reference.regular_replenishment_time,
    )?;
    dict.set_item(
        "published_situation1_optimal_r",
        reference.published_situation1_optimal_r,
    )?;
    dict.set_item(
        "published_situation1_cost",
        reference.published_situation1_cost,
    )?;
    dict.set_item(
        "published_situation3_optimal_r",
        reference.published_situation3_optimal_r,
    )?;
    dict.set_item(
        "published_situation3_cost",
        reference.published_situation3_cost,
    )?;
    dict.set_item(
        "published_cost_ratio_situation1_over_situation3",
        reference.published_cost_ratio_situation1_over_situation3,
    )?;
    dict.set_item("notes", reference.notes)?;
    Ok(dict.into_any().unbind().into())
}

fn kranenburg_situation_summary_to_py(
    py: Python<'_>,
    summary: &KranenburgSituationSummary,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("optimal_r", summary.optimal_r)?;
    dict.set_item("emergency_probability", summary.emergency_probability)?;
    dict.set_item("mean_waiting_time", summary.mean_waiting_time)?;
    dict.set_item(
        "transport_cost_per_request",
        summary.transport_cost_per_request,
    )?;
    dict.set_item("total_cost", summary.total_cost)?;
    dict.set_item(
        "waiting_constraint_binding",
        summary.waiting_constraint_binding,
    )?;
    Ok(dict.into_any().unbind().into())
}

fn kranenburg_comparison_to_py(
    py: Python<'_>,
    comparison: &KranenburgPublishedComparison,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("tolerance", comparison.tolerance)?;
    dict.set_item(
        "situation1_optimal_r_abs_diff",
        comparison.situation1_optimal_r_abs_diff,
    )?;
    dict.set_item(
        "situation1_cost_abs_diff",
        comparison.situation1_cost_abs_diff,
    )?;
    dict.set_item(
        "situation3_optimal_r_abs_diff",
        comparison.situation3_optimal_r_abs_diff,
    )?;
    dict.set_item(
        "situation3_cost_abs_diff",
        comparison.situation3_cost_abs_diff,
    )?;
    dict.set_item("cost_ratio_abs_diff", comparison.cost_ratio_abs_diff)?;
    dict.set_item(
        "matches_situation1_optimal_r",
        comparison.matches_situation1_optimal_r,
    )?;
    dict.set_item(
        "matches_situation1_cost",
        comparison.matches_situation1_cost,
    )?;
    dict.set_item(
        "matches_situation3_optimal_r",
        comparison.matches_situation3_optimal_r,
    )?;
    dict.set_item(
        "matches_situation3_cost",
        comparison.matches_situation3_cost,
    )?;
    dict.set_item("matches_cost_ratio", comparison.matches_cost_ratio)?;
    dict.set_item("all_within_tolerance", comparison.all_within_tolerance)?;
    Ok(dict.into_any().unbind().into())
}

fn kranenburg_summary_to_py(
    py: Python<'_>,
    evaluation: &KranenburgBenchmarkEvaluation,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "situation1",
        kranenburg_situation_summary_to_py(py, &evaluation.situation1)?,
    )?;
    dict.set_item(
        "situation2",
        evaluation
            .situation2
            .as_ref()
            .map(|summary| kranenburg_situation_summary_to_py(py, summary))
            .transpose()?,
    )?;
    dict.set_item(
        "situation3",
        kranenburg_situation_summary_to_py(py, &evaluation.situation3)?,
    )?;
    dict.set_item(
        "cost_ratio_situation1_over_situation3",
        evaluation.cost_ratio_situation1_over_situation3,
    )?;
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
    dict.set_item("installed_base", reference.installed_base)?;
    dict.set_item("procurement_lead_time", reference.procurement_lead_time)?;
    dict.set_item("repair_lead_time", reference.repair_lead_time)?;
    dict.set_item(
        "initial_on_hand_inventory",
        reference.initial_on_hand_inventory,
    )?;
    dict.set_item("initial_backlog", reference.initial_backlog)?;
    dict.set_item(
        "initial_procurement_pipeline",
        reference.initial_procurement_pipeline.to_vec(),
    )?;
    dict.set_item(
        "initial_repair_pipeline",
        reference.initial_repair_pipeline.to_vec(),
    )?;
    dict.set_item("failure_probability", reference.failure_probability)?;
    dict.set_item("holding_cost", reference.holding_cost)?;
    dict.set_item("downtime_cost", reference.downtime_cost)?;
    dict.set_item("procurement_cost", reference.procurement_cost)?;
    dict.set_item("max_order_quantity", reference.max_order_quantity)?;
    dict.set_item("base_stock_level", reference.base_stock_level)?;
    dict.set_item(
        "lead_time_mean_cover_safety_buffer",
        reference.lead_time_mean_cover_safety_buffer,
    )?;
    dict.set_item("notes", reference.notes)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn spare_parts_inventory_literature_benchmark_catalog(py: Python<'_>) -> PyResult<Vec<PyObject>> {
    VAN_OERS_2024_TABLE_1_SCENARIOS
        .iter()
        .map(|scenario| literature_benchmark_scenario_to_py(py, scenario))
        .collect()
}

#[pyfunction]
fn spare_parts_inventory_kranenburg_reference_instances(py: Python<'_>) -> PyResult<Vec<PyObject>> {
    KRANENBURG_2006_TABLE_5_2_ROWS
        .iter()
        .map(|reference| kranenburg_reference_to_py(py, reference))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (instance_name = None))]
fn spare_parts_inventory_kranenburg_exact_summary(
    py: Python<'_>,
    instance_name: Option<&str>,
) -> PyResult<PyObject> {
    let reference = match instance_name {
        Some(name) => KRANENBURG_2006_TABLE_5_2_ROWS
            .iter()
            .find(|reference| reference.name == name)
            .copied()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown Kranenburg spare-parts reference instance '{name}'"
                ))
            })?,
        None => KRANENBURG_2006_TABLE_5_2_BASE_CASE,
    };
    let evaluation = evaluate_reference_instance(&reference).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "failed to evaluate Kranenburg instance '{}': {err}",
            reference.name
        ))
    })?;
    let comparison =
        compare_to_published_table(&reference, &evaluation, KRANENBURG_TABLE_ROUNDING_TOLERANCE);

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "reference_instance",
        kranenburg_reference_to_py(py, &reference)?,
    )?;
    dict.set_item("evaluation", kranenburg_summary_to_py(py, &evaluation)?)?;
    dict.set_item(
        "published_table_comparison",
        kranenburg_comparison_to_py(py, &comparison)?,
    )?;
    Ok(dict.into_any().unbind().into())
}

fn simulation_summary_to_py(
    py: Python<'_>,
    summary: &PolicySimulationSummary,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("mean_discounted_cost", summary.mean_cost)?;
    dict.set_item("std_discounted_cost", summary.cost_std)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn spare_parts_inventory_primary_reference_instance(py: Python<'_>) -> PyResult<PyObject> {
    primary_reference_to_py(py, &PRIMARY_REFERENCE_INSTANCE)
}

#[pyfunction]
fn spare_parts_inventory_exact_verification_instance(py: Python<'_>) -> PyResult<PyObject> {
    verification_reference_to_py(py, &VERIFICATION_PROBLEM_INSTANCE)
}

#[pyfunction]
fn spare_parts_inventory_exact_dp_summary(py: Python<'_>) -> PyResult<PyObject> {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)?;
    let base_stock = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "base_stock")?;
    let mean_cover =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "lead_time_mean_cover")?;

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "verification_reference",
        verification_reference_to_py(py, &VERIFICATION_PROBLEM_INSTANCE)?,
    )?;
    dict.set_item("optimal_discounted_cost", optimal.discounted_cost)?;
    dict.set_item("optimal_first_action", optimal.first_action)?;
    dict.set_item("base_stock_discounted_cost", base_stock.discounted_cost)?;
    dict.set_item("base_stock_first_action", base_stock.first_action)?;
    dict.set_item(
        "lead_time_mean_cover_discounted_cost",
        mean_cover.discounted_cost,
    )?;
    dict.set_item("lead_time_mean_cover_first_action", mean_cover.first_action)?;
    dict.set_item(
        "base_stock_gap_to_optimal",
        base_stock.discounted_cost - optimal.discounted_cost,
    )?;
    dict.set_item(
        "lead_time_mean_cover_gap_to_optimal",
        mean_cover.discounted_cost - optimal.discounted_cost,
    )?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
#[pyo3(signature = (
    on_hand_inventory,
    backlog,
    procurement_pipeline,
    repair_pipeline,
    installed_base
))]
fn spare_parts_inventory_build_raw_state(
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
) -> PyResult<Vec<f32>> {
    let state = initialize_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    build_raw_state(&state)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    on_hand_inventory,
    backlog,
    procurement_pipeline,
    repair_pipeline,
    installed_base,
    periods,
    failure_probability,
    holding_cost,
    downtime_cost,
    procurement_cost,
    seed=1234,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None
))]
fn spare_parts_inventory_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    periods: usize,
    failure_probability: f64,
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
    seed: u64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    let config = SparePartsInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        installed_base,
        failure_probability,
        holding_cost,
        downtime_cost,
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
    on_hand_inventory,
    backlog,
    procurement_pipeline,
    repair_pipeline,
    installed_base,
    periods,
    failure_probability,
    holding_cost,
    downtime_cost,
    procurement_cost,
    seeds,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None
))]
fn spare_parts_inventory_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    periods: usize,
    failure_probability: f64,
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
    seeds: Vec<u64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let initial_state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    let config = SparePartsInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        installed_base,
        failure_probability,
        holding_cost,
        downtime_cost,
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
    on_hand_inventory,
    backlog,
    procurement_pipeline,
    repair_pipeline,
    installed_base,
    realized_failures,
    failure_probability,
    holding_cost,
    downtime_cost,
    procurement_cost,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None
))]
fn spare_parts_inventory_soft_tree_rollout_from_paths(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    realized_failures: Vec<usize>,
    failure_probability: f64,
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    let config = SparePartsInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods: realized_failures.len(),
        installed_base,
        failure_probability,
        holding_cost,
        downtime_cost,
        procurement_cost,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    rollout_from_paths(&flat_params, &config, &initial_state, &realized_failures)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    on_hand_inventory,
    backlog,
    procurement_pipeline,
    repair_pipeline,
    installed_base,
    realized_failures,
    failure_probability,
    holding_cost,
    downtime_cost,
    procurement_cost,
    discount_factor=0.99
))]
fn spare_parts_inventory_policy_rollout_from_paths(
    policy_name: &str,
    params: Vec<f64>,
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    realized_failures: Vec<usize>,
    failure_probability: f64,
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
    discount_factor: f64,
) -> PyResult<f64> {
    let initial_state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    policy_rollout_from_paths(
        policy_name,
        &params,
        &initial_state,
        installed_base,
        &realized_failures,
        holding_cost,
        downtime_cost,
        procurement_cost,
        failure_probability,
        discount_factor,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    on_hand_inventory,
    backlog,
    procurement_pipeline,
    repair_pipeline,
    installed_base,
    periods,
    failure_probability,
    holding_cost,
    downtime_cost,
    procurement_cost,
    replications=1000,
    seed=1234,
    discount_factor=0.99
))]
fn spare_parts_inventory_simulate_policy(
    py: Python<'_>,
    policy_name: &str,
    params: Vec<f64>,
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    periods: usize,
    failure_probability: f64,
    holding_cost: f64,
    downtime_cost: f64,
    procurement_cost: f64,
    replications: usize,
    seed: u64,
    discount_factor: f64,
) -> PyResult<PyObject> {
    let initial_state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    let summary = simulate_policy(
        policy_name,
        &params,
        &initial_state,
        periods,
        replications,
        seed,
        installed_base,
        failure_probability,
        holding_cost,
        downtime_cost,
        procurement_cost,
        discount_factor,
    )?;
    simulation_summary_to_py(py, &summary)
}

#[pyfunction]
#[pyo3(signature = (on_hand_inventory, backlog, procurement_pipeline, repair_pipeline, installed_base, base_stock_level))]
fn spare_parts_inventory_base_stock_order(
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    base_stock_level: usize,
) -> PyResult<usize> {
    let state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    base_stock_order_quantity(&state, base_stock_level)
}

#[pyfunction]
#[pyo3(signature = (
    on_hand_inventory,
    backlog,
    procurement_pipeline,
    repair_pipeline,
    installed_base,
    failure_probability,
    safety_buffer
))]
fn spare_parts_inventory_lead_time_mean_cover_order(
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
    installed_base: usize,
    failure_probability: f64,
    safety_buffer: f64,
) -> PyResult<usize> {
    let state = build_initial_state(
        on_hand_inventory,
        backlog,
        &procurement_pipeline,
        &repair_pipeline,
        installed_base,
    )?;
    lead_time_mean_cover_order_quantity(&state, installed_base, failure_probability, safety_buffer)
}

#[pyfunction]
#[pyo3(signature = (installed_base, failure_probability, procurement_lead_time, safety_buffer))]
fn spare_parts_inventory_lead_time_mean_cover_target(
    installed_base: usize,
    failure_probability: f64,
    procurement_lead_time: usize,
    safety_buffer: f64,
) -> PyResult<usize> {
    lead_time_mean_cover_target(
        installed_base,
        failure_probability,
        procurement_lead_time,
        safety_buffer,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_primary_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_literature_benchmark_catalog,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_kranenburg_reference_instances,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_kranenburg_exact_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_exact_verification_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(spare_parts_inventory_exact_dp_summary, m)?)?;
    m.add_function(wrap_pyfunction!(spare_parts_inventory_build_raw_state, m)?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_soft_tree_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_soft_tree_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_policy_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(spare_parts_inventory_simulate_policy, m)?)?;
    m.add_function(wrap_pyfunction!(spare_parts_inventory_base_stock_order, m)?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_lead_time_mean_cover_order,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spare_parts_inventory_lead_time_mean_cover_target,
        m
    )?)?;
    Ok(())
}
