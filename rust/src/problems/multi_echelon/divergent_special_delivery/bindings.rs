use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::multi_echelon::env::{
    build_raw_state, initialize_state, parse_allocation_mode, parse_inventory_dynamics_mode,
    parse_warehouse_base_stock_mode, AllocationMode,
};
use crate::problems::multi_echelon::exact_rollout::{
    population_rollout as exact_population_rollout, rollout as exact_rollout,
    MultiEchelonExactRolloutConfig,
};
use crate::problems::multi_echelon::finite_horizon_dp::{
    evaluate_soft_tree_policy as exact_evaluate_soft_tree_policy,
    search_best_stationary_policy as search_best_stationary_policy_exact, solve_optimal_policy,
    ExactHeuristicKind, ExactSoftTreeConfig,
};
use crate::problems::multi_echelon::heuristics::{
    parse_stationary_policy_kind, search_stationary_policy,
};
use crate::problems::multi_echelon::references::{
    ExactVerificationReference, MultiEchelonReferenceInstance, PublishedBenchmarkReference,
    GIJSBRECHTS_2022_REFERENCE, LITERATURE_REFERENCE_INSTANCES, PRIMARY_REFERENCE_INSTANCE,
    VAN_ROY_1997_CASE_STUDY, VERIFICATION_PROBLEM_INSTANCE, WORKED_TRANSITION_REFERENCE,
};
use crate::problems::multi_echelon::rollout::{
    build_policy_features_with_mode, parse_demand_distribution, parse_policy_action_mode,
    parse_policy_feature_mode, parse_rollout_objective, parse_state_normalizer,
    population_rollout as practical_population_rollout, rollout as practical_rollout,
    MultiEchelonRolloutConfig,
};
use crate::problems::multi_echelon::verification::{
    gijs_relative_verification_summary, van_roy_reproduction_summary, GijsRelativeVerificationRow,
    GijsRelativeVerificationSummary, VanRoyReproductionRow, VanRoyReproductionSummary,
    DEFAULT_GIJS_RELATIVE_VERIFICATION_REPLICATIONS, DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED,
    GIJS_RELATIVE_VERIFICATION_METRIC,
};

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

fn literature_reference_to_py(
    py: Python<'_>,
    reference: &MultiEchelonReferenceInstance,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    let has_gijs_relative_row = reference.published_a3c_savings_pct.is_some();
    let literature_metadata = PyDict::new_bound(py);
    literature_metadata.set_item("source", reference.source)?;
    literature_metadata.set_item("url", reference.url)?;
    literature_metadata.set_item(
        "literature_reference_present",
        has_gijs_relative_row || reference.published_constant_base_stock_mean_cost.is_some(),
    )?;
    literature_metadata.set_item(
        "implementation_literature_verified",
        reference.literature_verified,
    )?;
    literature_metadata.set_item(
        "repo_algorithm_literature_verified",
        reference.literature_verified,
    )?;
    literature_metadata.set_item(
        "literature_verification_metric",
        if has_gijs_relative_row {
            GIJS_RELATIVE_VERIFICATION_METRIC
        } else if reference.published_constant_base_stock_mean_cost.is_some() {
            "published_constant_base_stock_mean_cost"
        } else {
            "none"
        },
    )?;
    literature_metadata.set_item(
        "published_relative_policy",
        if has_gijs_relative_row { "a3c" } else { "none" },
    )?;
    literature_metadata.set_item("published_relative_baseline_policy", "constant_base_stock")?;
    literature_metadata.set_item(
        "published_relative_savings_pct",
        reference.published_a3c_savings_pct,
    )?;
    literature_metadata.set_item(
        "published_relative_confidence_half_width_pct",
        reference.published_a3c_confidence_half_width_pct,
    )?;
    literature_metadata.set_item(
        "published_relative_savings_source",
        if has_gijs_relative_row {
            "Gijsbrechts et al. (2022), Section 7.2"
        } else {
            "none"
        },
    )?;
    literature_metadata.set_item(
        "verification_scope",
        if has_gijs_relative_row {
            "benchmark_instance_and_published_comparison_row"
        } else {
            "benchmark_instance_and_published_constant_base_stock_row"
        },
    )?;
    literature_metadata.set_item(
        "repo_policy_reproduction_note",
        "The published A3C row is carried as a literature target; the repo does not currently reproduce that A3C policy.",
    )?;
    dict.set_item("name", reference.name)?;
    dict.set_item("source", reference.source)?;
    dict.set_item("url", reference.url)?;
    dict.set_item("literature_verified", reference.literature_verified)?;
    dict.set_item("warehouse_lead_time", reference.warehouse_lead_time)?;
    dict.set_item("retailer_lead_time", reference.retailer_lead_time)?;
    dict.set_item("num_retailers", reference.num_retailers)?;
    dict.set_item("warehouse_holding_cost", reference.warehouse_holding_cost)?;
    dict.set_item("retailer_holding_cost", reference.retailer_holding_cost)?;
    dict.set_item(
        "warehouse_expedited_cost",
        reference.warehouse_expedited_cost,
    )?;
    dict.set_item(
        "warehouse_lost_sale_cost",
        reference.warehouse_lost_sale_cost,
    )?;
    dict.set_item("expedited_service_prob", reference.expedited_service_prob)?;
    dict.set_item("warehouse_capacity", reference.warehouse_capacity)?;
    dict.set_item("warehouse_inventory_cap", reference.warehouse_inventory_cap)?;
    dict.set_item("retailer_inventory_cap", reference.retailer_inventory_cap)?;
    dict.set_item("inventory_dynamics_mode", reference.inventory_dynamics_mode)?;
    dict.set_item("demand_distribution", reference.demand_distribution)?;
    dict.set_item("demand_mean", reference.demand_mean)?;
    dict.set_item("demand_std", reference.demand_std)?;
    dict.set_item(
        "benchmark_search_horizon",
        reference.benchmark_search_horizon,
    )?;
    dict.set_item("benchmark_periods", reference.benchmark_periods)?;
    dict.set_item("benchmark_replications", reference.benchmark_replications)?;
    dict.set_item("warm_up_periods_ratio", reference.warm_up_periods_ratio)?;
    dict.set_item("rollout_objective", reference.rollout_objective)?;
    dict.set_item(
        "warehouse_base_stock_mode",
        reference.warehouse_base_stock_mode,
    )?;
    dict.set_item("policy_allocation_mode", reference.policy_allocation_mode)?;
    dict.set_item(
        "benchmark_warehouse_levels",
        reference.benchmark_warehouse_levels.to_vec(),
    )?;
    dict.set_item(
        "benchmark_retailer_levels",
        reference.benchmark_retailer_levels.to_vec(),
    )?;
    dict.set_item(
        "published_constant_base_stock_mean_cost",
        reference.published_constant_base_stock_mean_cost,
    )?;
    dict.set_item(
        "published_constant_base_stock_levels",
        reference.published_constant_base_stock_levels.to_vec(),
    )?;
    dict.set_item(
        "published_van_roy_best_ndp_mean_cost",
        reference.published_van_roy_best_ndp_mean_cost,
    )?;
    dict.set_item(
        "published_a3c_savings_pct",
        reference.published_a3c_savings_pct,
    )?;
    dict.set_item(
        "published_a3c_confidence_half_width_pct",
        reference.published_a3c_confidence_half_width_pct,
    )?;
    dict.set_item(
        "published_van_roy_savings_pct_approx",
        reference.published_van_roy_savings_pct_approx,
    )?;
    dict.set_item("tuned_learning_rate", reference.tuned_learning_rate)?;
    dict.set_item(
        "tuned_entropy_regularization",
        reference.tuned_entropy_regularization,
    )?;
    dict.set_item("tuned_buffer_length", reference.tuned_buffer_length)?;
    dict.set_item("literature_metadata", literature_metadata)?;
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
    dict.set_item("periods", reference.periods)?;
    dict.set_item("warehouse_lead_time", reference.warehouse_lead_time)?;
    dict.set_item("retailer_lead_time", reference.retailer_lead_time)?;
    dict.set_item("num_retailers", reference.num_retailers)?;
    dict.set_item("warehouse_holding_cost", reference.warehouse_holding_cost)?;
    dict.set_item("retailer_holding_cost", reference.retailer_holding_cost)?;
    dict.set_item(
        "warehouse_expedited_cost",
        reference.warehouse_expedited_cost,
    )?;
    dict.set_item(
        "warehouse_lost_sale_cost",
        reference.warehouse_lost_sale_cost,
    )?;
    dict.set_item("expedited_service_prob", reference.expedited_service_prob)?;
    dict.set_item("warehouse_capacity", reference.warehouse_capacity)?;
    dict.set_item("warehouse_inventory_cap", reference.warehouse_inventory_cap)?;
    dict.set_item("retailer_inventory_cap", reference.retailer_inventory_cap)?;
    dict.set_item("inventory_dynamics_mode", reference.inventory_dynamics_mode)?;
    dict.set_item("discount_factor", reference.discount_factor)?;
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
    dict.set_item("demand_support", reference.demand_support.to_vec())?;
    dict.set_item(
        "demand_probabilities",
        reference.demand_probabilities.to_vec(),
    )?;
    dict.set_item(
        "action_warehouse_levels",
        reference.action_warehouse_levels.to_vec(),
    )?;
    dict.set_item(
        "action_retailer_levels",
        reference.action_retailer_levels.to_vec(),
    )?;
    dict.set_item(
        "warehouse_base_stock_mode",
        reference.warehouse_base_stock_mode,
    )?;
    dict.set_item("allocation_mode", reference.allocation_mode)?;
    dict.set_item("notes", reference.notes)?;
    Ok(dict.into_any().unbind().into())
}

fn gijs_relative_verification_row_to_py(
    py: Python<'_>,
    row: &GijsRelativeVerificationRow,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("instance_name", row.instance_name)?;
    dict.set_item(
        "published_constant_base_stock_levels",
        row.published_constant_base_stock_levels.clone(),
    )?;
    dict.set_item(
        "published_constant_base_stock_mean_cost",
        row.published_constant_base_stock_mean_cost,
    )?;
    dict.set_item("published_a3c_savings_pct", row.published_a3c_savings_pct)?;
    dict.set_item(
        "published_a3c_confidence_half_width_pct",
        row.published_a3c_confidence_half_width_pct,
    )?;
    dict.set_item(
        "published_a3c_implied_mean_cost",
        row.published_a3c_implied_mean_cost,
    )?;
    dict.set_item(
        "published_van_roy_savings_pct_approx",
        row.published_van_roy_savings_pct_approx,
    )?;
    dict.set_item(
        "published_van_roy_implied_mean_cost",
        row.published_van_roy_implied_mean_cost,
    )?;
    dict.set_item(
        "repo_published_constant_base_stock_mean_cost",
        row.repo_published_constant_base_stock_mean_cost,
    )?;
    dict.set_item(
        "repo_published_constant_base_stock_cost_std",
        row.repo_published_constant_base_stock_cost_std,
    )?;
    dict.set_item(
        "repo_gap_vs_published_constant_cost",
        row.repo_gap_vs_published_constant_cost,
    )?;
    dict.set_item(
        "repo_gap_vs_published_constant_cost_pct",
        row.repo_gap_vs_published_constant_cost_pct,
    )?;
    dict.set_item(
        "published_constant_base_stock_reproduced_within_tolerance",
        row.published_constant_base_stock_reproduced_within_tolerance,
    )?;
    Ok(dict.into_any().unbind().into())
}

fn gijs_relative_verification_summary_to_py(
    py: Python<'_>,
    summary: &GijsRelativeVerificationSummary,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("source", summary.source)?;
    dict.set_item("url", summary.url)?;
    dict.set_item("repo_audit_replications", summary.repo_audit_replications)?;
    dict.set_item("seed", summary.seed)?;
    dict.set_item(
        "mean_published_a3c_savings_pct",
        summary.mean_published_a3c_savings_pct,
    )?;
    dict.set_item(
        "mean_repo_gap_vs_published_constant_cost",
        summary.mean_repo_gap_vs_published_constant_cost,
    )?;
    dict.set_item(
        "literature_reference_present",
        summary.literature_reference_present,
    )?;
    dict.set_item(
        "implementation_literature_verified",
        summary.implementation_literature_verified,
    )?;
    dict.set_item(
        "literature_verification_metric",
        summary.literature_verification_metric,
    )?;
    dict.set_item(
        "literature_verification_target_count",
        summary.literature_verification_target_count,
    )?;
    dict.set_item(
        "all_published_constant_base_stock_rows_reproduced_within_tolerance",
        summary.all_published_constant_base_stock_rows_reproduced_within_tolerance,
    )?;
    dict.set_item(
        "repo_generates_published_relative_rows",
        summary.repo_generates_published_relative_rows,
    )?;
    dict.set_item(
        "can_mark_literature_verified",
        summary.can_mark_literature_verified,
    )?;
    dict.set_item("verification_note", summary.verification_note)?;
    let rows = summary
        .rows
        .iter()
        .map(|row| gijs_relative_verification_row_to_py(py, row))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("rows", rows)?;
    Ok(dict.into_any().unbind().into())
}

fn van_roy_reproduction_row_to_py(
    py: Python<'_>,
    row: &VanRoyReproductionRow,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("instance_name", row.instance_name)?;
    dict.set_item("source", row.source)?;
    dict.set_item("url", row.url)?;
    dict.set_item(
        "published_constant_base_stock_levels",
        row.published_constant_base_stock_levels.clone(),
    )?;
    dict.set_item(
        "published_constant_base_stock_mean_cost",
        row.published_constant_base_stock_mean_cost,
    )?;
    dict.set_item(
        "repo_published_constant_base_stock_mean_cost",
        row.repo_published_constant_base_stock_mean_cost,
    )?;
    dict.set_item(
        "repo_published_constant_base_stock_cost_std",
        row.repo_published_constant_base_stock_cost_std,
    )?;
    dict.set_item(
        "repo_gap_vs_published_constant_cost",
        row.repo_gap_vs_published_constant_cost,
    )?;
    dict.set_item(
        "repo_gap_vs_published_constant_cost_pct",
        row.repo_gap_vs_published_constant_cost_pct,
    )?;
    dict.set_item(
        "reproduced_within_tolerance",
        row.reproduced_within_tolerance,
    )?;
    Ok(dict.into_any().unbind().into())
}

fn van_roy_reproduction_summary_to_py(
    py: Python<'_>,
    summary: &VanRoyReproductionSummary,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("source", summary.source)?;
    dict.set_item("url", summary.url)?;
    dict.set_item("repo_audit_replications", summary.repo_audit_replications)?;
    dict.set_item("seed", summary.seed)?;
    dict.set_item("tolerance_pct", summary.tolerance_pct)?;
    dict.set_item(
        "literature_reference_present",
        summary.literature_reference_present,
    )?;
    dict.set_item(
        "implementation_literature_verified",
        summary.implementation_literature_verified,
    )?;
    dict.set_item(
        "literature_verification_metric",
        summary.literature_verification_metric,
    )?;
    dict.set_item(
        "literature_verification_target_count",
        summary.literature_verification_target_count,
    )?;
    dict.set_item(
        "all_published_constant_base_stock_rows_reproduced_within_tolerance",
        summary.all_published_constant_base_stock_rows_reproduced_within_tolerance,
    )?;
    dict.set_item("verification_note", summary.verification_note)?;
    let rows = summary
        .rows
        .iter()
        .map(|row| van_roy_reproduction_row_to_py(py, row))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("rows", rows)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn multi_echelon_benchmark_reference(py: Python<'_>) -> PyResult<PyObject> {
    benchmark_reference_to_py(py, &GIJSBRECHTS_2022_REFERENCE)
}

#[pyfunction]
fn multi_echelon_list_reference_instances(py: Python<'_>) -> PyResult<Vec<PyObject>> {
    LITERATURE_REFERENCE_INSTANCES
        .iter()
        .map(|reference| literature_reference_to_py(py, reference))
        .collect()
}

#[pyfunction]
fn multi_echelon_get_reference_instance(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    let reference = LITERATURE_REFERENCE_INSTANCES
        .iter()
        .find(|reference| reference.name == name)
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "unknown reference instance '{name}'"
            ))
        })?;
    literature_reference_to_py(py, reference)
}

#[pyfunction]
fn multi_echelon_primary_reference_instance(py: Python<'_>) -> PyResult<PyObject> {
    literature_reference_to_py(py, PRIMARY_REFERENCE_INSTANCE)
}

#[pyfunction]
fn multi_echelon_van_roy_case_study(py: Python<'_>) -> PyResult<PyObject> {
    literature_reference_to_py(py, &VAN_ROY_1997_CASE_STUDY)
}

#[pyfunction]
fn multi_echelon_exact_verification_instance(py: Python<'_>) -> PyResult<PyObject> {
    exact_reference_to_py(py, &VERIFICATION_PROBLEM_INSTANCE)
}

#[pyfunction]
fn multi_echelon_exact_dp_summary(py: Python<'_>) -> PyResult<PyObject> {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)?;
    let best_sequential = search_best_stationary_policy_exact(
        &VERIFICATION_PROBLEM_INSTANCE,
        ExactHeuristicKind::RegularBaseStock,
        AllocationMode::SequentialIndex,
    )?;
    let best_proportional = search_best_stationary_policy_exact(
        &VERIFICATION_PROBLEM_INSTANCE,
        ExactHeuristicKind::RegularBaseStock,
        AllocationMode::Proportional,
    )?;
    let best_min_shortage = search_best_stationary_policy_exact(
        &VERIFICATION_PROBLEM_INSTANCE,
        ExactHeuristicKind::RegularBaseStock,
        AllocationMode::MinShortage,
    )?;

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "verification_reference",
        exact_reference_to_py(py, &VERIFICATION_PROBLEM_INSTANCE)?,
    )?;
    dict.set_item("optimal_discounted_cost", optimal.discounted_cost)?;
    dict.set_item("optimal_first_action", optimal.first_action)?;
    dict.set_item(
        "sequential_discounted_cost",
        best_sequential.2.discounted_cost,
    )?;
    dict.set_item(
        "sequential_first_action",
        best_sequential.2.first_action.clone(),
    )?;
    dict.set_item(
        "sequential_levels",
        vec![best_sequential.0, best_sequential.1],
    )?;
    dict.set_item(
        "proportional_discounted_cost",
        best_proportional.2.discounted_cost,
    )?;
    dict.set_item(
        "proportional_first_action",
        best_proportional.2.first_action.clone(),
    )?;
    dict.set_item(
        "proportional_levels",
        vec![best_proportional.0, best_proportional.1],
    )?;
    dict.set_item(
        "min_shortage_discounted_cost",
        best_min_shortage.2.discounted_cost,
    )?;
    dict.set_item(
        "min_shortage_first_action",
        best_min_shortage.2.first_action.clone(),
    )?;
    dict.set_item(
        "min_shortage_levels",
        vec![best_min_shortage.0, best_min_shortage.1],
    )?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
#[pyo3(signature = (
    repo_audit_replications=DEFAULT_GIJS_RELATIVE_VERIFICATION_REPLICATIONS,
    seed=DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED
))]
fn multi_echelon_gijs_relative_verification_summary(
    py: Python<'_>,
    repo_audit_replications: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let summary = gijs_relative_verification_summary(repo_audit_replications, seed)?;
    gijs_relative_verification_summary_to_py(py, &summary)
}

#[pyfunction]
#[pyo3(signature = (
    repo_audit_replications=DEFAULT_GIJS_RELATIVE_VERIFICATION_REPLICATIONS,
    seed=DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED
))]
fn multi_echelon_van_roy_reproduction_summary(
    py: Python<'_>,
    repo_audit_replications: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let summary = van_roy_reproduction_summary(repo_audit_replications, seed)?;
    van_roy_reproduction_summary_to_py(py, &summary)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    include_period_feature=true,
    warehouse_base_stock_mode="regular",
    allocation_mode="min_shortage",
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None
))]
fn multi_echelon_exact_evaluate_soft_tree(
    py: Python<'_>,
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    include_period_feature: bool,
    warehouse_base_stock_mode: &str,
    allocation_mode: &str,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<PyObject> {
    let evaluation = exact_evaluate_soft_tree_policy(
        &VERIFICATION_PROBLEM_INSTANCE,
        &ExactSoftTreeConfig {
            flat_params,
            input_dim,
            depth,
            action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
            include_period_feature,
            warehouse_base_stock_mode: parse_warehouse_base_stock_mode(warehouse_base_stock_mode)?,
            allocation_mode: parse_allocation_mode(allocation_mode)?,
            temperature,
            split_type: parse_split_type(split_type)?,
            leaf_type: parse_leaf_type(leaf_type)?,
        },
    )?;
    let dict = PyDict::new_bound(py);
    dict.set_item("warehouse_base_stock_mode", warehouse_base_stock_mode)?;
    dict.set_item("allocation_mode", allocation_mode)?;
    dict.set_item("discounted_cost", evaluation.discounted_cost)?;
    dict.set_item("first_action", evaluation.first_action)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
#[pyo3(signature = (
    policy_kind,
    allocation_mode,
    warehouse_levels,
    retailer_levels,
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
    inventory_dynamics_mode,
    demand_distribution,
    demand_mean,
    demand_std,
    horizon,
    replications,
    seed,
    warm_up_periods_ratio=0.0,
    discount_factor=1.0,
    objective="cumulative_cost",
    top_k=10
))]
fn multi_echelon_search_stationary_policy(
    policy_kind: &str,
    allocation_mode: &str,
    warehouse_levels: Vec<usize>,
    retailer_levels: Vec<usize>,
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
    inventory_dynamics_mode: &str,
    demand_distribution: &str,
    demand_mean: f64,
    demand_std: f64,
    horizon: usize,
    replications: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    discount_factor: f64,
    objective: &str,
    top_k: usize,
) -> PyResult<PyObject> {
    let kind = parse_stationary_policy_kind(policy_kind)?;
    let allocation_mode = parse_allocation_mode(allocation_mode)?;
    let (best, top_results) = search_stationary_policy(
        &warehouse_levels,
        &retailer_levels,
        kind,
        allocation_mode,
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
        parse_inventory_dynamics_mode(inventory_dynamics_mode)?,
        demand_distribution,
        demand_mean,
        demand_std,
        horizon,
        warm_up_periods_ratio,
        discount_factor,
        objective,
        replications,
        seed,
        top_k,
    )?;
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        let best_dict = PyDict::new_bound(py);
        best_dict.set_item("warehouse_level", best.0)?;
        best_dict.set_item("retailer_level", best.1)?;
        best_dict.set_item("mean_cost", best.2)?;
        best_dict.set_item("cost_std", best.3)?;
        dict.set_item("best_result", best_dict)?;
        let top_py = top_results
            .iter()
            .map(|row| {
                let row_dict = PyDict::new_bound(py);
                row_dict.set_item("warehouse_level", row.0)?;
                row_dict.set_item("retailer_level", row.1)?;
                row_dict.set_item("mean_cost", row.2)?;
                row_dict.set_item("cost_std", row.3)?;
                Ok(row_dict.into_any().unbind().into())
            })
            .collect::<PyResult<Vec<PyObject>>>()?;
        dict.set_item("top_results", top_py)?;
        Ok(dict.into_any().unbind().into())
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
    inventory_dynamics_mode,
    demand_distribution,
    demand_mean,
    demand_std,
    horizon,
    seed=1234,
    warm_up_periods_ratio=0.0,
    discount_factor=1.0,
    objective="cumulative_cost",
    include_period_feature=false,
    warehouse_base_stock_mode="regular",
    policy_feature_mode="full_decision_state",
    policy_action_mode="direct_base_stock",
    warehouse_anchor_level=0,
    retailer_anchor_level=0,
    reference_warehouse_levels=None,
    reference_retailer_levels=None,
    allocation_mode="min_shortage",
    state_normalizer="identity",
    state_scale=None,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
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
    inventory_dynamics_mode: &str,
    demand_distribution: &str,
    demand_mean: f64,
    demand_std: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    discount_factor: f64,
    objective: &str,
    include_period_feature: bool,
    warehouse_base_stock_mode: &str,
    policy_feature_mode: &str,
    policy_action_mode: &str,
    warehouse_anchor_level: usize,
    retailer_anchor_level: usize,
    reference_warehouse_levels: Option<Vec<usize>>,
    reference_retailer_levels: Option<Vec<usize>>,
    allocation_mode: &str,
    state_normalizer: &str,
    state_scale: Option<f64>,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    // Initialization levels only seed the zero-state shape (their numeric values are unused
    // by initialize_random_state), so they need not come from a discrete grid. For grid
    // policies default to the grid; for direct policies (vector_quantity / scalar)
    // allowed_values is None, so fall back to the reference levels or a placeholder.
    // build_action_spec validates that allowed_values is present iff the mode is discrete_grid.
    let initialization_warehouse_levels = reference_warehouse_levels
        .clone()
        .or_else(|| allowed_values.as_ref().and_then(|grid| grid.get(0).cloned()))
        .filter(|levels| !levels.is_empty())
        .unwrap_or_else(|| vec![0]);
    let initialization_retailer_levels = reference_retailer_levels
        .clone()
        .or_else(|| allowed_values.as_ref().and_then(|grid| grid.get(1).cloned()))
        .filter(|levels| !levels.is_empty())
        .unwrap_or_else(|| vec![0]);
    let config = MultiEchelonRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        policy_feature_mode: parse_policy_feature_mode(policy_feature_mode)?,
        policy_action_mode: parse_policy_action_mode(policy_action_mode)?,
        warehouse_anchor_level,
        retailer_anchor_level,
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
        inventory_dynamics_mode: parse_inventory_dynamics_mode(inventory_dynamics_mode)?,
        demand_distribution: parse_demand_distribution(demand_distribution)?,
        demand_mean,
        demand_std,
        horizon,
        warm_up_periods_ratio,
        discount_factor,
        objective: parse_rollout_objective(objective)?,
        include_period_feature,
        warehouse_base_stock_mode: parse_warehouse_base_stock_mode(warehouse_base_stock_mode)?,
        allocation_mode: parse_allocation_mode(allocation_mode)?,
        state_normalizer: parse_state_normalizer(state_normalizer)?,
        state_scale,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    practical_rollout(
        &flat_params,
        &config,
        seed,
        &initialization_warehouse_levels,
        &initialization_retailer_levels,
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
    inventory_dynamics_mode,
    demand_distribution,
    demand_mean,
    demand_std,
    seeds,
    horizon,
    warm_up_periods_ratio=0.0,
    discount_factor=1.0,
    objective="cumulative_cost",
    include_period_feature=false,
    warehouse_base_stock_mode="regular",
    policy_feature_mode="full_decision_state",
    policy_action_mode="direct_base_stock",
    warehouse_anchor_level=0,
    retailer_anchor_level=0,
    reference_warehouse_levels=None,
    reference_retailer_levels=None,
    allocation_mode="min_shortage",
    state_normalizer="identity",
    state_scale=None,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
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
    inventory_dynamics_mode: &str,
    demand_distribution: &str,
    demand_mean: f64,
    demand_std: f64,
    seeds: Vec<u64>,
    horizon: usize,
    warm_up_periods_ratio: f64,
    discount_factor: f64,
    objective: &str,
    include_period_feature: bool,
    warehouse_base_stock_mode: &str,
    policy_feature_mode: &str,
    policy_action_mode: &str,
    warehouse_anchor_level: usize,
    retailer_anchor_level: usize,
    reference_warehouse_levels: Option<Vec<usize>>,
    reference_retailer_levels: Option<Vec<usize>>,
    allocation_mode: &str,
    state_normalizer: &str,
    state_scale: Option<f64>,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    // Initialization levels only seed the zero-state shape (their numeric values are unused
    // by initialize_random_state), so they need not come from a discrete grid. For grid
    // policies default to the grid; for direct policies (vector_quantity / scalar)
    // allowed_values is None, so fall back to the reference levels or a placeholder.
    // build_action_spec validates that allowed_values is present iff the mode is discrete_grid.
    let initialization_warehouse_levels = reference_warehouse_levels
        .clone()
        .or_else(|| allowed_values.as_ref().and_then(|grid| grid.get(0).cloned()))
        .filter(|levels| !levels.is_empty())
        .unwrap_or_else(|| vec![0]);
    let initialization_retailer_levels = reference_retailer_levels
        .clone()
        .or_else(|| allowed_values.as_ref().and_then(|grid| grid.get(1).cloned()))
        .filter(|levels| !levels.is_empty())
        .unwrap_or_else(|| vec![0]);
    let config = MultiEchelonRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        policy_feature_mode: parse_policy_feature_mode(policy_feature_mode)?,
        policy_action_mode: parse_policy_action_mode(policy_action_mode)?,
        warehouse_anchor_level,
        retailer_anchor_level,
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
        inventory_dynamics_mode: parse_inventory_dynamics_mode(inventory_dynamics_mode)?,
        demand_distribution: parse_demand_distribution(demand_distribution)?,
        demand_mean,
        demand_std,
        horizon,
        warm_up_periods_ratio,
        discount_factor,
        objective: parse_rollout_objective(objective)?,
        include_period_feature,
        warehouse_base_stock_mode: parse_warehouse_base_stock_mode(warehouse_base_stock_mode)?,
        allocation_mode: parse_allocation_mode(allocation_mode)?,
        state_normalizer: parse_state_normalizer(state_normalizer)?,
        state_scale,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    practical_population_rollout(
        &params_batch,
        &config,
        &seeds,
        &initialization_warehouse_levels,
        &initialization_retailer_levels,
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
    initial_warehouse_inventory,
    initial_warehouse_pipeline,
    initial_retailer_inventory,
    initial_retailer_pipeline,
    demand_support,
    demand_probabilities,
    periods,
    discount_factor,
    warehouse_capacity,
    warehouse_inventory_cap,
    retailer_inventory_cap,
    warehouse_holding_cost,
    retailer_holding_cost,
    warehouse_expedited_cost,
    warehouse_lost_sale_cost,
    expedited_service_prob,
    seed=1234,
    include_period_feature=true,
    warehouse_base_stock_mode="regular",
    allocation_mode="min_shortage",
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None
))]
fn multi_echelon_exact_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_warehouse_inventory: i32,
    initial_warehouse_pipeline: Vec<u32>,
    initial_retailer_inventory: Vec<i32>,
    initial_retailer_pipeline: Vec<Vec<u32>>,
    demand_support: Vec<u32>,
    demand_probabilities: Vec<f64>,
    periods: usize,
    discount_factor: f64,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    expedited_service_prob: f64,
    seed: u64,
    include_period_feature: bool,
    warehouse_base_stock_mode: &str,
    allocation_mode: &str,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let config = MultiEchelonExactRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        discount_factor,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        warehouse_holding_cost,
        retailer_holding_cost,
        warehouse_expedited_cost,
        warehouse_lost_sale_cost,
        expedited_service_prob,
        demand_support,
        demand_probabilities,
        initial_warehouse_inventory,
        initial_warehouse_pipeline,
        initial_retailer_inventory,
        initial_retailer_pipeline,
        include_period_feature,
        warehouse_base_stock_mode: parse_warehouse_base_stock_mode(warehouse_base_stock_mode)?,
        allocation_mode: parse_allocation_mode(allocation_mode)?,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    exact_rollout(&flat_params, &config, seed)
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
    demand_support,
    demand_probabilities,
    periods,
    discount_factor,
    warehouse_capacity,
    warehouse_inventory_cap,
    retailer_inventory_cap,
    warehouse_holding_cost,
    retailer_holding_cost,
    warehouse_expedited_cost,
    warehouse_lost_sale_cost,
    expedited_service_prob,
    seeds,
    include_period_feature=true,
    warehouse_base_stock_mode="regular",
    allocation_mode="min_shortage",
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None
))]
fn multi_echelon_exact_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    initial_warehouse_inventory: i32,
    initial_warehouse_pipeline: Vec<u32>,
    initial_retailer_inventory: Vec<i32>,
    initial_retailer_pipeline: Vec<Vec<u32>>,
    demand_support: Vec<u32>,
    demand_probabilities: Vec<f64>,
    periods: usize,
    discount_factor: f64,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    expedited_service_prob: f64,
    seeds: Vec<u64>,
    include_period_feature: bool,
    warehouse_base_stock_mode: &str,
    allocation_mode: &str,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let config = MultiEchelonExactRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        discount_factor,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        warehouse_holding_cost,
        retailer_holding_cost,
        warehouse_expedited_cost,
        warehouse_lost_sale_cost,
        expedited_service_prob,
        demand_support,
        demand_probabilities,
        initial_warehouse_inventory,
        initial_warehouse_pipeline,
        initial_retailer_inventory,
        initial_retailer_pipeline,
        include_period_feature,
        warehouse_base_stock_mode: parse_warehouse_base_stock_mode(warehouse_base_stock_mode)?,
        allocation_mode: parse_allocation_mode(allocation_mode)?,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    exact_population_rollout(&params_batch, &config, &seeds)
}

#[pyfunction]
fn multi_echelon_worked_transition_reference(py: Python<'_>) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("source", WORKED_TRANSITION_REFERENCE.source)?;
    dict.set_item("url", WORKED_TRANSITION_REFERENCE.url)?;
    dict.set_item(
        "initial_warehouse_inventory",
        WORKED_TRANSITION_REFERENCE.initial_warehouse_inventory,
    )?;
    dict.set_item(
        "initial_warehouse_pipeline",
        WORKED_TRANSITION_REFERENCE
            .initial_warehouse_pipeline
            .to_vec(),
    )?;
    dict.set_item(
        "initial_retailer_inventory",
        WORKED_TRANSITION_REFERENCE
            .initial_retailer_inventory
            .to_vec(),
    )?;
    dict.set_item(
        "initial_retailer_pipeline",
        WORKED_TRANSITION_REFERENCE
            .initial_retailer_pipeline
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "warehouse_target",
        WORKED_TRANSITION_REFERENCE.warehouse_target,
    )?;
    dict.set_item(
        "retailer_target",
        WORKED_TRANSITION_REFERENCE.retailer_target,
    )?;
    dict.set_item(
        "realized_demands",
        WORKED_TRANSITION_REFERENCE.realized_demands.to_vec(),
    )?;
    dict.set_item(
        "accepted_emergency_shipments",
        WORKED_TRANSITION_REFERENCE.accepted_emergency_shipments,
    )?;
    dict.set_item(
        "warehouse_base_stock_mode",
        WORKED_TRANSITION_REFERENCE.warehouse_base_stock_mode,
    )?;
    dict.set_item(
        "allocation_mode",
        WORKED_TRANSITION_REFERENCE.allocation_mode,
    )?;
    dict.set_item(
        "expected_warehouse_order",
        WORKED_TRANSITION_REFERENCE.expected_warehouse_order,
    )?;
    dict.set_item(
        "expected_shipped_retail_orders",
        WORKED_TRANSITION_REFERENCE
            .expected_shipped_retail_orders
            .to_vec(),
    )?;
    dict.set_item(
        "expected_next_warehouse_inventory",
        WORKED_TRANSITION_REFERENCE.expected_next_warehouse_inventory,
    )?;
    dict.set_item(
        "expected_next_warehouse_pipeline",
        WORKED_TRANSITION_REFERENCE
            .expected_next_warehouse_pipeline
            .to_vec(),
    )?;
    dict.set_item(
        "expected_next_retailer_inventory",
        WORKED_TRANSITION_REFERENCE
            .expected_next_retailer_inventory
            .to_vec(),
    )?;
    dict.set_item(
        "expected_next_retailer_pipeline",
        WORKED_TRANSITION_REFERENCE
            .expected_next_retailer_pipeline
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "expected_period_cost",
        WORKED_TRANSITION_REFERENCE.expected_period_cost,
    )?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn multi_echelon_build_raw_state(
    warehouse_inventory: i32,
    warehouse_pipeline: Vec<u32>,
    retailer_inventory: Vec<i32>,
    retailer_pipeline: Vec<Vec<u32>>,
) -> PyResult<Vec<f32>> {
    let state = initialize_state(
        warehouse_inventory,
        &warehouse_pipeline,
        &retailer_inventory,
        &retailer_pipeline,
    )?;
    build_raw_state(&state)
}

/// Report the learned-policy input dimension for this problem, computed by the exact
/// feature builder the rollout uses. This makes the problem the single source of truth for
/// its own policy I/O contract, so the Python policy builder no longer has to re-derive the
/// dimension with a formula that can drift from the Rust decision-state layout.
#[pyfunction]
#[pyo3(signature = (
    num_retailers,
    warehouse_lead_time,
    retailer_lead_time,
    inventory_dynamics_mode,
    policy_feature_mode="full_decision_state",
    include_period_feature=false
))]
fn multi_echelon_policy_feature_dim(
    num_retailers: usize,
    warehouse_lead_time: usize,
    retailer_lead_time: usize,
    inventory_dynamics_mode: &str,
    policy_feature_mode: &str,
    include_period_feature: bool,
) -> PyResult<usize> {
    // A zero state of the correct shape; only the feature-vector LENGTH is needed (the
    // inventory caps scale the values, not the length), so unit caps are fine.
    let state = initialize_state(
        0,
        &vec![0u32; warehouse_lead_time],
        &vec![0i32; num_retailers.max(1)],
        &vec![vec![0u32; retailer_lead_time]; num_retailers.max(1)],
    )?;
    let features = build_policy_features_with_mode(
        &state,
        1,
        1,
        include_period_feature,
        1,
        parse_policy_feature_mode(policy_feature_mode)?,
        parse_inventory_dynamics_mode(inventory_dynamics_mode)?,
    )?;
    Ok(features.len())
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(multi_echelon_policy_feature_dim, m)?)?;
    m.add_function(wrap_pyfunction!(multi_echelon_benchmark_reference, m)?)?;
    m.add_function(wrap_pyfunction!(multi_echelon_list_reference_instances, m)?)?;
    m.add_function(wrap_pyfunction!(multi_echelon_get_reference_instance, m)?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_primary_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(multi_echelon_van_roy_case_study, m)?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_exact_verification_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(multi_echelon_exact_dp_summary, m)?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_gijs_relative_verification_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_van_roy_reproduction_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(multi_echelon_exact_evaluate_soft_tree, m)?)?;
    m.add_function(wrap_pyfunction!(multi_echelon_search_stationary_policy, m)?)?;
    m.add_function(wrap_pyfunction!(multi_echelon_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(multi_echelon_exact_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_exact_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_worked_transition_reference,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(multi_echelon_build_raw_state, m)?)?;
    Ok(())
}
