use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::perishable_inventory::env::{parse_issuing_policy, PerishableState};
use crate::problems::perishable_inventory::heuristics::{
    policy_discounted_return, policy_discounted_return_summary, policy_rollout,
    policy_rollout_from_demands, policy_trace_summary_from_demands, search_base_stock,
    search_base_stock_discounted_return, search_base_stock_discounted_return_summary,
    search_base_stock_from_demands, search_bsp_low_ew, search_bsp_low_ew_discounted_return,
    search_bsp_low_ew_discounted_return_summary, search_bsp_low_ew_from_demands,
    DiscountedReturnSummary, PolicyTraceSummary,
};
use crate::problems::perishable_inventory::references::{
    get_primary_reference_instance, get_reference_instance, list_reference_instances,
    PerishableReferenceInstance,
};
use crate::problems::perishable_inventory::rollout::{
    population_rollout as perishable_population_rollout,
    population_rollout_discounted_return as perishable_population_rollout_discounted_return,
    rollout as perishable_rollout,
    rollout_discounted_return as perishable_rollout_discounted_return,
    rollout_from_demands as perishable_rollout_from_demands,
    rollout_trace_summary_from_demands as perishable_rollout_trace_summary_from_demands,
    PerishableInventoryRolloutConfig,
};
use crate::problems::perishable_inventory::value_iteration_mdp::{
    best_base_stock_level_by_expected_return, build_exact_mdp, build_policy_table_9x9,
    expected_discounted_return_from_zero_state, value_iteration_best_action_values,
};

fn empirical_mean_demand(demands: &[usize]) -> f64 {
    if demands.is_empty() {
        0.0
    } else {
        demands.iter().copied().sum::<usize>() as f64 / demands.len() as f64
    }
}

fn policy_table_to_vec(table: &[[usize; 9]; 9]) -> Vec<Vec<usize>> {
    table.iter().map(|row| row.to_vec()).collect()
}

fn reference_instance_state_count(instance: &PerishableReferenceInstance) -> usize {
    let components = instance.shelf_life + instance.lead_time - 1;
    (instance.max_order_size + 1).pow(components as u32)
}

fn reference_instance_to_py(
    py: Python<'_>,
    instance: PerishableReferenceInstance,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", instance.name)?;
    dict.set_item("demand_mean", instance.demand_mean)?;
    dict.set_item("demand_cov", instance.demand_cov)?;
    dict.set_item("shelf_life", instance.shelf_life)?;
    dict.set_item("lead_time", instance.lead_time)?;
    dict.set_item("shortage_cost", instance.shortage_cost)?;
    dict.set_item("holding_cost", instance.holding_cost)?;
    dict.set_item("waste_cost", instance.waste_cost)?;
    dict.set_item("procurement_cost", instance.procurement_cost)?;
    dict.set_item("max_order_size", instance.max_order_size)?;
    dict.set_item(
        "issuing_policy",
        match instance.issuing_policy {
            crate::problems::perishable_inventory::env::IssuingPolicy::Fifo => "fifo",
            crate::problems::perishable_inventory::env::IssuingPolicy::Lifo => "lifo",
        },
    )?;
    dict.set_item("horizon", instance.horizon)?;
    dict.set_item("eval_horizon", instance.eval_horizon)?;
    dict.set_item("warm_up_periods_ratio", instance.warm_up_periods_ratio)?;
    dict.set_item("state_count", reference_instance_state_count(&instance))?;

    let published = PyDict::new_bound(py);
    if let Some(published_returns) = instance.published_scenario_a_returns {
        published.set_item("source", published_returns.source)?;
        published.set_item("url", published_returns.url)?;
        published.set_item(
            "value_iteration_mean_return",
            published_returns.value_iteration_mean_return,
        )?;
        published.set_item(
            "value_iteration_return_std",
            published_returns.value_iteration_return_std,
        )?;
        published.set_item(
            "best_base_stock_mean_return",
            published_returns.best_base_stock_mean_return,
        )?;
        published.set_item(
            "best_base_stock_return_std",
            published_returns.best_base_stock_return_std,
        )?;
        published.set_item("optimality_gap_pct", published_returns.optimality_gap_pct)?;
        dict.set_item("published_scenario_a_returns", published)?;
    } else {
        dict.set_item("published_scenario_a_returns", py.None())?;
    }

    if let Some(figure) = instance.published_figure3_verification {
        let figure_dict = PyDict::new_bound(py);
        figure_dict.set_item("source", figure.source)?;
        figure_dict.set_item("url", figure.url)?;
        figure_dict.set_item(
            "published_base_stock_level",
            figure.published_base_stock_level,
        )?;
        figure_dict.set_item(
            "published_optimal_policy",
            policy_table_to_vec(figure.published_optimal_policy),
        )?;
        dict.set_item("published_figure3_verification", figure_dict)?;
    } else {
        dict.set_item("published_figure3_verification", py.None())?;
    }

    Ok(dict.into_any().unbind().into())
}

fn heuristic_candidate_to_py(
    py: Python<'_>,
    params: &[usize],
    summary: &DiscountedReturnSummary,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("params", params.to_vec())?;
    dict.set_item("mean_return", summary.mean_return)?;
    dict.set_item("std_return", summary.std_return)?;
    dict.set_item("min_return", summary.min_return)?;
    dict.set_item("max_return", summary.max_return)?;
    dict.set_item("num_seeds", summary.num_seeds)?;
    Ok(dict.into_any().unbind().into())
}

fn heuristic_search_to_py(
    py: Python<'_>,
    best: PyObject,
    top: Vec<PyObject>,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("best", best)?;
    dict.set_item("top", top)?;
    Ok(dict.into_any().unbind().into())
}

fn trace_summary_to_py(py: Python<'_>, summary: &PolicyTraceSummary) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("periods", summary.periods)?;
    dict.set_item("total_cost", summary.total_cost)?;
    dict.set_item("mean_period_cost", summary.mean_period_cost)?;
    dict.set_item("total_demand", summary.total_demand)?;
    dict.set_item("total_shortage", summary.total_shortage)?;
    dict.set_item("fill_rate", summary.fill_rate)?;
    dict.set_item("cycle_service_level", summary.cycle_service_level)?;
    dict.set_item("total_waste", summary.total_waste)?;
    dict.set_item("waste_rate", summary.waste_rate)?;
    dict.set_item("mean_holding_inventory", summary.mean_holding_inventory)?;
    dict.set_item("mean_order_quantity", summary.mean_order_quantity)?;
    dict.set_item("positive_order_frequency", summary.positive_order_frequency)?;
    dict.set_item("ending_inventory", summary.ending_inventory)?;
    dict.set_item("ending_pipeline", summary.ending_pipeline)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn perishable_inventory_primary_reference_instance_name() -> &'static str {
    get_primary_reference_instance().name
}

#[pyfunction]
fn perishable_inventory_list_reference_instances(py: Python<'_>) -> PyResult<Vec<PyObject>> {
    list_reference_instances()
        .iter()
        .copied()
        .map(|instance| reference_instance_to_py(py, instance))
        .collect()
}

#[pyfunction]
fn perishable_inventory_get_reference_instance(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    let instance = get_reference_instance(name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown perishable-inventory reference instance '{name}'"
        ))
    })?;
    reference_instance_to_py(py, instance)
}

#[pyfunction]
fn perishable_inventory_exact_mdp_summary(
    py: Python<'_>,
    reference_instance_name: &str,
) -> PyResult<PyObject> {
    let instance = get_reference_instance(reference_instance_name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown perishable-inventory reference instance '{reference_instance_name}'"
        ))
    })?;
    let state_count = reference_instance_state_count(&instance);
    if state_count > 2_000 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "exact MDP summary is only enabled for small instances; '{reference_instance_name}' has {state_count} states"
        )));
    }

    let mdp = build_exact_mdp(reference_instance_name);
    let (policy, _) = value_iteration_best_action_values(&mdp, 0.99);
    let expected_return =
        expected_discounted_return_from_zero_state(reference_instance_name, &mdp, &policy);
    let best_base_stock_level =
        best_base_stock_level_by_expected_return(reference_instance_name, &mdp);

    let dict = PyDict::new_bound(py);
    dict.set_item("reference_instance_name", reference_instance_name)?;
    dict.set_item("state_count", state_count)?;
    dict.set_item("best_base_stock_level", best_base_stock_level)?;
    dict.set_item("value_iteration_mean_return", expected_return)?;
    dict.set_item(
        "value_iteration_mean_return_rounded",
        expected_return.round() as i32,
    )?;

    if instance.published_figure3_verification.is_some() {
        let policy_table = build_policy_table_9x9(&policy, &mdp);
        dict.set_item("policy_table_9x9", policy_table_to_vec(&policy_table))?;
        if let Some(figure) = instance.published_figure3_verification {
            dict.set_item(
                "matches_published_policy_table",
                policy_table == *figure.published_optimal_policy,
            )?;
            dict.set_item(
                "matches_published_base_stock_level",
                best_base_stock_level == figure.published_base_stock_level,
            )?;
        }
    }
    if let Some(published) = instance.published_scenario_a_returns {
        dict.set_item(
            "matches_published_value_iteration_mean_return",
            expected_return.round() as i32 == published.value_iteration_mean_return,
        )?;
        dict.set_item(
            "published_value_iteration_mean_return",
            published.value_iteration_mean_return,
        )?;
    }

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
    gamma=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    issuing_policy="fifo",
    allowed_values=None
))]
fn perishable_inventory_soft_tree_discounted_return(
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
    gamma: f64,
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
    perishable_rollout_discounted_return(&flat_params, &config, seed, gamma)
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
    gamma=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    issuing_policy="fifo",
    allowed_values=None
))]
fn perishable_inventory_soft_tree_population_discounted_return(
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
    gamma: f64,
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
    perishable_population_rollout_discounted_return(&params_batch, &config, &seeds, gamma)
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
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    issuing_policy="fifo",
    allowed_values=None,
    demand_mean=None
))]
fn perishable_inventory_soft_tree_trace_summary_from_demands(
    py: Python<'_>,
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
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    issuing_policy: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
    demand_mean: Option<f64>,
) -> PyResult<PyObject> {
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
        warm_up_periods_ratio: 0.0,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        issuing_policy: parse_issuing_policy(issuing_policy)?,
    };
    let state = PerishableState {
        on_hand,
        pipeline_orders,
    };
    let summary = perishable_rollout_trace_summary_from_demands(&flat_params, &config, state, &demands)?;
    trace_summary_to_py(py, &summary)
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
    issuing_policy="fifo"
))]
fn perishable_inventory_policy_trace_summary_from_demands(
    py: Python<'_>,
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
    issuing_policy: &str,
) -> PyResult<PyObject> {
    let state = PerishableState {
        on_hand,
        pipeline_orders,
    };
    let summary = policy_trace_summary_from_demands(
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
        parse_issuing_policy(issuing_policy)?,
    )?;
    trace_summary_to_py(py, &summary)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    on_hand,
    pipeline_orders,
    horizon,
    seed,
    max_order_size,
    demand_mean,
    demand_cov,
    holding_cost,
    shortage_cost,
    waste_cost,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    issuing_policy="fifo"
))]
fn perishable_inventory_policy_rollout(
    policy_name: &str,
    params: Vec<usize>,
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    horizon: usize,
    seed: u64,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: &str,
) -> PyResult<f64> {
    let state = PerishableState {
        on_hand,
        pipeline_orders: pipeline_orders.clone(),
    };
    policy_rollout(
        policy_name,
        &params,
        &state,
        horizon,
        seed,
        pipeline_orders.len() + 1,
        max_order_size,
        demand_mean,
        demand_cov,
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
    policy_name,
    params,
    on_hand,
    pipeline_orders,
    horizon,
    seed,
    max_order_size,
    demand_mean,
    demand_cov,
    holding_cost,
    shortage_cost,
    waste_cost,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    gamma=0.99,
    issuing_policy="fifo"
))]
fn perishable_inventory_policy_discounted_return(
    policy_name: &str,
    params: Vec<usize>,
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    horizon: usize,
    seed: u64,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: &str,
) -> PyResult<f64> {
    let state = PerishableState {
        on_hand,
        pipeline_orders: pipeline_orders.clone(),
    };
    policy_discounted_return(
        policy_name,
        &params,
        &state,
        horizon,
        seed,
        pipeline_orders.len() + 1,
        max_order_size,
        demand_mean,
        demand_cov,
        holding_cost,
        shortage_cost,
        waste_cost,
        procurement_cost,
        warm_up_periods_ratio,
        gamma,
        parse_issuing_policy(issuing_policy)?,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    on_hand,
    pipeline_orders,
    horizon,
    seeds,
    max_order_size,
    demand_mean,
    demand_cov,
    holding_cost,
    shortage_cost,
    waste_cost,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    gamma=0.99,
    issuing_policy="fifo"
))]
fn perishable_inventory_policy_discounted_return_summary(
    py: Python<'_>,
    policy_name: &str,
    params: Vec<usize>,
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    horizon: usize,
    seeds: Vec<u64>,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: &str,
) -> PyResult<PyObject> {
    let state = PerishableState {
        on_hand,
        pipeline_orders: pipeline_orders.clone(),
    };
    let summary = policy_discounted_return_summary(
        policy_name,
        &params,
        &state,
        horizon,
        &seeds,
        pipeline_orders.len() + 1,
        max_order_size,
        demand_mean,
        demand_cov,
        holding_cost,
        shortage_cost,
        waste_cost,
        procurement_cost,
        warm_up_periods_ratio,
        gamma,
        parse_issuing_policy(issuing_policy)?,
    )?;
    heuristic_candidate_to_py(py, &params, &summary)
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
    horizon,
    seed,
    max_order_size,
    demand_mean,
    demand_cov,
    holding_cost,
    shortage_cost,
    waste_cost,
    position_upper_bound,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    issuing_policy="fifo",
    top_k=12
))]
fn perishable_inventory_base_stock_search(
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    horizon: usize,
    seed: u64,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
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
        pipeline_orders: pipeline_orders.clone(),
    };
    search_base_stock(
        &state,
        horizon,
        seed,
        pipeline_orders.len() + 1,
        max_order_size,
        demand_mean,
        demand_cov,
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
    horizon,
    seed,
    max_order_size,
    demand_mean,
    demand_cov,
    holding_cost,
    shortage_cost,
    waste_cost,
    position_upper_bound,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    gamma=0.99,
    issuing_policy="fifo",
    top_k=12
))]
fn perishable_inventory_base_stock_search_discounted_return(
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    horizon: usize,
    seed: u64,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    position_upper_bound: usize,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: &str,
    top_k: usize,
) -> PyResult<((usize, f64), Vec<(usize, f64)>)> {
    let state = PerishableState {
        on_hand,
        pipeline_orders: pipeline_orders.clone(),
    };
    search_base_stock_discounted_return(
        &state,
        horizon,
        seed,
        pipeline_orders.len() + 1,
        max_order_size,
        demand_mean,
        demand_cov,
        holding_cost,
        shortage_cost,
        waste_cost,
        procurement_cost,
        warm_up_periods_ratio,
        gamma,
        parse_issuing_policy(issuing_policy)?,
        position_upper_bound,
        top_k,
    )
}

#[pyfunction]
#[pyo3(signature = (
    on_hand,
    pipeline_orders,
    horizon,
    seeds,
    max_order_size,
    demand_mean,
    demand_cov,
    holding_cost,
    shortage_cost,
    waste_cost,
    position_upper_bound,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    gamma=0.99,
    issuing_policy="fifo",
    top_k=12
))]
fn perishable_inventory_base_stock_search_discounted_return_summary(
    py: Python<'_>,
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    horizon: usize,
    seeds: Vec<u64>,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    position_upper_bound: usize,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: &str,
    top_k: usize,
) -> PyResult<PyObject> {
    let state = PerishableState {
        on_hand,
        pipeline_orders: pipeline_orders.clone(),
    };
    let (best, top) = search_base_stock_discounted_return_summary(
        &state,
        horizon,
        &seeds,
        pipeline_orders.len() + 1,
        max_order_size,
        demand_mean,
        demand_cov,
        holding_cost,
        shortage_cost,
        waste_cost,
        procurement_cost,
        warm_up_periods_ratio,
        gamma,
        parse_issuing_policy(issuing_policy)?,
        position_upper_bound,
        top_k,
    )?;
    let best_obj = heuristic_candidate_to_py(py, &[best.0], &best.1)?;
    let top_objs = top
        .iter()
        .map(|(level, summary)| heuristic_candidate_to_py(py, &[*level], summary))
        .collect::<PyResult<Vec<_>>>()?;
    heuristic_search_to_py(py, best_obj, top_objs)
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

#[pyfunction]
#[pyo3(signature = (
    on_hand,
    pipeline_orders,
    horizon,
    seed,
    max_order_size,
    demand_mean,
    demand_cov,
    holding_cost,
    shortage_cost,
    waste_cost,
    position_upper_bound,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    issuing_policy="fifo",
    top_k=12
))]
fn perishable_inventory_bsp_low_ew_search(
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    horizon: usize,
    seed: u64,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
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
        pipeline_orders: pipeline_orders.clone(),
    };
    search_bsp_low_ew(
        &state,
        horizon,
        seed,
        pipeline_orders.len() + 1,
        max_order_size,
        demand_mean,
        demand_cov,
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
    horizon,
    seed,
    max_order_size,
    demand_mean,
    demand_cov,
    holding_cost,
    shortage_cost,
    waste_cost,
    position_upper_bound,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    gamma=0.99,
    issuing_policy="fifo",
    top_k=12
))]
fn perishable_inventory_bsp_low_ew_search_discounted_return(
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    horizon: usize,
    seed: u64,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    position_upper_bound: usize,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: &str,
    top_k: usize,
) -> PyResult<((usize, usize, usize, f64), Vec<(usize, usize, usize, f64)>)> {
    let state = PerishableState {
        on_hand,
        pipeline_orders: pipeline_orders.clone(),
    };
    search_bsp_low_ew_discounted_return(
        &state,
        horizon,
        seed,
        pipeline_orders.len() + 1,
        max_order_size,
        demand_mean,
        demand_cov,
        holding_cost,
        shortage_cost,
        waste_cost,
        procurement_cost,
        warm_up_periods_ratio,
        gamma,
        parse_issuing_policy(issuing_policy)?,
        position_upper_bound,
        top_k,
    )
}

#[pyfunction]
#[pyo3(signature = (
    on_hand,
    pipeline_orders,
    horizon,
    seeds,
    max_order_size,
    demand_mean,
    demand_cov,
    holding_cost,
    shortage_cost,
    waste_cost,
    position_upper_bound,
    procurement_cost=0.0,
    warm_up_periods_ratio=0.2,
    gamma=0.99,
    issuing_policy="fifo",
    top_k=12
))]
fn perishable_inventory_bsp_low_ew_search_discounted_return_summary(
    py: Python<'_>,
    on_hand: Vec<usize>,
    pipeline_orders: Vec<usize>,
    horizon: usize,
    seeds: Vec<u64>,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    position_upper_bound: usize,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: &str,
    top_k: usize,
) -> PyResult<PyObject> {
    let state = PerishableState {
        on_hand,
        pipeline_orders: pipeline_orders.clone(),
    };
    let (best, top) = search_bsp_low_ew_discounted_return_summary(
        &state,
        horizon,
        &seeds,
        pipeline_orders.len() + 1,
        max_order_size,
        demand_mean,
        demand_cov,
        holding_cost,
        shortage_cost,
        waste_cost,
        procurement_cost,
        warm_up_periods_ratio,
        gamma,
        parse_issuing_policy(issuing_policy)?,
        position_upper_bound,
        top_k,
    )?;
    let best_obj = heuristic_candidate_to_py(py, &[best.0, best.1, best.2], &best.3)?;
    let top_objs = top
        .iter()
        .map(
            |(low_inventory_level, high_inventory_level, threshold, summary)| {
                heuristic_candidate_to_py(
                    py,
                    &[*low_inventory_level, *high_inventory_level, *threshold],
                    summary,
                )
            },
        )
        .collect::<PyResult<Vec<_>>>()?;
    heuristic_search_to_py(py, best_obj, top_objs)
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        perishable_inventory_primary_reference_instance_name,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_list_reference_instances,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_get_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(perishable_inventory_exact_mdp_summary, m)?)?;
    m.add_function(wrap_pyfunction!(perishable_inventory_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_soft_tree_discounted_return,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_soft_tree_population_discounted_return,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_soft_tree_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_soft_tree_trace_summary_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_policy_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_policy_trace_summary_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(perishable_inventory_policy_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_policy_discounted_return,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_policy_discounted_return_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_base_stock_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(perishable_inventory_base_stock_search, m)?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_base_stock_search_discounted_return,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_base_stock_search_discounted_return_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_bsp_low_ew_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(perishable_inventory_bsp_low_ew_search, m)?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_bsp_low_ew_search_discounted_return,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        perishable_inventory_bsp_low_ew_search_discounted_return_summary,
        m
    )?)?;
    Ok(())
}
