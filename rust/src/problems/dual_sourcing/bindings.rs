use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::dual_sourcing::bounded_dp::{
    benchmark_reference_instance, solve_bounded_average_cost_optimal_policy, BenchmarkReport,
    BoundedDpConfig, HeuristicBenchmarkResult,
};
use crate::problems::dual_sourcing::heuristics::{
    search_capped_dual_index_from_demands, search_dual_index_from_demands,
    search_single_index_from_demands, search_tailored_base_surge_from_demands,
};
use crate::problems::dual_sourcing::policies::parse_action_adapter;
use crate::problems::dual_sourcing::references::{
    get_figure_9_gap_reference, get_primary_reference_instance, get_reference_instance,
    list_reference_instances, DualSourcingReferenceInstance,
};
use crate::problems::dual_sourcing::rollout::{
    population_rollout as dual_sourcing_population_rollout, rollout as dual_sourcing_rollout,
    rollout_from_demands as dual_sourcing_rollout_from_demands, DualSourcingRolloutConfig,
};

const SYNTHETIC_REFERENCE_SOURCE: &str = "invman Rust bounded average-cost DP";
const SYNTHETIC_REFERENCE_URL: &str = "local";
const SYNTHETIC_REFERENCE_NOTES: &str = "Synthetic bounded-DP validation instance.";

fn bounded_dp_config(
    inventory_lower: i64,
    inventory_upper: i64,
    tolerance: f64,
    max_iterations: usize,
) -> BoundedDpConfig {
    BoundedDpConfig {
        inventory_lower,
        inventory_upper,
        tolerance,
        max_iterations,
    }
}

fn deterministic_initial_state(regular_lead_time: usize, demand_low: usize, demand_high: usize) -> Vec<i64> {
    let mean_demand = 0.5 * (demand_low + demand_high) as f64;
    let mut state = vec![((regular_lead_time + 1) as f64 * mean_demand).round() as i64];
    state.extend(vec![0; regular_lead_time.saturating_sub(1)]);
    state
}

fn reference_instance_to_py(
    py: Python<'_>,
    instance: &DualSourcingReferenceInstance,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", instance.name)?;
    dict.set_item("source", instance.source)?;
    dict.set_item("url", instance.url)?;
    dict.set_item("regular_lead_time", instance.regular_lead_time)?;
    dict.set_item("expedited_lead_time", instance.expedited_lead_time)?;
    dict.set_item("regular_order_cost", instance.regular_order_cost)?;
    dict.set_item("expedited_order_cost", instance.expedited_order_cost)?;
    dict.set_item("holding_cost", instance.holding_cost)?;
    dict.set_item("shortage_cost", instance.shortage_cost)?;
    dict.set_item("regular_max_order_size", instance.regular_max_order_size)?;
    dict.set_item("expedited_max_order_size", instance.expedited_max_order_size)?;
    dict.set_item("demand_low", instance.demand_low)?;
    dict.set_item("demand_high", instance.demand_high)?;
    dict.set_item("initial_state", deterministic_initial_state(
        instance.regular_lead_time,
        instance.demand_low,
        instance.demand_high,
    ))?;
    dict.set_item("notes", instance.notes)?;

    if let Some(figure_9) = get_figure_9_gap_reference(instance.name) {
        let published = PyDict::new_bound(py);
        published.set_item("source", figure_9.source)?;
        published.set_item("url", figure_9.url)?;
        published.set_item("capped_dual_index", figure_9.capped_dual_index_gap_pct)?;
        published.set_item("dual_index", figure_9.dual_index_gap_pct)?;
        published.set_item("single_index", figure_9.single_index_gap_pct)?;
        published.set_item("tailored_base_surge", figure_9.tailored_base_surge_gap_pct)?;
        published.set_item("a3c", figure_9.a3c_gap_pct)?;
        dict.set_item("published_optimality_gap_pct", published)?;
    } else {
        dict.set_item("published_optimality_gap_pct", py.None())?;
    }

    Ok(dict.into_any().unbind().into())
}

fn heuristic_benchmark_result_to_py(
    py: Python<'_>,
    heuristic: &HeuristicBenchmarkResult,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("policy_name", heuristic.policy_name)?;
    dict.set_item("params", heuristic.params.clone())?;
    dict.set_item("search_cost", heuristic.search_cost)?;
    dict.set_item("average_cost", heuristic.average_cost)?;
    dict.set_item("first_action", heuristic.first_action.to_vec())?;
    dict.set_item("optimality_gap_pct", heuristic.optimality_gap_pct)?;
    if let Some(published_gap) = heuristic.published_optimality_gap_pct {
        dict.set_item("published_optimality_gap_pct", published_gap)?;
        dict.set_item(
            "gap_delta_vs_literature_pct",
            heuristic.optimality_gap_pct - published_gap,
        )?;
    } else {
        dict.set_item("published_optimality_gap_pct", py.None())?;
        dict.set_item("gap_delta_vs_literature_pct", py.None())?;
    }
    Ok(dict.into_any().unbind().into())
}

fn benchmark_report_to_py(py: Python<'_>, report: &BenchmarkReport) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("reference_name", report.reference_name.clone())?;
    dict.set_item("initial_state", report.initial_state.clone())?;

    let optimal = PyDict::new_bound(py);
    optimal.set_item("average_cost", report.optimal.average_cost)?;
    optimal.set_item("first_action", report.optimal.first_action.to_vec())?;
    optimal.set_item("iterations", report.optimal.iterations)?;
    dict.set_item("optimal", optimal)?;

    let heuristics = report
        .heuristics
        .iter()
        .map(|heuristic| heuristic_benchmark_result_to_py(py, heuristic))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("heuristics", heuristics)?;

    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn dual_sourcing_primary_reference_instance_name() -> &'static str {
    get_primary_reference_instance().name
}

#[pyfunction]
fn dual_sourcing_list_reference_instances(py: Python<'_>) -> PyResult<Vec<PyObject>> {
    list_reference_instances()
        .iter()
        .map(|instance| reference_instance_to_py(py, instance))
        .collect()
}

#[pyfunction]
fn dual_sourcing_get_reference_instance(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    let instance = get_reference_instance(name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown dual-sourcing reference instance '{name}'"
        ))
    })?;
    reference_instance_to_py(py, instance)
}

#[pyfunction]
#[pyo3(signature = (
    regular_lead_time,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    regular_max_order_size,
    expedited_max_order_size,
    demand_low,
    demand_high,
    inventory_lower=-40,
    inventory_upper=60,
    tolerance=1e-8,
    max_iterations=400
))]
fn dual_sourcing_bounded_average_cost_optimal_summary(
    py: Python<'_>,
    regular_lead_time: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    demand_low: usize,
    demand_high: usize,
    inventory_lower: i64,
    inventory_upper: i64,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<PyObject> {
    let reference = DualSourcingReferenceInstance {
        name: "synthetic",
        source: SYNTHETIC_REFERENCE_SOURCE,
        url: SYNTHETIC_REFERENCE_URL,
        regular_lead_time,
        expedited_lead_time: 0,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        regular_max_order_size,
        expedited_max_order_size,
        demand_low,
        demand_high,
        notes: SYNTHETIC_REFERENCE_NOTES,
    };
    let config = bounded_dp_config(inventory_lower, inventory_upper, tolerance, max_iterations);
    let summary = solve_bounded_average_cost_optimal_policy(&reference, &config)?;

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "initial_state",
        deterministic_initial_state(regular_lead_time, demand_low, demand_high),
    )?;
    dict.set_item("average_cost", summary.average_cost)?;
    dict.set_item("first_action", summary.first_action.to_vec())?;
    dict.set_item("iterations", summary.iterations)?;
    dict.set_item("inventory_bounds", vec![inventory_lower, inventory_upper])?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
#[pyo3(signature = (
    reference_instance_name,
    inventory_lower=-40,
    inventory_upper=60,
    tolerance=1e-8,
    max_iterations=400,
    search_seed=123,
    search_horizon=6000,
    warm_up_periods_ratio=0.2
))]
fn dual_sourcing_reference_benchmark_summary(
    py: Python<'_>,
    reference_instance_name: &str,
    inventory_lower: i64,
    inventory_upper: i64,
    tolerance: f64,
    max_iterations: usize,
    search_seed: u64,
    search_horizon: usize,
    warm_up_periods_ratio: f64,
) -> PyResult<PyObject> {
    let config = bounded_dp_config(inventory_lower, inventory_upper, tolerance, max_iterations);
    let report = benchmark_reference_instance(
        reference_instance_name,
        &config,
        search_seed,
        search_horizon,
        warm_up_periods_ratio,
    )?;
    benchmark_report_to_py(py, &report)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    regular_lead_time,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    regular_max_order_size,
    expedited_max_order_size,
    demand_low,
    demand_high,
    horizon=2000,
    seed=1234,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    action_adapter="identity",
    allowed_values=None
))]
fn dual_sourcing_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    regular_lead_time: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    demand_low: usize,
    demand_high: usize,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    action_adapter: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let config = DualSourcingRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        regular_lead_time,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        regular_max_order_size,
        expedited_max_order_size,
        demand_low,
        demand_high,
        horizon,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        action_adapter: parse_action_adapter(action_adapter)?,
    };
    dual_sourcing_rollout(&flat_params, &config, seed)
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    regular_lead_time,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    regular_max_order_size,
    expedited_max_order_size,
    demand_low,
    demand_high,
    seeds,
    horizon=2000,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    action_adapter="identity",
    allowed_values=None
))]
fn dual_sourcing_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    regular_lead_time: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    demand_low: usize,
    demand_high: usize,
    seeds: Vec<u64>,
    horizon: usize,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    action_adapter: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let config = DualSourcingRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        regular_lead_time,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        regular_max_order_size,
        expedited_max_order_size,
        demand_low,
        demand_high,
        horizon,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        action_adapter: parse_action_adapter(action_adapter)?,
    };
    dual_sourcing_population_rollout(&params_batch, &config, &seeds)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    state,
    demands,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    regular_max_order_size,
    expedited_max_order_size,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    action_adapter="identity",
    allowed_values=None
))]
fn dual_sourcing_soft_tree_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    state: Vec<i64>,
    demands: Vec<usize>,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    action_adapter: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let config = DualSourcingRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        regular_lead_time: state.len(),
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        regular_max_order_size,
        expedited_max_order_size,
        demand_low: 0,
        demand_high: 0,
        horizon: demands.len(),
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        action_adapter: parse_action_adapter(action_adapter)?,
    };
    dual_sourcing_rollout_from_demands(&flat_params, &config, state, &demands)
}

#[pyfunction]
#[pyo3(signature = (
    state,
    demands,
    regular_max_order_size,
    expedited_max_order_size,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    warm_up_periods_ratio=0.2,
    target_upper_bound=20,
    top_k=10
))]
fn dual_sourcing_single_index_search_from_demands(
    state: Vec<i64>,
    demands: Vec<usize>,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    warm_up_periods_ratio: f64,
    target_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    search_single_index_from_demands(
        &state,
        &demands,
        regular_max_order_size,
        expedited_max_order_size,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        warm_up_periods_ratio,
        target_upper_bound,
        top_k,
    )
}

#[pyfunction]
#[pyo3(signature = (
    state,
    demands,
    regular_max_order_size,
    expedited_max_order_size,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    warm_up_periods_ratio=0.2,
    target_upper_bound=20,
    top_k=10
))]
fn dual_sourcing_dual_index_search_from_demands(
    state: Vec<i64>,
    demands: Vec<usize>,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    warm_up_periods_ratio: f64,
    target_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    search_dual_index_from_demands(
        &state,
        &demands,
        regular_max_order_size,
        expedited_max_order_size,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        warm_up_periods_ratio,
        target_upper_bound,
        top_k,
    )
}

#[pyfunction]
#[pyo3(signature = (
    state,
    demands,
    regular_max_order_size,
    expedited_max_order_size,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    warm_up_periods_ratio=0.2,
    target_upper_bound=20,
    top_k=10
))]
fn dual_sourcing_capped_dual_index_search_from_demands(
    state: Vec<i64>,
    demands: Vec<usize>,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    warm_up_periods_ratio: f64,
    target_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, usize, usize, f64), Vec<(usize, usize, usize, f64)>)> {
    search_capped_dual_index_from_demands(
        &state,
        &demands,
        regular_max_order_size,
        expedited_max_order_size,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        warm_up_periods_ratio,
        target_upper_bound,
        top_k,
    )
}

#[pyfunction]
#[pyo3(signature = (
    state,
    demands,
    regular_max_order_size,
    expedited_max_order_size,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    warm_up_periods_ratio=0.2,
    target_upper_bound=20,
    top_k=10
))]
fn dual_sourcing_tailored_base_surge_search_from_demands(
    state: Vec<i64>,
    demands: Vec<usize>,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    warm_up_periods_ratio: f64,
    target_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    search_tailored_base_surge_from_demands(
        &state,
        &demands,
        regular_max_order_size,
        expedited_max_order_size,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        warm_up_periods_ratio,
        target_upper_bound,
        top_k,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        dual_sourcing_primary_reference_instance_name,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_list_reference_instances,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_get_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_bounded_average_cost_optimal_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_reference_benchmark_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(dual_sourcing_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_soft_tree_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_single_index_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_dual_index_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_capped_dual_index_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_tailored_base_surge_search_from_demands,
        m
    )?)?;
    Ok(())
}
