use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::dual_sourcing::heuristics::{
    search_capped_dual_index_from_demands, search_dual_index_from_demands,
    search_single_index_from_demands, search_tailored_base_surge_from_demands,
};
use crate::problems::dual_sourcing::policies::parse_action_adapter;
use crate::problems::dual_sourcing::rollout::{
    population_rollout as dual_sourcing_population_rollout, rollout as dual_sourcing_rollout,
    rollout_from_demands as dual_sourcing_rollout_from_demands, DualSourcingRolloutConfig,
};

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
