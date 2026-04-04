use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::nonstationary_lot_sizing::demand::parse_demand_distribution_kind;
use crate::problems::nonstationary_lot_sizing::heuristics::{
    policy_rollout_from_demands, rolling_dp_s_s_levels, rolling_dp_s_s_sequence,
    simple_s_s_levels, simulate_periodic_s_s_policy, simulate_policy,
};
use crate::problems::nonstationary_lot_sizing::rollout::{
    build_initial_state_from_forecast, population_rollout as lot_sizing_population_rollout,
    rollout as lot_sizing_rollout, rollout_from_demands as lot_sizing_rollout_from_demands,
    NonstationaryLotSizingRolloutConfig,
};

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    forecast_means,
    forecast_horizon,
    initial_net_inventory,
    pipeline_orders,
    holding_cost,
    shortage_cost,
    fixed_order_cost,
    periods,
    seed=1234,
    demand_distribution="cv_normal",
    demand_cv=0.2,
    procurement_cost=0.0,
    lost_sales=true,
    warm_up_periods_ratio=0.0,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn nonstationary_lot_sizing_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    forecast_means: Vec<f64>,
    forecast_horizon: usize,
    initial_net_inventory: f64,
    pipeline_orders: Vec<f64>,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    periods: usize,
    seed: u64,
    demand_distribution: &str,
    demand_cv: f64,
    procurement_cost: f64,
    lost_sales: bool,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state_from_forecast(
        &forecast_means,
        forecast_horizon,
        initial_net_inventory,
        &pipeline_orders,
    )?;
    let config = NonstationaryLotSizingRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        lost_sales,
        demand_cv,
        demand_kind: parse_demand_distribution_kind(demand_distribution)?,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    lot_sizing_rollout(&flat_params, &config, &forecast_means, &initial_state, seed)
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    forecast_means,
    forecast_horizon,
    initial_net_inventory,
    pipeline_orders,
    holding_cost,
    shortage_cost,
    fixed_order_cost,
    periods,
    seeds,
    demand_distribution="cv_normal",
    demand_cv=0.2,
    procurement_cost=0.0,
    lost_sales=true,
    warm_up_periods_ratio=0.0,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn nonstationary_lot_sizing_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    forecast_means: Vec<f64>,
    forecast_horizon: usize,
    initial_net_inventory: f64,
    pipeline_orders: Vec<f64>,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    periods: usize,
    seeds: Vec<u64>,
    demand_distribution: &str,
    demand_cv: f64,
    procurement_cost: f64,
    lost_sales: bool,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let initial_state = build_initial_state_from_forecast(
        &forecast_means,
        forecast_horizon,
        initial_net_inventory,
        &pipeline_orders,
    )?;
    let config = NonstationaryLotSizingRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        lost_sales,
        demand_cv,
        demand_kind: parse_demand_distribution_kind(demand_distribution)?,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    lot_sizing_population_rollout(
        &params_batch,
        &config,
        &forecast_means,
        &initial_state,
        &seeds,
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
    forecast_means,
    forecast_horizon,
    initial_net_inventory,
    pipeline_orders,
    demands,
    holding_cost,
    shortage_cost,
    fixed_order_cost,
    demand_distribution="cv_normal",
    demand_cv=0.2,
    procurement_cost=0.0,
    lost_sales=true,
    warm_up_periods_ratio=0.0,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn nonstationary_lot_sizing_soft_tree_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    forecast_means: Vec<f64>,
    forecast_horizon: usize,
    initial_net_inventory: f64,
    pipeline_orders: Vec<f64>,
    demands: Vec<f64>,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    demand_distribution: &str,
    demand_cv: f64,
    procurement_cost: f64,
    lost_sales: bool,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state_from_forecast(
        &forecast_means,
        forecast_horizon,
        initial_net_inventory,
        &pipeline_orders,
    )?;
    let config = NonstationaryLotSizingRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods: demands.len(),
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        lost_sales,
        demand_cv,
        demand_kind: parse_demand_distribution_kind(demand_distribution)?,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    lot_sizing_rollout_from_demands(
        &flat_params,
        &config,
        &initial_state,
        &forecast_means,
        &demands,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    forecast_means,
    forecast_horizon,
    initial_net_inventory,
    pipeline_orders,
    demands,
    holding_cost,
    shortage_cost,
    fixed_order_cost,
    demand_distribution="cv_normal",
    demand_cv=0.2,
    procurement_cost=0.0,
    lost_sales=true,
    warm_up_periods_ratio=0.0
))]
fn nonstationary_lot_sizing_policy_rollout_from_demands(
    policy_name: &str,
    params: Vec<f64>,
    forecast_means: Vec<f64>,
    forecast_horizon: usize,
    initial_net_inventory: f64,
    pipeline_orders: Vec<f64>,
    demands: Vec<f64>,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    demand_distribution: &str,
    demand_cv: f64,
    procurement_cost: f64,
    lost_sales: bool,
    warm_up_periods_ratio: f64,
) -> PyResult<f64> {
    let initial_state = build_initial_state_from_forecast(
        &forecast_means,
        forecast_horizon,
        initial_net_inventory,
        &pipeline_orders,
    )?;
    policy_rollout_from_demands(
        policy_name,
        &params,
        &initial_state,
        &forecast_means,
        &demands,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        lost_sales,
        demand_cv,
        parse_demand_distribution_kind(demand_distribution)?,
        warm_up_periods_ratio,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    forecast_means,
    forecast_horizon,
    initial_net_inventory,
    pipeline_orders,
    periods,
    replications=1000,
    seed=1234,
    holding_cost=1.0,
    shortage_cost=5.0,
    fixed_order_cost=10.0,
    demand_distribution="cv_normal",
    demand_cv=0.2,
    procurement_cost=0.0,
    lost_sales=true
))]
fn nonstationary_lot_sizing_simulate_policy(
    policy_name: &str,
    params: Vec<f64>,
    forecast_means: Vec<f64>,
    forecast_horizon: usize,
    initial_net_inventory: f64,
    pipeline_orders: Vec<f64>,
    periods: usize,
    replications: usize,
    seed: u64,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    demand_distribution: &str,
    demand_cv: f64,
    procurement_cost: f64,
    lost_sales: bool,
) -> PyResult<(f64, f64, f64)> {
    let initial_state = build_initial_state_from_forecast(
        &forecast_means,
        forecast_horizon,
        initial_net_inventory,
        &pipeline_orders,
    )?;
    let summary = simulate_policy(
        policy_name,
        &params,
        &initial_state,
        &forecast_means,
        periods,
        replications,
        seed,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        lost_sales,
        demand_cv,
        parse_demand_distribution_kind(demand_distribution)?,
    )?;
    Ok((summary.mean_cost, summary.cost_std, summary.shortage_rate))
}

#[pyfunction]
#[pyo3(signature = (
    forecast_window,
    lead_time,
    holding_cost,
    shortage_cost,
    fixed_order_cost,
    demand_distribution="cv_normal",
    demand_cv=0.2
))]
fn nonstationary_lot_sizing_simple_s_s_levels(
    forecast_window: Vec<f64>,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    demand_distribution: &str,
    demand_cv: f64,
) -> PyResult<(f64, f64)> {
    Ok(simple_s_s_levels(
        &forecast_window,
        lead_time,
        holding_cost,
        shortage_cost,
        fixed_order_cost,
        demand_cv,
        parse_demand_distribution_kind(demand_distribution)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    forecast_window,
    lead_time,
    holding_cost,
    shortage_cost,
    fixed_order_cost,
    demand_distribution="poisson",
    discount_factor=0.99,
    stationary_tail_periods=32
))]
fn nonstationary_lot_sizing_rolling_dp_s_s_levels(
    forecast_window: Vec<f64>,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    demand_distribution: &str,
    discount_factor: f64,
    stationary_tail_periods: usize,
) -> PyResult<(f64, f64)> {
    let levels = rolling_dp_s_s_levels(
        &forecast_window,
        lead_time,
        holding_cost,
        shortage_cost,
        fixed_order_cost,
        parse_demand_distribution_kind(demand_distribution)?,
        discount_factor,
        stationary_tail_periods,
    )?;
    Ok((levels.reorder_point as f64, levels.order_up_to as f64))
}

#[pyfunction]
#[pyo3(signature = (
    forecast_means,
    periods,
    forecast_horizon,
    lead_time,
    holding_cost,
    shortage_cost,
    fixed_order_cost,
    demand_distribution="poisson",
    discount_factor=0.99,
    stationary_tail_periods=32
))]
fn nonstationary_lot_sizing_rolling_dp_s_s_sequence(
    forecast_means: Vec<f64>,
    periods: usize,
    forecast_horizon: usize,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    demand_distribution: &str,
    discount_factor: f64,
    stationary_tail_periods: usize,
) -> PyResult<Vec<(f64, f64)>> {
    Ok(rolling_dp_s_s_sequence(
        &forecast_means,
        periods,
        forecast_horizon,
        lead_time,
        holding_cost,
        shortage_cost,
        fixed_order_cost,
        parse_demand_distribution_kind(demand_distribution)?,
        discount_factor,
        stationary_tail_periods,
    )?
    .into_iter()
    .map(|levels| (levels.reorder_point as f64, levels.order_up_to as f64))
    .collect())
}

#[pyfunction]
#[pyo3(signature = (
    forecast_means,
    forecast_horizon,
    initial_net_inventory,
    pipeline_orders,
    periods,
    replications=1000,
    seed=1234,
    holding_cost=1.0,
    shortage_cost=5.0,
    fixed_order_cost=10.0,
    demand_distribution="poisson",
    demand_cv=0.0,
    procurement_cost=0.0,
    lost_sales=true,
    discount_factor=0.99,
    stationary_tail_periods=32
))]
fn nonstationary_lot_sizing_simulate_rolling_dp_policy(
    forecast_means: Vec<f64>,
    forecast_horizon: usize,
    initial_net_inventory: f64,
    pipeline_orders: Vec<f64>,
    periods: usize,
    replications: usize,
    seed: u64,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    demand_distribution: &str,
    demand_cv: f64,
    procurement_cost: f64,
    lost_sales: bool,
    discount_factor: f64,
    stationary_tail_periods: usize,
) -> PyResult<(f64, f64, f64)> {
    let demand_kind = parse_demand_distribution_kind(demand_distribution)?;
    let initial_state = build_initial_state_from_forecast(
        &forecast_means,
        forecast_horizon,
        initial_net_inventory,
        &pipeline_orders,
    )?;
    let sequence = rolling_dp_s_s_sequence(
        &forecast_means,
        periods,
        forecast_horizon,
        pipeline_orders.len(),
        holding_cost,
        shortage_cost,
        fixed_order_cost,
        demand_kind,
        discount_factor,
        stationary_tail_periods,
    )?;
    let summary = simulate_periodic_s_s_policy(
        &sequence,
        &initial_state,
        &forecast_means,
        replications,
        seed,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        lost_sales,
        demand_cv,
        demand_kind,
    )?;
    Ok((summary.mean_cost, summary.cost_std, summary.shortage_rate))
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        nonstationary_lot_sizing_soft_tree_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nonstationary_lot_sizing_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nonstationary_lot_sizing_soft_tree_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nonstationary_lot_sizing_policy_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nonstationary_lot_sizing_simulate_policy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nonstationary_lot_sizing_simple_s_s_levels,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nonstationary_lot_sizing_rolling_dp_s_s_levels,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nonstationary_lot_sizing_rolling_dp_s_s_sequence,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nonstationary_lot_sizing_simulate_rolling_dp_policy,
        m
    )?)?;
    Ok(())
}
