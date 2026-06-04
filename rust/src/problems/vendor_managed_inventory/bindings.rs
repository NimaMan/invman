use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::vendor_managed_inventory::demand::parse_demand_distribution_kind;
use crate::problems::vendor_managed_inventory::env::{
    build_paper_policy_state, build_policy_state, initialize_paper_state, initialize_state,
    step_state,
};
use crate::problems::vendor_managed_inventory::heuristics::{
    dc_reserve_base_stock_shipment_quantity, policy_rollout_from_demands,
    retailer_base_stock_shipment_quantity, simulate_policy, PolicySimulationSummary,
};
use crate::problems::vendor_managed_inventory::literature::references::{
    build_sui_gosavi_lin_2010_case, SUI_GOSAVI_LIN_2010_GOSAVI_CASE_STUDY_WORKED_EXAMPLE,
};
use crate::problems::vendor_managed_inventory::rollout::{
    paper_population_rollout, paper_rollout, population_rollout, rollout, rollout_from_demands,
    VendorManagedInventoryPaperRolloutConfig, VendorManagedInventoryRolloutConfig,
};
use crate::problems::vendor_managed_inventory::verification::newsvendor_case::evaluate_newsvendor_worked_case;

fn build_rollout_config(
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    periods: usize,
    demand_kind: &str,
    demand_mean: f64,
    dc_replenishment_quantity: usize,
    dc_capacity: usize,
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    salvage_value_per_unit: f64,
    max_shipment_quantity: usize,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<VendorManagedInventoryRolloutConfig> {
    Ok(VendorManagedInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_kind: parse_demand_distribution_kind(demand_kind)?,
        demand_mean,
        dc_replenishment_quantity,
        dc_capacity,
        shipment_cost_per_unit,
        dc_holding_cost_per_unit,
        retailer_holding_cost_per_unit,
        stockout_cost_per_unit,
        salvage_value_per_unit,
        max_shipment_quantity,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    })
}

fn build_paper_rollout_config(
    case_id: usize,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    warmup_time: f64,
    evaluation_time: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<VendorManagedInventoryPaperRolloutConfig> {
    Ok(VendorManagedInventoryPaperRolloutConfig {
        case_id,
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        warmup_time,
        evaluation_time,
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
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn vendor_managed_inventory_newsvendor_worked_case_summary(py: Python<'_>) -> PyResult<PyObject> {
    let summary = evaluate_newsvendor_worked_case(&SUI_GOSAVI_LIN_2010_GOSAVI_CASE_STUDY_WORKED_EXAMPLE)?;
    let dict = PyDict::new_bound(py);
    dict.set_item("source", SUI_GOSAVI_LIN_2010_GOSAVI_CASE_STUDY_WORKED_EXAMPLE.source)?;
    dict.set_item("url", SUI_GOSAVI_LIN_2010_GOSAVI_CASE_STUDY_WORKED_EXAMPLE.url)?;
    dict.set_item(
        "matlab_code_url",
        SUI_GOSAVI_LIN_2010_GOSAVI_CASE_STUDY_WORKED_EXAMPLE.matlab_code_url,
    )?;
    dict.set_item("mean_demand_rate", summary.mean_demand_rate)?;
    dict.set_item("demand_variance", summary.demand_variance)?;
    dict.set_item("cycle_time_mean", summary.cycle_time_mean)?;
    dict.set_item("cycle_time_variance", summary.cycle_time_variance)?;
    dict.set_item("cycle_demand_mean", summary.cycle_demand_mean)?;
    dict.set_item("cycle_demand_variance", summary.cycle_demand_variance)?;
    dict.set_item("cycle_demand_stddev", summary.cycle_demand_stddev)?;
    dict.set_item("critical_ratio", summary.critical_ratio)?;
    dict.set_item("k", summary.k)?;
    dict.set_item(
        "mean_demand_heuristic_order_up_to",
        summary.mean_demand_heuristic_order_up_to,
    )?;
    dict.set_item("six_sigma_order_up_to", summary.six_sigma_order_up_to)?;
    dict.set_item("newsvendor_order_up_to", summary.newsvendor_order_up_to)?;
    dict.set_item(
        "displayed_newsvendor_order_up_to",
        SUI_GOSAVI_LIN_2010_GOSAVI_CASE_STUDY_WORKED_EXAMPLE.displayed_newsvendor_order_up_to,
    )?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
#[pyo3(signature = (case_id, seed=1234))]
fn vendor_managed_inventory_paper_initial_policy_state(
    case_id: usize,
    seed: u64,
) -> PyResult<Vec<f32>> {
    use rand::SeedableRng;
    let model = build_sui_gosavi_lin_2010_case(case_id).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown Sui/Gosavi/Lin 2010 case_id {}",
            case_id
        ))
    })?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let state = initialize_paper_state(&model, &mut rng)?;
    build_paper_policy_state(&model, &state)
}

#[pyfunction]
#[pyo3(signature = (
    case_id,
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    seed=1234,
    warmup_time=1_000_000.0,
    evaluation_time=100_000.0,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn vendor_managed_inventory_paper_soft_tree_rollout(
    case_id: usize,
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    seed: u64,
    warmup_time: f64,
    evaluation_time: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let config = build_paper_rollout_config(
        case_id,
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        warmup_time,
        evaluation_time,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
    )?;
    paper_rollout(&flat_params, &config, seed)
}

#[pyfunction]
#[pyo3(signature = (
    case_id,
    params_batch,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    seeds,
    warmup_time=1_000_000.0,
    evaluation_time=100_000.0,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn vendor_managed_inventory_paper_soft_tree_population_rollout(
    case_id: usize,
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    seeds: Vec<u64>,
    warmup_time: f64,
    evaluation_time: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let config = build_paper_rollout_config(
        case_id,
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        warmup_time,
        evaluation_time,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
    )?;
    paper_population_rollout(&params_batch, &config, &seeds)
}

#[pyfunction]
fn vendor_managed_inventory_build_policy_state(
    dc_on_hand: usize,
    retailer_on_hand: usize,
    retailer_pipeline: usize,
    expected_demand: f64,
    periods: usize,
    dc_capacity: usize,
    dc_replenishment_quantity: usize,
) -> PyResult<Vec<f32>> {
    let state = initialize_state(dc_on_hand, retailer_on_hand, retailer_pipeline, dc_capacity)?;
    build_policy_state(
        &state,
        expected_demand,
        periods,
        dc_capacity,
        dc_replenishment_quantity,
    )
}

#[pyfunction]
#[pyo3(signature = (
    dc_on_hand,
    retailer_on_hand,
    retailer_pipeline,
    shipment_quantity,
    realized_demand,
    dc_replenishment_quantity,
    dc_capacity,
    shipment_cost_per_unit,
    dc_holding_cost_per_unit,
    retailer_holding_cost_per_unit,
    stockout_cost_per_unit
))]
fn vendor_managed_inventory_step(
    py: Python<'_>,
    dc_on_hand: usize,
    retailer_on_hand: usize,
    retailer_pipeline: usize,
    shipment_quantity: usize,
    realized_demand: usize,
    dc_replenishment_quantity: usize,
    dc_capacity: usize,
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
) -> PyResult<PyObject> {
    let state = initialize_state(dc_on_hand, retailer_on_hand, retailer_pipeline, dc_capacity)?;
    let outcome = step_state(
        &state,
        shipment_quantity,
        realized_demand,
        dc_replenishment_quantity,
        dc_capacity,
        shipment_cost_per_unit,
        dc_holding_cost_per_unit,
        retailer_holding_cost_per_unit,
        stockout_cost_per_unit,
    )?;
    let dict = PyDict::new_bound(py);
    dict.set_item("next_period", outcome.next_state.period)?;
    dict.set_item("next_dc_on_hand", outcome.next_state.dc_on_hand)?;
    dict.set_item("next_retailer_on_hand", outcome.next_state.retailer_on_hand)?;
    dict.set_item(
        "next_retailer_pipeline",
        outcome.next_state.retailer_pipeline,
    )?;
    dict.set_item("arrivals_to_retailer", outcome.arrivals_to_retailer)?;
    dict.set_item("sales", outcome.sales)?;
    dict.set_item("lost_sales", outcome.lost_sales)?;
    dict.set_item("dc_replenishment", outcome.dc_replenishment)?;
    dict.set_item("shipment_cost", outcome.shipment_cost)?;
    dict.set_item("dc_holding_cost", outcome.dc_holding_cost)?;
    dict.set_item("retailer_holding_cost", outcome.retailer_holding_cost)?;
    dict.set_item("stockout_cost", outcome.stockout_cost)?;
    dict.set_item("period_cost", outcome.period_cost)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn vendor_managed_inventory_retailer_base_stock_shipment(
    dc_on_hand: usize,
    retailer_on_hand: usize,
    retailer_pipeline: usize,
    dc_capacity: usize,
    retailer_base_stock_level: usize,
    max_shipment_quantity: usize,
) -> PyResult<usize> {
    let state = initialize_state(dc_on_hand, retailer_on_hand, retailer_pipeline, dc_capacity)?;
    retailer_base_stock_shipment_quantity(&state, retailer_base_stock_level, max_shipment_quantity)
}

#[pyfunction]
fn vendor_managed_inventory_dc_reserve_base_stock_shipment(
    dc_on_hand: usize,
    retailer_on_hand: usize,
    retailer_pipeline: usize,
    dc_capacity: usize,
    retailer_base_stock_level: usize,
    dc_reserve_quantity: usize,
    max_shipment_quantity: usize,
) -> PyResult<usize> {
    let state = initialize_state(dc_on_hand, retailer_on_hand, retailer_pipeline, dc_capacity)?;
    dc_reserve_base_stock_shipment_quantity(
        &state,
        retailer_base_stock_level,
        dc_reserve_quantity,
        max_shipment_quantity,
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
    dc_on_hand,
    retailer_on_hand,
    retailer_pipeline,
    periods,
    demand_kind,
    demand_mean,
    dc_replenishment_quantity,
    dc_capacity,
    shipment_cost_per_unit,
    dc_holding_cost_per_unit,
    retailer_holding_cost_per_unit,
    stockout_cost_per_unit,
    salvage_value_per_unit,
    max_shipment_quantity,
    seed=1234,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn vendor_managed_inventory_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    dc_on_hand: usize,
    retailer_on_hand: usize,
    retailer_pipeline: usize,
    periods: usize,
    demand_kind: &str,
    demand_mean: f64,
    dc_replenishment_quantity: usize,
    dc_capacity: usize,
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    salvage_value_per_unit: f64,
    max_shipment_quantity: usize,
    seed: u64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state =
        initialize_state(dc_on_hand, retailer_on_hand, retailer_pipeline, dc_capacity)?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        periods,
        demand_kind,
        demand_mean,
        dc_replenishment_quantity,
        dc_capacity,
        shipment_cost_per_unit,
        dc_holding_cost_per_unit,
        retailer_holding_cost_per_unit,
        stockout_cost_per_unit,
        salvage_value_per_unit,
        max_shipment_quantity,
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
    dc_on_hand,
    retailer_on_hand,
    retailer_pipeline,
    periods,
    demand_kind,
    demand_mean,
    dc_replenishment_quantity,
    dc_capacity,
    shipment_cost_per_unit,
    dc_holding_cost_per_unit,
    retailer_holding_cost_per_unit,
    stockout_cost_per_unit,
    salvage_value_per_unit,
    max_shipment_quantity,
    seeds,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn vendor_managed_inventory_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    dc_on_hand: usize,
    retailer_on_hand: usize,
    retailer_pipeline: usize,
    periods: usize,
    demand_kind: &str,
    demand_mean: f64,
    dc_replenishment_quantity: usize,
    dc_capacity: usize,
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    salvage_value_per_unit: f64,
    max_shipment_quantity: usize,
    seeds: Vec<u64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let initial_state =
        initialize_state(dc_on_hand, retailer_on_hand, retailer_pipeline, dc_capacity)?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        periods,
        demand_kind,
        demand_mean,
        dc_replenishment_quantity,
        dc_capacity,
        shipment_cost_per_unit,
        dc_holding_cost_per_unit,
        retailer_holding_cost_per_unit,
        stockout_cost_per_unit,
        salvage_value_per_unit,
        max_shipment_quantity,
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
    dc_on_hand,
    retailer_on_hand,
    retailer_pipeline,
    realized_demands,
    demand_mean,
    dc_replenishment_quantity,
    dc_capacity,
    shipment_cost_per_unit,
    dc_holding_cost_per_unit,
    retailer_holding_cost_per_unit,
    stockout_cost_per_unit,
    salvage_value_per_unit,
    max_shipment_quantity,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn vendor_managed_inventory_soft_tree_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    dc_on_hand: usize,
    retailer_on_hand: usize,
    retailer_pipeline: usize,
    realized_demands: Vec<usize>,
    demand_mean: f64,
    dc_replenishment_quantity: usize,
    dc_capacity: usize,
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    salvage_value_per_unit: f64,
    max_shipment_quantity: usize,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state =
        initialize_state(dc_on_hand, retailer_on_hand, retailer_pipeline, dc_capacity)?;
    let config = build_rollout_config(
        input_dim,
        depth,
        min_values,
        max_values,
        action_mode,
        realized_demands.len(),
        "deterministic",
        demand_mean,
        dc_replenishment_quantity,
        dc_capacity,
        shipment_cost_per_unit,
        dc_holding_cost_per_unit,
        retailer_holding_cost_per_unit,
        stockout_cost_per_unit,
        salvage_value_per_unit,
        max_shipment_quantity,
        discount_factor,
        temperature,
        split_type,
        leaf_type,
        allowed_values,
    )?;
    rollout_from_demands(&flat_params, &config, &initial_state, &realized_demands)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    dc_on_hand,
    retailer_on_hand,
    retailer_pipeline,
    realized_demands,
    dc_replenishment_quantity,
    dc_capacity,
    shipment_cost_per_unit,
    dc_holding_cost_per_unit,
    retailer_holding_cost_per_unit,
    stockout_cost_per_unit,
    max_shipment_quantity,
    discount_factor=0.99,
    salvage_value_per_unit=0.0
))]
fn vendor_managed_inventory_policy_rollout_from_demands(
    policy_name: &str,
    params: Vec<f64>,
    dc_on_hand: usize,
    retailer_on_hand: usize,
    retailer_pipeline: usize,
    realized_demands: Vec<usize>,
    dc_replenishment_quantity: usize,
    dc_capacity: usize,
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    max_shipment_quantity: usize,
    discount_factor: f64,
    salvage_value_per_unit: f64,
) -> PyResult<f64> {
    let initial_state =
        initialize_state(dc_on_hand, retailer_on_hand, retailer_pipeline, dc_capacity)?;
    policy_rollout_from_demands(
        policy_name,
        &params,
        &initial_state,
        &realized_demands,
        dc_replenishment_quantity,
        dc_capacity,
        shipment_cost_per_unit,
        dc_holding_cost_per_unit,
        retailer_holding_cost_per_unit,
        stockout_cost_per_unit,
        max_shipment_quantity,
        discount_factor,
        salvage_value_per_unit,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    dc_on_hand,
    retailer_on_hand,
    retailer_pipeline,
    periods,
    replications,
    seed,
    demand_mean,
    demand_kind,
    dc_replenishment_quantity,
    dc_capacity,
    shipment_cost_per_unit,
    dc_holding_cost_per_unit,
    retailer_holding_cost_per_unit,
    stockout_cost_per_unit,
    max_shipment_quantity,
    discount_factor=0.99,
    salvage_value_per_unit=0.0
))]
fn vendor_managed_inventory_simulate_policy(
    py: Python<'_>,
    policy_name: &str,
    params: Vec<f64>,
    dc_on_hand: usize,
    retailer_on_hand: usize,
    retailer_pipeline: usize,
    periods: usize,
    replications: usize,
    seed: u64,
    demand_mean: f64,
    demand_kind: &str,
    dc_replenishment_quantity: usize,
    dc_capacity: usize,
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    max_shipment_quantity: usize,
    discount_factor: f64,
    salvage_value_per_unit: f64,
) -> PyResult<PyObject> {
    let initial_state =
        initialize_state(dc_on_hand, retailer_on_hand, retailer_pipeline, dc_capacity)?;
    let summary = simulate_policy(
        policy_name,
        &params,
        &initial_state,
        periods,
        replications,
        seed,
        demand_mean,
        parse_demand_distribution_kind(demand_kind)?,
        dc_replenishment_quantity,
        dc_capacity,
        shipment_cost_per_unit,
        dc_holding_cost_per_unit,
        retailer_holding_cost_per_unit,
        stockout_cost_per_unit,
        max_shipment_quantity,
        discount_factor,
        salvage_value_per_unit,
    )?;
    simulation_summary_to_py(py, &summary)
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        vendor_managed_inventory_newsvendor_worked_case_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vendor_managed_inventory_paper_initial_policy_state,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vendor_managed_inventory_paper_soft_tree_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vendor_managed_inventory_paper_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vendor_managed_inventory_build_policy_state,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(vendor_managed_inventory_step, m)?)?;
    m.add_function(wrap_pyfunction!(
        vendor_managed_inventory_retailer_base_stock_shipment,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vendor_managed_inventory_dc_reserve_base_stock_shipment,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vendor_managed_inventory_soft_tree_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vendor_managed_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vendor_managed_inventory_soft_tree_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vendor_managed_inventory_policy_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vendor_managed_inventory_simulate_policy,
        m
    )?)?;
    Ok(())
}
