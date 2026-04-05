use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::one_warehouse_multi_retailer::allocation::{
    parse_allocation_policy,
};
use crate::problems::one_warehouse_multi_retailer::demand::{
    parse_demand_distribution_kind, DemandModel,
};
use crate::problems::one_warehouse_multi_retailer::env::{
    parse_customer_behavior_model, OneWarehouseMultiRetailerState,
};
use crate::problems::one_warehouse_multi_retailer::heuristics::{
    echelon_base_stock_orders, policy_rollout_from_paths,
};
use crate::problems::one_warehouse_multi_retailer::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_paths,
    OneWarehouseMultiRetailerRolloutConfig,
};

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
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
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
    retailer_target_inventory_positions=None,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
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
    retailer_target_inventory_positions: Option<Vec<usize>>,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
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
    retailer_target_inventory_positions=None,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
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
    retailer_target_inventory_positions: Option<Vec<usize>>,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
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
    retailer_target_inventory_positions=None,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
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
    retailer_target_inventory_positions: Option<Vec<usize>>,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
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
        temperature,
        split_type,
        leaf_type,
        allowed_values,
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
    echelon_base_stock_orders(&state, warehouse_base_stock_level, &retailer_base_stock_levels)
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    Ok(())
}
