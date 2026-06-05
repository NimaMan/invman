use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::multi_echelon::general_backorder_fixed_cost::env::{
    build_raw_state, initialize_zero_state, GeneralBackorderFixedCostNetwork, RetailConnectionEdge,
};
use crate::problems::multi_echelon::general_backorder_fixed_cost::heuristics::{
    parse_benchmark_order_routing_mode, simulate_node_base_stock_policy_audit_with_mode,
    simulate_node_base_stock_policy_with_mode,
};
use crate::problems::multi_echelon::general_backorder_fixed_cost::references::{
    parse_demand_mode, reference_instance_by_name, GeneralBackorderFixedCostReferenceInstance,
    GEEVERS_2023_REFERENCE, LITERATURE_REFERENCE_INSTANCES, PRIMARY_REFERENCE_INSTANCE,
};
use crate::problems::multi_echelon::general_backorder_fixed_cost::rollout::{
    build_initial_state, parse_policy_action_mode, parse_policy_feature_mode, population_rollout,
    rollout, GeneralBackorderFixedCostRolloutConfig,
};

fn build_network(
    reference: &GeneralBackorderFixedCostReferenceInstance,
) -> GeneralBackorderFixedCostNetwork {
    GeneralBackorderFixedCostNetwork {
        num_suppliers: reference.num_suppliers,
        num_warehouses: reference.num_warehouses,
        num_retailers: reference.num_retailers,
        supplier_lead_times: reference.supplier_lead_times.to_vec(),
        retail_edges: reference.retail_edges.to_vec(),
    }
}

fn retail_edge_rows_to_py(
    py: Python<'_>,
    edges: &[RetailConnectionEdge],
) -> PyResult<Vec<PyObject>> {
    edges
        .iter()
        .map(|edge| {
            let dict = PyDict::new_bound(py);
            dict.set_item("warehouse_idx", edge.warehouse_idx)?;
            dict.set_item("retailer_idx", edge.retailer_idx)?;
            dict.set_item("connection_weight", edge.connection_weight)?;
            dict.set_item("lead_time", edge.lead_time)?;
            Ok(dict.into_any().unbind().into())
        })
        .collect()
}

fn reference_to_py(
    py: Python<'_>,
    reference: &GeneralBackorderFixedCostReferenceInstance,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", reference.name)?;
    dict.set_item("source", reference.source)?;
    dict.set_item("url", reference.url)?;
    dict.set_item("literature_verified", reference.literature_verified)?;
    dict.set_item("num_suppliers", reference.num_suppliers)?;
    dict.set_item("num_warehouses", reference.num_warehouses)?;
    dict.set_item("num_retailers", reference.num_retailers)?;
    dict.set_item(
        "supplier_lead_times",
        reference.supplier_lead_times.to_vec(),
    )?;
    dict.set_item(
        "retail_edges",
        retail_edge_rows_to_py(py, reference.retail_edges)?,
    )?;
    dict.set_item("demand_mode", reference.demand_mode)?;
    dict.set_item("demand_alpha_min", reference.demand_alpha_min)?;
    dict.set_item("demand_alpha_max", reference.demand_alpha_max)?;
    dict.set_item("retailer_demand_mean", reference.retailer_demand_mean)?;
    dict.set_item(
        "warehouse_holding_costs",
        reference.warehouse_holding_costs.to_vec(),
    )?;
    dict.set_item(
        "retailer_holding_costs",
        reference.retailer_holding_costs.to_vec(),
    )?;
    dict.set_item(
        "warehouse_backorder_costs",
        reference.warehouse_backorder_costs.to_vec(),
    )?;
    dict.set_item(
        "retailer_backorder_costs",
        reference.retailer_backorder_costs.to_vec(),
    )?;
    dict.set_item(
        "benchmark_base_stock_levels",
        reference.benchmark_base_stock_levels.to_vec(),
    )?;
    dict.set_item("benchmark_periods", reference.benchmark_periods)?;
    dict.set_item(
        "benchmark_warm_up_periods",
        reference.benchmark_warm_up_periods,
    )?;
    dict.set_item("benchmark_replications", reference.benchmark_replications)?;
    dict.set_item(
        "benchmark_order_routing_mode",
        reference.benchmark_order_routing_mode,
    )?;
    dict.set_item("paper_action_space", reference.paper_action_space)?;
    dict.set_item(
        "published_benchmark_cost",
        reference.published_benchmark_cost,
    )?;
    dict.set_item("published_ppo_best_cost", reference.published_ppo_best_cost)?;
    dict.set_item(
        "published_ppo_average_cost",
        reference.published_ppo_average_cost,
    )?;
    dict.set_item("notes", reference.notes)?;
    Ok(dict.into_any().unbind().into())
}

fn rollout_config(
    reference: &GeneralBackorderFixedCostReferenceInstance,
    input_dim: usize,
    depth: usize,
    action_spec: crate::core::policies::soft_tree::SoftTreeActionSpec,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    policy_feature_mode: &str,
    policy_action_mode: &str,
) -> PyResult<GeneralBackorderFixedCostRolloutConfig> {
    Ok(GeneralBackorderFixedCostRolloutConfig {
        input_dim,
        depth,
        action_spec,
        periods: reference.benchmark_periods,
        warm_up_periods: reference.benchmark_warm_up_periods,
        network: build_network(reference),
        retailer_demand_mean: reference.retailer_demand_mean,
        demand_mode: parse_demand_mode(reference.demand_mode)?,
        demand_alpha_min: reference.demand_alpha_min,
        demand_alpha_max: reference.demand_alpha_max,
        warehouse_holding_costs: reference.warehouse_holding_costs.to_vec(),
        retailer_holding_costs: reference.retailer_holding_costs.to_vec(),
        warehouse_backorder_costs: reference.warehouse_backorder_costs.to_vec(),
        retailer_backorder_costs: reference.retailer_backorder_costs.to_vec(),
        benchmark_order_routing_mode: parse_benchmark_order_routing_mode(
            reference.benchmark_order_routing_mode,
        )?,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        policy_feature_mode: parse_policy_feature_mode(policy_feature_mode)?,
        policy_action_mode: parse_policy_action_mode(policy_action_mode)?,
    })
}

#[pyfunction]
fn multi_echelon_general_backorder_fixed_cost_benchmark_reference(
    py: Python<'_>,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("source", GEEVERS_2023_REFERENCE.source)?;
    dict.set_item("url", GEEVERS_2023_REFERENCE.url)?;
    dict.set_item(
        "benchmark_policies",
        GEEVERS_2023_REFERENCE.benchmark_policies.to_vec(),
    )?;
    dict.set_item("notes", GEEVERS_2023_REFERENCE.notes)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn multi_echelon_general_backorder_fixed_cost_list_reference_instances(
    py: Python<'_>,
) -> PyResult<Vec<PyObject>> {
    LITERATURE_REFERENCE_INSTANCES
        .iter()
        .map(|reference| reference_to_py(py, reference))
        .collect()
}

#[pyfunction]
fn multi_echelon_general_backorder_fixed_cost_get_reference_instance(
    py: Python<'_>,
    name: &str,
) -> PyResult<PyObject> {
    let reference = reference_instance_by_name(name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown general_backorder_fixed_cost reference '{name}'"
        ))
    })?;
    reference_to_py(py, reference)
}

#[pyfunction]
fn multi_echelon_general_backorder_fixed_cost_primary_reference_instance(
    py: Python<'_>,
) -> PyResult<PyObject> {
    reference_to_py(py, PRIMARY_REFERENCE_INSTANCE)
}

#[pyfunction]
#[pyo3(signature = (reference_name, base_stock_levels=None, replications=None, seed=1234, routing_mode=None))]
fn multi_echelon_general_backorder_fixed_cost_simulate_base_stock(
    py: Python<'_>,
    reference_name: &str,
    base_stock_levels: Option<Vec<usize>>,
    replications: Option<usize>,
    seed: u64,
    routing_mode: Option<&str>,
) -> PyResult<PyObject> {
    let reference = reference_instance_by_name(reference_name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown general_backorder_fixed_cost reference '{reference_name}'"
        ))
    })?;
    let levels =
        base_stock_levels.unwrap_or_else(|| reference.benchmark_base_stock_levels.to_vec());
    let replications = replications.unwrap_or(reference.benchmark_replications);
    let routing_mode = parse_benchmark_order_routing_mode(
        routing_mode.unwrap_or(reference.benchmark_order_routing_mode),
    )?;
    let costs = simulate_node_base_stock_policy_with_mode(
        reference,
        &levels,
        replications,
        seed,
        routing_mode,
    )?;
    let mean_cost = costs.iter().sum::<f64>() / costs.len() as f64;
    let variance = if costs.len() <= 1 {
        0.0
    } else {
        costs
            .iter()
            .map(|cost| {
                let delta = *cost - mean_cost;
                delta * delta
            })
            .sum::<f64>()
            / costs.len() as f64
    };
    let dict = PyDict::new_bound(py);
    dict.set_item("reference_name", reference.name)?;
    dict.set_item("base_stock_levels", levels)?;
    dict.set_item("replications", replications)?;
    dict.set_item("mean_cost", mean_cost)?;
    dict.set_item("std_cost", variance.sqrt())?;
    dict.set_item(
        "min_cost",
        costs.iter().cloned().fold(f64::INFINITY, f64::min),
    )?;
    dict.set_item(
        "max_cost",
        costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    )?;
    dict.set_item(
        "published_benchmark_cost",
        reference.published_benchmark_cost,
    )?;
    dict.set_item("published_ppo_best_cost", reference.published_ppo_best_cost)?;
    dict.set_item(
        "published_ppo_average_cost",
        reference.published_ppo_average_cost,
    )?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
#[pyo3(signature = (reference_name, base_stock_levels=None, replications=None, seed=1234, routing_mode=None))]
fn multi_echelon_general_backorder_fixed_cost_audit_base_stock(
    py: Python<'_>,
    reference_name: &str,
    base_stock_levels: Option<Vec<usize>>,
    replications: Option<usize>,
    seed: u64,
    routing_mode: Option<&str>,
) -> PyResult<PyObject> {
    let reference = reference_instance_by_name(reference_name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown general_backorder_fixed_cost reference '{reference_name}'"
        ))
    })?;
    let levels =
        base_stock_levels.unwrap_or_else(|| reference.benchmark_base_stock_levels.to_vec());
    let replications = replications.unwrap_or(reference.benchmark_replications);
    let routing_mode = parse_benchmark_order_routing_mode(
        routing_mode.unwrap_or(reference.benchmark_order_routing_mode),
    )?;
    let audit = simulate_node_base_stock_policy_audit_with_mode(
        reference,
        &levels,
        replications,
        seed,
        routing_mode,
    )?;
    let mean = |values: &[f64]| -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    };
    let std = |values: &[f64], mean_value: f64| -> f64 {
        if values.len() <= 1 {
            0.0
        } else {
            (values
                .iter()
                .map(|value| {
                    let delta = *value - mean_value;
                    delta * delta
                })
                .sum::<f64>()
                / values.len() as f64)
                .sqrt()
        }
    };
    let mean_cost = mean(&audit.total_costs);
    let mean_holding_cost = mean(&audit.holding_costs);
    let mean_warehouse_backorder_cost = mean(&audit.warehouse_backorder_costs);
    let mean_customer_backorder_cost = mean(&audit.customer_backorder_costs);
    let edge_fill_rates = audit
        .edge_demand_totals
        .iter()
        .zip(audit.edge_fulfilled_totals.iter())
        .map(|(demand, fulfilled)| {
            if *demand == 0 {
                1.0
            } else {
                *fulfilled as f64 / *demand as f64
            }
        })
        .collect::<Vec<_>>();
    let customer_fill_rates = audit
        .customer_demand_totals
        .iter()
        .zip(audit.customer_fulfilled_totals.iter())
        .map(|(demand, fulfilled)| {
            if *demand == 0 {
                1.0
            } else {
                *fulfilled as f64 / *demand as f64
            }
        })
        .collect::<Vec<_>>();
    let dict = PyDict::new_bound(py);
    dict.set_item("reference_name", reference.name)?;
    dict.set_item("base_stock_levels", levels)?;
    dict.set_item("replications", replications)?;
    dict.set_item("mean_cost", mean_cost)?;
    dict.set_item("std_cost", std(&audit.total_costs, mean_cost))?;
    dict.set_item("mean_holding_cost", mean_holding_cost)?;
    dict.set_item(
        "mean_warehouse_backorder_cost",
        mean_warehouse_backorder_cost,
    )?;
    dict.set_item("mean_customer_backorder_cost", mean_customer_backorder_cost)?;
    dict.set_item("edge_demand_totals", audit.edge_demand_totals)?;
    dict.set_item("edge_fulfilled_totals", audit.edge_fulfilled_totals)?;
    dict.set_item("edge_fill_rates", edge_fill_rates)?;
    dict.set_item("customer_demand_totals", audit.customer_demand_totals)?;
    dict.set_item("customer_fulfilled_totals", audit.customer_fulfilled_totals)?;
    dict.set_item("customer_fill_rates", customer_fill_rates)?;
    dict.set_item(
        "published_benchmark_cost",
        reference.published_benchmark_cost,
    )?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn multi_echelon_general_backorder_fixed_cost_build_raw_state(
    py: Python<'_>,
    reference_name: &str,
) -> PyResult<PyObject> {
    let reference = reference_instance_by_name(reference_name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown general_backorder_fixed_cost reference '{reference_name}'"
        ))
    })?;
    let network = build_network(reference);
    let state = initialize_zero_state(&network)?;
    let dict = PyDict::new_bound(py);
    dict.set_item("reference_name", reference.name)?;
    dict.set_item("raw_state", build_raw_state(&network, &state)?)?;
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
    reference_name,
    seed=1234,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None,
    policy_feature_mode="compact_summary",
    policy_action_mode="node_base_stock_targets"
))]
fn multi_echelon_general_backorder_fixed_cost_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    reference_name: &str,
    seed: u64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
    policy_feature_mode: &str,
    policy_action_mode: &str,
) -> PyResult<f64> {
    let reference = reference_instance_by_name(reference_name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown general_backorder_fixed_cost reference '{reference_name}'"
        ))
    })?;
    let action_spec = build_action_spec(action_mode, min_values, max_values, allowed_values)?;
    let config = rollout_config(
        reference,
        input_dim,
        depth,
        action_spec,
        temperature,
        split_type,
        leaf_type,
        policy_feature_mode,
        policy_action_mode,
    )?;
    let initial_state = initialize_zero_state(&config.network)?;
    rollout(&flat_params, &config, &initial_state, seed)
}

#[pyfunction]
#[pyo3(signature = (
    population,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    reference_name,
    seed=1234,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    allowed_values=None,
    policy_feature_mode="compact_summary",
    policy_action_mode="node_base_stock_targets"
))]
fn multi_echelon_general_backorder_fixed_cost_soft_tree_population_rollout(
    population: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    reference_name: &str,
    seed: u64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
    policy_feature_mode: &str,
    policy_action_mode: &str,
) -> PyResult<Vec<f64>> {
    let reference = reference_instance_by_name(reference_name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown general_backorder_fixed_cost reference '{reference_name}'"
        ))
    })?;
    let action_spec = build_action_spec(action_mode, min_values, max_values, allowed_values)?;
    let config = rollout_config(
        reference,
        input_dim,
        depth,
        action_spec,
        temperature,
        split_type,
        leaf_type,
        policy_feature_mode,
        policy_action_mode,
    )?;
    let initial_state = initialize_zero_state(&config.network)?;
    population_rollout(&population, &config, &initial_state, seed)
}

#[pyfunction]
#[pyo3(signature = (
    warehouse_inventory,
    retailer_inventory,
    supplier_orders_due,
    retailer_orders_due,
    supplier_deliveries_due,
    retailer_deliveries_due,
    supplier_in_transit,
    retailer_in_transit,
    retailer_backorders,
    customer_backorders,
    reference_name
))]
fn multi_echelon_general_backorder_fixed_cost_custom_raw_state(
    py: Python<'_>,
    warehouse_inventory: Vec<usize>,
    retailer_inventory: Vec<usize>,
    supplier_orders_due: Vec<usize>,
    retailer_orders_due: Vec<usize>,
    supplier_deliveries_due: Vec<usize>,
    retailer_deliveries_due: Vec<usize>,
    supplier_in_transit: Vec<usize>,
    retailer_in_transit: Vec<usize>,
    retailer_backorders: Vec<usize>,
    customer_backorders: Vec<usize>,
    reference_name: &str,
) -> PyResult<PyObject> {
    let reference = reference_instance_by_name(reference_name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown general_backorder_fixed_cost reference '{reference_name}'"
        ))
    })?;
    let network = build_network(reference);
    let state = build_initial_state(
        &network,
        &warehouse_inventory,
        &retailer_inventory,
        &supplier_orders_due,
        &retailer_orders_due,
        &supplier_deliveries_due,
        &retailer_deliveries_due,
        &supplier_in_transit,
        &retailer_in_transit,
        &retailer_backorders,
        &customer_backorders,
    )?;
    let dict = PyDict::new_bound(py);
    dict.set_item("reference_name", reference.name)?;
    dict.set_item("raw_state", build_raw_state(&network, &state)?)?;
    Ok(dict.into_any().unbind().into())
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        multi_echelon_general_backorder_fixed_cost_benchmark_reference,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_general_backorder_fixed_cost_list_reference_instances,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_general_backorder_fixed_cost_get_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_general_backorder_fixed_cost_primary_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_general_backorder_fixed_cost_simulate_base_stock,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_general_backorder_fixed_cost_audit_base_stock,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_general_backorder_fixed_cost_build_raw_state,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_general_backorder_fixed_cost_custom_raw_state,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_general_backorder_fixed_cost_soft_tree_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_general_backorder_fixed_cost_soft_tree_population_rollout,
        m
    )?)?;
    Ok(())
}
