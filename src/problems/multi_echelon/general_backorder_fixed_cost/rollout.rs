use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Poisson;
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::multi_echelon::general_backorder_fixed_cost::env::{
    advance_to_decision_state, apply_next_orders, build_raw_state,
    retailer_total_inventory_positions, validate_network, validate_state,
    warehouse_inventory_positions, GeneralBackorderFixedCostNetwork,
    GeneralBackorderFixedCostState,
};
use crate::problems::multi_echelon::general_backorder_fixed_cost::heuristics::{
    node_base_stock_orders, sample_period_demands, BenchmarkOrderRoutingMode,
};
use crate::problems::multi_echelon::general_backorder_fixed_cost::references::DemandMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyFeatureMode {
    RawState,
    CompactSummary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyActionMode {
    NodeBaseStockTargets,
}

#[derive(Clone)]
pub struct GeneralBackorderFixedCostRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub warm_up_periods: usize,
    pub network: GeneralBackorderFixedCostNetwork,
    pub retailer_demand_mean: f64,
    pub demand_mode: DemandMode,
    pub demand_alpha_min: f64,
    pub demand_alpha_max: f64,
    pub warehouse_holding_costs: Vec<f64>,
    pub retailer_holding_costs: Vec<f64>,
    pub warehouse_backorder_costs: Vec<f64>,
    pub retailer_backorder_costs: Vec<f64>,
    pub benchmark_order_routing_mode: BenchmarkOrderRoutingMode,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
    pub policy_feature_mode: PolicyFeatureMode,
    pub policy_action_mode: PolicyActionMode,
}

pub fn parse_policy_feature_mode(mode: &str) -> PyResult<PolicyFeatureMode> {
    match mode {
        "raw_state" => Ok(PolicyFeatureMode::RawState),
        "compact_summary" => Ok(PolicyFeatureMode::CompactSummary),
        _ => Err(PyValueError::new_err(format!(
            "unknown policy_feature_mode '{mode}'; expected 'raw_state' or 'compact_summary'"
        ))),
    }
}

pub fn parse_policy_action_mode(mode: &str) -> PyResult<PolicyActionMode> {
    match mode {
        "node_base_stock_targets" => Ok(PolicyActionMode::NodeBaseStockTargets),
        _ => Err(PyValueError::new_err(format!(
            "unknown policy_action_mode '{mode}'; expected 'node_base_stock_targets'"
        ))),
    }
}

fn validate_config(
    config: &GeneralBackorderFixedCostRolloutConfig,
    initial_state: &GeneralBackorderFixedCostState,
) -> PyResult<()> {
    validate_network(&config.network)?;
    validate_state(&config.network, initial_state)?;
    if !config.retailer_demand_mean.is_finite() || config.retailer_demand_mean < 0.0 {
        return Err(PyValueError::new_err(
            "retailer_demand_mean must be finite and non-negative",
        ));
    }
    if config.warm_up_periods > config.periods {
        return Err(PyValueError::new_err(
            "warm_up_periods must not exceed periods",
        ));
    }
    if config.warehouse_holding_costs.len() != config.network.num_warehouses
        || config.retailer_holding_costs.len() != config.network.num_retailers
        || config.warehouse_backorder_costs.len() != config.network.num_warehouses
        || config.retailer_backorder_costs.len() != config.network.num_retailers
    {
        return Err(PyValueError::new_err(
            "all cost vectors must match the network dimensions",
        ));
    }
    let expected_input_dim = match config.policy_feature_mode {
        PolicyFeatureMode::RawState => {
            config.network.num_warehouses * 4
                + config.network.num_retailers * 3
                + config.network.retail_edges.len() * 4
                + 1
        }
        PolicyFeatureMode::CompactSummary => {
            config.network.num_warehouses + config.network.num_retailers + 5
        }
    };
    if config.input_dim != expected_input_dim {
        return Err(PyValueError::new_err(format!(
            "input_dim {} does not match expected {} for feature mode",
            config.input_dim, expected_input_dim
        )));
    }
    let expected_action_dim = config.network.num_warehouses + config.network.num_retailers;
    if config.action_spec.action_dim != expected_action_dim {
        return Err(PyValueError::new_err(format!(
            "action_spec.action_dim {} does not match expected {}",
            config.action_spec.action_dim, expected_action_dim
        )));
    }
    Ok(())
}

pub fn build_policy_features(
    state: &GeneralBackorderFixedCostState,
    config: &GeneralBackorderFixedCostRolloutConfig,
) -> PyResult<Vec<f32>> {
    let mut features = match config.policy_feature_mode {
        PolicyFeatureMode::RawState => build_raw_state(&config.network, state)?,
        PolicyFeatureMode::CompactSummary => {
            let warehouse_positions = warehouse_inventory_positions(&config.network, state)?;
            let retailer_positions = retailer_total_inventory_positions(&config.network, state)?;
            let total_inventory = state
                .warehouse_inventory
                .iter()
                .chain(state.retailer_inventory.iter())
                .copied()
                .sum::<usize>() as f32;
            let total_backorders = state
                .retailer_backorders
                .iter()
                .chain(state.customer_backorders.iter())
                .copied()
                .sum::<usize>() as f32;
            let total_in_transit = state
                .supplier_in_transit
                .iter()
                .chain(state.retailer_in_transit.iter())
                .copied()
                .sum::<usize>() as f32;
            let max_abs_position = warehouse_positions
                .iter()
                .chain(retailer_positions.iter())
                .map(|value| value.unsigned_abs() as f32)
                .fold(1.0f32, f32::max);
            let scale = max_abs_position
                .max(total_inventory.max(total_backorders).max(total_in_transit))
                .max(config.retailer_demand_mean as f32);
            let mut summary = Vec::with_capacity(
                config.network.num_warehouses + config.network.num_retailers + 4,
            );
            summary.extend(
                warehouse_positions
                    .iter()
                    .map(|value| *value as f32 / scale),
            );
            summary.extend(retailer_positions.iter().map(|value| *value as f32 / scale));
            summary.push(total_inventory / scale);
            summary.push(total_backorders / scale);
            summary.push(total_in_transit / scale);
            summary.push(config.retailer_demand_mean as f32 / scale);
            summary
        }
    };
    let remaining_fraction = if config.periods == 0 {
        0.0
    } else {
        (config.periods.saturating_sub(state.period) as f32) / config.periods as f32
    };
    features.push(remaining_fraction);
    Ok(features)
}

fn decode_policy_action(
    flat_params: &[f32],
    state: &GeneralBackorderFixedCostState,
    config: &GeneralBackorderFixedCostRolloutConfig,
    rng: &mut StdRng,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let features = build_policy_features(state, config)?;
    let action = action_vector_from_flat_params(
        &features,
        flat_params,
        config.input_dim,
        config.depth,
        config.temperature,
        config.split_type,
        config.leaf_type,
        &config.action_spec,
    )?;
    match config.policy_action_mode {
        PolicyActionMode::NodeBaseStockTargets => node_base_stock_orders(
            &config.network,
            state,
            &action,
            config.benchmark_order_routing_mode,
            rng,
        ),
    }
}

pub fn rollout(
    flat_params: &[f32],
    config: &GeneralBackorderFixedCostRolloutConfig,
    initial_state: &GeneralBackorderFixedCostState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    let mut state = initial_state.clone();
    let mut rng = StdRng::seed_from_u64(seed);
    let demand_distribution = Poisson::new(config.retailer_demand_mean).map_err(|err| {
        PyValueError::new_err(format!(
            "invalid Poisson mean {}: {err}",
            config.retailer_demand_mean
        ))
    })?;
    let mut total_cost = 0.0;
    for period_idx in 0..config.periods {
        let realized_demands = sample_period_demands(
            &mut rng,
            config.network.num_retailers,
            config.demand_mode,
            &demand_distribution,
            config.demand_alpha_min,
            config.demand_alpha_max,
        )?;
        let decision = advance_to_decision_state(
            &config.network,
            &state,
            &realized_demands,
            &config.warehouse_holding_costs,
            &config.retailer_holding_costs,
            &config.warehouse_backorder_costs,
            &config.retailer_backorder_costs,
        )?;
        if period_idx >= config.warm_up_periods {
            total_cost += decision.period_cost;
        }
        let (warehouse_orders, retailer_orders) =
            decode_policy_action(flat_params, &decision.decision_state, config, &mut rng)?;
        state = apply_next_orders(
            &config.network,
            &decision.decision_state,
            &warehouse_orders,
            &retailer_orders,
        )?;
    }
    Ok(total_cost)
}

pub fn population_rollout(
    population: &[Vec<f32>],
    config: &GeneralBackorderFixedCostRolloutConfig,
    initial_state: &GeneralBackorderFixedCostState,
    seed: u64,
) -> PyResult<Vec<f64>> {
    validate_config(config, initial_state)?;
    let scores = population
        .par_iter()
        .enumerate()
        .map(|(idx, flat_params)| rollout(flat_params, config, initial_state, seed + idx as u64))
        .collect::<Vec<_>>();
    scores.into_iter().collect()
}

pub fn build_initial_state(
    network: &GeneralBackorderFixedCostNetwork,
    warehouse_inventory: &[usize],
    retailer_inventory: &[usize],
    supplier_orders_due: &[usize],
    retailer_orders_due: &[usize],
    supplier_deliveries_due: &[usize],
    retailer_deliveries_due: &[usize],
    supplier_in_transit: &[usize],
    retailer_in_transit: &[usize],
    retailer_backorders: &[usize],
    customer_backorders: &[usize],
) -> PyResult<GeneralBackorderFixedCostState> {
    let state = GeneralBackorderFixedCostState {
        period: 0,
        warehouse_inventory: warehouse_inventory.to_vec(),
        retailer_inventory: retailer_inventory.to_vec(),
        supplier_orders_due: supplier_orders_due.to_vec(),
        retailer_orders_due: retailer_orders_due.to_vec(),
        supplier_deliveries_due: supplier_deliveries_due.to_vec(),
        retailer_deliveries_due: retailer_deliveries_due.to_vec(),
        supplier_in_transit: supplier_in_transit.to_vec(),
        retailer_in_transit: retailer_in_transit.to_vec(),
        retailer_backorders: retailer_backorders.to_vec(),
        customer_backorders: customer_backorders.to_vec(),
    };
    validate_state(network, &state)?;
    Ok(state)
}
