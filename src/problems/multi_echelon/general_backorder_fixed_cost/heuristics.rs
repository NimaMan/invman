#![allow(dead_code)]

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Poisson};

use crate::problems::multi_echelon::general_backorder_fixed_cost::env::{
    advance_to_decision_state, apply_next_orders, incoming_retail_edge_indices,
    initialize_zero_state, retailer_selected_edge_inventory_position,
    retailer_total_inventory_positions, validate_network, warehouse_inventory_positions,
    GeneralBackorderFixedCostNetwork, GeneralBackorderFixedCostState,
};
use crate::problems::multi_echelon::general_backorder_fixed_cost::references::{
    parse_demand_mode, DemandMode, GeneralBackorderFixedCostReferenceInstance,
};

/// Sample one period's realized customer demand for every retailer.
///
/// `FixedPoisson`: each retailer draws `Poisson(fixed_mean)` -- the original set-1/2/3 behaviour.
/// `ResampledUniformPoisson`: each retailer first draws a fresh mean
/// `alpha ~ Uniform[alpha_min, alpha_max]` THIS period, then draws `Poisson(alpha)`
/// (nonstationary, per-retailer mean). The two paths consume the rng differently, so this is the
/// single place that branches on the demand mode; the fixed path must stay identical to the
/// pre-change code so set 1 keeps reproducing ~10,355.
pub fn sample_period_demands(
    rng: &mut StdRng,
    num_retailers: usize,
    mode: DemandMode,
    fixed_distribution: &Poisson<f64>,
    alpha_min: f64,
    alpha_max: f64,
) -> PyResult<Vec<usize>> {
    match mode {
        DemandMode::FixedPoisson => Ok((0..num_retailers)
            .map(|_| fixed_distribution.sample(rng) as usize)
            .collect()),
        DemandMode::ResampledUniformPoisson => (0..num_retailers)
            .map(|_| {
                let alpha = rng.gen_range(alpha_min..=alpha_max);
                let per_period = Poisson::new(alpha).map_err(|err| {
                    PyValueError::new_err(format!("invalid resampled Poisson mean {alpha}: {err}"))
                })?;
                Ok(per_period.sample(rng) as usize)
            })
            .collect(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkOrderRoutingMode {
    RandomSingleConnectionByWeight,
    SplitAcrossAllConnectionsByWeight,
    SplitAcrossAllConnectionsEvenly,
    DuplicateTargetAllConnections,
    WeightedTargetAllConnections,
}

pub fn parse_benchmark_order_routing_mode(mode: &str) -> PyResult<BenchmarkOrderRoutingMode> {
    match mode {
        "random_single_connection_by_weight" => {
            Ok(BenchmarkOrderRoutingMode::RandomSingleConnectionByWeight)
        }
        "split_across_all_connections_by_weight" => {
            Ok(BenchmarkOrderRoutingMode::SplitAcrossAllConnectionsByWeight)
        }
        "split_across_all_connections_evenly" => {
            Ok(BenchmarkOrderRoutingMode::SplitAcrossAllConnectionsEvenly)
        }
        "duplicate_target_all_connections" => {
            Ok(BenchmarkOrderRoutingMode::DuplicateTargetAllConnections)
        }
        "weighted_target_all_connections" => {
            Ok(BenchmarkOrderRoutingMode::WeightedTargetAllConnections)
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown benchmark_order_routing_mode '{mode}'"
        ))),
    }
}

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

fn split_by_weights(total_quantity: usize, weights: &[f64]) -> Vec<usize> {
    if total_quantity == 0 || weights.is_empty() {
        return vec![0usize; weights.len()];
    }
    let mut raw = weights
        .iter()
        .map(|weight| total_quantity as f64 * *weight)
        .collect::<Vec<_>>();
    let mut allocation = raw
        .iter()
        .map(|value| value.floor() as usize)
        .collect::<Vec<_>>();
    let assigned = allocation.iter().sum::<usize>();
    let remainder = total_quantity - assigned;
    if remainder == 0 {
        return allocation;
    }
    let mut order = (0..weights.len())
        .map(|idx| (idx, raw[idx] - allocation[idx] as f64))
        .collect::<Vec<_>>();
    order.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap().then(lhs.0.cmp(&rhs.0)));
    for (idx, _) in order.into_iter().take(remainder) {
        allocation[idx] += 1;
        raw[idx] = 0.0;
    }
    allocation
}

fn sample_single_edge_by_weight(
    rng: &mut StdRng,
    edge_indices: &[usize],
    weights: &[f64],
) -> usize {
    let draw = rng.gen::<f64>();
    let mut cumulative = 0.0;
    for (offset, weight) in weights.iter().enumerate() {
        cumulative += *weight;
        if draw <= cumulative || offset + 1 == weights.len() {
            return edge_indices[offset];
        }
    }
    edge_indices[edge_indices.len() - 1]
}

pub struct SimulationAuditSummary {
    pub total_costs: Vec<f64>,
    pub holding_costs: Vec<f64>,
    pub warehouse_backorder_costs: Vec<f64>,
    pub customer_backorder_costs: Vec<f64>,
    pub edge_demand_totals: Vec<usize>,
    pub edge_fulfilled_totals: Vec<usize>,
    pub customer_demand_totals: Vec<usize>,
    pub customer_fulfilled_totals: Vec<usize>,
}

pub fn node_base_stock_orders(
    network: &GeneralBackorderFixedCostNetwork,
    state: &GeneralBackorderFixedCostState,
    base_stock_levels: &[usize],
    routing_mode: BenchmarkOrderRoutingMode,
    rng: &mut StdRng,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    validate_network(network)?;
    if base_stock_levels.len() != network.num_warehouses + network.num_retailers {
        return Err(PyValueError::new_err(
            "base_stock_levels must contain one level for each warehouse and retailer",
        ));
    }

    let warehouse_positions = warehouse_inventory_positions(network, state)?;
    let retailer_total_positions = retailer_total_inventory_positions(network, state)?;

    let mut warehouse_orders = vec![0usize; network.num_warehouses];
    for warehouse_idx in 0..network.num_warehouses {
        let current_level = warehouse_positions[warehouse_idx].max(0) as usize;
        warehouse_orders[warehouse_idx] =
            base_stock_levels[warehouse_idx].saturating_sub(current_level);
    }

    let mut retailer_orders_by_edge = vec![0usize; network.retail_edges.len()];
    for retailer_idx in 0..network.num_retailers {
        let target_level = base_stock_levels[network.num_warehouses + retailer_idx];
        let incoming_edges = incoming_retail_edge_indices(network, retailer_idx);
        let incoming_weights = incoming_edges
            .iter()
            .map(|edge_idx| network.retail_edges[*edge_idx].connection_weight)
            .collect::<Vec<_>>();
        match routing_mode {
            BenchmarkOrderRoutingMode::RandomSingleConnectionByWeight => {
                let selected_edge =
                    sample_single_edge_by_weight(rng, &incoming_edges, &incoming_weights);
                let current_level = retailer_selected_edge_inventory_position(
                    network,
                    state,
                    selected_edge,
                )?
                .max(0) as usize;
                retailer_orders_by_edge[selected_edge] = target_level.saturating_sub(current_level);
            }
            BenchmarkOrderRoutingMode::SplitAcrossAllConnectionsByWeight => {
                let current_level = retailer_total_positions[retailer_idx].max(0) as usize;
                let local_gap = target_level.saturating_sub(current_level);
                let split = split_by_weights(local_gap, &incoming_weights);
                for (offset, edge_idx) in incoming_edges.iter().enumerate() {
                    retailer_orders_by_edge[*edge_idx] = split[offset];
                }
            }
            BenchmarkOrderRoutingMode::SplitAcrossAllConnectionsEvenly => {
                let current_level = retailer_total_positions[retailer_idx].max(0) as usize;
                let local_gap = target_level.saturating_sub(current_level);
                let equal_weights = vec![1.0 / incoming_edges.len() as f64; incoming_edges.len()];
                let split = split_by_weights(local_gap, &equal_weights);
                for (offset, edge_idx) in incoming_edges.iter().enumerate() {
                    retailer_orders_by_edge[*edge_idx] = split[offset];
                }
            }
            BenchmarkOrderRoutingMode::DuplicateTargetAllConnections => {
                for edge_idx in incoming_edges {
                    let current_level =
                        retailer_selected_edge_inventory_position(network, state, edge_idx)?.max(0)
                            as usize;
                    retailer_orders_by_edge[edge_idx] = target_level.saturating_sub(current_level);
                }
            }
            BenchmarkOrderRoutingMode::WeightedTargetAllConnections => {
                let edge_targets = split_by_weights(target_level, &incoming_weights);
                for (offset, edge_idx) in incoming_edges.iter().enumerate() {
                    let current_level = retailer_selected_edge_inventory_position(
                        network, state, *edge_idx,
                    )?
                    .max(0) as usize;
                    retailer_orders_by_edge[*edge_idx] =
                        edge_targets[offset].saturating_sub(current_level);
                }
            }
        }
    }

    Ok((warehouse_orders, retailer_orders_by_edge))
}

pub fn zero_state_from_reference(
    reference: &GeneralBackorderFixedCostReferenceInstance,
) -> PyResult<GeneralBackorderFixedCostState> {
    initialize_zero_state(&build_network(reference))
}

pub fn simulate_node_base_stock_policy(
    reference: &GeneralBackorderFixedCostReferenceInstance,
    base_stock_levels: &[usize],
    replications: usize,
    seed: u64,
) -> PyResult<Vec<f64>> {
    simulate_node_base_stock_policy_with_mode(
        reference,
        base_stock_levels,
        replications,
        seed,
        parse_benchmark_order_routing_mode(reference.benchmark_order_routing_mode)?,
    )
}

pub fn simulate_node_base_stock_policy_with_mode(
    reference: &GeneralBackorderFixedCostReferenceInstance,
    base_stock_levels: &[usize],
    replications: usize,
    seed: u64,
    routing_mode: BenchmarkOrderRoutingMode,
) -> PyResult<Vec<f64>> {
    let network = build_network(reference);
    validate_network(&network)?;
    let demand_mode = parse_demand_mode(reference.demand_mode)?;
    let demand_distribution = Poisson::new(reference.retailer_demand_mean).map_err(|err| {
        PyValueError::new_err(format!(
            "invalid Poisson mean {}: {err}",
            reference.retailer_demand_mean
        ))
    })?;
    let mut totals = Vec::with_capacity(replications);
    for replication_idx in 0..replications {
        let mut rng = StdRng::seed_from_u64(seed + replication_idx as u64);
        let mut state = initialize_zero_state(&network)?;
        let mut total_cost = 0.0;
        for period_idx in 0..reference.benchmark_periods {
            let realized_demands = sample_period_demands(
                &mut rng,
                reference.num_retailers,
                demand_mode,
                &demand_distribution,
                reference.demand_alpha_min,
                reference.demand_alpha_max,
            )?;
            let decision = advance_to_decision_state(
                &network,
                &state,
                &realized_demands,
                reference.warehouse_holding_costs,
                reference.retailer_holding_costs,
                reference.warehouse_backorder_costs,
                reference.retailer_backorder_costs,
            )?;
            if period_idx >= reference.benchmark_warm_up_periods {
                total_cost += decision.period_cost;
            }
            let (warehouse_orders, retailer_orders) = node_base_stock_orders(
                &network,
                &decision.decision_state,
                base_stock_levels,
                routing_mode,
                &mut rng,
            )?;
            state = apply_next_orders(
                &network,
                &decision.decision_state,
                &warehouse_orders,
                &retailer_orders,
            )?;
        }
        totals.push(total_cost);
    }
    Ok(totals)
}

pub fn simulate_node_base_stock_policy_audit_with_mode(
    reference: &GeneralBackorderFixedCostReferenceInstance,
    base_stock_levels: &[usize],
    replications: usize,
    seed: u64,
    routing_mode: BenchmarkOrderRoutingMode,
) -> PyResult<SimulationAuditSummary> {
    let network = build_network(reference);
    validate_network(&network)?;
    let demand_mode = parse_demand_mode(reference.demand_mode)?;
    let demand_distribution = Poisson::new(reference.retailer_demand_mean).map_err(|err| {
        PyValueError::new_err(format!(
            "invalid Poisson mean {}: {err}",
            reference.retailer_demand_mean
        ))
    })?;
    let mut total_costs = Vec::with_capacity(replications);
    let mut holding_costs = Vec::with_capacity(replications);
    let mut warehouse_backorder_costs = Vec::with_capacity(replications);
    let mut customer_backorder_costs = Vec::with_capacity(replications);
    let mut edge_demand_totals = vec![0usize; network.retail_edges.len()];
    let mut edge_fulfilled_totals = vec![0usize; network.retail_edges.len()];
    let mut customer_demand_totals = vec![0usize; network.num_retailers];
    let mut customer_fulfilled_totals = vec![0usize; network.num_retailers];
    for replication_idx in 0..replications {
        let mut rng = StdRng::seed_from_u64(seed + replication_idx as u64);
        let mut state = initialize_zero_state(&network)?;
        let mut total_cost = 0.0;
        let mut total_holding_cost = 0.0;
        let mut total_warehouse_backorder_cost = 0.0;
        let mut total_customer_backorder_cost = 0.0;
        for period_idx in 0..reference.benchmark_periods {
            let realized_demands = sample_period_demands(
                &mut rng,
                reference.num_retailers,
                demand_mode,
                &demand_distribution,
                reference.demand_alpha_min,
                reference.demand_alpha_max,
            )?;
            let decision = advance_to_decision_state(
                &network,
                &state,
                &realized_demands,
                reference.warehouse_holding_costs,
                reference.retailer_holding_costs,
                reference.warehouse_backorder_costs,
                reference.retailer_backorder_costs,
            )?;
            if period_idx >= reference.benchmark_warm_up_periods {
                total_cost += decision.period_cost;
                total_holding_cost += decision.holding_cost;
                total_warehouse_backorder_cost += decision.warehouse_backorder_cost;
                total_customer_backorder_cost += decision.customer_backorder_cost;
                for retailer_idx in 0..network.num_retailers {
                    customer_demand_totals[retailer_idx] += decision.realized_demands[retailer_idx];
                    customer_fulfilled_totals[retailer_idx] +=
                        decision.fulfilled_customer_demands[retailer_idx];
                }
                for edge_idx in 0..network.retail_edges.len() {
                    edge_fulfilled_totals[edge_idx] +=
                        decision.fulfilled_current_retail_orders[edge_idx];
                }
            }
            let (warehouse_orders, retailer_orders) = node_base_stock_orders(
                &network,
                &decision.decision_state,
                base_stock_levels,
                routing_mode,
                &mut rng,
            )?;
            if period_idx + 1 < reference.benchmark_periods
                && period_idx + 1 >= reference.benchmark_warm_up_periods
            {
                for edge_idx in 0..network.retail_edges.len() {
                    edge_demand_totals[edge_idx] += retailer_orders[edge_idx];
                }
            }
            state = apply_next_orders(
                &network,
                &decision.decision_state,
                &warehouse_orders,
                &retailer_orders,
            )?;
        }
        total_costs.push(total_cost);
        holding_costs.push(total_holding_cost);
        warehouse_backorder_costs.push(total_warehouse_backorder_cost);
        customer_backorder_costs.push(total_customer_backorder_cost);
    }
    Ok(SimulationAuditSummary {
        total_costs,
        holding_costs,
        warehouse_backorder_costs,
        customer_backorder_costs,
        edge_demand_totals,
        edge_fulfilled_totals,
        customer_demand_totals,
        customer_fulfilled_totals,
    })
}
