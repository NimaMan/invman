#![allow(dead_code)]

mod base_stock;

pub use base_stock::node_base_stock_requests;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::problems::network_inventory::demand::{sample_demand, DemandModel};
use crate::problems::network_inventory::env::{
    step_state, validate_state, NetworkInventoryGraph, NetworkInventoryState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicySimulationSummary {
    pub mean_cost: f64,
    pub cost_std: f64,
}

fn policy_requests(
    policy_name: &str,
    params: &[f64],
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
) -> PyResult<Vec<usize>> {
    match policy_name {
        "node_base_stock" => {
            if params.len() != graph.num_nodes {
                return Err(PyValueError::new_err(format!(
                    "node_base_stock expects {} parameters",
                    graph.num_nodes
                )));
            }
            let levels = params
                .iter()
                .map(|value| value.round().max(0.0) as usize)
                .collect::<Vec<_>>();
            node_base_stock_requests(graph, state, &levels)
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported policy '{policy_name}'"
        ))),
    }
}

pub fn policy_rollout_from_paths(
    policy_name: &str,
    params: &[f64],
    graph: &NetworkInventoryGraph,
    initial_state: &NetworkInventoryState,
    realized_demands: &[Vec<usize>],
    holding_costs: &[f64],
    backlog_costs: &[f64],
    discount_factor: f64,
) -> PyResult<f64> {
    validate_state(graph, initial_state)?;
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }

    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;

    for demand in realized_demands.iter() {
        if demand.len() != graph.num_nodes {
            return Err(PyValueError::new_err(
                "each realized demand vector must match num_nodes",
            ));
        }
        let edge_requests = policy_requests(policy_name, params, graph, &state)?;
        let outcome = step_state(
            graph,
            &state,
            &edge_requests,
            demand,
            holding_costs,
            backlog_costs,
        )?;
        discounted_cost += discount * outcome.period_cost;
        discount *= discount_factor;
        state = outcome.next_state;
    }

    Ok(discounted_cost)
}

pub fn simulate_policy(
    policy_name: &str,
    params: &[f64],
    graph: &NetworkInventoryGraph,
    initial_state: &NetworkInventoryState,
    periods: usize,
    replications: usize,
    seed: u64,
    demand_models: &[DemandModel],
    holding_costs: &[f64],
    backlog_costs: &[f64],
    discount_factor: f64,
) -> PyResult<PolicySimulationSummary> {
    validate_state(graph, initial_state)?;
    if demand_models.len() != graph.num_nodes {
        return Err(PyValueError::new_err(
            "demand_models length must match num_nodes",
        ));
    }
    if periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if replications == 0 {
        return Err(PyValueError::new_err("replications must be at least 1"));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut discounted_costs = Vec::with_capacity(replications);
    for _ in 0..replications {
        let mut state = initial_state.clone();
        let mut discounted_cost = 0.0;
        let mut discount = 1.0;

        for _ in 0..periods {
            let realized_demands = demand_models
                .iter()
                .map(|model| sample_demand(&mut rng, model))
                .collect::<PyResult<Vec<_>>>()?;
            let edge_requests = policy_requests(policy_name, params, graph, &state)?;
            let outcome = step_state(
                graph,
                &state,
                &edge_requests,
                &realized_demands,
                holding_costs,
                backlog_costs,
            )?;
            discounted_cost += discount * outcome.period_cost;
            discount *= discount_factor;
            state = outcome.next_state;
        }

        discounted_costs.push(discounted_cost);
    }

    let mean_cost = discounted_costs.iter().sum::<f64>() / discounted_costs.len() as f64;
    let variance = discounted_costs
        .iter()
        .map(|value| (value - mean_cost).powi(2))
        .sum::<f64>()
        / discounted_costs.len() as f64;
    Ok(PolicySimulationSummary {
        mean_cost,
        cost_std: variance.sqrt(),
    })
}
