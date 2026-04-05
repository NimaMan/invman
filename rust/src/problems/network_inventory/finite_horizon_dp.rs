use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::network_inventory::env::{
    initialize_state, step_state, NetworkInventoryGraph, NetworkInventoryState,
};
use crate::problems::network_inventory::heuristics::node_base_stock_requests;
use crate::problems::network_inventory::references::ExactVerificationReference;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactStateKey {
    period: usize,
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    edge_pipelines: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactPolicyEvaluation {
    pub discounted_cost: f64,
    pub first_action: Vec<usize>,
}

fn build_graph(reference: &ExactVerificationReference) -> NetworkInventoryGraph {
    NetworkInventoryGraph {
        num_nodes: reference.num_nodes,
        source_nodes: reference.source_nodes.to_vec(),
        edges: reference.edges.to_vec(),
    }
}

fn nested_vec(rows: &[&[usize]]) -> Vec<Vec<usize>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

fn validate_exact_reference(reference: &ExactVerificationReference) -> PyResult<()> {
    if reference.source_nodes.len() != reference.num_nodes
        || reference.initial_on_hand_inventory.len() != reference.num_nodes
        || reference.initial_backlog.len() != reference.num_nodes
        || reference.holding_costs.len() != reference.num_nodes
        || reference.backlog_costs.len() != reference.num_nodes
        || reference.demand_supports.len() != reference.num_nodes
        || reference.demand_probabilities.len() != reference.num_nodes
        || reference.base_stock_levels.len() != reference.num_nodes
    {
        return Err(PyValueError::new_err(
            "all node-wise verification arrays must match num_nodes",
        ));
    }
    if reference.initial_edge_pipelines.len() != reference.edges.len()
        || reference.max_edge_requests.len() != reference.edges.len()
    {
        return Err(PyValueError::new_err(
            "all edge-wise verification arrays must match the number of edges",
        ));
    }
    for node_idx in 0..reference.num_nodes {
        if reference.demand_supports[node_idx].len() != reference.demand_probabilities[node_idx].len()
        {
            return Err(PyValueError::new_err(format!(
                "demand support and probabilities must have the same length for node {node_idx}"
            )));
        }
        let total_probability = reference.demand_probabilities[node_idx].iter().sum::<f64>();
        if (total_probability - 1.0).abs() > 1e-12 {
            return Err(PyValueError::new_err(format!(
                "node {node_idx} demand probabilities must sum to 1, found {total_probability}"
            )));
        }
    }
    Ok(())
}

fn build_initial_state(
    reference: &ExactVerificationReference,
) -> PyResult<NetworkInventoryState> {
    let graph = build_graph(reference);
    initialize_state(
        &graph,
        reference.initial_on_hand_inventory,
        reference.initial_backlog,
        &nested_vec(reference.initial_edge_pipelines),
    )
}

fn as_state_key(state: &NetworkInventoryState) -> ExactStateKey {
    ExactStateKey {
        period: state.period,
        on_hand_inventory: state.on_hand_inventory.clone(),
        backlog: state.backlog.clone(),
        edge_pipelines: state.edge_pipelines.clone(),
    }
}

fn enumerate_edge_actions(max_edge_requests: &[usize]) -> Vec<Vec<usize>> {
    fn recurse(
        edge_idx: usize,
        max_edge_requests: &[usize],
        partial: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if edge_idx == max_edge_requests.len() {
            output.push(partial.clone());
            return;
        }
        for request in 0..=max_edge_requests[edge_idx] {
            partial.push(request);
            recurse(edge_idx + 1, max_edge_requests, partial, output);
            partial.pop();
        }
    }

    let mut output = Vec::new();
    recurse(0, max_edge_requests, &mut Vec::new(), &mut output);
    output
}

fn demand_scenarios(reference: &ExactVerificationReference) -> Vec<(Vec<usize>, f64)> {
    fn recurse(
        node_idx: usize,
        reference: &ExactVerificationReference,
        partial: &mut Vec<usize>,
        probability: f64,
        output: &mut Vec<(Vec<usize>, f64)>,
    ) {
        if node_idx == reference.num_nodes {
            output.push((partial.clone(), probability));
            return;
        }
        for (demand, demand_probability) in reference.demand_supports[node_idx]
            .iter()
            .zip(reference.demand_probabilities[node_idx].iter())
        {
            partial.push(*demand as usize);
            recurse(
                node_idx + 1,
                reference,
                partial,
                probability * demand_probability,
                output,
            );
            partial.pop();
        }
    }

    let mut output = Vec::new();
    recurse(0, reference, &mut Vec::new(), 1.0, &mut output);
    output
}

fn solve_optimal_from_state(
    graph: &NetworkInventoryGraph,
    state: NetworkInventoryState,
    reference: &ExactVerificationReference,
    action_grid: &[Vec<usize>],
    demand_scenarios: &[(Vec<usize>, f64)],
    cache: &mut HashMap<ExactStateKey, ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: vec![0; graph.edges.len()],
        });
    }
    let key = as_state_key(&state);
    if let Some(cached) = cache.get(&key) {
        return Ok(cached.clone());
    }

    let mut best_cost = f64::INFINITY;
    let mut best_action = vec![0; graph.edges.len()];

    for action in action_grid.iter() {
        let mut expected_cost = 0.0;
        for (demands, probability) in demand_scenarios.iter() {
            let outcome = step_state(
                graph,
                &state,
                action,
                demands,
                reference.holding_costs,
                reference.backlog_costs,
            )?;
            let continuation = solve_optimal_from_state(
                graph,
                outcome.next_state,
                reference,
                action_grid,
                demand_scenarios,
                cache,
            )?;
            expected_cost += probability
                * (outcome.period_cost + reference.discount_factor * continuation.discounted_cost);
        }
        if expected_cost < best_cost {
            best_cost = expected_cost;
            best_action = action.clone();
        }
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: best_cost,
        first_action: best_action,
    };
    cache.insert(key, result.clone());
    Ok(result)
}

fn evaluate_base_stock_from_state(
    graph: &NetworkInventoryGraph,
    state: NetworkInventoryState,
    reference: &ExactVerificationReference,
    demand_scenarios: &[(Vec<usize>, f64)],
    cache: &mut HashMap<ExactStateKey, ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: vec![0; graph.edges.len()],
        });
    }
    let key = as_state_key(&state);
    if let Some(cached) = cache.get(&key) {
        return Ok(cached.clone());
    }

    let action = node_base_stock_requests(graph, &state, reference.base_stock_levels)?;
    let mut expected_cost = 0.0;
    for (demands, probability) in demand_scenarios.iter() {
        let outcome = step_state(
            graph,
            &state,
            &action,
            demands,
            reference.holding_costs,
            reference.backlog_costs,
        )?;
        let continuation =
            evaluate_base_stock_from_state(graph, outcome.next_state, reference, demand_scenarios, cache)?;
        expected_cost +=
            probability * (outcome.period_cost + reference.discount_factor * continuation.discounted_cost);
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: expected_cost,
        first_action: action,
    };
    cache.insert(key, result.clone());
    Ok(result)
}

pub fn solve_optimal_policy(
    reference: &ExactVerificationReference,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let graph = build_graph(reference);
    let initial_state = build_initial_state(reference)?;
    let action_grid = enumerate_edge_actions(reference.max_edge_requests);
    let scenarios = demand_scenarios(reference);
    let mut cache = HashMap::new();
    solve_optimal_from_state(&graph, initial_state, reference, &action_grid, &scenarios, &mut cache)
}

pub fn evaluate_named_heuristic(
    reference: &ExactVerificationReference,
    policy_name: &str,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let graph = build_graph(reference);
    let initial_state = build_initial_state(reference)?;
    let scenarios = demand_scenarios(reference);
    let mut cache = HashMap::new();
    match policy_name {
        "node_base_stock" => {
            evaluate_base_stock_from_state(&graph, initial_state, reference, &scenarios, &mut cache)
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported heuristic '{policy_name}'"
        ))),
    }
}
