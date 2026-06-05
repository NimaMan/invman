use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::multi_echelon::production_assembly_distribution_network::env::{
    initialize_state, step_state, supply_relation_count, validate_graph, NetworkInventoryGraph,
    NetworkInventoryState,
};
use crate::problems::multi_echelon::production_assembly_distribution_network::heuristics::pairwise_base_stock_requests;
use crate::problems::multi_echelon::production_assembly_distribution_network::literature::ExactVerificationReference;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactStateKey {
    period: usize,
    finished_inventory: Vec<usize>,
    raw_inventory_by_relation: Vec<usize>,
    internal_backlog_by_edge: Vec<usize>,
    external_backlog: Vec<usize>,
    supply_pipelines: Vec<Vec<usize>>,
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
        node_modes: reference.node_modes.to_vec(),
        external_supplier_lead_times: reference.external_supplier_lead_times.to_vec(),
        edges: reference.edges.to_vec(),
    }
}

fn nested_vec(rows: &[&[usize]]) -> Vec<Vec<usize>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

fn validate_exact_reference(reference: &ExactVerificationReference) -> PyResult<()> {
    let graph = build_graph(reference);
    validate_graph(&graph)?;
    let relation_count = supply_relation_count(&graph);
    if reference.source_nodes.len() != reference.num_nodes
        || reference.node_modes.len() != reference.num_nodes
        || reference.external_supplier_lead_times.len() != reference.num_nodes
        || reference.initial_finished_inventory.len() != reference.num_nodes
        || reference.initial_external_backlog.len() != reference.num_nodes
        || reference.holding_costs.len() != reference.num_nodes
        || reference.backlog_costs.len() != reference.num_nodes
        || reference.demand_supports.len() != reference.num_nodes
        || reference.demand_probabilities.len() != reference.num_nodes
    {
        return Err(PyValueError::new_err(
            "all node-wise verification arrays must match num_nodes",
        ));
    }
    if reference.initial_raw_inventory_by_relation.len() != relation_count
        || reference.initial_supply_pipelines.len() != relation_count
        || reference.max_supply_requests.len() != relation_count
        || reference.base_stock_levels.len() != relation_count
    {
        return Err(PyValueError::new_err(
            "all supply-relation arrays must match the number of supply relations",
        ));
    }
    if reference.initial_internal_backlog_by_edge.len() != graph.edges.len() {
        return Err(PyValueError::new_err(
            "initial_internal_backlog_by_edge must match the number of internal edges",
        ));
    }
    for node_idx in 0..reference.num_nodes {
        if reference.demand_supports[node_idx].len()
            != reference.demand_probabilities[node_idx].len()
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

fn build_initial_state(reference: &ExactVerificationReference) -> PyResult<NetworkInventoryState> {
    let graph = build_graph(reference);
    initialize_state(
        &graph,
        reference.initial_finished_inventory,
        reference.initial_raw_inventory_by_relation,
        reference.initial_internal_backlog_by_edge,
        reference.initial_external_backlog,
        &nested_vec(reference.initial_supply_pipelines),
    )
}

fn as_state_key(state: &NetworkInventoryState) -> ExactStateKey {
    ExactStateKey {
        period: state.period,
        finished_inventory: state.finished_inventory.clone(),
        raw_inventory_by_relation: state.raw_inventory_by_relation.clone(),
        internal_backlog_by_edge: state.internal_backlog_by_edge.clone(),
        external_backlog: state.external_backlog.clone(),
        supply_pipelines: state.supply_pipelines.clone(),
    }
}

fn enumerate_supply_actions(max_supply_requests: &[usize]) -> Vec<Vec<usize>> {
    fn recurse(
        relation_idx: usize,
        max_supply_requests: &[usize],
        partial: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if relation_idx == max_supply_requests.len() {
            output.push(partial.clone());
            return;
        }
        for request in 0..=max_supply_requests[relation_idx] {
            partial.push(request);
            recurse(relation_idx + 1, max_supply_requests, partial, output);
            partial.pop();
        }
    }

    let mut output = Vec::new();
    recurse(0, max_supply_requests, &mut Vec::new(), &mut output);
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
            first_action: vec![0; supply_relation_count(graph)],
        });
    }
    let key = as_state_key(&state);
    if let Some(cached) = cache.get(&key) {
        return Ok(cached.clone());
    }

    let mut expected_cost = 0.0;
    let mut representative_action = vec![0; supply_relation_count(graph)];
    let mut representative_probability = -1.0f64;

    for (demands, probability) in demand_scenarios.iter() {
        let mut best_scenario_cost = f64::INFINITY;
        let mut best_scenario_action = vec![0; supply_relation_count(graph)];

        for action in action_grid.iter() {
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
            let scenario_cost =
                outcome.period_cost + reference.discount_factor * continuation.discounted_cost;
            if scenario_cost < best_scenario_cost {
                best_scenario_cost = scenario_cost;
                best_scenario_action = action.clone();
            }
        }

        if *probability > representative_probability {
            representative_probability = *probability;
            representative_action = best_scenario_action;
        }
        expected_cost += probability * best_scenario_cost;
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: expected_cost,
        first_action: representative_action,
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
            first_action: vec![0; supply_relation_count(graph)],
        });
    }
    let key = as_state_key(&state);
    if let Some(cached) = cache.get(&key) {
        return Ok(cached.clone());
    }

    let mut expected_cost = 0.0;
    let mut representative_action = vec![0; supply_relation_count(graph)];
    let mut representative_probability = -1.0f64;

    for (demands, probability) in demand_scenarios.iter() {
        let action =
            pairwise_base_stock_requests(graph, &state, reference.base_stock_levels, demands)?;
        let outcome = step_state(
            graph,
            &state,
            &action,
            demands,
            reference.holding_costs,
            reference.backlog_costs,
        )?;
        let continuation = evaluate_base_stock_from_state(
            graph,
            outcome.next_state,
            reference,
            demand_scenarios,
            cache,
        )?;
        let scenario_cost =
            outcome.period_cost + reference.discount_factor * continuation.discounted_cost;
        if *probability > representative_probability {
            representative_probability = *probability;
            representative_action = action;
        }
        expected_cost += probability * scenario_cost;
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: expected_cost,
        first_action: representative_action,
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
    let action_grid = enumerate_supply_actions(reference.max_supply_requests);
    let scenarios = demand_scenarios(reference);
    let mut cache = HashMap::new();
    solve_optimal_from_state(
        &graph,
        initial_state,
        reference,
        &action_grid,
        &scenarios,
        &mut cache,
    )
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
        "pairwise_base_stock" | "node_base_stock" => {
            evaluate_base_stock_from_state(&graph, initial_state, reference, &scenarios, &mut cache)
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported heuristic '{policy_name}'"
        ))),
    }
}
