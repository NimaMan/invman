use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::decentralized_inventory_control::env::{
    initialize_state, step_state, DecentralizedInventoryControlState,
};
use crate::problems::decentralized_inventory_control::heuristics::{
    base_stock_orders, sterman_anchor_adjust_orders,
};
use crate::problems::decentralized_inventory_control::references::ExactVerificationReference;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactStateKey {
    period: usize,
    on_hand_inventory: Vec<usize>,
    backlog: Vec<usize>,
    shipment_pipelines: Vec<Vec<usize>>,
    order_pipelines: Vec<Vec<usize>>,
    last_received_shipments: Vec<usize>,
    last_received_orders: Vec<usize>,
    forecast_order_bits: Vec<u64>,
    last_actions: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactPolicyEvaluation {
    pub discounted_cost: f64,
    pub first_action: Vec<usize>,
}

fn validate_exact_reference(reference: &ExactVerificationReference) -> PyResult<()> {
    let num_agents = reference.initial_on_hand_inventory.len();
    if num_agents < 2 {
        return Err(PyValueError::new_err(
            "exact verifier requires at least two agents",
        ));
    }
    if reference.initial_backlog.len() != num_agents
        || reference.initial_shipment_pipelines.len() != num_agents
        || reference.initial_order_pipelines.len() != num_agents
        || reference.initial_last_received_shipments.len() != num_agents
        || reference.initial_last_received_orders.len() != num_agents
        || reference.initial_forecast_orders.len() != num_agents
        || reference.initial_last_actions.len() != num_agents
        || reference.demand_smoothing_factors.len() != num_agents
        || reference.holding_costs.len() != num_agents
        || reference.backlog_costs.len() != num_agents
        || reference.max_order_quantities.len() != num_agents
        || reference.base_stock_levels.len() != num_agents
        || reference.sterman_target_positions.len() != num_agents
        || reference.sterman_adjustment_times.len() != num_agents
        || reference.sterman_supply_line_weights.len() != num_agents
    {
        return Err(PyValueError::new_err(
            "all exact-reference vectors must match the number of agents",
        ));
    }
    if !reference.initial_order_pipelines[0].is_empty() {
        return Err(PyValueError::new_err(
            "retailer order pipeline must be empty in the exact reference",
        ));
    }
    if reference
        .customer_demand_support
        .len()
        != reference.customer_demand_probabilities.len()
    {
        return Err(PyValueError::new_err(
            "customer_demand_support and probabilities must have the same length",
        ));
    }
    let total_probability = reference.customer_demand_probabilities.iter().sum::<f64>();
    if (total_probability - 1.0).abs() > 1e-12 {
        return Err(PyValueError::new_err(format!(
            "customer_demand_probabilities must sum to 1, found {total_probability}"
        )));
    }
    Ok(())
}

fn nested_vec(values: &[&[usize]]) -> Vec<Vec<usize>> {
    values.iter().map(|row| row.to_vec()).collect()
}

fn build_initial_state(
    reference: &ExactVerificationReference,
) -> PyResult<DecentralizedInventoryControlState> {
    initialize_state(
        reference.initial_on_hand_inventory,
        reference.initial_backlog,
        &nested_vec(reference.initial_shipment_pipelines),
        &nested_vec(reference.initial_order_pipelines),
        reference.initial_last_received_shipments,
        reference.initial_last_received_orders,
        reference.initial_forecast_orders,
        reference.initial_last_actions,
    )
}

fn as_state_key(state: &DecentralizedInventoryControlState) -> ExactStateKey {
    ExactStateKey {
        period: state.period,
        on_hand_inventory: state.on_hand_inventory.clone(),
        backlog: state.backlog.clone(),
        shipment_pipelines: state.shipment_pipelines.clone(),
        order_pipelines: state.order_pipelines.clone(),
        last_received_shipments: state.last_received_shipments.clone(),
        last_received_orders: state.last_received_orders.clone(),
        forecast_order_bits: state
            .forecast_orders
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        last_actions: state.last_actions.clone(),
    }
}

fn enumerate_actions(max_order_quantities: &[usize]) -> Vec<Vec<usize>> {
    fn recurse(
        agent_idx: usize,
        max_order_quantities: &[usize],
        partial: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if agent_idx == max_order_quantities.len() {
            output.push(partial.clone());
            return;
        }
        for action in 0..=max_order_quantities[agent_idx] {
            partial.push(action);
            recurse(agent_idx + 1, max_order_quantities, partial, output);
            partial.pop();
        }
    }

    let mut output = Vec::new();
    recurse(0, max_order_quantities, &mut Vec::new(), &mut output);
    output
}

fn solve_optimal_from_state(
    state: DecentralizedInventoryControlState,
    reference: &ExactVerificationReference,
    action_grid: &[Vec<usize>],
    cache: &mut HashMap<ExactStateKey, ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: vec![0; reference.initial_on_hand_inventory.len()],
        });
    }
    let key = as_state_key(&state);
    if let Some(cached) = cache.get(&key) {
        return Ok(cached.clone());
    }

    let mut best_cost = f64::INFINITY;
    let mut best_action = vec![0; reference.initial_on_hand_inventory.len()];

    for action in action_grid.iter() {
        let mut expected_cost = 0.0;
        for (demand, probability) in reference
            .customer_demand_support
            .iter()
            .zip(reference.customer_demand_probabilities.iter())
        {
            let outcome = step_state(
                &state,
                action,
                *demand as usize,
                reference.demand_smoothing_factors,
                reference.holding_costs,
                reference.backlog_costs,
            )?;
            let continuation =
                solve_optimal_from_state(outcome.next_state, reference, action_grid, cache)?;
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

fn heuristic_actions(
    state: &DecentralizedInventoryControlState,
    reference: &ExactVerificationReference,
    policy_name: &str,
) -> PyResult<Vec<usize>> {
    match policy_name {
        "base_stock" => base_stock_orders(state, reference.base_stock_levels),
        "sterman_anchor_adjust" => sterman_anchor_adjust_orders(
            state,
            reference.sterman_target_positions,
            reference.sterman_adjustment_times,
            reference.sterman_supply_line_weights,
        ),
        _ => Err(PyValueError::new_err(format!(
            "unsupported heuristic '{policy_name}'"
        ))),
    }
}

fn evaluate_heuristic_from_state(
    state: DecentralizedInventoryControlState,
    reference: &ExactVerificationReference,
    policy_name: &str,
    cache: &mut HashMap<ExactStateKey, ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: vec![0; reference.initial_on_hand_inventory.len()],
        });
    }
    let key = as_state_key(&state);
    if let Some(cached) = cache.get(&key) {
        return Ok(cached.clone());
    }

    let action = heuristic_actions(&state, reference, policy_name)?;
    let mut expected_cost = 0.0;
    for (demand, probability) in reference
        .customer_demand_support
        .iter()
        .zip(reference.customer_demand_probabilities.iter())
    {
        let outcome = step_state(
            &state,
            &action,
            *demand as usize,
            reference.demand_smoothing_factors,
            reference.holding_costs,
            reference.backlog_costs,
        )?;
        let continuation =
            evaluate_heuristic_from_state(outcome.next_state, reference, policy_name, cache)?;
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
    let initial_state = build_initial_state(reference)?;
    let action_grid = enumerate_actions(reference.max_order_quantities);
    let mut cache = HashMap::new();
    solve_optimal_from_state(initial_state, reference, &action_grid, &mut cache)
}

pub fn evaluate_named_heuristic(
    reference: &ExactVerificationReference,
    policy_name: &str,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let initial_state = build_initial_state(reference)?;
    let mut cache = HashMap::new();
    evaluate_heuristic_from_state(initial_state, reference, policy_name, &mut cache)
}
