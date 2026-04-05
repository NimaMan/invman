use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::joint_replenishment::demand::{support, DemandRange};
use crate::problems::joint_replenishment::env::{step_state, JointReplenishmentState};
use crate::problems::joint_replenishment::heuristics::{
    dynamic_order_up_to_order_quantities, minimum_order_quantity_order_quantities,
};
use crate::problems::joint_replenishment::references::ExactVerificationReference;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ExactStateKey {
    period: usize,
    inventory_levels: [i32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactPolicyEvaluation {
    pub discounted_cost: f64,
    pub first_action: [usize; 2],
}

fn validate_exact_reference(reference: &ExactVerificationReference) -> PyResult<()> {
    let expected_len = 2usize;
    if reference.max_order_quantities.len() != expected_len
        || reference.initial_inventory_levels.len() != expected_len
        || reference.minor_order_costs.len() != expected_len
        || reference.holding_costs.len() != expected_len
        || reference.shortage_costs.len() != expected_len
        || reference.demand_ranges.len() != expected_len
        || reference.moq_item_targets.len() != expected_len
        || reference.dynout_item_targets.len() != expected_len
    {
        return Err(PyValueError::new_err(
            "exact joint_replenishment verifier currently supports exactly two items",
        ));
    }
    if reference.periods == 0 {
        return Err(PyValueError::new_err("periods must be positive"));
    }
    if !(0.0..=1.0).contains(&reference.discount_factor) {
        return Err(PyValueError::new_err(
            "discount_factor must lie in [0, 1]",
        ));
    }
    Ok(())
}

fn as_state_key(period: usize, inventory_levels: &[i32]) -> ExactStateKey {
    ExactStateKey {
        period,
        inventory_levels: [inventory_levels[0], inventory_levels[1]],
    }
}

fn to_state(state: ExactStateKey) -> JointReplenishmentState {
    JointReplenishmentState {
        period: state.period,
        inventory_levels: vec![state.inventory_levels[0], state.inventory_levels[1]],
    }
}

fn lexicographically_smaller(lhs: &[usize; 2], rhs: &[usize; 2]) -> bool {
    lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1])
}

fn demand_scenarios(demand_ranges: &[DemandRange]) -> PyResult<Vec<([usize; 2], f64)>> {
    let left_support = support(demand_ranges[0])?;
    let right_support = support(demand_ranges[1])?;
    let mut scenarios = Vec::new();
    for (left_demand, left_probability) in left_support.iter() {
        for (right_demand, right_probability) in right_support.iter() {
            scenarios.push(([*left_demand, *right_demand], left_probability * right_probability));
        }
    }
    Ok(scenarios)
}

fn solve_optimal_from_state(
    state: ExactStateKey,
    reference: &ExactVerificationReference,
    scenarios: &[([usize; 2], f64)],
    cache: &mut HashMap<ExactStateKey, ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: [0, 0],
        });
    }
    if let Some(cached) = cache.get(&state) {
        return Ok(*cached);
    }

    let state_for_step = to_state(state);
    let mut best_cost = f64::INFINITY;
    let mut best_action = [0usize, 0usize];

    for action_0 in 0..=reference.max_order_quantities[0] {
        for action_1 in 0..=reference.max_order_quantities[1] {
            let action = [action_0, action_1];
            let mut expected_cost = 0.0;
            for (demands, probability) in scenarios.iter() {
                let outcome = step_state(
                    &state_for_step,
                    &action,
                    demands,
                    reference.truck_capacity,
                    reference.minor_order_costs,
                    reference.major_order_cost,
                    reference.holding_costs,
                    reference.shortage_costs,
                )?;
                let next_state = as_state_key(
                    state.period + 1,
                    &outcome.next_state.inventory_levels,
                );
                let continuation =
                    solve_optimal_from_state(next_state, reference, scenarios, cache)?;
                expected_cost += probability
                    * (outcome.period_cost
                        + reference.discount_factor * continuation.discounted_cost);
            }
            if expected_cost < best_cost - 1e-12
                || ((expected_cost - best_cost).abs() < 1e-12
                    && lexicographically_smaller(&action, &best_action))
            {
                best_cost = expected_cost;
                best_action = action;
            }
        }
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: best_cost,
        first_action: best_action,
    };
    cache.insert(state, result);
    Ok(result)
}

pub fn solve_optimal_policy(
    reference: &ExactVerificationReference,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let scenarios = demand_scenarios(reference.demand_ranges)?;
    let initial_state = as_state_key(0, reference.initial_inventory_levels);
    let mut cache = HashMap::new();
    solve_optimal_from_state(initial_state, reference, &scenarios, &mut cache)
}

fn heuristic_action(
    heuristic_name: &str,
    state: ExactStateKey,
    reference: &ExactVerificationReference,
) -> PyResult<[usize; 2]> {
    let state_for_heuristic = to_state(state);
    let action = match heuristic_name {
        "minimum_order_quantity" | "moq" => minimum_order_quantity_order_quantities(
            &state_for_heuristic,
            reference.moq_item_targets,
            reference.moq_review_period,
            reference.moq_rounding_threshold,
            reference.truck_capacity,
        )?,
        "dynamic_order_up_to" | "dynout" => dynamic_order_up_to_order_quantities(
            &state_for_heuristic,
            reference.dynout_item_targets,
            reference.truck_capacity,
            reference.demand_ranges,
            reference.holding_costs,
            reference.shortage_costs,
        )?,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported heuristic '{heuristic_name}'",
            )))
        }
    };
    Ok([
        action[0].min(reference.max_order_quantities[0]),
        action[1].min(reference.max_order_quantities[1]),
    ])
}

fn evaluate_heuristic_from_state(
    heuristic_name: &'static str,
    state: ExactStateKey,
    reference: &ExactVerificationReference,
    scenarios: &[([usize; 2], f64)],
    cache: &mut HashMap<(ExactStateKey, &'static str), ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: [0, 0],
        });
    }
    let key = (state, heuristic_name);
    if let Some(cached) = cache.get(&key) {
        return Ok(*cached);
    }

    let state_for_step = to_state(state);
    let action = heuristic_action(heuristic_name, state, reference)?;
    let mut expected_cost = 0.0;
    for (demands, probability) in scenarios.iter() {
        let outcome = step_state(
            &state_for_step,
            &action,
            demands,
            reference.truck_capacity,
            reference.minor_order_costs,
            reference.major_order_cost,
            reference.holding_costs,
            reference.shortage_costs,
        )?;
        let next_state = as_state_key(state.period + 1, &outcome.next_state.inventory_levels);
        let continuation =
            evaluate_heuristic_from_state(heuristic_name, next_state, reference, scenarios, cache)?;
        expected_cost += probability
            * (outcome.period_cost + reference.discount_factor * continuation.discounted_cost);
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: expected_cost,
        first_action: action,
    };
    cache.insert(key, result);
    Ok(result)
}

pub fn evaluate_named_heuristic(
    reference: &ExactVerificationReference,
    heuristic_name: &'static str,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let scenarios = demand_scenarios(reference.demand_ranges)?;
    let initial_state = as_state_key(0, reference.initial_inventory_levels);
    let mut cache = HashMap::new();
    evaluate_heuristic_from_state(
        heuristic_name,
        initial_state,
        reference,
        &scenarios,
        &mut cache,
    )
}
