use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use statrs::distribution::{Discrete, DiscreteCDF, Poisson};

use crate::problems::lost_sales_fixed_order_cost::heuristics::{
    modified_s_s_q_order_quantity, s_nq_order_quantity, s_s_order_quantity,
};
use crate::problems::lost_sales_fixed_order_cost::references::FixedCostLostSalesReferenceInstance;

const DEFAULT_EPSILON: f64 = 1e-4;
const DEFAULT_MAX_ITERATIONS: usize = 20_000;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExactStateKey {
    pub on_hand_inventory: usize,
    pub outstanding_orders: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactPolicyEvaluation {
    pub average_cost: f64,
    pub first_action: usize,
    pub iterations: usize,
    pub final_span: f64,
    pub inventory_position_cap: usize,
    pub state_space_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactPolicyKind {
    Optimal,
    Ss { s: usize, s_up_to: usize },
    Snq { s: usize, q: usize },
    ModifiedSsQ { s: usize, s_up_to: usize, q: usize },
}

struct DemandModel {
    distribution: Poisson,
    mean: f64,
}

struct StateSpace {
    states: Vec<ExactStateKey>,
    index_by_state: HashMap<ExactStateKey, usize>,
}

fn build_demand_model(reference: &FixedCostLostSalesReferenceInstance) -> PyResult<DemandModel> {
    if reference.demand_distribution != "poisson" {
        return Err(PyValueError::new_err(
            "exact fixed-cost lost-sales solver currently supports only Poisson demand",
        ));
    }
    if reference.review_periods != 1 {
        return Err(PyValueError::new_err(
            "exact fixed-cost lost-sales solver currently supports review_periods == 1 only",
        ));
    }
    let distribution = Poisson::new(reference.demand_mean_per_review_period).map_err(|err| {
        PyValueError::new_err(format!("failed to build Poisson distribution: {err}"))
    })?;
    Ok(DemandModel {
        distribution,
        mean: reference.demand_mean_per_review_period,
    })
}

fn validate_reference(reference: &FixedCostLostSalesReferenceInstance) -> PyResult<()> {
    if reference.lead_time == 0 {
        return Err(PyValueError::new_err("lead_time must be at least 1"));
    }
    if reference.holding_cost < 0.0 || reference.shortage_cost < 0.0 || reference.fixed_order_cost < 0.0 {
        return Err(PyValueError::new_err(
            "cost parameters must be nonnegative",
        ));
    }
    build_demand_model(reference).map(|_| ())
}

fn expected_period_cost(
    demand: &DemandModel,
    holding_cost: f64,
    shortage_cost: f64,
    on_hand_inventory: usize,
) -> f64 {
    let overage = (0..on_hand_inventory)
        .map(|demand_value| {
            (on_hand_inventory - demand_value) as f64
                * demand.distribution.pmf(demand_value as u64)
        })
        .sum::<f64>();
    let underage = demand.mean - on_hand_inventory as f64 + overage;
    holding_cost * overage + shortage_cost * underage
}

fn total_inventory_position(state: &ExactStateKey) -> usize {
    state.on_hand_inventory + state.outstanding_orders.iter().sum::<usize>()
}

fn enumerate_exact_compositions(
    slots_remaining: usize,
    total_remaining: usize,
    current: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
) {
    if slots_remaining == 1 {
        current.push(total_remaining);
        output.push(current.clone());
        current.pop();
        return;
    }
    for value in 0..=total_remaining {
        current.push(value);
        enumerate_exact_compositions(slots_remaining - 1, total_remaining - value, current, output);
        current.pop();
    }
}

fn build_state_space(lead_time: usize, inventory_position_cap: usize) -> StateSpace {
    let slots = lead_time;
    let mut states = Vec::new();
    let mut index_by_state = HashMap::new();

    for total in 0..=inventory_position_cap {
        let mut tuples = Vec::new();
        enumerate_exact_compositions(slots, total, &mut Vec::new(), &mut tuples);
        for tuple in tuples {
            let state = ExactStateKey {
                on_hand_inventory: tuple[0],
                outstanding_orders: tuple[1..].to_vec(),
            };
            let idx = states.len();
            index_by_state.insert(state.clone(), idx);
            states.push(state);
        }
    }

    StateSpace {
        states,
        index_by_state,
    }
}

fn next_state(
    state: &ExactStateKey,
    demand: usize,
    order_quantity: usize,
    lead_time: usize,
) -> ExactStateKey {
    let remaining_inventory = state.on_hand_inventory.saturating_sub(demand);
    if lead_time == 1 {
        return ExactStateKey {
            on_hand_inventory: remaining_inventory + order_quantity,
            outstanding_orders: Vec::new(),
        };
    }

    let arriving_order = state.outstanding_orders[0];
    let mut next_pipeline = state.outstanding_orders[1..].to_vec();
    next_pipeline.push(order_quantity);
    ExactStateKey {
        on_hand_inventory: remaining_inventory + arriving_order,
        outstanding_orders: next_pipeline,
    }
}

fn greedy_action(
    state: &ExactStateKey,
    inventory_position_cap: usize,
    policy: ExactPolicyKind,
) -> PyResult<usize> {
    let inventory_position = total_inventory_position(state) as i64;
    let residual_capacity = inventory_position_cap.saturating_sub(total_inventory_position(state));
    let action = match policy {
        ExactPolicyKind::Optimal => {
            return Err(PyValueError::new_err(
                "greedy_action should not be called for ExactPolicyKind::Optimal",
            ))
        }
        ExactPolicyKind::Ss { s, s_up_to } => {
            s_s_order_quantity(inventory_position, s, s_up_to, residual_capacity)
        }
        ExactPolicyKind::Snq { s, q } => {
            s_nq_order_quantity(inventory_position, s, q, residual_capacity)?
        }
        ExactPolicyKind::ModifiedSsQ { s, s_up_to, q } => {
            modified_s_s_q_order_quantity(inventory_position, s, s_up_to, q, residual_capacity)?
        }
    };
    Ok(action)
}

fn candidate_actions(state: &ExactStateKey, inventory_position_cap: usize) -> Vec<usize> {
    let residual_capacity = inventory_position_cap.saturating_sub(total_inventory_position(state));
    (0..=residual_capacity).collect()
}

fn evaluate_action(
    state: &ExactStateKey,
    order_quantity: usize,
    value_function: &[f64],
    state_space: &StateSpace,
    reference: &FixedCostLostSalesReferenceInstance,
    demand: &DemandModel,
) -> PyResult<f64> {
    let mut continuation = 0.0;
    for demand_value in 0..state.on_hand_inventory {
        let probability = demand.distribution.pmf(demand_value as u64);
        if probability <= 0.0 {
            continue;
        }
        let successor = next_state(state, demand_value, order_quantity, reference.lead_time);
        let successor_idx = *state_space.index_by_state.get(&successor).ok_or_else(|| {
            PyValueError::new_err("successor state fell outside the bounded state space")
        })?;
        continuation += probability * value_function[successor_idx];
    }

    let tail_probability = if state.on_hand_inventory == 0 {
        1.0
    } else {
        1.0 - demand.distribution.cdf((state.on_hand_inventory - 1) as u64)
    };
    if tail_probability > 0.0 {
        let successor = next_state(
            state,
            state.on_hand_inventory,
            order_quantity,
            reference.lead_time,
        );
        let successor_idx = *state_space.index_by_state.get(&successor).ok_or_else(|| {
            PyValueError::new_err("tail successor state fell outside the bounded state space")
        })?;
        continuation += tail_probability * value_function[successor_idx];
    }

    Ok(expected_period_cost(
        demand,
        reference.holding_cost,
        reference.shortage_cost,
        state.on_hand_inventory,
    ) + if order_quantity > 0 {
        reference.fixed_order_cost
    } else {
        0.0
    } + continuation)
}

fn solve_bounded_average_cost_policy(
    reference: &FixedCostLostSalesReferenceInstance,
    inventory_position_cap: usize,
    policy: ExactPolicyKind,
    epsilon: f64,
    max_iterations: usize,
) -> PyResult<(ExactPolicyEvaluation, Vec<usize>, StateSpace)> {
    validate_reference(reference)?;
    let demand = build_demand_model(reference)?;
    let state_space = build_state_space(reference.lead_time, inventory_position_cap);
    let mut values = vec![0.0; state_space.states.len()];
    let mut next_values = vec![0.0; state_space.states.len()];
    let mut policy_actions = vec![0usize; state_space.states.len()];
    let mut final_span = f64::INFINITY;
    let mut iterations = None;

    for iteration in 1..=max_iterations {
        let mut min_delta = f64::INFINITY;
        let mut max_delta = -f64::INFINITY;

        for (state_idx, state) in state_space.states.iter().enumerate() {
            let (best_action, best_value) = match policy {
                ExactPolicyKind::Optimal => {
                    let mut best_action = 0usize;
                    let mut best_value = f64::INFINITY;
                    for action in candidate_actions(state, inventory_position_cap) {
                        let value = evaluate_action(
                            state,
                            action,
                            &values,
                            &state_space,
                            reference,
                            &demand,
                        )?;
                        if value < best_value - 1e-12 {
                            best_value = value;
                            best_action = action;
                        }
                    }
                    (best_action, best_value)
                }
                _ => {
                    let action = greedy_action(state, inventory_position_cap, policy)?;
                    let value =
                        evaluate_action(state, action, &values, &state_space, reference, &demand)?;
                    (action, value)
                }
            };
            policy_actions[state_idx] = best_action;
            next_values[state_idx] = best_value;
            let delta = next_values[state_idx] - values[state_idx];
            min_delta = min_delta.min(delta);
            max_delta = max_delta.max(delta);
        }

        let previous_values = values.clone();
        values.clone_from_slice(&next_values);
        final_span = max_delta - min_delta;
        iterations = Some(iteration);
        if final_span < epsilon {
            let initial_state = ExactStateKey {
                on_hand_inventory: 0,
                outstanding_orders: vec![0usize; reference.lead_time.saturating_sub(1)],
            };
            let initial_idx = *state_space.index_by_state.get(&initial_state).ok_or_else(|| {
                PyValueError::new_err("initial state missing from bounded state space")
            })?;
            return Ok((
                ExactPolicyEvaluation {
                    average_cost: average_cost_for_values(&previous_values, &values),
                    first_action: policy_actions[initial_idx],
                    iterations: iterations.expect("iteration count must be set"),
                    final_span,
                    inventory_position_cap,
                    state_space_size: state_space.states.len(),
                },
                policy_actions,
                state_space,
            ));
        }
    }

    Err(PyValueError::new_err(format!(
        "average-cost value iteration did not converge within {max_iterations} iterations; final span={final_span}"
    )))
}

fn average_cost_for_values(previous_values: &[f64], current_values: &[f64]) -> f64 {
    let mut min_delta = f64::INFINITY;
    let mut max_delta = -f64::INFINITY;
    for (new, old) in current_values.iter().zip(previous_values.iter()) {
        let delta = new - old;
        min_delta = min_delta.min(delta);
        max_delta = max_delta.max(delta);
    }
    (min_delta + max_delta) / 2.0
}

pub fn solve_optimal_policy(
    reference: &FixedCostLostSalesReferenceInstance,
    inventory_position_cap: usize,
) -> PyResult<ExactPolicyEvaluation> {
    validate_reference(reference)?;
    let demand = build_demand_model(reference)?;
    let state_space = build_state_space(reference.lead_time, inventory_position_cap);
    let mut values = vec![0.0; state_space.states.len()];
    let mut next_values = vec![0.0; state_space.states.len()];
    let mut policy_actions = vec![0usize; state_space.states.len()];
    let mut final_span = f64::INFINITY;
    let mut iterations = None;

    for iteration in 1..=DEFAULT_MAX_ITERATIONS {
        let mut min_delta = f64::INFINITY;
        let mut max_delta = -f64::INFINITY;
        for (state_idx, state) in state_space.states.iter().enumerate() {
            let mut best_action = 0usize;
            let mut best_value = f64::INFINITY;
            for action in candidate_actions(state, inventory_position_cap) {
                let value =
                    evaluate_action(state, action, &values, &state_space, reference, &demand)?;
                if value < best_value - 1e-12 {
                    best_value = value;
                    best_action = action;
                }
            }
            policy_actions[state_idx] = best_action;
            next_values[state_idx] = best_value;
            let delta = next_values[state_idx] - values[state_idx];
            min_delta = min_delta.min(delta);
            max_delta = max_delta.max(delta);
        }
        let previous_values = values.clone();
        values.clone_from_slice(&next_values);
        final_span = max_delta - min_delta;
        iterations = Some(iteration);
        if final_span < DEFAULT_EPSILON {
            let initial_idx = *state_space
                .index_by_state
                .get(&ExactStateKey {
                    on_hand_inventory: 0,
                    outstanding_orders: vec![0usize; reference.lead_time.saturating_sub(1)],
                })
                .ok_or_else(|| PyValueError::new_err("initial state missing from state space"))?;
            return Ok(ExactPolicyEvaluation {
                average_cost: average_cost_for_values(&previous_values, &values),
                first_action: policy_actions[initial_idx],
                iterations: iterations.expect("iteration count must be set"),
                final_span,
                inventory_position_cap,
                state_space_size: state_space.states.len(),
            });
        }
    }

    Err(PyValueError::new_err(format!(
        "optimal average-cost value iteration did not converge within {DEFAULT_MAX_ITERATIONS} iterations; final span={final_span}"
    )))
}

pub fn evaluate_policy(
    reference: &FixedCostLostSalesReferenceInstance,
    inventory_position_cap: usize,
    policy: ExactPolicyKind,
) -> PyResult<ExactPolicyEvaluation> {
    validate_reference(reference)?;
    let demand = build_demand_model(reference)?;
    let state_space = build_state_space(reference.lead_time, inventory_position_cap);
    let mut values = vec![0.0; state_space.states.len()];
    let mut next_values = vec![0.0; state_space.states.len()];
    let mut policy_actions = vec![0usize; state_space.states.len()];
    let mut final_span = f64::INFINITY;
    let mut iterations = None;

    for iteration in 1..=DEFAULT_MAX_ITERATIONS {
        let mut min_delta = f64::INFINITY;
        let mut max_delta = -f64::INFINITY;
        for (state_idx, state) in state_space.states.iter().enumerate() {
            let action = greedy_action(state, inventory_position_cap, policy)?;
            let value =
                evaluate_action(state, action, &values, &state_space, reference, &demand)?;
            policy_actions[state_idx] = action;
            next_values[state_idx] = value;
            let delta = next_values[state_idx] - values[state_idx];
            min_delta = min_delta.min(delta);
            max_delta = max_delta.max(delta);
        }
        let previous_values = values.clone();
        values.clone_from_slice(&next_values);
        final_span = max_delta - min_delta;
        iterations = Some(iteration);
        if final_span < DEFAULT_EPSILON {
            let initial_idx = *state_space
                .index_by_state
                .get(&ExactStateKey {
                    on_hand_inventory: 0,
                    outstanding_orders: vec![0usize; reference.lead_time.saturating_sub(1)],
                })
                .ok_or_else(|| PyValueError::new_err("initial state missing from state space"))?;
            return Ok(ExactPolicyEvaluation {
                average_cost: average_cost_for_values(&previous_values, &values),
                first_action: policy_actions[initial_idx],
                iterations: iterations.expect("iteration count must be set"),
                final_span,
                inventory_position_cap,
                state_space_size: state_space.states.len(),
            });
        }
    }

    Err(PyValueError::new_err(format!(
        "policy evaluation did not converge within {DEFAULT_MAX_ITERATIONS} iterations; final span={final_span}"
    )))
}

pub fn optimal_action_for_state(
    reference: &FixedCostLostSalesReferenceInstance,
    inventory_position_cap: usize,
    state: &ExactStateKey,
) -> PyResult<usize> {
    let (solution, policy_actions, state_space) = solve_bounded_average_cost_policy(
        reference,
        inventory_position_cap,
        ExactPolicyKind::Optimal,
        DEFAULT_EPSILON,
        DEFAULT_MAX_ITERATIONS,
    )?;
    let _ = solution;
    let state_idx = *state_space.index_by_state.get(state).ok_or_else(|| {
        PyValueError::new_err("requested state is not in the bounded state space")
    })?;
    Ok(policy_actions[state_idx])
}
