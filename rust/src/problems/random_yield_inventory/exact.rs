use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::random_yield_inventory::env::initialize_state;
use crate::problems::random_yield_inventory::heuristics::{
    weighted_newsvendor_order_quantity, yield_inflated_base_stock_order_quantity,
};
use crate::problems::random_yield_inventory::references::ExactVerificationReference;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ExactStateKey {
    period: usize,
    inventory_level: i32,
    pipeline_orders: [u32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactPolicyEvaluation {
    pub discounted_cost: f64,
    pub first_action: usize,
}

fn validate_exact_reference(reference: &ExactVerificationReference) -> PyResult<()> {
    if reference.lead_time != 2 {
        return Err(PyValueError::new_err(
            "exact verifier currently supports lead_time == 2 only",
        ));
    }
    if reference.demand_support.len() != reference.demand_probabilities.len() {
        return Err(PyValueError::new_err(
            "demand_support and demand_probabilities must have the same length",
        ));
    }
    let total_probability = reference.demand_probabilities.iter().sum::<f64>();
    if (total_probability - 1.0).abs() > 1e-12 {
        return Err(PyValueError::new_err(format!(
            "demand_probabilities must sum to 1, found {total_probability}"
        )));
    }
    Ok(())
}

fn as_state_key(
    period: usize,
    inventory_level: i32,
    pipeline_orders: &[u32],
) -> ExactStateKey {
    ExactStateKey {
        period,
        inventory_level,
        pipeline_orders: [pipeline_orders[0], pipeline_orders[1]],
    }
}

fn transition_inventory(
    state: &ExactStateKey,
    order_quantity: usize,
    demand: u32,
    arrival_succeeds: bool,
) -> (i32, [u32; 2]) {
    let realized_arrival = if arrival_succeeds {
        state.pipeline_orders[0] as i32
    } else {
        0
    };
    let ending_inventory = state.inventory_level + realized_arrival - demand as i32;
    (
        ending_inventory,
        [state.pipeline_orders[1], order_quantity as u32],
    )
}

fn period_cost(
    ending_inventory: i32,
    order_quantity: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
) -> f64 {
    procurement_cost * order_quantity as f64
        + holding_cost * ending_inventory.max(0) as f64
        + shortage_cost * (-ending_inventory).max(0) as f64
}

fn solve_optimal_from_state(
    state: ExactStateKey,
    reference: &ExactVerificationReference,
    cache: &mut HashMap<ExactStateKey, ExactPolicyEvaluation>,
) -> ExactPolicyEvaluation {
    if state.period == reference.periods {
        return ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: 0,
        };
    }
    if let Some(cached) = cache.get(&state) {
        return *cached;
    }

    let mut best_cost = f64::INFINITY;
    let mut best_action = 0usize;

    for action in 0..=reference.max_order_quantity {
        let mut expected_cost = 0.0;
        for (demand, probability) in reference
            .demand_support
            .iter()
            .zip(reference.demand_probabilities.iter())
        {
            for (arrival_succeeds, yield_probability) in [
                (false, 1.0 - reference.success_probability),
                (true, reference.success_probability),
            ] {
                let branch_probability = probability * yield_probability;
                let (ending_inventory, next_pipeline) =
                    transition_inventory(&state, action, *demand, arrival_succeeds);
                let next_state = as_state_key(state.period + 1, ending_inventory, &next_pipeline);
                let continuation = solve_optimal_from_state(next_state, reference, cache);
                expected_cost += branch_probability
                    * (period_cost(
                        ending_inventory,
                        action,
                        reference.holding_cost,
                        reference.shortage_cost,
                        reference.procurement_cost,
                    ) + reference.discount_factor * continuation.discounted_cost);
            }
        }
        if expected_cost < best_cost - 1e-12 {
            best_cost = expected_cost;
            best_action = action;
        }
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: best_cost,
        first_action: best_action,
    };
    cache.insert(state, result);
    result
}

pub fn solve_optimal_policy(
    reference: &ExactVerificationReference,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let initial_state = as_state_key(
        0,
        reference.initial_inventory_level,
        reference.initial_pipeline_orders,
    );
    let mut cache = HashMap::new();
    Ok(solve_optimal_from_state(initial_state, reference, &mut cache))
}

fn evaluate_heuristic_from_state(
    state: ExactStateKey,
    reference: &ExactVerificationReference,
    heuristic_name: &str,
    cache: &mut HashMap<(ExactStateKey, &'static str), ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: 0,
        });
    }
    let cache_key = (
        state,
        match heuristic_name {
            "linear_inflation" => "linear_inflation",
            "weighted_newsvendor" => "weighted_newsvendor",
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported heuristic '{heuristic_name}'"
                )))
            }
        },
    );
    if let Some(cached) = cache.get(&cache_key) {
        return Ok(*cached);
    }

    let state_for_heuristic = initialize_state(
        state.inventory_level as f64,
        &[state.pipeline_orders[0] as f64, state.pipeline_orders[1] as f64],
    )?;
    let raw_action = match heuristic_name {
        "linear_inflation" => yield_inflated_base_stock_order_quantity(
            &state_for_heuristic,
            reference
                .demand_support
                .iter()
                .zip(reference.demand_probabilities.iter())
                .map(|(value, probability)| *value as f64 * probability)
                .sum(),
            reference.success_probability,
            reference.holding_cost,
            reference.shortage_cost,
        )?,
        "weighted_newsvendor" => weighted_newsvendor_order_quantity(
            &state_for_heuristic,
            reference
                .demand_support
                .iter()
                .zip(reference.demand_probabilities.iter())
                .map(|(value, probability)| *value as f64 * probability)
                .sum(),
            reference.success_probability,
            reference.holding_cost,
            reference.shortage_cost,
        )?,
        _ => unreachable!(),
    };
    let action = raw_action.round().clamp(0.0, reference.max_order_quantity as f64) as usize;

    let mut expected_cost = 0.0;
    for (demand, probability) in reference
        .demand_support
        .iter()
        .zip(reference.demand_probabilities.iter())
    {
        for (arrival_succeeds, yield_probability) in [
            (false, 1.0 - reference.success_probability),
            (true, reference.success_probability),
        ] {
            let branch_probability = probability * yield_probability;
            let (ending_inventory, next_pipeline) =
                transition_inventory(&state, action, *demand, arrival_succeeds);
            let next_state = as_state_key(state.period + 1, ending_inventory, &next_pipeline);
            let continuation =
                evaluate_heuristic_from_state(next_state, reference, heuristic_name, cache)?;
            expected_cost += branch_probability
                * (period_cost(
                    ending_inventory,
                    action,
                    reference.holding_cost,
                    reference.shortage_cost,
                    reference.procurement_cost,
                ) + reference.discount_factor * continuation.discounted_cost);
        }
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: expected_cost,
        first_action: action,
    };
    cache.insert(cache_key, result);
    Ok(result)
}

pub fn evaluate_named_heuristic(
    reference: &ExactVerificationReference,
    heuristic_name: &str,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let initial_state = as_state_key(
        0,
        reference.initial_inventory_level,
        reference.initial_pipeline_orders,
    );
    let mut cache = HashMap::new();
    evaluate_heuristic_from_state(initial_state, reference, heuristic_name, &mut cache)
}
