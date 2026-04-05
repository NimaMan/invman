use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::procurement_removal_inventory::env::{
    initialize_state, step_state, terminal_salvage_credit, ProcurementRemovalState,
};
use crate::problems::procurement_removal_inventory::heuristics::{
    interval_stock_action, returnability_buffer_interval_stock_action,
};
use crate::problems::procurement_removal_inventory::references::ExactVerificationReference;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactStateKey {
    period: usize,
    inventory_level: usize,
    returnable_inventory: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactPolicyEvaluation {
    pub discounted_cost: f64,
    pub first_action: (usize, usize),
}

fn validate_exact_reference(reference: &ExactVerificationReference) -> PyResult<()> {
    if reference.periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if reference.demand_support.len() != reference.demand_probabilities.len() {
        return Err(PyValueError::new_err(
            "demand_support and demand_probabilities must have the same length",
        ));
    }
    if reference.demand_support.is_empty() {
        return Err(PyValueError::new_err(
            "demand_support must be non-empty",
        ));
    }
    let probability_sum = reference.demand_probabilities.iter().sum::<f64>();
    if (probability_sum - 1.0).abs() > 1e-12 {
        return Err(PyValueError::new_err(format!(
            "demand_probabilities must sum to 1, found {probability_sum}"
        )));
    }
    if !(0.0..=1.0).contains(&reference.discount_factor) {
        return Err(PyValueError::new_err(
            "discount_factor must lie in [0, 1]",
        ));
    }
    Ok(())
}

fn state_key_from_state(state: &ProcurementRemovalState) -> ExactStateKey {
    ExactStateKey {
        period: state.period,
        inventory_level: state.inventory_level,
        returnable_inventory: state.returnable_inventory,
    }
}

fn state_from_key(state: &ExactStateKey) -> PyResult<ProcurementRemovalState> {
    let mut rebuilt = initialize_state(state.inventory_level, state.returnable_inventory)?;
    rebuilt.period = state.period;
    Ok(rebuilt)
}

fn terminal_cost(
    state: &ProcurementRemovalState,
    reference: &ExactVerificationReference,
) -> PyResult<f64> {
    Ok(-terminal_salvage_credit(
        state,
        reference.return_value_per_unit,
        reference.liquidation_value_per_unit,
    )?)
}

fn solve_optimal_from_state(
    state: ExactStateKey,
    reference: &ExactVerificationReference,
    cache: &mut HashMap<ExactStateKey, ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: terminal_cost(&state_from_key(&state)?, reference)?,
            first_action: (0, 0),
        });
    }
    if let Some(cached) = cache.get(&state) {
        return Ok(*cached);
    }

    let concrete_state = state_from_key(&state)?;
    let mut best_cost = f64::INFINITY;
    let mut best_action = (0usize, 0usize);
    for purchase_quantity in 0..=reference.max_purchase_quantity {
        let removal_limit = reference
            .max_removal_quantity
            .min(concrete_state.inventory_level + purchase_quantity);
        for removal_quantity in 0..=removal_limit {
            let mut expected_cost = 0.0;
            for (demand, probability) in reference
                .demand_support
                .iter()
                .zip(reference.demand_probabilities.iter())
            {
                let outcome = step_state(
                    &concrete_state,
                    purchase_quantity,
                    removal_quantity,
                    *demand as usize,
                    reference.returnable_purchase_cap,
                    reference.purchase_cost_per_unit,
                    reference.return_value_per_unit,
                    reference.liquidation_value_per_unit,
                    reference.holding_cost_per_unit,
                    reference.shortage_cost_per_unit,
                )?;
                let continuation = solve_optimal_from_state(
                    state_key_from_state(&outcome.next_state),
                    reference,
                    cache,
                )?;
                expected_cost += probability
                    * (outcome.period_cost
                        + reference.discount_factor * continuation.discounted_cost);
            }
            if expected_cost < best_cost - 1e-12 {
                best_cost = expected_cost;
                best_action = (purchase_quantity, removal_quantity);
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
    let initial_state =
        initialize_state(reference.initial_inventory_level, reference.initial_returnable_inventory)?;
    let mut cache = HashMap::new();
    solve_optimal_from_state(state_key_from_state(&initial_state), reference, &mut cache)
}

fn evaluate_heuristic_from_state(
    state: ExactStateKey,
    reference: &ExactVerificationReference,
    heuristic_name: &str,
    cache: &mut HashMap<(ExactStateKey, &'static str), ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: terminal_cost(&state_from_key(&state)?, reference)?,
            first_action: (0, 0),
        });
    }

    let normalized_name = match heuristic_name {
        "interval_stock" => "interval_stock",
        "returnability_buffer_interval_stock" => "returnability_buffer_interval_stock",
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported heuristic '{heuristic_name}'"
            )))
        }
    };
    let cache_key = (state.clone(), normalized_name);
    if let Some(cached) = cache.get(&cache_key) {
        return Ok(*cached);
    }

    let concrete_state = state_from_key(&state)?;
    let first_action = match normalized_name {
        "interval_stock" => interval_stock_action(
            &concrete_state,
            reference.interval_stock_order_up_to,
            reference.interval_stock_remove_down_to,
            reference.max_purchase_quantity,
            reference.max_removal_quantity,
        )?,
        "returnability_buffer_interval_stock" => returnability_buffer_interval_stock_action(
            &concrete_state,
            reference.returnability_buffer_order_up_to,
            reference.returnability_buffer_remove_down_to,
            reference.returnability_buffer,
            reference.max_purchase_quantity,
            reference.max_removal_quantity,
        )?,
        _ => unreachable!(),
    };

    let mut expected_cost = 0.0;
    for (demand, probability) in reference
        .demand_support
        .iter()
        .zip(reference.demand_probabilities.iter())
    {
        let outcome = step_state(
            &concrete_state,
            first_action.0,
            first_action.1,
            *demand as usize,
            reference.returnable_purchase_cap,
            reference.purchase_cost_per_unit,
            reference.return_value_per_unit,
            reference.liquidation_value_per_unit,
            reference.holding_cost_per_unit,
            reference.shortage_cost_per_unit,
        )?;
        let continuation = evaluate_heuristic_from_state(
            state_key_from_state(&outcome.next_state),
            reference,
            normalized_name,
            cache,
        )?;
        expected_cost += probability
            * (outcome.period_cost
                + reference.discount_factor * continuation.discounted_cost);
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: expected_cost,
        first_action,
    };
    cache.insert(cache_key, result);
    Ok(result)
}

pub fn evaluate_named_heuristic(
    reference: &ExactVerificationReference,
    heuristic_name: &str,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let initial_state =
        initialize_state(reference.initial_inventory_level, reference.initial_returnable_inventory)?;
    let mut cache = HashMap::new();
    evaluate_heuristic_from_state(
        state_key_from_state(&initial_state),
        reference,
        heuristic_name,
        &mut cache,
    )
}
