use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::joint_pricing_inventory::env::{
    initialize_state, step_state, terminal_salvage_credit, JointPricingInventoryState,
};
use crate::problems::joint_pricing_inventory::heuristics::{
    inventory_sensitive_base_stock_action, static_price_base_stock_action,
};
use crate::problems::joint_pricing_inventory::references::ExactVerificationReference;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactStateKey {
    period: usize,
    inventory_level: usize,
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
    if reference.price_levels.is_empty() {
        return Err(PyValueError::new_err("price_levels must be non-empty"));
    }
    if reference.price_levels.len() != reference.price_demand_supports.len()
        || reference.price_levels.len() != reference.price_demand_probabilities.len()
    {
        return Err(PyValueError::new_err(
            "price_levels, price_demand_supports, and price_demand_probabilities must have the same length",
        ));
    }
    for (support, probabilities) in reference
        .price_demand_supports
        .iter()
        .zip(reference.price_demand_probabilities.iter())
    {
        if support.len() != probabilities.len() {
            return Err(PyValueError::new_err(
                "each demand support must match its probability vector length",
            ));
        }
        let probability_sum = probabilities.iter().sum::<f64>();
        if (probability_sum - 1.0).abs() > 1e-12 {
            return Err(PyValueError::new_err(format!(
                "demand probabilities must sum to 1, found {probability_sum}"
            )));
        }
    }
    if !(0.0..=1.0).contains(&reference.discount_factor) {
        return Err(PyValueError::new_err(
            "discount_factor must lie in [0, 1]",
        ));
    }
    Ok(())
}

fn state_key_from_state(state: &JointPricingInventoryState) -> ExactStateKey {
    ExactStateKey {
        period: state.period,
        inventory_level: state.inventory_level,
    }
}

fn state_from_key(state: &ExactStateKey) -> PyResult<JointPricingInventoryState> {
    let mut rebuilt = initialize_state(state.inventory_level)?;
    rebuilt.period = state.period;
    Ok(rebuilt)
}

fn terminal_cost(
    state: &JointPricingInventoryState,
    reference: &ExactVerificationReference,
) -> PyResult<f64> {
    Ok(-terminal_salvage_credit(
        state,
        reference.salvage_value_per_unit,
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
    for order_quantity in 0..=reference.max_order_quantity {
        for price_index in 0..reference.price_levels.len() {
            let mut expected_cost = 0.0;
            for (demand, probability) in reference.price_demand_supports[price_index]
                .iter()
                .zip(reference.price_demand_probabilities[price_index].iter())
            {
                let outcome = step_state(
                    &concrete_state,
                    order_quantity,
                    price_index,
                    *demand as usize,
                    reference.price_levels,
                    reference.procurement_cost_per_unit,
                    reference.holding_cost_per_unit,
                    reference.stockout_cost_per_unit,
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
                best_action = (order_quantity, price_index);
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
    let initial_state = initialize_state(reference.initial_inventory_level)?;
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
        "static_price_base_stock" => "static_price_base_stock",
        "inventory_sensitive_base_stock" => "inventory_sensitive_base_stock",
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
        "static_price_base_stock" => static_price_base_stock_action(
            concrete_state.inventory_level,
            reference.static_order_up_to,
            reference.static_price_index,
            reference.max_order_quantity,
            reference.price_levels.len(),
        )?,
        "inventory_sensitive_base_stock" => inventory_sensitive_base_stock_action(
            concrete_state.inventory_level,
            reference.inventory_sensitive_order_up_to,
            reference.markdown_threshold,
            reference.high_price_index,
            reference.low_price_index,
            reference.max_order_quantity,
            reference.price_levels.len(),
        )?,
        _ => unreachable!(),
    };

    let price_index = first_action.1;
    let mut expected_cost = 0.0;
    for (demand, probability) in reference.price_demand_supports[price_index]
        .iter()
        .zip(reference.price_demand_probabilities[price_index].iter())
    {
        let outcome = step_state(
            &concrete_state,
            first_action.0,
            first_action.1,
            *demand as usize,
            reference.price_levels,
            reference.procurement_cost_per_unit,
            reference.holding_cost_per_unit,
            reference.stockout_cost_per_unit,
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
    let initial_state = initialize_state(reference.initial_inventory_level)?;
    let mut cache = HashMap::new();
    evaluate_heuristic_from_state(
        state_key_from_state(&initial_state),
        reference,
        heuristic_name,
        &mut cache,
    )
}
