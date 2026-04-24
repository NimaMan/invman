use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::ameliorating_inventory::env::{
    initialize_state, step_state, AmelioratingInventoryState,
};
use crate::problems::ameliorating_inventory::heuristics::{
    newsvendor_purchase_order_quantity, two_dimensional_order_up_to_order_quantity,
};
use crate::problems::ameliorating_inventory::literature::ExactVerificationReference;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactStateKey {
    period: usize,
    inventory_by_age: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactPolicyEvaluation {
    pub discounted_cost: f64,
    pub first_action: usize,
}

fn validate_exact_reference(reference: &ExactVerificationReference) -> PyResult<()> {
    if reference.periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if reference.target_ages.len() != reference.product_prices.len() {
        return Err(PyValueError::new_err(
            "target_ages and product_prices must have the same length",
        ));
    }
    if reference.age_retention.len() != reference.initial_inventory_by_age.len()
        || reference.decay_salvage_values.len() != reference.initial_inventory_by_age.len()
    {
        return Err(PyValueError::new_err(
            "age_retention and decay_salvage_values must match the number of age classes",
        ));
    }
    if reference.demand_scenarios.len() != reference.demand_probabilities.len() {
        return Err(PyValueError::new_err(
            "demand_scenarios and demand_probabilities must have the same length",
        ));
    }
    let total_probability = reference.demand_probabilities.iter().sum::<f64>();
    if (total_probability - 1.0).abs() > 1e-12 {
        return Err(PyValueError::new_err(format!(
            "demand_probabilities must sum to 1, found {total_probability}"
        )));
    }
    if reference
        .demand_scenarios
        .iter()
        .any(|scenario| scenario.len() != reference.target_ages.len())
    {
        return Err(PyValueError::new_err(
            "each demand scenario must match the number of products",
        ));
    }
    Ok(())
}

fn state_key_from_state(state: &AmelioratingInventoryState) -> ExactStateKey {
    ExactStateKey {
        period: state.period,
        inventory_by_age: state.inventory_by_age.clone(),
    }
}

fn state_from_key(state: &ExactStateKey) -> PyResult<AmelioratingInventoryState> {
    let mut rebuilt = initialize_state(&state.inventory_by_age)?;
    rebuilt.period = state.period;
    Ok(rebuilt)
}

fn solve_optimal_from_state(
    state: ExactStateKey,
    reference: &ExactVerificationReference,
    cache: &mut HashMap<ExactStateKey, ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: 0,
        });
    }
    if let Some(cached) = cache.get(&state) {
        return Ok(*cached);
    }

    let concrete_state = state_from_key(&state)?;
    let mut best_cost = f64::INFINITY;
    let mut best_action = 0usize;
    for action in 0..=reference.max_purchase_quantity {
        let mut expected_cost = 0.0;
        for (scenario, probability) in reference
            .demand_scenarios
            .iter()
            .zip(reference.demand_probabilities.iter())
        {
            let outcome = step_state(
                &concrete_state,
                action,
                &scenario
                    .iter()
                    .map(|value| *value as usize)
                    .collect::<Vec<_>>(),
                reference.target_ages,
                reference.product_prices,
                reference.age_retention,
                reference.purchase_cost_per_unit,
                reference.holding_cost_per_unit,
                reference.decay_salvage_values,
            )?;
            let continuation = solve_optimal_from_state(
                state_key_from_state(&outcome.next_state),
                reference,
                cache,
            )?;
            expected_cost += probability
                * (outcome.period_cost + reference.discount_factor * continuation.discounted_cost);
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
    Ok(result)
}

pub fn solve_optimal_policy(
    reference: &ExactVerificationReference,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let initial_state = initialize_state(reference.initial_inventory_by_age)?;
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
            discounted_cost: 0.0,
            first_action: 0,
        });
    }

    let normalized_name = match heuristic_name {
        "newsvendor_purchase" => "newsvendor_purchase",
        "two_dimensional_order_up_to" => "two_dimensional_order_up_to",
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
    let action = match normalized_name {
        "newsvendor_purchase" => {
            newsvendor_purchase_order_quantity(&concrete_state, reference.newsvendor_total_target)?
        }
        "two_dimensional_order_up_to" => two_dimensional_order_up_to_order_quantity(
            &concrete_state,
            reference.two_dimensional_total_target,
            reference.two_dimensional_young_target,
            reference.young_age_cutoff,
        )?,
        _ => unreachable!(),
    }
    .min(reference.max_purchase_quantity);

    let mut expected_cost = 0.0;
    for (scenario, probability) in reference
        .demand_scenarios
        .iter()
        .zip(reference.demand_probabilities.iter())
    {
        let outcome = step_state(
            &concrete_state,
            action,
            &scenario
                .iter()
                .map(|value| *value as usize)
                .collect::<Vec<_>>(),
            reference.target_ages,
            reference.product_prices,
            reference.age_retention,
            reference.purchase_cost_per_unit,
            reference.holding_cost_per_unit,
            reference.decay_salvage_values,
        )?;
        let continuation = evaluate_heuristic_from_state(
            state_key_from_state(&outcome.next_state),
            reference,
            normalized_name,
            cache,
        )?;
        expected_cost += probability
            * (outcome.period_cost + reference.discount_factor * continuation.discounted_cost);
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
    let initial_state = initialize_state(reference.initial_inventory_by_age)?;
    let mut cache = HashMap::new();
    evaluate_heuristic_from_state(
        state_key_from_state(&initial_state),
        reference,
        heuristic_name,
        &mut cache,
    )
}
