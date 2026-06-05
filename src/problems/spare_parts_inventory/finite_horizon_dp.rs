use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::spare_parts_inventory::demand::failure_probabilities;
use crate::problems::spare_parts_inventory::env::{
    initialize_state, operational_units, step_state, SparePartsInventoryState,
};
use crate::problems::spare_parts_inventory::heuristics::{
    base_stock_order_quantity, lead_time_mean_cover_order_quantity,
};
use crate::problems::spare_parts_inventory::references::ExactVerificationReference;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactStateKey {
    period: usize,
    on_hand_inventory: usize,
    backlog: usize,
    procurement_pipeline: Vec<usize>,
    repair_pipeline: Vec<usize>,
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
    if reference.procurement_lead_time == 0 || reference.repair_lead_time == 0 {
        return Err(PyValueError::new_err(
            "procurement_lead_time and repair_lead_time must be at least 1",
        ));
    }
    if reference.initial_procurement_pipeline.len() != reference.procurement_lead_time {
        return Err(PyValueError::new_err(
            "initial_procurement_pipeline length does not match procurement_lead_time",
        ));
    }
    if reference.initial_repair_pipeline.len() != reference.repair_lead_time {
        return Err(PyValueError::new_err(
            "initial_repair_pipeline length does not match repair_lead_time",
        ));
    }
    Ok(())
}

fn state_key_from_state(state: &SparePartsInventoryState) -> ExactStateKey {
    ExactStateKey {
        period: state.period,
        on_hand_inventory: state.on_hand_inventory,
        backlog: state.backlog,
        procurement_pipeline: state.procurement_pipeline.clone(),
        repair_pipeline: state.repair_pipeline.clone(),
    }
}

fn state_from_key(
    state: &ExactStateKey,
    reference: &ExactVerificationReference,
) -> PyResult<SparePartsInventoryState> {
    let mut rebuilt = initialize_state(
        state.on_hand_inventory,
        state.backlog,
        &state.procurement_pipeline,
        &state.repair_pipeline,
        reference.installed_base,
    )?;
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

    let concrete_state = state_from_key(&state, reference)?;
    let failure_distribution = failure_probabilities(
        operational_units(&concrete_state, reference.installed_base)?,
        reference.failure_probability,
    )?;

    let mut best_cost = f64::INFINITY;
    let mut best_action = 0usize;
    for action in 0..=reference.max_order_quantity {
        let mut expected_cost = 0.0;
        for (failures, probability) in failure_distribution.iter().enumerate() {
            if *probability <= 0.0 {
                continue;
            }
            let outcome = step_state(
                &concrete_state,
                action,
                failures,
                reference.installed_base,
                reference.holding_cost,
                reference.downtime_cost,
                reference.procurement_cost,
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
    let initial_state = initialize_state(
        reference.initial_on_hand_inventory,
        reference.initial_backlog,
        reference.initial_procurement_pipeline,
        reference.initial_repair_pipeline,
        reference.installed_base,
    )?;
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
    let normalized_policy_name = match heuristic_name {
        "base_stock" => "base_stock",
        "lead_time_mean_cover" => "lead_time_mean_cover",
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported heuristic '{heuristic_name}'"
            )))
        }
    };
    let cache_key = (state.clone(), normalized_policy_name);
    if let Some(cached) = cache.get(&cache_key) {
        return Ok(*cached);
    }

    let concrete_state = state_from_key(&state, reference)?;
    let action = match normalized_policy_name {
        "base_stock" => base_stock_order_quantity(&concrete_state, reference.base_stock_level)?,
        "lead_time_mean_cover" => lead_time_mean_cover_order_quantity(
            &concrete_state,
            reference.installed_base,
            reference.failure_probability,
            reference.lead_time_mean_cover_safety_buffer,
        )?,
        _ => unreachable!(),
    }
    .min(reference.max_order_quantity);

    let failure_distribution = failure_probabilities(
        operational_units(&concrete_state, reference.installed_base)?,
        reference.failure_probability,
    )?;
    let mut expected_cost = 0.0;
    for (failures, probability) in failure_distribution.iter().enumerate() {
        if *probability <= 0.0 {
            continue;
        }
        let outcome = step_state(
            &concrete_state,
            action,
            failures,
            reference.installed_base,
            reference.holding_cost,
            reference.downtime_cost,
            reference.procurement_cost,
        )?;
        let continuation = evaluate_heuristic_from_state(
            state_key_from_state(&outcome.next_state),
            reference,
            normalized_policy_name,
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
    let initial_state = initialize_state(
        reference.initial_on_hand_inventory,
        reference.initial_backlog,
        reference.initial_procurement_pipeline,
        reference.initial_repair_pipeline,
        reference.installed_base,
    )?;
    let mut cache = HashMap::new();
    evaluate_heuristic_from_state(
        state_key_from_state(&initial_state),
        reference,
        heuristic_name,
        &mut cache,
    )
}
