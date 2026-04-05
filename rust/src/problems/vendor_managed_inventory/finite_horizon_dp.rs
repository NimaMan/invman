use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::vendor_managed_inventory::env::{
    initialize_state, step_state, terminal_salvage_credit, VendorManagedInventoryState,
};
use crate::problems::vendor_managed_inventory::heuristics::{
    dc_reserve_base_stock_shipment_quantity, retailer_base_stock_shipment_quantity,
};
use crate::problems::vendor_managed_inventory::references::ExactVerificationReference;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactStateKey {
    period: usize,
    dc_on_hand: usize,
    retailer_on_hand: usize,
    retailer_pipeline: usize,
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
    if reference.demand_support.len() != reference.demand_probabilities.len() {
        return Err(PyValueError::new_err(
            "demand_support and demand_probabilities must have the same length",
        ));
    }
    if reference.demand_support.is_empty() {
        return Err(PyValueError::new_err("demand_support must be non-empty"));
    }
    let probability_sum = reference.demand_probabilities.iter().sum::<f64>();
    if (probability_sum - 1.0).abs() > 1e-12 {
        return Err(PyValueError::new_err(format!(
            "demand_probabilities must sum to 1, found {probability_sum}"
        )));
    }
    if !(0.0..=1.0).contains(&reference.discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }
    Ok(())
}

fn state_key_from_state(state: &VendorManagedInventoryState) -> ExactStateKey {
    ExactStateKey {
        period: state.period,
        dc_on_hand: state.dc_on_hand,
        retailer_on_hand: state.retailer_on_hand,
        retailer_pipeline: state.retailer_pipeline,
    }
}

fn state_from_key(
    state: &ExactStateKey,
    reference: &ExactVerificationReference,
) -> PyResult<VendorManagedInventoryState> {
    let mut rebuilt = initialize_state(
        state.dc_on_hand,
        state.retailer_on_hand,
        state.retailer_pipeline,
        reference.dc_capacity,
    )?;
    rebuilt.period = state.period;
    Ok(rebuilt)
}

fn terminal_cost(
    state: &VendorManagedInventoryState,
    reference: &ExactVerificationReference,
) -> PyResult<f64> {
    Ok(-terminal_salvage_credit(
        state,
        reference.dc_capacity,
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
            discounted_cost: terminal_cost(&state_from_key(&state, reference)?, reference)?,
            first_action: 0,
        });
    }
    if let Some(cached) = cache.get(&state) {
        return Ok(*cached);
    }

    let concrete_state = state_from_key(&state, reference)?;
    let mut best_cost = f64::INFINITY;
    let mut best_action = 0usize;
    for shipment_quantity in 0..=reference
        .max_shipment_quantity
        .min(concrete_state.dc_on_hand)
    {
        let mut expected_cost = 0.0;
        for (demand, probability) in reference
            .demand_support
            .iter()
            .zip(reference.demand_probabilities.iter())
        {
            let outcome = step_state(
                &concrete_state,
                shipment_quantity,
                *demand as usize,
                reference.dc_replenishment_quantity,
                reference.dc_capacity,
                reference.shipment_cost_per_unit,
                reference.dc_holding_cost_per_unit,
                reference.retailer_holding_cost_per_unit,
                reference.stockout_cost_per_unit,
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
            best_action = shipment_quantity;
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
        reference.initial_dc_on_hand,
        reference.initial_retailer_on_hand,
        reference.initial_retailer_pipeline,
        reference.dc_capacity,
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
            discounted_cost: terminal_cost(&state_from_key(&state, reference)?, reference)?,
            first_action: 0,
        });
    }

    let normalized_name = match heuristic_name {
        "retailer_base_stock" => "retailer_base_stock",
        "dc_reserve_base_stock" => "dc_reserve_base_stock",
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

    let concrete_state = state_from_key(&state, reference)?;
    let first_action = match normalized_name {
        "retailer_base_stock" => retailer_base_stock_shipment_quantity(
            &concrete_state,
            reference.retailer_base_stock_level,
            reference.max_shipment_quantity,
        )?,
        "dc_reserve_base_stock" => dc_reserve_base_stock_shipment_quantity(
            &concrete_state,
            reference.dc_reserve_base_stock_level,
            reference.dc_reserve_quantity,
            reference.max_shipment_quantity,
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
            first_action,
            *demand as usize,
            reference.dc_replenishment_quantity,
            reference.dc_capacity,
            reference.shipment_cost_per_unit,
            reference.dc_holding_cost_per_unit,
            reference.retailer_holding_cost_per_unit,
            reference.stockout_cost_per_unit,
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
    let initial_state = initialize_state(
        reference.initial_dc_on_hand,
        reference.initial_retailer_on_hand,
        reference.initial_retailer_pipeline,
        reference.dc_capacity,
    )?;
    let mut cache = HashMap::new();
    evaluate_heuristic_from_state(
        state_key_from_state(&initial_state),
        reference,
        heuristic_name,
        &mut cache,
    )
}
