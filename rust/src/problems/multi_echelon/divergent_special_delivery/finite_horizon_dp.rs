#![allow(dead_code)]

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::multi_echelon::env::{
    build_decision_state, parse_allocation_mode, parse_warehouse_base_stock_mode, AllocationMode,
    MultiEchelonState, WarehouseBaseStockMode, initialize_state, step_state,
};
use crate::problems::multi_echelon::references::ExactVerificationReference;
use crate::problems::multi_echelon::rollout::build_policy_features;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactStateKey {
    period: usize,
    warehouse_inventory: i32,
    warehouse_pipeline: Vec<u32>,
    retailer_inventory: Vec<i32>,
    retailer_pipeline: Vec<Vec<u32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactPolicyEvaluation {
    pub discounted_cost: f64,
    pub first_action: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct ExactSoftTreeConfig {
    pub flat_params: Vec<f32>,
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub include_period_feature: bool,
    pub warehouse_base_stock_mode: WarehouseBaseStockMode,
    pub allocation_mode: AllocationMode,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExactHeuristicKind {
    RegularBaseStock,
    EchelonBaseStock,
}

fn validate_exact_reference(reference: &ExactVerificationReference) -> PyResult<()> {
    if reference.num_retailers == 0 {
        return Err(PyValueError::new_err(
            "exact multi-echelon reference must contain at least one retailer",
        ));
    }
    if reference.initial_retailer_inventory.len() != reference.num_retailers {
        return Err(PyValueError::new_err(
            "initial_retailer_inventory length does not match num_retailers",
        ));
    }
    if reference.initial_retailer_pipeline.len() != reference.num_retailers {
        return Err(PyValueError::new_err(
            "initial_retailer_pipeline length does not match num_retailers",
        ));
    }
    if reference.demand_support.len() != reference.demand_probabilities.len() {
        return Err(PyValueError::new_err(
            "demand_support and demand_probabilities must have the same length",
        ));
    }
    let probability_sum = reference.demand_probabilities.iter().sum::<f64>();
    if (probability_sum - 1.0).abs() > 1e-12 {
        return Err(PyValueError::new_err(format!(
            "demand_probabilities must sum to 1, found {probability_sum}"
        )));
    }
    parse_warehouse_base_stock_mode(reference.warehouse_base_stock_mode)?;
    parse_allocation_mode(reference.allocation_mode)?;
    Ok(())
}

fn initial_state(reference: &ExactVerificationReference) -> PyResult<MultiEchelonState> {
    initialize_state(
        reference.initial_warehouse_inventory,
        reference.initial_warehouse_pipeline,
        reference.initial_retailer_inventory,
        &reference
            .initial_retailer_pipeline
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )
}

fn as_state_key(state: &MultiEchelonState) -> ExactStateKey {
    ExactStateKey {
        period: state.period,
        warehouse_inventory: state.warehouse_inventory,
        warehouse_pipeline: state.warehouse_pipeline.clone(),
        retailer_inventory: state.retailer_inventory.clone(),
        retailer_pipeline: state.retailer_pipeline.clone(),
    }
}

fn from_state_key(state_key: &ExactStateKey) -> PyResult<MultiEchelonState> {
    let mut state = initialize_state(
        state_key.warehouse_inventory,
        &state_key.warehouse_pipeline,
        &state_key.retailer_inventory,
        &state_key.retailer_pipeline,
    )?;
    state.period = state_key.period;
    Ok(state)
}

fn enumerate_demand_combinations(
    num_retailers: usize,
    demand_support: &[u32],
    demand_probabilities: &[f64],
) -> Vec<(Vec<u32>, f64)> {
    fn recurse(
        num_retailers: usize,
        demand_support: &[u32],
        demand_probabilities: &[f64],
        retailer_idx: usize,
        current: &mut Vec<u32>,
        current_probability: f64,
        output: &mut Vec<(Vec<u32>, f64)>,
    ) {
        if retailer_idx == num_retailers {
            output.push((current.clone(), current_probability));
            return;
        }
        for (value, probability) in demand_support.iter().zip(demand_probabilities.iter()) {
            current.push(*value);
            recurse(
                num_retailers,
                demand_support,
                demand_probabilities,
                retailer_idx + 1,
                current,
                current_probability * probability,
                output,
            );
            current.pop();
        }
    }

    let mut output = Vec::new();
    recurse(
        num_retailers,
        demand_support,
        demand_probabilities,
        0,
        &mut Vec::new(),
        1.0,
        &mut output,
    );
    output
}

fn binomial_probability(n: usize, k: usize, p: f64) -> f64 {
    if k > n {
        return 0.0;
    }
    let combinations = (0..k).fold(1.0, |acc, idx| {
        acc * (n - idx) as f64 / (idx + 1) as f64
    });
    combinations * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
}

fn total_unmet_without_emergency(
    state: &MultiEchelonState,
    realized_demands: &[u32],
) -> PyResult<usize> {
    let decision_state = build_decision_state(state)?;
    Ok(realized_demands
        .iter()
        .enumerate()
        .map(|(retailer_idx, demand)| {
            let served =
                (*demand).min(decision_state.retailer_available[retailer_idx].max(0) as u32);
            (*demand - served) as usize
        })
        .sum::<usize>())
}

fn solve_optimal_from_state(
    state_key: ExactStateKey,
    reference: &ExactVerificationReference,
    warehouse_base_stock_mode: WarehouseBaseStockMode,
    allocation_mode: AllocationMode,
    demand_combinations: &[(Vec<u32>, f64)],
    cache: &mut HashMap<ExactStateKey, ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state_key.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: vec![0, 0],
        });
    }
    if let Some(cached) = cache.get(&state_key) {
        return Ok(cached.clone());
    }

    let state = from_state_key(&state_key)?;
    let mut best_cost = f64::INFINITY;
    let mut best_action = vec![0usize, 0usize];

    for warehouse_level in reference.action_warehouse_levels.iter().copied() {
        for retailer_level in reference.action_retailer_levels.iter().copied() {
            let mut expected_cost = 0.0;
            for (demands, demand_probability) in demand_combinations.iter() {
                let total_unmet = total_unmet_without_emergency(&state, demands)?;
                for accepted_emergency_shipments in 0..=total_unmet {
                    let acceptance_probability = binomial_probability(
                        total_unmet,
                        accepted_emergency_shipments,
                        reference.expedited_service_prob,
                    );
                    if acceptance_probability <= 0.0 {
                        continue;
                    }
                    let outcome = step_state(
                        &state,
                        warehouse_level,
                        retailer_level,
                        demands,
                        accepted_emergency_shipments,
                        reference.warehouse_capacity,
                        reference.warehouse_inventory_cap,
                        reference.retailer_inventory_cap,
                        reference.warehouse_holding_cost,
                        reference.retailer_holding_cost,
                        reference.warehouse_expedited_cost,
                        reference.warehouse_lost_sale_cost,
                        warehouse_base_stock_mode,
                        allocation_mode,
                    )?;
                    let continuation = solve_optimal_from_state(
                        as_state_key(&outcome.next_state),
                        reference,
                        warehouse_base_stock_mode,
                        allocation_mode,
                        demand_combinations,
                        cache,
                    )?;
                    expected_cost += demand_probability
                        * acceptance_probability
                        * (outcome.period_cost
                            + reference.discount_factor * continuation.discounted_cost);
                }
            }
            if expected_cost < best_cost - 1e-12 {
                best_cost = expected_cost;
                best_action = vec![warehouse_level, retailer_level];
            }
        }
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: best_cost,
        first_action: best_action,
    };
    cache.insert(state_key, result.clone());
    Ok(result)
}

fn evaluate_heuristic_from_state(
    state_key: ExactStateKey,
    reference: &ExactVerificationReference,
    demand_combinations: &[(Vec<u32>, f64)],
    heuristic_kind: ExactHeuristicKind,
    allocation_mode: AllocationMode,
    warehouse_level: usize,
    retailer_level: usize,
    cache: &mut HashMap<(ExactStateKey, ExactHeuristicKind, AllocationMode, usize, usize), ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state_key.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: vec![warehouse_level, retailer_level],
        });
    }
    let cache_key = (
        state_key.clone(),
        heuristic_kind,
        allocation_mode,
        warehouse_level,
        retailer_level,
    );
    if let Some(cached) = cache.get(&cache_key) {
        return Ok(cached.clone());
    }

    let state = from_state_key(&state_key)?;
    let warehouse_base_stock_mode = match heuristic_kind {
        ExactHeuristicKind::RegularBaseStock => WarehouseBaseStockMode::Regular,
        ExactHeuristicKind::EchelonBaseStock => WarehouseBaseStockMode::Echelon,
    };

    let mut expected_cost = 0.0;
    for (demands, demand_probability) in demand_combinations.iter() {
        let total_unmet = total_unmet_without_emergency(&state, demands)?;
        for accepted_emergency_shipments in 0..=total_unmet {
            let acceptance_probability = binomial_probability(
                total_unmet,
                accepted_emergency_shipments,
                reference.expedited_service_prob,
            );
            if acceptance_probability <= 0.0 {
                continue;
            }
            let outcome = step_state(
                &state,
                warehouse_level,
                retailer_level,
                demands,
                accepted_emergency_shipments,
                reference.warehouse_capacity,
                reference.warehouse_inventory_cap,
                reference.retailer_inventory_cap,
                reference.warehouse_holding_cost,
                reference.retailer_holding_cost,
                reference.warehouse_expedited_cost,
                reference.warehouse_lost_sale_cost,
                warehouse_base_stock_mode,
                allocation_mode,
            )?;
            let continuation = evaluate_heuristic_from_state(
                as_state_key(&outcome.next_state),
                reference,
                demand_combinations,
                heuristic_kind,
                allocation_mode,
                warehouse_level,
                retailer_level,
                cache,
            )?;
            expected_cost += demand_probability
                * acceptance_probability
                * (outcome.period_cost + reference.discount_factor * continuation.discounted_cost);
        }
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: expected_cost,
        first_action: vec![warehouse_level, retailer_level],
    };
    cache.insert(cache_key, result.clone());
    Ok(result)
}

fn soft_tree_action(
    state: &MultiEchelonState,
    reference: &ExactVerificationReference,
    config: &ExactSoftTreeConfig,
) -> PyResult<Vec<usize>> {
    let policy_state = build_policy_features(
        state,
        reference.warehouse_inventory_cap,
        reference.retailer_inventory_cap,
        config.include_period_feature,
        reference.periods,
    )?;
    if policy_state.len() != config.input_dim {
        return Err(PyValueError::new_err(
            "policy state length does not match input_dim",
        ));
    }
    action_vector_from_flat_params(
        &policy_state,
        &config.flat_params,
        config.input_dim,
        config.depth,
        config.temperature,
        config.split_type,
        config.leaf_type,
        &config.action_spec,
    )
}

fn evaluate_soft_tree_from_state(
    state_key: ExactStateKey,
    reference: &ExactVerificationReference,
    config: &ExactSoftTreeConfig,
    demand_combinations: &[(Vec<u32>, f64)],
    cache: &mut HashMap<ExactStateKey, ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state_key.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: vec![0, 0],
        });
    }
    if let Some(cached) = cache.get(&state_key) {
        return Ok(cached.clone());
    }

    let state = from_state_key(&state_key)?;
    let action = soft_tree_action(&state, reference, config)?;
    if action.len() != 2 {
        return Err(PyValueError::new_err(
            "multi-echelon soft-tree exact evaluation expects a 2D action",
        ));
    }

    let mut expected_cost = 0.0;
    for (demands, demand_probability) in demand_combinations.iter() {
        let total_unmet = total_unmet_without_emergency(&state, demands)?;
        for accepted_emergency_shipments in 0..=total_unmet {
            let acceptance_probability = binomial_probability(
                total_unmet,
                accepted_emergency_shipments,
                reference.expedited_service_prob,
            );
            if acceptance_probability <= 0.0 {
                continue;
            }
            let outcome = step_state(
                &state,
                action[0],
                action[1],
                demands,
                accepted_emergency_shipments,
                reference.warehouse_capacity,
                reference.warehouse_inventory_cap,
                reference.retailer_inventory_cap,
                reference.warehouse_holding_cost,
                reference.retailer_holding_cost,
                reference.warehouse_expedited_cost,
                reference.warehouse_lost_sale_cost,
                config.warehouse_base_stock_mode,
                config.allocation_mode,
            )?;
            let continuation = evaluate_soft_tree_from_state(
                as_state_key(&outcome.next_state),
                reference,
                config,
                demand_combinations,
                cache,
            )?;
            expected_cost += demand_probability
                * acceptance_probability
                * (outcome.period_cost + reference.discount_factor * continuation.discounted_cost);
        }
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: expected_cost,
        first_action: action,
    };
    cache.insert(state_key, result.clone());
    Ok(result)
}

pub fn solve_optimal_policy(reference: &ExactVerificationReference) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let demand_combinations = enumerate_demand_combinations(
        reference.num_retailers,
        reference.demand_support,
        reference.demand_probabilities,
    );
    let mut cache = HashMap::new();
    solve_optimal_from_state(
        as_state_key(&initial_state(reference)?),
        reference,
        parse_warehouse_base_stock_mode(reference.warehouse_base_stock_mode)?,
        parse_allocation_mode(reference.allocation_mode)?,
        &demand_combinations,
        &mut cache,
    )
}

pub fn evaluate_stationary_policy(
    reference: &ExactVerificationReference,
    heuristic_kind: ExactHeuristicKind,
    allocation_mode: AllocationMode,
    warehouse_level: usize,
    retailer_level: usize,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let demand_combinations = enumerate_demand_combinations(
        reference.num_retailers,
        reference.demand_support,
        reference.demand_probabilities,
    );
    let mut cache = HashMap::new();
    evaluate_heuristic_from_state(
        as_state_key(&initial_state(reference)?),
        reference,
        &demand_combinations,
        heuristic_kind,
        allocation_mode,
        warehouse_level,
        retailer_level,
        &mut cache,
    )
}

pub fn search_best_stationary_policy(
    reference: &ExactVerificationReference,
    heuristic_kind: ExactHeuristicKind,
    allocation_mode: AllocationMode,
) -> PyResult<(usize, usize, ExactPolicyEvaluation)> {
    let mut best_result: Option<(usize, usize, ExactPolicyEvaluation)> = None;
    for warehouse_level in reference.action_warehouse_levels.iter().copied() {
        for retailer_level in reference.action_retailer_levels.iter().copied() {
            let evaluation = evaluate_stationary_policy(
                reference,
                heuristic_kind,
                allocation_mode,
                warehouse_level,
                retailer_level,
            )?;
            match &best_result {
                Some((_, _, current_best))
                    if evaluation.discounted_cost >= current_best.discounted_cost - 1e-12 => {}
                _ => best_result = Some((warehouse_level, retailer_level, evaluation)),
            }
        }
    }
    best_result.ok_or_else(|| PyValueError::new_err("failed to search stationary policy"))
}

pub fn evaluate_soft_tree_policy(
    reference: &ExactVerificationReference,
    config: &ExactSoftTreeConfig,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let demand_combinations = enumerate_demand_combinations(
        reference.num_retailers,
        reference.demand_support,
        reference.demand_probabilities,
    );
    let mut cache = HashMap::new();
    evaluate_soft_tree_from_state(
        as_state_key(&initial_state(reference)?),
        reference,
        config,
        &demand_combinations,
        &mut cache,
    )
}
