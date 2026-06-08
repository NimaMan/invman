use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::one_warehouse_multi_retailer::allocation::{
    min_shortage_shipments, proportional_shipments, AllocationPolicy,
};
use crate::problems::one_warehouse_multi_retailer::env::{
    retailer_inventory_positions, step_state, validate_state, warehouse_echelon_inventory_position,
    OneWarehouseMultiRetailerState,
};
use crate::problems::one_warehouse_multi_retailer::heuristics::echelon_base_stock_orders;
use crate::problems::one_warehouse_multi_retailer::references::ExactVerificationReference;
use crate::problems::one_warehouse_multi_retailer::rollout::{PolicyActionMode, PolicyStateMode};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactStateKey {
    period: usize,
    warehouse_inventory: i32,
    warehouse_pipeline: Vec<usize>,
    retailer_inventory: Vec<i32>,
    retailer_pipeline: Vec<Vec<usize>>,
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
    pub allocation_policy: AllocationPolicy,
    pub policy_action_mode: PolicyActionMode,
    pub policy_state_mode: PolicyStateMode,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

fn validate_exact_reference(reference: &ExactVerificationReference) -> PyResult<()> {
    let num_retailers = reference.retailer_lead_times.len();
    if num_retailers == 0 {
        return Err(PyValueError::new_err(
            "exact verification requires at least one retailer",
        ));
    }
    if reference.initial_warehouse_pipeline.len() != reference.warehouse_lead_time
        || reference.initial_retailer_inventory.len() != num_retailers
        || reference.initial_retailer_pipeline.len() != num_retailers
        || reference.holding_cost_retailers.len() != num_retailers
        || reference.penalty_costs_retailers.len() != num_retailers
        || reference.heuristic_retailer_base_stock_levels.len() != num_retailers
        || reference.demand_supports.len() != num_retailers
        || reference.demand_probabilities.len() != num_retailers
        || reference.max_action_levels.len() != num_retailers + 1
    {
        return Err(PyValueError::new_err(
            "reference arrays must match the number of retailers and lead times",
        ));
    }
    for retailer_idx in 0..num_retailers {
        if reference.initial_retailer_pipeline[retailer_idx].len()
            != reference.retailer_lead_times[retailer_idx]
        {
            return Err(PyValueError::new_err(format!(
                "initial retailer pipeline {} length does not match the retailer lead time",
                retailer_idx
            )));
        }
        if reference.demand_supports[retailer_idx].len()
            != reference.demand_probabilities[retailer_idx].len()
        {
            return Err(PyValueError::new_err(format!(
                "demand support/probability lengths do not match for retailer {}",
                retailer_idx
            )));
        }
        let probability_sum = reference.demand_probabilities[retailer_idx]
            .iter()
            .sum::<f64>();
        if (probability_sum - 1.0).abs() > 1e-12 {
            return Err(PyValueError::new_err(format!(
                "demand probabilities for retailer {} sum to {}, expected 1",
                retailer_idx, probability_sum
            )));
        }
    }
    Ok(())
}

fn as_state_key(period: usize, state: &OneWarehouseMultiRetailerState) -> ExactStateKey {
    ExactStateKey {
        period,
        warehouse_inventory: state.warehouse_inventory,
        warehouse_pipeline: state.warehouse_pipeline.clone(),
        retailer_inventory: state.retailer_inventory.clone(),
        retailer_pipeline: state.retailer_pipeline.clone(),
    }
}

fn to_state(key: &ExactStateKey) -> OneWarehouseMultiRetailerState {
    OneWarehouseMultiRetailerState {
        period: key.period,
        warehouse_inventory: key.warehouse_inventory,
        warehouse_pipeline: key.warehouse_pipeline.clone(),
        retailer_inventory: key.retailer_inventory.clone(),
        retailer_pipeline: key.retailer_pipeline.clone(),
    }
}

fn enumerate_feasible_shipments(
    max_shipments: &[usize],
    available_inventory: usize,
) -> Vec<Vec<usize>> {
    fn recurse(
        index: usize,
        max_shipments: &[usize],
        remaining_inventory: usize,
        prefix: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if index == max_shipments.len() {
            output.push(prefix.clone());
            return;
        }
        let upper = max_shipments[index].min(remaining_inventory);
        for shipment in 0..=upper {
            prefix.push(shipment);
            recurse(
                index + 1,
                max_shipments,
                remaining_inventory - shipment,
                prefix,
                output,
            );
            prefix.pop();
        }
    }

    let mut output = Vec::new();
    recurse(
        0,
        max_shipments,
        available_inventory,
        &mut Vec::new(),
        &mut output,
    );
    output
}

fn demand_scenarios(reference: &ExactVerificationReference) -> Vec<(Vec<usize>, f64)> {
    let mut scenarios = vec![(Vec::new(), 1.0)];
    for retailer_idx in 0..reference.retailer_lead_times.len() {
        let mut next = Vec::new();
        for (prefix, prefix_probability) in scenarios.into_iter() {
            for (demand, probability) in reference.demand_supports[retailer_idx]
                .iter()
                .zip(reference.demand_probabilities[retailer_idx].iter())
            {
                let mut scenario = prefix.clone();
                scenario.push(*demand as usize);
                next.push((scenario, prefix_probability * probability));
            }
        }
        scenarios = next;
    }
    scenarios
}

fn retailer_shipments_for_action(
    state: &OneWarehouseMultiRetailerState,
    retailer_orders: &[usize],
    allocation_policy: AllocationPolicy,
    retailer_base_stock_levels: &[usize],
) -> PyResult<Vec<usize>> {
    let available_warehouse_inventory =
        (state.warehouse_inventory + state.warehouse_pipeline[0] as i32).max(0) as usize;
    match allocation_policy {
        AllocationPolicy::Proportional => {
            proportional_shipments(available_warehouse_inventory, retailer_orders)
        }
        AllocationPolicy::MinShortage => min_shortage_shipments(
            available_warehouse_inventory,
            retailer_orders,
            &retailer_inventory_positions(state)?,
            retailer_base_stock_levels,
        ),
        AllocationPolicy::RandomSequential => Err(PyValueError::new_err(
            "finite_horizon_dp does not support random_sequential allocation",
        )),
    }
}

fn solve_optimal_from_state(
    state_key: &ExactStateKey,
    reference: &ExactVerificationReference,
    demand_scenarios: &[(Vec<usize>, f64)],
    cache: &mut HashMap<ExactStateKey, ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state_key.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: vec![0; reference.max_action_levels.len()],
        });
    }
    if let Some(cached) = cache.get(state_key) {
        return Ok(cached.clone());
    }

    let state = to_state(state_key);
    validate_state(&state)?;
    let mut best_cost = f64::INFINITY;
    let mut best_action = vec![0usize; reference.max_action_levels.len()];
    let available_warehouse_inventory =
        (state.warehouse_inventory + state.warehouse_pipeline[0] as i32).max(0) as usize;
    let feasible_shipments = enumerate_feasible_shipments(
        &reference.max_action_levels[1..],
        available_warehouse_inventory,
    );

    for warehouse_order in 0..=reference.max_action_levels[0] {
        for retailer_shipments in feasible_shipments.iter() {
            let mut action = vec![warehouse_order];
            action.extend(retailer_shipments.iter().copied());
            let mut expected_cost = 0.0;
            for (demands, probability) in demand_scenarios.iter() {
                let outcome = step_state(
                    &state,
                    warehouse_order,
                    retailer_shipments,
                    demands,
                    reference.holding_cost_warehouse,
                    reference.holding_cost_retailers,
                    reference.penalty_costs_retailers,
                    reference.customer_behavior,
                    reference.emergency_shipment_probability,
                    None,
                )?;
                let next_key = as_state_key(state_key.period + 1, &outcome.next_state);
                let continuation =
                    solve_optimal_from_state(&next_key, reference, demand_scenarios, cache)?;
                expected_cost += probability
                    * (outcome.period_cost
                        + reference.discount_factor * continuation.discounted_cost);
            }

            if expected_cost < best_cost - 1e-12 {
                best_cost = expected_cost;
                best_action = action;
            }
        }
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: best_cost,
        first_action: best_action,
    };
    cache.insert(state_key.clone(), result.clone());
    Ok(result)
}

pub fn solve_optimal_policy(
    reference: &ExactVerificationReference,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    let initial_state = OneWarehouseMultiRetailerState {
        period: 0,
        warehouse_inventory: reference.initial_warehouse_inventory,
        warehouse_pipeline: reference.initial_warehouse_pipeline.to_vec(),
        retailer_inventory: reference.initial_retailer_inventory.to_vec(),
        retailer_pipeline: reference
            .initial_retailer_pipeline
            .iter()
            .map(|pipeline| pipeline.to_vec())
            .collect(),
    };
    let initial_key = as_state_key(0, &initial_state);
    let scenarios = demand_scenarios(reference);
    let mut cache = HashMap::new();
    solve_optimal_from_state(&initial_key, reference, &scenarios, &mut cache)
}

fn heuristic_action(
    state: &OneWarehouseMultiRetailerState,
    warehouse_base_stock_level: usize,
    retailer_base_stock_levels: &[usize],
) -> PyResult<Vec<usize>> {
    echelon_base_stock_orders(
        state,
        warehouse_base_stock_level,
        retailer_base_stock_levels,
    )
}

fn build_policy_state(
    state: &OneWarehouseMultiRetailerState,
    total_periods: usize,
    mode: PolicyStateMode,
) -> PyResult<Vec<f32>> {
    validate_state(state)?;
    let total_system_position = warehouse_echelon_inventory_position(state)?;
    let scale = state
        .warehouse_inventory
        .abs()
        .max(total_system_position.abs())
        .max(
            state
                .retailer_inventory
                .iter()
                .map(|value| value.abs())
                .max()
                .unwrap_or(1),
        )
        .max(1) as f32;

    let normalized_dim = 1
        + state.warehouse_pipeline.len()
        + state.retailer_inventory.len()
        + state
            .retailer_pipeline
            .iter()
            .map(|pipeline| pipeline.len())
            .sum::<usize>()
        + 2;
    let augmented_dim = normalized_dim + 2 + state.retailer_inventory.len();
    let mut features = Vec::with_capacity(match mode {
        PolicyStateMode::Normalized => normalized_dim,
        PolicyStateMode::AbsoluteAugmented => augmented_dim,
    });
    features.push(state.warehouse_inventory as f32 / scale);
    features.extend(
        state
            .warehouse_pipeline
            .iter()
            .map(|value| *value as f32 / scale),
    );
    features.extend(
        state
            .retailer_inventory
            .iter()
            .map(|value| *value as f32 / scale),
    );
    for pipeline in state.retailer_pipeline.iter() {
        features.extend(pipeline.iter().map(|value| *value as f32 / scale));
    }
    features.push(total_system_position as f32 / scale);
    let remaining_fraction = if total_periods == 0 {
        0.0
    } else {
        (total_periods.saturating_sub(state.period) as f32) / total_periods as f32
    };
    features.push(remaining_fraction);
    if mode == PolicyStateMode::AbsoluteAugmented {
        features.push(scale);
        features.push(total_system_position as f32);
        features.extend(
            retailer_inventory_positions(state)?
                .iter()
                .map(|value| *value as f32),
        );
    }
    Ok(features)
}

fn soft_tree_action(
    state: &OneWarehouseMultiRetailerState,
    reference: &ExactVerificationReference,
    config: &ExactSoftTreeConfig,
) -> PyResult<(Vec<usize>, Option<Vec<usize>>)> {
    let policy_state = build_policy_state(state, reference.periods, config.policy_state_mode)?;
    if policy_state.len() != config.input_dim {
        return Err(PyValueError::new_err(
            "soft tree input_dim does not match the policy-state dimension",
        ));
    }
    let controls = action_vector_from_flat_params(
        &policy_state,
        &config.flat_params,
        config.input_dim,
        config.depth,
        config.temperature,
        config.split_type,
        config.leaf_type,
        &config.action_spec,
    )?;
    match config.policy_action_mode {
        PolicyActionMode::DirectOrders => Ok((controls, None)),
        PolicyActionMode::EchelonTargets => Ok((
            echelon_base_stock_orders(state, controls[0], &controls[1..])?,
            Some(controls[1..].to_vec()),
        )),
        PolicyActionMode::EchelonTargetsWithAllocTargets => {
            let num_retailers = state.retailer_inventory.len();
            if controls.len() != 1 + 2 * num_retailers {
                return Err(PyValueError::new_err(
                    "echelon target mode with allocation targets requires warehouse, retailer order targets, and retailer allocation targets",
                ));
            }
            let order_targets_end = 1 + num_retailers;
            Ok((
                echelon_base_stock_orders(state, controls[0], &controls[1..order_targets_end])?,
                Some(controls[order_targets_end..].to_vec()),
            ))
        }
        PolicyActionMode::SymmetricEchelonTargets => {
            if controls.len() != 2 {
                return Err(PyValueError::new_err(
                    "symmetric echelon target mode requires exactly two controls",
                ));
            }
            let retailer_targets = vec![controls[1]; state.retailer_inventory.len()];
            Ok((
                echelon_base_stock_orders(state, controls[0], &retailer_targets)?,
                Some(retailer_targets),
            ))
        }
        // The exact finite-horizon DP simulates the bounded action vector only; it has no
        // release-capacity (holdback) plumbing. The holdback head is trained/scored via the
        // stochastic rollout oracle, not the tiny-instance exact DP, so this fails loudly.
        PolicyActionMode::EchelonTargetsWithHoldback => Err(PyValueError::new_err(
            "echelon_targets_with_holdback is not supported by the exact finite-horizon DP; use the rollout oracle",
        )),
    }
}

fn evaluate_soft_tree_from_state(
    state_key: &ExactStateKey,
    reference: &ExactVerificationReference,
    config: &ExactSoftTreeConfig,
    demand_scenarios: &[(Vec<usize>, f64)],
    cache: &mut HashMap<ExactStateKey, ExactPolicyEvaluation>,
) -> PyResult<ExactPolicyEvaluation> {
    if state_key.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: vec![0; reference.max_action_levels.len()],
        });
    }
    if let Some(cached) = cache.get(state_key) {
        return Ok(cached.clone());
    }
    let state = to_state(state_key);
    let (action, retailer_targets) = soft_tree_action(&state, reference, config)?;
    let retailer_shipments = retailer_shipments_for_action(
        &state,
        &action[1..],
        config.allocation_policy,
        retailer_targets.as_deref().unwrap_or(&action[1..]),
    )?;

    let mut expected_cost = 0.0;
    for (demands, probability) in demand_scenarios.iter() {
        let outcome = step_state(
            &state,
            action[0],
            &retailer_shipments,
            demands,
            reference.holding_cost_warehouse,
            reference.holding_cost_retailers,
            reference.penalty_costs_retailers,
            reference.customer_behavior,
            reference.emergency_shipment_probability,
            None,
        )?;
        let next_key = as_state_key(state_key.period + 1, &outcome.next_state);
        let continuation =
            evaluate_soft_tree_from_state(&next_key, reference, config, demand_scenarios, cache)?;
        expected_cost += probability
            * (outcome.period_cost + reference.discount_factor * continuation.discounted_cost);
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: expected_cost,
        first_action: action,
    };
    cache.insert(state_key.clone(), result.clone());
    Ok(result)
}

fn evaluate_heuristic_from_state(
    state_key: &ExactStateKey,
    reference: &ExactVerificationReference,
    warehouse_base_stock_level: usize,
    retailer_base_stock_levels: &[usize],
    allocation_policy: AllocationPolicy,
    demand_scenarios: &[(Vec<usize>, f64)],
    cache: &mut HashMap<
        (ExactStateKey, usize, Vec<usize>, AllocationPolicy),
        ExactPolicyEvaluation,
    >,
) -> PyResult<ExactPolicyEvaluation> {
    if state_key.period == reference.periods {
        return Ok(ExactPolicyEvaluation {
            discounted_cost: 0.0,
            first_action: vec![0; reference.max_action_levels.len()],
        });
    }
    let cache_key = (
        state_key.clone(),
        warehouse_base_stock_level,
        retailer_base_stock_levels.to_vec(),
        allocation_policy,
    );
    if let Some(cached) = cache.get(&cache_key) {
        return Ok(cached.clone());
    }
    let state = to_state(state_key);
    let action = heuristic_action(
        &state,
        warehouse_base_stock_level,
        retailer_base_stock_levels,
    )?;
    let retailer_shipments = retailer_shipments_for_action(
        &state,
        &action[1..],
        allocation_policy,
        retailer_base_stock_levels,
    )?;

    let mut expected_cost = 0.0;
    for (demands, probability) in demand_scenarios.iter() {
        let outcome = step_state(
            &state,
            action[0],
            &retailer_shipments,
            demands,
            reference.holding_cost_warehouse,
            reference.holding_cost_retailers,
            reference.penalty_costs_retailers,
            reference.customer_behavior,
            reference.emergency_shipment_probability,
            None,
        )?;
        let next_key = as_state_key(state_key.period + 1, &outcome.next_state);
        let continuation = evaluate_heuristic_from_state(
            &next_key,
            reference,
            warehouse_base_stock_level,
            retailer_base_stock_levels,
            allocation_policy,
            demand_scenarios,
            cache,
        )?;
        expected_cost += probability
            * (outcome.period_cost + reference.discount_factor * continuation.discounted_cost);
    }

    let result = ExactPolicyEvaluation {
        discounted_cost: expected_cost,
        first_action: action,
    };
    cache.insert(cache_key, result.clone());
    Ok(result)
}

pub fn evaluate_echelon_base_stock_policy(
    reference: &ExactVerificationReference,
    warehouse_base_stock_level: usize,
    retailer_base_stock_levels: &[usize],
    allocation_policy: AllocationPolicy,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    if retailer_base_stock_levels.len() != reference.retailer_lead_times.len() {
        return Err(PyValueError::new_err(
            "retailer_base_stock_levels length must match the number of retailers",
        ));
    }
    if allocation_policy == AllocationPolicy::RandomSequential {
        return Err(PyValueError::new_err(
            "finite_horizon_dp does not support random_sequential allocation",
        ));
    };
    let initial_state = OneWarehouseMultiRetailerState {
        period: 0,
        warehouse_inventory: reference.initial_warehouse_inventory,
        warehouse_pipeline: reference.initial_warehouse_pipeline.to_vec(),
        retailer_inventory: reference.initial_retailer_inventory.to_vec(),
        retailer_pipeline: reference
            .initial_retailer_pipeline
            .iter()
            .map(|pipeline| pipeline.to_vec())
            .collect(),
    };
    let initial_key = as_state_key(0, &initial_state);
    let scenarios = demand_scenarios(reference);
    let mut cache = HashMap::new();
    evaluate_heuristic_from_state(
        &initial_key,
        reference,
        warehouse_base_stock_level,
        retailer_base_stock_levels,
        allocation_policy,
        &scenarios,
        &mut cache,
    )
}

pub fn evaluate_named_heuristic(
    reference: &ExactVerificationReference,
    heuristic_name: &str,
) -> PyResult<ExactPolicyEvaluation> {
    let allocation_policy = match heuristic_name {
        "echelon_base_stock_proportional" => AllocationPolicy::Proportional,
        "echelon_base_stock_min_shortage" => AllocationPolicy::MinShortage,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported heuristic '{heuristic_name}'",
            )))
        }
    };
    evaluate_echelon_base_stock_policy(
        reference,
        reference.heuristic_warehouse_base_stock_level,
        reference.heuristic_retailer_base_stock_levels,
        allocation_policy,
    )
}

pub fn evaluate_soft_tree_policy(
    reference: &ExactVerificationReference,
    config: &ExactSoftTreeConfig,
) -> PyResult<ExactPolicyEvaluation> {
    validate_exact_reference(reference)?;
    if config.allocation_policy == AllocationPolicy::RandomSequential {
        return Err(PyValueError::new_err(
            "finite_horizon_dp does not support random_sequential allocation",
        ));
    }
    let initial_state = OneWarehouseMultiRetailerState {
        period: 0,
        warehouse_inventory: reference.initial_warehouse_inventory,
        warehouse_pipeline: reference.initial_warehouse_pipeline.to_vec(),
        retailer_inventory: reference.initial_retailer_inventory.to_vec(),
        retailer_pipeline: reference
            .initial_retailer_pipeline
            .iter()
            .map(|pipeline| pipeline.to_vec())
            .collect(),
    };
    let initial_key = as_state_key(0, &initial_state);
    let scenarios = demand_scenarios(reference);
    let mut cache = HashMap::new();
    evaluate_soft_tree_from_state(&initial_key, reference, config, &scenarios, &mut cache)
}
