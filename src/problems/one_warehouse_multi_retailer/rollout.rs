use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::one_warehouse_multi_retailer::allocation::{
    min_shortage_shipments, proportional_shipments, random_sequential_shipments, AllocationPolicy,
};
use crate::problems::one_warehouse_multi_retailer::demand::{
    sample_demand, validate_demand_models, DemandModel,
};
use crate::problems::one_warehouse_multi_retailer::env::{
    retailer_inventory_positions, step_state, validate_state, warehouse_echelon_inventory_position,
    CustomerBehaviorModel, OneWarehouseMultiRetailerState,
};
use crate::problems::one_warehouse_multi_retailer::heuristics::echelon_base_stock_orders;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyActionMode {
    DirectOrders,
    EchelonTargets,
    EchelonTargetsWithAllocTargets,
    SymmetricEchelonTargets,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyStateMode {
    Normalized,
    AbsoluteAugmented,
}

#[derive(Clone)]
pub struct OneWarehouseMultiRetailerRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub periods: usize,
    pub demand_models: Vec<DemandModel>,
    pub allocation_policy: AllocationPolicy,
    pub retailer_target_inventory_positions: Option<Vec<usize>>,
    pub holding_cost_warehouse: f64,
    pub holding_cost_retailers: Vec<f64>,
    pub penalty_costs_retailers: Vec<f64>,
    pub customer_behavior: CustomerBehaviorModel,
    pub emergency_shipment_probability: f64,
    pub discount_factor: f64,
    pub policy_action_mode: PolicyActionMode,
    pub policy_state_mode: PolicyStateMode,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyAction {
    pub orders: Vec<usize>,
    pub retailer_target_inventory_positions: Option<Vec<usize>>,
}

pub fn parse_policy_state_mode(value: &str) -> PyResult<PolicyStateMode> {
    match value {
        "normalized" | "default" => Ok(PolicyStateMode::Normalized),
        "absolute_augmented" | "augmented" | "absolute" => Ok(PolicyStateMode::AbsoluteAugmented),
        other => Err(PyValueError::new_err(format!(
            "unsupported policy_state_mode '{other}'; expected 'normalized' or 'absolute_augmented'"
        ))),
    }
}

pub fn parse_policy_action_mode(value: &str) -> PyResult<PolicyActionMode> {
    match value {
        "direct_orders" | "direct" | "order_quantities" => Ok(PolicyActionMode::DirectOrders),
        "echelon_targets" | "echelon_base_stock_targets" | "targets" => {
            Ok(PolicyActionMode::EchelonTargets)
        }
        "echelon_targets_with_alloc_targets"
        | "echelon_base_stock_targets_with_alloc_targets"
        | "targets_with_alloc_targets" => Ok(PolicyActionMode::EchelonTargetsWithAllocTargets),
        "symmetric_echelon_targets" | "symmetric_targets" | "shared_retailer_targets" => {
            Ok(PolicyActionMode::SymmetricEchelonTargets)
        }
        other => Err(PyValueError::new_err(format!(
            "unsupported policy_action_mode '{other}'; expected 'direct_orders', 'echelon_targets', 'echelon_targets_with_alloc_targets', or 'symmetric_echelon_targets'"
        ))),
    }
}

fn validate_config(
    config: &OneWarehouseMultiRetailerRolloutConfig,
    initial_state: &OneWarehouseMultiRetailerState,
) -> PyResult<()> {
    validate_state(initial_state)?;
    validate_demand_models(&config.demand_models)?;
    let num_retailers = initial_state.retailer_inventory.len();
    if config.demand_models.len() != num_retailers
        || config.holding_cost_retailers.len() != num_retailers
        || config.penalty_costs_retailers.len() != num_retailers
    {
        return Err(PyValueError::new_err(
            "all retailer-wise config vectors must match the number of retailers",
        ));
    }
    if let Some(ref targets) = config.retailer_target_inventory_positions {
        if targets.len() != num_retailers {
            return Err(PyValueError::new_err(
                "retailer_target_inventory_positions length must match the number of retailers",
            ));
        }
    }
    let expected_action_dim = match config.policy_action_mode {
        PolicyActionMode::DirectOrders | PolicyActionMode::EchelonTargets => num_retailers + 1,
        PolicyActionMode::EchelonTargetsWithAllocTargets => 1 + 2 * num_retailers,
        PolicyActionMode::SymmetricEchelonTargets => 2,
    };
    if config.action_spec.action_dim != expected_action_dim {
        return Err(PyValueError::new_err(format!(
            "action_spec.action_dim {} does not match expected {} for {:?}",
            config.action_spec.action_dim, expected_action_dim, config.policy_action_mode
        )));
    }
    if config.policy_action_mode == PolicyActionMode::SymmetricEchelonTargets && num_retailers == 0
    {
        return Err(PyValueError::new_err(
            "symmetric_echelon_targets requires at least one retailer",
        ));
    }
    let expected_input_dim = expected_policy_input_dim(initial_state, config.policy_state_mode);
    if config.input_dim != expected_input_dim {
        return Err(PyValueError::new_err(format!(
            "input_dim {} does not match expected {} for {:?}",
            config.input_dim, expected_input_dim, config.policy_state_mode
        )));
    }
    if !(0.0..=1.0).contains(&config.discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }
    if config.allocation_policy == AllocationPolicy::MinShortage
        && config.retailer_target_inventory_positions.is_none()
        && config.policy_action_mode != PolicyActionMode::EchelonTargets
        && config.policy_action_mode != PolicyActionMode::EchelonTargetsWithAllocTargets
        && config.policy_action_mode != PolicyActionMode::SymmetricEchelonTargets
    {
        return Err(PyValueError::new_err(
            "min_shortage rollout requires retailer_target_inventory_positions",
        ));
    }
    Ok(())
}

fn normalized_policy_input_dim(state: &OneWarehouseMultiRetailerState) -> usize {
    1 + state.warehouse_pipeline.len()
        + state.retailer_inventory.len()
        + state
            .retailer_pipeline
            .iter()
            .map(|pipeline| pipeline.len())
            .sum::<usize>()
        + 2
}

pub fn expected_policy_input_dim(
    state: &OneWarehouseMultiRetailerState,
    mode: PolicyStateMode,
) -> usize {
    let normalized_dim = normalized_policy_input_dim(state);
    match mode {
        PolicyStateMode::Normalized => normalized_dim,
        PolicyStateMode::AbsoluteAugmented => normalized_dim + 2 + state.retailer_inventory.len(),
    }
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

    let mut features = Vec::with_capacity(expected_policy_input_dim(state, mode));
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

fn action_vector(
    flat_params: &[f32],
    state: &OneWarehouseMultiRetailerState,
    config: &OneWarehouseMultiRetailerRolloutConfig,
) -> PyResult<Vec<usize>> {
    let policy_state = build_policy_state(state, config.periods, config.policy_state_mode)?;
    action_vector_from_flat_params(
        &policy_state,
        flat_params,
        config.input_dim,
        config.depth,
        config.temperature,
        config.split_type,
        config.leaf_type,
        &config.action_spec,
    )
}

pub fn policy_action_from_tree(
    flat_params: &[f32],
    state: &OneWarehouseMultiRetailerState,
    config: &OneWarehouseMultiRetailerRolloutConfig,
) -> PyResult<PolicyAction> {
    let actions = action_vector(flat_params, state, config)?;
    match config.policy_action_mode {
        PolicyActionMode::DirectOrders => Ok(PolicyAction {
            orders: actions,
            retailer_target_inventory_positions: config.retailer_target_inventory_positions.clone(),
        }),
        PolicyActionMode::EchelonTargets => {
            if actions.len() < 2 {
                return Err(PyValueError::new_err(
                    "echelon target mode requires warehouse and retailer targets",
                ));
            }
            Ok(PolicyAction {
                orders: echelon_base_stock_orders(state, actions[0], &actions[1..])?,
                retailer_target_inventory_positions: Some(actions[1..].to_vec()),
            })
        }
        PolicyActionMode::EchelonTargetsWithAllocTargets => {
            let num_retailers = state.retailer_inventory.len();
            if actions.len() != 1 + 2 * num_retailers {
                return Err(PyValueError::new_err(
                    "echelon target mode with allocation targets requires warehouse, retailer order targets, and retailer allocation targets",
                ));
            }
            let order_targets_end = 1 + num_retailers;
            Ok(PolicyAction {
                orders: echelon_base_stock_orders(
                    state,
                    actions[0],
                    &actions[1..order_targets_end],
                )?,
                retailer_target_inventory_positions: Some(actions[order_targets_end..].to_vec()),
            })
        }
        PolicyActionMode::SymmetricEchelonTargets => {
            if actions.len() != 2 {
                return Err(PyValueError::new_err(
                    "symmetric echelon target mode requires exactly two controls",
                ));
            }
            let retailer_targets = vec![actions[1]; state.retailer_inventory.len()];
            Ok(PolicyAction {
                orders: echelon_base_stock_orders(state, actions[0], &retailer_targets)?,
                retailer_target_inventory_positions: Some(retailer_targets),
            })
        }
    }
}

fn retailer_shipments<R: Rng + ?Sized>(
    rng: &mut R,
    state: &OneWarehouseMultiRetailerState,
    policy_action: &PolicyAction,
    config: &OneWarehouseMultiRetailerRolloutConfig,
) -> PyResult<Vec<usize>> {
    let available_warehouse_inventory =
        (state.warehouse_inventory + state.warehouse_pipeline[0] as i32).max(0) as usize;
    match config.allocation_policy {
        AllocationPolicy::Proportional => {
            proportional_shipments(available_warehouse_inventory, &policy_action.orders[1..])
        }
        AllocationPolicy::RandomSequential => random_sequential_shipments(
            rng,
            available_warehouse_inventory,
            &policy_action.orders[1..],
        ),
        AllocationPolicy::MinShortage => min_shortage_shipments(
            available_warehouse_inventory,
            &policy_action.orders[1..],
            &retailer_inventory_positions(state)?,
            policy_action
                .retailer_target_inventory_positions
                .as_deref()
                .or(config.retailer_target_inventory_positions.as_deref())
                .expect("validated above"),
        ),
    }
}

pub fn rollout(
    flat_params: &[f32],
    config: &OneWarehouseMultiRetailerRolloutConfig,
    initial_state: &OneWarehouseMultiRetailerState,
    seed: u64,
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut total_discounted_cost = 0.0;
    let mut discount = 1.0;

    for _ in 0..config.periods {
        let realized_demands = config
            .demand_models
            .iter()
            .map(|model| sample_demand(&mut rng, model))
            .collect::<PyResult<Vec<_>>>()?;
        let policy_action = policy_action_from_tree(flat_params, &state, config)?;
        let retailer_shipments = retailer_shipments(&mut rng, &state, &policy_action, config)?;
        let emergency_draws = if config.customer_behavior == CustomerBehaviorModel::PartialBackorder
        {
            Some(
                (0..state.retailer_inventory.len())
                    .map(|_| rng.gen_bool(config.emergency_shipment_probability))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let outcome = step_state(
            &state,
            policy_action.orders[0],
            &retailer_shipments,
            &realized_demands,
            config.holding_cost_warehouse,
            &config.holding_cost_retailers,
            &config.penalty_costs_retailers,
            config.customer_behavior,
            config.emergency_shipment_probability,
            emergency_draws.as_deref(),
        )?;
        total_discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }

    Ok(total_discounted_cost)
}

pub fn rollout_from_paths(
    flat_params: &[f32],
    config: &OneWarehouseMultiRetailerRolloutConfig,
    initial_state: &OneWarehouseMultiRetailerState,
    demands: &[Vec<usize>],
    seed: u64,
) -> PyResult<f64> {
    validate_config(config, initial_state)?;
    if demands.len() != config.periods {
        return Err(PyValueError::new_err(
            "demands length must match config.periods",
        ));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut total_discounted_cost = 0.0;
    let mut discount = 1.0;

    for demand in demands.iter() {
        if demand.len() != state.retailer_inventory.len() {
            return Err(PyValueError::new_err(
                "each realized demand vector must match the number of retailers",
            ));
        }
        let policy_action = policy_action_from_tree(flat_params, &state, config)?;
        let retailer_shipments = retailer_shipments(&mut rng, &state, &policy_action, config)?;
        let emergency_draws = if config.customer_behavior == CustomerBehaviorModel::PartialBackorder
        {
            Some(
                (0..state.retailer_inventory.len())
                    .map(|_| rng.gen_bool(config.emergency_shipment_probability))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let outcome = step_state(
            &state,
            policy_action.orders[0],
            &retailer_shipments,
            demand,
            config.holding_cost_warehouse,
            &config.holding_cost_retailers,
            &config.penalty_costs_retailers,
            config.customer_behavior,
            config.emergency_shipment_probability,
            emergency_draws.as_deref(),
        )?;
        total_discounted_cost += discount * outcome.period_cost;
        discount *= config.discount_factor;
        state = outcome.next_state;
    }

    Ok(total_discounted_cost)
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &OneWarehouseMultiRetailerRolloutConfig,
    initial_state: &OneWarehouseMultiRetailerState,
    seeds: &[u64],
) -> PyResult<Vec<f64>> {
    if params_batch.len() != seeds.len() {
        return Err(PyValueError::new_err(
            "params_batch and seeds must have the same length",
        ));
    }
    validate_config(config, initial_state)?;
    params_batch
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(flat_params, seed)| rollout(flat_params, config, initial_state, *seed))
        .collect()
}

pub fn build_initial_state(
    warehouse_inventory: i32,
    warehouse_pipeline: &[usize],
    retailer_inventory: &[i32],
    retailer_pipeline: &[Vec<usize>],
) -> PyResult<OneWarehouseMultiRetailerState> {
    crate::problems::one_warehouse_multi_retailer::env::initialize_state(
        warehouse_inventory,
        warehouse_pipeline,
        retailer_inventory,
        retailer_pipeline,
    )
}
