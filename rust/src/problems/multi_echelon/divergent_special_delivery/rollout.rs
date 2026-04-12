use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Poisson};
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::multi_echelon::env::{
    build_decision_state_with_mode, initialize_random_state,
    step_state_with_explicit_warehouse_order_and_mode, step_state_with_mode, AllocationMode,
    InventoryDynamicsMode, WarehouseBaseStockMode,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymmetricDemandDistribution {
    NormalRoundedClipped,
    Poisson,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RolloutObjective {
    AverageCostAfterWarmup,
    CumulativeCost,
    DiscountedCost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyActionMode {
    DirectBaseStock,
    AnchorAdjustment,
    DirectWarehouseOrderStoreTarget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyFeatureMode {
    FullDecisionState,
    SymmetricSummary,
    CompactSummary,
}

#[derive(Clone)]
pub struct MultiEchelonRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub policy_feature_mode: PolicyFeatureMode,
    pub policy_action_mode: PolicyActionMode,
    pub warehouse_anchor_level: usize,
    pub retailer_anchor_level: usize,
    pub warehouse_lead_time: usize,
    pub retailer_lead_time: usize,
    pub num_retailers: usize,
    pub warehouse_holding_cost: f64,
    pub retailer_holding_cost: f64,
    pub warehouse_expedited_cost: f64,
    pub warehouse_lost_sale_cost: f64,
    pub expedited_service_prob: f64,
    pub warehouse_capacity: usize,
    pub warehouse_inventory_cap: usize,
    pub retailer_inventory_cap: usize,
    pub inventory_dynamics_mode: InventoryDynamicsMode,
    pub demand_distribution: SymmetricDemandDistribution,
    pub demand_mean: f64,
    pub demand_std: f64,
    pub horizon: usize,
    pub warm_up_periods_ratio: f64,
    pub discount_factor: f64,
    pub objective: RolloutObjective,
    pub include_period_feature: bool,
    pub warehouse_base_stock_mode: WarehouseBaseStockMode,
    pub allocation_mode: AllocationMode,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

pub fn parse_demand_distribution(value: &str) -> PyResult<SymmetricDemandDistribution> {
    match value {
        "normal" | "normal_rounded_clipped" => Ok(SymmetricDemandDistribution::NormalRoundedClipped),
        "poisson" => Ok(SymmetricDemandDistribution::Poisson),
        other => Err(PyValueError::new_err(format!(
            "unsupported demand_distribution '{other}'"
        ))),
    }
}

pub fn parse_rollout_objective(value: &str) -> PyResult<RolloutObjective> {
    match value {
        "average_cost_after_warmup" | "average_cost" => Ok(RolloutObjective::AverageCostAfterWarmup),
        "cumulative_cost" | "total_cost" => Ok(RolloutObjective::CumulativeCost),
        "discounted_cost" => Ok(RolloutObjective::DiscountedCost),
        other => Err(PyValueError::new_err(format!(
            "unsupported rollout objective '{other}'"
        ))),
    }
}

pub fn parse_policy_action_mode(value: &str) -> PyResult<PolicyActionMode> {
    match value {
        "direct_base_stock" | "direct" => Ok(PolicyActionMode::DirectBaseStock),
        "anchor_adjustment" | "adjustment" => Ok(PolicyActionMode::AnchorAdjustment),
        "direct_warehouse_order_store_target"
        | "warehouse_order_store_target"
        | "direct_order_store_target"
        | "warehouse_order" => Ok(PolicyActionMode::DirectWarehouseOrderStoreTarget),
        other => Err(PyValueError::new_err(format!(
            "unsupported policy_action_mode '{other}'"
        ))),
    }
}

pub fn parse_policy_feature_mode(value: &str) -> PyResult<PolicyFeatureMode> {
    match value {
        "full_decision_state" | "full" => Ok(PolicyFeatureMode::FullDecisionState),
        "symmetric_summary" | "summary" => Ok(PolicyFeatureMode::SymmetricSummary),
        "compact_summary" => Ok(PolicyFeatureMode::CompactSummary),
        other => Err(PyValueError::new_err(format!(
            "unsupported policy_feature_mode '{other}'"
        ))),
    }
}

fn build_full_decision_state_features(
    state: &crate::problems::multi_echelon::env::MultiEchelonState,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    inventory_dynamics_mode: InventoryDynamicsMode,
) -> PyResult<Vec<f32>> {
    let decision_state = build_decision_state_with_mode(state, inventory_dynamics_mode)?;
    let warehouse_scale = warehouse_inventory_cap.max(1) as f32;
    let retailer_scale = retailer_inventory_cap.max(1) as f32;

    let mut features = vec![decision_state.warehouse_available as f32 / warehouse_scale];
    features.extend(
        decision_state
            .warehouse_future
            .iter()
            .map(|value| *value as f32 / warehouse_scale),
    );
    features.extend(
        decision_state
            .retailer_available
            .iter()
            .map(|value| *value as f32 / retailer_scale),
    );
    for retailer_future in &decision_state.retailer_future {
        features.extend(retailer_future.iter().map(|value| *value as f32 / retailer_scale));
    }
    Ok(features)
}

fn build_symmetric_summary_features(
    state: &crate::problems::multi_echelon::env::MultiEchelonState,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    inventory_dynamics_mode: InventoryDynamicsMode,
) -> PyResult<Vec<f32>> {
    let decision_state = build_decision_state_with_mode(state, inventory_dynamics_mode)?;
    let warehouse_scale = warehouse_inventory_cap.max(1) as f32;
    let retailer_scale = retailer_inventory_cap.max(1) as f32;
    let total_scale = (warehouse_inventory_cap
        + decision_state.retailer_available.len() * retailer_inventory_cap)
        .max(1) as f32;

    let retailer_count = decision_state.retailer_available.len().max(1) as f32;
    let available_mean = decision_state
        .retailer_available
        .iter()
        .map(|value| *value as f32)
        .sum::<f32>()
        / retailer_count;
    let available_min = decision_state
        .retailer_available
        .iter()
        .copied()
        .min()
        .unwrap_or(0) as f32;
    let available_max = decision_state
        .retailer_available
        .iter()
        .copied()
        .max()
        .unwrap_or(0) as f32;
    let position_mean = decision_state
        .retailer_inventory_positions
        .iter()
        .map(|value| *value as f32)
        .sum::<f32>()
        / retailer_count;
    let position_min = decision_state
        .retailer_inventory_positions
        .iter()
        .copied()
        .min()
        .unwrap_or(0) as f32;
    let position_max = decision_state
        .retailer_inventory_positions
        .iter()
        .copied()
        .max()
        .unwrap_or(0) as f32;
    let stockout_share = decision_state
        .retailer_available
        .iter()
        .filter(|value| **value <= 0)
        .count() as f32
        / retailer_count;

    let mut features = vec![decision_state.warehouse_available as f32 / warehouse_scale];
    features.extend(
        decision_state
            .warehouse_future
            .iter()
            .map(|value| *value as f32 / warehouse_scale),
    );
    features.push(decision_state.warehouse_regular_inventory_position as f32 / warehouse_scale);
    features.push(decision_state.warehouse_echelon_inventory_position as f32 / total_scale);
    features.push(available_mean / retailer_scale);
    features.push(available_min / retailer_scale);
    features.push(available_max / retailer_scale);
    features.push(position_mean / retailer_scale);
    features.push(position_min / retailer_scale);
    features.push(position_max / retailer_scale);
    features.push(stockout_share);
    if let Some(first_pipeline) = decision_state.retailer_future.first() {
        for stage_idx in 0..first_pipeline.len() {
            let stage_mean = decision_state
                .retailer_future
                .iter()
                .map(|row| row[stage_idx] as f32)
                .sum::<f32>()
                / retailer_count;
            features.push(stage_mean / retailer_scale);
        }
    }
    Ok(features)
}

fn padded_stage_sum(rows: &[Vec<u32>], stage_idx: usize) -> f32 {
    rows.iter()
        .map(|row| row.get(stage_idx).copied().unwrap_or(0) as f32)
        .sum::<f32>()
}

fn retailer_stage_inventory(retailer_idx: usize, decision_state: &crate::problems::multi_echelon::env::DecisionState, future_horizon: usize) -> f32 {
    decision_state.retailer_available[retailer_idx] as f32
        + (0..future_horizon)
            .map(|stage_idx| {
                decision_state.retailer_future[retailer_idx]
                    .get(stage_idx)
                    .copied()
                    .unwrap_or(0) as f32
            })
            .sum::<f32>()
}

fn variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f32>()
        / values.len() as f32
}

fn build_compact_summary_features(
    state: &crate::problems::multi_echelon::env::MultiEchelonState,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    inventory_dynamics_mode: InventoryDynamicsMode,
) -> PyResult<Vec<f32>> {
    let decision_state = build_decision_state_with_mode(state, inventory_dynamics_mode)?;
    let warehouse_scale = warehouse_inventory_cap.max(1) as f32;
    let store_total_scale =
        (decision_state.retailer_available.len() * retailer_inventory_cap.max(1)).max(1) as f32;
    let retailer_position_scale =
        (retailer_inventory_cap.max(1) * (1 + decision_state.retailer_future.first().map(|row| row.len()).unwrap_or(0)))
            .max(1) as f32;

    let store_on_hand = decision_state
        .retailer_available
        .iter()
        .map(|value| *value as f32)
        .sum::<f32>();
    let store_arrive_1 = padded_stage_sum(&decision_state.retailer_future, 0);
    let store_arrive_2 = padded_stage_sum(&decision_state.retailer_future, 1);
    let warehouse_available = decision_state.warehouse_available as f32;
    let warehouse_arrive_1 = decision_state.warehouse_future.get(0).copied().unwrap_or(0) as f32;
    let warehouse_arrive_2 = decision_state.warehouse_future.get(1).copied().unwrap_or(0) as f32;
    let warehouse_arrive_3 = decision_state.warehouse_future.get(2).copied().unwrap_or(0) as f32;

    let f1 = store_on_hand / store_total_scale;
    let f2 = store_arrive_1 / store_total_scale;
    let f3 = store_arrive_2 / store_total_scale;
    let f4 = warehouse_available / warehouse_scale;
    let f5 = warehouse_arrive_1 / warehouse_scale;
    let f6 = warehouse_arrive_2 / warehouse_scale;
    let f7 = warehouse_arrive_3 / warehouse_scale;

    let retailer_stage_0 = (0..decision_state.retailer_available.len())
        .map(|retailer_idx| retailer_stage_inventory(retailer_idx, &decision_state, 0) / retailer_position_scale)
        .collect::<Vec<_>>();
    let retailer_stage_1 = (0..decision_state.retailer_available.len())
        .map(|retailer_idx| retailer_stage_inventory(retailer_idx, &decision_state, 1) / retailer_position_scale)
        .collect::<Vec<_>>();
    let retailer_stage_2 = (0..decision_state.retailer_available.len())
        .map(|retailer_idx| retailer_stage_inventory(retailer_idx, &decision_state, 2) / retailer_position_scale)
        .collect::<Vec<_>>();

    Ok(vec![
        f1,
        f2,
        f3,
        f4,
        f5,
        f6,
        f7,
        f1 * f1,
        f2 * f2,
        f3 * f3,
        f4 * f4,
        f5 * f5,
        f6 * f6,
        f7 * f7,
        variance(&retailer_stage_0),
        variance(&retailer_stage_1),
        variance(&retailer_stage_2),
        f1 * f4,
        f4 * (f1 + f2 + f3),
        (f4 + f5 + f6 + f7) * (f1 + f2 + f3),
        (f4 + f5 + f6) * (f1 + f2 + f3),
        f3 * f4 * f7,
    ])
}

pub fn build_policy_features(
    state: &crate::problems::multi_echelon::env::MultiEchelonState,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    include_period_feature: bool,
    horizon: usize,
) -> PyResult<Vec<f32>> {
    build_policy_features_with_mode(
        state,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        include_period_feature,
        horizon,
        PolicyFeatureMode::FullDecisionState,
        InventoryDynamicsMode::Gijs2022,
    )
}

pub fn build_policy_features_with_mode(
    state: &crate::problems::multi_echelon::env::MultiEchelonState,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    include_period_feature: bool,
    horizon: usize,
    policy_feature_mode: PolicyFeatureMode,
    inventory_dynamics_mode: InventoryDynamicsMode,
) -> PyResult<Vec<f32>> {
    let mut features = match policy_feature_mode {
        PolicyFeatureMode::FullDecisionState => {
            build_full_decision_state_features(
                state,
                warehouse_inventory_cap,
                retailer_inventory_cap,
                inventory_dynamics_mode,
            )?
        }
        PolicyFeatureMode::SymmetricSummary => {
            build_symmetric_summary_features(
                state,
                warehouse_inventory_cap,
                retailer_inventory_cap,
                inventory_dynamics_mode,
            )?
        }
        PolicyFeatureMode::CompactSummary => {
            build_compact_summary_features(
                state,
                warehouse_inventory_cap,
                retailer_inventory_cap,
                inventory_dynamics_mode,
            )?
        }
    };
    if include_period_feature {
        features.push(state.period as f32 / horizon.max(1) as f32);
    }
    Ok(features)
}

fn sample_demands(
    rng: &mut StdRng,
    config: &MultiEchelonRolloutConfig,
) -> PyResult<Vec<u32>> {
    match config.demand_distribution {
        SymmetricDemandDistribution::NormalRoundedClipped => {
            let distribution = Normal::new(config.demand_mean, config.demand_std.max(1e-6))
                .map_err(|err| PyValueError::new_err(format!("invalid normal demand parameters: {err}")))?;
            Ok((0..config.num_retailers)
                .map(|_| distribution.sample(rng).round().max(0.0) as u32)
                .collect())
        }
        SymmetricDemandDistribution::Poisson => {
            let distribution = Poisson::new(config.demand_mean.max(1e-9))
                .map_err(|err| PyValueError::new_err(format!("invalid poisson demand_mean: {err}")))?;
            Ok((0..config.num_retailers)
                .map(|_| distribution.sample(rng) as u32)
                .collect())
        }
    }
}

fn sample_accepted_emergency_shipments(
    rng: &mut StdRng,
    total_unmet_demand: usize,
    expedited_service_prob: f64,
) -> usize {
    (0..total_unmet_demand)
        .filter(|_| rng.gen::<f64>() < expedited_service_prob)
        .count()
}

pub fn rollout_metric(period_costs: &[f64], warm_up_periods_ratio: f64, discount_factor: f64, objective: RolloutObjective) -> f64 {
    match objective {
        RolloutObjective::AverageCostAfterWarmup => {
            let horizon = period_costs.len();
            let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
            let active_costs = if warm_up_periods < period_costs.len() {
                &period_costs[warm_up_periods..]
            } else {
                period_costs
            };
            active_costs.iter().sum::<f64>() / active_costs.len().max(1) as f64
        }
        RolloutObjective::CumulativeCost => period_costs.iter().sum(),
        RolloutObjective::DiscountedCost => period_costs
            .iter()
            .enumerate()
            .map(|(period, cost)| discount_factor.powi(period as i32) * *cost)
            .sum(),
    }
}

pub fn rollout(
    flat_params: &[f32],
    config: &MultiEchelonRolloutConfig,
    seed: u64,
    initialization_warehouse_levels: &[usize],
    initialization_retailer_levels: &[usize],
) -> PyResult<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initialize_random_state(
        config.warehouse_lead_time,
        config.retailer_lead_time,
        config.num_retailers,
        initialization_warehouse_levels,
        initialization_retailer_levels,
        config.demand_mean,
        seed,
    )?;
    let mut period_costs = Vec::with_capacity(config.horizon);

    for _ in 0..config.horizon {
        let policy_state = build_policy_features_with_mode(
            &state,
            config.warehouse_inventory_cap,
            config.retailer_inventory_cap,
            config.include_period_feature,
            config.horizon,
            config.policy_feature_mode,
            config.inventory_dynamics_mode,
        )?;
        if policy_state.len() != config.input_dim {
            return Err(PyValueError::new_err(
                "policy state length does not match input_dim",
            ));
        }
        let action = action_vector_from_flat_params(
            &policy_state,
            flat_params,
            config.input_dim,
            config.depth,
            config.temperature,
            config.split_type,
            config.leaf_type,
            &config.action_spec,
        )?;
        let (warehouse_target, retailer_target, explicit_warehouse_order) =
            match config.policy_action_mode {
                PolicyActionMode::DirectBaseStock => (action[0], action[1], None),
                PolicyActionMode::AnchorAdjustment => (
                    config.warehouse_anchor_level.saturating_sub(action[0]),
                    config
                        .retailer_anchor_level
                        .saturating_add(action[1])
                        .min(config.retailer_inventory_cap),
                    None,
                ),
                PolicyActionMode::DirectWarehouseOrderStoreTarget => {
                    (0usize, action[1], Some(action[0]))
                }
        };

        let demands = sample_demands(&mut rng, config)?;
        let decision_state = build_decision_state_with_mode(&state, config.inventory_dynamics_mode)?;
        let total_unmet_demand = demands
            .iter()
            .enumerate()
            .map(|(retailer_idx, demand)| {
                let served = (*demand).min(decision_state.retailer_available[retailer_idx].max(0) as u32);
                (*demand - served) as usize
            })
            .sum::<usize>();
        let accepted_emergency_shipments = sample_accepted_emergency_shipments(
            &mut rng,
            total_unmet_demand,
            config.expedited_service_prob,
        );

        let outcome = match explicit_warehouse_order {
            Some(warehouse_order) => step_state_with_explicit_warehouse_order_and_mode(
                &state,
                warehouse_order,
                retailer_target,
                &demands,
                accepted_emergency_shipments,
                config.warehouse_capacity,
                config.warehouse_inventory_cap,
                config.retailer_inventory_cap,
                config.warehouse_holding_cost,
                config.retailer_holding_cost,
                config.warehouse_expedited_cost,
                config.warehouse_lost_sale_cost,
                config.allocation_mode,
                config.inventory_dynamics_mode,
            )?,
            None => step_state_with_mode(
                &state,
                warehouse_target,
                retailer_target,
                &demands,
                accepted_emergency_shipments,
                config.warehouse_capacity,
                config.warehouse_inventory_cap,
                config.retailer_inventory_cap,
                config.warehouse_holding_cost,
                config.retailer_holding_cost,
                config.warehouse_expedited_cost,
                config.warehouse_lost_sale_cost,
                config.warehouse_base_stock_mode,
                config.allocation_mode,
                config.inventory_dynamics_mode,
            )?,
        };
        period_costs.push(outcome.period_cost);
        state = outcome.next_state;
    }

    Ok(rollout_metric(
        &period_costs,
        config.warm_up_periods_ratio,
        config.discount_factor,
        config.objective,
    ))
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &MultiEchelonRolloutConfig,
    seeds: &[u64],
    initialization_warehouse_levels: &[usize],
    initialization_retailer_levels: &[usize],
) -> PyResult<Vec<f64>> {
    if params_batch.len() != seeds.len() {
        return Err(PyValueError::new_err(
            "params batch size must match seeds size",
        ));
    }
    let results: Vec<PyResult<f64>> = params_batch
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(params, seed)| {
            rollout(
                params,
                config,
                *seed,
                initialization_warehouse_levels,
                initialization_retailer_levels,
            )
        })
        .collect();
    let mut values = Vec::with_capacity(results.len());
    for result in results {
        values.push(result?);
    }
    Ok(values)
}
