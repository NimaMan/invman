use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::problems::multi_echelon::env::{
    build_decision_state_with_mode, initialize_random_state, step_state_with_mode, AllocationMode,
    InventoryDynamicsMode, WarehouseBaseStockMode,
};
use crate::problems::multi_echelon::rollout::{
    parse_demand_distribution, parse_rollout_objective, rollout_metric, MultiEchelonRolloutConfig,
    RolloutObjective, SymmetricDemandDistribution,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StationaryPolicyKind {
    RegularBaseStock,
    EchelonBaseStock,
}

#[derive(Clone)]
pub struct HeuristicSimulationConfig {
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
}

pub fn parse_stationary_policy_kind(value: &str) -> PyResult<StationaryPolicyKind> {
    match value {
        "constant_base_stock" => Ok(StationaryPolicyKind::RegularBaseStock),
        "regular_base_stock" => Ok(StationaryPolicyKind::RegularBaseStock),
        "echelon_base_stock" => Ok(StationaryPolicyKind::EchelonBaseStock),
        other => Err(PyValueError::new_err(format!(
            "unsupported stationary policy kind '{other}'"
        ))),
    }
}

fn sample_demands(
    rng: &mut rand::rngs::StdRng,
    config: &HeuristicSimulationConfig,
) -> PyResult<Vec<u32>> {
    match config.demand_distribution {
        SymmetricDemandDistribution::NormalRoundedClipped => {
            let distribution = rand_distr::Normal::new(config.demand_mean, config.demand_std.max(1e-6))
                .map_err(|err| PyValueError::new_err(format!("invalid normal demand parameters: {err}")))?;
            Ok((0..config.num_retailers)
                .map(|_| rand_distr::Distribution::sample(&distribution, rng).round().max(0.0) as u32)
                .collect())
        }
        SymmetricDemandDistribution::Poisson => {
            let distribution = rand_distr::Poisson::new(config.demand_mean.max(1e-9))
                .map_err(|err| PyValueError::new_err(format!("invalid poisson demand_mean: {err}")))?;
            Ok((0..config.num_retailers)
                .map(|_| rand_distr::Distribution::sample(&distribution, rng) as u32)
                .collect())
        }
    }
}

fn sample_accepted_emergency_shipments(
    rng: &mut rand::rngs::StdRng,
    total_unmet_demand: usize,
    expedited_service_prob: f64,
) -> usize {
    (0..total_unmet_demand)
        .filter(|_| rand::Rng::gen::<f64>(rng) < expedited_service_prob)
        .count()
}

fn simulate_stationary_policy_once(
    config: &HeuristicSimulationConfig,
    warehouse_levels: &[usize],
    retailer_levels: &[usize],
    warehouse_level: usize,
    retailer_level: usize,
    warehouse_base_stock_mode: WarehouseBaseStockMode,
    allocation_mode: AllocationMode,
    seed: u64,
) -> PyResult<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut state = initialize_random_state(
        config.warehouse_lead_time,
        config.retailer_lead_time,
        config.num_retailers,
        warehouse_levels,
        retailer_levels,
        config.demand_mean,
        seed,
    )?;
    let mut period_costs = Vec::with_capacity(config.horizon);

    for _ in 0..config.horizon {
        let demands = sample_demands(&mut rng, config)?;
        let decision_state =
            build_decision_state_with_mode(&state, config.inventory_dynamics_mode)?;
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
        let outcome = step_state_with_mode(
            &state,
            warehouse_level,
            retailer_level,
            &demands,
            accepted_emergency_shipments,
            config.warehouse_capacity,
            config.warehouse_inventory_cap,
            config.retailer_inventory_cap,
            config.warehouse_holding_cost,
            config.retailer_holding_cost,
            config.warehouse_expedited_cost,
            config.warehouse_lost_sale_cost,
            warehouse_base_stock_mode,
            allocation_mode,
            config.inventory_dynamics_mode,
        )?;
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

pub fn evaluate_stationary_policy(
    config: &HeuristicSimulationConfig,
    warehouse_levels: &[usize],
    retailer_levels: &[usize],
    warehouse_level: usize,
    retailer_level: usize,
    policy_kind: StationaryPolicyKind,
    allocation_mode: AllocationMode,
    replications: usize,
    seed: u64,
) -> PyResult<(f64, f64)> {
    let warehouse_base_stock_mode = match policy_kind {
        StationaryPolicyKind::RegularBaseStock => WarehouseBaseStockMode::Regular,
        StationaryPolicyKind::EchelonBaseStock => WarehouseBaseStockMode::Echelon,
    };

    let results: Vec<PyResult<f64>> = (0..replications)
        .into_par_iter()
        .map(|replication_idx| {
            simulate_stationary_policy_once(
                config,
                warehouse_levels,
                retailer_levels,
                warehouse_level,
                retailer_level,
                warehouse_base_stock_mode,
                allocation_mode,
                seed + replication_idx as u64,
            )
        })
        .collect();

    let mut values = Vec::with_capacity(results.len());
    for result in results {
        values.push(result?);
    }
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    let variance = values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / values.len().max(1) as f64;
    Ok((mean, variance.sqrt()))
}

#[allow(clippy::too_many_arguments)]
pub fn search_stationary_policy(
    warehouse_levels: &[usize],
    retailer_levels: &[usize],
    policy_kind: StationaryPolicyKind,
    allocation_mode: AllocationMode,
    warehouse_lead_time: usize,
    retailer_lead_time: usize,
    num_retailers: usize,
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    expedited_service_prob: f64,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    inventory_dynamics_mode: InventoryDynamicsMode,
    demand_distribution: &str,
    demand_mean: f64,
    demand_std: f64,
    horizon: usize,
    warm_up_periods_ratio: f64,
    discount_factor: f64,
    objective: &str,
    replications: usize,
    seed: u64,
    top_k: usize,
) -> PyResult<((usize, usize, f64, f64), Vec<(usize, usize, f64, f64)>)> {
    let config = HeuristicSimulationConfig {
        warehouse_lead_time,
        retailer_lead_time,
        num_retailers,
        warehouse_holding_cost,
        retailer_holding_cost,
        warehouse_expedited_cost,
        warehouse_lost_sale_cost,
        expedited_service_prob,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        inventory_dynamics_mode,
        demand_distribution: parse_demand_distribution(demand_distribution)?,
        demand_mean,
        demand_std,
        horizon,
        warm_up_periods_ratio,
        discount_factor,
        objective: parse_rollout_objective(objective)?,
    };

    let mut results = Vec::new();
    for warehouse_level in warehouse_levels.iter().copied() {
        for retailer_level in retailer_levels.iter().copied() {
            let (mean, std) = evaluate_stationary_policy(
                &config,
                warehouse_levels,
                retailer_levels,
                warehouse_level,
                retailer_level,
                policy_kind,
                allocation_mode,
                replications,
                seed,
            )?;
            results.push((warehouse_level, retailer_level, mean, std));
        }
    }
    results.sort_by(|left, right| left.2.partial_cmp(&right.2).unwrap());
    Ok((results[0], results.into_iter().take(top_k).collect()))
}

impl From<&MultiEchelonRolloutConfig> for HeuristicSimulationConfig {
    fn from(config: &MultiEchelonRolloutConfig) -> Self {
        Self {
            warehouse_lead_time: config.warehouse_lead_time,
            retailer_lead_time: config.retailer_lead_time,
            num_retailers: config.num_retailers,
            warehouse_holding_cost: config.warehouse_holding_cost,
            retailer_holding_cost: config.retailer_holding_cost,
            warehouse_expedited_cost: config.warehouse_expedited_cost,
            warehouse_lost_sale_cost: config.warehouse_lost_sale_cost,
            expedited_service_prob: config.expedited_service_prob,
            warehouse_capacity: config.warehouse_capacity,
            warehouse_inventory_cap: config.warehouse_inventory_cap,
            retailer_inventory_cap: config.retailer_inventory_cap,
            inventory_dynamics_mode: config.inventory_dynamics_mode,
            demand_distribution: config.demand_distribution,
            demand_mean: config.demand_mean,
            demand_std: config.demand_std,
            horizon: config.horizon,
            warm_up_periods_ratio: config.warm_up_periods_ratio,
            discount_factor: config.discount_factor,
            objective: config.objective,
        }
    }
}
