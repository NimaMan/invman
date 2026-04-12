#![allow(dead_code)]

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::SeedableRng;
use statrs::distribution::{DiscreteCDF, Poisson};

use crate::problems::lost_sales::demand::{
    build_demand_process, sample_demand, LostSalesDemandConfig, LostSalesDemandKind,
};
use crate::problems::lost_sales::env::{epoch_cost, initialize_state};
use crate::problems::lost_sales::rollout::{
    linear_rollout, neural_rollout, rollout, LostSalesLinearRolloutConfig,
    LostSalesNeuralRolloutConfig, LostSalesRolloutConfig,
};

const DEMAND_SUPPORT_TAIL_MASS: f64 = 1e-14;
const HEURISTIC_DISCOUNT_FACTOR: f64 = 0.995;

pub const VANILLA_L4_P4_POISSON5_VERIFICATION_REFERENCE: &str = "vanilla_l4_p4_poisson5";
pub const VANILLA_L4_P4_POISSON5_VERIFICATION_HORIZON: usize = 100_000;
pub const VANILLA_L4_P4_POISSON5_VERIFICATION_SEED: u64 = 123;

pub const VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG: LostSalesHeuristicVerificationConfig =
    LostSalesHeuristicVerificationConfig {
        reference_name: VANILLA_L4_P4_POISSON5_VERIFICATION_REFERENCE,
        horizon: VANILLA_L4_P4_POISSON5_VERIFICATION_HORIZON,
        seed: VANILLA_L4_P4_POISSON5_VERIFICATION_SEED,
        warm_up_periods_ratio: 0.2,
        order_search_upper_bound: 200,
        lead_time: 4,
        holding_cost: 1.0,
        shortage_cost: 4.0,
        procurement_cost: 0.0,
        fixed_order_cost: 0.0,
        heuristic_discount_factor: HEURISTIC_DISCOUNT_FACTOR,
        demand_config: LostSalesDemandConfig {
            kind: LostSalesDemandKind::Poisson,
            demand_rate: 5.0,
            demand_lambda_low: 0.0,
            demand_lambda_high: 0.0,
            demand_p00: 0.0,
            demand_p11: 0.0,
        },
    };

#[derive(Clone, Copy)]
pub struct LostSalesHeuristicVerificationConfig {
    pub reference_name: &'static str,
    pub horizon: usize,
    pub seed: u64,
    pub warm_up_periods_ratio: f64,
    pub order_search_upper_bound: usize,
    pub lead_time: usize,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub procurement_cost: f64,
    pub fixed_order_cost: f64,
    pub heuristic_discount_factor: f64,
    pub demand_config: LostSalesDemandConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyVerificationRole {
    OptimalReference,
    Heuristic,
    LearnedPolicyThreshold,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyPerformanceTarget {
    pub policy_name: &'static str,
    pub role: PolicyVerificationRole,
    pub expected_mean_cost: f64,
    pub tolerance: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LostSalesHeuristicPolicyKind {
    Myopic1,
    Myopic2,
    StandardVectorBaseStock,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyPerformanceMeasurement {
    pub policy_name: &'static str,
    pub mean_cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyPerformanceVerificationResult {
    pub target: PolicyPerformanceTarget,
    pub observed_mean_cost: Option<f64>,
    pub abs_gap: Option<f64>,
    pub within_tolerance: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyPerformanceVerificationSummary {
    pub reference_name: &'static str,
    pub horizon: usize,
    pub seed: u64,
    pub results: Vec<PolicyPerformanceVerificationResult>,
    pub untargeted_measurements: Vec<PolicyPerformanceMeasurement>,
}

pub const VANILLA_L4_P4_POISSON5_POLICY_TARGETS: &[PolicyPerformanceTarget] = &[
    PolicyPerformanceTarget {
        policy_name: "optimal_reference",
        role: PolicyVerificationRole::OptimalReference,
        expected_mean_cost: 4.73,
        tolerance: 0.12,
    },
    PolicyPerformanceTarget {
        policy_name: "capped_base_stock",
        role: PolicyVerificationRole::Heuristic,
        expected_mean_cost: 4.80,
        tolerance: 0.12,
    },
    PolicyPerformanceTarget {
        policy_name: "myopic2",
        role: PolicyVerificationRole::Heuristic,
        expected_mean_cost: 4.82,
        tolerance: 0.03,
    },
    PolicyPerformanceTarget {
        policy_name: "myopic1",
        role: PolicyVerificationRole::Heuristic,
        expected_mean_cost: 5.06,
        tolerance: 0.08,
    },
    PolicyPerformanceTarget {
        policy_name: "svbs",
        role: PolicyVerificationRole::Heuristic,
        expected_mean_cost: 5.83,
        tolerance: 0.03,
    },
];

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RecursiveCostKey {
    lookahead_depth: usize,
    state: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ActionStateKey {
    order_quantity: usize,
    state: Vec<usize>,
}

struct LostSalesHeuristicEvaluator {
    config: LostSalesHeuristicVerificationConfig,
    demand_support: Vec<(usize, f64)>,
    one_period_cache: HashMap<usize, f64>,
    q_l_cache: HashMap<RecursiveCostKey, f64>,
    q_l_from_state_action_cache: HashMap<ActionStateKey, f64>,
    myopic2_cache: HashMap<ActionStateKey, f64>,
    svbs_levels: Option<Vec<usize>>,
}

impl LostSalesHeuristicPolicyKind {
    pub fn policy_name(self) -> &'static str {
        match self {
            Self::Myopic1 => "myopic1",
            Self::Myopic2 => "myopic2",
            Self::StandardVectorBaseStock => "svbs",
        }
    }

    fn all() -> [Self; 3] {
        [Self::Myopic2, Self::Myopic1, Self::StandardVectorBaseStock]
    }
}

impl PolicyPerformanceVerificationSummary {
    pub fn observed_mean_cost(&self, policy_name: &str) -> Option<f64> {
        self.results
            .iter()
            .find(|result| result.target.policy_name == policy_name)
            .and_then(|result| result.observed_mean_cost)
    }

    pub fn executable_results(&self) -> Vec<&PolicyPerformanceVerificationResult> {
        self.results
            .iter()
            .filter(|result| result.observed_mean_cost.is_some())
            .collect()
    }

    pub fn executable_targets_are_sorted_from_best_to_worst(&self) -> bool {
        let executable = self.executable_results();
        executable.windows(2).all(|window| {
            window[0].observed_mean_cost.unwrap_or(f64::INFINITY)
                <= window[1].observed_mean_cost.unwrap_or(f64::INFINITY)
        })
    }

    pub fn all_executable_targets_within_tolerance(&self) -> bool {
        self.results
            .iter()
            .filter(|result| result.observed_mean_cost.is_some())
            .all(|result| result.within_tolerance.unwrap_or(false))
    }

    pub fn untargeted_measurement(
        &self,
        policy_name: &str,
    ) -> Option<&PolicyPerformanceMeasurement> {
        self.untargeted_measurements
            .iter()
            .find(|measurement| measurement.policy_name == policy_name)
    }
}

impl LostSalesHeuristicEvaluator {
    fn new(config: LostSalesHeuristicVerificationConfig) -> Result<Self, String> {
        validate_verification_config(&config)?;
        Ok(Self {
            demand_support: iid_demand_support(&config.demand_config)?,
            config,
            one_period_cache: HashMap::new(),
            q_l_cache: HashMap::new(),
            q_l_from_state_action_cache: HashMap::new(),
            myopic2_cache: HashMap::new(),
            svbs_levels: None,
        })
    }

    fn get_one_period_cost(&mut self, y: usize) -> f64 {
        if let Some(cost) = self.one_period_cache.get(&y) {
            return *cost;
        }

        let expected_overage = self
            .demand_support
            .iter()
            .map(|(demand, probability)| y.saturating_sub(*demand) as f64 * probability)
            .sum::<f64>();
        let expected_underage = self
            .demand_support
            .iter()
            .map(|(demand, probability)| demand.saturating_sub(y) as f64 * probability)
            .sum::<f64>();
        let total_cost = self.config.procurement_cost * y as f64
            + self.config.holding_cost * expected_overage
            + self.config.shortage_cost * expected_underage;
        self.one_period_cache.insert(y, total_cost);
        total_cost
    }

    fn q_l(&mut self, state: &[usize], lookahead_depth: usize) -> Result<f64, String> {
        let key = RecursiveCostKey {
            lookahead_depth,
            state: state.to_vec(),
        };
        if let Some(value) = self.q_l_cache.get(&key) {
            return Ok(*value);
        }

        let value = if lookahead_depth == 0 {
            self.get_one_period_cost(state[0])
        } else {
            let demand_support = self.demand_support.clone();
            let expected_cost = demand_support
                .iter()
                .map(|(demand, probability)| {
                    let next_state =
                        state_after_demand_and_pipeline_advance(state, *demand, lookahead_depth);
                    self.q_l(&next_state, lookahead_depth - 1)
                        .map(|cost| probability * cost)
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .sum::<f64>();
            self.config.heuristic_discount_factor * expected_cost
        };

        self.q_l_cache.insert(key, value);
        Ok(value)
    }

    fn q_l_from_state_action(
        &mut self,
        state: &[usize],
        order_quantity: usize,
    ) -> Result<f64, String> {
        let key = ActionStateKey {
            order_quantity,
            state: state.to_vec(),
        };
        if let Some(value) = self.q_l_from_state_action_cache.get(&key) {
            return Ok(*value);
        }

        let extended_state = state_with_appended_order(state, order_quantity);
        let demand_support = self.demand_support.clone();
        let expected_cost = demand_support
            .iter()
            .map(|(demand, probability)| {
                let next_state = state_after_demand_and_pipeline_advance(
                    &extended_state,
                    *demand,
                    self.config.lead_time,
                );
                self.q_l(&next_state, self.config.lead_time - 1)
                    .map(|cost| probability * cost)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<f64>();
        let value = self.config.heuristic_discount_factor * expected_cost;

        self.q_l_from_state_action_cache.insert(key, value);
        Ok(value)
    }

    fn best_quantity(
        &mut self,
        state: &[usize],
        evaluator: fn(&mut Self, &[usize], usize) -> Result<f64, String>,
    ) -> Result<(usize, f64), String> {
        let mut best_quantity = 0usize;
        let mut current_value = evaluator(self, state, 0)?;
        let mut previous_value = f64::INFINITY;

        while best_quantity < self.config.order_search_upper_bound && previous_value > current_value
        {
            best_quantity += 1;
            previous_value = current_value;
            current_value = evaluator(self, state, best_quantity)?;
        }

        if previous_value > current_value {
            Ok((best_quantity, current_value))
        } else {
            Ok((best_quantity.saturating_sub(1), previous_value))
        }
    }

    fn myopic_1_order_quantity(&mut self, state: &[usize]) -> Result<(usize, f64), String> {
        self.best_quantity(state, Self::q_l_from_state_action)
    }

    fn myopic_2_q_value(&mut self, state: &[usize], order_quantity: usize) -> Result<f64, String> {
        let key = ActionStateKey {
            order_quantity,
            state: state.to_vec(),
        };
        if let Some(value) = self.myopic2_cache.get(&key) {
            return Ok(*value);
        }

        let extended_state = state_with_appended_order(state, order_quantity);
        let q_hat_z = self.q_l_from_state_action(state, order_quantity)?;
        let demand_support = self.demand_support.clone();
        let future_value = demand_support
            .iter()
            .map(|(demand, probability)| {
                let next_state = state_after_demand_and_pipeline_advance(
                    &extended_state,
                    *demand,
                    self.config.lead_time,
                );
                self.myopic_1_order_quantity(&next_state)
                    .map(|(_, q_hat)| probability * q_hat)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<f64>();
        let value = q_hat_z + self.config.heuristic_discount_factor * future_value;

        self.myopic2_cache.insert(key, value);
        Ok(value)
    }

    fn myopic_2_order_quantity(&mut self, state: &[usize]) -> Result<(usize, f64), String> {
        self.best_quantity(state, Self::myopic_2_q_value)
    }

    fn standard_vector_base_stock_levels(&mut self) -> Result<&[usize], String> {
        if self.svbs_levels.is_none() {
            let levels = (0..=self.config.lead_time)
                .map(|l| {
                    let mut s = 0usize;
                    while (1.0
                        - cumulative_demand_cdf(
                            &self.config.demand_config,
                            s,
                            self.config.lead_time - l + 1,
                        )?)
                        >= critical_fractile(&self.config)
                    {
                        s += 1;
                    }
                    Ok(s)
                })
                .collect::<Result<Vec<_>, String>>()?;
            self.svbs_levels = Some(levels);
        }

        Ok(self.svbs_levels.as_deref().unwrap_or(&[]))
    }

    fn standard_vector_base_stock_order_quantity(
        &mut self,
        state: &[usize],
    ) -> Result<usize, String> {
        let levels = self.standard_vector_base_stock_levels()?.to_vec();
        let order_quantity = (0..=self.config.lead_time)
            .map(|l| {
                let pipeline_partial_sum =
                    order_pipeline_partial_sum(self.config.lead_time, l, state);
                levels[l].saturating_sub(pipeline_partial_sum)
            })
            .min()
            .unwrap_or(0);
        Ok(order_quantity.min(self.config.order_search_upper_bound))
    }
}

pub fn policy_targets_are_sorted_from_best_to_worst(targets: &[PolicyPerformanceTarget]) -> bool {
    targets
        .windows(2)
        .all(|window| window[0].expected_mean_cost <= window[1].expected_mean_cost)
}

pub fn target_for_policy_name(
    targets: &[PolicyPerformanceTarget],
    policy_name: &str,
) -> Option<PolicyPerformanceTarget> {
    targets
        .iter()
        .copied()
        .find(|target| target.policy_name == policy_name)
}

pub fn compare_observed_policy_cost(
    targets: &[PolicyPerformanceTarget],
    policy_name: &str,
    observed_mean_cost: f64,
) -> Option<PolicyPerformanceVerificationResult> {
    target_for_policy_name(targets, policy_name).map(|target| {
        let abs_gap = (observed_mean_cost - target.expected_mean_cost).abs();
        PolicyPerformanceVerificationResult {
            target,
            observed_mean_cost: Some(observed_mean_cost),
            abs_gap: Some(abs_gap),
            within_tolerance: Some(abs_gap <= target.tolerance),
        }
    })
}

pub fn measurement_from_observed_mean_cost(
    policy_name: &'static str,
    mean_cost: f64,
) -> PolicyPerformanceMeasurement {
    PolicyPerformanceMeasurement {
        policy_name,
        mean_cost,
    }
}

pub fn evaluate_soft_tree_policy(
    policy_name: &'static str,
    flat_params: &[f32],
    config: &LostSalesRolloutConfig,
    seed: u64,
) -> Result<PolicyPerformanceMeasurement, String> {
    rollout(flat_params, config, seed)
        .map(|mean_cost| measurement_from_observed_mean_cost(policy_name, mean_cost))
        .map_err(|err| err.to_string())
}

pub fn evaluate_linear_policy(
    policy_name: &'static str,
    flat_params: &[f32],
    config: &LostSalesLinearRolloutConfig,
    seed: u64,
) -> Result<PolicyPerformanceMeasurement, String> {
    linear_rollout(flat_params, config, seed)
        .map(|mean_cost| measurement_from_observed_mean_cost(policy_name, mean_cost))
        .map_err(|err| err.to_string())
}

pub fn evaluate_neural_policy(
    policy_name: &'static str,
    flat_params: &[f32],
    config: &LostSalesNeuralRolloutConfig,
    seed: u64,
) -> Result<PolicyPerformanceMeasurement, String> {
    neural_rollout(flat_params, config, seed)
        .map(|mean_cost| measurement_from_observed_mean_cost(policy_name, mean_cost))
        .map_err(|err| err.to_string())
}

pub fn summarize_policy_measurements(
    reference_name: &'static str,
    horizon: usize,
    seed: u64,
    targets: &[PolicyPerformanceTarget],
    measurements: &[PolicyPerformanceMeasurement],
) -> PolicyPerformanceVerificationSummary {
    let measurement_map: HashMap<&'static str, f64> = measurements
        .iter()
        .map(|measurement| (measurement.policy_name, measurement.mean_cost))
        .collect();

    let results = targets
        .iter()
        .copied()
        .map(|target| {
            measurement_map
                .get(target.policy_name)
                .copied()
                .and_then(|observed_mean_cost| {
                    compare_observed_policy_cost(targets, target.policy_name, observed_mean_cost)
                })
                .unwrap_or(PolicyPerformanceVerificationResult {
                    target,
                    observed_mean_cost: None,
                    abs_gap: None,
                    within_tolerance: None,
                })
        })
        .collect::<Vec<_>>();

    let untargeted_measurements = measurements
        .iter()
        .filter(|measurement| target_for_policy_name(targets, measurement.policy_name).is_none())
        .cloned()
        .collect::<Vec<_>>();

    PolicyPerformanceVerificationSummary {
        reference_name,
        horizon,
        seed,
        results,
        untargeted_measurements,
    }
}

pub fn evaluate_heuristic_policy(
    config: LostSalesHeuristicVerificationConfig,
    policy: LostSalesHeuristicPolicyKind,
) -> Result<PolicyPerformanceMeasurement, String> {
    validate_verification_config(&config)?;
    let demand_mean = config.demand_config.implied_mean()?;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut demand_process = build_demand_process(config.demand_config, &mut rng)?;
    let mut env_state =
        initialize_state(demand_mean, config.lead_time, &mut rng, &mut demand_process);
    let mut heuristic = LostSalesHeuristicEvaluator::new(config)?;
    let mut epoch_costs = Vec::with_capacity(config.horizon);

    for _period in 0..config.horizon {
        let state = pipeline_state_with_inventory_folded_into_first_slot(
            env_state.current_inventory,
            &env_state.lead_time_orders,
        );
        let action = match policy {
            LostSalesHeuristicPolicyKind::Myopic1 => heuristic.myopic_1_order_quantity(&state)?.0,
            LostSalesHeuristicPolicyKind::Myopic2 => heuristic.myopic_2_order_quantity(&state)?.0,
            LostSalesHeuristicPolicyKind::StandardVectorBaseStock => {
                heuristic.standard_vector_base_stock_order_quantity(&state)?
            }
        };

        let arriving_order = env_state.lead_time_orders.remove(0);
        env_state.lead_time_orders.push(action);
        env_state.current_inventory = env_state
            .current_inventory
            .saturating_add(arriving_order.min(i64::MAX as usize) as i64);

        let demand = sample_demand(&mut rng, &mut demand_process);
        let cost = epoch_cost(
            &mut env_state.current_inventory,
            demand,
            action,
            config.holding_cost,
            config.shortage_cost,
            config.procurement_cost,
            config.fixed_order_cost,
        );
        epoch_costs.push(cost);
    }

    Ok(PolicyPerformanceMeasurement {
        policy_name: policy.policy_name(),
        mean_cost: mean_after_warmup_like_rollout(&epoch_costs, config.warm_up_periods_ratio),
    })
}

pub fn verify_policy_targets(
    config: LostSalesHeuristicVerificationConfig,
    targets: &[PolicyPerformanceTarget],
) -> Result<PolicyPerformanceVerificationSummary, String> {
    verify_policy_targets_with_additional_measurements(config, targets, &[])
}

pub fn verify_policy_targets_with_additional_measurements(
    config: LostSalesHeuristicVerificationConfig,
    targets: &[PolicyPerformanceTarget],
    additional_measurements: &[PolicyPerformanceMeasurement],
) -> Result<PolicyPerformanceVerificationSummary, String> {
    let mut measurements = HashMap::new();
    for policy in LostSalesHeuristicPolicyKind::all() {
        let measurement = evaluate_heuristic_policy(config, policy)?;
        measurements.insert(measurement.policy_name, measurement.mean_cost);
    }
    for measurement in additional_measurements {
        measurements.insert(measurement.policy_name, measurement.mean_cost);
    }

    let collected_measurements = measurements
        .into_iter()
        .map(|(policy_name, mean_cost)| measurement_from_observed_mean_cost(policy_name, mean_cost))
        .collect::<Vec<_>>();

    Ok(summarize_policy_measurements(
        config.reference_name,
        config.horizon,
        config.seed,
        targets,
        &collected_measurements,
    ))
}

pub fn verify_canonical_vanilla_l4_p4_poisson5_policy_targets(
) -> Result<PolicyPerformanceVerificationSummary, String> {
    verify_policy_targets(
        VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG,
        VANILLA_L4_P4_POISSON5_POLICY_TARGETS,
    )
}

fn validate_verification_config(
    config: &LostSalesHeuristicVerificationConfig,
) -> Result<(), String> {
    if config.lead_time < 1 {
        return Err(String::from("lead_time must be at least 1"));
    }
    if config.horizon < 1 {
        return Err(String::from("horizon must be at least 1"));
    }
    if !(0.0..=1.0).contains(&config.warm_up_periods_ratio) {
        return Err(String::from("warm_up_periods_ratio must be in [0, 1]"));
    }
    if !config.holding_cost.is_finite()
        || !config.shortage_cost.is_finite()
        || !config.procurement_cost.is_finite()
        || !config.fixed_order_cost.is_finite()
    {
        return Err(String::from("cost coefficients must be finite"));
    }
    if !matches!(
        config.demand_config.kind,
        LostSalesDemandKind::Poisson | LostSalesDemandKind::Geometric
    ) {
        return Err(String::from(
            "current lost-sales heuristic verification supports only IID Poisson and Geometric demand",
        ));
    }
    Ok(())
}

fn iid_demand_support(config: &LostSalesDemandConfig) -> Result<Vec<(usize, f64)>, String> {
    match config.kind {
        LostSalesDemandKind::Poisson => truncated_poisson_support(config.demand_rate),
        LostSalesDemandKind::Geometric => truncated_geometric_support(config.demand_rate),
        LostSalesDemandKind::MarkovModulatedPoisson2 => Err(String::from(
            "current lost-sales heuristic verification supports only IID demand",
        )),
    }
}

fn truncated_poisson_support(mean: f64) -> Result<Vec<(usize, f64)>, String> {
    if !mean.is_finite() || mean < 0.0 {
        return Err(String::from("Poisson mean must be finite and non-negative"));
    }
    if mean == 0.0 {
        return Ok(vec![(0, 1.0)]);
    }

    let mut support = Vec::new();
    let mut probability = (-mean).exp();
    let mut cumulative = probability;
    support.push((0, probability));

    for demand in 1..=10_000usize {
        if cumulative >= 1.0 - DEMAND_SUPPORT_TAIL_MASS {
            break;
        }
        probability *= mean / demand as f64;
        support.push((demand, probability));
        cumulative += probability;
    }

    normalize_support(&mut support)?;
    Ok(support)
}

fn truncated_geometric_support(mean: f64) -> Result<Vec<(usize, f64)>, String> {
    if !mean.is_finite() || mean < 0.0 {
        return Err(String::from(
            "Geometric mean must be finite and non-negative",
        ));
    }

    let success_probability = 1.0 / (1.0 + mean);
    let mut support = Vec::new();
    let mut probability = success_probability;
    let mut cumulative = probability;
    support.push((0, probability));

    for demand in 1..=100_000usize {
        if cumulative >= 1.0 - DEMAND_SUPPORT_TAIL_MASS {
            break;
        }
        probability *= 1.0 - success_probability;
        support.push((demand, probability));
        cumulative += probability;
    }

    normalize_support(&mut support)?;
    Ok(support)
}

fn normalize_support(support: &mut Vec<(usize, f64)>) -> Result<(), String> {
    let total_probability = support
        .iter()
        .map(|(_, probability)| *probability)
        .sum::<f64>();
    if !total_probability.is_finite() || total_probability <= 0.0 {
        return Err(String::from(
            "demand support probabilities must sum to a positive value",
        ));
    }
    for (_, probability) in support.iter_mut() {
        *probability /= total_probability;
    }
    Ok(())
}

fn cumulative_demand_cdf(
    config: &LostSalesDemandConfig,
    k: usize,
    periods: usize,
) -> Result<f64, String> {
    if periods < 1 {
        return Err(String::from("periods must be at least 1"));
    }
    match config.kind {
        LostSalesDemandKind::Poisson => {
            let distribution = Poisson::new(periods as f64 * config.demand_rate)
                .map_err(|err| format!("invalid Poisson mean {}: {err}", periods as f64 * config.demand_rate))?;
            Ok(distribution.cdf(k as u64))
        }
        LostSalesDemandKind::Geometric => cumulative_geometric_sum_cdf(config.demand_rate, k, periods),
        LostSalesDemandKind::MarkovModulatedPoisson2 => Err(String::from(
            "current lost-sales heuristic verification does not implement MMPP2 cumulative demand CDF",
        )),
    }
}

fn cumulative_geometric_sum_cdf(mean: f64, k: usize, periods: usize) -> Result<f64, String> {
    if !mean.is_finite() || mean < 0.0 {
        return Err(String::from(
            "Geometric mean must be finite and non-negative",
        ));
    }
    let success_probability = 1.0 / (1.0 + mean);
    let failure_probability = 1.0 - success_probability;
    let mut probability = success_probability.powi(periods as i32);
    let mut cumulative = probability;

    for demand in 1..=k {
        probability *= ((periods + demand - 1) as f64 / demand as f64) * failure_probability;
        cumulative += probability;
    }

    Ok(cumulative.clamp(0.0, 1.0))
}

fn critical_fractile(config: &LostSalesHeuristicVerificationConfig) -> f64 {
    (config.procurement_cost + config.holding_cost) / (config.holding_cost + config.shortage_cost)
}

fn mean_after_warmup_like_rollout(epoch_costs: &[f64], warm_up_periods_ratio: f64) -> f64 {
    let horizon = epoch_costs.len();
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let active_costs = if warm_up_periods < epoch_costs.len() {
        &epoch_costs[warm_up_periods..]
    } else {
        epoch_costs
    };
    active_costs.iter().sum::<f64>() / active_costs.len() as f64
}

fn order_pipeline_partial_sum(lead_time: usize, l: usize, state: &[usize]) -> usize {
    if l == lead_time {
        0
    } else {
        state[l..].iter().sum()
    }
}

fn pipeline_state_with_inventory_folded_into_first_slot(
    current_inventory: i64,
    lead_time_orders: &[usize],
) -> Vec<usize> {
    let mut state = lead_time_orders.to_vec();
    if let Some(first_slot) = state.first_mut() {
        *first_slot = first_slot.saturating_add(current_inventory.max(0) as usize);
    }
    state
}

fn state_with_appended_order(state: &[usize], order_quantity: usize) -> Vec<usize> {
    let mut extended_state = state.to_vec();
    extended_state.push(order_quantity);
    extended_state
}

fn state_after_demand_and_pipeline_advance(
    state: &[usize],
    demand: usize,
    lookahead_depth: usize,
) -> Vec<usize> {
    let mut next_state = state[..state.len() - 1].to_vec();
    next_state[0] = state[0].saturating_sub(demand).saturating_add(state[1]);
    for i in 1..lookahead_depth {
        next_state[i] = state[i + 1];
    }
    next_state
}

#[cfg(test)]
mod tests {
    use super::{
        compare_observed_policy_cost, evaluate_heuristic_policy, evaluate_linear_policy,
        measurement_from_observed_mean_cost, policy_targets_are_sorted_from_best_to_worst,
        summarize_policy_measurements, target_for_policy_name,
        verify_canonical_vanilla_l4_p4_poisson5_policy_targets, LostSalesHeuristicPolicyKind,
        PolicyPerformanceTarget, PolicyVerificationRole, VANILLA_L4_P4_POISSON5_POLICY_TARGETS,
        VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG,
    };
    use crate::core::policies::dense::DensePolicyHead;
    use crate::problems::lost_sales::env::StateNormalizer;
    use crate::problems::lost_sales::rollout::LostSalesLinearRolloutConfig;

    #[test]
    fn canonical_policy_targets_are_sorted_from_best_to_worst() {
        assert!(policy_targets_are_sorted_from_best_to_worst(
            VANILLA_L4_P4_POISSON5_POLICY_TARGETS
        ));
    }

    #[test]
    fn canonical_policy_targets_cover_optimal_and_heuristic_roles() {
        assert!(VANILLA_L4_P4_POISSON5_POLICY_TARGETS
            .iter()
            .any(|target| target.role == PolicyVerificationRole::OptimalReference));
        assert!(VANILLA_L4_P4_POISSON5_POLICY_TARGETS
            .iter()
            .any(|target| target.role == PolicyVerificationRole::Heuristic));
    }

    #[test]
    fn sorting_helper_rejects_descending_targets() {
        let targets = [
            PolicyPerformanceTarget {
                policy_name: "worse",
                role: PolicyVerificationRole::Heuristic,
                expected_mean_cost: 5.0,
                tolerance: 0.1,
            },
            PolicyPerformanceTarget {
                policy_name: "better",
                role: PolicyVerificationRole::Heuristic,
                expected_mean_cost: 4.5,
                tolerance: 0.1,
            },
        ];

        assert!(!policy_targets_are_sorted_from_best_to_worst(&targets));
    }

    #[test]
    fn target_lookup_and_gap_comparison_work() {
        let target = target_for_policy_name(VANILLA_L4_P4_POISSON5_POLICY_TARGETS, "myopic2")
            .expect("missing myopic2 target");
        let comparison = compare_observed_policy_cost(
            VANILLA_L4_P4_POISSON5_POLICY_TARGETS,
            "myopic2",
            target.expected_mean_cost + 0.01,
        )
        .expect("comparison should exist");

        assert_eq!(comparison.target.policy_name, "myopic2");
        assert!(comparison
            .within_tolerance
            .expect("within_tolerance should exist"));
        assert!(comparison.abs_gap.expect("abs_gap should exist") <= target.tolerance);
    }

    #[test]
    fn learned_policy_measurement_smoke_test_flows_into_summary() -> Result<(), String> {
        let rollout_config = LostSalesLinearRolloutConfig {
            input_dim: VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG.lead_time,
            output_dim: 8,
            policy_max_quantity: Some(7),
            state_scale: Some(20.0),
            state_normalizer: StateNormalizer::DivideByScale,
            policy_head: DensePolicyHead::CategoricalQuantity,
            demand_config: VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG.demand_config,
            lead_time: VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG.lead_time,
            holding_cost: VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG.holding_cost,
            shortage_cost: VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG.shortage_cost,
            procurement_cost: VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG.procurement_cost,
            fixed_order_cost: VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG.fixed_order_cost,
            horizon: 512,
            warm_up_periods_ratio: VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG.warm_up_periods_ratio,
        };
        let flat_params = vec![
            0.0_f32;
            rollout_config.output_dim * rollout_config.input_dim
                + rollout_config.output_dim
        ];
        let learned_measurement = evaluate_linear_policy(
            "linear_categorical_quantity_q8_smoke",
            &flat_params,
            &rollout_config,
            VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG.seed,
        )?;
        let targets = [PolicyPerformanceTarget {
            policy_name: "linear_categorical_quantity_q8_smoke",
            role: PolicyVerificationRole::LearnedPolicyThreshold,
            expected_mean_cost: learned_measurement.mean_cost,
            tolerance: 0.0,
        }];
        let summary = summarize_policy_measurements(
            "learned_policy_smoke",
            rollout_config.horizon,
            VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG.seed,
            &targets,
            &[measurement_from_observed_mean_cost(
                learned_measurement.policy_name,
                learned_measurement.mean_cost,
            )],
        );

        assert_eq!(
            summary.observed_mean_cost("linear_categorical_quantity_q8_smoke"),
            Some(learned_measurement.mean_cost)
        );
        assert!(summary.untargeted_measurements.is_empty());
        assert!(summary.all_executable_targets_within_tolerance());
        Ok(())
    }

    #[test]
    fn canonical_heuristic_measurements_follow_expected_ordering() -> Result<(), String> {
        let myopic2 = evaluate_heuristic_policy(
            VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG,
            LostSalesHeuristicPolicyKind::Myopic2,
        )?;
        let myopic1 = evaluate_heuristic_policy(
            VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG,
            LostSalesHeuristicPolicyKind::Myopic1,
        )?;
        let svbs = evaluate_heuristic_policy(
            VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG,
            LostSalesHeuristicPolicyKind::StandardVectorBaseStock,
        )?;

        assert!(myopic2.mean_cost < myopic1.mean_cost);
        assert!(myopic1.mean_cost < svbs.mean_cost);
        Ok(())
    }

    #[test]
    fn canonical_heuristic_verification_matches_literature_targets() -> Result<(), String> {
        let summary = verify_canonical_vanilla_l4_p4_poisson5_policy_targets()?;

        assert!(summary.executable_targets_are_sorted_from_best_to_worst());
        assert!(summary.all_executable_targets_within_tolerance());
        assert!(summary
            .results
            .iter()
            .any(|result| result.target.policy_name == "optimal_reference"
                && result.observed_mean_cost.is_none()));
        assert!(summary
            .results
            .iter()
            .any(|result| result.target.policy_name == "capped_base_stock"
                && result.observed_mean_cost.is_none()));
        Ok(())
    }
}
