// Flownet verification of lost-sales policy performance.
//
// This module owns the VERIFICATION machinery only: literature targets, target
// roles, the comparison/summary types, and the `verify_*` entry points that run
// a sweep and check observed mean costs against trusted numbers. The actual
// heuristic POLICY logic (Myopic-1 / Myopic-2 / SVBS evaluator, demand support,
// the canonical vanilla instance config, and `evaluate_heuristic_policy`) lives
// in `crate::problems::lost_sales::heuristics` and is reused here, so there is a
// single source of truth for the heuristics.
//
// In addition to the heuristic rollouts this module provides thin wrappers to
// measure learned policies (soft-tree, linear, neural) via the rollout crate, so
// they can be folded into the same target/summary comparison.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::problems::lost_sales::rollout::{
    linear_rollout, neural_rollout, rollout, LostSalesLinearRolloutConfig,
    LostSalesNeuralRolloutConfig, LostSalesRolloutConfig,
};

// Re-export the heuristic policy/config/measurement types, the rollout entry
// point, and the canonical vanilla instance from the dedicated `heuristics`
// module so existing `verification::*` / `flownet::*` consumers keep working
// unchanged against this single source of truth.
pub use crate::problems::lost_sales::heuristics::{
    evaluate_heuristic_policy, measurement_from_observed_mean_cost, LostSalesHeuristicPolicyKind,
    LostSalesHeuristicVerificationConfig, PolicyPerformanceMeasurement,
    VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG, VANILLA_L4_P4_POISSON5_VERIFICATION_HORIZON,
    VANILLA_L4_P4_POISSON5_VERIFICATION_REFERENCE, VANILLA_L4_P4_POISSON5_VERIFICATION_SEED,
};

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
