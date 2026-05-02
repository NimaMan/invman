use crate::problems::core::flownet::{
    summarize_policy_performance, PolicyPerformanceMeasurement, PolicyPerformanceTarget,
    PolicyPerformanceVerificationSummary, PolicyScoreOrdering, PolicyVerificationRole,
};
use crate::problems::network_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::network_inventory::literature::VERIFICATION_PROBLEM_INSTANCE;

pub const EXACT_VERIFICATION_OPTIMAL_POLICY_NAME: &str = "optimal_reference";
pub const EXACT_VERIFICATION_PAIRWISE_BASE_STOCK_POLICY_NAME: &str = "pairwise_base_stock";

pub fn verify_exact_reference_policy_performance(
) -> Result<PolicyPerformanceVerificationSummary, String> {
    let optimal =
        solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE).map_err(|err| err.to_string())?;
    let pairwise_base_stock = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        EXACT_VERIFICATION_PAIRWISE_BASE_STOCK_POLICY_NAME,
    )
    .map_err(|err| err.to_string())?;

    Ok(summarize_policy_performance(
        "network_inventory_exact_verification_reference",
        Some(VERIFICATION_PROBLEM_INSTANCE.periods),
        PolicyScoreOrdering::LowerIsBetter,
        vec![
            PolicyPerformanceTarget {
                policy_name: String::from(EXACT_VERIFICATION_OPTIMAL_POLICY_NAME),
                role: PolicyVerificationRole::OptimalReference,
                expected_score: optimal.discounted_cost,
                tolerance: 1e-9,
            },
            PolicyPerformanceTarget {
                policy_name: String::from(EXACT_VERIFICATION_PAIRWISE_BASE_STOCK_POLICY_NAME),
                role: PolicyVerificationRole::Heuristic,
                expected_score: pairwise_base_stock.discounted_cost,
                tolerance: 1e-9,
            },
        ],
        vec![
            PolicyPerformanceMeasurement {
                policy_name: String::from(EXACT_VERIFICATION_OPTIMAL_POLICY_NAME),
                observed_score: optimal.discounted_cost,
            },
            PolicyPerformanceMeasurement {
                policy_name: String::from(EXACT_VERIFICATION_PAIRWISE_BASE_STOCK_POLICY_NAME),
                observed_score: pairwise_base_stock.discounted_cost,
            },
        ],
        vec![],
    ))
}

#[cfg(test)]
mod tests {
    use super::verify_exact_reference_policy_performance;

    #[test]
    fn exact_reference_policy_performance_matches_reference_targets() {
        let summary = verify_exact_reference_policy_performance()
            .expect("network FlowNet policy verification must succeed");

        assert!(summary.all_observed_targets_within_tolerance());
        assert!(summary.observed_targets_are_sorted_from_best_to_worst());
    }
}
