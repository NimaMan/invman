use crate::problems::core::flownet::{
    summarize_policy_performance, PolicyPerformanceMeasurement, PolicyPerformanceTarget,
    PolicyPerformanceVerificationSummary, PolicyScoreOrdering, PolicyVerificationRole,
};
use crate::problems::perishable_inventory::heuristics::base_stock_order_quantity;
use crate::problems::perishable_inventory::references::get_primary_reference_instance;
use crate::problems::perishable_inventory::value_iteration_mdp::{
    best_base_stock_level_by_expected_return, build_exact_mdp,
    expected_discounted_return_from_zero_state, value_iteration_best_action_values,
};

pub const PRIMARY_REFERENCE_OPTIMAL_POLICY_NAME: &str = "optimal_reference";
pub const PRIMARY_REFERENCE_BASE_STOCK_POLICY_NAME: &str = "base_stock";

#[derive(Clone, Debug, PartialEq)]
pub struct PerishablePolicyPerformanceReport {
    pub summary: PolicyPerformanceVerificationSummary,
    pub exact_best_base_stock_level: usize,
    pub published_base_stock_level: usize,
}

pub fn verify_primary_reference_policy_performance(
) -> Result<PerishablePolicyPerformanceReport, String> {
    let reference = get_primary_reference_instance();
    let published_returns = reference
        .published_scenario_a_returns
        .ok_or_else(|| String::from("primary reference is missing published Scenario A returns"))?;
    let published_figure = reference.published_figure3_verification.ok_or_else(|| {
        String::from("primary reference is missing published Figure 3 base-stock verification")
    })?;

    let mdp = build_exact_mdp(reference.name);
    let (optimal_policy, _) = value_iteration_best_action_values(&mdp, 0.99);
    let optimal_return =
        expected_discounted_return_from_zero_state(reference.name, &mdp, &optimal_policy);

    let best_base_stock_level = best_base_stock_level_by_expected_return(reference.name, &mdp);
    let base_stock_policy = mdp
        .states
        .iter()
        .map(|state| {
            base_stock_order_quantity(state, best_base_stock_level, reference.max_order_size)
        })
        .collect::<Vec<_>>();
    let best_base_stock_return =
        expected_discounted_return_from_zero_state(reference.name, &mdp, &base_stock_policy);

    let targets = vec![
        PolicyPerformanceTarget {
            policy_name: String::from(PRIMARY_REFERENCE_OPTIMAL_POLICY_NAME),
            role: PolicyVerificationRole::OptimalReference,
            expected_score: published_returns.value_iteration_mean_return as f64,
            tolerance: 1e-9,
        },
        PolicyPerformanceTarget {
            policy_name: String::from(PRIMARY_REFERENCE_BASE_STOCK_POLICY_NAME),
            role: PolicyVerificationRole::Heuristic,
            expected_score: published_returns.best_base_stock_mean_return as f64,
            tolerance: 1.0,
        },
    ];
    let measurements = vec![
        PolicyPerformanceMeasurement {
            policy_name: String::from(PRIMARY_REFERENCE_OPTIMAL_POLICY_NAME),
            observed_score: optimal_return.round(),
        },
        PolicyPerformanceMeasurement {
            policy_name: String::from(PRIMARY_REFERENCE_BASE_STOCK_POLICY_NAME),
            observed_score: best_base_stock_return.round(),
        },
    ];

    Ok(PerishablePolicyPerformanceReport {
        summary: summarize_policy_performance(
            reference.name,
            Some(reference.eval_horizon),
            PolicyScoreOrdering::HigherIsBetter,
            targets,
            measurements,
            vec![],
        ),
        exact_best_base_stock_level: best_base_stock_level,
        published_base_stock_level: published_figure.published_base_stock_level,
    })
}

#[cfg(test)]
mod tests {
    use super::verify_primary_reference_policy_performance;

    #[test]
    fn primary_reference_policy_performance_matches_published_targets() {
        let report = verify_primary_reference_policy_performance()
            .expect("perishable FlowNet policy verification must succeed");

        assert_eq!(
            report.exact_best_base_stock_level,
            report.published_base_stock_level
        );
        assert!(report.summary.all_observed_targets_within_tolerance());
        assert!(report
            .summary
            .observed_targets_are_sorted_from_best_to_worst());
    }
}
