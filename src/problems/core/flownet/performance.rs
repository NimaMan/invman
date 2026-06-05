#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyVerificationRole {
    OptimalReference,
    Heuristic,
    LearnedPolicyThreshold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyScoreOrdering {
    LowerIsBetter,
    HigherIsBetter,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyPerformanceTarget {
    pub policy_name: String,
    pub role: PolicyVerificationRole,
    pub expected_score: f64,
    pub tolerance: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyPerformanceMeasurement {
    pub policy_name: String,
    pub observed_score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyPerformanceVerificationResult {
    pub target: PolicyPerformanceTarget,
    pub observed_score: Option<f64>,
    pub abs_gap: Option<f64>,
    pub within_tolerance: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyPerformanceVerificationSummary {
    pub reference_name: String,
    pub horizon_periods: Option<usize>,
    pub score_ordering: PolicyScoreOrdering,
    pub results: Vec<PolicyPerformanceVerificationResult>,
    pub untargeted_measurements: Vec<PolicyPerformanceMeasurement>,
}

impl PolicyPerformanceVerificationSummary {
    pub fn observed_score(&self, policy_name: &str) -> Option<f64> {
        self.results
            .iter()
            .find(|result| result.target.policy_name == policy_name)
            .and_then(|result| result.observed_score)
    }

    pub fn all_observed_targets_within_tolerance(&self) -> bool {
        self.results
            .iter()
            .filter(|result| result.observed_score.is_some())
            .all(|result| result.within_tolerance.unwrap_or(false))
    }

    pub fn observed_results(&self) -> Vec<&PolicyPerformanceVerificationResult> {
        self.results
            .iter()
            .filter(|result| result.observed_score.is_some())
            .collect()
    }

    pub fn observed_targets_are_sorted_from_best_to_worst(&self) -> bool {
        let observed = self.observed_results();
        observed.windows(2).all(|window| match self.score_ordering {
            PolicyScoreOrdering::LowerIsBetter => {
                window[0].observed_score.unwrap_or(f64::INFINITY)
                    <= window[1].observed_score.unwrap_or(f64::INFINITY)
            }
            PolicyScoreOrdering::HigherIsBetter => {
                window[0].observed_score.unwrap_or(f64::NEG_INFINITY)
                    >= window[1].observed_score.unwrap_or(f64::NEG_INFINITY)
            }
        })
    }
}

pub fn summarize_policy_performance(
    reference_name: impl Into<String>,
    horizon_periods: Option<usize>,
    score_ordering: PolicyScoreOrdering,
    targets: Vec<PolicyPerformanceTarget>,
    measurements: Vec<PolicyPerformanceMeasurement>,
    untargeted_measurements: Vec<PolicyPerformanceMeasurement>,
) -> PolicyPerformanceVerificationSummary {
    let results = targets
        .into_iter()
        .map(|target| {
            let measurement = measurements
                .iter()
                .find(|measurement| measurement.policy_name == target.policy_name);
            let observed_score = measurement.map(|measurement| measurement.observed_score);
            let abs_gap = observed_score.map(|score| (score - target.expected_score).abs());
            let within_tolerance = abs_gap.map(|gap| gap <= target.tolerance);
            PolicyPerformanceVerificationResult {
                target,
                observed_score,
                abs_gap,
                within_tolerance,
            }
        })
        .collect::<Vec<_>>();

    PolicyPerformanceVerificationSummary {
        reference_name: reference_name.into(),
        horizon_periods,
        score_ordering,
        results,
        untargeted_measurements,
    }
}
