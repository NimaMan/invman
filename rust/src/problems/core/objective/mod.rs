pub mod cost_term;
pub mod discounting;
pub mod metrics;
pub mod reward;

#[allow(unused_imports)]
pub use cost_term::ObjectiveTerm;
#[allow(unused_imports)]
pub use discounting::Discounting;
#[allow(unused_imports)]
pub use metrics::PerformanceMetric;
#[allow(unused_imports)]
pub use reward::RewardConvention;

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectiveLayer {
    pub terms: Vec<ObjectiveTerm>,
    pub discounting: Discounting,
    pub reward_convention: RewardConvention,
    pub tracked_metrics: Vec<PerformanceMetric>,
}

impl ObjectiveLayer {
    pub fn has_scoring_terms(&self) -> bool {
        !self.terms.is_empty()
    }
}
