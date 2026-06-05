use crate::problems::core::stochastic::demand::DemandProcessSpec;
use crate::problems::core::stochastic::disruption::{
    DisruptionProcessSpec, TransitDelayProcessSpec,
};
use crate::problems::core::stochastic::failure::FailureProcessSpec;
use crate::problems::core::stochastic::forecast::ForecastProcessSpec;
use crate::problems::core::stochastic::r#yield::YieldProcessSpec;
use crate::problems::core::stochastic::return_flow::ReturnArrivalProcessSpec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomProcessSpec {
    pub target: String,
    pub model: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StochasticProcess {
    Demand(DemandProcessSpec),
    Yield(YieldProcessSpec),
    Failure(FailureProcessSpec),
    ReturnArrival(ReturnArrivalProcessSpec),
    ForecastEvolution(ForecastProcessSpec),
    Disruption(DisruptionProcessSpec),
    TransitDelay(TransitDelayProcessSpec),
    Custom(CustomProcessSpec),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StochasticLayer {
    pub processes: Vec<StochasticProcess>,
}

impl StochasticLayer {
    pub fn has_random_events(&self) -> bool {
        !self.processes.is_empty()
    }
}
