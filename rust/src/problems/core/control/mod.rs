pub mod action;
pub mod constraints;
pub mod observation;
pub mod service;

#[allow(unused_imports)]
pub use action::{ActionShape, ActionSpec};
#[allow(unused_imports)]
pub use constraints::FeasibilityConstraint;
#[allow(unused_imports)]
pub use observation::{ObservationMode, ObservationSpec};
#[allow(unused_imports)]
pub use service::{IssuanceRule, ServiceSpec, ShortageReaction};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlLayer {
    pub actions: Vec<ActionSpec>,
    pub observations: Vec<ObservationSpec>,
    pub service_policies: Vec<ServiceSpec>,
    pub feasibility_constraints: Vec<FeasibilityConstraint>,
}

impl ControlLayer {
    pub fn has_actions(&self) -> bool {
        !self.actions.is_empty()
    }

    pub fn has_observations(&self) -> bool {
        !self.observations.is_empty()
    }

    pub fn has_service_policies(&self) -> bool {
        !self.service_policies.is_empty()
    }
}
