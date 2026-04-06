use crate::problems::core::control::ControlLayer;
use crate::problems::core::flownet::question::{FlowNetQuestion, FLOWNET_QUESTIONS};
use crate::problems::core::objective::ObjectiveLayer;
use crate::problems::core::physical::PhysicalLayer;
use crate::problems::core::stochastic::StochasticLayer;
use crate::problems::core::timing::TimingLayer;

#[derive(Clone, Debug, PartialEq)]
pub struct FlowNetFormulation {
    pub name: String,
    pub physical: PhysicalLayer,
    pub stochastic: StochasticLayer,
    pub control: ControlLayer,
    pub objective: ObjectiveLayer,
    pub timing: TimingLayer,
}

impl FlowNetFormulation {
    pub fn questions() -> &'static [FlowNetQuestion] {
        &FLOWNET_QUESTIONS
    }

    pub fn answers_all_questions(&self) -> bool {
        self.physical.has_inventory_states()
            && self.physical.has_material_movement()
            && self.stochastic.has_random_events()
            && self.control.has_actions()
            && self.control.has_observations()
            && self.objective.has_scoring_terms()
            && self.timing.has_schedule()
            && self.timing.schedule_references_known_events()
    }
}
