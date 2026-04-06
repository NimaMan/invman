use crate::problems::core::control::ControlLayer;
use crate::problems::core::objective::ObjectiveLayer;
use crate::problems::core::physical::PhysicalLayer;
use crate::problems::core::stochastic::StochasticLayer;
use crate::problems::core::timing::TimingLayer;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FundamentalQuestion {
    InventoryStates,
    MaterialMovement,
    RandomEvents,
    ControllerChoices,
    ControllerObservations,
    PerformanceScoring,
    TimingAndConstraints,
}

impl FundamentalQuestion {
    pub const fn prompt(self) -> &'static str {
        match self {
            Self::InventoryStates => "What inventory states exist?",
            Self::MaterialMovement => "How can material move or transform?",
            Self::RandomEvents => "What random events occur?",
            Self::ControllerChoices => "What can the controller choose?",
            Self::ControllerObservations => "What can the controller observe, and when?",
            Self::PerformanceScoring => "How is performance scored?",
            Self::TimingAndConstraints => {
                "What timing rules and feasibility constraints shape the system?"
            }
        }
    }
}

pub const FUNDAMENTAL_QUESTIONS: [FundamentalQuestion; 7] = [
    FundamentalQuestion::InventoryStates,
    FundamentalQuestion::MaterialMovement,
    FundamentalQuestion::RandomEvents,
    FundamentalQuestion::ControllerChoices,
    FundamentalQuestion::ControllerObservations,
    FundamentalQuestion::PerformanceScoring,
    FundamentalQuestion::TimingAndConstraints,
];

#[derive(Clone, Debug, PartialEq)]
pub struct InventoryProblemBlueprint {
    pub name: String,
    pub physical: PhysicalLayer,
    pub stochastic: StochasticLayer,
    pub control: ControlLayer,
    pub objective: ObjectiveLayer,
    pub timing: TimingLayer,
}

impl InventoryProblemBlueprint {
    pub fn fundamental_questions() -> &'static [FundamentalQuestion] {
        &FUNDAMENTAL_QUESTIONS
    }

    pub fn answers_all_fundamental_questions(&self) -> bool {
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
