#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlowNetQuestion {
    InventoryStates,
    MaterialMovement,
    RandomEvents,
    ControllerChoices,
    ControllerObservations,
    PerformanceScoring,
    TimingAndConstraints,
}

impl FlowNetQuestion {
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

pub const FLOWNET_QUESTIONS: [FlowNetQuestion; 7] = [
    FlowNetQuestion::InventoryStates,
    FlowNetQuestion::MaterialMovement,
    FlowNetQuestion::RandomEvents,
    FlowNetQuestion::ControllerChoices,
    FlowNetQuestion::ControllerObservations,
    FlowNetQuestion::PerformanceScoring,
    FlowNetQuestion::TimingAndConstraints,
];
