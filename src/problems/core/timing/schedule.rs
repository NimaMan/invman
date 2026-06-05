use crate::problems::core::events::EventCatalog;
use crate::problems::core::timing::constraints::TimingConstraint;
use crate::problems::core::timing::scheduled_event::ScheduledEvent;
use crate::problems::core::timing::stage::Stage;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimingLayer {
    pub events: EventCatalog,
    pub stages: Vec<Stage>,
    pub schedule: Vec<ScheduledEvent>,
    pub feasibility_constraints: Vec<TimingConstraint>,
}

impl TimingLayer {
    pub fn has_schedule(&self) -> bool {
        !self.stages.is_empty() && !self.schedule.is_empty() && self.events.has_events()
    }

    pub fn schedule_references_known_events(&self) -> bool {
        self.schedule
            .iter()
            .all(|scheduled| self.events.contains_named_event(&scheduled.event))
    }
}
