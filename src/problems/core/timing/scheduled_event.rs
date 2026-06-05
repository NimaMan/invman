use crate::problems::core::timing::stage::Stage;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledEvent {
    pub stage: Stage,
    pub event: String,
}
