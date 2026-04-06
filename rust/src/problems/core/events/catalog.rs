use crate::problems::core::events::kind::EventKind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventSpec {
    pub name: String,
    pub kind: EventKind,
    pub source: Option<String>,
    pub target: Option<String>,
    pub notes: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EventCatalog {
    pub events: Vec<EventSpec>,
}

impl EventCatalog {
    pub fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    pub fn contains_named_event(&self, event_name: &str) -> bool {
        self.events.iter().any(|event| event.name == event_name)
    }
}
