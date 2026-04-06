#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventSemantics {
    pub event_name: String,
    pub preconditions: Vec<String>,
    pub postconditions: Vec<String>,
    pub accounting_implications: Vec<String>,
}
