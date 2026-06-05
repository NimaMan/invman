#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObservationMode {
    FullState,
    LocalState,
    ReducedState,
    ForecastAugmented,
    Delayed,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservationSpec {
    pub name: String,
    pub mode: ObservationMode,
    pub channels: Vec<String>,
}
