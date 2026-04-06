#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlowMode {
    Procurement,
    Shipment,
    Transformation,
    Aging,
    Repair,
    Removal,
    Return,
    DemandFulfillment,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlowEdgeSpec {
    pub name: String,
    pub from: String,
    pub to: String,
    pub mode: FlowMode,
}
