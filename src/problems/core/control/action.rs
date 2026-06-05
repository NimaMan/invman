#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionShape {
    ScalarOrder,
    VectorOrder,
    Allocation,
    Routing,
    DualSourceOrderPair,
    PurchaseAndRemoval,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionSpec {
    pub name: String,
    pub target: String,
    pub shape: ActionShape,
}
