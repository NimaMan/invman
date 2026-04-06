#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StockRole {
    OnHand,
    Pipeline,
    Backlog,
    Reserve,
    WorkInProcess,
    DemandSink,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StockNodeSpec {
    pub name: String,
    pub role: StockRole,
    pub attributes: Vec<String>,
}
