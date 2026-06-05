#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlowNetParameter {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlowNetInstance {
    pub name: String,
    pub flownet_name: String,
    pub parameters: Vec<FlowNetParameter>,
    pub horizon_periods: Option<usize>,
    pub notes: Vec<String>,
}
