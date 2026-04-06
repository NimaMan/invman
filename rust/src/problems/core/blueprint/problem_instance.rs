#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstanceParameter {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InventoryProblemInstance {
    pub name: String,
    pub blueprint_name: String,
    pub parameters: Vec<InstanceParameter>,
    pub horizon_periods: Option<usize>,
    pub notes: Vec<String>,
}
