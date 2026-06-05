#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IssuanceRule {
    Fifo,
    Lifo,
    FixedPriority(Vec<String>),
    Configurable(Vec<String>),
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShortageReaction {
    LostSales,
    Backorder,
    EmergencyFulfillment,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceSpec {
    pub name: String,
    pub demand_target: String,
    pub inventory_sources: Vec<String>,
    pub issuance_rule: IssuanceRule,
    pub shortage_reaction: ShortageReaction,
}
