#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisruptionProcessSpec {
    pub target: String,
    pub model: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitDelayProcessSpec {
    pub target: String,
    pub model: String,
}
