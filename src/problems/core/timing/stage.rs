#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    StartOfPeriod,
    AfterReceipts,
    AfterAction,
    AfterTransformations,
    AfterDemand,
    EndOfPeriod,
    Custom(String),
}
