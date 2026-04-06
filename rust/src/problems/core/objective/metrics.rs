#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PerformanceMetric {
    FillRate,
    CycleServiceLevel,
    AverageInventory,
    AverageWaste,
    AverageBacklog,
    TotalCost,
    Custom(String),
}
