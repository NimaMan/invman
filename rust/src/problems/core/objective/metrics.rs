#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PerformanceMetric {
    FillRate,
    CycleServiceLevel,
    AverageInventory,
    AverageBacklog,
    TotalCost,
    Custom(String),
}
