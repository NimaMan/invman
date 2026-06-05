#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExogenousEventKind {
    DemandArrival,
    FailureOccurrence,
    YieldRealization,
    ReturnArrival,
    ForecastUpdate,
    DisruptionStart,
    DisruptionEnd,
    TransitDelay,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControlEventKind {
    ProcurementDecision,
    ShipmentDecision,
    AllocationDecision,
    RemovalDecision,
    PricingDecision,
    ReserveReleaseDecision,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaterialEventKind {
    Receipt,
    Dispatch,
    Delivery,
    Transfer,
    ReturnReceipt,
    RepairCompletion,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransformationEventKind {
    Aging,
    Decay,
    RepairStart,
    Refinement,
    Reclassification,
    Conversion,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServiceEventKind {
    DemandServed,
    DemandBackordered,
    DemandLost,
    EmergencyFulfillment,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccountingEventKind {
    HoldingCostCharged,
    BacklogCostCharged,
    LostSalesPenaltyCharged,
    ProcurementCostCharged,
    FixedOrderCostCharged,
    WasteCostCharged,
    SalvageCreditApplied,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventKind {
    Exogenous(ExogenousEventKind),
    Control(ControlEventKind),
    Material(MaterialEventKind),
    Transformation(TransformationEventKind),
    Service(ServiceEventKind),
    Accounting(AccountingEventKind),
}
