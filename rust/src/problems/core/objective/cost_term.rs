#[derive(Clone, Debug, PartialEq)]
pub enum ObjectiveTerm {
    HoldingCost { target: String },
    BacklogCost { target: String },
    LostSalesPenalty { target: String },
    ProcurementCost { target: String },
    FixedOrderCost { target: String },
    WasteCost { target: String },
    SalvageCredit { target: String },
    EmergencyFulfillmentCost { target: String },
    Custom { name: String, target: String },
}
