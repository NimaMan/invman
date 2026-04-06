use crate::problems::core::control::{
    ActionShape, ActionSpec, ControlLayer, FeasibilityConstraint, ObservationMode, ObservationSpec,
};
use crate::problems::core::events::{
    AccountingEventKind, ControlEventKind, EventCatalog, EventKind, EventSpec, ExogenousEventKind,
    MaterialEventKind, ServiceEventKind,
};
use crate::problems::core::flownet::FlowNetFormulation;
use crate::problems::core::objective::{
    Discounting, ObjectiveLayer, ObjectiveTerm, PerformanceMetric, RewardConvention,
};
use crate::problems::core::physical::{
    FlowEdgeSpec, FlowMode, MaterialAttribute, PhysicalLayer, PipelineSpec, StockNodeSpec,
    StockRole, Topology,
};
use crate::problems::core::stochastic::{DemandProcessSpec, StochasticLayer, StochasticProcess};
use crate::problems::core::timing::{ScheduledEvent, Stage, TimingConstraint, TimingLayer};

pub const LOST_SALES_FLOWNET_NAME: &str = "lost_sales";

pub fn canonical_lost_sales_flownet() -> FlowNetFormulation {
    FlowNetFormulation {
        name: String::from(LOST_SALES_FLOWNET_NAME),
        physical: PhysicalLayer {
            topology: Topology::SingleLocation,
            stock_nodes: vec![
                StockNodeSpec {
                    name: String::from("on_hand_inventory"),
                    role: StockRole::OnHand,
                    attributes: vec![String::from("single_item"), String::from("nonperishable")],
                },
                StockNodeSpec {
                    name: String::from("customer_demand_sink"),
                    role: StockRole::DemandSink,
                    attributes: vec![String::from("lost_sales_service")],
                },
            ],
            pipelines: vec![PipelineSpec {
                name: String::from("inbound_pipeline"),
                from: String::from("supplier"),
                to: String::from("on_hand_inventory"),
                stages: 1,
            }],
            flow_edges: vec![
                FlowEdgeSpec {
                    name: String::from("supplier_to_pipeline"),
                    from: String::from("supplier"),
                    to: String::from("inbound_pipeline"),
                    mode: FlowMode::Procurement,
                },
                FlowEdgeSpec {
                    name: String::from("pipeline_to_stock"),
                    from: String::from("inbound_pipeline"),
                    to: String::from("on_hand_inventory"),
                    mode: FlowMode::Shipment,
                },
                FlowEdgeSpec {
                    name: String::from("stock_to_customer"),
                    from: String::from("on_hand_inventory"),
                    to: String::from("customer_demand_sink"),
                    mode: FlowMode::DemandFulfillment,
                },
            ],
            material_attributes: vec![
                MaterialAttribute {
                    name: String::from("item_class"),
                    allowed_values: vec![String::from("single_sku")],
                },
                MaterialAttribute {
                    name: String::from("service_semantics"),
                    allowed_values: vec![String::from("lost_sales")],
                },
            ],
        },
        stochastic: StochasticLayer {
            processes: vec![StochasticProcess::Demand(DemandProcessSpec {
                target: String::from("customer_demand_sink"),
                model: String::from(
                    "configurable LostSalesDemandProcess: Poisson | Geometric | MarkovModulatedPoisson2",
                ),
            })],
        },
        control: ControlLayer {
            actions: vec![ActionSpec {
                name: String::from("replenishment_order"),
                target: String::from("inbound_pipeline"),
                shape: ActionShape::ScalarOrder,
            }],
            observations: vec![ObservationSpec {
                name: String::from("pipeline_state"),
                mode: ObservationMode::FullState,
                channels: vec![
                    String::from("current_inventory"),
                    String::from("lead_time_orders"),
                    String::from("pipeline_vector_with_inventory_folded_into_first_slot"),
                ],
            }],
            feasibility_constraints: vec![
                FeasibilityConstraint {
                    name: String::from("nonnegative_order"),
                    description: String::from("the replenishment action must be a nonnegative integer"),
                },
                FeasibilityConstraint {
                    name: String::from("positive_lead_time"),
                    description: String::from("the lost-sales implementation assumes lead_time >= 1"),
                },
            ],
        },
        objective: ObjectiveLayer {
            terms: vec![
                ObjectiveTerm::HoldingCost {
                    target: String::from("on_hand_inventory"),
                },
                ObjectiveTerm::LostSalesPenalty {
                    target: String::from("customer_demand_sink"),
                },
                ObjectiveTerm::ProcurementCost {
                    target: String::from("replenishment_order"),
                },
                ObjectiveTerm::FixedOrderCost {
                    target: String::from("replenishment_order"),
                },
            ],
            discounting: Discounting::None,
            reward_convention: RewardConvention::MinimizeCost,
            tracked_metrics: vec![
                PerformanceMetric::TotalCost,
                PerformanceMetric::FillRate,
                PerformanceMetric::AverageInventory,
            ],
        },
        timing: TimingLayer {
            events: EventCatalog {
                events: vec![
                    EventSpec {
                        name: String::from("replenishment_decision"),
                        kind: EventKind::Control(ControlEventKind::ProcurementDecision),
                        source: None,
                        target: Some(String::from("inbound_pipeline")),
                        notes: Some(String::from(
                            "the controller observes the pipeline state and chooses an order quantity",
                        )),
                    },
                    EventSpec {
                        name: String::from("inbound_receipt"),
                        kind: EventKind::Material(MaterialEventKind::Receipt),
                        source: Some(String::from("inbound_pipeline")),
                        target: Some(String::from("on_hand_inventory")),
                        notes: Some(String::from("the oldest pipeline order reaches stock")),
                    },
                    EventSpec {
                        name: String::from("order_dispatch"),
                        kind: EventKind::Material(MaterialEventKind::Dispatch),
                        source: Some(String::from("supplier")),
                        target: Some(String::from("inbound_pipeline")),
                        notes: Some(String::from("the new order is appended to the pipeline tail")),
                    },
                    EventSpec {
                        name: String::from("customer_demand_arrival"),
                        kind: EventKind::Exogenous(ExogenousEventKind::DemandArrival),
                        source: None,
                        target: Some(String::from("customer_demand_sink")),
                        notes: Some(String::from("demand is sampled from the configured demand process")),
                    },
                    EventSpec {
                        name: String::from("customer_demand_served"),
                        kind: EventKind::Service(ServiceEventKind::DemandServed),
                        source: Some(String::from("on_hand_inventory")),
                        target: Some(String::from("customer_demand_sink")),
                        notes: Some(String::from("available stock is used to satisfy demand first")),
                    },
                    EventSpec {
                        name: String::from("customer_demand_lost"),
                        kind: EventKind::Service(ServiceEventKind::DemandLost),
                        source: Some(String::from("customer_demand_sink")),
                        target: None,
                        notes: Some(String::from("unmet demand leaves the system immediately")),
                    },
                    EventSpec {
                        name: String::from("procurement_cost_accounted"),
                        kind: EventKind::Accounting(AccountingEventKind::ProcurementCostCharged),
                        source: Some(String::from("replenishment_order")),
                        target: None,
                        notes: Some(String::from("variable procurement cost is charged on the action")),
                    },
                    EventSpec {
                        name: String::from("fixed_order_cost_accounted"),
                        kind: EventKind::Accounting(AccountingEventKind::FixedOrderCostCharged),
                        source: Some(String::from("replenishment_order")),
                        target: None,
                        notes: Some(String::from("fixed order cost is charged if the action is positive")),
                    },
                    EventSpec {
                        name: String::from("holding_cost_accounted"),
                        kind: EventKind::Accounting(AccountingEventKind::HoldingCostCharged),
                        source: Some(String::from("on_hand_inventory")),
                        target: None,
                        notes: Some(String::from("holding cost is charged on ending on-hand inventory")),
                    },
                    EventSpec {
                        name: String::from("lost_sales_penalty_accounted"),
                        kind: EventKind::Accounting(AccountingEventKind::LostSalesPenaltyCharged),
                        source: Some(String::from("customer_demand_sink")),
                        target: None,
                        notes: Some(String::from("lost-sales penalty is charged on unmet demand")),
                    },
                ],
            },
            stages: vec![
                Stage::StartOfPeriod,
                Stage::AfterReceipts,
                Stage::AfterAction,
                Stage::AfterDemand,
                Stage::EndOfPeriod,
            ],
            schedule: vec![
                ScheduledEvent {
                    stage: Stage::StartOfPeriod,
                    event: String::from("replenishment_decision"),
                },
                ScheduledEvent {
                    stage: Stage::AfterReceipts,
                    event: String::from("inbound_receipt"),
                },
                ScheduledEvent {
                    stage: Stage::AfterAction,
                    event: String::from("order_dispatch"),
                },
                ScheduledEvent {
                    stage: Stage::AfterDemand,
                    event: String::from("customer_demand_arrival"),
                },
                ScheduledEvent {
                    stage: Stage::AfterDemand,
                    event: String::from("customer_demand_served"),
                },
                ScheduledEvent {
                    stage: Stage::AfterDemand,
                    event: String::from("customer_demand_lost"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("procurement_cost_accounted"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("fixed_order_cost_accounted"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("holding_cost_accounted"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("lost_sales_penalty_accounted"),
                },
            ],
            feasibility_constraints: vec![TimingConstraint {
                name: String::from("observe_then_receive_then_demand"),
                description: String::from(
                    "the current lost-sales rollout observes the pipeline state, then advances the pipeline, then realizes demand, and finally accounts for period cost",
                ),
            }],
        },
    }
}
