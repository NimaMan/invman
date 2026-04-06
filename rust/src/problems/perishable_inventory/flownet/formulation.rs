use crate::problems::core::control::{
    ActionShape, ActionSpec, ControlLayer, FeasibilityConstraint, IssuanceRule, ObservationMode,
    ObservationSpec, ServiceSpec, ShortageReaction,
};
use crate::problems::core::events::{
    AccountingEventKind, ControlEventKind, EventCatalog, EventKind, EventSpec, ExogenousEventKind,
    MaterialEventKind, ServiceEventKind, TransformationEventKind,
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

pub const PERISHABLE_INVENTORY_FLOWNET_NAME: &str = "perishable_inventory";

pub fn canonical_perishable_inventory_flownet() -> FlowNetFormulation {
    FlowNetFormulation {
        name: String::from(PERISHABLE_INVENTORY_FLOWNET_NAME),
        physical: PhysicalLayer {
            topology: Topology::SingleLocation,
            stock_nodes: vec![
                StockNodeSpec {
                    name: String::from("on_hand_age_buckets"),
                    role: StockRole::AgeBucket,
                    attributes: vec![
                        String::from("single_item"),
                        String::from("youngest_to_oldest"),
                        String::from("perishable_inventory"),
                    ],
                },
                StockNodeSpec {
                    name: String::from("customer_demand_sink"),
                    role: StockRole::DemandSink,
                    attributes: vec![String::from("lost_sales_service")],
                },
                StockNodeSpec {
                    name: String::from("expired_inventory_sink"),
                    role: StockRole::Custom(String::from("waste_sink")),
                    attributes: vec![String::from("expired_units")],
                },
            ],
            pipelines: vec![PipelineSpec {
                name: String::from("inbound_pipeline"),
                from: String::from("supplier"),
                to: String::from("on_hand_age_buckets"),
                stages: 1,
                attributes: vec![String::from(
                    "single inbound lead-time pipeline whose arrival enters the youngest age bucket",
                )],
            }],
            flow_edges: vec![
                FlowEdgeSpec {
                    name: String::from("supplier_to_pipeline"),
                    from: String::from("supplier"),
                    to: String::from("inbound_pipeline"),
                    mode: FlowMode::Procurement,
                },
                FlowEdgeSpec {
                    name: String::from("pipeline_to_youngest_age_bucket"),
                    from: String::from("inbound_pipeline"),
                    to: String::from("on_hand_age_buckets"),
                    mode: FlowMode::Shipment,
                },
                FlowEdgeSpec {
                    name: String::from("age_buckets_to_customer"),
                    from: String::from("on_hand_age_buckets"),
                    to: String::from("customer_demand_sink"),
                    mode: FlowMode::DemandFulfillment,
                },
                FlowEdgeSpec {
                    name: String::from("expiring_inventory_to_waste"),
                    from: String::from("on_hand_age_buckets"),
                    to: String::from("expired_inventory_sink"),
                    mode: FlowMode::Removal,
                },
                FlowEdgeSpec {
                    name: String::from("age_bucket_progression"),
                    from: String::from("on_hand_age_buckets"),
                    to: String::from("on_hand_age_buckets"),
                    mode: FlowMode::Aging,
                },
            ],
            material_attributes: vec![
                MaterialAttribute {
                    name: String::from("item_class"),
                    allowed_values: vec![String::from("single_sku")],
                },
                MaterialAttribute {
                    name: String::from("inventory_age_structure"),
                    allowed_values: vec![
                        String::from("youngest_to_oldest"),
                        String::from("fixed_shelf_life"),
                    ],
                },
                MaterialAttribute {
                    name: String::from("issuing_policy"),
                    allowed_values: vec![String::from("fifo"), String::from("lifo")],
                },
                MaterialAttribute {
                    name: String::from("shortage_semantics"),
                    allowed_values: vec![String::from("lost_sales")],
                },
            ],
        },
        stochastic: StochasticLayer {
            processes: vec![StochasticProcess::Demand(DemandProcessSpec {
                target: String::from("customer_demand_sink"),
                model: String::from("rounded Gamma demand with configurable mean and coefficient of variation"),
            })],
        },
        control: ControlLayer {
            actions: vec![ActionSpec {
                name: String::from("replenishment_order"),
                target: String::from("inbound_pipeline"),
                shape: ActionShape::ScalarOrder,
            }],
            observations: vec![ObservationSpec {
                name: String::from("pipeline_and_age_bucket_state"),
                mode: ObservationMode::FullState,
                channels: vec![
                    String::from("pipeline_orders"),
                    String::from("on_hand_by_age_bucket"),
                    String::from("pipeline_then_on_hand_observation_layout"),
                ],
            }],
            service_policies: vec![ServiceSpec {
                name: String::from("perishable_lost_sales_service"),
                demand_target: String::from("customer_demand_sink"),
                inventory_sources: vec![String::from("on_hand_age_buckets")],
                issuance_rule: IssuanceRule::Configurable(vec![
                    String::from("fifo"),
                    String::from("lifo"),
                ]),
                shortage_reaction: ShortageReaction::LostSales,
            }],
            feasibility_constraints: vec![
                FeasibilityConstraint {
                    name: String::from("nonnegative_order"),
                    description: String::from("the replenishment action must be a nonnegative integer"),
                },
                FeasibilityConstraint {
                    name: String::from("positive_shelf_life"),
                    description: String::from("the perishable-inventory implementation assumes shelf_life >= 1"),
                },
                FeasibilityConstraint {
                    name: String::from("positive_lead_time"),
                    description: String::from("the perishable-inventory implementation assumes lead_time >= 1"),
                },
            ],
        },
        objective: ObjectiveLayer {
            terms: vec![
                ObjectiveTerm::HoldingCost {
                    target: String::from("on_hand_age_buckets"),
                },
                ObjectiveTerm::LostSalesPenalty {
                    target: String::from("customer_demand_sink"),
                },
                ObjectiveTerm::WasteCost {
                    target: String::from("expired_inventory_sink"),
                },
                ObjectiveTerm::ProcurementCost {
                    target: String::from("replenishment_order"),
                },
            ],
            discounting: Discounting::None,
            reward_convention: RewardConvention::MinimizeCost,
            tracked_metrics: vec![
                PerformanceMetric::TotalCost,
                PerformanceMetric::FillRate,
                PerformanceMetric::AverageInventory,
                PerformanceMetric::AverageWaste,
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
                        notes: Some(String::from("the controller observes the age-structured state and chooses an order quantity")),
                    },
                    EventSpec {
                        name: String::from("order_dispatch"),
                        kind: EventKind::Material(MaterialEventKind::Dispatch),
                        source: Some(String::from("supplier")),
                        target: Some(String::from("inbound_pipeline")),
                        notes: Some(String::from("the new order is inserted at the head of the inbound pipeline")),
                    },
                    EventSpec {
                        name: String::from("customer_demand_arrival"),
                        kind: EventKind::Exogenous(ExogenousEventKind::DemandArrival),
                        source: None,
                        target: Some(String::from("customer_demand_sink")),
                        notes: Some(String::from("demand is sampled from the configured rounded-Gamma demand process")),
                    },
                    EventSpec {
                        name: String::from("customer_demand_served"),
                        kind: EventKind::Service(ServiceEventKind::DemandServed),
                        source: Some(String::from("on_hand_age_buckets")),
                        target: Some(String::from("customer_demand_sink")),
                        notes: Some(String::from("demand is issued from age buckets using the configured FIFO or LIFO rule")),
                    },
                    EventSpec {
                        name: String::from("customer_demand_lost"),
                        kind: EventKind::Service(ServiceEventKind::DemandLost),
                        source: Some(String::from("customer_demand_sink")),
                        target: None,
                        notes: Some(String::from("unmet demand leaves the system immediately as lost sales")),
                    },
                    EventSpec {
                        name: String::from("inventory_expiration"),
                        kind: EventKind::Transformation(TransformationEventKind::Decay),
                        source: Some(String::from("on_hand_age_buckets")),
                        target: Some(String::from("expired_inventory_sink")),
                        notes: Some(String::from("the oldest leftover inventory expires after demand is served")),
                    },
                    EventSpec {
                        name: String::from("inventory_aging"),
                        kind: EventKind::Transformation(TransformationEventKind::Aging),
                        source: Some(String::from("on_hand_age_buckets")),
                        target: Some(String::from("on_hand_age_buckets")),
                        notes: Some(String::from("remaining inventory shifts one bucket older at the end of the period")),
                    },
                    EventSpec {
                        name: String::from("inbound_receipt"),
                        kind: EventKind::Material(MaterialEventKind::Receipt),
                        source: Some(String::from("inbound_pipeline")),
                        target: Some(String::from("on_hand_age_buckets")),
                        notes: Some(String::from("the oldest pipeline order arrives into the youngest age bucket")),
                    },
                    EventSpec {
                        name: String::from("procurement_cost_accounted"),
                        kind: EventKind::Accounting(AccountingEventKind::ProcurementCostCharged),
                        source: Some(String::from("replenishment_order")),
                        target: None,
                        notes: Some(String::from("procurement cost is charged on the order quantity")),
                    },
                    EventSpec {
                        name: String::from("holding_cost_accounted"),
                        kind: EventKind::Accounting(AccountingEventKind::HoldingCostCharged),
                        source: Some(String::from("on_hand_age_buckets")),
                        target: None,
                        notes: Some(String::from("holding cost is charged on non-expired inventory carried forward")),
                    },
                    EventSpec {
                        name: String::from("waste_cost_accounted"),
                        kind: EventKind::Accounting(AccountingEventKind::WasteCostCharged),
                        source: Some(String::from("expired_inventory_sink")),
                        target: None,
                        notes: Some(String::from("waste cost is charged on expired units")),
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
                    event: String::from("inventory_expiration"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("inventory_aging"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("inbound_receipt"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("procurement_cost_accounted"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("holding_cost_accounted"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("waste_cost_accounted"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("lost_sales_penalty_accounted"),
                },
            ],
            feasibility_constraints: vec![TimingConstraint {
                name: String::from("demand_before_expiration_and_receipt"),
                description: String::from(
                    "the current perishable step semantics serve demand first, then expire the oldest leftover inventory, then age the remaining units and inject the inbound receipt into the youngest bucket",
                ),
            }],
        },
    }
}
