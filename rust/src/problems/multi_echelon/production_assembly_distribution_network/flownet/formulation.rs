#![allow(dead_code)]

use crate::problems::core::control::{
    ActionShape, ActionSpec, ControlLayer, FeasibilityConstraint, IssuanceRule, ObservationMode,
    ObservationSpec, ServiceSpec, ShortageReaction,
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

pub const PRODUCTION_ASSEMBLY_DISTRIBUTION_NETWORK_FLOWNET_NAME: &str = "production_assembly_distribution_network";

pub fn canonical_production_assembly_distribution_network_flownet() -> FlowNetFormulation {
    FlowNetFormulation {
        name: String::from(PRODUCTION_ASSEMBLY_DISTRIBUTION_NETWORK_FLOWNET_NAME),
        physical: PhysicalLayer {
            topology: Topology::DirectedNetwork,
            stock_nodes: vec![
                StockNodeSpec {
                    name: String::from("source_supply_nodes"),
                    role: StockRole::SupplySource,
                    attributes: vec![
                        String::from("subset_of_graph_nodes_flagged_as_sources"),
                        String::from("outbound_dispatch_does_not_consume_local_on_hand"),
                    ],
                },
                StockNodeSpec {
                    name: String::from("graph_node_inventory"),
                    role: StockRole::OnHand,
                    attributes: vec![
                        String::from("inventory_indexed_by_node"),
                        String::from("directed_network_stocking_points"),
                    ],
                },
                StockNodeSpec {
                    name: String::from("graph_node_backlog"),
                    role: StockRole::Backlog,
                    attributes: vec![
                        String::from("backlog_indexed_by_node"),
                        String::from("unmet_demand_carries_forward"),
                    ],
                },
                StockNodeSpec {
                    name: String::from("node_demand_sink"),
                    role: StockRole::DemandSink,
                    attributes: vec![
                        String::from("graph_indexed_demand"),
                        String::from("backorder_service"),
                    ],
                },
            ],
            pipelines: vec![PipelineSpec {
                name: String::from("directed_edge_pipelines"),
                from: String::from("graph_node_inventory"),
                to: String::from("graph_node_inventory"),
                stages: 1,
                attributes: vec![
                    String::from("one transit pipeline per directed edge"),
                    String::from("edge lead times are configurable and must be at least one"),
                    String::from(
                        "shipments dispatched by source nodes enter the same edge-indexed transit buffers",
                    ),
                ],
            }],
            flow_edges: vec![
                FlowEdgeSpec {
                    name: String::from("source_supply_dispatch_to_edge_pipelines"),
                    from: String::from("source_supply_nodes"),
                    to: String::from("directed_edge_pipelines"),
                    mode: FlowMode::Procurement,
                },
                FlowEdgeSpec {
                    name: String::from("internal_node_dispatch_to_edge_pipelines"),
                    from: String::from("graph_node_inventory"),
                    to: String::from("directed_edge_pipelines"),
                    mode: FlowMode::Shipment,
                },
                FlowEdgeSpec {
                    name: String::from("edge_pipeline_receipts_to_inventory"),
                    from: String::from("directed_edge_pipelines"),
                    to: String::from("graph_node_inventory"),
                    mode: FlowMode::Shipment,
                },
                FlowEdgeSpec {
                    name: String::from("inventory_service_to_demand_sink"),
                    from: String::from("graph_node_inventory"),
                    to: String::from("node_demand_sink"),
                    mode: FlowMode::DemandFulfillment,
                },
                FlowEdgeSpec {
                    name: String::from("unmet_demand_to_backlog"),
                    from: String::from("node_demand_sink"),
                    to: String::from("graph_node_backlog"),
                    mode: FlowMode::Custom(String::from("backorder_carryover")),
                },
            ],
            material_attributes: vec![
                MaterialAttribute {
                    name: String::from("graph_structure"),
                    allowed_values: vec![
                        String::from("directed_edges"),
                        String::from("edge_specific_lead_times"),
                        String::from("source_node_flags"),
                    ],
                },
                MaterialAttribute {
                    name: String::from("shipment_allocation_rule"),
                    allowed_values: vec![
                        String::from("source_nodes_unconstrained"),
                        String::from("internal_nodes_proportional_allocation"),
                    ],
                },
                MaterialAttribute {
                    name: String::from("service_semantics"),
                    allowed_values: vec![String::from("backorder")],
                },
                MaterialAttribute {
                    name: String::from("demand_process_family"),
                    allowed_values: vec![
                        String::from("per_node_deterministic"),
                        String::from("per_node_poisson"),
                    ],
                },
            ],
        },
        stochastic: StochasticLayer {
            processes: vec![StochasticProcess::Demand(DemandProcessSpec {
                target: String::from("node_demand_sink"),
                model: String::from(
                    "graph-indexed demand vector with deterministic or Poisson marginals at each node",
                ),
            })],
        },
        control: ControlLayer {
            actions: vec![ActionSpec {
                name: String::from("edge_shipment_requests"),
                target: String::from("directed_edge_pipelines"),
                shape: ActionShape::VectorOrder,
            }],
            observations: vec![ObservationSpec {
                name: String::from("network_state"),
                mode: ObservationMode::FullState,
                channels: vec![
                    String::from("on_hand_inventory_by_node"),
                    String::from("backlog_by_node"),
                    String::from("inbound_pipeline_totals_by_node"),
                    String::from("in_transit_totals_by_edge"),
                    String::from("demand_means_by_node"),
                    String::from("remaining_horizon_fraction"),
                ],
            }],
            service_policies: vec![ServiceSpec {
                name: String::from("backorder_service_rule"),
                demand_target: String::from("node_demand_sink"),
                inventory_sources: vec![String::from("graph_node_inventory")],
                issuance_rule: IssuanceRule::FixedPriority(vec![String::from(
                    "graph_node_inventory",
                )]),
                shortage_reaction: ShortageReaction::Backorder,
            }],
            feasibility_constraints: vec![
                FeasibilityConstraint {
                    name: String::from("nonnegative_edge_requests"),
                    description: String::from(
                        "each directed-edge shipment request must be a nonnegative integer",
                    ),
                },
                FeasibilityConstraint {
                    name: String::from("graph_dimension_consistency"),
                    description: String::from(
                        "node-wise demand and cost vectors must match num_nodes and pipeline vectors must match the edge lead times",
                    ),
                },
                FeasibilityConstraint {
                    name: String::from("source_node_dispatch_override"),
                    description: String::from(
                        "source nodes dispatch requested quantities without depleting on-hand inventory",
                    ),
                },
                FeasibilityConstraint {
                    name: String::from("proportional_internal_allocation"),
                    description: String::from(
                        "non-source nodes allocate limited on-hand inventory proportionally across outgoing edge requests when requests exceed availability",
                    ),
                },
            ],
        },
        objective: ObjectiveLayer {
            terms: vec![
                ObjectiveTerm::HoldingCost {
                    target: String::from("graph_node_inventory"),
                },
                ObjectiveTerm::BacklogCost {
                    target: String::from("graph_node_backlog"),
                },
            ],
            discounting: Discounting::None,
            reward_convention: RewardConvention::MinimizeCost,
            tracked_metrics: vec![
                PerformanceMetric::TotalCost,
                PerformanceMetric::AverageInventory,
                PerformanceMetric::AverageBacklog,
                PerformanceMetric::FillRate,
            ],
        },
        timing: TimingLayer {
            events: EventCatalog {
                events: vec![
                    EventSpec {
                        name: String::from("edge_shipment_decision"),
                        kind: EventKind::Control(ControlEventKind::ShipmentDecision),
                        source: None,
                        target: Some(String::from("directed_edge_pipelines")),
                        notes: Some(String::from(
                            "the controller observes the pre-receipt network state and requests shipments on each directed edge",
                        )),
                    },
                    EventSpec {
                        name: String::from("inbound_edge_receipts"),
                        kind: EventKind::Material(MaterialEventKind::Receipt),
                        source: Some(String::from("directed_edge_pipelines")),
                        target: Some(String::from("graph_node_inventory")),
                        notes: Some(String::from(
                            "the oldest in-transit shipment on each directed edge is received into the destination node",
                        )),
                    },
                    EventSpec {
                        name: String::from("edge_dispatch"),
                        kind: EventKind::Material(MaterialEventKind::Dispatch),
                        source: Some(String::from("graph_node_inventory")),
                        target: Some(String::from("directed_edge_pipelines")),
                        notes: Some(String::from(
                            "source nodes dispatch their full requests, while non-source nodes dispatch from on-hand stock using proportional allocation when needed",
                        )),
                    },
                    EventSpec {
                        name: String::from("node_demand_arrival"),
                        kind: EventKind::Exogenous(ExogenousEventKind::DemandArrival),
                        source: None,
                        target: Some(String::from("node_demand_sink")),
                        notes: Some(String::from(
                            "a graph-indexed demand vector is realized at the nodes",
                        )),
                    },
                    EventSpec {
                        name: String::from("node_demand_served"),
                        kind: EventKind::Service(ServiceEventKind::DemandServed),
                        source: Some(String::from("graph_node_inventory")),
                        target: Some(String::from("node_demand_sink")),
                        notes: Some(String::from(
                            "available inventory serves existing backlog and newly arrived demand at each node",
                        )),
                    },
                    EventSpec {
                        name: String::from("node_demand_backordered"),
                        kind: EventKind::Service(ServiceEventKind::DemandBackordered),
                        source: Some(String::from("node_demand_sink")),
                        target: Some(String::from("graph_node_backlog")),
                        notes: Some(String::from(
                            "unmet demand is retained as node-indexed backlog",
                        )),
                    },
                    EventSpec {
                        name: String::from("holding_cost_accounted"),
                        kind: EventKind::Accounting(AccountingEventKind::HoldingCostCharged),
                        source: Some(String::from("graph_node_inventory")),
                        target: None,
                        notes: Some(String::from(
                            "holding cost is charged on ending on-hand inventory by node",
                        )),
                    },
                    EventSpec {
                        name: String::from("backlog_cost_accounted"),
                        kind: EventKind::Accounting(AccountingEventKind::BacklogCostCharged),
                        source: Some(String::from("graph_node_backlog")),
                        target: None,
                        notes: Some(String::from(
                            "backlog cost is charged on ending backlog by node",
                        )),
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
                    event: String::from("edge_shipment_decision"),
                },
                ScheduledEvent {
                    stage: Stage::AfterReceipts,
                    event: String::from("inbound_edge_receipts"),
                },
                ScheduledEvent {
                    stage: Stage::AfterAction,
                    event: String::from("edge_dispatch"),
                },
                ScheduledEvent {
                    stage: Stage::AfterDemand,
                    event: String::from("node_demand_arrival"),
                },
                ScheduledEvent {
                    stage: Stage::AfterDemand,
                    event: String::from("node_demand_served"),
                },
                ScheduledEvent {
                    stage: Stage::AfterDemand,
                    event: String::from("node_demand_backordered"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("holding_cost_accounted"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("backlog_cost_accounted"),
                },
            ],
            feasibility_constraints: vec![TimingConstraint {
                name: String::from("network_step_order"),
                description: String::from(
                    "the current network step semantics choose edge requests from the start-of-period state, receive in-transit shipments, dispatch new shipments, then serve backlog plus new demand and finally charge holding and backlog costs",
                ),
            }],
        },
    }
}
