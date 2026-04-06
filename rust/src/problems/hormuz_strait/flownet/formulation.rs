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
use crate::problems::core::stochastic::{
    CustomProcessSpec, DemandProcessSpec, DisruptionProcessSpec, StochasticLayer,
    StochasticProcess, TransitDelayProcessSpec,
};
use crate::problems::core::timing::{ScheduledEvent, Stage, TimingConstraint, TimingLayer};

pub const HORMUZ_STRAIT_FLOWNET_NAME: &str = "hormuz_strait";

pub fn canonical_hormuz_strait_flownet() -> FlowNetFormulation {
    FlowNetFormulation {
        name: String::from(HORMUZ_STRAIT_FLOWNET_NAME),
        physical: PhysicalLayer {
            topology: Topology::DirectedNetwork,
            stock_nodes: vec![
                StockNodeSpec {
                    name: String::from("exporter_supply"),
                    role: StockRole::SupplySource,
                    attributes: vec![
                        String::from("7 exporter nodes"),
                        String::from("baseline flow weights derived from EIA 2024 Hormuz origin data"),
                    ],
                },
                StockNodeSpec {
                    name: String::from("destination_market_inventory"),
                    role: StockRole::OnHand,
                    attributes: vec![
                        String::from("9 destination market nodes"),
                        String::from("working stocks at destination markets"),
                    ],
                },
                StockNodeSpec {
                    name: String::from("gulf_refining_and_storage_hub"),
                    role: StockRole::OnHand,
                    attributes: vec![
                        String::from("local Gulf refining and storage absorption"),
                        String::from("captures diversion to regional demand inside the Gulf"),
                    ],
                },
                StockNodeSpec {
                    name: String::from("strategic_reserve_and_floating_storage"),
                    role: StockRole::Reserve,
                    attributes: vec![
                        String::from("emergency inventory release buffer"),
                        String::from("synthetic buffer node introduced for policy modeling"),
                    ],
                },
                StockNodeSpec {
                    name: String::from("market_demand"),
                    role: StockRole::DemandSink,
                    attributes: vec![
                        String::from("destination demand weights follow EIA 2024 Hormuz destinations"),
                        String::from("service failure is interpreted as rationing or unmet market demand"),
                    ],
                },
                StockNodeSpec {
                    name: String::from("unserved_market_demand"),
                    role: StockRole::Backlog,
                    attributes: vec![
                        String::from("tracks severe shortage or deferred demand"),
                        String::from("proxy for physical shortage and demand destruction pressure"),
                    ],
                },
            ],
            pipelines: vec![
                PipelineSpec {
                    name: String::from("hormuz_transit_lane"),
                    from: String::from("exporter_supply"),
                    to: String::from("destination_market_inventory"),
                    stages: 1,
                    attributes: vec![
                        String::from("main chokepoint flow path"),
                        String::from("subject to disruption-driven capacity loss"),
                    ],
                },
                PipelineSpec {
                    name: String::from("aggregate_bypass_capacity"),
                    from: String::from("exporter_supply"),
                    to: String::from("destination_market_inventory"),
                    stages: 1,
                    attributes: vec![
                        String::from("aggregate Saudi and UAE bypass asset"),
                        String::from("effective unused capacity initialized from EIA 2025 estimate"),
                    ],
                },
                PipelineSpec {
                    name: String::from("open_water_delivery_lane"),
                    from: String::from("hormuz_transit_lane"),
                    to: String::from("destination_market_inventory"),
                    stages: 1,
                    attributes: vec![
                        String::from("post-Hormuz maritime delivery path"),
                        String::from("can accumulate congestion-driven delay"),
                    ],
                },
            ],
            flow_edges: vec![
                FlowEdgeSpec {
                    name: String::from("exporters_to_hormuz"),
                    from: String::from("exporter_supply"),
                    to: String::from("hormuz_transit_lane"),
                    mode: FlowMode::Shipment,
                },
                FlowEdgeSpec {
                    name: String::from("exporters_to_bypass"),
                    from: String::from("exporter_supply"),
                    to: String::from("aggregate_bypass_capacity"),
                    mode: FlowMode::Custom(String::from("rerouted_shipment")),
                },
                FlowEdgeSpec {
                    name: String::from("hormuz_to_delivery_lane"),
                    from: String::from("hormuz_transit_lane"),
                    to: String::from("open_water_delivery_lane"),
                    mode: FlowMode::Custom(String::from("maritime_transfer")),
                },
                FlowEdgeSpec {
                    name: String::from("bypass_to_delivery_lane"),
                    from: String::from("aggregate_bypass_capacity"),
                    to: String::from("open_water_delivery_lane"),
                    mode: FlowMode::Custom(String::from("bypass_transfer")),
                },
                FlowEdgeSpec {
                    name: String::from("delivery_to_markets"),
                    from: String::from("open_water_delivery_lane"),
                    to: String::from("destination_market_inventory"),
                    mode: FlowMode::Custom(String::from("market_delivery")),
                },
                FlowEdgeSpec {
                    name: String::from("reserve_release_to_markets"),
                    from: String::from("strategic_reserve_and_floating_storage"),
                    to: String::from("destination_market_inventory"),
                    mode: FlowMode::Procurement,
                },
                FlowEdgeSpec {
                    name: String::from("market_service"),
                    from: String::from("destination_market_inventory"),
                    to: String::from("market_demand"),
                    mode: FlowMode::DemandFulfillment,
                },
                FlowEdgeSpec {
                    name: String::from("local_gulf_absorption"),
                    from: String::from("exporter_supply"),
                    to: String::from("gulf_refining_and_storage_hub"),
                    mode: FlowMode::Transformation,
                },
                FlowEdgeSpec {
                    name: String::from("unserved_demand_carryover"),
                    from: String::from("market_demand"),
                    to: String::from("unserved_market_demand"),
                    mode: FlowMode::Custom(String::from("rationing_or_shortage_state")),
                },
            ],
            material_attributes: vec![
                MaterialAttribute {
                    name: String::from("flow_node_layout"),
                    allowed_values: vec![
                        String::from("7 origin exporters"),
                        String::from("1 chokepoint"),
                        String::from("1 aggregate bypass asset"),
                        String::from("9 destination markets"),
                        String::from("1 Gulf refining and storage hub"),
                        String::from("1 strategic reserve buffer"),
                    ],
                },
                MaterialAttribute {
                    name: String::from("units"),
                    allowed_values: vec![String::from("million barrels per day")],
                },
                MaterialAttribute {
                    name: String::from("baseline_year"),
                    allowed_values: vec![String::from("2024")],
                },
                MaterialAttribute {
                    name: String::from("closure_mode"),
                    allowed_values: vec![
                        String::from("full_closure"),
                        String::from("partial_capacity_loss"),
                    ],
                },
            ],
        },
        stochastic: StochasticLayer {
            processes: vec![
                StochasticProcess::Demand(DemandProcessSpec {
                    target: String::from("market_demand"),
                    model: String::from(
                        "destination demand weights anchored to 2024 Hormuz destination shares and then stressed by scenario demand shocks",
                    ),
                }),
                StochasticProcess::Disruption(DisruptionProcessSpec {
                    target: String::from("hormuz_transit_lane"),
                    model: String::from(
                        "closure state with scenario-defined onset, duration, and reopening conditions",
                    ),
                }),
                StochasticProcess::TransitDelay(TransitDelayProcessSpec {
                    target: String::from("open_water_delivery_lane"),
                    model: String::from(
                        "rerouting congestion and extended voyage time after disruption",
                    ),
                }),
                StochasticProcess::Custom(CustomProcessSpec {
                    target: String::from("destination_market_inventory"),
                    model: String::from(
                        "inventory draw and replenishment lag outside the Gulf can be layered here later",
                    ),
                }),
            ],
        },
        control: ControlLayer {
            actions: vec![
                ActionSpec {
                    name: String::from("reroute_export_flows"),
                    target: String::from("aggregate_bypass_capacity"),
                    shape: ActionShape::Routing,
                },
                ActionSpec {
                    name: String::from("reserve_release_allocation"),
                    target: String::from("strategic_reserve_and_floating_storage"),
                    shape: ActionShape::Allocation,
                },
                ActionSpec {
                    name: String::from("destination_supply_allocation"),
                    target: String::from("destination_market_inventory"),
                    shape: ActionShape::Allocation,
                },
            ],
            observations: vec![ObservationSpec {
                name: String::from("hormuz_oil_network_state"),
                mode: ObservationMode::FullState,
                channels: vec![
                    String::from("closure_status"),
                    String::from("exporter_supply_by_origin"),
                    String::from("destination_inventory_by_market"),
                    String::from("reserve_volume"),
                    String::from("bypass_available_capacity"),
                    String::from("market_demand_weights"),
                    String::from("remaining_horizon_fraction"),
                ],
            }],
            service_policies: vec![ServiceSpec {
                name: String::from("market_supply_and_rationing"),
                demand_target: String::from("market_demand"),
                inventory_sources: vec![
                    String::from("destination_market_inventory"),
                    String::from("strategic_reserve_and_floating_storage"),
                    String::from("gulf_refining_and_storage_hub"),
                ],
                issuance_rule: IssuanceRule::FixedPriority(vec![
                    String::from("destination_market_inventory"),
                    String::from("strategic_reserve_and_floating_storage"),
                    String::from("gulf_refining_and_storage_hub"),
                ]),
                shortage_reaction: ShortageReaction::Custom(String::from(
                    "rationing_or_demand_destruction",
                )),
            }],
            feasibility_constraints: vec![
                FeasibilityConstraint {
                    name: String::from("nonnegative_flow_allocations"),
                    description: String::from(
                        "all rerouting, reserve-release, and market-allocation decisions must be nonnegative",
                    ),
                },
                FeasibilityConstraint {
                    name: String::from("bypass_capacity_limit"),
                    description: String::from(
                        "aggregate rerouted flow cannot exceed the scenario bypass capacity derived from the EIA estimate",
                    ),
                },
                FeasibilityConstraint {
                    name: String::from("origin_flow_limit"),
                    description: String::from(
                        "allocated origin flow cannot exceed the baseline exporter supply under the current disruption state",
                    ),
                },
                FeasibilityConstraint {
                    name: String::from("reserve_stock_limit"),
                    description: String::from(
                        "reserve releases cannot exceed available reserve and floating storage inventory",
                    ),
                },
            ],
        },
        objective: ObjectiveLayer {
            terms: vec![
                ObjectiveTerm::HoldingCost {
                    target: String::from("destination_market_inventory"),
                },
                ObjectiveTerm::HoldingCost {
                    target: String::from("strategic_reserve_and_floating_storage"),
                },
                ObjectiveTerm::BacklogCost {
                    target: String::from("unserved_market_demand"),
                },
                ObjectiveTerm::EmergencyFulfillmentCost {
                    target: String::from("strategic_reserve_and_floating_storage"),
                },
                ObjectiveTerm::Custom {
                    name: String::from("rerouting_cost"),
                    target: String::from("aggregate_bypass_capacity"),
                },
                ObjectiveTerm::Custom {
                    name: String::from("price_shock_penalty"),
                    target: String::from("market_demand"),
                },
            ],
            discounting: Discounting::None,
            reward_convention: RewardConvention::MinimizeCost,
            tracked_metrics: vec![
                PerformanceMetric::TotalCost,
                PerformanceMetric::FillRate,
                PerformanceMetric::AverageInventory,
                PerformanceMetric::AverageBacklog,
                PerformanceMetric::Custom(String::from("reserve_draw_million_bpd")),
                PerformanceMetric::Custom(String::from("rerouted_million_bpd")),
            ],
        },
        timing: TimingLayer {
            events: EventCatalog {
                events: vec![
                    EventSpec {
                        name: String::from("closure_status_update"),
                        kind: EventKind::Exogenous(ExogenousEventKind::DisruptionStart),
                        source: None,
                        target: Some(String::from("hormuz_transit_lane")),
                        notes: Some(String::from(
                            "the disruption state is updated at the start of the period",
                        )),
                    },
                    EventSpec {
                        name: String::from("controller_observes_network"),
                        kind: EventKind::Exogenous(ExogenousEventKind::Custom(String::from(
                            "observation_refresh",
                        ))),
                        source: None,
                        target: None,
                        notes: Some(String::from(
                            "the controller observes supply, reserve, and closure status",
                        )),
                    },
                    EventSpec {
                        name: String::from("rerouting_and_reserve_decision"),
                        kind: EventKind::Control(ControlEventKind::ReserveReleaseDecision),
                        source: None,
                        target: Some(String::from("aggregate_bypass_capacity")),
                        notes: Some(String::from(
                            "rerouting, reserve release, and allocation decisions are chosen",
                        )),
                    },
                    EventSpec {
                        name: String::from("export_dispatch"),
                        kind: EventKind::Material(MaterialEventKind::Dispatch),
                        source: Some(String::from("exporter_supply")),
                        target: Some(String::from("hormuz_transit_lane")),
                        notes: Some(String::from(
                            "available exporter flow is dispatched into Hormuz and bypass assets",
                        )),
                    },
                    EventSpec {
                        name: String::from("destination_delivery"),
                        kind: EventKind::Material(MaterialEventKind::Delivery),
                        source: Some(String::from("open_water_delivery_lane")),
                        target: Some(String::from("destination_market_inventory")),
                        notes: Some(String::from(
                            "surviving routed volumes arrive at destination inventories",
                        )),
                    },
                    EventSpec {
                        name: String::from("market_demand_realization"),
                        kind: EventKind::Exogenous(ExogenousEventKind::DemandArrival),
                        source: None,
                        target: Some(String::from("market_demand")),
                        notes: Some(String::from(
                            "market demand is realized across the destination blocs",
                        )),
                    },
                    EventSpec {
                        name: String::from("market_service_and_rationing"),
                        kind: EventKind::Service(ServiceEventKind::EmergencyFulfillment),
                        source: Some(String::from("destination_market_inventory")),
                        target: Some(String::from("market_demand")),
                        notes: Some(String::from(
                            "demand is served from market inventories and reserves; residual shortage becomes rationing state",
                        )),
                    },
                    EventSpec {
                        name: String::from("period_accounting"),
                        kind: EventKind::Accounting(AccountingEventKind::Custom(String::from(
                            "oil_market_disruption_costs",
                        ))),
                        source: None,
                        target: None,
                        notes: Some(String::from(
                            "inventory, shortage, rerouting, and price-shock costs are charged",
                        )),
                    },
                ],
            },
            stages: vec![
                Stage::StartOfPeriod,
                Stage::AfterAction,
                Stage::AfterReceipts,
                Stage::AfterDemand,
                Stage::EndOfPeriod,
            ],
            schedule: vec![
                ScheduledEvent {
                    stage: Stage::StartOfPeriod,
                    event: String::from("closure_status_update"),
                },
                ScheduledEvent {
                    stage: Stage::StartOfPeriod,
                    event: String::from("controller_observes_network"),
                },
                ScheduledEvent {
                    stage: Stage::AfterAction,
                    event: String::from("rerouting_and_reserve_decision"),
                },
                ScheduledEvent {
                    stage: Stage::AfterAction,
                    event: String::from("export_dispatch"),
                },
                ScheduledEvent {
                    stage: Stage::AfterReceipts,
                    event: String::from("destination_delivery"),
                },
                ScheduledEvent {
                    stage: Stage::AfterDemand,
                    event: String::from("market_demand_realization"),
                },
                ScheduledEvent {
                    stage: Stage::AfterDemand,
                    event: String::from("market_service_and_rationing"),
                },
                ScheduledEvent {
                    stage: Stage::EndOfPeriod,
                    event: String::from("period_accounting"),
                },
            ],
            feasibility_constraints: vec![
                TimingConstraint {
                    name: String::from("disruption_before_dispatch"),
                    description: String::from(
                        "closure status must be known before export routing is chosen",
                    ),
                },
                TimingConstraint {
                    name: String::from("delivery_before_service"),
                    description: String::from(
                        "delivered volumes and reserve releases must be available before demand is served",
                    ),
                },
                TimingConstraint {
                    name: String::from("accounting_after_rationing"),
                    description: String::from(
                        "costs are charged only after unmet demand and reserve releases are known",
                    ),
                },
            ],
        },
    }
}
