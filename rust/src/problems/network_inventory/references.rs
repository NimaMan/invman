use crate::problems::network_inventory::demand::{DemandDistributionKind, DemandModel};
use crate::problems::network_inventory::env::NetworkEdge;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetworkInventoryReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub num_nodes: usize,
    pub source_nodes: &'static [bool],
    pub edges: &'static [NetworkEdge],
    pub demand_models: &'static [DemandModel],
    pub holding_costs: &'static [f64],
    pub backlog_costs: &'static [f64],
    pub base_stock_levels: &'static [usize],
    pub initial_on_hand_inventory: &'static [usize],
    pub initial_backlog: &'static [usize],
    pub initial_edge_pipelines: &'static [&'static [usize]],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkedTransitionReference {
    pub source: &'static str,
    pub url: &'static str,
    pub num_nodes: usize,
    pub source_nodes: &'static [bool],
    pub edges: &'static [NetworkEdge],
    pub initial_on_hand_inventory: &'static [usize],
    pub initial_backlog: &'static [usize],
    pub initial_edge_pipelines: &'static [&'static [usize]],
    pub action: &'static [usize],
    pub realized_demands: &'static [usize],
    pub holding_costs: &'static [f64],
    pub backlog_costs: &'static [f64],
    pub expected_received_shipments_by_node: &'static [usize],
    pub expected_shipments_on_edges: &'static [usize],
    pub expected_next_on_hand_inventory: &'static [usize],
    pub expected_next_backlog: &'static [usize],
    pub expected_next_edge_pipelines: &'static [&'static [usize]],
    pub expected_period_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
    pub periods: usize,
    pub discount_factor: f64,
    pub num_nodes: usize,
    pub source_nodes: &'static [bool],
    pub edges: &'static [NetworkEdge],
    pub initial_on_hand_inventory: &'static [usize],
    pub initial_backlog: &'static [usize],
    pub initial_edge_pipelines: &'static [&'static [usize]],
    pub holding_costs: &'static [f64],
    pub backlog_costs: &'static [f64],
    pub demand_supports: &'static [&'static [u32]],
    pub demand_probabilities: &'static [&'static [f64]],
    pub max_edge_requests: &'static [usize],
    pub base_stock_levels: &'static [usize],
    pub expected_optimal_discounted_cost: f64,
    pub expected_optimal_first_action: &'static [usize],
    pub expected_base_stock_discounted_cost: f64,
    pub expected_base_stock_first_action: &'static [usize],
    pub notes: &'static str,
}

pub const PIRHOOSHYARAN_2021_REFERENCE: PublishedBenchmarkReference =
    PublishedBenchmarkReference {
        source: "Pirhooshyaran and Snyder (2021), arXiv:2006.05608",
        url: "https://arxiv.org/abs/2006.05608",
        benchmark_policies: &["pairwise_dnn", "node_base_stock"],
        notes: "The paper studies general stochastic inventory networks and motivates pairwise decision makers on adjacent nodes rather than a single handcrafted policy for each special topology. The repo starts with a graph-generalized node-base-stock baseline and a reduced exact verifier.",
    };

pub const DETERMINISTIC_ZERO: DemandModel = DemandModel {
    kind: DemandDistributionKind::Deterministic,
    param1: 0.0,
};
pub const POISSON_TWO: DemandModel = DemandModel {
    kind: DemandDistributionKind::Poisson,
    param1: 2.0,
};

pub const DIAMOND_EDGES: &[NetworkEdge] = &[
    NetworkEdge {
        from: 0,
        to: 1,
        lead_time: 1,
    },
    NetworkEdge {
        from: 0,
        to: 2,
        lead_time: 1,
    },
    NetworkEdge {
        from: 1,
        to: 3,
        lead_time: 1,
    },
    NetworkEdge {
        from: 2,
        to: 3,
        lead_time: 1,
    },
];

pub const PRIMARY_REFERENCE_INSTANCE: NetworkInventoryReferenceInstance =
    NetworkInventoryReferenceInstance {
        name: "pirhooshyaran2021_diamond_distribution_network",
        source: PIRHOOSHYARAN_2021_REFERENCE.source,
        url: PIRHOOSHYARAN_2021_REFERENCE.url,
        num_nodes: 4,
        source_nodes: &[true, false, false, false],
        edges: DIAMOND_EDGES,
        demand_models: &[DETERMINISTIC_ZERO, DETERMINISTIC_ZERO, DETERMINISTIC_ZERO, POISSON_TWO],
        holding_costs: &[0.0, 0.5, 0.5, 1.0],
        backlog_costs: &[0.0, 0.0, 0.0, 8.0],
        base_stock_levels: &[0, 3, 3, 4],
        initial_on_hand_inventory: &[0, 2, 2, 1],
        initial_backlog: &[0, 0, 0, 0],
        initial_edge_pipelines: &[&[1], &[1], &[0], &[1]],
        notes: "Repo canonical first benchmark for the general network family. This is a literature-shaped diamond distribution network with a single source, two intermediate stocking nodes, and one customer-facing sink supplied by two upstream nodes.",
    };

pub const WORKED_TRANSITION_REFERENCE: WorkedTransitionReference = WorkedTransitionReference {
    source: PIRHOOSHYARAN_2021_REFERENCE.source,
    url: PIRHOOSHYARAN_2021_REFERENCE.url,
    num_nodes: 4,
    source_nodes: &[true, false, false, false],
    edges: DIAMOND_EDGES,
    initial_on_hand_inventory: &[0, 2, 1, 0],
    initial_backlog: &[0, 0, 0, 0],
    initial_edge_pipelines: &[&[1], &[0], &[0], &[1]],
    action: &[2, 1, 1, 2],
    realized_demands: &[0, 0, 0, 2],
    holding_costs: &[0.0, 0.5, 0.5, 1.0],
    backlog_costs: &[0.0, 0.0, 0.0, 8.0],
    expected_received_shipments_by_node: &[0, 1, 0, 1],
    expected_shipments_on_edges: &[2, 1, 1, 1],
    expected_next_on_hand_inventory: &[0, 2, 0, 0],
    expected_next_backlog: &[0, 0, 0, 1],
    expected_next_edge_pipelines: &[&[2], &[1], &[1], &[1]],
    expected_period_cost: 9.0,
};

pub const VERIFICATION_ZERO_SUPPORT: &[u32] = &[0];
pub const VERIFICATION_ZERO_PROBABILITIES: &[f64] = &[1.0];
pub const VERIFICATION_SINK_SUPPORT: &[u32] = &[0, 1];
pub const VERIFICATION_SINK_PROBABILITIES: &[f64] = &[0.5, 0.5];

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: PIRHOOSHYARAN_2021_REFERENCE.source,
    url: PIRHOOSHYARAN_2021_REFERENCE.url,
    periods: 3,
    discount_factor: 0.99,
    num_nodes: 4,
    source_nodes: &[true, false, false, false],
    edges: DIAMOND_EDGES,
    initial_on_hand_inventory: &[0, 1, 1, 0],
    initial_backlog: &[0, 0, 0, 0],
    initial_edge_pipelines: &[&[1], &[0], &[0], &[1]],
    holding_costs: &[0.0, 0.5, 0.5, 1.0],
    backlog_costs: &[0.0, 0.0, 0.0, 6.0],
    demand_supports: &[
        VERIFICATION_ZERO_SUPPORT,
        VERIFICATION_ZERO_SUPPORT,
        VERIFICATION_ZERO_SUPPORT,
        VERIFICATION_SINK_SUPPORT,
    ],
    demand_probabilities: &[
        VERIFICATION_ZERO_PROBABILITIES,
        VERIFICATION_ZERO_PROBABILITIES,
        VERIFICATION_ZERO_PROBABILITIES,
        VERIFICATION_SINK_PROBABILITIES,
    ],
    max_edge_requests: &[2, 2, 2, 2],
    base_stock_levels: &[0, 2, 2, 3],
    expected_optimal_discounted_cost: 4.2126,
    expected_optimal_first_action: &[0, 0, 0, 1],
    expected_base_stock_discounted_cost: 7.152850000000001,
    expected_base_stock_first_action: &[0, 1, 1, 1],
    notes: "Repo-native exact verifier on a reduced general network with one source, two parallel intermediate nodes, and one customer-facing sink supplied by two upstream edges. This keeps the graph genuinely networked while staying small enough for routine finite-horizon DP checks.",
};
