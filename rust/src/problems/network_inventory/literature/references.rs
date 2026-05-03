#![allow(dead_code)]

use crate::problems::network_inventory::demand::{DemandDistributionKind, DemandModel};
use crate::problems::network_inventory::env::{NetworkEdge, NetworkNodeMode};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SingleNodeBenchmarkRow {
    pub case_idx: usize,
    pub demand_mean: f64,
    pub demand_stddev: f64,
    pub lead_time: usize,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub horizon_periods: usize,
    pub published_analytical_oul: f64,
    pub published_analytical_average_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SerialBenchmarkRow {
    pub case_idx: usize,
    pub num_echelons: usize,
    pub demand_mean: f64,
    pub demand_stddev: f64,
    pub holding_costs: &'static [f64],
    pub shortage_costs: &'static [f64],
    pub lead_times: &'static [usize],
    pub horizon_periods: usize,
    pub published_analytical_ouls: &'static [f64],
    pub published_average_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetworkInventoryReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub verification_source: &'static str,
    pub periods: usize,
    pub num_nodes: usize,
    pub source_nodes: &'static [bool],
    pub node_modes: &'static [NetworkNodeMode],
    pub external_supplier_lead_times: &'static [usize],
    pub edges: &'static [NetworkEdge],
    pub demand_models: &'static [DemandModel],
    pub holding_costs: &'static [f64],
    pub backlog_costs: &'static [f64],
    pub pairwise_oul_levels: &'static [f64],
    pub initial_finished_inventory: &'static [usize],
    pub initial_raw_inventory_by_relation: &'static [usize],
    pub initial_internal_backlog_by_edge: &'static [usize],
    pub initial_external_backlog: &'static [usize],
    pub initial_supply_pipelines: &'static [&'static [usize]],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub verification_source: &'static str,
    pub periods: usize,
    pub discount_factor: f64,
    pub num_nodes: usize,
    pub source_nodes: &'static [bool],
    pub node_modes: &'static [NetworkNodeMode],
    pub external_supplier_lead_times: &'static [usize],
    pub edges: &'static [NetworkEdge],
    pub initial_finished_inventory: &'static [usize],
    pub initial_raw_inventory_by_relation: &'static [usize],
    pub initial_internal_backlog_by_edge: &'static [usize],
    pub initial_external_backlog: &'static [usize],
    pub initial_supply_pipelines: &'static [&'static [usize]],
    pub holding_costs: &'static [f64],
    pub backlog_costs: &'static [f64],
    pub demand_supports: &'static [&'static [u32]],
    pub demand_probabilities: &'static [&'static [f64]],
    pub max_supply_requests: &'static [usize],
    pub base_stock_levels: &'static [usize],
    pub notes: &'static str,
}

pub const PIRHOOSHYARAN_2021_REFERENCE: PublishedBenchmarkReference =
    PublishedBenchmarkReference {
        source: "Pirhooshyaran and Snyder (2021), arXiv:2006.05608",
        url: "https://arxiv.org/abs/2006.05608",
        benchmark_policies: &["pairwise_dnn", "pairwise_base_stock"],
        notes: "The paper studies finite-horizon stochastic supply-chain networks with pairwise order-up-to decisions, raw-material inventories, finished-goods inventories, and both assembly and distribution structures. Table 1 gives single-node analytical newsvendor rows. Tables 2 and 3 give serial benchmark settings and analytical OUL plus average-cost rows.",
    };

pub const SINGLE_NODE_BENCHMARK_ROWS: &[SingleNodeBenchmarkRow] = &[
    SingleNodeBenchmarkRow {
        case_idx: 1,
        demand_mean: 10.0,
        demand_stddev: 1.0,
        lead_time: 1,
        holding_cost: 10.0,
        shortage_cost: 30.0,
        horizon_periods: 2,
        published_analytical_oul: 10.67,
        published_analytical_average_cost: 12.71,
    },
    SingleNodeBenchmarkRow {
        case_idx: 2,
        demand_mean: 10.0,
        demand_stddev: 2.0,
        lead_time: 1,
        holding_cost: 10.0,
        shortage_cost: 30.0,
        horizon_periods: 2,
        published_analytical_oul: 11.35,
        published_analytical_average_cost: 25.42,
    },
    SingleNodeBenchmarkRow {
        case_idx: 3,
        demand_mean: 50.0,
        demand_stddev: 1.0,
        lead_time: 1,
        holding_cost: 10.0,
        shortage_cost: 30.0,
        horizon_periods: 2,
        published_analytical_oul: 50.67,
        published_analytical_average_cost: 12.71,
    },
    SingleNodeBenchmarkRow {
        case_idx: 4,
        demand_mean: 50.0,
        demand_stddev: 5.0,
        lead_time: 1,
        holding_cost: 10.0,
        shortage_cost: 30.0,
        horizon_periods: 2,
        published_analytical_oul: 53.37,
        published_analytical_average_cost: 63.56,
    },
    SingleNodeBenchmarkRow {
        case_idx: 5,
        demand_mean: 100.0,
        demand_stddev: 1.0,
        lead_time: 1,
        holding_cost: 10.0,
        shortage_cost: 30.0,
        horizon_periods: 2,
        published_analytical_oul: 100.67,
        published_analytical_average_cost: 12.71,
    },
    SingleNodeBenchmarkRow {
        case_idx: 6,
        demand_mean: 100.0,
        demand_stddev: 5.0,
        lead_time: 1,
        holding_cost: 10.0,
        shortage_cost: 30.0,
        horizon_periods: 2,
        published_analytical_oul: 103.37,
        published_analytical_average_cost: 63.56,
    },
    SingleNodeBenchmarkRow {
        case_idx: 7,
        demand_mean: 100.0,
        demand_stddev: 10.0,
        lead_time: 1,
        holding_cost: 10.0,
        shortage_cost: 30.0,
        horizon_periods: 2,
        published_analytical_oul: 106.74,
        published_analytical_average_cost: 127.11,
    },
];

pub const SERIAL_BENCHMARK_ROWS: &[SerialBenchmarkRow] = &[
    SerialBenchmarkRow {
        case_idx: 1,
        num_echelons: 2,
        demand_mean: 3.0,
        demand_stddev: 0.5,
        holding_costs: &[5.0, 8.2],
        shortage_costs: &[0.0, 25.5],
        lead_times: &[1, 1],
        horizon_periods: 10,
        published_analytical_ouls: &[2.91, 3.64],
        published_average_cost: 22.21,
    },
    SerialBenchmarkRow {
        case_idx: 2,
        num_echelons: 2,
        demand_mean: 6.0,
        demand_stddev: 1.5,
        holding_costs: &[1.9, 4.1],
        shortage_costs: &[0.0, 11.3],
        lead_times: &[2, 1],
        horizon_periods: 10,
        published_analytical_ouls: &[12.58, 7.60],
        published_average_cost: 23.07,
    },
    SerialBenchmarkRow {
        case_idx: 3,
        num_echelons: 3,
        demand_mean: 5.0,
        demand_stddev: 1.0,
        holding_costs: &[2.0, 4.0, 7.0],
        shortage_costs: &[0.0, 0.0, 37.12],
        lead_times: &[2, 1, 1],
        horizon_periods: 10,
        published_analytical_ouls: &[10.69, 5.53, 6.49],
        published_average_cost: 47.65,
    },
    SerialBenchmarkRow {
        case_idx: 4,
        num_echelons: 3,
        demand_mean: 50.0,
        demand_stddev: 3.0,
        holding_costs: &[5.0, 10.0, 25.0],
        shortage_costs: &[0.0, 0.0, 50.0],
        lead_times: &[2, 1, 1],
        horizon_periods: 10,
        published_analytical_ouls: &[101.45, 51.40, 52.7040],
        published_average_cost: 879.88,
    },
    SerialBenchmarkRow {
        case_idx: 5,
        num_echelons: 3,
        demand_mean: 100.0,
        demand_stddev: 5.0,
        holding_costs: &[25.0, 25.0, 50.0],
        shortage_costs: &[0.0, 0.0, 100.0],
        lead_times: &[1, 2, 2],
        horizon_periods: 10,
        published_analytical_ouls: &[71.026, 228.29, 207.04],
        published_average_cost: 10568.23,
    },
    SerialBenchmarkRow {
        case_idx: 6,
        num_echelons: 3,
        demand_mean: 100.0,
        demand_stddev: 10.0,
        holding_costs: &[10.0, 20.0, 30.0],
        shortage_costs: &[0.0, 0.0, 100.0],
        lead_times: &[1, 1, 1],
        horizon_periods: 10,
        published_analytical_ouls: &[99.53, 102.58, 114.05],
        published_average_cost: 3630.14,
    },
    SerialBenchmarkRow {
        case_idx: 7,
        num_echelons: 4,
        demand_mean: 3.0,
        demand_stddev: 0.4,
        holding_costs: &[4.0, 5.75, 7.90, 10.8],
        shortage_costs: &[0.0, 0.0, 0.0, 35.5],
        lead_times: &[1, 1, 1, 1],
        horizon_periods: 10,
        published_analytical_ouls: &[2.78, 3.13, 3.19, 3.60],
        published_average_cost: 63.39,
    },
    SerialBenchmarkRow {
        case_idx: 8,
        num_echelons: 4,
        demand_mean: 5.0,
        demand_stddev: 1.2,
        holding_costs: &[5.0, 5.0, 5.0, 10.0],
        shortage_costs: &[0.0, 0.0, 0.0, 30.0],
        lead_times: &[1, 1, 1, 1],
        horizon_periods: 10,
        published_analytical_ouls: &[-3.80, 9.80, 9.80, 6.35],
        published_average_cost: 101.48,
    },
    SerialBenchmarkRow {
        case_idx: 9,
        num_echelons: 5,
        demand_mean: 80.0,
        demand_stddev: 4.0,
        holding_costs: &[10.0, 20.0, 30.0, 40.0, 50.0],
        shortage_costs: &[0.0, 0.0, 0.0, 0.0, 200.0],
        lead_times: &[1, 1, 1, 1, 1],
        horizon_periods: 10,
        published_analytical_ouls: &[80.15, 80.15, 81.17, 81.68, 86.99],
        published_average_cost: 8559.85,
    },
    SerialBenchmarkRow {
        case_idx: 10,
        num_echelons: 5,
        demand_mean: 25.0,
        demand_stddev: 2.0,
        holding_costs: &[5.0, 10.0, 25.0, 50.0, 50.0],
        shortage_costs: &[0.0, 0.0, 0.0, 0.0, 150.0],
        lead_times: &[2, 1, 1, 1, 1],
        horizon_periods: 10,
        published_analytical_ouls: &[51.57, 26.30, 25.05, 20.25, 33.01],
        published_average_cost: 2500.79,
    },
];

pub const DETERMINISTIC_ZERO: DemandModel = DemandModel {
    kind: DemandDistributionKind::Deterministic,
    param1: 0.0,
    param2: 0.0,
};

pub const DETERMINISTIC_ONE: DemandModel = DemandModel {
    kind: DemandDistributionKind::Deterministic,
    param1: 1.0,
    param2: 0.0,
};

pub const NORMAL_FIVE_ONE: DemandModel = DemandModel {
    kind: DemandDistributionKind::Normal,
    param1: 5.0,
    param2: 1.0,
};

pub const SERIAL_THREE_ECHELON_EDGES: &[NetworkEdge] = &[
    NetworkEdge {
        from: 0,
        to: 1,
        lead_time: 1,
    },
    NetworkEdge {
        from: 1,
        to: 2,
        lead_time: 1,
    },
];

pub const SERIAL_SINGLE_MODES: &[NetworkNodeMode] = &[
    NetworkNodeMode::Single,
    NetworkNodeMode::Single,
    NetworkNodeMode::Single,
];

pub const PRIMARY_REFERENCE_INSTANCE: NetworkInventoryReferenceInstance =
    NetworkInventoryReferenceInstance {
        name: "pirhooshyaran2021_serial_case3",
        source: PIRHOOSHYARAN_2021_REFERENCE.source,
        url: PIRHOOSHYARAN_2021_REFERENCE.url,
        literature_verified: false,
        verification_source: "single_node_rows_verified_serial_rows_cataloged_only",
        periods: 10,
        num_nodes: 3,
        source_nodes: &[true, false, false],
        node_modes: SERIAL_SINGLE_MODES,
        external_supplier_lead_times: &[2, 0, 0],
        edges: SERIAL_THREE_ECHELON_EDGES,
        demand_models: &[DETERMINISTIC_ZERO, DETERMINISTIC_ZERO, NORMAL_FIVE_ONE],
        holding_costs: &[2.0, 4.0, 7.0],
        backlog_costs: &[0.0, 0.0, 37.12],
        pairwise_oul_levels: &[10.69, 5.53, 6.49],
        initial_finished_inventory: &[10, 5, 5],
        initial_raw_inventory_by_relation: &[0, 0, 0],
        initial_internal_backlog_by_edge: &[0, 0],
        initial_external_backlog: &[0, 0, 0],
        initial_supply_pipelines: &[&[0], &[0], &[0, 0]],
        notes: "Paper-shaped three-echelon serial case from Tables 2 and 3. The analytical OULs are carried in the paper's published upstream-to-downstream edge order [supplier->0, 0->1, 1->2], not the internal Rust supply-relation order. The current executable package is still discrete and not literature-verified on this row.",
    };

pub const VERIFICATION_SERIAL_EDGES: &[NetworkEdge] = &[NetworkEdge {
    from: 0,
    to: 1,
    lead_time: 1,
}];

pub const VERIFICATION_ZERO_SUPPORT: &[u32] = &[0];
pub const VERIFICATION_ZERO_PROBABILITIES: &[f64] = &[1.0];
pub const VERIFICATION_ONE_SUPPORT: &[u32] = &[1];
pub const VERIFICATION_ONE_PROBABILITIES: &[f64] = &[1.0];

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: "Repo exact verification instance for the paper-shaped network-inventory family",
    url: PIRHOOSHYARAN_2021_REFERENCE.url,
    literature_verified: false,
    verification_source: "repo_exact_solver_not_verified_against_literature",
    periods: 3,
    discount_factor: 0.99,
    num_nodes: 2,
    source_nodes: &[true, false],
    node_modes: &[NetworkNodeMode::Single, NetworkNodeMode::Single],
    external_supplier_lead_times: &[1, 0],
    edges: VERIFICATION_SERIAL_EDGES,
    initial_finished_inventory: &[1, 0],
    initial_raw_inventory_by_relation: &[0, 0],
    initial_internal_backlog_by_edge: &[0],
    initial_external_backlog: &[0, 0],
    initial_supply_pipelines: &[&[0], &[1]],
    holding_costs: &[1.0, 1.0],
    backlog_costs: &[0.0, 5.0],
    demand_supports: &[VERIFICATION_ZERO_SUPPORT, VERIFICATION_ONE_SUPPORT],
    demand_probabilities: &[VERIFICATION_ZERO_PROBABILITIES, VERIFICATION_ONE_PROBABILITIES],
    max_supply_requests: &[2, 2],
    base_stock_levels: &[1, 1],
    notes: "Repo-native exact verifier on a paper-shaped serial network with explicit raw-material inventory, finished-goods inventory, an external supplier relation, and an external customer demand relation.",
};
