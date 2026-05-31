use crate::problems::multi_echelon::general_backorder_fixed_cost::env::RetailConnectionEdge;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneralBackorderFixedCostReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub num_suppliers: usize,
    pub num_warehouses: usize,
    pub num_retailers: usize,
    pub supplier_lead_times: &'static [usize],
    pub retail_edges: &'static [RetailConnectionEdge],
    pub retailer_demand_mean: f64,
    pub warehouse_holding_costs: &'static [f64],
    pub retailer_holding_costs: &'static [f64],
    pub warehouse_backorder_costs: &'static [f64],
    pub retailer_backorder_costs: &'static [f64],
    pub benchmark_base_stock_levels: &'static [usize],
    pub benchmark_periods: usize,
    pub benchmark_warm_up_periods: usize,
    pub benchmark_replications: usize,
    pub benchmark_order_routing_mode: &'static str,
    pub paper_action_space: &'static str,
    pub published_benchmark_cost: f64,
    pub published_ppo_best_cost: Option<f64>,
    pub published_ppo_average_cost: Option<f64>,
    pub notes: &'static str,
}

pub const GEEVERS_2023_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Geevers, van Hezewijk & Mes (2024), Central European Journal of Operations Research 32(3):653-683 (online first 2023)",
    url: "https://doi.org/10.1007/s10100-023-00872-2",
    benchmark_policies: &["node_base_stock", "ppo"],
    notes:
        "The general-network (CardBoard Company) section reports three benchmark rows. The three experiment sets differ in the agent/benchmark action space: set 1 places one order per stock point (relative-rationing routing to a single upstream connection), set 2 places one order per edge, and set 3 places one order per edge with a restricted transition function. All three reuse the Kunnumkal & Topaloglu (2011) holding/backorder costs (warehouse holding 0.6, retailer holding 1.0, retailer backorder 19.0, no warehouse backorder cost), Poisson(15) retailer demand, and unit lead times. The benchmark base-stock levels are tuned to a 98% fill-rate target on the corrugated-plant (retailer) connections. The cost window the repo accumulates (periods 50..100) matches the paper's 50-period warm-up + 50-period evaluation window. NOTE: the earlier MA thesis (essay.utwente.nl/85432) reports only the set-1 benchmark (cost 10467, base-stock [82,100,64,83,35,35,35,35,35], 50 periods x 500 reps); the set 2/3 rows (4797) come only from the journal version, whose per-edge benchmark mechanics could not be recovered verbatim from open sources.",
};

pub const CBC_SUPPLIER_LEAD_TIMES: &[usize] = &[1, 1, 1, 1];

pub const CBC_RETAIL_EDGES: &[RetailConnectionEdge] = &[
    RetailConnectionEdge {
        warehouse_idx: 0,
        retailer_idx: 0,
        connection_weight: 0.60,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 0,
        retailer_idx: 1,
        connection_weight: 0.50,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 0,
        retailer_idx: 2,
        connection_weight: 0.15,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 1,
        retailer_idx: 0,
        connection_weight: 0.30,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 1,
        retailer_idx: 1,
        connection_weight: 0.40,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 1,
        retailer_idx: 2,
        connection_weight: 0.80,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 1,
        retailer_idx: 3,
        connection_weight: 0.10,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 2,
        retailer_idx: 3,
        connection_weight: 0.80,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 2,
        retailer_idx: 4,
        connection_weight: 0.70,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 3,
        retailer_idx: 0,
        connection_weight: 0.10,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 3,
        retailer_idx: 1,
        connection_weight: 0.10,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 3,
        retailer_idx: 2,
        connection_weight: 0.05,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 3,
        retailer_idx: 3,
        connection_weight: 0.10,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 3,
        retailer_idx: 4,
        connection_weight: 0.30,
        lead_time: 1,
    },
];

pub const CBC_WAREHOUSE_HOLDING_COSTS: &[f64] = &[0.6, 0.6, 0.6, 0.6];
pub const CBC_RETAILER_HOLDING_COSTS: &[f64] = &[1.0, 1.0, 1.0, 1.0, 1.0];
pub const CBC_WAREHOUSE_BACKORDER_COSTS: &[f64] = &[0.0, 0.0, 0.0, 0.0];
pub const CBC_RETAILER_BACKORDER_COSTS: &[f64] = &[19.0, 19.0, 19.0, 19.0, 19.0];

pub const GEEVERS_SET1_BASE_STOCK_LEVELS: &[usize] = &[82, 100, 64, 83, 35, 35, 35, 35, 35];
pub const GEEVERS_SET23_BASE_STOCK_LEVELS: &[usize] = &[37, 47, 33, 63, 30, 30, 30, 30, 30];

pub const LITERATURE_REFERENCE_INSTANCES: &[GeneralBackorderFixedCostReferenceInstance] = &[
    GeneralBackorderFixedCostReferenceInstance {
        name: "geevers2023_general_set1",
        source: GEEVERS_2023_REFERENCE.source,
        url: GEEVERS_2023_REFERENCE.url,
        literature_verified: false,
        num_suppliers: 4,
        num_warehouses: 4,
        num_retailers: 5,
        supplier_lead_times: CBC_SUPPLIER_LEAD_TIMES,
        retail_edges: CBC_RETAIL_EDGES,
        retailer_demand_mean: 15.0,
        warehouse_holding_costs: CBC_WAREHOUSE_HOLDING_COSTS,
        retailer_holding_costs: CBC_RETAILER_HOLDING_COSTS,
        warehouse_backorder_costs: CBC_WAREHOUSE_BACKORDER_COSTS,
        retailer_backorder_costs: CBC_RETAILER_BACKORDER_COSTS,
        benchmark_base_stock_levels: GEEVERS_SET1_BASE_STOCK_LEVELS,
        benchmark_periods: 100,
        benchmark_warm_up_periods: 50,
        benchmark_replications: 500,
        benchmark_order_routing_mode: "random_single_connection_by_weight",
        paper_action_space: "order_per_stock_point",
        published_benchmark_cost: 10_467.0,
        published_ppo_best_cost: Some(8_714.0),
        published_ppo_average_cost: Some(630_401.0),
        notes:
            "Experiment set 1 (order per stock point). Each retailer order is routed to exactly one upstream warehouse drawn according to the historical connection weights (relative rationing). REPRODUCED: the repo node-base-stock simulation gives mean cost ~10355 vs published 10467 (gap -1.1%, 500 reps x 3 seeds) with warehouse and retailer fill rates in the 98-99% band, so this row is reproduced within the simulation-protocol tolerance. (The thesis also reports this exact row.)",
    },
    GeneralBackorderFixedCostReferenceInstance {
        name: "geevers2023_general_set2",
        source: GEEVERS_2023_REFERENCE.source,
        url: GEEVERS_2023_REFERENCE.url,
        literature_verified: false,
        num_suppliers: 4,
        num_warehouses: 4,
        num_retailers: 5,
        supplier_lead_times: CBC_SUPPLIER_LEAD_TIMES,
        retail_edges: CBC_RETAIL_EDGES,
        retailer_demand_mean: 15.0,
        warehouse_holding_costs: CBC_WAREHOUSE_HOLDING_COSTS,
        retailer_holding_costs: CBC_RETAILER_HOLDING_COSTS,
        warehouse_backorder_costs: CBC_WAREHOUSE_BACKORDER_COSTS,
        retailer_backorder_costs: CBC_RETAILER_BACKORDER_COSTS,
        benchmark_base_stock_levels: GEEVERS_SET23_BASE_STOCK_LEVELS,
        benchmark_periods: 100,
        benchmark_warm_up_periods: 50,
        benchmark_replications: 500,
        benchmark_order_routing_mode: "split_across_all_connections_by_weight",
        paper_action_space: "order_per_edge",
        published_benchmark_cost: 4_797.0,
        published_ppo_best_cost: Some(4_175.0),
        published_ppo_average_cost: Some(314_923.0),
        notes:
            "Experiment set 2 (order per edge). NOT REPRODUCED. Under the configured split_across_all_connections_by_weight routing the repo gives ~15306 (gap +219%); no available routing mode (by-weight / evenly / duplicate / weighted / single) reproduces 4797 at the published base-stock [.,.,.,.,30,30,30,30,30]. ROOT CAUSE (diagnostic): with evenly-split per-edge ordering the repo needs retailer order-up-to ~36-37 (not 30) to hit BOTH cost ~4797 AND the paper's ~98% retailer fill simultaneously - a consistent ~6-7 unit offset in the retailer order-up-to level. This offset is the signature of a different per-edge inventory-position / order-up-to timing convention in the journal's order-per-edge transition (the exact equation is in the gated journal full text and could not be recovered). Carried as a published row; verification target = reproduce 4797 with the published level 30 once the per-edge transition is specified.",
    },
    GeneralBackorderFixedCostReferenceInstance {
        name: "geevers2023_general_set3",
        source: GEEVERS_2023_REFERENCE.source,
        url: GEEVERS_2023_REFERENCE.url,
        literature_verified: false,
        num_suppliers: 4,
        num_warehouses: 4,
        num_retailers: 5,
        supplier_lead_times: CBC_SUPPLIER_LEAD_TIMES,
        retail_edges: CBC_RETAIL_EDGES,
        retailer_demand_mean: 15.0,
        warehouse_holding_costs: CBC_WAREHOUSE_HOLDING_COSTS,
        retailer_holding_costs: CBC_RETAILER_HOLDING_COSTS,
        warehouse_backorder_costs: CBC_WAREHOUSE_BACKORDER_COSTS,
        retailer_backorder_costs: CBC_RETAILER_BACKORDER_COSTS,
        benchmark_base_stock_levels: GEEVERS_SET23_BASE_STOCK_LEVELS,
        benchmark_periods: 100,
        benchmark_warm_up_periods: 50,
        benchmark_replications: 500,
        benchmark_order_routing_mode: "split_across_all_connections_by_weight",
        paper_action_space: "order_per_edge_with_transition_limit",
        published_benchmark_cost: 4_797.0,
        published_ppo_best_cost: Some(3_935.0),
        published_ppo_average_cost: Some(4_481.0),
        notes:
            "Experiment set 3 (order per edge with a RESTRICTED transition function). NOT REPRODUCED. The benchmark base-stock policy is the same as set 2 (cost 4797); the difference vs set 2 is the restricted transition function used during PPO training (it stabilises learning, which is why the PPO average 4481 is close to the PPO best 3935 here, unlike set 2's 314923). The repo does not implement the restricted transition, so set 3 inherits set 2's per-edge reproduction gap. The restricted-transition specification is in the gated journal full text.",
    },
];

pub const PRIMARY_REFERENCE_INSTANCE: &GeneralBackorderFixedCostReferenceInstance =
    &LITERATURE_REFERENCE_INSTANCES[2];

pub fn reference_instance_by_name(
    name: &str,
) -> Option<&'static GeneralBackorderFixedCostReferenceInstance> {
    LITERATURE_REFERENCE_INSTANCES
        .iter()
        .find(|instance| instance.name == name)
}
