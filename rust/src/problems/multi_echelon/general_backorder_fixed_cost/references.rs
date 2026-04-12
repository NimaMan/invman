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
    source: "Geevers et al. (2023), Central European Journal of Operations Research 32:157-187",
    url: "https://doi.org/10.1007/s10100-023-00872-2",
    benchmark_policies: &["node_base_stock", "ppo"],
    notes:
        "Section 5.4 / Table 7 reports the three benchmark rows for the CardBoard Company general network. The paper uses the same holding/backorder costs as the earlier thesis implementation, but evaluates the benchmark over 100 periods with a 50-period warm-up and 500 replications.",
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
            "Table 7, Experiment set 1. Orders are specified per stock point and each retailer order is routed to exactly one upstream warehouse according to the historical connection weights from the CardBoard Company network.",
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
            "Table 7, Experiment set 2. Orders are split across all upstream connections, which lowers the effective uncertainty seen by the warehouses compared with set 1.",
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
            "Table 7, Experiment set 3. The benchmark is the same base-stock policy as in set 2; only the PPO training protocol changes.",
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
