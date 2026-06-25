use crate::problems::multi_echelon::general_backorder_fixed_cost::env::RetailConnectionEdge;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

/// Per-retailer customer-demand process used by the simulator/rollout.
///
/// `FixedPoisson` (the historical, set-1/2/3 behaviour): every retailer draws
/// `Poisson(retailer_demand_mean)` every period -- a single fixed mean shared by all retailers
/// and all periods. This is byte-for-byte the original code path; set 1 still reproduces ~10,355.
///
/// `ResampledUniformPoisson` (the general-network divergent Kunnumkal & Topaloglu base case, via
/// the open Geevers 2020 thesis): every period and every retailer first draws a fresh Poisson
/// mean `alpha ~ Uniform[demand_alpha_min, demand_alpha_max]`, then draws `Poisson(alpha)`. The
/// mean is nonstationary (resampled each period per retailer); `retailer_demand_mean` is ignored
/// for the demand draw and only carried as a representative scalar for feature scaling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemandMode {
    FixedPoisson,
    ResampledUniformPoisson,
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
    /// Demand process selector. Sets 1/2/3 use `"fixed_poisson"`; the general-network divergent
    /// instance uses `"resampled_uniform_poisson"`. Parsed by `parse_demand_mode`.
    pub demand_mode: &'static str,
    /// Lower/upper bound of the per-period, per-retailer Poisson mean for
    /// `resampled_uniform_poisson`. Ignored for `fixed_poisson`.
    pub demand_alpha_min: f64,
    pub demand_alpha_max: f64,
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
        "The general-network (CardBoard Company) section reports three benchmark rows. The three experiment sets differ in the agent/benchmark action space: set 1 places one order per stock point (relative-rationing routing to a single upstream connection), set 2 places one order per edge, and set 3 places one order per edge with a restricted transition function. All three reuse the Kunnumkal & Topaloglu (2011) holding/backorder costs (warehouse holding 0.6, retailer holding 1.0, retailer backorder 19.0, no warehouse backorder cost), Poisson(15) retailer demand, and unit lead times. The benchmark base-stock levels are tuned to a 98% fill-rate target on the corrugated-plant (retailer) connections. NOTE ON THE EVALUATION WINDOW: the open MA thesis (essay.utwente.nl/85432, Sec. 6.6) measures the benchmark over a 50-period planning horizon replicated 500 times, and the thesis warm-up convention (Sec. 5.6) is a 25-period warm-up applied to a 75-period run (first 25 removed) -- NOT the 50-warm-up + 50-eval window stated in earlier drafts of this note. The repo accumulates periods 50..100 (benchmark_periods=100, benchmark_warm_up_periods=50); this is the repo's own protocol choice and is NOT verbatim from the open thesis (whose set-1 benchmark uses a 50-period horizon, 25-period warm-up). The set-1 reproduction (~10355 vs 10467, gap -1.1%) holds under the repo protocol, so the window difference is not the source of any set-1 gap, but the field values should not be presented as quoted from the paper. VERIFIED FROM OPEN THESIS (Sec. 6.6, verbatim): set-1 base-stock [82,100,64,83,35,35,35,35,35], cost 10467, 50 periods x 500 reps; costs (Sec. 2.2.5 / 6.1) warehouse holding 0.6, retailer holding 1.0, retailer backorder 19.0, no warehouse backorder, attributed to Kunnumkal & Topaloglu (2011); Poisson(lambda=15) demand (Sec. 6.1). NOT INDEPENDENTLY VERIFIABLE FROM OPEN SOURCES: the set 2/3 rows (benchmark 4797, levels [37,47,33,63,30,30,30,30,30], PPO 4175/3935, averages 314923/4481) appear only in the gated journal full text (thesis covers set 1 only; SSRN preprint returns 403); they are carried as published rows and have NOT been confirmed against an open copy of the paper.",
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

// ---------------------------------------------------------------------------------------------
// General-network DIVERGENT instance (Kunnumkal & Topaloglu 2011 base case, via the open Geevers
// 2020 thesis, essay.utwente.nl/85432, Ch. 5). This is a SEPARATE topology from the CardBoard
// Company general network above: 1 supplier -> 1 warehouse -> 3 identical retailers (a pure
// divergent two-echelon tree). It is also DISTINCT from the paper's Section-6
// Gijsbrechts/Van-Roy "divergent special-delivery" model -- name it unambiguously the
// "general-network divergent (Kunnumkal-Topaloglu)" instance.
//
// Each retailer is fed by exactly one unit-weight edge from the single warehouse, so there is no
// rationing/routing ambiguity. Costs are the SAME Kunnumkal & Topaloglu (2011) costs the env
// already uses for set 1 (warehouse holding 0.6, retailer holding 1.0, retailer backorder 19.0,
// no warehouse backorder cost). Lead times are all 1. Customer demand is Poisson with a
// per-period, per-retailer mean alpha ~ Uniform[5,15] (resampled_uniform_poisson demand mode).
pub const DIVERGENT_KT_SUPPLIER_LEAD_TIMES: &[usize] = &[1];

pub const DIVERGENT_KT_RETAIL_EDGES: &[RetailConnectionEdge] = &[
    RetailConnectionEdge {
        warehouse_idx: 0,
        retailer_idx: 0,
        connection_weight: 1.0,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 0,
        retailer_idx: 1,
        connection_weight: 1.0,
        lead_time: 1,
    },
    RetailConnectionEdge {
        warehouse_idx: 0,
        retailer_idx: 2,
        connection_weight: 1.0,
        lead_time: 1,
    },
];

pub const DIVERGENT_KT_WAREHOUSE_HOLDING_COSTS: &[f64] = &[0.6];
pub const DIVERGENT_KT_RETAILER_HOLDING_COSTS: &[f64] = &[1.0, 1.0, 1.0];
pub const DIVERGENT_KT_WAREHOUSE_BACKORDER_COSTS: &[f64] = &[0.0];
pub const DIVERGENT_KT_RETAILER_BACKORDER_COSTS: &[f64] = &[19.0, 19.0, 19.0];

// Published benchmark base-stock levels: warehouse 124, each retailer 30.
pub const DIVERGENT_KT_BASE_STOCK_LEVELS: &[usize] = &[124, 30, 30, 30];

pub const LITERATURE_REFERENCE_INSTANCES: &[GeneralBackorderFixedCostReferenceInstance] = &[
    GeneralBackorderFixedCostReferenceInstance {
        name: "geevers2023_general_set1",
        source: GEEVERS_2023_REFERENCE.source,
        url: GEEVERS_2023_REFERENCE.url,
        literature_verified: true,
        num_suppliers: 4,
        num_warehouses: 4,
        num_retailers: 5,
        supplier_lead_times: CBC_SUPPLIER_LEAD_TIMES,
        retail_edges: CBC_RETAIL_EDGES,
        demand_mode: "fixed_poisson",
        demand_alpha_min: 15.0,
        demand_alpha_max: 15.0,
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
            "Experiment set 1 (order per stock point). Each retailer order is routed to exactly one upstream warehouse drawn according to the historical connection weights (relative rationing). REPRODUCED: the repo node-base-stock simulation gives mean cost ~10355 vs published 10467 (gap -1.1%, 500 reps x 3 seeds) with warehouse and retailer fill rates in the 98-99% band, so this row is reproduced within the simulation-protocol tolerance. (The thesis also reports this exact row.) Asserted in-crate by tests::verification::set1_benchmark_reproduces_geevers_published_cost (mean within 5% of 10467).",
    },
    GeneralBackorderFixedCostReferenceInstance {
        name: "geevers2023_general_set2",
        source: GEEVERS_2023_REFERENCE.source,
        url: GEEVERS_2023_REFERENCE.url,
        // carried as published-row provenance only; env does NOT reproduce this cost (+~223%),
        // see VERIFICATION_LEDGER/README.md debt D2.
        literature_verified: false,
        num_suppliers: 4,
        num_warehouses: 4,
        num_retailers: 5,
        supplier_lead_times: CBC_SUPPLIER_LEAD_TIMES,
        retail_edges: CBC_RETAIL_EDGES,
        demand_mode: "fixed_poisson",
        demand_alpha_min: 15.0,
        demand_alpha_max: 15.0,
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
        // NOTE: set 2/3 stay literature_verified=false (not reproduced). Only set 1 is verified
        // in-crate (general_backorder_fixed_cost::tests::verification::set1_benchmark_reproduces_geevers_published_cost).
    },
    GeneralBackorderFixedCostReferenceInstance {
        name: "geevers2023_general_set3",
        source: GEEVERS_2023_REFERENCE.source,
        url: GEEVERS_2023_REFERENCE.url,
        // carried as published-row provenance only; env does NOT reproduce this cost (+~223%),
        // see VERIFICATION_LEDGER/README.md debt D2.
        literature_verified: false,
        num_suppliers: 4,
        num_warehouses: 4,
        num_retailers: 5,
        supplier_lead_times: CBC_SUPPLIER_LEAD_TIMES,
        retail_edges: CBC_RETAIL_EDGES,
        demand_mode: "fixed_poisson",
        demand_alpha_min: 15.0,
        demand_alpha_max: 15.0,
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
    GeneralBackorderFixedCostReferenceInstance {
        name: "kunnumkal_topaloglu_divergent",
        source: "Kunnumkal & Topaloglu (2011) base case, reported via the open MSc thesis Geevers (2020), essay.utwente.nl/85432, Ch. 5",
        url: "https://essay.utwente.nl/85432/",
        // VERIFIED (2026-06-05): the executing verification test confirms the env reproduces the
        // published benchmark 4,059 within tolerance -- observed mean ~3,931 (gap -3.0%, 3 seeds x
        // 1000 reps) with all three retailer fill rates ~98.6% (matching the K&T ~98% design
        // target). Same direction/magnitude as set 1's -1.1% simulator residual. See
        // tests::verification::divergent_kt_benchmark_reproduces_published_cost.
        literature_verified: true,
        num_suppliers: 1,
        num_warehouses: 1,
        num_retailers: 3,
        supplier_lead_times: DIVERGENT_KT_SUPPLIER_LEAD_TIMES,
        retail_edges: DIVERGENT_KT_RETAIL_EDGES,
        demand_mode: "resampled_uniform_poisson",
        demand_alpha_min: 5.0,
        demand_alpha_max: 15.0,
        // Representative scalar mean (E[Uniform[5,15]] = 10), used only for feature scaling in the
        // compact-summary policy; the demand draw itself uses the per-period resampled alpha.
        retailer_demand_mean: 10.0,
        warehouse_holding_costs: DIVERGENT_KT_WAREHOUSE_HOLDING_COSTS,
        retailer_holding_costs: DIVERGENT_KT_RETAILER_HOLDING_COSTS,
        warehouse_backorder_costs: DIVERGENT_KT_WAREHOUSE_BACKORDER_COSTS,
        retailer_backorder_costs: DIVERGENT_KT_RETAILER_BACKORDER_COSTS,
        benchmark_base_stock_levels: DIVERGENT_KT_BASE_STOCK_LEVELS,
        // Thesis protocol (Ch. 5): horizon 50, 25-period warm-up applied to a 75-period run (first
        // 25 dropped), 1000 replications. The repo accumulates periods 25..75.
        benchmark_periods: 75,
        benchmark_warm_up_periods: 25,
        benchmark_replications: 1000,
        // Single unit-weight edge per retailer => routing is degenerate; any single-connection mode
        // sends the whole order down the one edge. Use the same mode as set 1 for clarity.
        benchmark_order_routing_mode: "random_single_connection_by_weight",
        paper_action_space: "order_per_stock_point",
        published_benchmark_cost: 4_059.0,
        published_ppo_best_cost: Some(3_724.0),
        published_ppo_average_cost: None,
        notes:
            "General-network DIVERGENT instance (Kunnumkal & Topaloglu 2011 base case, via the open Geevers 2020 thesis, essay.utwente.nl/85432, Ch. 5). Topology: 1 supplier -> 1 warehouse -> 3 identical retailers (NOT the 4-warehouse/5-retailer CardBoard network of set 1/2/3, and DISTINCT from the paper's Section-6 Gijsbrechts/Van-Roy divergent special-delivery model). Costs are the SAME Kunnumkal & Topaloglu (2011) costs as set 1 (warehouse holding 0.6, retailer holding 1.0, retailer backorder 19.0, no warehouse backorder). Lead time 1 everywhere. Demand: Poisson(alpha) with alpha ~ Uniform[5,15] RESAMPLED EVERY PERIOD PER RETAILER (nonstationary mean); warehouse demand = sum of retailer demands. Protocol: horizon 50, 25-period warm-up on a 75-period run, 1000 reps. Published constant node-base-stock benchmark cost 4,059 at levels [warehouse 124, each retailer 30] (confirmed open). Published DRL 3,724 (cross-protocol context, not a head-to-head beat). VERIFICATION: asserted in-crate by tests::verification::divergent_kt_benchmark_reproduces_published_cost (re-runs the env's node-base-stock gate at [124,30,30,30] under the resampled_uniform_poisson demand mode and asserts the mean is within tolerance of 4,059).",
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

/// Map the reference's `demand_mode` string to the typed `DemandMode`. Unknown strings are an
/// error so a typo cannot silently fall back to the fixed-Poisson default and mask a divergent
/// instance using stationary demand.
pub fn parse_demand_mode(mode: &str) -> pyo3::PyResult<DemandMode> {
    match mode {
        "fixed_poisson" => Ok(DemandMode::FixedPoisson),
        "resampled_uniform_poisson" => Ok(DemandMode::ResampledUniformPoisson),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown demand_mode '{mode}'; expected 'fixed_poisson' or 'resampled_uniform_poisson'"
        ))),
    }
}
