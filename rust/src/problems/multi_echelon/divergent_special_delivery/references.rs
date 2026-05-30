#![allow(dead_code)]

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiEchelonReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub warehouse_lead_time: usize,
    pub retailer_lead_time: usize,
    pub num_retailers: usize,
    pub warehouse_holding_cost: f64,
    pub retailer_holding_cost: f64,
    pub warehouse_expedited_cost: f64,
    pub warehouse_lost_sale_cost: f64,
    pub expedited_service_prob: f64,
    pub warehouse_capacity: usize,
    pub warehouse_inventory_cap: usize,
    pub retailer_inventory_cap: usize,
    pub inventory_dynamics_mode: &'static str,
    pub demand_distribution: &'static str,
    pub demand_mean: f64,
    pub demand_std: f64,
    pub benchmark_search_horizon: usize,
    pub benchmark_periods: usize,
    pub benchmark_replications: usize,
    pub warm_up_periods_ratio: f64,
    pub rollout_objective: &'static str,
    pub warehouse_base_stock_mode: &'static str,
    pub policy_allocation_mode: &'static str,
    pub benchmark_warehouse_levels: &'static [usize],
    pub benchmark_retailer_levels: &'static [usize],
    pub published_constant_base_stock_mean_cost: Option<f64>,
    pub published_constant_base_stock_levels: &'static [usize],
    pub published_van_roy_best_ndp_mean_cost: Option<f64>,
    pub published_a3c_savings_pct: Option<f64>,
    pub published_a3c_confidence_half_width_pct: Option<f64>,
    pub published_van_roy_savings_pct_approx: Option<f64>,
    pub tuned_learning_rate: Option<f64>,
    pub tuned_entropy_regularization: Option<f64>,
    pub tuned_buffer_length: Option<usize>,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub periods: usize,
    pub warehouse_lead_time: usize,
    pub retailer_lead_time: usize,
    pub num_retailers: usize,
    pub warehouse_holding_cost: f64,
    pub retailer_holding_cost: f64,
    pub warehouse_expedited_cost: f64,
    pub warehouse_lost_sale_cost: f64,
    pub expedited_service_prob: f64,
    pub warehouse_capacity: usize,
    pub warehouse_inventory_cap: usize,
    pub retailer_inventory_cap: usize,
    pub inventory_dynamics_mode: &'static str,
    pub warehouse_base_stock_mode: &'static str,
    pub allocation_mode: &'static str,
    pub discount_factor: f64,
    pub initial_warehouse_inventory: i32,
    pub initial_warehouse_pipeline: &'static [u32],
    pub initial_retailer_inventory: &'static [i32],
    pub initial_retailer_pipeline: &'static [&'static [u32]],
    pub demand_support: &'static [u32],
    pub demand_probabilities: &'static [f64],
    pub action_warehouse_levels: &'static [usize],
    pub action_retailer_levels: &'static [usize],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkedTransitionReference {
    pub source: &'static str,
    pub url: &'static str,
    pub initial_warehouse_inventory: i32,
    pub initial_warehouse_pipeline: &'static [u32],
    pub initial_retailer_inventory: &'static [i32],
    pub initial_retailer_pipeline: &'static [&'static [u32]],
    pub warehouse_target: usize,
    pub retailer_target: usize,
    pub realized_demands: &'static [u32],
    pub accepted_emergency_shipments: usize,
    pub warehouse_base_stock_mode: &'static str,
    pub allocation_mode: &'static str,
    pub expected_warehouse_order: usize,
    pub expected_shipped_retail_orders: &'static [usize],
    pub expected_next_warehouse_inventory: i32,
    pub expected_next_warehouse_pipeline: &'static [u32],
    pub expected_next_retailer_inventory: &'static [i32],
    pub expected_next_retailer_pipeline: &'static [&'static [u32]],
    pub expected_period_cost: f64,
}

pub const GIJSBRECHTS_2022_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source:
        "Gijsbrechts et al. (2022), Manufacturing & Service Operations Management 24(3):1349-1368",
    url: "https://doi.org/10.1287/msom.2021.1064",
    benchmark_policies: &[
        "constant_base_stock",
        "van_roy_neuro_dynamic_programming",
        "a3c",
    ],
    notes:
        "Section 7.2 / Table 3 reuses the two Van Roy case-study settings and reports A3C relative improvements over constant base-stock: 8.95% and 12.09%. The absolute constant base-stock costs for those same settings come from the original Van Roy full-length report.",
};

pub const VAN_ROY_1997_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source:
        "Van Roy et al. (1997), Proceedings of the 36th IEEE Conference on Decision and Control",
    url: "https://www.mit.edu/~jnt/Papers/C-97-bvr-retail-CDC.pdf",
    benchmark_policies: &["constant_base_stock", "neuro_dynamic_programming"],
    notes:
        "The original CDC paper and the linked full-length report give executable demand-generation, heuristic, and NDP benchmark rows for one simple problem and two complex case studies. Transportation costs in the model are associated only with special deliveries.",
};

pub const GIJS_SETTING_WAREHOUSE_LEVELS: &[usize] = &[50, 60, 70, 80, 90, 100];
pub const GIJS_SETTING1_RETAILER_LEVELS: &[usize] = &[0, 5, 10, 15, 20, 25, 30, 35, 40];
pub const GIJS_SETTING2_RETAILER_LEVELS: &[usize] = &[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
pub const VAN_ROY_SIMPLE_ORDER_LEVELS: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
pub const VAN_ROY_SIMPLE_RETAILER_LEVELS: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50,
];
pub const VAN_ROY_SIMPLE_LEVELS: &[usize] = &[10, 16];
pub const VAN_ROY_CASE_STUDY_LEVELS: &[usize] = &[330, 23];
pub const VAN_ROY_CASE_STUDY2_LEVELS: &[usize] = &[460, 22];

pub const LITERATURE_REFERENCE_INSTANCES: &[MultiEchelonReferenceInstance] = &[
    MultiEchelonReferenceInstance {
        name: "van_roy1997_simple_problem",
        source: VAN_ROY_1997_REFERENCE.source,
        url: "https://www.stanford.edu/~bvr/pubs/retail.pdf",
        literature_verified: false,
        warehouse_lead_time: 0,
        retailer_lead_time: 1,
        num_retailers: 1,
        warehouse_holding_cost: 1.0,
        retailer_holding_cost: 2.0,
        warehouse_expedited_cost: 10.0,
        warehouse_lost_sale_cost: 50.0,
        expedited_service_prob: 1.0,
        warehouse_capacity: 10,
        warehouse_inventory_cap: 50,
        retailer_inventory_cap: 50,
        inventory_dynamics_mode: "van_roy_1997",
        demand_distribution: "normal_rounded_clipped",
        demand_mean: 6.294,
        demand_std: 6.2,
        benchmark_search_horizon: 100_000,
        benchmark_periods: 100_000,
        benchmark_replications: 100,
        warm_up_periods_ratio: 0.0,
        rollout_objective: "average_cost_after_warmup",
        warehouse_base_stock_mode: "regular",
        policy_allocation_mode: "min_shortage",
        benchmark_warehouse_levels: VAN_ROY_SIMPLE_ORDER_LEVELS,
        benchmark_retailer_levels: VAN_ROY_SIMPLE_RETAILER_LEVELS,
        published_constant_base_stock_mean_cost: Some(51.7),
        published_constant_base_stock_levels: VAN_ROY_SIMPLE_LEVELS,
        published_van_roy_best_ndp_mean_cost: Some(52.6),
        published_a3c_savings_pct: None,
        published_a3c_confidence_half_width_pct: None,
        published_van_roy_savings_pct_approx: None,
        tuned_learning_rate: None,
        tuned_entropy_regularization: None,
        tuned_buffer_length: None,
        notes:
            "Simple one-store Van Roy problem from the full-length report. Van Roy specifies demand as N(5, 8). The exact effective moments of that distribution after rounding and clipping to non-negative integers are mean=6.2937 and std=6.2374 (the README approximation of 6.2/6.2 was too coarse). Using demand_mean=6.294 (3-dp rounding of 6.2937) and demand_std=6.2 as latent parameters reproduces the published constant-base-stock cost 51.7 within 0.1% (cost 51.718 at 100 replications). Using (6.2, 6.2) instead produced a -1.4% gap because the effective mean is off by 0.09 units.",
    },
    MultiEchelonReferenceInstance {
        // van_roy_1997-mode REPRODUCTION instance for Van Roy / Gijs complex case study 1.
        // It exists to reproduce the published absolute constant-base-stock cost (1302) and
        // to carry the Gijs A3C relative-savings row. It is NOT the paper-faithful MDP used
        // as the search target -- see gijsbrechts2022_setting1 for that.
        name: "van_roy1997_case_study1",
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        literature_verified: false,
        warehouse_lead_time: 2,
        retailer_lead_time: 2,
        num_retailers: 10,
        warehouse_holding_cost: 3.0,
        retailer_holding_cost: 3.0,
        warehouse_expedited_cost: 0.0,
        warehouse_lost_sale_cost: 60.0,
        expedited_service_prob: 0.8,
        warehouse_capacity: 100,
        warehouse_inventory_cap: 1000,
        retailer_inventory_cap: 100,
        inventory_dynamics_mode: "van_roy_1997",
        demand_distribution: "normal_rounded_clipped",
        demand_mean: 5.0,
        demand_std: 14.0,
        benchmark_search_horizon: 10_000,
        benchmark_periods: 100_000,
        benchmark_replications: 100,
        warm_up_periods_ratio: 0.0,
        rollout_objective: "average_cost_after_warmup",
        warehouse_base_stock_mode: "regular",
        policy_allocation_mode: "min_shortage",
        benchmark_warehouse_levels: GIJS_SETTING_WAREHOUSE_LEVELS,
        benchmark_retailer_levels: GIJS_SETTING1_RETAILER_LEVELS,
        published_constant_base_stock_mean_cost: Some(1302.0),
        published_constant_base_stock_levels: VAN_ROY_CASE_STUDY_LEVELS,
        published_van_roy_best_ndp_mean_cost: Some(1179.0),
        published_a3c_savings_pct: Some(8.95),
        published_a3c_confidence_half_width_pct: Some(0.13),
        published_van_roy_savings_pct_approx: Some(10.0),
        tuned_learning_rate: Some(5.78e-5),
        tuned_entropy_regularization: Some(4.68e-5),
        tuned_buffer_length: Some(100),
        notes:
            "Van Roy complex case study 1, reused by Gijs as setting 1, in van_roy_1997 reproduction mode (post-shipment warehouse order convention + pre-demand holding timing). Latent normal demand mean 5, stdev 14 (effective ~8.4/9.8 after rounding+clipping). Reproduces the published constant base-stock cost 1302 at levels (330,23) within ~1.3%; best NDP cost 1179; A3C improves 8.95% over constant base-stock.",
    },
    MultiEchelonReferenceInstance {
        // van_roy_1997-mode REPRODUCTION instance for Van Roy / Gijs complex case study 2.
        // demand_mean is deliberately kept at the calibrated value 1.0 (not the paper's
        // Table-3 mu=0) because that is the value under which Van Roy's published absolute
        // cost 1449 reproduces. The paper-faithful MDP (mu=0, gijs_2022) is the separate
        // gijsbrechts2022_setting2 instance used as the search target.
        name: "van_roy1997_case_study2",
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        literature_verified: false,
        warehouse_lead_time: 5,
        retailer_lead_time: 3,
        num_retailers: 10,
        warehouse_holding_cost: 3.0,
        retailer_holding_cost: 3.0,
        warehouse_expedited_cost: 0.0,
        warehouse_lost_sale_cost: 60.0,
        expedited_service_prob: 0.8,
        warehouse_capacity: 100,
        warehouse_inventory_cap: 1000,
        retailer_inventory_cap: 100,
        inventory_dynamics_mode: "van_roy_1997",
        demand_distribution: "normal_rounded_clipped",
        demand_mean: 1.0,
        demand_std: 20.0,
        benchmark_search_horizon: 10_000,
        benchmark_periods: 100_000,
        benchmark_replications: 100,
        warm_up_periods_ratio: 0.0,
        rollout_objective: "average_cost_after_warmup",
        warehouse_base_stock_mode: "regular",
        policy_allocation_mode: "min_shortage",
        benchmark_warehouse_levels: GIJS_SETTING_WAREHOUSE_LEVELS,
        benchmark_retailer_levels: GIJS_SETTING2_RETAILER_LEVELS,
        published_constant_base_stock_mean_cost: Some(1449.0),
        published_constant_base_stock_levels: VAN_ROY_CASE_STUDY2_LEVELS,
        published_van_roy_best_ndp_mean_cost: Some(1318.0),
        published_a3c_savings_pct: Some(12.09),
        published_a3c_confidence_half_width_pct: Some(0.39),
        published_van_roy_savings_pct_approx: Some(10.0),
        tuned_learning_rate: Some(1.74e-4),
        tuned_entropy_regularization: Some(1.46e-8),
        tuned_buffer_length: Some(100),
        notes:
            "Van Roy complex case study 2, reused by Gijs as setting 2, in van_roy_1997 reproduction mode. demand_mean is the CALIBRATED value 1.0 (latent), not the paper's Table-3 mu=0: under van_roy_1997 dynamics this reproduces the published constant base-stock cost 1449 at levels (460,22) within ~0.7%. The paper-faithful mu=0 / gijs_2022 environment is gijsbrechts2022_setting2. Best NDP cost 1318; A3C improves 12.09% over constant base-stock.",
    },
    MultiEchelonReferenceInstance {
        // PAPER-FAITHFUL Gijsbrechts (2022) setting 1, used as the autosearch + CMA-ES
        // search target. gijs_2022 dynamics: pre-shipment warehouse order (Eq. (2)) and
        // end-of-period holding. Table-3 latent demand mean 5, stdev 14. No published
        // absolute/relative rows are attached (the faithful MDP does not reproduce Van
        // Roy's absolute cost, which was computed under van_roy_1997 dynamics); correctness
        // is checked by the gijs_2022 exact-DP / worked-transition tests instead. The A3C
        // relative-savings target lives on van_roy1997_case_study1.
        name: "gijsbrechts2022_setting1",
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        literature_verified: false,
        warehouse_lead_time: 2,
        retailer_lead_time: 2,
        num_retailers: 10,
        warehouse_holding_cost: 3.0,
        retailer_holding_cost: 3.0,
        warehouse_expedited_cost: 0.0,
        warehouse_lost_sale_cost: 60.0,
        expedited_service_prob: 0.8,
        warehouse_capacity: 100,
        warehouse_inventory_cap: 1000,
        retailer_inventory_cap: 100,
        inventory_dynamics_mode: "gijs_2022",
        demand_distribution: "normal_rounded_clipped",
        demand_mean: 5.0,
        demand_std: 14.0,
        benchmark_search_horizon: 10_000,
        benchmark_periods: 100_000,
        benchmark_replications: 100,
        warm_up_periods_ratio: 0.0,
        rollout_objective: "average_cost_after_warmup",
        warehouse_base_stock_mode: "regular",
        policy_allocation_mode: "min_shortage",
        benchmark_warehouse_levels: GIJS_SETTING_WAREHOUSE_LEVELS,
        benchmark_retailer_levels: GIJS_SETTING1_RETAILER_LEVELS,
        published_constant_base_stock_mean_cost: None,
        published_constant_base_stock_levels: &[],
        published_van_roy_best_ndp_mean_cost: None,
        published_a3c_savings_pct: None,
        published_a3c_confidence_half_width_pct: None,
        published_van_roy_savings_pct_approx: None,
        tuned_learning_rate: None,
        tuned_entropy_regularization: None,
        tuned_buffer_length: None,
        notes:
            "Paper-faithful Gijsbrechts (2022) setting 1 (Table 3: lw=2, lr=2, mu=5, sigma=20->14, K=10, hw=hr=3, cw=0, p=60, Pw=0.8, Cm=100, Cw=1000, Cr=100) under gijs_2022 dynamics (pre-shipment warehouse order Eq. (2), end-of-period holding). Search target for policy design; not a Van Roy absolute-cost reproduction instance.",
    },
    MultiEchelonReferenceInstance {
        // PAPER-FAITHFUL Gijsbrechts (2022) setting 2, the PRIMARY autosearch + CMA-ES
        // search target. gijs_2022 dynamics + Table-3 latent demand mean 0 (NOT the
        // calibrated 1.0 used by van_roy1997_case_study2). Correctness validated by the
        // gijs_2022 exact-DP / worked-transition tests, not by Van Roy absolute reproduction.
        name: "gijsbrechts2022_setting2",
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        literature_verified: false,
        warehouse_lead_time: 5,
        retailer_lead_time: 3,
        num_retailers: 10,
        warehouse_holding_cost: 3.0,
        retailer_holding_cost: 3.0,
        warehouse_expedited_cost: 0.0,
        warehouse_lost_sale_cost: 60.0,
        expedited_service_prob: 0.8,
        warehouse_capacity: 100,
        warehouse_inventory_cap: 1000,
        retailer_inventory_cap: 100,
        inventory_dynamics_mode: "gijs_2022",
        demand_distribution: "normal_rounded_clipped",
        demand_mean: 0.0,
        demand_std: 20.0,
        benchmark_search_horizon: 10_000,
        benchmark_periods: 100_000,
        benchmark_replications: 100,
        warm_up_periods_ratio: 0.0,
        rollout_objective: "average_cost_after_warmup",
        warehouse_base_stock_mode: "regular",
        policy_allocation_mode: "min_shortage",
        benchmark_warehouse_levels: GIJS_SETTING_WAREHOUSE_LEVELS,
        benchmark_retailer_levels: GIJS_SETTING2_RETAILER_LEVELS,
        published_constant_base_stock_mean_cost: None,
        published_constant_base_stock_levels: &[],
        published_van_roy_best_ndp_mean_cost: None,
        published_a3c_savings_pct: None,
        published_a3c_confidence_half_width_pct: None,
        published_van_roy_savings_pct_approx: None,
        tuned_learning_rate: None,
        tuned_entropy_regularization: None,
        tuned_buffer_length: None,
        notes:
            "Paper-faithful Gijsbrechts (2022) setting 2 (Table 3: lw=5, lr=3, mu=0, sigma=20, K=10, hw=hr=3, cw=0, p=60, Pw=0.8, Cm=100, Cw=1000, Cr=100) under gijs_2022 dynamics. demand_mean is the paper's Table-3 value 0 (latent), NOT the calibrated 1.0. Primary search target for policy design.",
    },
];

// Catalog order: [0] van_roy1997_simple_problem, [1] van_roy1997_case_study1,
// [2] van_roy1997_case_study2, [3] gijsbrechts2022_setting1 (faithful),
// [4] gijsbrechts2022_setting2 (faithful). The primary search target is the faithful
// setting 2; the VAN_ROY_1997_CASE_STUDY consts point at the reproduction instances.
pub const PRIMARY_REFERENCE_INSTANCE: &MultiEchelonReferenceInstance =
    &LITERATURE_REFERENCE_INSTANCES[4];

pub const VAN_ROY_1997_CASE_STUDY1: MultiEchelonReferenceInstance =
    LITERATURE_REFERENCE_INSTANCES[1];
pub const VAN_ROY_1997_CASE_STUDY2: MultiEchelonReferenceInstance =
    LITERATURE_REFERENCE_INSTANCES[2];
pub const VAN_ROY_1997_CASE_STUDY: MultiEchelonReferenceInstance = VAN_ROY_1997_CASE_STUDY1;

pub const EXACT_WAREHOUSE_LEVELS: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7, 8];
pub const EXACT_RETAILER_LEVELS: &[usize] = &[0, 1, 2, 3, 4];
pub const EXACT_INITIAL_WAREHOUSE_PIPELINE: &[u32] = &[1];
pub const EXACT_INITIAL_RETAILER_INVENTORY: &[i32] = &[1, 2];
pub const EXACT_INITIAL_RETAILER_PIPELINE_0: &[u32] = &[1];
pub const EXACT_INITIAL_RETAILER_PIPELINE_1: &[u32] = &[0];
pub const EXACT_INITIAL_RETAILER_PIPELINES: &[&[u32]] = &[
    EXACT_INITIAL_RETAILER_PIPELINE_0,
    EXACT_INITIAL_RETAILER_PIPELINE_1,
];
pub const WORKED_NEXT_RETAILER_PIPELINE_0: &[u32] = &[0];
pub const WORKED_NEXT_RETAILER_PIPELINE_1: &[u32] = &[0];
pub const WORKED_NEXT_RETAILER_PIPELINES: &[&[u32]] = &[
    WORKED_NEXT_RETAILER_PIPELINE_0,
    WORKED_NEXT_RETAILER_PIPELINE_1,
];
pub const EXACT_DEMAND_SUPPORT: &[u32] = &[0, 1, 2, 3];
pub const EXACT_DEMAND_PROBABILITIES: &[f64] = &[0.2, 0.4, 0.3, 0.1];

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: "Repo exact verification instance for the Van Roy / Gijs multi-echelon family",
    url: GIJSBRECHTS_2022_REFERENCE.url,
    literature_verified: false,
    periods: 4,
    warehouse_lead_time: 1,
    retailer_lead_time: 1,
    num_retailers: 2,
    warehouse_holding_cost: 0.5,
    retailer_holding_cost: 1.0,
    warehouse_expedited_cost: 0.0,
    warehouse_lost_sale_cost: 9.0,
    expedited_service_prob: 0.8,
    warehouse_capacity: 6,
    warehouse_inventory_cap: 8,
    retailer_inventory_cap: 4,
    inventory_dynamics_mode: "gijs_2022",
    warehouse_base_stock_mode: "regular",
    allocation_mode: "min_shortage",
    discount_factor: 0.95,
    initial_warehouse_inventory: 4,
    initial_warehouse_pipeline: EXACT_INITIAL_WAREHOUSE_PIPELINE,
    initial_retailer_inventory: EXACT_INITIAL_RETAILER_INVENTORY,
    initial_retailer_pipeline: EXACT_INITIAL_RETAILER_PIPELINES,
    demand_support: EXACT_DEMAND_SUPPORT,
    demand_probabilities: EXACT_DEMAND_PROBABILITIES,
    action_warehouse_levels: EXACT_WAREHOUSE_LEVELS,
    action_retailer_levels: EXACT_RETAILER_LEVELS,
    notes:
        "Reduced finite-horizon verifier for the regular-base-stock plus min-shortage-allocation formulation used in the Van Roy / Gijs benchmark family. The reference stores only the problem instance; optimal and heuristic values are generated by the exact Rust routines at verification time.",
};

pub const WORKED_TRANSITION_REFERENCE: WorkedTransitionReference = WorkedTransitionReference {
    source: VERIFICATION_PROBLEM_INSTANCE.source,
    url: VERIFICATION_PROBLEM_INSTANCE.url,
    initial_warehouse_inventory: 4,
    initial_warehouse_pipeline: EXACT_INITIAL_WAREHOUSE_PIPELINE,
    initial_retailer_inventory: EXACT_INITIAL_RETAILER_INVENTORY,
    initial_retailer_pipeline: EXACT_INITIAL_RETAILER_PIPELINES,
    warehouse_target: 7,
    retailer_target: 2,
    realized_demands: &[2, 3],
    accepted_emergency_shipments: 1,
    warehouse_base_stock_mode: "regular",
    allocation_mode: "min_shortage",
    expected_warehouse_order: 2,
    expected_shipped_retail_orders: &[0, 0],
    expected_next_warehouse_inventory: 4,
    expected_next_warehouse_pipeline: &[2],
    expected_next_retailer_inventory: &[0, 0],
    expected_next_retailer_pipeline: WORKED_NEXT_RETAILER_PIPELINES,
    expected_period_cost: 2.0,
};
