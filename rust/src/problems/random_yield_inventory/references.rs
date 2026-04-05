#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomYieldReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub periods: usize,
    pub lead_time: usize,
    pub demand_mean: f64,
    pub success_probability: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub procurement_cost: f64,
    pub discount_factor: f64,
    pub initial_inventory_level: f64,
    pub initial_pipeline_orders: &'static [f64],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkedTransitionReference {
    pub source: &'static str,
    pub url: &'static str,
    pub lead_time: usize,
    pub initial_inventory_level: f64,
    pub initial_pipeline_orders: &'static [f64],
    pub action: f64,
    pub realized_demand: f64,
    pub arrival_succeeds: bool,
    pub expected_arrival: f64,
    pub expected_next_inventory_level: f64,
    pub expected_next_pipeline_orders: &'static [f64],
    pub expected_period_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
    pub periods: usize,
    pub lead_time: usize,
    pub success_probability: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub procurement_cost: f64,
    pub discount_factor: f64,
    pub initial_inventory_level: i32,
    pub initial_pipeline_orders: &'static [u32],
    pub demand_support: &'static [u32],
    pub demand_probabilities: &'static [f64],
    pub max_order_quantity: usize,
    pub expected_optimal_discounted_cost: f64,
    pub expected_optimal_first_action: usize,
    pub expected_linear_inflation_discounted_cost: f64,
    pub expected_linear_inflation_first_action: usize,
    pub expected_weighted_newsvendor_discounted_cost: f64,
    pub expected_weighted_newsvendor_first_action: usize,
}

pub const YAN_2026_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Yan et al. (2026), Computers & Operations Research 186, 107305",
    url: "https://doi.org/10.1016/j.cor.2025.107305",
    benchmark_policies: &["optimal_dp", "linear_inflation", "weighted_newsvendor", "gdsh", "drl"],
    notes: "This paper defines the finite-horizon discounted all-or-nothing random-yield problem with positive lead time and backlogging. The article preview exposes the model and benchmark policy families, but not a clean table of exact per-instance benchmark rows accessible for repo assertions.",
};

pub const INDERFURTH_2015_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Inderfurth and Kiesmuller (2015), linear-inflation policies under random yield",
    url: "https://www.fww.ovgu.de/fww_media/femm/femm_2013/2013_07.pdf",
    benchmark_policies: &["linear_inflation"],
    notes: "This paper gives the canonical linear-inflation rule q = F * (S - X)^+ and the standard choice F = 1/p for binomial yield. It also reports the lead-time-zero benchmark grid with mean demand 20, h = 1, p in {0.5, 0.7, 0.9}, and several critical ratios.",
};

pub const CHEN_2018_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Chen et al. (2018), Heuristics and Bounds for an Inventory System with an All-or-Nothing Yield Pattern and Lead-times",
    url: "https://dblp.org/rec/conf/soli/ChenHYY18",
    benchmark_policies: &["weighted_newsvendor"],
    notes: "The later Yan et al. (2026) paper identifies this weighted newsvendor heuristic as the main all-or-nothing benchmark policy and describes it as a sample-path weighted-average gap rule over pipeline-yield realizations.",
};

pub const PRIMARY_REFERENCE_INSTANCE: RandomYieldReferenceInstance = RandomYieldReferenceInstance {
    name: "yan2026_style_lt2_p075_discounted",
    source: YAN_2026_REFERENCE.source,
    url: YAN_2026_REFERENCE.url,
    periods: 12,
    lead_time: 2,
    demand_mean: 4.0,
    success_probability: 0.75,
    holding_cost: 1.0,
    shortage_cost: 9.0,
    procurement_cost: 1.0,
    discount_factor: 0.99,
    initial_inventory_level: 6.0,
    initial_pipeline_orders: &[4.0, 3.0],
    notes: "Canonical first learned-policy benchmark for the repo. This is a repo-native instance shaped after the small-lead-time discounted setting emphasized by Yan et al. (2026), not a verbatim table row from the paper.",
};

pub const WORKED_TRANSITION_REFERENCE: WorkedTransitionReference = WorkedTransitionReference {
    source: YAN_2026_REFERENCE.source,
    url: YAN_2026_REFERENCE.url,
    lead_time: 2,
    initial_inventory_level: 3.0,
    initial_pipeline_orders: &[5.0, 2.0],
    action: 4.0,
    realized_demand: 6.0,
    arrival_succeeds: true,
    expected_arrival: 5.0,
    expected_next_inventory_level: 2.0,
    expected_next_pipeline_orders: &[2.0, 4.0],
    expected_period_cost: 6.0,
};

pub const VERIFICATION_DEMAND_SUPPORT: &[u32] = &[0, 1, 2, 3, 4, 5];
pub const VERIFICATION_DEMAND_PROBABILITIES: &[f64] = &[0.05, 0.15, 0.30, 0.25, 0.15, 0.10];

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: YAN_2026_REFERENCE.source,
    url: YAN_2026_REFERENCE.url,
    periods: 5,
    lead_time: 2,
    success_probability: 0.75,
    holding_cost: 1.0,
    shortage_cost: 9.0,
    procurement_cost: 1.0,
    discount_factor: 0.99,
    initial_inventory_level: 4,
    initial_pipeline_orders: &[3, 2],
    demand_support: VERIFICATION_DEMAND_SUPPORT,
    demand_probabilities: VERIFICATION_DEMAND_PROBABILITIES,
    max_order_quantity: 8,
    expected_optimal_discounted_cost: 40.05989760985441,
    expected_optimal_first_action: 4,
    expected_linear_inflation_discounted_cost: 47.71379457283354,
    expected_linear_inflation_first_action: 4,
    expected_weighted_newsvendor_discounted_cost: 60.3935751430189,
    expected_weighted_newsvendor_first_action: 8,
};
