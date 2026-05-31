#![allow(dead_code)]

use crate::problems::ameliorating_inventory::demand::{DemandDistributionKind, DemandModel};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub reported_numbers_available: bool,
    pub numbers_anchor_repo_assertions: bool,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmelioratingInventoryReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub num_ages: usize,
    pub target_ages: &'static [usize],
    pub product_prices: &'static [f64],
    pub demand_models: &'static [DemandModel],
    pub age_retention: &'static [f64],
    pub purchase_cost_per_unit: f64,
    pub holding_cost_per_unit: f64,
    pub decay_salvage_values: &'static [f64],
    pub benchmark_newsvendor_total_target: usize,
    pub benchmark_two_dimensional_total_target: usize,
    pub benchmark_two_dimensional_young_target: usize,
    pub benchmark_young_age_cutoff: usize,
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
    pub initial_inventory_by_age: &'static [usize],
    pub target_ages: &'static [usize],
    pub product_prices: &'static [f64],
    pub age_retention: &'static [f64],
    pub purchase_cost_per_unit: f64,
    pub holding_cost_per_unit: f64,
    pub decay_salvage_values: &'static [f64],
    pub demand_scenarios: &'static [&'static [u32]],
    pub demand_probabilities: &'static [f64],
    pub max_purchase_quantity: usize,
    pub newsvendor_total_target: usize,
    pub two_dimensional_total_target: usize,
    pub two_dimensional_young_target: usize,
    pub young_age_cutoff: usize,
    pub notes: &'static str,
}

pub const PAHR_GRUNOW_2025_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Pahr and Grunow (2025), Production and Operations Management, Vol. 35 No. 5 (DOI 10.1177/10591478251387795)",
    url: "https://journals.sagepub.com/doi/10.1177/10591478251387795",
    benchmark_policies: &[
        "newsvendor_purchase",
        "two_dimensional_order_up_to",
        "rolling_lp",
        "drl",
    ],
    reported_numbers_available: true,
    numbers_anchor_repo_assertions: false,
    notes: "The paper studies an ameliorating inventory MDP with age-differentiated products, stochastic sales prices, stochastic decay, and blending-based issuance. The current Rust package is a reduced approximation of that family rather than a faithful executable port.",
};

pub const PAHR_GRUNOW_2025_REPOSITORY_REFERENCE: PublishedBenchmarkReference =
    PublishedBenchmarkReference {
        source: "Pahr and Grunow (2025) companion code repository",
        url: "https://github.com/amelioratinginventory/ameliorating_inventory",
        benchmark_policies: &[
            "newsvendor_purchase",
            "two_dimensional_order_up_to",
            "rolling_lp",
        ],
        reported_numbers_available: true,
        numbers_anchor_repo_assertions: false,
        notes: "The public companion repository defaults to ten age classes, three products, stochastic sales prices, stochastic beta decay processes, and LP-based issuance machinery. Those settings are not the executable model used by the current Rust package.",
    };

/// Exact title of the source paper, recorded for provenance.
pub const PAHR_GRUNOW_2025_TITLE: &str =
    "The Value of Blending - Managing Ameliorating Inventory Using Deep Reinforcement Learning";

/// A published perfect-information upper bound recorded from the companion repository's
/// `problem_configurations/<instance>/upper_bound.json`. The paper reports policy
/// performance as the gap to this upper bound under long-run average profit.
///
/// IMPORTANT: these numbers are recorded for provenance only. They do NOT anchor any
/// executable assertion in this package, because the current Rust env optimizes a
/// finite-horizon discounted COST with a purchase-only action and fixed price/decay
/// processes, whereas the upper bound is the long-run average PROFIT of the paper's full
/// model (three-part action of purchasing + production + issuance, stochastic purchase
/// price, stochastic beta decay, processing capacity, and Gaussian-copula demand/price).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedUpperBoundAnchor {
    pub instance: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub num_ages: usize,
    pub num_products: usize,
    /// Perfect-information LP upper bound on long-run average profit (`max_reward`).
    pub upper_bound_average_profit: f64,
    /// Whether this number can currently anchor a repo executable assertion.
    pub anchors_repo_assertion: bool,
    pub notes: &'static str,
}

/// Companion-repo default generic instance `spirits_0001`.
/// Source: problem_configurations/spirits_0001/{config,upper_bound}.json.
pub const PAHR_GRUNOW_2025_SPIRITS_0001_UPPER_BOUND: PublishedUpperBoundAnchor =
    PublishedUpperBoundAnchor {
        instance: "spirits_0001",
        source: PAHR_GRUNOW_2025_REPOSITORY_REFERENCE.source,
        url: "https://github.com/amelioratinginventory/ameliorating_inventory/blob/main/problem_configurations/spirits_0001/upper_bound.json",
        num_ages: 10,
        num_products: 3,
        upper_bound_average_profit: 1991.9344293376805,
        anchors_repo_assertion: false,
        notes: "Generic default instance: target ages [2,4,6], demand means [10,7,5] (CoV 0.25, Gaussian copula with price), purchase price (config price_mean 200, price_std 50, price_truncation 70), sales-price means [250,350,500] (CoV 0.1), age-dependent beta decay (CoV 0.8) plus 0.03 evaporation, capacity 50, holding 2.5. Config fields and the upper bound (max_reward 1991.9344293376805) were read directly from the companion repo's spirits_0001/{config,upper_bound}.json. The paper reports its average-age-blending policy reaching about 3.5% below this upper bound on the generic instance set.",
    };

/// Companion-repo port-wine industry case study.
/// Source: problem_configurations/port_wine/upper_bound.json.
pub const PAHR_GRUNOW_2025_PORT_WINE_UPPER_BOUND: PublishedUpperBoundAnchor =
    PublishedUpperBoundAnchor {
        instance: "port_wine",
        source: PAHR_GRUNOW_2025_REPOSITORY_REFERENCE.source,
        url: "https://github.com/amelioratinginventory/ameliorating_inventory/blob/main/problem_configurations/port_wine/upper_bound.json",
        num_ages: 25,
        num_products: 3,
        upper_bound_average_profit: 2444.80,
        anchors_repo_assertion: false,
        notes: "Port-wine industry case study (25 age classes). The paper reports a gap to this upper bound of about 2.8% for the learned policy.",
    };

/// Headline performance figures reported in Pahr and Grunow (2025), recorded for
/// provenance. These are reductions/gaps relative to baselines, not absolute costs, and
/// do NOT anchor executable assertions in this reduced package.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedPerformanceFigures {
    pub source: &'static str,
    /// DRL reduces the rolling-horizon LP (RLP) gap to the upper bound by this fraction.
    pub drl_vs_rlp_gap_reduction: f64,
    /// DRL improvement over the industry-practice heuristic (NVP+VOL).
    pub drl_vs_industry_heuristic: f64,
    /// Average-profit increase from full average-age blending vs. no blending.
    pub value_of_average_age_blending: f64,
    /// Average-profit increase from minimum-age blending vs. no blending.
    pub value_of_minimum_age_blending: f64,
    /// Generic-instance learned-policy gap to the perfect-information upper bound.
    pub generic_gap_to_upper_bound: f64,
    /// Port-wine case-study gap to the perfect-information upper bound.
    pub port_wine_gap_to_upper_bound: f64,
    pub anchors_repo_assertion: bool,
    pub notes: &'static str,
}

pub const PAHR_GRUNOW_2025_PERFORMANCE: PublishedPerformanceFigures =
    PublishedPerformanceFigures {
        source: PAHR_GRUNOW_2025_REFERENCE.source,
        drl_vs_rlp_gap_reduction: 0.169,
        drl_vs_industry_heuristic: 0.277,
        value_of_average_age_blending: 0.181,
        value_of_minimum_age_blending: 0.086,
        generic_gap_to_upper_bound: 0.035,
        port_wine_gap_to_upper_bound: 0.028,
        anchors_repo_assertion: false,
        notes: "Reported in the paper text. Recorded for provenance only; the reduced Rust env (purchase-only action, fixed price/decay, discounted-cost objective) cannot reproduce these numbers.",
    };

pub const PRIMARY_DEMAND_MODELS: &[DemandModel] = &[
    DemandModel {
        kind: DemandDistributionKind::Poisson,
        param1: 10.0,
    },
    DemandModel {
        kind: DemandDistributionKind::Poisson,
        param1: 6.0,
    },
];

pub const PRIMARY_REFERENCE_INSTANCE: AmelioratingInventoryReferenceInstance =
    AmelioratingInventoryReferenceInstance {
        name: "pahr_grunow2025_default_spirits_shape",
        source: PAHR_GRUNOW_2025_REFERENCE.source,
        url: PAHR_GRUNOW_2025_REFERENCE.url,
        num_ages: 5,
        target_ages: &[1, 3],
        product_prices: &[300.0, 500.0],
        demand_models: PRIMARY_DEMAND_MODELS,
        age_retention: &[0.98, 0.98, 0.98, 0.98, 0.98],
        purchase_cost_per_unit: 250.0,
        holding_cost_per_unit: 25.0,
        decay_salvage_values: &[50.0, 60.0, 70.0, 80.0, 90.0],
        benchmark_newsvendor_total_target: 24,
        benchmark_two_dimensional_total_target: 24,
        benchmark_two_dimensional_young_target: 8,
        benchmark_young_age_cutoff: 1,
        notes: "Repo-native five-age, two-product discrete reduction loosely inspired by the paper's default spirits family. It is NOT the companion default (which is ten ages, three products, target ages [2,4,6], stochastic purchase/sales prices, stochastic beta decay, evaporation, capacity 50; see PAHR_GRUNOW_2025_SPIRITS_0001_UPPER_BOUND). The numeric targets here are repo-chosen, not published. This instance is benchmark-shaped but not literature-verified.",
    };

pub const VERIFICATION_DEMAND_SCENARIOS: &[&[u32]] =
    &[&[0, 0], &[1, 0], &[0, 1], &[1, 1], &[0, 2], &[1, 2]];
pub const VERIFICATION_DEMAND_PROBABILITIES: &[f64] = &[0.10, 0.20, 0.20, 0.25, 0.10, 0.15];

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: PAHR_GRUNOW_2025_REFERENCE.source,
    url: PAHR_GRUNOW_2025_REFERENCE.url,
    literature_verified: false,
    verification_source: "repo_exact_solver_not_verified_against_literature",
    periods: 4,
    discount_factor: 0.99,
    initial_inventory_by_age: &[1, 1, 0],
    target_ages: &[1, 2],
    product_prices: &[5.0, 9.0],
    age_retention: &[1.0, 1.0, 1.0],
    purchase_cost_per_unit: 3.0,
    holding_cost_per_unit: 0.5,
    decay_salvage_values: &[0.0, 0.0, 0.0],
    demand_scenarios: VERIFICATION_DEMAND_SCENARIOS,
    demand_probabilities: VERIFICATION_DEMAND_PROBABILITIES,
    max_purchase_quantity: 4,
    newsvendor_total_target: 3,
    two_dimensional_total_target: 4,
    two_dimensional_young_target: 2,
    young_age_cutoff: 0,
    notes: "Repo-native exact verifier on a reduced average-age blending instance with three age classes and two products. This is an internal implementation check, not a published benchmark row.",
};
