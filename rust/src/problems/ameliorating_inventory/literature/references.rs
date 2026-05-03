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
    source: "Pahr and Grunow (2025), Production and Operations Management",
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
        notes: "Repo-native five-age, two-product discrete reduction inspired by the paper's default spirits setting. This instance is benchmark-shaped but not literature-verified.",
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
