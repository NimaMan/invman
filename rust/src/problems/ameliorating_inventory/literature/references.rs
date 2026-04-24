#![allow(dead_code)]

use crate::problems::ameliorating_inventory::demand::{DemandDistributionKind, DemandModel};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
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
pub struct WorkedTransitionReference {
    pub source: &'static str,
    pub url: &'static str,
    pub initial_inventory_by_age: &'static [usize],
    pub target_ages: &'static [usize],
    pub product_prices: &'static [f64],
    pub age_retention: &'static [f64],
    pub purchase_cost_per_unit: f64,
    pub holding_cost_per_unit: f64,
    pub decay_salvage_values: &'static [f64],
    pub purchase_quantity: usize,
    pub realized_demands: &'static [usize],
    pub expected_shipments_by_product_age: &'static [&'static [usize]],
    pub expected_shipped_by_product: &'static [usize],
    pub expected_lost_sales_by_product: &'static [usize],
    pub expected_next_inventory_by_age: &'static [usize],
    pub expected_decayed_units_by_age: &'static [usize],
    pub expected_revenue: f64,
    pub expected_purchase_cost: f64,
    pub expected_holding_cost: f64,
    pub expected_salvage_credit: f64,
    pub expected_period_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
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

pub const WORKED_TRANSITION_REFERENCE: WorkedTransitionReference = WorkedTransitionReference {
    source: PAHR_GRUNOW_2025_REFERENCE.source,
    url: PAHR_GRUNOW_2025_REFERENCE.url,
    initial_inventory_by_age: &[1, 2, 1],
    target_ages: &[1, 2],
    product_prices: &[5.0, 9.0],
    age_retention: &[1.0, 1.0, 1.0],
    purchase_cost_per_unit: 3.0,
    holding_cost_per_unit: 0.5,
    decay_salvage_values: &[0.0, 0.0, 0.0],
    purchase_quantity: 1,
    realized_demands: &[1, 1],
    expected_shipments_by_product_age: &[&[0, 1, 0], &[0, 0, 1]],
    expected_shipped_by_product: &[1, 1],
    expected_lost_sales_by_product: &[0, 0],
    expected_next_inventory_by_age: &[0, 2, 1],
    expected_decayed_units_by_age: &[0, 0, 0],
    expected_revenue: 14.0,
    expected_purchase_cost: 3.0,
    expected_holding_cost: 1.5,
    expected_salvage_credit: 0.0,
    expected_period_cost: -9.5,
};

pub const VERIFICATION_DEMAND_SCENARIOS: &[&[u32]] =
    &[&[0, 0], &[1, 0], &[0, 1], &[1, 1], &[0, 2], &[1, 2]];
pub const VERIFICATION_DEMAND_PROBABILITIES: &[f64] = &[0.10, 0.20, 0.20, 0.25, 0.10, 0.15];

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: PAHR_GRUNOW_2025_REFERENCE.source,
    url: PAHR_GRUNOW_2025_REFERENCE.url,
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
