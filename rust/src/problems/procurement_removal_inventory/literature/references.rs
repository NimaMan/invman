#![allow(dead_code)]

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
pub struct ProcurementRemovalReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub periods: usize,
    pub demand_distribution_kind: &'static str,
    pub demand_mean: f64,
    pub initial_inventory_level: usize,
    pub initial_returnable_inventory: usize,
    pub returnable_purchase_cap: usize,
    pub purchase_cost_per_unit: f64,
    pub return_value_per_unit: f64,
    pub liquidation_value_per_unit: f64,
    pub holding_cost_per_unit: f64,
    pub shortage_cost_per_unit: f64,
    pub max_purchase_quantity: usize,
    pub max_removal_quantity: usize,
    pub benchmark_order_up_to: usize,
    pub benchmark_remove_down_to: usize,
    pub benchmark_returnable_buffer: usize,
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
    pub initial_inventory_level: usize,
    pub initial_returnable_inventory: usize,
    pub returnable_purchase_cap: usize,
    pub purchase_cost_per_unit: f64,
    pub return_value_per_unit: f64,
    pub liquidation_value_per_unit: f64,
    pub holding_cost_per_unit: f64,
    pub shortage_cost_per_unit: f64,
    pub demand_support: &'static [u32],
    pub demand_probabilities: &'static [f64],
    pub max_purchase_quantity: usize,
    pub max_removal_quantity: usize,
    pub interval_stock_order_up_to: usize,
    pub interval_stock_remove_down_to: usize,
    pub returnability_buffer_order_up_to: usize,
    pub returnability_buffer_remove_down_to: usize,
    pub returnability_buffer: usize,
    pub notes: &'static str,
}

pub const MAGGIAR_2017_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Maggiar and Sadighian (2017), Joint Inventory and Revenue Management with Removal Decisions",
    url: "https://assets.amazon.science/7b/48/bc8c1c21450b9dac198e1f4ed13a/joint-inventory-and-revenue-management-with-removal-decisions.pdf",
    benchmark_policies: &["optimal_interval_stock", "order_up_to_remove_down_to", "pricing_and_markdown_variants"],
    reported_numbers_available: false,
    numbers_anchor_repo_assertions: false,
    notes: "The paper studies a richer finite-horizon joint replenishment, pricing, and removal revenue-management model under return and liquidation credits. It gives structural policy results and graphical numerical examples, but no exact benchmark row for this simplified repo control-only package.",
};

pub const MAGGIAR_2025_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Maggiar et al. (2025), Structure-Informed Deep Reinforcement Learning for Inventory Management",
    url: "https://openreview.net/pdf?id=asKybwTGUt",
    benchmark_policies: &["directbackprop_drl", "structure_informed_policy_network", "interval_stock"],
    reported_numbers_available: false,
    numbers_anchor_repo_assertions: false,
    notes: "The DRL paper explicitly includes joint procurement-removal decisions as one of the benchmark families and reports that the learned policy recovers interval-stock structure, but it does not expose a public exact cost row for this repo package.",
};

pub const PRIMARY_REFERENCE_INSTANCE: ProcurementRemovalReferenceInstance =
    ProcurementRemovalReferenceInstance {
        name: "maggiar2017_style_fixed_returnability",
        source: MAGGIAR_2017_REFERENCE.source,
        url: MAGGIAR_2017_REFERENCE.url,
        periods: 16,
        demand_distribution_kind: "poisson",
        demand_mean: 4.0,
        initial_inventory_level: 5,
        initial_returnable_inventory: 3,
        returnable_purchase_cap: 2,
        purchase_cost_per_unit: 6.0,
        return_value_per_unit: 4.0,
        liquidation_value_per_unit: 1.0,
        holding_cost_per_unit: 0.5,
        shortage_cost_per_unit: 9.0,
        max_purchase_quantity: 6,
        max_removal_quantity: 6,
        benchmark_order_up_to: 6,
        benchmark_remove_down_to: 8,
        benchmark_returnable_buffer: 2,
        notes: "Canonical repo interpretation of procurement-removal inventory: a single-item finite-horizon system with a fixed per-period cap on returnable purchases, explicit return and liquidation credits, and shortage penalties. This strips away pricing while keeping the procurement-removal structure highlighted by the literature; it is therefore a repo-native instance, not a literature-verified benchmark row.",
    };

/// Repo-native benchmark instance on which the REMOVAL channel actually binds.
///
/// Why this exists: the `PRIMARY_REFERENCE_INSTANCE` runs a Poisson demand of mean 4 over 16
/// periods starting from only 5 units. Demand drains inventory faster than it can accumulate, so
/// the system almost never overstocks and the `remove_down_to` threshold is essentially never
/// triggered — the interval-stock policy degenerates to a pure order-up-to policy there (verified
/// empirically: the best constant interval-stock is `(order_up_to=6, remove_down_to=6)`, i.e. the
/// removal level collapses onto the order level). That makes the primary instance a poor benchmark
/// for the distinguishing feature of THIS problem (procurement vs removal).
///
/// This instance starts overstocked (high initial inventory, lower demand, nontrivial holding
/// cost), so carrying excess hurts and returning/liquidating becomes worthwhile. Empirically the
/// best constant interval-stock is `(order_up_to=4, remove_down_to=9)` and it beats both the
/// never-remove and aggressive-remove extremes, confirming the removal lever is active.
///
/// It is NOT a literature-verified row; like the primary instance it is a repo-native instance of
/// the control-only procurement-removal slice. It is used by
/// `scripts/procurement_removal_inventory/benchmark_procurement_removal.py` (which passes the
/// fields directly to the simulator, so no binding rebuild is required). It is recorded here as the
/// source of truth so a future rebuild can expose it through a binding.
pub const REMOVAL_ACTIVE_REFERENCE_INSTANCE: ProcurementRemovalReferenceInstance =
    ProcurementRemovalReferenceInstance {
        name: "removal_active_returnability",
        source: "repo_native_removal_active_instance",
        url: MAGGIAR_2017_REFERENCE.url,
        periods: 16,
        demand_distribution_kind: "poisson",
        demand_mean: 3.0,
        initial_inventory_level: 12,
        initial_returnable_inventory: 8,
        returnable_purchase_cap: 2,
        purchase_cost_per_unit: 6.0,
        return_value_per_unit: 4.0,
        liquidation_value_per_unit: 1.0,
        holding_cost_per_unit: 1.0,
        shortage_cost_per_unit: 9.0,
        max_purchase_quantity: 6,
        max_removal_quantity: 8,
        benchmark_order_up_to: 4,
        benchmark_remove_down_to: 9,
        benchmark_returnable_buffer: 0,
        notes: "Repo-native procurement-removal instance with an active removal channel: high initial inventory and lower demand make overstock occur, so the remove-down-to threshold binds and the procurement-versus-removal tradeoff is observable. Best constant interval-stock is (4, 9). Not a literature-verified benchmark row.",
    };

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: MAGGIAR_2017_REFERENCE.source,
    url: MAGGIAR_2017_REFERENCE.url,
    literature_verified: false,
    verification_source: "repo_exact_solver_not_verified_against_literature",
    periods: 5,
    discount_factor: 0.99,
    initial_inventory_level: 2,
    initial_returnable_inventory: 1,
    returnable_purchase_cap: 1,
    purchase_cost_per_unit: 5.0,
    return_value_per_unit: 3.0,
    liquidation_value_per_unit: 1.0,
    holding_cost_per_unit: 0.5,
    shortage_cost_per_unit: 7.0,
    demand_support: &[0, 1, 2, 3],
    demand_probabilities: &[0.2, 0.3, 0.3, 0.2],
    max_purchase_quantity: 4,
    max_removal_quantity: 4,
    interval_stock_order_up_to: 3,
    interval_stock_remove_down_to: 4,
    returnability_buffer_order_up_to: 3,
    returnability_buffer_remove_down_to: 4,
    returnability_buffer: 1,
    notes: "Repo-native exact verifier on a reduced procurement-removal instance with a small discrete demand support. This preserves the returnable-quota state and the order/remove action pair while keeping the finite-horizon DP small enough for exact regression tests.",
};
