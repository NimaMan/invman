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
    source: "Alvaro Maggiar and Ali Sadighian (Amazon.com), Joint Inventory and Revenue Management with Removal Decisions, working paper, August 14, 2017 (SSRN 3018984)",
    url: "https://assets.amazon.science/7b/48/bc8c1c21450b9dac198e1f4ed13a/joint-inventory-and-revenue-management-with-removal-decisions.pdf",
    benchmark_policies: &["optimal_interval_stock", "order_up_to_remove_down_to", "pricing_and_markdown_variants"],
    reported_numbers_available: false,
    numbers_anchor_repo_assertions: false,
    notes: "Working paper (SSRN abstract 3018984; PDF mirrored at amazon.science; also titled 'Joint Inventory and Revenue Management with Removal Decisions'). Citation independently verified against the paper PDF on 2026-05-31: Theorem 3.4 'interval-stock list-prices policy' with stock levels (x*, xbar*), Corollary 1 'never optimal to liquidate a unit that could be returned', Assumption 2(ii) c>s and 2(iii) l<s, Section 3.2 fixed returnability (per-period cap on returnable purchases), Assumption 4 terminal value VT(x,y)=s*min(x,y)+l*max(x-y,0), and additive price-dependent log-linear demand with Gamma noise (CV 1). The model studies a richer finite-horizon joint replenishment, pricing, and removal revenue-management problem; the only numerical example (Section 7, Table 1: p0=90, c=75, s=30, l=5, h+=2, k=15.5, elasticity -2; 40 periods; discount 0.9984) reports a pricing-coupled NPV surface (axis ~84000), not a standalone control-only cost row, so no exact benchmark row anchors this simplified repo package.",
};

// ----------------------------------------------------------------------------
// Faithful Maggiar & Sadighian (2017) joint pricing / inventory / removal model.
//
// This is the literature anchor for the FAITHFUL environment in
// joint_pricing_removal_env.rs / joint_pricing_removal_dp.rs, which adds the
// pricing/markdown decision the control-only legacy slice omitted. The Section 7
// "Numerical Results" example (Table 1) is the only fully specified numerical
// instance in the paper. Its headline output is a NPV SURFACE plotted at fixed
// period t = 24 whose top labelled contour is ~84000 (Figure 7, Section 7.2.1).
//
// WHAT THE PAPER ACTUALLY PRINTS (and what is therefore reproducible):
//   - Table 1 parameter set: p0=90, c=75, s=30, l=5, h+=2, k=15.5, E=-2.
//   - 40 periods, discount gamma = 0.9984 (~8%/yr), 99 demand quantiles.
//   - Log-linear additive demand d_t(p) = mu_t exp(-beta(p-p0)), with mu_t+eps
//     Gamma-distributed, mean mu_t, coefficient of variation 1.
//   - mu_t profile: constant ~50 with a peak around period 20 (Figure 6) -- but
//     the exact per-period mu_t values are given ONLY GRAPHICALLY, never as a
//     table. The fixed-returnability quota is the MEDIAN of the base-price
//     forecast demand.
//   - The NPV surface peak at t=24 is ~84000 (Figure 7, top contour label).
//
// Because mu_t is graphical, the exact 84000 cannot be reproduced to tight
// tolerance: it depends on the unspecified peak shape and the conditional t=24
// plotting window. We therefore (a) reproduce the paper's PROVEN exact
// structural properties (Lemma 3.1 monotonicity bullets) by re-running the env
// DP -- these are exact, executable claims independent of the mu_t shape -- and
// (b) characterize the NPV magnitude against ~84000 with a wide honest tolerance.
// literature_verified stays FALSE for the NPV figure; the structural properties
// are exactly reproduced.
// ----------------------------------------------------------------------------

/// Published faithful-model anchor: Table 1 parameters + the reported NPV-surface
/// peak magnitude. `reported_npv_surface_peak` is the ~84000 top contour at
/// t = 24; `npv_peak_is_exact` is false because mu_t is graphical.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaggiarSadighian2017FaithfulInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub base_price: f64,            // p0
    pub purchase_cost: f64,         // c
    pub refund_value: f64,          // s
    pub liquidation_value: f64,     // l
    pub holding_cost: f64,          // h+
    pub backorder_supplement: f64,  // k  (h- = c + k)
    pub elasticity_at_base_price: f64, // E
    pub periods: usize,             // 40
    pub discount_factor: f64,       // 0.9984
    pub coefficient_of_variation: f64, // 1
    pub num_demand_quantiles: usize, // 99
    pub baseline_mean_demand: f64,  // ~50 constant base level
    pub peak_mean_demand: f64,      // ~500 at the period-20 peak (Figure 6)
    pub peak_period: usize,         // ~20
    pub reported_npv_surface_peak: f64, // ~84000 (Figure 7, t=24)
    pub npv_peak_is_exact: bool,    // false: mu_t given only graphically
    pub notes: &'static str,
}

pub const MAGGIAR_SADIGHIAN_2017_FAITHFUL_INSTANCE: MaggiarSadighian2017FaithfulInstance =
    MaggiarSadighian2017FaithfulInstance {
        name: "maggiar_sadighian_2017_table1_full_returnability",
        source: MAGGIAR_2017_REFERENCE.source,
        url: MAGGIAR_2017_REFERENCE.url,
        base_price: 90.0,
        purchase_cost: 75.0,
        refund_value: 30.0,
        liquidation_value: 5.0,
        holding_cost: 2.0,
        backorder_supplement: 15.5,
        elasticity_at_base_price: -2.0,
        periods: 40,
        discount_factor: 0.9984,
        coefficient_of_variation: 1.0,
        num_demand_quantiles: 99,
        baseline_mean_demand: 50.0,
        peak_mean_demand: 500.0,
        peak_period: 20,
        reported_npv_surface_peak: 84000.0,
        npv_peak_is_exact: false,
        notes: "Section 7 (Table 1, Figures 6-7) full-returnability example. p0=90, c=75, s=30, l=5, h+=2, k=15.5, E=-2, 40 periods, gamma=0.9984, 99 demand quantiles, additive log-linear price-dependent Gamma demand with CV=1. Headline output is a NPV surface at t=24 with top contour ~84000. The mu_t profile is given only graphically (Figure 6: ~50 baseline, peak ~500 near period 20), so the exact 84000 is NOT reproducible to tight tolerance; the env reproduces the paper's exact PROVEN monotonicity properties (Lemma 3.1) instead, and brackets the NPV magnitude.",
    };

/// Small faithful verification instance: the SAME faithful dynamics (pricing +
/// removal + backlogging with h-=c+k, log-linear price-dependent Gamma demand
/// CV=1), shrunk to a coarse grid so the finite-horizon DP solves exactly and
/// quickly inside `cargo test`, while still exhibiting the paper's proven
/// structural properties. Parameters keep the c > s > l ordering and the
/// elasticity sign, scaled down from Table 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaithfulVerificationInstance {
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub verification_source: &'static str,
    pub base_price: f64,
    pub purchase_cost: f64,
    pub refund_value: f64,
    pub liquidation_value: f64,
    pub holding_cost: f64,
    pub backorder_supplement: f64,
    pub elasticity_at_base_price: f64,
    pub periods: usize,
    pub discount_factor: f64,
    pub coefficient_of_variation: f64,
    pub baseline_mean_demand: f64,
    pub num_demand_quantiles: usize,
    pub num_price_points: usize,
    pub max_demand_multiple: f64,
    pub returnable_purchase_cap: i64,
    pub max_inventory_level: i64,
    pub max_purchase_quantity: i64,
    pub notes: &'static str,
}

pub const FAITHFUL_VERIFICATION_INSTANCE: FaithfulVerificationInstance =
    FaithfulVerificationInstance {
        source: MAGGIAR_2017_REFERENCE.source,
        url: MAGGIAR_2017_REFERENCE.url,
        literature_verified: false,
        verification_source: "faithful_dp_reproduces_paper_proven_structural_properties",
        base_price: 90.0,
        purchase_cost: 75.0,
        refund_value: 30.0,
        liquidation_value: 5.0,
        holding_cost: 2.0,
        backorder_supplement: 15.5,
        elasticity_at_base_price: -2.0,
        periods: 6,
        discount_factor: 0.9984,
        coefficient_of_variation: 1.0,
        baseline_mean_demand: 4.0,
        num_demand_quantiles: 25,
        num_price_points: 9,
        max_demand_multiple: 2.0,
        returnable_purchase_cap: 4,
        max_inventory_level: 18,
        max_purchase_quantity: 12,
        notes: "Faithful-model regression instance: same pricing + removal + backlogging dynamics (h-=c+k) and log-linear additive Gamma demand (CV=1) as Table 1, with the c > s > l ordering and E=-2 elasticity preserved, shrunk to a coarse (x,y) grid (mean demand 4, x_max 18) so the finite-horizon DP solves exactly inside cargo test. Used to reproduce the paper's PROVEN Lemma 3.1 monotonicity properties of the optimal policy, not a published cost row.",
    };

pub const MAGGIAR_2025_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Alvaro Maggiar, Sohrab Andaz, Akhil Bagaria, Carson Eisenach, Dean Foster, Omer Gottesman, Dominique Perrault-Joncas (2025), Structure-Informed Deep Reinforcement Learning for Inventory Management, NeurIPS 2025 (arXiv:2507.22040)",
    url: "https://openreview.net/pdf?id=asKybwTGUt",
    benchmark_policies: &["directbackprop_drl", "structure_informed_policy_network", "interval_stock"],
    reported_numbers_available: false,
    numbers_anchor_repo_assertions: false,
    notes: "NeurIPS 2025; arXiv:2507.22040; OpenReview id asKybwTGUt. Citation independently verified against the arXiv HTML on 2026-05-31: Section 4.6 'Multi-Period Inventory Management with Returns' cites Maggiar & Sadighian (2017) and describes the optimal interval-stock policy (order-up-to / remove-down-to). Section 4.6.4 explicitly states it does NOT report the measured average expected reward for the returns family (because its steady state matches the basic multi-period problem) and 'focus[es] exclusively on observing the structure of the learned policy' (Figure 23) -- so the paper exposes NO public exact procurement-removal cost row for this repo package.",
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
