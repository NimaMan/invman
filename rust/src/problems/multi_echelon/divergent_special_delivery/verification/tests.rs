use crate::problems::multi_echelon::verification::{
    gijs_relative_reference_instances, gijs_relative_verification_summary,
    published_constant_base_stock_reference_instances, van_roy_reproduction_summary,
    DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED, GIJS_RELATIVE_VERIFICATION_METRIC,
    VAN_ROY_REPRODUCTION_METRIC,
};

// ---------------------------------------------------------------------------
// Honest-status drift guard for the Van Roy / Gijs divergent special-delivery
// constant base-stock benchmark rows.
//
// Algorithmic description
// -----------------------
// This test re-runs THIS family's executable environment + constant base-stock
// heuristic at the published Van Roy (1997) order-up-to levels for both complex
// case studies (the same two settings Gijsbrechts et al. 2022 reuse), under BOTH
// transition conventions the crate supports:
//
//   * `van_roy_1997`  -- post-shipment warehouse order (Van Roy 1997 heuristic:
//                        warehouse order-up-to is applied AFTER store orders are
//                        deducted, full-length report Section 4, p.10-11).
//   * `gijs_2022`     -- pre-shipment warehouse order (Gijsbrechts et al. 2022
//                        Eq. (2) MDP: the warehouse order raises its inventory
//                        position to its base-stock level FIRST, then "After the
//                        warehouse has ordered, each retailer places its order",
//                        MSOM 24(3) p.1365-1366). This is the faithful policy-
//                        search target.
//
// For each (setting, mode) it measures the simulated average cost and compares
// it to the absolute number PRINTED in Van Roy et al. (1997) full-length report
// (1302 at (330,23) for case study 1, Figure 6 / p.886,935; 1449 at (460,22) for
// case study 2, p.1147-1148,1166-1167). It then ASSERTS the honest finding:
//
//   1. The faithful `gijs_2022` MDP does NOT reproduce the published absolute
//      anchor: it lands roughly 19%-21% BELOW it (structurally different model;
//      Gijs itself prints NO absolute cost, only ~9%/~12% relative savings).
//   2. The `van_roy_1997` reproduction mode approaches but does NOT match the
//      published numbers within the repo's 1% literature-verification tolerance
//      (~ -1.3% for setting 1, ~ -7% for setting 2).
//
// Therefore neither executable mode reproduces a paper-printed absolute number
// within tolerance, so `literature_verified` MUST stay `false` for these rows.
// This test exists so that a future change which silently makes the env "match"
// 1302/1449 (and would tempt someone to flip the flag) trips this guard, and so
// that the structural pre-/post-shipment separation cannot regress unnoticed.
//
// Runtime: horizon 20_000 x 4 replications x 4 (setting,mode) pairs (~7s). The
// measured means are within ~0.5 of the long-run (100_000 x 20) values, so the
// honest bands below are not protocol-sensitive.
// ---------------------------------------------------------------------------

const PUBLISHED_SETTING1_CONSTANT_BASE_STOCK_COST: f64 = 1302.0;
const PUBLISHED_SETTING2_CONSTANT_BASE_STOCK_COST: f64 = 1449.0;

struct DivergentBenchmarkCase {
    name: &'static str,
    warehouse_lead_time: usize,
    retailer_lead_time: usize,
    demand_mean: f64,
    demand_std: f64,
    warehouse_level: usize,
    retailer_level: usize,
    published_constant_base_stock_cost: f64,
}

fn simulate_constant_base_stock(
    case: &DivergentBenchmarkCase,
    mode: crate::problems::multi_echelon::env::InventoryDynamicsMode,
) -> f64 {
    use crate::problems::multi_echelon::env::AllocationMode;
    use crate::problems::multi_echelon::heuristics::{
        evaluate_stationary_policy, HeuristicSimulationConfig, StationaryPolicyKind,
    };
    use crate::problems::multi_echelon::rollout::{RolloutObjective, SymmetricDemandDistribution};

    let config = HeuristicSimulationConfig {
        warehouse_lead_time: case.warehouse_lead_time,
        retailer_lead_time: case.retailer_lead_time,
        num_retailers: 10,
        warehouse_holding_cost: 3.0,
        retailer_holding_cost: 3.0,
        warehouse_expedited_cost: 0.0,
        warehouse_lost_sale_cost: 60.0,
        expedited_service_prob: 0.8,
        warehouse_capacity: 100,
        warehouse_inventory_cap: 1000,
        retailer_inventory_cap: 100,
        inventory_dynamics_mode: mode,
        demand_distribution: SymmetricDemandDistribution::NormalRoundedClipped,
        demand_mean: case.demand_mean,
        demand_std: case.demand_std,
        horizon: 20_000,
        warm_up_periods_ratio: 0.0,
        discount_factor: 1.0,
        objective: RolloutObjective::AverageCostAfterWarmup,
    };
    let (mean, _std) = evaluate_stationary_policy(
        &config,
        &[case.warehouse_level],
        &[case.retailer_level],
        case.warehouse_level,
        case.retailer_level,
        StationaryPolicyKind::RegularBaseStock,
        AllocationMode::MinShortage,
        4,
        123,
    )
    .expect("constant base-stock evaluation must succeed");
    mean
}

#[test]
fn neither_dynamics_mode_reproduces_published_absolute_cost_within_tolerance() {
    use crate::problems::multi_echelon::env::InventoryDynamicsMode;

    let cases = [
        DivergentBenchmarkCase {
            name: "gijsbrechts2022_setting1",
            warehouse_lead_time: 2,
            retailer_lead_time: 2,
            demand_mean: 5.0,
            demand_std: 14.0,
            warehouse_level: 330,
            retailer_level: 23,
            published_constant_base_stock_cost: PUBLISHED_SETTING1_CONSTANT_BASE_STOCK_COST,
        },
        DivergentBenchmarkCase {
            name: "gijsbrechts2022_setting2",
            warehouse_lead_time: 5,
            retailer_lead_time: 3,
            demand_mean: 0.0,
            demand_std: 20.0,
            warehouse_level: 460,
            retailer_level: 22,
            published_constant_base_stock_cost: PUBLISHED_SETTING2_CONSTANT_BASE_STOCK_COST,
        },
    ];

    // The repo's literature-verification tolerance for these rows is 1%
    // (see verification::PUBLISHED_CONSTANT_BASE_STOCK_RELATIVE_TOLERANCE_PCT).
    const LITERATURE_TOLERANCE_PCT: f64 = 1.0;

    for case in &cases {
        let van_roy_cost = simulate_constant_base_stock(case, InventoryDynamicsMode::VanRoy1997);
        let gijs_cost = simulate_constant_base_stock(case, InventoryDynamicsMode::Gijs2022);

        let van_roy_gap_pct =
            100.0 * (van_roy_cost - case.published_constant_base_stock_cost)
                / case.published_constant_base_stock_cost;
        let gijs_gap_pct = 100.0 * (gijs_cost - case.published_constant_base_stock_cost)
            / case.published_constant_base_stock_cost;

        // (1) Faithful gijs_2022 MDP is a structurally different model: the
        //     pre-shipment warehouse order drives the cost well below the
        //     published Van Roy heuristic cost (>10% under). It does NOT and is
        //     not expected to reproduce any published absolute anchor.
        assert!(
            gijs_gap_pct < -10.0,
            "faithful gijs_2022 MDP unexpectedly close to published cost for {}: gap {:.2}% (cost {:.2} vs {})",
            case.name,
            gijs_gap_pct,
            gijs_cost,
            case.published_constant_base_stock_cost
        );
        // Pin the structural separation so a pre-/post-shipment regression trips.
        assert!(
            gijs_cost < van_roy_cost - 50.0,
            "gijs_2022 cost ({:.2}) should sit well below van_roy_1997 cost ({:.2}) for {}",
            gijs_cost,
            van_roy_cost,
            case.name
        );

        // (2) The van_roy_1997 reproduction mode approaches the published number
        //     but does NOT match it within the repo's 1% tolerance, so it does
        //     not constitute an executable literature reproduction either.
        assert!(
            van_roy_gap_pct.abs() > LITERATURE_TOLERANCE_PCT,
            "van_roy_1997 mode reproduced {} within {}% (gap {:.2}%, cost {:.2}); if this becomes true, re-examine literature_verified flag",
            case.name,
            LITERATURE_TOLERANCE_PCT,
            van_roy_gap_pct,
            van_roy_cost
        );

        // Drift bands around the observed values (long-run stable to ~0.5):
        //   setting1: van_roy ~1285 (-1.3%), gijs ~1052 (-19.2%)
        //   setting2: van_roy ~1345 (-7.2%), gijs ~1139 (-21.4%)
        assert!(
            van_roy_gap_pct > -12.0 && van_roy_gap_pct < 0.0,
            "van_roy_1997 gap drifted outside expected band for {}: {:.2}%",
            case.name,
            van_roy_gap_pct
        );
        assert!(
            gijs_gap_pct > -26.0 && gijs_gap_pct < -14.0,
            "gijs_2022 gap drifted outside expected band for {}: {:.2}%",
            case.name,
            gijs_gap_pct
        );
    }
}

#[test]
fn gijs_relative_reference_rows_are_present_and_frozen() {
    let references = gijs_relative_reference_instances();
    assert_eq!(references.len(), 2);
    assert_eq!(references[0].name, "van_roy1997_case_study1");
    assert_eq!(references[1].name, "van_roy1997_case_study2");
    assert_eq!(
        references[0].published_constant_base_stock_levels,
        &[330, 23]
    );
    assert_eq!(
        references[1].published_constant_base_stock_levels,
        &[460, 22]
    );
    assert_eq!(references[0].published_a3c_savings_pct, Some(8.95));
    assert_eq!(references[1].published_a3c_savings_pct, Some(12.09));
}

#[test]
fn gijs_relative_verification_summary_computes_carried_implied_costs() {
    let summary = gijs_relative_verification_summary(2, DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED)
        .expect("summary must build");

    assert_eq!(summary.rows.len(), 2);
    assert_eq!(summary.source.contains("Gijsbrechts"), true);
    assert!(summary.literature_reference_present);
    assert!(!summary.implementation_literature_verified);
    assert_eq!(
        summary.literature_verification_metric,
        GIJS_RELATIVE_VERIFICATION_METRIC
    );
    assert_eq!(summary.literature_verification_target_count, 2);
    assert!((summary.rows[0].published_a3c_implied_mean_cost - 1185.471).abs() < 1e-9);
    assert!((summary.rows[1].published_a3c_implied_mean_cost - 1273.8159).abs() < 1e-9);
    assert!((summary.rows[0].published_van_roy_implied_mean_cost - 1171.8).abs() < 1e-9);
    assert!((summary.rows[1].published_van_roy_implied_mean_cost - 1304.1).abs() < 1e-9);
}

#[test]
fn gijs_relative_verification_summary_is_honest_about_current_status() {
    let summary = gijs_relative_verification_summary(2, DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED)
        .expect("summary must build");

    assert!(summary
        .rows
        .iter()
        .all(|row| row.repo_published_constant_base_stock_mean_cost.is_finite()));
    assert!(summary.mean_repo_gap_vs_published_constant_cost.is_finite());
    assert!(summary.all_published_constant_base_stock_rows_reproduced_within_tolerance);
    assert!(!summary.repo_generates_published_relative_rows);
    assert!(!summary.can_mark_literature_verified);
}

#[test]
fn van_roy_reproduction_summary_checks_absolute_published_rows() {
    let references = published_constant_base_stock_reference_instances();
    assert_eq!(references.len(), 3);
    assert_eq!(
        references[0].published_constant_base_stock_levels,
        &[10, 16]
    );
    assert_eq!(
        references[1].published_constant_base_stock_levels,
        &[330, 23]
    );
    assert_eq!(
        references[2].published_constant_base_stock_levels,
        &[460, 22]
    );

    let summary = van_roy_reproduction_summary(2, DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED)
        .expect("summary must build");

    assert_eq!(summary.rows.len(), 3);
    assert!(summary.literature_reference_present);
    assert_eq!(
        summary.literature_verification_metric,
        VAN_ROY_REPRODUCTION_METRIC
    );
    assert!(!summary.implementation_literature_verified);
    assert!(summary.all_published_constant_base_stock_rows_reproduced_within_tolerance);
    assert_eq!(summary.rows[1].instance_name, "van_roy1997_case_study1");
    assert!(
        summary.rows[1]
            .repo_gap_vs_published_constant_cost_pct
            .abs()
            > 1.0
    );
}
