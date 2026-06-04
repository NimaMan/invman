// =============================================================================
// nonstationary_lot_sizing / tests/verification.rs
//
// WHAT THIS VERIFIER ACTUALLY ASSERTS (and what it does NOT)
//   This file is the executable correctness anchor for the single-item
//   non-stationary rolling-forecast lot-sizing family. Per the repo rule, a
//   family is "literature-verified" ONLY when an in-crate test re-runs the
//   env/solver and reproduces a number PRINTED IN A PAPER within a stated
//   tolerance. NONE of these tests meet that bar, and they do not claim to:
//
//     * `policy_state_layout_*` and `worked_example_transition_is_internally_consistent`
//       are INTERNAL mechanics / self-consistency checks of `build_policy_state`
//       and `step_state`. The Section-4 worked transition reward -130 is
//       recomputed from our own arithmetic; the EJOR article full text was not
//       accessible, so we make NO claim the article prints -130.
//
//     * `simple_s_s_matches_author_testbed_csv_row_*` and
//       `rolling_dp_matches_author_testbed_csv_row_*` re-run our env/solver and
//       reproduce a row of the author's PUBLIC COMPANION-CODE testbed CSVs
//       (HenriDeh/DRL_MMULS, branch single-item). That is a
//       REFERENCE-IMPLEMENTATION match, NOT a paper-printed table value. The
//       carried rows come from a testbed grid that differs from the article's
//       reported experiment grid (see references.rs header).
//
//   `honest_status_flags_are_false` asserts the literature_verified flags stay
//   false so the status cannot silently drift to an overclaim.
// =============================================================================

use crate::problems::nonstationary_lot_sizing::demand::DemandDistributionKind;
use crate::problems::nonstationary_lot_sizing::env::{
    build_policy_state, initialize_state, step_state, NonstationaryLotSizingState,
};
use crate::problems::nonstationary_lot_sizing::heuristics::{
    rolling_dp_s_s_levels, rolling_dp_s_s_sequence, simple_s_s_levels,
    simulate_periodic_s_s_policy, simulate_policy,
};
use crate::problems::nonstationary_lot_sizing::references::{
    build_forecast_path, get_primary_reference_instance, get_reference_instance,
    list_forecast_definitions, list_reference_instances, DEHAYBE_2024_REFERENCE,
    DRL_MMULS_SINGLE_ITEM_REFERENCE, ROLLING_DP_DISCOUNT_FACTOR,
    ROLLING_DP_STATIONARY_TAIL_PERIODS, VERIFICATION_PROBLEM_INSTANCE, WORKED_EXAMPLE_REFERENCE,
};

const PRIMARY_REFERENCE_REPO_DERIVED_ROLLING_DP_LEVELS: (i32, i32) = (28, 42);

#[test]
fn reference_set_has_expected_shape() {
    let primary = get_primary_reference_instance();
    let forecasts = list_forecast_definitions();
    let instances = list_reference_instances();
    let seasonal = build_forecast_path(4, 136).expect("forecast id 4 must exist");

    assert_eq!(forecasts.len(), 8);
    assert_eq!(instances.len(), 8);
    assert_eq!(primary.name, "dehaybe2024_lostsales_lt2_b5_k10_constant_10");
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.reference_instance_name,
        primary.name
    );
    assert_eq!(DEHAYBE_2024_REFERENCE.benchmark_policies.len(), 3);
    assert_eq!(DRL_MMULS_SINGLE_ITEM_REFERENCE.benchmark_policies.len(), 3);
    assert_eq!(seasonal.len(), 136);
    assert!((seasonal[0] - 10.30189248711143).abs() < 1e-12);
    assert_eq!(
        primary
            .published_simple_benchmark
            .expect("primary simple benchmark must exist")
            .demand_kind,
        DemandDistributionKind::CvNormal
    );
    assert_eq!(
        primary
            .published_rolling_dp_benchmark
            .expect("primary rolling benchmark must exist")
            .demand_kind,
        DemandDistributionKind::Poisson
    );
}

#[test]
fn policy_state_layout_matches_section_4_1() {
    let state = NonstationaryLotSizingState {
        forecast_window: vec![5.0, 10.0, 15.0, 20.0],
        net_inventory: -3.0,
        pipeline_orders: vec![7.0, 9.0],
    };
    let policy_state = build_policy_state(&state);

    assert_eq!(policy_state, vec![0.4, 0.8, 1.2, 1.6, -0.24, 0.56, 0.72]);
}

/// Internal mechanics / self-consistency check of `step_state` on the Section-4
/// illustrative transition (h=1, b=10, K=100, LT=1, backorders). The reward -130
/// is recomputed from our own cost arithmetic. This is NOT literature
/// verification: the EJOR full text was not accessible, so we do not claim the
/// article prints -130 (`WORKED_EXAMPLE_REFERENCE.literature_verified == false`).
#[test]
fn worked_example_transition_is_internally_consistent() {
    let worked = WORKED_EXAMPLE_REFERENCE;
    assert!(
        !worked.literature_verified,
        "worked transition must stay flagged as a mechanics check, not a paper number"
    );
    let state = NonstationaryLotSizingState {
        forecast_window: worked.initial_forecast_window.to_vec(),
        net_inventory: worked.initial_net_inventory,
        pipeline_orders: worked.initial_pipeline.to_vec(),
    };
    let outcome = step_state(
        &state,
        worked.action,
        worked.realized_demand,
        worked.next_forecast_mean,
        worked.holding_cost,
        worked.shortage_cost,
        worked.procurement_cost,
        worked.fixed_order_cost,
        worked.lost_sales,
    )
    .expect("worked example transition must be valid");

    assert_eq!(
        outcome.next_state.forecast_window,
        worked.expected_next_forecast_window.to_vec()
    );
    assert_eq!(
        outcome.next_state.net_inventory,
        worked.expected_next_net_inventory
    );
    assert_eq!(
        outcome.next_state.pipeline_orders,
        worked.expected_next_pipeline.to_vec()
    );
    assert_eq!(outcome.reward, worked.expected_reward);
    assert_eq!(outcome.period_cost, 130.0);
}

#[test]
fn simple_s_s_levels_follow_the_literature_formula() {
    let forecast = build_forecast_path(2, 136).expect("constant forecast must exist");
    let window = &forecast[..32];
    let (s, s_up_to) = simple_s_s_levels(
        window,
        2,
        1.0,
        5.0,
        10.0,
        0.2,
        DemandDistributionKind::CvNormal,
    );

    assert!((s - 33.351246609652).abs() < 1e-9);
    assert!((s_up_to - 47.49338223338295).abs() < 1e-9);
}

/// Re-runs the env + simple (s,S) heuristic and reproduces the author's PUBLIC
/// TESTBED CSV row (scarf_testbed_simple_lostsales.csv, LT=2/b=5/K=10 family)
/// within tolerance. This is a reference-implementation match, NOT a
/// paper-printed value (`literature_verified == false`).
#[test]
fn simple_s_s_matches_author_testbed_csv_row_within_tolerance() {
    let instance = get_primary_reference_instance();
    assert!(
        !instance.literature_verified,
        "benchmark row is author-testbed reference-impl, not a paper-printed number"
    );
    let published = instance
        .published_simple_benchmark
        .expect("verification instance must include a simple benchmark");
    let forecast = build_forecast_path(
        instance.forecast_id,
        instance.periods + instance.forecast_horizon,
    )
    .expect("reference forecast must exist");
    let initial_state = initialize_state(
        &forecast[..instance.forecast_horizon],
        instance.initial_net_inventory,
        instance.lead_time,
    )
    .expect("initial state must build");
    let summary = simulate_policy(
        "simple_s_s",
        &[],
        &initial_state,
        &forecast,
        instance.periods,
        25_000,
        1234,
        instance.holding_cost,
        instance.shortage_cost,
        instance.procurement_cost,
        instance.fixed_order_cost,
        instance.lost_sales,
        published.demand_cv,
        published.demand_kind,
    )
    .expect("policy simulation must run");

    assert!(
        (summary.mean_cost - published.mean_cost).abs() <= 35.0,
        "mean cost {} differed from published {} by more than {}",
        summary.mean_cost,
        published.mean_cost,
        35.0
    );
    assert!(
        (summary.shortage_rate - published.shortage_rate).abs() <= 0.01,
        "shortage rate {} differed from published {} by more than {}",
        summary.shortage_rate,
        published.shortage_rate,
        0.01
    );
}

/// Re-runs the env + rolling Scarf DP (s,S) solver and reproduces the author's
/// PUBLIC TESTBED CSV row (scarf_testbed_DP_lostsales.csv, LT=2/b=5/K=10
/// constant-10 instance) within tolerance. This is a reference-implementation
/// match, NOT a paper-printed value (`literature_verified == false`).
#[test]
fn rolling_dp_matches_author_testbed_csv_row_within_tolerance() {
    let verification = VERIFICATION_PROBLEM_INSTANCE;
    assert!(
        !verification.literature_verified,
        "rolling-DP row is author-testbed reference-impl, not a paper-printed number"
    );
    let instance = get_reference_instance(verification.reference_instance_name)
        .expect("verification instance must exist");
    let published = instance
        .published_rolling_dp_benchmark
        .expect("verification instance must include a rolling DP benchmark");
    let forecast = build_forecast_path(
        instance.forecast_id,
        instance.periods + instance.forecast_horizon,
    )
    .expect("reference forecast must exist");
    let initial_state = initialize_state(
        &forecast[..instance.forecast_horizon],
        instance.initial_net_inventory,
        instance.lead_time,
    )
    .expect("initial state must build");
    let first_period_levels = rolling_dp_s_s_levels(
        &forecast[..instance.forecast_horizon],
        instance.lead_time,
        instance.holding_cost,
        instance.shortage_cost,
        instance.fixed_order_cost,
        published.demand_kind,
        ROLLING_DP_DISCOUNT_FACTOR,
        ROLLING_DP_STATIONARY_TAIL_PERIODS,
    )
    .expect("rolling DP levels must solve");
    let sequence = rolling_dp_s_s_sequence(
        &forecast,
        instance.periods,
        instance.forecast_horizon,
        instance.lead_time,
        instance.holding_cost,
        instance.shortage_cost,
        instance.fixed_order_cost,
        published.demand_kind,
        ROLLING_DP_DISCOUNT_FACTOR,
        ROLLING_DP_STATIONARY_TAIL_PERIODS,
    )
    .expect("rolling DP sequence must solve");
    let summary = simulate_periodic_s_s_policy(
        &sequence,
        &initial_state,
        &forecast,
        verification.simulation_replications,
        1234,
        instance.holding_cost,
        instance.shortage_cost,
        instance.procurement_cost,
        instance.fixed_order_cost,
        instance.lost_sales,
        published.demand_cv,
        published.demand_kind,
    )
    .expect("rolling DP simulation must run");

    assert_eq!(
        (
            first_period_levels.reorder_point,
            first_period_levels.order_up_to
        ),
        PRIMARY_REFERENCE_REPO_DERIVED_ROLLING_DP_LEVELS
    );
    assert!(sequence.iter().all(|levels| *levels == first_period_levels));
    assert!(
        (summary.mean_cost - published.mean_cost).abs() <= verification.mean_cost_tolerance,
        "mean cost {} differed from published {} by more than {}",
        summary.mean_cost,
        published.mean_cost,
        verification.mean_cost_tolerance
    );
    assert!(
        (summary.shortage_rate - published.shortage_rate).abs()
            <= verification.shortage_rate_tolerance,
        "shortage rate {} differed from published {} by more than {}",
        summary.shortage_rate,
        published.shortage_rate,
        verification.shortage_rate_tolerance
    );
}

/// Drift guard for the family's HONEST verification status. Nothing in this
/// family reproduces a number printed in the Dehaybe et al. (2024) EJOR
/// article: the per-instance benchmark rows are reproduced from the author's
/// public companion-code testbed CSVs (reference implementation), and the
/// Section-4 worked transition is only an internal `step_state` mechanics
/// check. This test fails if any literature_verified flag is flipped to true
/// or any verification_source is silently changed, so the status cannot drift
/// into an overclaim without an explicit code edit that re-runs a real
/// paper-printed number.
#[test]
fn honest_status_flags_are_false() {
    assert!(!WORKED_EXAMPLE_REFERENCE.literature_verified);
    assert_eq!(
        WORKED_EXAMPLE_REFERENCE.verification_source,
        "internal_step_state_mechanics_self_consistency_not_a_paper_printed_number"
    );
    assert!(!VERIFICATION_PROBLEM_INSTANCE.literature_verified);
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.verification_source,
        "henrideh_drl_mmuls_public_testbed_csv_reference_impl_not_paper_table"
    );
    for instance in list_reference_instances() {
        assert!(
            !instance.literature_verified,
            "instance {} must stay literature_verified=false (author-testbed reference-impl, not a paper-printed number)",
            instance.name
        );
        assert_eq!(
            instance.verification_source,
            "henrideh_drl_mmuls_public_testbed_csv_reference_impl_not_paper_table",
            "instance {} verification_source drifted",
            instance.name
        );
    }
}
