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

#[test]
fn worked_example_transition_matches_section_4_2() {
    let worked = WORKED_EXAMPLE_REFERENCE;
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

#[test]
fn simple_s_s_matches_author_reference_row_within_tolerance() {
    let instance = get_primary_reference_instance();
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

#[test]
fn rolling_dp_matches_author_reference_row_within_tolerance() {
    let verification = VERIFICATION_PROBLEM_INSTANCE;
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
