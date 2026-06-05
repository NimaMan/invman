use crate::problems::decentralized_inventory_control::env::{
    build_local_policy_state, current_received_orders, initialize_state, step_state,
};
use crate::problems::decentralized_inventory_control::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::decentralized_inventory_control::heuristics::{
    base_stock_orders, policy_rollout_from_paths, sterman_anchor_adjust_orders,
};
use crate::problems::decentralized_inventory_control::literature::references::{
    CANER_2014_REFERENCE, CLASSIC_BEER_GAME_CUSTOMER_DEMANDS, MOUSA_2024_REFERENCE,
    OROOJLOYJADID_2021_REFERENCE, PRIMARY_REFERENCE_INSTANCE, STERMAN_1989_CLASSIC_BENCHMARK,
    STERMAN_1989_REFERENCE, VERIFICATION_PROBLEM_INSTANCE,
};
use crate::problems::decentralized_inventory_control::verification::classic_board_game::simulate_classic_sterman_benchmark;

#[derive(Clone, Copy, Debug)]
struct WorkedTransitionCase {
    initial_on_hand_inventory: &'static [usize],
    initial_backlog: &'static [usize],
    initial_shipment_pipelines: &'static [&'static [usize]],
    initial_order_pipelines: &'static [&'static [usize]],
    initial_last_received_shipments: &'static [usize],
    initial_last_received_orders: &'static [usize],
    initial_forecast_orders: &'static [f64],
    initial_last_actions: &'static [usize],
    action: &'static [usize],
    realized_customer_demand: usize,
    demand_smoothing_factors: &'static [f64],
    holding_costs: &'static [f64],
    backlog_costs: &'static [f64],
    expected_received_shipments: &'static [usize],
    expected_received_orders: &'static [usize],
    expected_downstream_shipments: &'static [usize],
    expected_next_on_hand_inventory: &'static [usize],
    expected_next_backlog: &'static [usize],
    expected_next_shipment_pipelines: &'static [&'static [usize]],
    expected_next_order_pipelines: &'static [&'static [usize]],
    expected_period_cost: f64,
}

const WORKED_TRANSITION_CASE: WorkedTransitionCase = WorkedTransitionCase {
    initial_on_hand_inventory: &[12, 12, 12, 12],
    initial_backlog: &[0, 0, 0, 0],
    initial_shipment_pipelines: &[&[4, 4], &[4, 4], &[4, 4], &[4, 4]],
    initial_order_pipelines: &[&[], &[4], &[4], &[4]],
    initial_last_received_shipments: &[4, 4, 4, 4],
    initial_last_received_orders: &[4, 4, 4, 4],
    initial_forecast_orders: &[4.0, 4.0, 4.0, 4.0],
    initial_last_actions: &[4, 4, 4, 4],
    action: &[4, 4, 4, 4],
    realized_customer_demand: 4,
    demand_smoothing_factors: &[0.0, 0.0, 0.0, 0.0],
    holding_costs: &[0.5, 0.5, 0.5, 0.5],
    backlog_costs: &[1.0, 1.0, 1.0, 1.0],
    expected_received_shipments: &[4, 4, 4, 4],
    expected_received_orders: &[4, 4, 4, 4],
    expected_downstream_shipments: &[4, 4, 4, 4],
    expected_next_on_hand_inventory: &[12, 12, 12, 12],
    expected_next_backlog: &[0, 0, 0, 0],
    expected_next_shipment_pipelines: &[&[4, 4], &[4, 4], &[4, 4], &[4, 4]],
    expected_next_order_pipelines: &[&[], &[4], &[4], &[4]],
    expected_period_cost: 24.0,
};

fn nested_vec(rows: &[&[usize]]) -> Vec<Vec<usize>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(OROOJLOYJADID_2021_REFERENCE.benchmark_policies.len(), 3);
    assert_eq!(
        STERMAN_1989_REFERENCE.benchmark_policies,
        &["sterman_anchor_adjust"]
    );
    assert_eq!(
        CANER_2014_REFERENCE.benchmark_policies,
        &["sterman_anchor_adjust"]
    );
    assert_eq!(STERMAN_1989_CLASSIC_BENCHMARK.total_mean_cost, 204.0);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.num_agents, 4);
    assert_eq!(
        PRIMARY_REFERENCE_INSTANCE.benchmark_customer_demands,
        Some(CLASSIC_BEER_GAME_CUSTOMER_DEMANDS)
    );
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.sterman_target_positions[3], 20.0);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_order_quantities, &[4, 4]);
    assert!(!VERIFICATION_PROBLEM_INSTANCE.literature_verified);
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.verification_source,
        "repo_exact_solver_not_verified_against_literature"
    );
    assert_eq!(MOUSA_2024_REFERENCE.benchmark_policies.len(), 2);
}

#[test]
fn classic_sterman_benchmark_matches_literature() {
    let summary = simulate_classic_sterman_benchmark();

    assert_eq!(summary.per_agent_costs, [46.0, 50.0, 54.0, 54.0]);
    assert_eq!(summary.total_cost, 204.0);
}

#[test]
fn local_policy_state_layout_matches_expected_shape() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_on_hand_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_backlog,
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_shipment_pipelines),
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_order_pipelines),
        VERIFICATION_PROBLEM_INSTANCE.initial_last_received_shipments,
        VERIFICATION_PROBLEM_INSTANCE.initial_last_received_orders,
        VERIFICATION_PROBLEM_INSTANCE.initial_forecast_orders,
        VERIFICATION_PROBLEM_INSTANCE.initial_last_actions,
    )
    .expect("state must build");
    let features = build_local_policy_state(
        &state,
        0,
        VERIFICATION_PROBLEM_INSTANCE.periods,
        VERIFICATION_PROBLEM_INSTANCE.holding_costs,
        VERIFICATION_PROBLEM_INSTANCE.backlog_costs,
        1,
    )
    .expect("local policy state must build");

    assert_eq!(features.len(), 12);
    assert!((features[0] - 1.0).abs() < 1e-6);
    assert!((features[2] - 1.0).abs() < 1e-6);
    assert!((features[6] - 0.5).abs() < 1e-6);
    assert!((features[8] - 0.5).abs() < 1e-6);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_CASE;
    let state = initialize_state(
        worked.initial_on_hand_inventory,
        worked.initial_backlog,
        &nested_vec(worked.initial_shipment_pipelines),
        &nested_vec(worked.initial_order_pipelines),
        worked.initial_last_received_shipments,
        worked.initial_last_received_orders,
        worked.initial_forecast_orders,
        worked.initial_last_actions,
    )
    .expect("state must build");
    let outcome = step_state(
        &state,
        worked.action,
        worked.realized_customer_demand,
        worked.demand_smoothing_factors,
        worked.holding_costs,
        worked.backlog_costs,
    )
    .expect("step must succeed");

    assert_eq!(
        outcome.received_shipments,
        worked.expected_received_shipments
    );
    assert_eq!(outcome.received_orders, worked.expected_received_orders);
    assert_eq!(
        outcome.downstream_shipments,
        worked.expected_downstream_shipments
    );
    assert_eq!(
        outcome.next_state.on_hand_inventory,
        worked.expected_next_on_hand_inventory
    );
    assert_eq!(outcome.next_state.backlog, worked.expected_next_backlog);
    assert_eq!(
        outcome.next_state.shipment_pipelines,
        nested_vec(worked.expected_next_shipment_pipelines)
    );
    assert_eq!(
        outcome.next_state.order_pipelines,
        nested_vec(worked.expected_next_order_pipelines)
    );
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
}

#[test]
fn heuristic_first_actions_match_named_heuristic_evaluators() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_on_hand_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_backlog,
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_shipment_pipelines),
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_order_pipelines),
        VERIFICATION_PROBLEM_INSTANCE.initial_last_received_shipments,
        VERIFICATION_PROBLEM_INSTANCE.initial_last_received_orders,
        VERIFICATION_PROBLEM_INSTANCE.initial_forecast_orders,
        VERIFICATION_PROBLEM_INSTANCE.initial_last_actions,
    )
    .expect("state must build");
    let base_stock_eval = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "base_stock")
        .expect("base-stock evaluation must solve");
    let sterman_eval =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "sterman_anchor_adjust")
            .expect("Sterman evaluation must solve");

    for demand in VERIFICATION_PROBLEM_INSTANCE
        .customer_demand_support
        .iter()
        .copied()
    {
        let observed_orders =
            current_received_orders(&state, demand as usize).expect("observed orders must build");
        let base_stock = base_stock_orders(
            &state,
            &observed_orders,
            VERIFICATION_PROBLEM_INSTANCE.base_stock_levels,
        )
        .expect("base-stock must compute");
        let sterman = sterman_anchor_adjust_orders(
            &state,
            &observed_orders,
            VERIFICATION_PROBLEM_INSTANCE.sterman_target_positions,
            VERIFICATION_PROBLEM_INSTANCE.sterman_adjustment_times,
            VERIFICATION_PROBLEM_INSTANCE.sterman_supply_line_weights,
        )
        .expect("Sterman heuristic must compute");

        let base_stock_first_action = base_stock_eval
            .first_actions_by_customer_demand
            .iter()
            .find(|(supported_demand, _)| *supported_demand == demand)
            .map(|(_, action)| action.clone())
            .expect("base-stock branch action must exist");
        let sterman_first_action = sterman_eval
            .first_actions_by_customer_demand
            .iter()
            .find(|(supported_demand, _)| *supported_demand == demand)
            .map(|(_, action)| action.clone())
            .expect("Sterman branch action must exist");

        assert_eq!(base_stock, base_stock_first_action);
        assert_eq!(sterman, sterman_first_action);
    }
}

#[test]
fn exact_dp_dominates_repo_heuristics() {
    let optimal =
        solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE).expect("optimal policy must solve");
    let base_stock = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "base_stock")
        .expect("base-stock evaluation must solve");
    let sterman = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "sterman_anchor_adjust")
        .expect("Sterman evaluation must solve");

    assert!(
        optimal.discounted_cost <= base_stock.discounted_cost + 1e-9,
        "optimal={} base_stock={}",
        optimal.discounted_cost,
        base_stock.discounted_cost
    );
    assert!(
        optimal.discounted_cost <= sterman.discounted_cost + 1e-9,
        "optimal={} sterman={}",
        optimal.discounted_cost,
        sterman.discounted_cost
    );
}

/// Characterization (executing, NOT a snapshot): env.rs is a distinct decentralized serial MDP
/// (mandatory order/information lead time + installation base-stock + post-fulfillment cost). Under
/// the SAME published Sterman parameters and 36-week path, env.rs + sterman_anchor_adjust totals
/// 378, NOT the closed-form board-game benchmark 204 ([46,50,54,54]). The 204 is reproduced only by
/// the closed-form simulator (classic_sterman_benchmark_matches_literature). No published *cost* is
/// reproducible by this env (every published Beer-Game benchmark uses echelon control on a
/// zero-order-delay MDP), so literature_verified stays false; this test pins the honest structural
/// gap so a future env change that alters it must update this assertion deliberately.
/// (Mirrors scripts/decentralized_inventory_control/measure_env_vs_closedform.py.)
#[test]
fn env_sterman_anchor_adjust_does_not_reproduce_closed_form_204() {
    let ship_pipelines = vec![vec![4usize, 4], vec![4, 4], vec![4, 4], vec![4, 4]];
    let order_pipelines = vec![vec![], vec![4usize], vec![4], vec![4]];
    let state = initialize_state(
        &[12, 12, 12, 12],
        &[0, 0, 0, 0],
        &ship_pipelines,
        &order_pipelines,
        &[4, 4, 4, 4],
        &[4, 4, 4, 4],
        &[4.0, 4.0, 4.0, 4.0],
        &[4, 4, 4, 4],
    )
    .expect("Beer Game initial state must build");

    // Sterman params: targets(4) | adjustment_times(4) | supply_line_weights(4).
    let sterman_params = [28.0, 28.0, 28.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let mut customer_demands = vec![4usize, 4, 4, 4];
    customer_demands.extend(std::iter::repeat(8).take(32)); // canonical 36-week path

    let env_total = policy_rollout_from_paths(
        "sterman_anchor_adjust",
        &sterman_params,
        &state,
        &customer_demands,
        &[0.0, 0.0, 0.0, 0.0],
        &[0.5, 0.5, 0.5, 0.5],
        &[1.0, 1.0, 1.0, 1.0],
        1.0,
    )
    .expect("env rollout must succeed");

    assert!(
        (env_total - 378.0).abs() < 1e-9,
        "env.rs sterman_anchor_adjust total {env_total} should be 378 (closed-form benchmark is 204)"
    );
    let closed_form = simulate_classic_sterman_benchmark();
    assert_eq!(closed_form.total_cost, 204.0);
    assert!(
        env_total - closed_form.total_cost > 100.0,
        "the env-vs-closed-form gap ({env_total} vs 204) is structural, not a tuning artifact"
    );
}

/// Method-level literature check (executing): for CONSTANT demand the Clark-Scarf serial optimum is
/// the deterministic zero-inventory policy (cost 0). env.rs attains exactly that under an
/// installation base-stock with S_k = d * (shipment_lead_k + order_lead_{k+1}) when primed at the
/// steady state -- a genuine serial-optimum property of this MDP. This is a literature-METHOD check,
/// not a printed published number, so literature_verified stays false.
#[test]
fn clark_scarf_constant_demand_serial_optimum_is_zero() {
    // 4-stage chain, d=8, shipment leads [2,2,2,4], order leads [0,2,2,2] => S_k = 32 for all k.
    let demand = 8usize;
    let state4 = initialize_state(
        &[0, 0, 0, 0],
        &[0, 0, 0, 0],
        &vec![vec![demand; 2], vec![demand; 2], vec![demand; 2], vec![demand; 4]],
        &vec![vec![], vec![demand; 2], vec![demand; 2], vec![demand; 2]],
        &[demand; 4],
        &[demand; 4],
        &[demand as f64; 4],
        &[demand; 4],
    )
    .expect("primed steady-state must build");
    let cost4 = policy_rollout_from_paths(
        "base_stock",
        &[32.0, 32.0, 32.0, 32.0],
        &state4,
        &vec![demand; 100],
        &[0.0; 4],
        &[0.5; 4],
        &[1.0; 4],
        1.0,
    )
    .expect("rollout must succeed");
    assert!(cost4.abs() < 1e-9, "4-stage constant-demand serial optimum should be 0, got {cost4}");

    // 2-stage chain, d=2, shipment leads [1,2], order leads [0,1] => S = [4,4].
    let d2 = 2usize;
    let state2 = initialize_state(
        &[0, 0],
        &[0, 0],
        &vec![vec![d2; 1], vec![d2; 2]],
        &vec![vec![], vec![d2; 1]],
        &[d2; 2],
        &[d2; 2],
        &[d2 as f64; 2],
        &[d2; 2],
    )
    .expect("primed steady-state must build");
    let cost2 = policy_rollout_from_paths(
        "base_stock",
        &[4.0, 4.0],
        &state2,
        &vec![d2; 100],
        &[0.0; 2],
        &[0.5; 2],
        &[1.0; 2],
        1.0,
    )
    .expect("rollout must succeed");
    assert!(cost2.abs() < 1e-9, "2-stage constant-demand serial optimum should be 0, got {cost2}");
}
