use crate::problems::multi_echelon::production_assembly_distribution_network::env::{
    build_policy_state, initialize_state, step_state, supply_relation_count, NetworkInventoryGraph,
    NetworkNodeMode,
};
use crate::problems::multi_echelon::production_assembly_distribution_network::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::multi_echelon::production_assembly_distribution_network::heuristics::pairwise_base_stock_requests;
use crate::problems::multi_echelon::production_assembly_distribution_network::literature::{
    ExactVerificationReference, PIRHOOSHYARAN_2021_REFERENCE, PRIMARY_REFERENCE_INSTANCE,
    SERIAL_BENCHMARK_ROWS, SINGLE_NODE_BENCHMARK_ROWS, VERIFICATION_PROBLEM_INSTANCE,
};
use crate::problems::multi_echelon::production_assembly_distribution_network::verification::fixtures::WORKED_TRANSITION_CASE;
use crate::problems::multi_echelon::production_assembly_distribution_network::verification::literature_benchmarks::literature_benchmark_summary;

fn build_graph(
    num_nodes: usize,
    source_nodes: &[bool],
    node_modes: &[crate::problems::multi_echelon::production_assembly_distribution_network::env::NetworkNodeMode],
    external_supplier_lead_times: &[usize],
    edges: &[crate::problems::multi_echelon::production_assembly_distribution_network::env::NetworkEdge],
) -> NetworkInventoryGraph {
    NetworkInventoryGraph {
        num_nodes,
        source_nodes: source_nodes.to_vec(),
        node_modes: node_modes.to_vec(),
        external_supplier_lead_times: external_supplier_lead_times.to_vec(),
        edges: edges.to_vec(),
    }
}

fn nested_vec(rows: &[&[usize]]) -> Vec<Vec<usize>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(PIRHOOSHYARAN_2021_REFERENCE.benchmark_policies.len(), 2);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.num_nodes, 3);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.edges.len(), 2);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.pairwise_oul_levels.len(), 3);
    assert!(!PRIMARY_REFERENCE_INSTANCE.literature_verified);
    assert_eq!(
        PRIMARY_REFERENCE_INSTANCE.verification_source,
        "pirhooshyaran_network_env_not_literature_verified_serial_optimum_lives_in_multi_echelon_serial"
    );
    assert_eq!(SINGLE_NODE_BENCHMARK_ROWS.len(), 7);
    assert_eq!(SERIAL_BENCHMARK_ROWS.len(), 10);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_supply_requests, &[2, 2]);
    assert!(!VERIFICATION_PROBLEM_INSTANCE.literature_verified);
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.verification_source,
        "repo_exact_solver_not_verified_against_literature"
    );
}

#[test]
fn single_node_rows_match_analytical_newsvendor_values() {
    let summary = literature_benchmark_summary(32, 1234);
    assert_eq!(summary.single_node_results.len(), 7);
    for result in summary.single_node_results.iter() {
        assert!(
            (result.reproduced_analytical_oul - result.published_analytical_oul).abs() <= 0.02,
            "case {} OUL published={} reproduced={}",
            result.case_idx,
            result.published_analytical_oul,
            result.reproduced_analytical_oul
        );
        assert!(
            (result.reproduced_analytical_average_cost - result.published_analytical_average_cost)
                .abs()
                <= 0.02,
            "case {} cost published={} reproduced={}",
            result.case_idx,
            result.published_analytical_average_cost,
            result.reproduced_analytical_average_cost
        );
    }
}

#[test]
fn serial_rows_reproduced_by_exact_clark_scarf_solver() {
    use crate::problems::multi_echelon::serial::exact::{
        solve_from_local_costs, GridParams, SerialDemand,
    };

    assert_eq!(SERIAL_BENCHMARK_ROWS.len(), 10);
    // Case 3 is Snyder & Shen Example 6.1 with published optimal cost 47.65.
    assert_eq!(SERIAL_BENCHMARK_ROWS[2].published_average_cost, 47.65);

    // The exact Clark-Scarf decomposition reproduces every published serial optimal
    // cost within 0.5% relative error (it lands within 0.05% in practice). This is the
    // exact-theory literature anchor for the serial family; the env-simulation
    // reproduction of these analytical costs is tracked separately.
    for row in SERIAL_BENCHMARK_ROWS.iter() {
        let solution = solve_from_local_costs(
            row.holding_costs,
            row.lead_times,
            *row.shortage_costs.last().unwrap(),
            SerialDemand::Normal {
                mean: row.demand_mean,
                std: row.demand_stddev,
            },
            GridParams::default(),
        );
        let relative_error = (solution.optimal_cost - row.published_average_cost).abs()
            / row.published_average_cost;
        assert!(
            relative_error < 0.005,
            "serial case {} reproduced={:.4} published={:.4} rel_err={:.4}",
            row.case_idx,
            solution.optimal_cost,
            row.published_average_cost,
            relative_error
        );
    }
}

#[test]
fn policy_state_layout_matches_expected_shape() {
    let graph = build_graph(
        VERIFICATION_PROBLEM_INSTANCE.num_nodes,
        VERIFICATION_PROBLEM_INSTANCE.source_nodes,
        VERIFICATION_PROBLEM_INSTANCE.node_modes,
        VERIFICATION_PROBLEM_INSTANCE.external_supplier_lead_times,
        VERIFICATION_PROBLEM_INSTANCE.edges,
    );
    let state = initialize_state(
        &graph,
        VERIFICATION_PROBLEM_INSTANCE.initial_finished_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_raw_inventory_by_relation,
        VERIFICATION_PROBLEM_INSTANCE.initial_internal_backlog_by_edge,
        VERIFICATION_PROBLEM_INSTANCE.initial_external_backlog,
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_supply_pipelines),
    )
    .expect("state must build");
    let demand_means = VERIFICATION_PROBLEM_INSTANCE
        .demand_supports
        .iter()
        .zip(VERIFICATION_PROBLEM_INSTANCE.demand_probabilities.iter())
        .map(|(support, probabilities)| {
            support
                .iter()
                .zip(probabilities.iter())
                .map(|(value, probability)| *value as f64 * probability)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let current_demands = vec![0usize, 1usize];
    let features = build_policy_state(
        &graph,
        &state,
        &demand_means,
        &current_demands,
        VERIFICATION_PROBLEM_INSTANCE.periods,
    )
    .expect("policy state must build");

    let expected_len =
        7 * graph.num_nodes + 2 * supply_relation_count(&graph) + graph.edges.len() + 1;
    assert_eq!(features.len(), expected_len);
    assert!((features.last().copied().unwrap_or(-1.0) - 1.0).abs() < 1e-6);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_CASE;
    let graph = build_graph(
        worked.num_nodes,
        worked.source_nodes,
        worked.node_modes,
        worked.external_supplier_lead_times,
        worked.edges,
    );
    let state = initialize_state(
        &graph,
        worked.initial_finished_inventory,
        worked.initial_raw_inventory_by_relation,
        worked.initial_internal_backlog_by_edge,
        worked.initial_external_backlog,
        &nested_vec(worked.initial_supply_pipelines),
    )
    .expect("state must build");
    let outcome = step_state(
        &graph,
        &state,
        worked.action,
        worked.realized_external_demands,
        worked.holding_costs,
        worked.backlog_costs,
    )
    .expect("step must succeed");

    assert_eq!(
        outcome.received_shipments_by_relation,
        worked.expected_received_shipments_by_relation
    );
    assert_eq!(
        outcome.produced_finished_goods,
        worked.expected_produced_finished_goods
    );
    assert_eq!(
        outcome.shipped_on_internal_edges,
        worked.expected_shipped_on_internal_edges
    );
    assert_eq!(
        outcome.shipped_to_external_customer,
        worked.expected_shipped_to_external_customer
    );
    assert_eq!(
        outcome.next_state.finished_inventory,
        worked.expected_next_finished_inventory
    );
    assert_eq!(
        outcome.next_state.raw_inventory_by_relation,
        worked.expected_next_raw_inventory_by_relation
    );
    assert_eq!(
        outcome.next_state.internal_backlog_by_edge,
        worked.expected_next_internal_backlog_by_edge
    );
    assert_eq!(
        outcome.next_state.external_backlog,
        worked.expected_next_external_backlog
    );
    assert_eq!(
        outcome.next_state.supply_pipelines,
        nested_vec(worked.expected_next_supply_pipelines)
    );
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
}

#[test]
fn pairwise_base_stock_first_action_matches_reference_freeze() {
    let graph = build_graph(
        VERIFICATION_PROBLEM_INSTANCE.num_nodes,
        VERIFICATION_PROBLEM_INSTANCE.source_nodes,
        VERIFICATION_PROBLEM_INSTANCE.node_modes,
        VERIFICATION_PROBLEM_INSTANCE.external_supplier_lead_times,
        VERIFICATION_PROBLEM_INSTANCE.edges,
    );
    let state = initialize_state(
        &graph,
        VERIFICATION_PROBLEM_INSTANCE.initial_finished_inventory,
        VERIFICATION_PROBLEM_INSTANCE.initial_raw_inventory_by_relation,
        VERIFICATION_PROBLEM_INSTANCE.initial_internal_backlog_by_edge,
        VERIFICATION_PROBLEM_INSTANCE.initial_external_backlog,
        &nested_vec(VERIFICATION_PROBLEM_INSTANCE.initial_supply_pipelines),
    )
    .expect("state must build");
    let realized_demands = vec![0usize, 1usize];
    let action = pairwise_base_stock_requests(
        &graph,
        &state,
        VERIFICATION_PROBLEM_INSTANCE.base_stock_levels,
        &realized_demands,
    )
    .expect("pairwise base-stock requests must compute");

    let base_stock =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "pairwise_base_stock")
            .expect("base-stock evaluation must solve");
    assert_eq!(action, base_stock.first_action);
}

#[test]
fn exact_dp_dominates_pairwise_base_stock() {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)
        .expect("exact optimal policy must solve");
    let base_stock =
        evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "pairwise_base_stock")
            .expect("base-stock evaluation must solve");

    assert!(
        optimal.discounted_cost <= base_stock.discounted_cost + 1e-9,
        "optimal={} pairwise_base_stock={}",
        optimal.discounted_cost,
        base_stock.discounted_cost
    );
}

/// Literature verification (executing, env-native): the env's OWN finite-horizon DP reproduces a
/// PUBLISHED Pirhooshyaran & Snyder (2021) Table 1 single-node newsvendor cost (h=10, p=30, L=1) by
/// SIMULATING its step_state dynamics -- not by the closed-form newsvendor formula (that is the
/// separate single_node_rows_match_analytical_newsvendor_values test). For the (mu=100, sigma=10)
/// row the env DP at the published order-up-to level reproduces the published 127.11 to ~0.01%; the
/// tiny residual is integer demand/OUL discretization (it vanishes as the scale grows).
///
/// SCOPE: this verifies the SINGLE-NODE mode only. The serial/network optimum 47.65 stays
/// structurally unreachable by this env (eq. 5's local raw-material position lets finished goods
/// accumulate); see serial_echelon_simulation.rs, and PRIMARY_REFERENCE_INSTANCE stays
/// literature_verified=false. 47.65's verified home is the multi_echelon/serial family.
#[test]
fn single_node_newsvendor_cost_reproduced_by_exact_env_dp() {
    use statrs::distribution::{ContinuousCDF, Normal};

    let row = SINGLE_NODE_BENCHMARK_ROWS
        .iter()
        .find(|r| r.demand_mean == 100.0 && r.demand_stddev == 10.0)
        .expect("Table 1 (mu=100, sigma=10) single-node row must exist");

    // Discretize N(mu, sigma) onto integer demand over +-6 sigma via midpoint probability mass.
    let dist = Normal::new(row.demand_mean, row.demand_stddev).expect("normal distribution");
    let lo = (row.demand_mean - 6.0 * row.demand_stddev).floor().max(0.0) as u32;
    let hi = (row.demand_mean + 6.0 * row.demand_stddev).ceil() as u32;
    let mut support: Vec<u32> = Vec::new();
    let mut probabilities: Vec<f64> = Vec::new();
    for demand in lo..=hi {
        support.push(demand);
        probabilities.push(dist.cdf(demand as f64 + 0.5) - dist.cdf(demand as f64 - 0.5));
    }
    let total: f64 = probabilities.iter().sum();
    for probability in probabilities.iter_mut() {
        *probability /= total;
    }
    let order_up_to = row.published_analytical_oul.round() as usize; // 107

    // One source node, one period, supply pipeline warmed to the published OUL so the newsvendor
    // arrival meets demand (the order placed now arrives after the horizon, so cost = newsvendor at
    // S*). The reference struct is 'static; build the runtime demand pmf and leak it (one-shot test).
    // The reference struct holds 'static slices; leak the runtime-derived data (one-shot test).
    let support: &'static [u32] = Box::leak(support.into_boxed_slice());
    let probabilities: &'static [f64] = Box::leak(probabilities.into_boxed_slice());
    let pipeline: &'static [usize] = Box::leak(vec![order_up_to].into_boxed_slice());
    let lead_times: &'static [usize] = Box::leak(vec![row.lead_time].into_boxed_slice());
    let holding_costs: &'static [f64] = Box::leak(vec![row.holding_cost].into_boxed_slice());
    let backlog_costs: &'static [f64] = Box::leak(vec![row.shortage_cost].into_boxed_slice());
    let base_stock_levels: &'static [usize] = Box::leak(vec![order_up_to].into_boxed_slice());
    let reference = ExactVerificationReference {
        source: PIRHOOSHYARAN_2021_REFERENCE.source,
        url: PIRHOOSHYARAN_2021_REFERENCE.url,
        literature_verified: true,
        verification_source: "pirhooshyaran_table1_single_node_newsvendor_reproduced_by_env_dp",
        periods: 1,
        discount_factor: 1.0,
        num_nodes: 1,
        source_nodes: &[true],
        node_modes: &[NetworkNodeMode::Single],
        external_supplier_lead_times: lead_times,
        edges: &[],
        initial_finished_inventory: &[0],
        initial_raw_inventory_by_relation: &[0],
        initial_internal_backlog_by_edge: &[],
        initial_external_backlog: &[0],
        initial_supply_pipelines: Box::leak(vec![pipeline].into_boxed_slice()),
        holding_costs,
        backlog_costs,
        demand_supports: Box::leak(vec![support].into_boxed_slice()),
        demand_probabilities: Box::leak(vec![probabilities].into_boxed_slice()),
        max_supply_requests: &[1],
        base_stock_levels,
        notes: "Table 1 single-node newsvendor case reproduced by env DP (test-constructed).",
    };

    let optimal = solve_optimal_policy(&reference).expect("exact DP must solve");
    let published = row.published_analytical_average_cost;
    let relative_gap = (optimal.discounted_cost - published).abs() / published;
    assert!(
        relative_gap < 0.01,
        "env single-node DP cost {} should reproduce published Table-1 cost {} (rel gap {})",
        optimal.discounted_cost,
        published,
        relative_gap
    );
}
