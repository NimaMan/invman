use std::collections::HashMap;

use crate::problems::multi_echelon::env::{
    build_decision_state, build_order_plan_with_mode, build_raw_state, initialize_state,
    parse_allocation_mode, parse_warehouse_base_stock_mode, step_state, AllocationMode,
    InventoryDynamicsMode, MultiEchelonState, WarehouseBaseStockMode,
};
use crate::problems::multi_echelon::finite_horizon_dp::{
    evaluate_stationary_policy, search_best_stationary_policy, solve_optimal_policy,
    ExactHeuristicKind, ExactPolicyEvaluation,
};
use crate::problems::multi_echelon::references::{
    GIJSBRECHTS_2022_REFERENCE, LITERATURE_REFERENCE_INSTANCES, PRIMARY_REFERENCE_INSTANCE,
    VAN_ROY_1997_CASE_STUDY, VERIFICATION_PROBLEM_INSTANCE, WORKED_TRANSITION_REFERENCE,
};

fn nested_pipeline_vec(pipelines: &[&[u32]]) -> Vec<Vec<u32>> {
    pipelines.iter().map(|pipeline| pipeline.to_vec()).collect()
}

fn enumerate_demand_combinations(
    num_retailers: usize,
    demand_support: &[u32],
    demand_probabilities: &[f64],
) -> Vec<(Vec<u32>, f64)> {
    fn recurse(
        retailer_idx: usize,
        num_retailers: usize,
        demand_support: &[u32],
        demand_probabilities: &[f64],
        current: &mut Vec<u32>,
        probability: f64,
        output: &mut Vec<(Vec<u32>, f64)>,
    ) {
        if retailer_idx == num_retailers {
            output.push((current.clone(), probability));
            return;
        }
        for (demand, demand_probability) in demand_support.iter().zip(demand_probabilities.iter()) {
            current.push(*demand);
            recurse(
                retailer_idx + 1,
                num_retailers,
                demand_support,
                demand_probabilities,
                current,
                probability * *demand_probability,
                output,
            );
            current.pop();
        }
    }

    let mut output = Vec::new();
    recurse(
        0,
        num_retailers,
        demand_support,
        demand_probabilities,
        &mut Vec::new(),
        1.0,
        &mut output,
    );
    output
}

fn binomial_probability(trials: usize, successes: usize, success_probability: f64) -> f64 {
    if successes > trials {
        return 0.0;
    }
    let combinations = if successes == 0 || successes == trials {
        1.0
    } else {
        let effective_successes = successes.min(trials - successes);
        let mut numerator = 1.0;
        let mut denominator = 1.0;
        for offset in 0..effective_successes {
            numerator *= (trials - offset) as f64;
            denominator *= (offset + 1) as f64;
        }
        numerator / denominator
    };
    combinations
        * success_probability.powi(successes as i32)
        * (1.0 - success_probability).powi((trials - successes) as i32)
}

fn total_unmet_without_emergency(state: &MultiEchelonState, realized_demands: &[u32]) -> usize {
    let decision_state = build_decision_state(state).expect("decision state must build");
    realized_demands
        .iter()
        .enumerate()
        .map(|(retailer_idx, demand)| {
            demand.saturating_sub(decision_state.retailer_available[retailer_idx].max(0) as u32)
                as usize
        })
        .sum()
}

fn independent_optimal_policy(
    reference: &crate::problems::multi_echelon::references::ExactVerificationReference,
) -> ExactPolicyEvaluation {
    fn solve_from_state(
        state: &MultiEchelonState,
        reference: &crate::problems::multi_echelon::references::ExactVerificationReference,
        warehouse_base_stock_mode: WarehouseBaseStockMode,
        allocation_mode: AllocationMode,
        demand_combinations: &[(Vec<u32>, f64)],
        cache: &mut HashMap<MultiEchelonState, ExactPolicyEvaluation>,
    ) -> ExactPolicyEvaluation {
        if state.period == reference.periods {
            return ExactPolicyEvaluation {
                discounted_cost: 0.0,
                first_action: vec![0, 0],
            };
        }
        if let Some(cached) = cache.get(state) {
            return cached.clone();
        }

        let mut best: Option<ExactPolicyEvaluation> = None;
        for warehouse_level in reference.action_warehouse_levels.iter().copied() {
            for retailer_level in reference.action_retailer_levels.iter().copied() {
                let mut expected_cost = 0.0;
                for (demands, demand_probability) in demand_combinations.iter() {
                    let total_unmet = total_unmet_without_emergency(state, demands);
                    for accepted_emergency_shipments in 0..=total_unmet {
                        let acceptance_probability = binomial_probability(
                            total_unmet,
                            accepted_emergency_shipments,
                            reference.expedited_service_prob,
                        );
                        if acceptance_probability <= 0.0 {
                            continue;
                        }
                        let outcome = step_state(
                            state,
                            warehouse_level,
                            retailer_level,
                            demands,
                            accepted_emergency_shipments,
                            reference.warehouse_capacity,
                            reference.warehouse_inventory_cap,
                            reference.retailer_inventory_cap,
                            reference.warehouse_holding_cost,
                            reference.retailer_holding_cost,
                            reference.warehouse_expedited_cost,
                            reference.warehouse_lost_sale_cost,
                            warehouse_base_stock_mode,
                            allocation_mode,
                        )
                        .expect("step must succeed");
                        let continuation = solve_from_state(
                            &outcome.next_state,
                            reference,
                            warehouse_base_stock_mode,
                            allocation_mode,
                            demand_combinations,
                            cache,
                        );
                        expected_cost += demand_probability
                            * acceptance_probability
                            * (outcome.period_cost
                                + reference.discount_factor * continuation.discounted_cost);
                    }
                }
                let candidate = ExactPolicyEvaluation {
                    discounted_cost: expected_cost,
                    first_action: vec![warehouse_level, retailer_level],
                };
                let should_replace = match best.as_ref() {
                    Some(current) => candidate.discounted_cost < current.discounted_cost - 1e-12,
                    None => true,
                };
                if should_replace {
                    best = Some(candidate);
                }
            }
        }

        let result = best.expect("there must be at least one action candidate");
        cache.insert(state.clone(), result.clone());
        result
    }

    let warehouse_base_stock_mode =
        parse_warehouse_base_stock_mode(reference.warehouse_base_stock_mode)
            .expect("warehouse mode must parse");
    let allocation_mode =
        parse_allocation_mode(reference.allocation_mode).expect("allocation mode must parse");
    let demand_combinations = enumerate_demand_combinations(
        reference.num_retailers,
        reference.demand_support,
        reference.demand_probabilities,
    );
    let initial_state = initialize_state(
        reference.initial_warehouse_inventory,
        reference.initial_warehouse_pipeline,
        reference.initial_retailer_inventory,
        &nested_pipeline_vec(reference.initial_retailer_pipeline),
    )
    .expect("state must build");
    let mut cache = HashMap::new();
    solve_from_state(
        &initial_state,
        reference,
        warehouse_base_stock_mode,
        allocation_mode,
        &demand_combinations,
        &mut cache,
    )
}

fn brute_force_best_stationary_policy(
    reference: &crate::problems::multi_echelon::references::ExactVerificationReference,
    heuristic_kind: ExactHeuristicKind,
    allocation_mode: AllocationMode,
) -> (usize, usize, ExactPolicyEvaluation) {
    let mut best: Option<(usize, usize, ExactPolicyEvaluation)> = None;
    for warehouse_level in reference.action_warehouse_levels.iter().copied() {
        for retailer_level in reference.action_retailer_levels.iter().copied() {
            let evaluation = evaluate_stationary_policy(
                reference,
                heuristic_kind,
                allocation_mode,
                warehouse_level,
                retailer_level,
            )
            .expect("stationary policy must evaluate");
            let should_replace = match best.as_ref() {
                Some((_, _, current)) => {
                    evaluation.discounted_cost < current.discounted_cost - 1e-12
                }
                None => true,
            };
            if should_replace {
                best = Some((warehouse_level, retailer_level, evaluation));
            }
        }
    }
    best.expect("there must be at least one stationary candidate")
}

#[test]
fn reference_catalog_matches_gijs_and_van_roy() {
    assert_eq!(GIJSBRECHTS_2022_REFERENCE.benchmark_policies.len(), 3);
    // Catalog order: [0] simple, [1] case_study1, [2] case_study2 (van_roy_1997
    // reproduction instances), [3] setting1, [4] setting2 (paper-faithful gijs_2022
    // search targets).
    assert_eq!(LITERATURE_REFERENCE_INSTANCES.len(), 5);
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[0].name,
        "van_roy1997_simple_problem"
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[1].name,
        "van_roy1997_case_study1"
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[2].name,
        "van_roy1997_case_study2"
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[3].name,
        "gijsbrechts2022_setting1"
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[4].name,
        "gijsbrechts2022_setting2"
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[0].published_constant_base_stock_mean_cost,
        Some(51.7)
    );

    // The primary search target is the paper-faithful setting 2: gijs_2022 dynamics,
    // Table-3 demand mean 0, and NO published reproduction rows attached (those live on
    // the van_roy1997_* reproduction instances).
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.name, "gijsbrechts2022_setting2");
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.inventory_dynamics_mode, "gijs_2022");
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.demand_mean, 0.0);
    assert_eq!(
        PRIMARY_REFERENCE_INSTANCE.published_constant_base_stock_mean_cost,
        None
    );
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.published_a3c_savings_pct, None);
    assert_eq!(
        PRIMARY_REFERENCE_INSTANCE.benchmark_warehouse_levels,
        &[50, 60, 70, 80, 90, 100]
    );

    // The A3C relative-savings rows and the calibrated reproduction live on the
    // van_roy1997_case_study* instances.
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[1].published_a3c_savings_pct,
        Some(8.95)
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[1].published_a3c_confidence_half_width_pct,
        Some(0.13)
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[2].published_a3c_savings_pct,
        Some(12.09)
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[2].published_a3c_confidence_half_width_pct,
        Some(0.39)
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[2].inventory_dynamics_mode,
        "van_roy_1997"
    );
    assert_eq!(LITERATURE_REFERENCE_INSTANCES[2].demand_mean, 1.0);
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[2].published_constant_base_stock_mean_cost,
        Some(1449.0)
    );
    assert_eq!(
        VAN_ROY_1997_CASE_STUDY.published_constant_base_stock_mean_cost,
        Some(1302.0)
    );
    assert_eq!(
        VAN_ROY_1997_CASE_STUDY.published_constant_base_stock_levels,
        &[330, 23]
    );
}

#[test]
fn warehouse_order_uses_pre_shipment_position_in_gijs_mode() {
    // Regression test for the Eq. (2) warehouse-order rule. With a warehouse that ships a
    // positive quantity to its retailers this period, the gijs_2022 (paper-faithful) order
    // must be sized from the PRE-shipment inventory position and therefore be INVARIANT to
    // how much is shipped downstream; the van_roy_1997 reproduction mode instead re-orders
    // to cover the shipment (POST-shipment position) and so depends on it.
    //
    // State: warehouse on-hand 5, one outstanding warehouse order of 3 (lw=1); two
    // retailers at zero on-hand with one empty pipeline slot each (lr=1); warehouse
    // base-stock target y^w = 12, ample capacities.
    let make_plan = |retailer_target: usize, mode: InventoryDynamicsMode| {
        let state = initialize_state(5, &[3], &[0, 0], &[vec![0], vec![0]])
            .expect("state must build");
        build_order_plan_with_mode(
            &state,
            12,
            retailer_target,
            100,
            1000,
            100,
            WarehouseBaseStockMode::Regular,
            AllocationMode::MinShortage,
            mode,
        )
        .expect("order plan must build")
    };

    // gijs_2022: pre-shipment IP^w = 5 + 3 = 8, so q^w = 12 - 8 = 4 whether the retailers
    // are replenished (ships 6) or not (ships 0). The shipment is NOT deducted.
    let gijs_ship = make_plan(3, InventoryDynamicsMode::Gijs2022);
    let gijs_no_ship = make_plan(0, InventoryDynamicsMode::Gijs2022);
    assert_eq!(gijs_ship.shipped_retail_orders.iter().sum::<usize>(), 6);
    assert_eq!(gijs_no_ship.shipped_retail_orders.iter().sum::<usize>(), 0);
    assert_eq!(gijs_ship.warehouse_order, 4);
    assert_eq!(gijs_no_ship.warehouse_order, 4);
    // The old (buggy) post-shipment rule would have ordered 12 - (8 - 6) = 10 here.
    assert_ne!(gijs_ship.warehouse_order, 10);

    // van_roy_1997: warehouse_available = 5 (arrival not folded in), future = 3. With
    // retailer_target 3 the warehouse ships 5 and re-orders to restore the post-shipment
    // position (0 + 3 = 3) to 12 -> q^w = 9; with no shipment the position is 5 + 3 = 8 ->
    // q^w = 4. The order therefore DEPENDS on the shipment under the reproduction mode.
    let van_roy_ship = make_plan(3, InventoryDynamicsMode::VanRoy1997);
    let van_roy_no_ship = make_plan(0, InventoryDynamicsMode::VanRoy1997);
    assert_eq!(van_roy_ship.warehouse_order, 9);
    assert_eq!(van_roy_no_ship.warehouse_order, 4);
}

#[test]
fn proportional_allocation_exhausts_available_inventory() {
    let state =
        initialize_state(5, &[], &[0, 0, 4], &[vec![], vec![], vec![]]).expect("state must build");
    let plan = build_order_plan_with_mode(
        &state,
        0,
        5,
        100,
        100,
        100,
        WarehouseBaseStockMode::Regular,
        AllocationMode::Proportional,
        InventoryDynamicsMode::VanRoy1997,
    )
    .expect("order plan must build");

    assert_eq!(plan.desired_retail_orders, vec![5, 5, 1]);
    assert_eq!(plan.shipped_retail_orders.iter().sum::<usize>(), 5);
    assert_eq!(plan.remaining_warehouse_inventory_after_regular, 0);
}

#[test]
fn raw_state_layout_matches_expected_shape() {
    let state =
        initialize_state(3, &[2, 2], &[1, 0], &vec![vec![1], vec![0]]).expect("state must build");
    let raw_state = build_raw_state(&state).expect("raw state must build");
    assert_eq!(raw_state, vec![3.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn van_roy_feature_layout_matches_expected_shape() {
    let state =
        initialize_state(3, &[2, 2], &[1, 0], &vec![vec![1], vec![0]]).expect("state must build");
    let features = build_policy_features_with_mode(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.warehouse_inventory_cap,
        VERIFICATION_PROBLEM_INSTANCE.retailer_inventory_cap,
        false,
        VERIFICATION_PROBLEM_INSTANCE.periods,
        PolicyFeatureMode::CompactSummary,
        InventoryDynamicsMode::Gijs2022,
    )
    .expect("features must build");

    // Compact-summary layout characterization (NOT a literature claim; the repo
    // rule explicitly excludes frozen snapshots from literature verification).
    // Indices 14-16 are the per-retailer normalized stage-inventory variances.
    // Under the faithful gijs_2022 decision state, each retailer's available
    // inventory includes the first inbound pipeline unit (pre-shipment / arrives
    // this period), so for state retailer_inventory=[1,0], retailer_pipeline=
    // [[1],[0]] the stage-0 inventories are [2,0]/4 = [0.5,0] with population
    // variance 0.0625. (The earlier carried value 0.015625 was the pre-fix
    // van-roy-style decision state that omitted the arriving pipeline unit; the
    // mode-dependent fix made the gijs_2022 decision state faithful.)
    let expected = vec![
        0.25, 0.0, 0.0, 0.625, 0.25, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.390625, 0.0625, 0.0, 0.0,
        0.0625, 0.0625, 0.0625, 0.15625, 0.15625, 0.21875, 0.21875, 0.0,
    ];
    assert_eq!(features.len(), 22);
    for (observed, target) in features.iter().zip(expected.iter()) {
        assert!((observed - target).abs() < 1e-6);
    }
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_REFERENCE;
    let state = initialize_state(
        worked.initial_warehouse_inventory,
        worked.initial_warehouse_pipeline,
        worked.initial_retailer_inventory,
        &nested_pipeline_vec(worked.initial_retailer_pipeline),
    )
    .expect("state must build");
    let outcome = step_state(
        &state,
        worked.warehouse_target,
        worked.retailer_target,
        worked.realized_demands,
        worked.accepted_emergency_shipments,
        VERIFICATION_PROBLEM_INSTANCE.warehouse_capacity,
        VERIFICATION_PROBLEM_INSTANCE.warehouse_inventory_cap,
        VERIFICATION_PROBLEM_INSTANCE.retailer_inventory_cap,
        VERIFICATION_PROBLEM_INSTANCE.warehouse_holding_cost,
        VERIFICATION_PROBLEM_INSTANCE.retailer_holding_cost,
        VERIFICATION_PROBLEM_INSTANCE.warehouse_expedited_cost,
        VERIFICATION_PROBLEM_INSTANCE.warehouse_lost_sale_cost,
        parse_warehouse_base_stock_mode(worked.warehouse_base_stock_mode)
            .expect("warehouse mode must parse"),
        parse_allocation_mode(worked.allocation_mode).expect("allocation mode must parse"),
    )
    .expect("step must succeed");

    assert_eq!(
        outcome.order_plan.warehouse_order,
        worked.expected_warehouse_order
    );
    assert_eq!(
        outcome.order_plan.shipped_retail_orders,
        worked.expected_shipped_retail_orders.to_vec()
    );
    assert_eq!(
        outcome.next_state.warehouse_inventory,
        worked.expected_next_warehouse_inventory
    );
    assert_eq!(
        outcome.next_state.warehouse_pipeline,
        worked.expected_next_warehouse_pipeline.to_vec()
    );
    assert_eq!(
        outcome.next_state.retailer_inventory,
        worked.expected_next_retailer_inventory.to_vec()
    );
    assert_eq!(
        outcome.next_state.retailer_pipeline,
        nested_pipeline_vec(worked.expected_next_retailer_pipeline)
    );
    assert!((outcome.period_cost - worked.expected_period_cost).abs() < 1e-12);
}

#[test]
fn exact_dp_and_heuristics_match_generated_oracles() {
    let optimal =
        solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE).expect("optimal DP must solve");
    let independent_optimal = independent_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE);
    let sequential = search_best_stationary_policy(
        &VERIFICATION_PROBLEM_INSTANCE,
        ExactHeuristicKind::RegularBaseStock,
        parse_allocation_mode("sequential_index").expect("mode"),
    )
    .expect("sequential heuristic search must solve");
    let proportional = search_best_stationary_policy(
        &VERIFICATION_PROBLEM_INSTANCE,
        ExactHeuristicKind::RegularBaseStock,
        parse_allocation_mode("proportional").expect("mode"),
    )
    .expect("proportional heuristic search must solve");
    let min_shortage = search_best_stationary_policy(
        &VERIFICATION_PROBLEM_INSTANCE,
        ExactHeuristicKind::RegularBaseStock,
        parse_allocation_mode("min_shortage").expect("mode"),
    )
    .expect("min-shortage heuristic search must solve");
    let brute_force_sequential = brute_force_best_stationary_policy(
        &VERIFICATION_PROBLEM_INSTANCE,
        ExactHeuristicKind::RegularBaseStock,
        parse_allocation_mode("sequential_index").expect("mode"),
    );
    let brute_force_proportional = brute_force_best_stationary_policy(
        &VERIFICATION_PROBLEM_INSTANCE,
        ExactHeuristicKind::RegularBaseStock,
        parse_allocation_mode("proportional").expect("mode"),
    );
    let brute_force_min_shortage = brute_force_best_stationary_policy(
        &VERIFICATION_PROBLEM_INSTANCE,
        ExactHeuristicKind::RegularBaseStock,
        parse_allocation_mode("min_shortage").expect("mode"),
    );

    assert!((optimal.discounted_cost - independent_optimal.discounted_cost).abs() < 1e-12);
    assert_eq!(optimal.first_action, independent_optimal.first_action);

    assert_eq!(
        [sequential.0, sequential.1],
        [brute_force_sequential.0, brute_force_sequential.1]
    );
    assert!(
        (sequential.2.discounted_cost - brute_force_sequential.2.discounted_cost).abs() < 1e-12
    );
    assert_eq!(
        sequential.2.first_action,
        brute_force_sequential.2.first_action
    );

    assert_eq!(
        [proportional.0, proportional.1],
        [brute_force_proportional.0, brute_force_proportional.1]
    );
    assert!(
        (proportional.2.discounted_cost - brute_force_proportional.2.discounted_cost).abs() < 1e-12
    );
    assert_eq!(
        proportional.2.first_action,
        brute_force_proportional.2.first_action
    );

    assert_eq!(
        [min_shortage.0, min_shortage.1],
        [brute_force_min_shortage.0, brute_force_min_shortage.1]
    );
    assert!(
        (min_shortage.2.discounted_cost - brute_force_min_shortage.2.discounted_cost).abs() < 1e-12
    );
    assert_eq!(
        min_shortage.2.first_action,
        brute_force_min_shortage.2.first_action
    );
    assert!(optimal.discounted_cost <= proportional.2.discounted_cost + 1e-12);
    assert!(optimal.discounted_cost <= sequential.2.discounted_cost + 1e-12);
    assert!(optimal.discounted_cost <= min_shortage.2.discounted_cost + 1e-12);
}
