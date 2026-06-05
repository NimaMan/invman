use crate::problems::joint_replenishment::env::{
    build_raw_state, initialize_state, step_state, JointReplenishmentState,
};
use crate::problems::joint_replenishment::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::joint_replenishment::heuristics::{
    dynamic_order_up_to_order_quantities, minimum_order_quantity_order_quantities,
};
use crate::problems::joint_replenishment::literature::references::{
    PRIMARY_REFERENCE_INSTANCE, SMALL_SCALE_SETTINGS, VANVUCHELEN_2020_FIGURE3_ANCHOR,
    VANVUCHELEN_2020_REFERENCE, VERIFICATION_PROBLEM_INSTANCE,
};

#[derive(Clone, Copy)]
struct WorkedTransitionCase {
    initial_inventory_levels: &'static [i32],
    action: &'static [usize],
    realized_demands: &'static [usize],
    truck_capacity: usize,
    major_order_cost: f64,
    minor_order_costs: &'static [f64],
    holding_costs: &'static [f64],
    shortage_costs: &'static [f64],
    expected_next_inventory_levels: &'static [i32],
    expected_trucks_used: usize,
    expected_order_cost: f64,
    expected_holding_cost: f64,
    expected_shortage_cost: f64,
    expected_period_cost: f64,
}

const WORKED_TRANSITION_CASE: WorkedTransitionCase = WorkedTransitionCase {
    initial_inventory_levels: &[1, -2],
    action: &[4, 2],
    realized_demands: &[3, 0],
    truck_capacity: 6,
    major_order_cost: 75.0,
    minor_order_costs: &[10.0, 10.0],
    holding_costs: &[1.0, 1.0],
    shortage_costs: &[19.0, 19.0],
    expected_next_inventory_levels: &[2, 0],
    expected_trucks_used: 1,
    expected_order_cost: 95.0,
    expected_holding_cost: 2.0,
    expected_shortage_cost: 0.0,
    expected_period_cost: 97.0,
};

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(VANVUCHELEN_2020_REFERENCE.benchmark_policies.len(), 3);
    assert!(VANVUCHELEN_2020_REFERENCE.reported_numbers_available);
    // The paper exposes one exact, executable anchor (Figure 3 optimal action for setting 5),
    // so the reference is now treated as anchoring a repo assertion.
    assert!(VANVUCHELEN_2020_REFERENCE.numbers_anchor_repo_assertions);
    assert_eq!(SMALL_SCALE_SETTINGS.len(), 16);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.num_items, 2);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.truck_capacity, 6);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.minor_order_costs[0], 40.0);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.demand_ranges[0].high, 5);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.demand_ranges[1].high, 3);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.periods, 4);
    assert!(!VERIFICATION_PROBLEM_INSTANCE.literature_verified);
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.verification_source,
        "repo_finite_horizon_self_consistency_comparator"
    );
}

#[test]
fn raw_state_layout_matches_expected_shape() {
    let state = JointReplenishmentState {
        period: 2,
        inventory_levels: vec![2, -1],
    };
    let raw_state = build_raw_state(&state).expect("raw state must build");
    assert_eq!(raw_state, vec![2.0, -1.0, 2.0]);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_CASE;
    let state = initialize_state(worked.initial_inventory_levels).expect("state must build");
    let outcome = step_state(
        &state,
        worked.action,
        worked.realized_demands,
        worked.truck_capacity,
        worked.minor_order_costs,
        worked.major_order_cost,
        worked.holding_costs,
        worked.shortage_costs,
    )
    .expect("step must succeed");

    assert_eq!(
        outcome.next_state.inventory_levels,
        worked.expected_next_inventory_levels.to_vec()
    );
    assert_eq!(outcome.trucks_used, worked.expected_trucks_used);
    assert_eq!(outcome.order_cost, worked.expected_order_cost);
    assert_eq!(outcome.holding_cost, worked.expected_holding_cost);
    assert_eq!(outcome.shortage_cost, worked.expected_shortage_cost);
    assert_eq!(outcome.period_cost, worked.expected_period_cost);
    assert_eq!(outcome.reward, -worked.expected_period_cost);
}

#[test]
fn partial_truck_actions_are_rejected() {
    let state = initialize_state(&[1, -2]).expect("state must build");
    let result = step_state(
        &state,
        &[4, 1],
        &[3, 0],
        6,
        &[10.0, 10.0],
        75.0,
        &[1.0, 1.0],
        &[19.0, 19.0],
    );
    assert!(result.is_err());
}

#[test]
fn heuristic_initial_orders_match_named_heuristic_evaluators() {
    let reference = VERIFICATION_PROBLEM_INSTANCE;
    let state = initialize_state(reference.initial_inventory_levels).expect("state must build");
    let moq_action = minimum_order_quantity_order_quantities(
        &state,
        reference.moq_item_targets,
        reference.moq_review_period,
        reference.moq_rounding_threshold,
        reference.truck_capacity,
    )
    .expect("MOQ heuristic must succeed");
    let dynout_action = dynamic_order_up_to_order_quantities(
        &state,
        reference.dynout_item_targets,
        reference.truck_capacity,
        reference.demand_ranges,
        reference.holding_costs,
        reference.shortage_costs,
    )
    .expect("DYN-OUT heuristic must succeed");

    let moq = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "minimum_order_quantity")
        .expect("MOQ evaluation must solve");
    let dynout = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "dynamic_order_up_to")
        .expect("DYN-OUT evaluation must solve");

    assert_eq!(moq_action, moq.first_action.to_vec());
    assert_eq!(dynout_action, dynout.first_action.to_vec());
}

#[test]
fn exact_dp_dominates_repo_heuristics() {
    let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)
        .expect("exact optimal policy must solve");
    let moq = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "minimum_order_quantity")
        .expect("MOQ evaluation must solve");
    let dynout = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "dynamic_order_up_to")
        .expect("DYN-OUT evaluation must solve");

    assert_eq!(optimal.first_action.len(), 2);
    assert!(
        optimal.discounted_cost <= moq.discounted_cost + 1e-9,
        "optimal={} moq={}",
        optimal.discounted_cost,
        moq.discounted_cost
    );
    assert!(
        optimal.discounted_cost <= dynout.discounted_cost + 1e-9,
        "optimal={} dynout={}",
        optimal.discounted_cost,
        dynout.discounted_cost
    );
}

#[test]
fn published_figure3_anchor_has_expected_shape() {
    // The carried anchor must point at setting 5 (the family Figure 3/4 visualise) and at the
    // exact state/action the paper states in prose: state (5,0), optimal q=(0,6), heuristic q=(2,4).
    let anchor = VANVUCHELEN_2020_FIGURE3_ANCHOR;
    assert_eq!(anchor.setting_name, "vanvuchelen2020_small_scale_setting_5");
    assert_eq!(anchor.setting_name, PRIMARY_REFERENCE_INSTANCE.name);
    assert_eq!(anchor.state_inventory_levels, &[5, 0]);
    assert_eq!(anchor.optimal_action, &[0, 6]);
    assert_eq!(anchor.heuristic_action, &[2, 4]);
    assert_eq!(anchor.discount_factor, 0.99);

    // Both the published optimal action and the published heuristic action ship exactly one full
    // truckload (aggregate = V = 6), consistent with the setting-5 truck capacity.
    let truck_capacity = PRIMARY_REFERENCE_INSTANCE.truck_capacity;
    let optimal_total: usize = anchor.optimal_action.iter().sum();
    let heuristic_total: usize = anchor.heuristic_action.iter().sum();
    assert_eq!(optimal_total % truck_capacity, 0);
    assert_eq!(heuristic_total % truck_capacity, 0);
    assert_eq!(optimal_total / truck_capacity, 1);
    assert_eq!(heuristic_total / truck_capacity, 1);
}

#[test]
fn env_reproduces_figure3_anchor_one_period_cost() {
    // Numerically verify the env one-period accounting (paper Eq. 2 / Eq. 4) at the published
    // anchor state-action for setting 5. We do NOT re-derive the optimal action here (that needs an
    // infinite-horizon solver, exercised independently in the benchmark script); we confirm that the
    // env evaluates the published optimal action with the paper's cost convention.
    //
    // Setting 5: K=75, k=[40,10], h=[1,1], b=[19,19], V=6.
    // State (5,0), optimal action q=(0,6): only shipper 2 orders one FTL.
    //   trucks = 1  -> major cost = 75
    //   minor  = k2 = 10 (only item 2 ordered)            -> order cost = 85
    // Take a worked demand d=(2,4):
    //   I1 = 5 + 0 - 2 = 3  (holding 1*3 = 3)
    //   I2 = 0 + 6 - 4 = 2  (holding 1*2 = 2)              -> holding cost = 5, shortage = 0
    //   period cost = 85 + 5 = 90
    let setting = PRIMARY_REFERENCE_INSTANCE;
    let anchor = VANVUCHELEN_2020_FIGURE3_ANCHOR;
    let state = initialize_state(anchor.state_inventory_levels).expect("state must build");
    let demand = [2usize, 4usize];
    let outcome = step_state(
        &state,
        anchor.optimal_action,
        &demand,
        setting.truck_capacity,
        setting.minor_order_costs,
        setting.major_order_cost,
        setting.holding_costs,
        setting.shortage_costs,
    )
    .expect("step must succeed");

    assert_eq!(outcome.trucks_used, 1);
    assert_eq!(outcome.order_cost, 85.0);
    assert_eq!(outcome.next_state.inventory_levels, vec![3, 2]);
    assert_eq!(outcome.holding_cost, 5.0);
    assert_eq!(outcome.shortage_cost, 0.0);
    assert_eq!(outcome.period_cost, 90.0);
}

/// Literature verification (executing): infinite-horizon value iteration over the env's own Eq.2
/// cost and Eq.4 transition (step_state, gamma=0.99) RE-DERIVES the published Vanvuchelen et al.
/// (2020) Figure-3 optimal action for setting 5. At state (I1,I2)=(5,0) the optimum must be the
/// published q=(0,6) (one full truckload to shipper 2), strictly cheaper than the heuristics' q=(2,4).
/// This was previously reproduced only by an external Python script; here it is an in-crate optimality
/// check. The published anchor is an optimal ACTION, not a cost (the paper reports per-setting gaps
/// only as a figure), so no absolute-cost assertion is possible.
///
/// Inventory grid [-12, 18] (the optimum at (5,0) is invariant up to [-18, 24]); 24 equiprobable
/// demand scenarios = U[0,5] x U[0,3]; actions = total order in {0, V, 2V, 3V} split between items.
#[test]
fn value_iteration_reproduces_figure3_optimal_action() {
    let setting = PRIMARY_REFERENCE_INSTANCE;
    let anchor = VANVUCHELEN_2020_FIGURE3_ANCHOR;
    let gamma = anchor.discount_factor; // 0.99

    const LO: i32 = -12;
    const HI: i32 = 18;
    let width = (HI - LO + 1) as usize;
    let n_states = width * width;
    let to_index = |i1: i32, i2: i32| ((i1 - LO) as usize) * width + (i2 - LO) as usize;
    let clip = |x: i32| x.clamp(LO, HI);

    // Action set: total order is 0 or a multiple of the truck capacity (paper Eq. 1), up to 3 trucks.
    let v_cap = setting.truck_capacity;
    let mut actions: Vec<(usize, usize)> = Vec::new();
    for total in (0..=3 * v_cap).step_by(v_cap) {
        for q1 in 0..=total {
            actions.push((q1, total - q1));
        }
    }
    let n_actions = actions.len();

    // 24 equiprobable demand scenarios from the inclusive ranges U[0,5] x U[0,3].
    let ranges = setting.demand_ranges;
    let mut demands: Vec<(usize, usize)> = Vec::new();
    for d1 in ranges[0].low..=ranges[0].high {
        for d2 in ranges[1].low..=ranges[1].high {
            demands.push((d1, d2));
        }
    }
    let n_demands = demands.len();
    let inv_n_demands = 1.0 / n_demands as f64;

    // Precompute per (state, action): expected one-period cost (from the env's own step_state) and
    // the clipped next-state indices for every demand scenario.
    let mut immediate = vec![0.0f64; n_states * n_actions];
    let mut next_index = vec![0usize; n_states * n_actions * n_demands];
    for s in 0..n_states {
        let i1 = LO + (s / width) as i32;
        let i2 = LO + (s % width) as i32;
        let state = JointReplenishmentState {
            period: 0,
            inventory_levels: vec![i1, i2],
        };
        for (a, &(q1, q2)) in actions.iter().enumerate() {
            let mut cost_sum = 0.0;
            for (k, &(d1, d2)) in demands.iter().enumerate() {
                let outcome = step_state(
                    &state,
                    &[q1, q2],
                    &[d1, d2],
                    setting.truck_capacity,
                    setting.minor_order_costs,
                    setting.major_order_cost,
                    setting.holding_costs,
                    setting.shortage_costs,
                )
                .expect("step must succeed");
                cost_sum += outcome.period_cost;
                let levels = &outcome.next_state.inventory_levels;
                next_index[(s * n_actions + a) * n_demands + k] =
                    to_index(clip(levels[0]), clip(levels[1]));
            }
            immediate[s * n_actions + a] = cost_sum * inv_n_demands;
        }
    }

    // Discounted value iteration: V(s) <- min_a [ E_d(cost) + gamma * E_d(V(next)) ].
    let mut value = vec![0.0f64; n_states];
    let q_value = |value: &[f64], s: usize, a: usize| -> f64 {
        let base = s * n_actions + a;
        let mut future = 0.0;
        for k in 0..n_demands {
            future += value[next_index[base * n_demands + k]];
        }
        immediate[base] + gamma * future * inv_n_demands
    };
    for _ in 0..5000 {
        let mut max_delta = 0.0f64;
        let mut next_value = vec![0.0f64; n_states];
        for s in 0..n_states {
            let mut best = f64::INFINITY;
            for a in 0..n_actions {
                best = best.min(q_value(&value, s, a));
            }
            next_value[s] = best;
            max_delta = max_delta.max((best - value[s]).abs());
        }
        value = next_value;
        if max_delta < 1e-6 {
            break;
        }
    }

    // Greedy action at the published state (5,0), keeping the first minimizer (lexicographic
    // tie-break by action-generation order: (0,6) precedes the other one-truck splits).
    let state_index = to_index(
        anchor.state_inventory_levels[0],
        anchor.state_inventory_levels[1],
    );
    let mut best_action = actions[0];
    let mut best_q = f64::INFINITY;
    for (a, &action) in actions.iter().enumerate() {
        let q = q_value(&value, state_index, a);
        if q < best_q - 1e-9 {
            best_q = q;
            best_action = action;
        }
    }
    assert_eq!(
        vec![best_action.0, best_action.1],
        anchor.optimal_action.to_vec(),
        "value-iteration greedy action {best_action:?} must equal published optimal {:?}",
        anchor.optimal_action
    );

    // The published optimal action is strictly cheaper than the published heuristic action (2,4)
    // by a clear margin (independent VI observed +36.16 in discounted value units).
    let action_index = |q1: usize, q2: usize| {
        actions
            .iter()
            .position(|&a| a == (q1, q2))
            .expect("action must be in the enumerated set")
    };
    let q_optimal = q_value(&value, state_index, action_index(0, 6));
    let q_heuristic = q_value(&value, state_index, action_index(2, 4));
    assert!(
        q_heuristic - q_optimal > 30.0,
        "optimal q=(0,6) value {q_optimal} should beat heuristic q=(2,4) value {q_heuristic} by >30"
    );
}
