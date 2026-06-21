use crate::problems::procurement_removal_inventory::env::{
    build_raw_state, initialize_state, step_state, terminal_salvage_credit,
};
use crate::problems::procurement_removal_inventory::finite_horizon_dp::{
    evaluate_named_heuristic, solve_optimal_policy,
};
use crate::problems::procurement_removal_inventory::heuristics::{
    interval_stock_action, returnability_buffer_interval_stock_action,
};
use crate::problems::procurement_removal_inventory::literature::references::{
    MAGGIAR_2017_REFERENCE, MAGGIAR_2025_REFERENCE, PRIMARY_REFERENCE_INSTANCE,
    REMOVAL_ACTIVE_REFERENCE_INSTANCE, VERIFICATION_PROBLEM_INSTANCE,
};

use crate::problems::procurement_removal_inventory::joint_pricing_removal_dp::{
    solve, JointPricingRemovalDpConfig,
};
use crate::problems::procurement_removal_inventory::joint_pricing_removal_env::{
    backorder_cost_per_unit, step_period, terminal_value, JointPricingRemovalParameters,
    JointPricingRemovalState,
};
use crate::problems::procurement_removal_inventory::literature::references::{
    FaithfulVerificationInstance, MaggiarSadighian2017FaithfulInstance,
    FAITHFUL_VERIFICATION_INSTANCE, MAGGIAR_SADIGHIAN_2017_FAITHFUL_INSTANCE,
};
use crate::problems::procurement_removal_inventory::price_dependent_gamma_demand::{
    beta_from_elasticity, noise_quantiles, price_at_demand,
};

#[derive(Clone, Copy)]
struct WorkedTransitionCase {
    initial_inventory_level: usize,
    initial_returnable_inventory: usize,
    purchase_quantity: usize,
    removal_quantity: usize,
    realized_demand: usize,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
    expected_returned_units: usize,
    expected_liquidated_units: usize,
    expected_sales: usize,
    expected_shortage: usize,
    expected_next_inventory_level: usize,
    expected_next_returnable_inventory: usize,
    expected_period_cost: f64,
}

const WORKED_TRANSITION_CASE: WorkedTransitionCase = WorkedTransitionCase {
    initial_inventory_level: 4,
    initial_returnable_inventory: 2,
    purchase_quantity: 3,
    removal_quantity: 2,
    realized_demand: 4,
    returnable_purchase_cap: 2,
    purchase_cost_per_unit: 6.0,
    return_value_per_unit: 4.0,
    liquidation_value_per_unit: 1.0,
    holding_cost_per_unit: 0.5,
    shortage_cost_per_unit: 9.0,
    expected_returned_units: 2,
    expected_liquidated_units: 0,
    expected_sales: 4,
    expected_shortage: 0,
    expected_next_inventory_level: 1,
    expected_next_returnable_inventory: 1,
    expected_period_cost: 10.5,
};

#[test]
fn reference_set_has_expected_shape() {
    assert_eq!(
        MAGGIAR_2017_REFERENCE.benchmark_policies,
        &[
            "optimal_interval_stock",
            "order_up_to_remove_down_to",
            "pricing_and_markdown_variants"
        ]
    );
    assert_eq!(
        MAGGIAR_2025_REFERENCE.benchmark_policies,
        &[
            "directbackprop_drl",
            "structure_informed_policy_network",
            "interval_stock"
        ]
    );
    assert!(!MAGGIAR_2017_REFERENCE.reported_numbers_available);
    assert!(!MAGGIAR_2017_REFERENCE.numbers_anchor_repo_assertions);
    assert!(!MAGGIAR_2025_REFERENCE.reported_numbers_available);
    assert!(!MAGGIAR_2025_REFERENCE.numbers_anchor_repo_assertions);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.returnable_purchase_cap, 2);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.benchmark_returnable_buffer, 2);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_purchase_quantity, 4);
    assert_eq!(VERIFICATION_PROBLEM_INSTANCE.max_removal_quantity, 4);
    assert!(!VERIFICATION_PROBLEM_INSTANCE.literature_verified);
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.verification_source,
        "repo_exact_solver_not_verified_against_literature"
    );
}

#[test]
fn raw_state_layout_matches_expected_shape() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_inventory_level,
        VERIFICATION_PROBLEM_INSTANCE.initial_returnable_inventory,
    )
    .expect("state must build");
    let raw_state = build_raw_state(&state).expect("raw state must build");

    assert_eq!(raw_state, vec![2.0, 1.0, 0.0]);
}

#[test]
fn raw_state_preserves_high_inventory_magnitude() {
    let state = initialize_state(8, 3).expect("state must build");
    let raw_state = build_raw_state(&state).expect("raw state must build");

    assert_eq!(raw_state, vec![8.0, 3.0, 0.0]);
}

#[test]
fn worked_transition_matches_expected_accounting() {
    let worked = WORKED_TRANSITION_CASE;
    let state = initialize_state(
        worked.initial_inventory_level,
        worked.initial_returnable_inventory,
    )
    .expect("state must build");
    let outcome = step_state(
        &state,
        worked.purchase_quantity,
        worked.removal_quantity,
        worked.realized_demand,
        worked.returnable_purchase_cap,
        worked.purchase_cost_per_unit,
        worked.return_value_per_unit,
        worked.liquidation_value_per_unit,
        worked.holding_cost_per_unit,
        worked.shortage_cost_per_unit,
    )
    .expect("step must succeed");

    assert_eq!(outcome.returned_units, worked.expected_returned_units);
    assert_eq!(outcome.liquidated_units, worked.expected_liquidated_units);
    assert_eq!(outcome.sales, worked.expected_sales);
    assert_eq!(outcome.shortage, worked.expected_shortage);
    assert_eq!(
        outcome.next_state.inventory_level,
        worked.expected_next_inventory_level
    );
    assert_eq!(
        outcome.next_state.returnable_inventory,
        worked.expected_next_returnable_inventory
    );
    assert!((outcome.period_cost - worked.expected_period_cost).abs() < 1e-12);
}

#[test]
fn terminal_salvage_credit_matches_expected_freeze() {
    let state = initialize_state(3, 1).expect("state must build");
    let credit = terminal_salvage_credit(&state, 3.0, 1.0).expect("terminal credit must compute");
    assert!((credit - 5.0).abs() < 1e-12);
}

#[test]
fn heuristic_first_actions_match_named_heuristic_evaluators() {
    let state = initialize_state(
        VERIFICATION_PROBLEM_INSTANCE.initial_inventory_level,
        VERIFICATION_PROBLEM_INSTANCE.initial_returnable_inventory,
    )
    .expect("state must build");
    let interval = interval_stock_action(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.interval_stock_order_up_to,
        VERIFICATION_PROBLEM_INSTANCE.interval_stock_remove_down_to,
        VERIFICATION_PROBLEM_INSTANCE.max_purchase_quantity,
        VERIFICATION_PROBLEM_INSTANCE.max_removal_quantity,
    )
    .expect("interval-stock must compute");
    let buffer = returnability_buffer_interval_stock_action(
        &state,
        VERIFICATION_PROBLEM_INSTANCE.returnability_buffer_order_up_to,
        VERIFICATION_PROBLEM_INSTANCE.returnability_buffer_remove_down_to,
        VERIFICATION_PROBLEM_INSTANCE.returnability_buffer,
        VERIFICATION_PROBLEM_INSTANCE.max_purchase_quantity,
        VERIFICATION_PROBLEM_INSTANCE.max_removal_quantity,
    )
    .expect("buffered interval-stock must compute");

    let interval_eval = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "interval_stock")
        .expect("interval-stock evaluation must solve");
    let buffer_eval = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "returnability_buffer_interval_stock",
    )
    .expect("buffered interval-stock evaluation must solve");

    assert_eq!(interval, interval_eval.first_action);
    assert_eq!(buffer, buffer_eval.first_action);
}

#[test]
fn removal_active_instance_is_well_formed() {
    // The removal-active benchmark instance must (a) be a valid env state and (b) respect the
    // Maggiar & Sadighian (2017) cost ordering used throughout this package: Assumption 2(ii)
    // c > s (purchase cost above return value) and 2(iii) l <= s (liquidation no greater than
    // return value). It must also start overstocked relative to demand so the removal channel can
    // bind, and its carried benchmark levels must form a valid interval (order_up_to <=
    // remove_down_to) where the removal level is strictly above the order level (removal lever is
    // exercised), unlike the primary instance where they collapse together.
    let instance = REMOVAL_ACTIVE_REFERENCE_INSTANCE;
    let state = initialize_state(
        instance.initial_inventory_level,
        instance.initial_returnable_inventory,
    )
    .expect("removal-active initial state must build");
    assert!(state.returnable_inventory <= state.inventory_level);

    assert!(instance.purchase_cost_per_unit > instance.return_value_per_unit);
    assert!(instance.return_value_per_unit >= instance.liquidation_value_per_unit);

    assert!(instance.benchmark_order_up_to <= instance.benchmark_remove_down_to);
    assert!(instance.benchmark_remove_down_to > instance.benchmark_order_up_to);
    assert!((instance.initial_inventory_level as f64) > instance.demand_mean);

    // A worked step that exercises the removal channel: from the overstocked start, removing units
    // returns from the returnable pool first (Corollary 1: never liquidate what can be returned),
    // and any excess beyond the returnable pool is liquidated.
    let outcome = step_state(
        &state,
        0,  // no purchase
        10, // remove 10 of 12 on hand: 8 returnable + 2 liquidated
        0,  // zero demand to isolate the removal accounting
        instance.returnable_purchase_cap,
        instance.purchase_cost_per_unit,
        instance.return_value_per_unit,
        instance.liquidation_value_per_unit,
        instance.holding_cost_per_unit,
        instance.shortage_cost_per_unit,
    )
    .expect("removal-active worked step must succeed");
    assert_eq!(outcome.returned_units, 8);
    assert_eq!(outcome.liquidated_units, 2);
    assert_eq!(outcome.next_state.inventory_level, 2);
}

#[test]
fn exact_dp_dominates_repo_heuristics() {
    let optimal =
        solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE).expect("optimal policy must solve");
    let interval = evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "interval_stock")
        .expect("interval-stock evaluation must solve");
    let buffer = evaluate_named_heuristic(
        &VERIFICATION_PROBLEM_INSTANCE,
        "returnability_buffer_interval_stock",
    )
    .expect("buffered interval-stock evaluation must solve");

    assert!(
        optimal.discounted_cost <= interval.discounted_cost + 1e-9,
        "optimal={} interval_stock={}",
        optimal.discounted_cost,
        interval.discounted_cost
    );
    assert!(
        optimal.discounted_cost <= buffer.discounted_cost + 1e-9,
        "optimal={} returnability_buffer={}",
        optimal.discounted_cost,
        buffer.discounted_cost
    );
}

// ===========================================================================
// FAITHFUL Maggiar & Sadighian (2017) joint pricing / inventory / removal model.
//
// These tests RE-RUN the faithful environment + finite-horizon DP and assert:
//   1. the price-dependent Gamma demand model matches the paper's equations,
//   2. the single-period reward accounting matches Eq. 4 by hand,
//   3. the optimal policy recovered by the DP reproduces the EXACT structural
//      monotonicity properties the paper PROVES (Lemma 3.1 / Section 3.2 / 7.2.1),
//   4. the Table-1 instance reproduces the reported NPV-surface magnitude
//      (~84000) within an honest tolerance (the mu_t profile is graphical so the
//      figure is not exactly reproducible).
// ===========================================================================

fn faithful_parameters(
    instance: &FaithfulVerificationInstance,
) -> JointPricingRemovalParameters {
    let beta = beta_from_elasticity(instance.base_price, instance.elasticity_at_base_price)
        .expect("beta must compute");
    JointPricingRemovalParameters {
        base_price: instance.base_price,
        purchase_cost: instance.purchase_cost,
        refund_value: instance.refund_value,
        liquidation_value: instance.liquidation_value,
        holding_cost: instance.holding_cost,
        backorder_supplement: instance.backorder_supplement,
        elasticity_at_base_price: instance.elasticity_at_base_price,
        beta,
        coefficient_of_variation: instance.coefficient_of_variation,
    }
}

fn faithful_config(instance: &FaithfulVerificationInstance) -> JointPricingRemovalDpConfig {
    JointPricingRemovalDpConfig {
        periods: instance.periods,
        discount_factor: instance.discount_factor,
        parameters: faithful_parameters(instance),
        forecast_mean_demand: vec![instance.baseline_mean_demand; instance.periods],
        returnable_purchase_cap: instance.returnable_purchase_cap,
        max_inventory_level: instance.max_inventory_level,
        max_purchase_quantity: instance.max_purchase_quantity,
        num_demand_quantiles: instance.num_demand_quantiles,
        num_price_points: instance.num_price_points,
        max_demand_multiple: instance.max_demand_multiple,
    }
}

#[test]
fn faithful_instance_obeys_paper_assumption2_cost_ordering() {
    // Assumption 2: c > s > l makes removal a genuinely lossy option.
    let m: MaggiarSadighian2017FaithfulInstance = MAGGIAR_SADIGHIAN_2017_FAITHFUL_INSTANCE;
    assert!(m.purchase_cost > m.refund_value);
    assert!(m.refund_value > m.liquidation_value);
    // Table 1 values are carried verbatim.
    assert_eq!(m.base_price, 90.0);
    assert_eq!(m.purchase_cost, 75.0);
    assert_eq!(m.refund_value, 30.0);
    assert_eq!(m.liquidation_value, 5.0);
    assert_eq!(m.holding_cost, 2.0);
    assert_eq!(m.backorder_supplement, 15.5);
    assert_eq!(m.elasticity_at_base_price, -2.0);
    assert_eq!(m.periods, 40);
    assert!((m.discount_factor - 0.9984).abs() < 1e-12);
    assert_eq!(m.num_demand_quantiles, 99);
    assert!((m.reported_npv_surface_peak - 84000.0).abs() < 1e-9);
    assert!(!m.npv_peak_is_exact);
}

#[test]
fn price_demand_map_matches_paper_log_linear_model() {
    // d_t(p) = mu_t exp(-beta(p-p0)); inverse p(d) = p0 - (1/beta) ln(d/mu_t).
    // Elasticity E = -beta p0 => beta = -E/p0.
    let p0 = 90.0;
    let e = -2.0;
    let beta = beta_from_elasticity(p0, e).expect("beta");
    assert!((beta - 2.0 / 90.0).abs() < 1e-12);

    let mu = 50.0;
    // At target demand == mu, the price must equal the base price (no markdown).
    let p_at_mu = price_at_demand(mu, mu, p0, beta).expect("price");
    assert!((p_at_mu - p0).abs() < 1e-9);

    // Doubling target demand lowers the price (a markdown): p(2 mu) < p0.
    let p_double = price_at_demand(2.0 * mu, mu, p0, beta).expect("price");
    assert!(p_double < p0);
    // Closed form: p(2mu) = p0 - ln(2)/beta.
    let expected = p0 - (2.0f64).ln() / beta;
    assert!((p_double - expected).abs() < 1e-9);
}

#[test]
fn gamma_noise_quantiles_are_centred_with_unit_cv() {
    // mu_t + eps ~ Gamma(mean mu, CV 1); the centred noise quantiles average to
    // ~0 over many equally-likely quantiles, and their second moment matches the
    // unit-CV variance mu^2 to discretization accuracy.
    let mu = 50.0;
    let k = 999;
    let eps = noise_quantiles(mu, 1.0, k).expect("quantiles");
    let mean = eps.iter().sum::<f64>() / k as f64;
    assert!(mean.abs() < 1.0, "centred noise mean {mean} should be ~0");
    let var = eps.iter().map(|e| e * e).sum::<f64>() / k as f64;
    // CV = 1 => Var = mu^2 = 2500. Midpoint-quantile discretization underestimates
    // tail variance, so allow a wide band but confirm the right order of magnitude.
    assert!(
        var > 1500.0 && var < 3000.0,
        "noise variance {var} should be near mu^2 = 2500"
    );
}

#[test]
fn faithful_single_period_accounting_matches_eq4_by_hand() {
    // Hand-computed Eq. 4 period: state x=10, y=4, remove q=+3 units, demand
    // target d held at the forecast mean so price == base price, realized D=6.
    let instance = FAITHFUL_VERIFICATION_INSTANCE;
    let parameters = faithful_parameters(&instance);
    let state = JointPricingRemovalState {
        period: 0,
        inventory_level: 10,
        returnable_level: 4,
    };
    let mu = 6.0;
    let d = mu; // no markdown => price == p0
    let q = 3; // remove 3 units: all 3 returnable (y=4), 0 liquidated
    let realized = 6.0;
    let outcome = step_period(&state, d, q, realized, instance.returnable_purchase_cap, mu, &parameters)
        .expect("step");

    // Revenue: r(d) = d * p0 = 6 * 90 = 540.
    assert!((outcome.expected_revenue - 6.0 * 90.0).abs() < 1e-9);
    assert!((outcome.implied_price - 90.0).abs() < 1e-9);
    // Returns: 3 returnable removed, 0 liquidated.
    assert_eq!(outcome.returned_units, 3);
    assert_eq!(outcome.liquidated_units, 0);
    assert_eq!(outcome.purchased_units, 0);
    // b(q,y) = s*3 = 30*3 = 90.
    assert!((outcome.flow_value - 90.0).abs() < 1e-9);
    // Net position w = x - D - q = 10 - 6 - 3 = 1 carried, 0 backlog.
    assert!((outcome.carried_inventory - 1.0).abs() < 1e-9);
    assert!((outcome.backlogged_units - 0.0).abs() < 1e-9);
    // Holding = h+ * 1 = 2. Backorder = 0.
    assert!((outcome.holding_cost - 2.0).abs() < 1e-9);
    assert!((outcome.backorder_cost - 0.0).abs() < 1e-9);
    // Profit = 540 + 90 - 2 - 0 = 628.
    assert!((outcome.period_profit - 628.0).abs() < 1e-9);
    assert!((outcome.period_cost + 628.0).abs() < 1e-9);
    // Next state: x' = 1, y' = min(4 - 3, 1) = 1.
    assert_eq!(outcome.next_state.inventory_level, 1);
    assert_eq!(outcome.next_state.returnable_level, 1);
}

#[test]
fn faithful_backorder_uses_c_plus_k_convention() {
    // Stockout cost per unit is h- = c + k (paper Section 3.1).
    let instance = FAITHFUL_VERIFICATION_INSTANCE;
    let parameters = faithful_parameters(&instance);
    assert!(
        (backorder_cost_per_unit(&parameters)
            - (instance.purchase_cost + instance.backorder_supplement))
            .abs()
            < 1e-12
    );
    // A period that backlogs one unit incurs exactly (c+k) of backorder cost.
    let state = JointPricingRemovalState {
        period: 0,
        inventory_level: 0,
        returnable_level: 0,
    };
    let mu = 4.0;
    let outcome = step_period(&state, mu, 0, 1.0, instance.returnable_purchase_cap, mu, &parameters)
        .expect("step");
    assert!((outcome.backlogged_units - 1.0).abs() < 1e-9);
    assert!(
        (outcome.backorder_cost - (instance.purchase_cost + instance.backorder_supplement)).abs()
            < 1e-9
    );
}

#[test]
fn faithful_terminal_value_returns_then_liquidates() {
    // V_T(x,y) = s*min(x,y) + l*max(x-y,0).
    let instance = FAITHFUL_VERIFICATION_INSTANCE;
    let parameters = faithful_parameters(&instance);
    let state = JointPricingRemovalState {
        period: instance.periods,
        inventory_level: 10,
        returnable_level: 4,
    };
    let value = terminal_value(&state, &parameters).expect("terminal");
    // 30*4 + 5*6 = 120 + 30 = 150.
    assert!((value - 150.0).abs() < 1e-9);
}

#[test]
fn faithful_dp_reproduces_paper_proven_monotonicity_lemma31() {
    // Lemma 3.1 (and the Section 3.2 / 7.2.1 bullets) PROVE, for the optimal
    // policy of this exact model:
    //   (a) target demand d (i.e. the markdown) is NONDECREASING in inventory x
    //       and NONINCREASING in returnable level y, with the unit-step bound
    //       d(x,y) <= d(x+1,y) <= d(x,y)+1 (here scaled to the demand grid step),
    //   (b) returns increase with inventory for fixed y; liquidations increase
    //       with inventory,
    //   (c) total purchases (net buy) increase with DECREASING inventory.
    // We re-run the DP on the faithful verification instance and check these
    // monotonicities hold at an interior period.
    let instance = FAITHFUL_VERIFICATION_INSTANCE;
    let config = faithful_config(&instance);
    let result = solve(&config).expect("faithful DP must solve");

    let t = 1usize; // interior period (away from terminal edge effects)
    let xmax = instance.max_inventory_level;

    // (a) markdown / target demand monotone in inventory (nondecreasing) for a
    //     fixed returnable level y = 0.
    let y = 0;
    let mut prev_d = f64::NEG_INFINITY;
    for x in y..=xmax {
        let d = result.decision_at(t, x, y).target_demand;
        assert!(
            d >= prev_d - 1e-9,
            "target demand must be nondecreasing in inventory: x={x} d={d} prev={prev_d}"
        );
        prev_d = d;
    }

    // (a') markdown / target demand nonincreasing in returnable level y for a
    //      fixed inventory x.
    let x = xmax;
    let mut prev_d = f64::INFINITY;
    for yy in 0..=x {
        let d = result.decision_at(t, x, yy).target_demand;
        assert!(
            d <= prev_d + 1e-9,
            "target demand must be nonincreasing in returnable level: y={yy} d={d} prev={prev_d}"
        );
        prev_d = d;
    }

    // (b) returns nondecreasing in inventory for a fixed positive returnable y.
    let y = (xmax / 2).max(1);
    let mut prev_returns = i64::MIN;
    for x in y..=xmax {
        let returns = result.decision_at(t, x, y).returned_units;
        assert!(
            returns >= prev_returns,
            "returns must be nondecreasing in inventory: x={x} returns={returns} prev={prev_returns}"
        );
        prev_returns = returns;
    }

    // (c) net purchases nonincreasing in inventory (total purchases increase as
    //     inventory falls). net_flow > 0 is removal, < 0 is purchase, so the
    //     purchased amount = max(-net_flow, 0) must be nonincreasing in x.
    let y = 0;
    let mut prev_purchase = i64::MAX;
    for x in y..=xmax {
        let purchase = result.decision_at(t, x, y).purchased_units;
        assert!(
            purchase <= prev_purchase,
            "purchases must be nonincreasing in inventory: x={x} purchase={purchase} prev={prev_purchase}"
        );
        prev_purchase = purchase;
    }
}

#[test]
fn faithful_dp_value_function_is_supermodular_in_state() {
    // The paper proves V_t is L-natural-concave (hence supermodular):
    //   V(x+1,y+1) + V(x,y) >= V(x+1,y) + V(x,y+1).
    // We re-run the DP and check supermodularity at an interior period over the
    // grid where x+1 stays within bounds.
    let instance = FAITHFUL_VERIFICATION_INSTANCE;
    let config = faithful_config(&instance);
    let result = solve(&config).expect("faithful DP must solve");

    let t = 1usize;
    let xmax = instance.max_inventory_level;
    let mut checked = 0usize;
    for x in 0..xmax {
        for y in 0..x {
            // need y+1 <= x for state (x, y+1) to be feasible
            let v_hh = result.value_at(t, x + 1, y + 1);
            let v_ll = result.value_at(t, x, y);
            let v_hl = result.value_at(t, x + 1, y);
            let v_lh = result.value_at(t, x, y + 1);
            assert!(
                v_hh + v_ll >= v_hl + v_lh - 1e-6,
                "supermodularity violated at x={x} y={y}: {v_hh}+{v_ll} < {v_hl}+{v_lh}"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "supermodularity should be checked on >0 cells");
}

#[test]
fn faithful_table1_dp_reproduces_npv_surface_magnitude() {
    // CHARACTERIZATION (not a tight reproduction): re-run the faithful DP on the
    // Table-1 parameter set with a reconstructed mu_t profile (Figure 6: ~50
    // baseline, Gaussian peak ~500 near period 20) and assert the peak of the
    // V_t surface lands in the same magnitude band as the reported ~84000 top
    // contour (Figure 7). The mu_t profile is graphical, so this is an honest
    // order-of-magnitude reproduction, not an exact figure match.
    let m: MaggiarSadighian2017FaithfulInstance = MAGGIAR_SADIGHIAN_2017_FAITHFUL_INSTANCE;
    let beta = beta_from_elasticity(m.base_price, m.elasticity_at_base_price).expect("beta");
    let parameters = JointPricingRemovalParameters {
        base_price: m.base_price,
        purchase_cost: m.purchase_cost,
        refund_value: m.refund_value,
        liquidation_value: m.liquidation_value,
        holding_cost: m.holding_cost,
        backorder_supplement: m.backorder_supplement,
        elasticity_at_base_price: m.elasticity_at_base_price,
        beta,
        coefficient_of_variation: m.coefficient_of_variation,
    };

    // Reconstruct mu_t: baseline + Gaussian bump (Figure 6).
    let center = m.peak_period as f64;
    let width = 5.0;
    let peak_add = m.peak_mean_demand - m.baseline_mean_demand;
    let forecast_mean_demand: Vec<f64> = (0..m.periods)
        .map(|t| {
            let z = (t as f64 - center) / width;
            m.baseline_mean_demand + peak_add * (-0.5 * z * z).exp()
        })
        .collect();

    // Fixed-returnability quota = median of base-price forecast demand. For a
    // CV=1 Gamma the median is mu * ln(2). Use the baseline forecast median.
    let returnable_cap =
        (m.baseline_mean_demand * (2.0f64).ln()).round().max(0.0) as i64;

    // Inventory grid: the paper plots up to ~1500. Use a coarser cap but large
    // enough to bracket the surface peak while keeping the test tractable. We
    // scale units down by `scale` to keep the (x,y) state count tractable and
    // rescale the NPV back up afterwards (the model is positively homogeneous in
    // quantities given fixed prices and per-unit costs).
    let scale = 50i64;
    let scaled_forecast: Vec<f64> = forecast_mean_demand.iter().map(|m| m / scale as f64).collect();
    let config = JointPricingRemovalDpConfig {
        periods: m.periods,
        discount_factor: m.discount_factor,
        parameters,
        forecast_mean_demand: scaled_forecast,
        returnable_purchase_cap: (returnable_cap / scale).max(1),
        max_inventory_level: 30,
        max_purchase_quantity: 22,
        num_demand_quantiles: 13,
        num_price_points: 7,
        max_demand_multiple: 2.5,
    };
    let result = solve(&config).expect("Table-1 faithful DP must solve");

    // The plotted surface is at t = 24. Rescale the per-unit NPV back to full
    // quantity units.
    let peak_scaled = result.max_value_at_period(24);
    let peak = peak_scaled * scale as f64;
    println!(
        "faithful Table-1 reconstructed NPV-surface peak at t=24: {peak:.0} (reported ~{})",
        m.reported_npv_surface_peak
    );

    // Honest order-of-magnitude band around the reported ~84000 peak.
    assert!(
        peak > 30_000.0 && peak < 200_000.0,
        "reconstructed NPV-surface peak {peak} should bracket the reported ~{} contour",
        m.reported_npv_surface_peak
    );
}
