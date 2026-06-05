// ============================================================================
// tests/verification.rs  (faithful model)
//
// EXECUTING literature-verification for the ameliorating-inventory family.
//
// Each test below RE-SOLVES the perfect-information (steady-state, expected-
// value) LP from the carried companion dataset and asserts the freshly computed
// average-profit upper bound reproduces the value PRINTED in the Pahr & Grunow
// (2025) companion repository's `problem_configurations/<instance>/upper_bound.json`
// within the stated tolerance. This is a reproduction, not a frozen snapshot.
//
// Anchors:
//   - spirits_0001 : published max_reward = 1991.9344293376805
//   - port_wine    : published max_reward = 2444.8010643781136
// ============================================================================

use crate::problems::ameliorating_inventory::average_profit_blending_env::{
    initialize_state, step_state, AverageProfitBlendingConfig,
};
use crate::problems::ameliorating_inventory::lp_dataset_loader::{
    load_port_wine, load_spirits_0001, LoadedLpDataset,
};
use crate::problems::ameliorating_inventory::perfect_information_lp::solve_upper_bound;
use crate::problems::ameliorating_inventory::references::{
    PORT_WINE_REFERENCE_INSTANCE, PORT_WINE_VERIFICATION_ANCHOR, PRIMARY_REFERENCE_INSTANCE,
    REFERENCE_INSTANCES, VERIFICATION_PROBLEM_INSTANCE,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn reference_catalogue_has_expected_shape() {
    assert_eq!(REFERENCE_INSTANCES.len(), 2);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.name, "pahr_grunow2025_spirits_0001");
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.num_ages, 10);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.num_products, 3);
    assert_eq!(PRIMARY_REFERENCE_INSTANCE.target_ages, &[2, 4, 6]);
    assert!(!PRIMARY_REFERENCE_INSTANCE.allow_blending);
    assert!(PRIMARY_REFERENCE_INSTANCE.literature_verified);
    assert_eq!(
        VERIFICATION_PROBLEM_INSTANCE.instance_name,
        PRIMARY_REFERENCE_INSTANCE.name
    );
    assert!(PORT_WINE_REFERENCE_INSTANCE.allow_blending);
    assert_eq!(PORT_WINE_REFERENCE_INSTANCE.num_ages, 25);
}

#[test]
fn dataset_inputs_match_reference_metadata() {
    let LoadedLpDataset { inputs, anchor } = load_spirits_0001();
    assert_eq!(inputs.instance, "spirits_0001");
    assert_eq!(inputs.num_ages, PRIMARY_REFERENCE_INSTANCE.num_ages);
    assert_eq!(inputs.num_products, PRIMARY_REFERENCE_INSTANCE.num_products);
    assert_eq!(inputs.target_ages, PRIMARY_REFERENCE_INSTANCE.target_ages);
    assert!((inputs.max_inventory - PRIMARY_REFERENCE_INSTANCE.max_inventory).abs() < 1e-12);
    assert!((inputs.holding_costs - PRIMARY_REFERENCE_INSTANCE.holding_cost).abs() < 1e-12);
    assert!((inputs.evaporation - PRIMARY_REFERENCE_INSTANCE.evaporation).abs() < 1e-12);
    // expected_revenue/slope tables aligned with the production grid
    for p in 0..inputs.num_products {
        assert_eq!(inputs.expected_revenue[p].len(), inputs.slope[p].len());
        assert!(inputs.expected_revenue[p].len() > 1000);
    }
    assert!(
        (anchor.max_reward - PRIMARY_REFERENCE_INSTANCE.published_max_reward).abs() < 1e-9,
        "dataset-carried published bound must match reference metadata"
    );
}

/// PRIMARY EXECUTING REPRODUCTION: re-solve the spirits_0001 perfect-information
/// LP and reproduce the published average-profit upper bound.
#[test]
fn spirits_0001_perfect_information_upper_bound_reproduces_published_max_reward() {
    let LoadedLpDataset { inputs, anchor } = load_spirits_0001();
    let solution = solve_upper_bound(&inputs);

    let gap = (solution.max_reward - anchor.max_reward).abs();
    println!(
        "spirits_0001 perfect-information LP: re-solved max_reward = {:.10}, \
         published = {:.10}, gap = {:.3e}",
        solution.max_reward, anchor.max_reward, gap
    );
    assert!(
        gap < VERIFICATION_PROBLEM_INSTANCE.max_reward_tolerance,
        "re-solved upper bound {} must reproduce published {} within {} (gap {})",
        solution.max_reward,
        anchor.max_reward,
        VERIFICATION_PROBLEM_INSTANCE.max_reward_tolerance,
        gap
    );

    // sanity: the recovered purchase/production volumes are in a plausible range
    assert!(solution.purchasing > 0.0 && solution.purchasing <= inputs.max_inventory);
    assert_eq!(solution.production.len(), inputs.num_products);
    assert_eq!(solution.inventory_position.len(), inputs.num_ages);
}

/// SECONDARY EXECUTING REPRODUCTION: port_wine perfect-information upper bound.
#[test]
fn port_wine_perfect_information_upper_bound_reproduces_published_max_reward() {
    let LoadedLpDataset { inputs, anchor } = load_port_wine();
    let solution = solve_upper_bound(&inputs);

    let gap = (solution.max_reward - anchor.max_reward).abs();
    println!(
        "port_wine perfect-information LP: re-solved max_reward = {:.10}, \
         published = {:.10}, gap = {:.3e}",
        solution.max_reward, anchor.max_reward, gap
    );
    assert!(
        gap < PORT_WINE_VERIFICATION_ANCHOR.max_reward_tolerance,
        "re-solved upper bound {} must reproduce published {} within {} (gap {})",
        solution.max_reward,
        anchor.max_reward,
        PORT_WINE_VERIFICATION_ANCHOR.max_reward_tolerance,
        gap
    );
}

/// COMPANION ADDITION: spirits_0002 is spirits_0001 with BLENDING ENABLED.
/// Re-solve its perfect-information LP and reproduce the published upper bound
/// from problem_configurations/spirits_0002/upper_bound.json.
#[test]
fn spirits_0002_blending_upper_bound_reproduces_published_max_reward() {
    use crate::problems::ameliorating_inventory::lp_dataset_loader::load_spirits_0002;
    let LoadedLpDataset { inputs, anchor } = load_spirits_0002();
    assert!(inputs.allow_blending, "spirits_0002 must have blending enabled");
    assert!((inputs.max_inventory - 50.0).abs() < 1e-12);
    let solution = solve_upper_bound(&inputs);
    let gap = (solution.max_reward - anchor.max_reward).abs();
    println!(
        "spirits_0002 perfect-information LP: re-solved max_reward = {:.10}, \
         published = {:.10}, gap = {:.3e}",
        solution.max_reward, anchor.max_reward, gap
    );
    assert!(
        gap < 1e-3,
        "re-solved upper bound {} must reproduce published {} within 1e-3 (gap {})",
        solution.max_reward,
        anchor.max_reward,
        gap
    );
}

/// COMPANION ADDITION: spirits_1002 is the processing-capacity-constrained
/// variant (blending ON, maxInventory = 30). Re-solve its perfect-information
/// LP and reproduce problem_configurations/spirits_1002/upper_bound.json.
#[test]
fn spirits_1002_capacity_upper_bound_reproduces_published_max_reward() {
    use crate::problems::ameliorating_inventory::lp_dataset_loader::load_spirits_1002;
    let LoadedLpDataset { inputs, anchor } = load_spirits_1002();
    assert!(inputs.allow_blending, "spirits_1002 must have blending enabled");
    assert!((inputs.max_inventory - 30.0).abs() < 1e-12);
    let solution = solve_upper_bound(&inputs);
    let gap = (solution.max_reward - anchor.max_reward).abs();
    println!(
        "spirits_1002 perfect-information LP: re-solved max_reward = {:.10}, \
         published = {:.10}, gap = {:.3e}",
        solution.max_reward, anchor.max_reward, gap
    );
    assert!(
        gap < 1e-3,
        "re-solved upper bound {} must reproduce published {} within 1e-3 (gap {})",
        solution.max_reward,
        anchor.max_reward,
        gap
    );
}

/// MECHANICS CHECK: the faithful average-profit env runs a long trajectory and
/// its realised average profit stays at or below the perfect-information upper
/// bound (the bound's defining property), and the step accounting is internally
/// consistent.
#[test]
fn average_profit_env_respects_upper_bound_and_accounting() {
    let LoadedLpDataset { inputs, anchor } = load_spirits_0001();

    // decay_cov for spirits_0001 is 0.8 across all ages (companion default).
    let decay_cov = vec![0.8; inputs.num_ages];
    let config = AverageProfitBlendingConfig {
        num_ages: inputs.num_ages,
        num_products: inputs.num_products,
        target_ages: inputs.target_ages.clone(),
        max_inventory: inputs.max_inventory,
        evaporation: inputs.evaporation,
        decay_mean: inputs.decay_mean.clone(),
        decay_cov,
        holding_costs: inputs.holding_costs,
        outdating_costs: inputs.outdating_costs,
        decay_salvage: inputs.decay_salvage.clone(),
        allow_blending: inputs.allow_blending,
        blending_range: inputs.blending_range,
        price_mean: inputs.price_mean,
        price_std: inputs.price_std,
        price_truncation: inputs.price_truncation,
        demand_means: vec![10.0, 7.0, 5.0],
        demand_covs: vec![0.25, 0.25, 0.25],
        sales_means: vec![250.0, 350.0, 500.0],
        sales_covs: vec![0.1, 0.1, 0.1],
        correlation_demand_salesprice: vec![0.5, 0.5, 0.5],
        production_step_size: inputs.production_step_size,
        sales_bound: inputs.sales_bound.clone(),
        expected_revenue: inputs.expected_revenue.clone(),
    };

    // start from the LP's steady-state inventory position
    let mut state = initialize_state(&config, &solve_upper_bound_inventory(&inputs));
    let mut rng = StdRng::seed_from_u64(20250604);
    let periods = 2000usize;
    let mut total_reward = 0.0f64;

    for _ in 0..periods {
        // myopic order-up-to: buy toward the LP's age-0 steady-state level
        let target_age0 = state.inventory_position.first().copied().unwrap_or(0.0);
        let purchase = (config.max_inventory * 0.5 - target_age0).max(0.0);
        let outcome = step_state(&mut rng, &config, &state, purchase);

        // step-level accounting identity
        let recomputed = outcome.revenue - outcome.purchase_cost - outcome.holding_cost
            + outcome.decay_salvage_credit
            - outcome.outdating_cost;
        assert!(
            (recomputed - outcome.reward).abs() < 1e-9,
            "per-period reward accounting must be internally consistent"
        );
        // inventory never negative and respects per-age capacity logic
        assert!(outcome
            .next_state
            .inventory_position
            .iter()
            .all(|&v| v >= -1e-9));

        total_reward += outcome.reward;
        state = outcome.next_state;
    }

    let average_profit = total_reward / periods as f64;
    println!(
        "spirits_0001 faithful env: realised average profit over {} periods = {:.4} \
         (perfect-information upper bound = {:.4})",
        periods, average_profit, anchor.max_reward
    );
    // defining property of the bound: realised average profit <= upper bound
    assert!(
        average_profit <= anchor.max_reward + 1e-6,
        "realised average profit {} must not exceed the perfect-information bound {}",
        average_profit,
        anchor.max_reward
    );
}

/// Helper: recover the LP's steady-state inventory position for env warm-start.
fn solve_upper_bound_inventory(
    inputs: &crate::problems::ameliorating_inventory::perfect_information_lp::PerfectInformationLpInputs,
) -> Vec<f64> {
    solve_upper_bound(inputs).inventory_position
}
