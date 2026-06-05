//! Vanilla lost-sales heuristics: Myopic-1, Myopic-2, and Standard Vector Base
//! Stock (SVBS).
//!
//! This module is a first-class, reusable home for the three classical
//! lost-sales ordering heuristics and a rollout-based average-cost evaluator. It
//! is independent of the flownet verification/targets machinery: its job is
//! "given a per-instance config, compute order quantities and roll out average
//! cost". The flownet verification module reuses these types rather than holding
//! its own copy.
//!
//! ## Algorithmic overview
//!
//! State. The lost-sales pipeline state is `[x_0, ..., x_{L-1}]` where `x_0` is
//! on-hand inventory plus the order arriving this period and `x_j` (j >= 1) is
//! the order that arrives in `j` periods. Ordering `z` appends a slot. After
//! demand `d`, on-hand becomes `(x_0 - d)^+ + x_1` and later slots shift down.
//!
//! One-period cost. `c_p * y + c_h * E[(y - D)^+] + c_s * E[(D - y)^+]` over the
//! truncated demand support, for inventory position `y`.
//!
//! Demand law. The order-quantity math requires a closed-form per-period demand
//! PMF/CDF. IID Poisson and Geometric are used directly. Markov-Modulated
//! Poisson (MMPP2) demand is supported via its **stationary marginal**: the
//! mixture `prob_low * Poisson(lambda_low) + prob_high * Poisson(lambda_high)`,
//! where `prob_low`/`prob_high` are the stationary regime occupancies. The
//! multi-period demand used by SVBS is the self-convolution of this marginal
//! (treating periods as independent). This ignores regime autocorrelation and is
//! an approximation used only for *choosing order quantities*; the rollout cost
//! is always measured on the true autocorrelated MMPP2 process. Consequently the
//! MMPP2 heuristic mean costs are repo-computed, not literature numbers. See
//! `demand_support.rs` for the construction.
//!
//! Lookahead value `q_l(state, l)`. Expected discounted cost-to-go over an
//! `l`-period lookahead with NO further ordering (myopic continuation); depth-0
//! is the one-period cost of slot 0, deeper levels take the demand expectation
//! of `q_{l-1}` over the advanced pipeline, discounted by `beta`. All values are
//! memoised.
//!
//! Order-quantity search (`best_quantity`). Each heuristic scans
//! `z = 0, 1, 2, ...` and stops at the first local minimum of its action value,
//! capped at `order_search_upper_bound`.
//!
//! Myopic-1. Minimise the lead-time-deep lookahead value of ordering `z`
//! (`q_l_from_state_action`), i.e. a single newsvendor lookahead assuming no
//! future ordering.
//!
//! Myopic-2. Minimise `q_l_from_state_action(z)` PLUS a discounted one-step
//! continuation equal to the expected Myopic-1 quantity chosen next period. This
//! two-period view typically beats Myopic-1.
//!
//! SVBS. For each pipeline position `l = 0..=L`, set base-stock level `S_l` to
//! the critical-fractile quantile of `(L - l + 1)`-period demand, where the
//! critical fractile is `(c_p + c_h) / (c_h + c_s)`. Order
//! `min_l (S_l - sum of pipeline from position l)`, clamped to
//! `[0, order_search_upper_bound]`.
//!
//! ## Average cost
//!
//! `evaluate_heuristic_policy` simulates `horizon` periods under a policy and
//! returns the mean per-period cost after discarding the first
//! `floor(warm_up_periods_ratio * horizon)` periods (warm-up), matching the
//! learned-policy rollout convention.
//!
//! ## Verification
//!
//! `VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG` is the canonical lost-sales
//! instance (lead time 4, holding 1, shortage 4, IID Poisson(5)). Rolling out
//! the three heuristics on it reproduces the literature average costs:
//! myopic2 ~= 4.82, myopic1 ~= 5.06, svbs ~= 5.83, with myopic2 the best of the
//! three. The unit tests below assert these numbers; see `README.md` for how to
//! run them.

pub mod demand_support;
pub mod evaluator;
pub mod policy_kind;
pub mod rollout;

use crate::problems::lost_sales::demand::{LostSalesDemandConfig, LostSalesDemandKind};

pub use evaluator::{
    validate_heuristic_config, LostSalesHeuristicEvaluator, LostSalesHeuristicVerificationConfig,
};
pub use policy_kind::LostSalesHeuristicPolicyKind;
pub use rollout::{
    evaluate_heuristic_policy, measurement_from_observed_mean_cost, PolicyPerformanceMeasurement,
};

/// Reference name for the canonical vanilla lost-sales verification instance.
pub const VANILLA_L4_P4_POISSON5_VERIFICATION_REFERENCE: &str = "vanilla_l4_p4_poisson5";
/// Rollout horizon used to obtain a stable steady-state cost estimate.
pub const VANILLA_L4_P4_POISSON5_VERIFICATION_HORIZON: usize = 100_000;
/// Fixed RNG seed for reproducible verification rollouts.
pub const VANILLA_L4_P4_POISSON5_VERIFICATION_SEED: u64 = 123;

const HEURISTIC_DISCOUNT_FACTOR: f64 = 0.995;

/// Canonical lost-sales instance: lead time 4, holding 1, shortage 4, IID
/// Poisson(5). Rolling out the heuristics on this instance reproduces the
/// trusted literature average costs (myopic2 ~= 4.82, myopic1 ~= 5.06,
/// svbs ~= 5.83).
pub const VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG: LostSalesHeuristicVerificationConfig =
    LostSalesHeuristicVerificationConfig {
        reference_name: VANILLA_L4_P4_POISSON5_VERIFICATION_REFERENCE,
        horizon: VANILLA_L4_P4_POISSON5_VERIFICATION_HORIZON,
        seed: VANILLA_L4_P4_POISSON5_VERIFICATION_SEED,
        warm_up_periods_ratio: 0.2,
        order_search_upper_bound: 200,
        lead_time: 4,
        holding_cost: 1.0,
        shortage_cost: 4.0,
        procurement_cost: 0.0,
        fixed_order_cost: 0.0,
        heuristic_discount_factor: HEURISTIC_DISCOUNT_FACTOR,
        demand_config: LostSalesDemandConfig {
            kind: LostSalesDemandKind::Poisson,
            demand_rate: 5.0,
            demand_lambda_low: 0.0,
            demand_lambda_high: 0.0,
            demand_p00: 0.0,
            demand_p11: 0.0,
        },
    };

#[cfg(test)]
mod tests {
    use super::{
        evaluate_heuristic_policy, LostSalesHeuristicPolicyKind, LostSalesDemandConfig,
        LostSalesDemandKind, LostSalesHeuristicVerificationConfig, HEURISTIC_DISCOUNT_FACTOR,
        VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG,
    };

    // Trusted literature average costs for the canonical vanilla instance.
    const EXPECTED_MYOPIC1_MEAN_COST: f64 = 5.06;
    const EXPECTED_MYOPIC2_MEAN_COST: f64 = 4.82;
    const EXPECTED_SVBS_MEAN_COST: f64 = 5.83;
    // Absolute tolerance consistent with the flownet verification targets.
    const MEAN_COST_TOLERANCE: f64 = 0.12;

    #[test]
    fn vanilla_heuristic_mean_costs_match_literature_numbers() -> Result<(), String> {
        let myopic1 = evaluate_heuristic_policy(
            VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG,
            LostSalesHeuristicPolicyKind::Myopic1,
        )?;
        let myopic2 = evaluate_heuristic_policy(
            VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG,
            LostSalesHeuristicPolicyKind::Myopic2,
        )?;
        let svbs = evaluate_heuristic_policy(
            VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG,
            LostSalesHeuristicPolicyKind::StandardVectorBaseStock,
        )?;

        assert!(
            (myopic1.mean_cost - EXPECTED_MYOPIC1_MEAN_COST).abs() <= MEAN_COST_TOLERANCE,
            "myopic1 mean cost {} not within {} of {}",
            myopic1.mean_cost,
            MEAN_COST_TOLERANCE,
            EXPECTED_MYOPIC1_MEAN_COST
        );
        assert!(
            (myopic2.mean_cost - EXPECTED_MYOPIC2_MEAN_COST).abs() <= MEAN_COST_TOLERANCE,
            "myopic2 mean cost {} not within {} of {}",
            myopic2.mean_cost,
            MEAN_COST_TOLERANCE,
            EXPECTED_MYOPIC2_MEAN_COST
        );
        assert!(
            (svbs.mean_cost - EXPECTED_SVBS_MEAN_COST).abs() <= MEAN_COST_TOLERANCE,
            "svbs mean cost {} not within {} of {}",
            svbs.mean_cost,
            MEAN_COST_TOLERANCE,
            EXPECTED_SVBS_MEAN_COST
        );

        Ok(())
    }

    #[test]
    fn vanilla_heuristic_ordering_holds_myopic2_is_best() -> Result<(), String> {
        let myopic1 = evaluate_heuristic_policy(
            VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG,
            LostSalesHeuristicPolicyKind::Myopic1,
        )?;
        let myopic2 = evaluate_heuristic_policy(
            VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG,
            LostSalesHeuristicPolicyKind::Myopic2,
        )?;
        let svbs = evaluate_heuristic_policy(
            VANILLA_L4_P4_POISSON5_VERIFICATION_CONFIG,
            LostSalesHeuristicPolicyKind::StandardVectorBaseStock,
        )?;

        assert!(
            myopic2.mean_cost <= myopic1.mean_cost,
            "expected myopic2 ({}) <= myopic1 ({})",
            myopic2.mean_cost,
            myopic1.mean_cost
        );
        assert!(
            myopic2.mean_cost <= svbs.mean_cost,
            "expected myopic2 ({}) <= svbs ({})",
            myopic2.mean_cost,
            svbs.mean_cost
        );
        Ok(())
    }

    // MMPP2 "lit_mmpp2_pos_p4_l4" parameters (mean demand 5): positively
    // autocorrelated regime switching with lambda_low=3, lambda_high=7,
    // p00=p11=0.9. Lead time 4, holding 1, shortage 4 — same costs as the
    // vanilla instance. The heuristic order quantities are computed on the
    // stationary marginal; the rollout cost is measured on the true MMPP2
    // process. These numbers are repo-computed (not literature).
    #[test]
    fn mmpp2_heuristics_run_and_return_finite_positive_costs() -> Result<(), String> {
        let config = LostSalesHeuristicVerificationConfig {
            reference_name: "lit_mmpp2_pos_p4_l4",
            // Short horizon: this test only checks the heuristics complete and
            // return sane numbers under MMPP2, not a precise steady-state value.
            horizon: 20_000,
            seed: 123,
            warm_up_periods_ratio: 0.2,
            order_search_upper_bound: 200,
            lead_time: 4,
            holding_cost: 1.0,
            shortage_cost: 4.0,
            procurement_cost: 0.0,
            fixed_order_cost: 0.0,
            heuristic_discount_factor: HEURISTIC_DISCOUNT_FACTOR,
            demand_config: LostSalesDemandConfig {
                kind: LostSalesDemandKind::MarkovModulatedPoisson2,
                demand_rate: 5.0,
                demand_lambda_low: 3.0,
                demand_lambda_high: 7.0,
                demand_p00: 0.9,
                demand_p11: 0.9,
            },
        };

        let myopic1 =
            evaluate_heuristic_policy(config, LostSalesHeuristicPolicyKind::Myopic1)?;
        let myopic2 =
            evaluate_heuristic_policy(config, LostSalesHeuristicPolicyKind::Myopic2)?;
        let svbs = evaluate_heuristic_policy(
            config,
            LostSalesHeuristicPolicyKind::StandardVectorBaseStock,
        )?;

        // All three must complete and return finite, strictly positive costs.
        for (name, cost) in [
            ("myopic1", myopic1.mean_cost),
            ("myopic2", myopic2.mean_cost),
            ("svbs", svbs.mean_cost),
        ] {
            assert!(
                cost.is_finite() && cost > 0.0,
                "{name} MMPP2 mean cost {cost} must be finite and positive"
            );
        }

        // Sanity check on ordering. Under MMPP2 the regime autocorrelation is
        // ignored by the order-quantity math, so the strict literature ordering
        // need not hold exactly; we only assert that Myopic-2 is no worse than
        // Myopic-1, which is the robust comparison (Myopic-2 strictly extends the
        // Myopic-1 lookahead). SVBS ordering is intentionally not asserted.
        assert!(
            myopic2.mean_cost <= myopic1.mean_cost + 0.25,
            "expected myopic2 ({}) to be roughly <= myopic1 ({}) under MMPP2",
            myopic2.mean_cost,
            myopic1.mean_cost
        );
        Ok(())
    }
}
