// ============================================================================
// issuance_blending_lp.rs
//
// PURPOSE
// -------
// Single-period blending issuance LP, the per-step subproblem the companion env
// solves with Gurobi inside `step_continuous_issuance_lp`. Given the current
// on-hand inventory by age class, choose issuance volumes iss[p,a] >= 0 that
//   - never exceed what is on hand in each age class:
//         sum_p iss[p,a] <= inventory[a]
//   - respect each product's TARGET-AGE blending rule, i.e. the issued blend's
//     volume-weighted (post-evaporation) mean age meets/exceeds the target:
//         sum_a iss[p,a]*a*evap[a] >= targetAge[p] * sum_a iss[p,a]*evap[a]
//     (young + old stock may be combined when blending is allowed),
//   - when blending is NOT allowed, draw nothing younger than the target age:
//         sum_{a<targetAge[p]} iss[p,a] = 0,
//   - when a blendingRange is set, draw only within [target-range, target+range],
//   - produce at most sales_bound[p] of effective (post-evaporation) volume:
//         sum_a iss[p,a]*evap[a] <= sales_bound[p].
//
// OBJECTIVE
// ---------
// Maximise total expected sales value of issued (post-evaporation) volume,
//   sum_p unit_revenue[p] * sum_a iss[p,a]*evap[a],
// which prioritises issuing scarce stock to the highest-value products while
// honouring the maturation (target-age) constraints. This mirrors the intent of
// the companion issuance LP (maximise revenue contribution of produced volume),
// reduced to a single decision period.
//
// The env consumes the resulting plan to compute production volumes and the
// post-issuance inventory; the period revenue itself is then scored from the
// companion's expected-revenue table (see `average_profit_blending_env.rs`).
// ============================================================================

use microlp::{ComparisonOp, OptimizationDirection, Problem};

/// Specification of the single-period issuance LP.
#[derive(Clone, Debug)]
pub struct IssuanceLpSpec {
    pub num_ages: usize,
    pub num_products: usize,
    pub target_ages: Vec<usize>,
    pub allow_blending: bool,
    pub blending_range: Option<usize>,
    pub evap_remains: Vec<f64>,
    pub sales_bound: Vec<f64>,
    /// Per-product marginal value used to prioritise issuance.
    pub unit_revenue: Vec<f64>,
}

/// Solve the single-period issuance LP and return iss[p][a].
pub fn solve_single_period_issuance(spec: &IssuanceLpSpec, inventory: &[f64]) -> Vec<Vec<f64>> {
    let a_n = spec.num_ages;
    let p_n = spec.num_products;

    let mut lp = Problem::new(OptimizationDirection::Maximize);

    // iss[p][a], objective coeff = unit_revenue[p]*evap[a] (value of issuing)
    let mut iss: Vec<Vec<microlp::Variable>> = Vec::with_capacity(p_n);
    for p in 0..p_n {
        let row: Vec<microlp::Variable> = (0..a_n)
            .map(|a| {
                let cap = inventory[a].max(0.0);
                lp.add_var(spec.unit_revenue[p] * spec.evap_remains[a], (0.0, cap))
            })
            .collect();
        iss.push(row);
    }

    // per-age availability: sum_p iss[p,a] <= inventory[a]
    for a in 0..a_n {
        let terms: Vec<(microlp::Variable, f64)> =
            (0..p_n).map(|p| (iss[p][a], 1.0)).collect();
        lp.add_constraint(&terms, ComparisonOp::Le, inventory[a].max(0.0));
    }

    for p in 0..p_n {
        // production cap: sum_a iss[p,a]*evap[a] <= sales_bound[p]
        let prod_terms: Vec<(microlp::Variable, f64)> =
            (0..a_n).map(|a| (iss[p][a], spec.evap_remains[a])).collect();
        lp.add_constraint(&prod_terms, ComparisonOp::Le, spec.sales_bound[p]);

        // target age (mean blend age >= target):
        //   sum_a iss[p,a]*evap[a]*(target - a) <= 0
        let target_terms: Vec<(microlp::Variable, f64)> = (0..a_n)
            .map(|a| {
                let coeff =
                    (spec.target_ages[p] as f64) * spec.evap_remains[a] - (a as f64) * spec.evap_remains[a];
                (iss[p][a], coeff)
            })
            .collect();
        lp.add_constraint(&target_terms, ComparisonOp::Le, 0.0);

        // blending restrictions
        if !spec.allow_blending {
            let young: Vec<(microlp::Variable, f64)> =
                (0..spec.target_ages[p]).map(|a| (iss[p][a], 1.0)).collect();
            if !young.is_empty() {
                lp.add_constraint(&young, ComparisonOp::Le, 0.0);
            }
        }
        if let Some(range) = spec.blending_range {
            let lo = spec.target_ages[p].saturating_sub(range);
            let hi = spec.target_ages[p] + range;
            let outside: Vec<(microlp::Variable, f64)> = (0..a_n)
                .filter(|&a| a < lo || a > hi)
                .map(|a| (iss[p][a], 1.0))
                .collect();
            if !outside.is_empty() {
                lp.add_constraint(&outside, ComparisonOp::Le, 0.0);
            }
        }
    }

    match lp.solve() {
        Ok(solution) => (0..p_n)
            .map(|p| (0..a_n).map(|a| solution[iss[p][a]].max(0.0)).collect())
            .collect(),
        // infeasible (e.g. empty inventory): issue nothing
        Err(_) => vec![vec![0.0; a_n]; p_n],
    }
}
