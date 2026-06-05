// ============================================================================
// perfect_information_lp.rs
//
// PURPOSE
// -------
// Faithful Rust port of the perfect-information (steady-state, expected-value)
// LINEAR PROGRAM that the Pahr & Grunow (2025) companion code uses to compute
// the long-run AVERAGE-PROFIT UPPER BOUND for the ameliorating-inventory
// problem. This bound (`max_reward`) is the reference value DRL policies are
// normalised against in the paper and is published per instance in the
// companion repository under `problem_configurations/<instance>/upper_bound.json`.
//
// The companion implementation lives in `AmelioratingInventoryPOM.py`,
// function `upper_bound(env, discr_step)`, and is solved there with Gurobi.
// We reproduce it exactly with the pure-Rust `microlp` simplex solver, reading
// the deterministic preprocessing inputs (per-product expected-revenue and
// slope curves, decay means, salvage, prices) from a checked-in dataset that
// mirrors the companion `config.json` / `expected_revenue.json` for each
// instance.
//
// WHY THIS IS THE FAITHFUL STEADY-STATE MODEL
// -------------------------------------------
// The companion env optimises long-run average profit of an age-structured
// ameliorating inventory with:
//   - stochastic truncated-Normal purchase price (mean 200, std 50, truncated
//     at +-70 std-units around the mean),
//   - Gaussian-copula-correlated sales prices and demand (rho = 0.5),
//   - age-dependent stochastic Beta decay proportions whose MEAN per age class
//     equals `decay_mean[a]` (Beta mean = alpha/(alpha+beta) = decay_mean),
//   - 0.03 (spirits) / 0.02 (port wine) multiplicative evaporation, so a unit
//     of age `a` retains `(1-evaporation)^(a+1)` of its volume,
//   - a per-product target age that the issued blend's (volume-weighted) mean
//     age must meet or exceed (blending of young + old stock),
//   - per-age-class capacity `maxInventory`.
//
// The perfect-information bound replaces every random quantity by its
// expectation and looks for the best STATIONARY (time-invariant) operating
// point: a single set of per-age inventory positions, per-age/per-product
// issuance volumes, per-product production volumes, and an expected purchase
// quantity, that maximises one-period expected profit subject to the
// steady-state inventory-balance (aging + mean decay) recursion. Because it
// (a) uses expectations, (b) allows continuous decisions, and (c) imposes only
// the steady-state balance (not the full stochastic information constraints),
// its optimal objective is a valid upper bound on achievable average profit.
//
// LP FORMULATION (matches companion `upper_bound`)
// ------------------------------------------------
// Indices: ages a in 0..A, products p in 0..P, purchase price levels l,
// revenue break points b (per product).
//
// Variables (all continuous, >= 0):
//   inv[a]   in [0, maxInventory]        steady-state on-hand of age class a
//   iss[p,a] in [0, maxInventory]        volume issued from age a toward prod p
//   pur[l]   in [0, maxInventory]        purchase volume at price level l
//   ff[p,b]  in [0, 1]                   convex weight on revenue break point b
//   out      in [0, maxInventory]        outdated (oldest unissued) volume
//
// Preprocessing of the concave expected-revenue curve R_p(.) into a PIECEWISE-
// LINEAR OUTER (upper) approximation:
//   - tangent points = the production grid 0..sales_bound[p] step `discr_step`
//     (here discr_step = production_step_size = 0.01, matching the published
//     bound),
//   - at each tangent point we know R_p(x) (`expected_revenue[p][x]`) and its
//     slope R_p'(x) (`slope[p][x]`); consecutive tangent lines intersect at
//     break points, and the revenue at a break point is the height of the
//     intersection. The last break point is pinned to (sales_bound, R(sales_bound)).
//   The convex combination sum_b ff[p,b]=1 with sum_b ff[p,b]*b <= produced
//   volume selects the tightest tangent line -> exact concave envelope value.
//
// Purchase-price discretisation:
//   price levels l = arange(ppf(0), ppf(1), discr_step) of the truncated-Normal
//   purchase-price distribution; prob[l] = cdf(l+discr_step) - cdf(l).
//
// Objective (MAXIMISE), one-period expected profit:
//   sum_{p,b} ff[p,b]*R_bp[p,b]                            (expected revenue)
//   - sum_l pur[l]*l*prob[l]                               (expected purchase cost)
//   + sum_a inv[a]*(salvage[a]*meanDecay[a] - holding)/(1-meanDecay[a])
//                                            (decay-salvage credit net of holding)
//   - out*outdatingCosts                                  (outdating penalty)
//
// Constraints:
//   inv[a] == (inv[a-1] - sum_p iss[p,a-1]) * (1 - meanDecay[a])   for a>=1
//   inv[0] == (sum_l pur[l]*prob[l]) * (1 - meanDecay[0])
//   out    == inv[A-1] - sum_p iss[p,A-1]
//   for each product p:
//     sum_b ff[p,b] == 1
//     sum_b ff[p,b]*b <= sum_a iss[p,a]*evapRemains[a]   (production <= issued volume)
//     sum_a iss[p,a]*a*evapRemains[a] >= targetAge[p] * sum_a iss[p,a]*evapRemains[a]
//                                              (blend mean age >= target age)
//     if NOT allowBlending: sum_{a<targetAge[p]} iss[p,a] <= 0  (no young stock issued)
//     if blendingRange set:  iss outside [target-range, target+range] <= 0
//     elif ageRange set:     iss outside ageRange[p] <= 0
//
// where evapRemains[a] = (1-evaporation)^(a+1), meanDecay[a] = decay_mean[a].
//
// VERIFICATION
// ------------
// `tests/verification.rs` loads the checked-in dataset, runs `solve_upper_bound`,
// and asserts the LP optimal reproduces the published `max_reward` within 1e-4
// (absolute). The two anchored instances are:
//   - spirits_0001 : A=10, P=3, targets [2,4,6], cap 50, holding 2.5,
//                     published max_reward = 1991.9344293376805
//   - port_wine    : A=25, P=2, targets [9,19], blending enabled,
//                     published max_reward = 2444.8010643781136
// ============================================================================

use microlp::{ComparisonOp, OptimizationDirection, Problem};
use statrs::distribution::{ContinuousCDF, Normal};

/// Deterministic inputs of the perfect-information LP for one instance.
/// Mirrors the relevant fields of the companion `config.json`.
#[derive(Clone, Debug)]
pub struct PerfectInformationLpInputs {
    pub instance: String,
    pub num_ages: usize,
    pub num_products: usize,
    pub target_ages: Vec<usize>,
    pub max_inventory: f64,
    pub evaporation: f64,
    /// Per-age mean decay proportion (Beta mean == decay_mean).
    pub decay_mean: Vec<f64>,
    pub holding_costs: f64,
    pub outdating_costs: f64,
    /// Per-age salvage value of decayed volume.
    pub decay_salvage: Vec<f64>,
    pub allow_blending: bool,
    pub blending_range: Option<usize>,
    pub age_range: Option<Vec<Vec<usize>>>,
    pub price_mean: f64,
    pub price_std: f64,
    /// Truncation expressed in raw price units (companion `price_truncation`).
    pub price_truncation: f64,
    /// Discretisation step for both the revenue PWL grid and price levels.
    pub production_step_size: f64,
    /// Per-product maximum production volume (top of the revenue grid).
    pub sales_bound: Vec<f64>,
    /// expected_revenue[p][i] for production level i*step, i = 0..=len-1.
    pub expected_revenue: Vec<Vec<f64>>,
    /// slope[p][i] = R_p'(i*step), aligned with `expected_revenue`.
    pub slope: Vec<Vec<f64>>,
}

/// Result of solving the perfect-information LP.
#[derive(Clone, Debug)]
pub struct PerfectInformationLpSolution {
    /// Long-run average-profit upper bound (`max_reward`).
    pub max_reward: f64,
    /// Expected purchase quantity at the optimum.
    pub purchasing: f64,
    /// Per-product production volume at the optimum.
    pub production: Vec<f64>,
    /// Per-age steady-state inventory position at the optimum.
    pub inventory_position: Vec<f64>,
}

/// Truncated-Normal purchase-price distribution used by the companion env:
/// `truncnorm(loc=mean, scale=std, a=-trunc/std, b=+trunc/std)`.
struct TruncatedNormalPrice {
    normal: Normal,
    lower: f64,
    upper: f64,
    cdf_lower: f64,
    cdf_span: f64,
}

impl TruncatedNormalPrice {
    fn new(mean: f64, std: f64, truncation: f64) -> Self {
        let normal = Normal::new(mean, std).expect("price std must be positive");
        let lower = mean - truncation;
        let upper = mean + truncation;
        let cdf_lower = normal.cdf(lower);
        let cdf_upper = normal.cdf(upper);
        Self {
            normal,
            lower,
            upper,
            cdf_lower,
            cdf_span: cdf_upper - cdf_lower,
        }
    }

    /// CDF of the truncated distribution at `x`.
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.lower {
            0.0
        } else if x >= self.upper {
            1.0
        } else {
            (self.normal.cdf(x) - self.cdf_lower) / self.cdf_span
        }
    }

    /// Support endpoints == ppf(0), ppf(1).
    fn ppf0(&self) -> f64 {
        self.lower
    }
    fn ppf1(&self) -> f64 {
        self.upper
    }
}

/// Build `arange(start, stop, step)` exactly like numpy: include start, stop
/// while value < stop (last value strictly below stop).
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil();
    let n = if n < 0.0 { 0 } else { n as usize };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = start + (i as f64) * step;
        if v < stop {
            out.push(v);
        }
    }
    out
}

/// Round to 2 decimals (companion uses `round(.,2)` on the production grid).
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Solve the perfect-information LP and return the average-profit upper bound.
pub fn solve_upper_bound(inputs: &PerfectInformationLpInputs) -> PerfectInformationLpSolution {
    let a_n = inputs.num_ages;
    let p_n = inputs.num_products;
    let step = inputs.production_step_size;

    let evap_remains: Vec<f64> = (0..a_n)
        .map(|a| (1.0 - inputs.evaporation).powi((a + 1) as i32))
        .collect();
    let mean_decay = &inputs.decay_mean;

    // -------- piecewise-linear outer approximation of each R_p(.) --------
    // tangent points = production grid 0..sales_bound[p] step `step`, aligned
    // index-for-index with expected_revenue[p] / slope[p].
    let mut break_points: Vec<Vec<f64>> = Vec::with_capacity(p_n);
    let mut revenue_at_break: Vec<Vec<f64>> = Vec::with_capacity(p_n);
    for p in 0..p_n {
        let tp: Vec<f64> = {
            let mut v = arange(0.0, inputs.sales_bound[p] + step, step)
                .into_iter()
                .map(round2)
                .collect::<Vec<_>>();
            // numpy arange(0, sales_bound+step, step) yields exactly the grid;
            // its length must match the expected_revenue table.
            v.truncate(inputs.expected_revenue[p].len());
            v
        };
        let er = &inputs.expected_revenue[p];
        let sl = &inputs.slope[p];
        assert_eq!(
            tp.len(),
            er.len(),
            "tangent grid / expected_revenue length mismatch for product {p}"
        );

        let m = tp.len();
        let mut bp = Vec::with_capacity(m);
        let mut rbp = Vec::with_capacity(m);
        // first break point = first tangent point
        bp.push(tp[0]);
        // interior break points = intersection of consecutive tangent lines.
        //
        // For a concave R_p with strictly decreasing slopes, the tangent lines
        // at tp[i] and tp[i+1] MUST intersect at an abscissa x* in the closed
        // interval [tp[i], tp[i+1]]. Near the top of the revenue grid the
        // sampled slopes become nearly equal (|sl[i]-sl[i+1]| ~ 1e-9 because the
        // finite-precision companion `slope` table flattens as demand is almost
        // surely covered), so the raw quotient below is numerically unstable and
        // can overshoot well past sales_bound (e.g. 27.89 > 27.59), producing a
        // NON-MONOTONE, out-of-range break-point sequence. The Gurobi-based
        // companion absorbs that inconsistency in presolve; the in-crate simplex
        // (microlp) instead reports the model infeasible. Clamping each interior
        // break point back into its only valid interval [tp[i], tp[i+1]] is the
        // exact location in infinite precision and removes only finite-precision
        // overshoot, so it keeps the formulation faithful while making it robust.
        for i in 0..(m - 1) {
            let denom = sl[i] - sl[i + 1];
            let raw = if denom.abs() < 1e-12 {
                // tangent lines parallel within numerical noise: the intersection
                // is undefined / irrelevant; collapse to the tangent point.
                tp[i]
            } else {
                (er[i + 1] - er[i] + sl[i] * tp[i] - sl[i + 1] * tp[i + 1]) / denom
            };
            let lo = tp[i].min(tp[i + 1]);
            let hi = tp[i].max(tp[i + 1]);
            bp.push(raw.clamp(lo, hi));
        }
        // last break point pinned to (sales_bound, R(sales_bound))
        bp.push(tp[m - 1]);
        // revenue at break points l = 0..m uses the tangent line through tp[l]
        // (matches companion: er_bp[bp[l]] = er[l] + slope[l]*(bp[l]-tp[l]) for
        // l in 0..m, then the final break point pinned to (sales_bound, er[m-1])).
        for l in 0..m {
            rbp.push(er[l] + sl[l] * (bp[l] - tp[l]));
        }
        rbp.push(er[m - 1]);
        break_points.push(bp);
        revenue_at_break.push(rbp);
    }

    // -------- purchase price discretisation --------
    let price = TruncatedNormalPrice::new(inputs.price_mean, inputs.price_std, inputs.price_truncation);
    let price_levels = arange(price.ppf0(), price.ppf1(), step);
    let price_probs: Vec<f64> = price_levels
        .iter()
        .map(|&l| price.cdf(l + step) - price.cdf(l))
        .collect();

    // -------- build the LP --------
    let mut lp = Problem::new(OptimizationDirection::Maximize);

    // ff[p][b]
    let mut ff: Vec<Vec<microlp::Variable>> = Vec::with_capacity(p_n);
    for p in 0..p_n {
        let mut row = Vec::with_capacity(break_points[p].len());
        for b in 0..break_points[p].len() {
            // objective coeff = revenue at break point
            row.push(lp.add_var(revenue_at_break[p][b], (0.0, 1.0)));
        }
        ff.push(row);
    }
    // pur[l]
    let pur: Vec<microlp::Variable> = price_levels
        .iter()
        .zip(price_probs.iter())
        .map(|(&l, &prob)| lp.add_var(-l * prob, (0.0, inputs.max_inventory)))
        .collect();
    // inv[a]
    let inv: Vec<microlp::Variable> = (0..a_n)
        .map(|a| {
            let coeff = (inputs.decay_salvage[a] * mean_decay[a] - inputs.holding_costs)
                / (1.0 - mean_decay[a]);
            lp.add_var(coeff, (0.0, inputs.max_inventory))
        })
        .collect();
    // iss[p][a]
    let mut iss: Vec<Vec<microlp::Variable>> = Vec::with_capacity(p_n);
    for _ in 0..p_n {
        let row: Vec<microlp::Variable> = (0..a_n)
            .map(|_| lp.add_var(0.0, (0.0, inputs.max_inventory)))
            .collect();
        iss.push(row);
    }
    // out
    let out = lp.add_var(-inputs.outdating_costs, (0.0, inputs.max_inventory));

    // inventory balance for a>=1:
    //   inv[a] - (1-decay[a])*inv[a-1] + (1-decay[a])*sum_p iss[p,a-1] == 0
    for a in 1..a_n {
        let mut terms: Vec<(microlp::Variable, f64)> = Vec::with_capacity(2 + p_n);
        terms.push((inv[a], 1.0));
        terms.push((inv[a - 1], -(1.0 - mean_decay[a])));
        for p in 0..p_n {
            terms.push((iss[p][a - 1], 1.0 - mean_decay[a]));
        }
        lp.add_constraint(&terms, ComparisonOp::Eq, 0.0);
    }
    // inv[0] - (1-decay[0])*sum_l prob[l]*pur[l] == 0
    {
        let mut terms: Vec<(microlp::Variable, f64)> = Vec::with_capacity(1 + pur.len());
        terms.push((inv[0], 1.0));
        for (l, &v) in pur.iter().enumerate() {
            terms.push((v, -(1.0 - mean_decay[0]) * price_probs[l]));
        }
        lp.add_constraint(&terms, ComparisonOp::Eq, 0.0);
    }
    // out - inv[A-1] + sum_p iss[p,A-1] == 0
    {
        let mut terms: Vec<(microlp::Variable, f64)> = Vec::with_capacity(2 + p_n);
        terms.push((out, 1.0));
        terms.push((inv[a_n - 1], -1.0));
        for p in 0..p_n {
            terms.push((iss[p][a_n - 1], 1.0));
        }
        lp.add_constraint(&terms, ComparisonOp::Eq, 0.0);
    }

    // per-product constraints
    for p in 0..p_n {
        // sum_b ff[p,b] == 1
        let conv: Vec<(microlp::Variable, f64)> =
            ff[p].iter().map(|&v| (v, 1.0)).collect();
        lp.add_constraint(&conv, ComparisonOp::Eq, 1.0);

        // sum_b ff[p,b]*b - sum_a iss[p,a]*evap[a] <= 0
        let mut prod_le: Vec<(microlp::Variable, f64)> =
            Vec::with_capacity(break_points[p].len() + a_n);
        for (b, &v) in ff[p].iter().enumerate() {
            prod_le.push((v, break_points[p][b]));
        }
        for a in 0..a_n {
            prod_le.push((iss[p][a], -evap_remains[a]));
        }
        lp.add_constraint(&prod_le, ComparisonOp::Le, 0.0);

        // target age: sum_a iss[p,a]*evap[a]*(target - a) <= 0
        let mut target: Vec<(microlp::Variable, f64)> = Vec::with_capacity(a_n);
        for a in 0..a_n {
            let coeff = (inputs.target_ages[p] as f64) * evap_remains[a] - (a as f64) * evap_remains[a];
            target.push((iss[p][a], coeff));
        }
        lp.add_constraint(&target, ComparisonOp::Le, 0.0);

        // blending restrictions
        if !inputs.allow_blending {
            // no issuance from age classes younger than the target age
            let mut young: Vec<(microlp::Variable, f64)> = Vec::new();
            for a in 0..inputs.target_ages[p] {
                young.push((iss[p][a], 1.0));
            }
            if !young.is_empty() {
                lp.add_constraint(&young, ComparisonOp::Le, 0.0);
            }
        }
        if let Some(range) = inputs.blending_range {
            let lo = inputs.target_ages[p].saturating_sub(range);
            let hi = inputs.target_ages[p] + range;
            let mut outside: Vec<(microlp::Variable, f64)> = Vec::new();
            for a in 0..a_n {
                if a < lo || a > hi {
                    outside.push((iss[p][a], 1.0));
                }
            }
            if !outside.is_empty() {
                lp.add_constraint(&outside, ComparisonOp::Le, 0.0);
            }
        } else if let Some(age_range) = &inputs.age_range {
            let allowed = &age_range[p];
            let mut outside: Vec<(microlp::Variable, f64)> = Vec::new();
            for a in 0..a_n {
                if !allowed.contains(&a) {
                    outside.push((iss[p][a], 1.0));
                }
            }
            if !outside.is_empty() {
                lp.add_constraint(&outside, ComparisonOp::Le, 0.0);
            }
        }
    }

    let solution = lp.solve().expect("perfect-information LP must be solvable");

    let max_reward = solution.objective();
    let purchasing: f64 = pur
        .iter()
        .enumerate()
        .map(|(l, &v)| solution[v] * price_probs[l])
        .sum();
    let production: Vec<f64> = (0..p_n)
        .map(|p| (0..a_n).map(|a| solution[iss[p][a]] * evap_remains[a]).sum())
        .collect();
    let inventory_position: Vec<f64> = (0..a_n).map(|a| solution[inv[a]]).collect();

    PerfectInformationLpSolution {
        max_reward,
        purchasing,
        production,
        inventory_position,
    }
}
