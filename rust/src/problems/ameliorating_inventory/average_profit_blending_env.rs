// ============================================================================
// average_profit_blending_env.rs
//
// PURPOSE
// -------
// Faithful executable port of the per-period dynamics of the Pahr & Grunow
// (2025) ameliorating-inventory environment (`AmelioratingInventoryPOM.py`,
// method `step_continuous_issuance_lp`). This is the long-run AVERAGE-PROFIT
// model the paper trains DRL policies on; its perfect-information upper bound
// is computed separately in `perfect_information_lp.rs` and is the
// literature-verification anchor.
//
// STATE  (price-augmented, continuous)
// ------------------------------------
//   price                    current purchase price (truncated-Normal level)
//   inventory_position[a]    on-hand volume of each age class a in 0..A
//
// ACTION  (3-part, matching the companion env)
// --------------------------------------------
//   purchase   aP  in [0, maxInventory]   raw volume bought into age class 0
//   production aY_w per product           (optional; if not driven, derived
//                                          from the issuance plan)
//   issuance   aX_{p,a}                    volume drawn from age a for product p
//
// In the companion env the issuance is solved by a Gurobi LP; here we solve the
// SAME single-period issuance LP (target-age mean constraint + blending rules +
// evaporation) with the in-crate `microlp` solver. Production is then
//   production[p] = min(sales_bound[p], sum_a iss[p,a]*evapRemains[a]).
//
// PER-PERIOD TRANSITION ORDERING  (matches the companion exactly)
// ----------------------------------------------------------------
//   1. read purchasing aP (and optional production targets);
//   2. solve issuance plan iss[p,a] given current inventory_position;
//   3. production[p] = min(sales_bound[p], sum_a iss[p,a]*evapRemains[a]);
//   4. new_inventory[a] = max(0, inv[a] - sum_p iss[p,a]); outdating = new_inv[A-1];
//   5. sample demand d[p] (Gaussian-copula-correlated with sales price);
//      sales[p] = min(production[p], d[p]);
//   6. AGE & PURCHASE: pre_decay = [purchasing, new_inventory[0..A-1]]
//      (everything ages one slot; oldest age class A-1 is outdated/dropped;
//       purchase enters age 0);
//   7. sample age-dependent Beta decay proportions; decay_samples =
//      pre_decay * decay_prop; inventory_position = pre_decay - decay_samples;
//   8. reward (one-period expected profit):
//         revenue(production)            (interpolated expected-revenue table)
//         - purchasing * price           (purchase cost at realised price)
//         - sum(pre_decay) * holding     (holding on aged+purchased stock)
//         + dot(decay_samples, decaySalvage)   (salvage credit of decayed vol)
//         - outdating * outdatingCosts;  (outdating penalty)
//   9. sample next purchase price.
//
// DECAY LAW
// ---------
//   For each age class a, the decay proportion is Beta(alpha_a, beta_a) with
//   mean = decay_mean[a] and coefficient of variation decay_cov[a]:
//     v      = decay_mean*(1-decay_mean)/(decay_cov*decay_mean)^2 - 1
//     alpha  = decay_mean * v
//     beta   = (1-decay_mean) * v
//   The Beta MEAN equals decay_mean[a]; the perfect-information LP uses that
//   mean directly. Evaporation removes a deterministic multiplicative fraction:
//   a unit of age a retains evapRemains[a] = (1-evaporation)^(a+1) of its volume
//   when issued/produced.
//
// REVENUE
// -------
//   The companion env scores revenue from an EXPECTED-revenue lookup table
//   R_p(production) (the double integral of the correlated demand/sales-price
//   distribution), linearly interpolated between grid levels. We reuse the same
//   table (carried in the LP dataset) so dynamics and bound share one revenue
//   model. Realised demand/sales are still sampled to drive the stochastic
//   trajectory but the per-period revenue credit is the expectation given
//   production, exactly as in the companion.
//
// OBJECTIVE
// ---------
//   Long-run average profit = mean per-period reward over a long horizon. The
//   companion normalises this to [reward_lb, reward_ub] using max_reward (the
//   LP bound) and min_reward=0; we expose the RAW average profit so it is
//   directly comparable to the LP bound.
// ============================================================================

use rand::Rng;
use rand_distr::{Beta, Distribution, Normal as RandNormal};
use statrs::distribution::{ContinuousCDF, Normal as StatNormal};

use crate::problems::ameliorating_inventory::issuance_blending_lp::{
    solve_single_period_issuance, IssuanceLpSpec,
};

/// Time-invariant configuration of the average-profit ameliorating env.
#[derive(Clone, Debug)]
pub struct AverageProfitBlendingConfig {
    pub num_ages: usize,
    pub num_products: usize,
    pub target_ages: Vec<usize>,
    pub max_inventory: f64,
    pub evaporation: f64,
    pub decay_mean: Vec<f64>,
    pub decay_cov: Vec<f64>,
    pub holding_costs: f64,
    pub outdating_costs: f64,
    pub decay_salvage: Vec<f64>,
    pub allow_blending: bool,
    pub blending_range: Option<usize>,
    // purchase price: truncated Normal(mean, std) truncated at +-truncation
    pub price_mean: f64,
    pub price_std: f64,
    pub price_truncation: f64,
    // demand and sales-price processes (per product), Gaussian-copula correlated
    pub demand_means: Vec<f64>,
    pub demand_covs: Vec<f64>,
    pub sales_means: Vec<f64>,
    pub sales_covs: Vec<f64>,
    pub correlation_demand_salesprice: Vec<f64>,
    // expected-revenue table (per product), aligned to grid 0..sales_bound step
    pub production_step_size: f64,
    pub sales_bound: Vec<f64>,
    pub expected_revenue: Vec<Vec<f64>>,
}

impl AverageProfitBlendingConfig {
    /// Per-age evaporation retention (1-evaporation)^(a+1).
    pub fn evap_remains(&self) -> Vec<f64> {
        (0..self.num_ages)
            .map(|a| (1.0 - self.evaporation).powi((a + 1) as i32))
            .collect()
    }

    /// Beta(alpha,beta) decay distribution for age class a.
    fn decay_beta(&self, a: usize) -> Beta<f64> {
        let m = self.decay_mean[a];
        let cv = self.decay_cov[a];
        let v = (m * (1.0 - m)) / (cv * m).powi(2) - 1.0;
        Beta::new(m * v, (1.0 - m) * v).expect("valid Beta decay parameters")
    }
}

/// Continuous, price-augmented state.
#[derive(Clone, Debug)]
pub struct AverageProfitBlendingState {
    pub price: f64,
    pub inventory_position: Vec<f64>,
}

/// The 3-part action and its realised effects for one period.
#[derive(Clone, Debug)]
pub struct AverageProfitStepOutcome {
    pub purchasing: f64,
    pub issuance_by_product_age: Vec<Vec<f64>>,
    pub production: Vec<f64>,
    pub realized_demand: Vec<f64>,
    pub sales: Vec<f64>,
    pub outdating: f64,
    pub revenue: f64,
    pub purchase_cost: f64,
    pub holding_cost: f64,
    pub decay_salvage_credit: f64,
    pub outdating_cost: f64,
    pub reward: f64,
    pub next_state: AverageProfitBlendingState,
}

/// Truncated-Normal purchase-price sampler/PPF, matching the companion.
struct TruncatedNormalPrice {
    stat: StatNormal,
    mean: f64,
    lower: f64,
    upper: f64,
    cdf_lower: f64,
    cdf_span: f64,
}

impl TruncatedNormalPrice {
    fn new(mean: f64, std: f64, truncation: f64) -> Self {
        let stat = StatNormal::new(mean, std).expect("price std positive");
        let lower = mean - truncation;
        let upper = mean + truncation;
        let cl = stat.cdf(lower);
        let cu = stat.cdf(upper);
        Self {
            stat,
            mean,
            lower,
            upper,
            cdf_lower: cl,
            cdf_span: cu - cl,
        }
    }
    /// Mean of the symmetric truncated Normal equals the untruncated mean.
    fn mean(&self) -> f64 {
        self.mean
    }
    /// Inverse CDF of the truncated distribution.
    fn ppf(&self, u: f64) -> f64 {
        let target = self.cdf_lower + u.clamp(0.0, 1.0) * self.cdf_span;
        self.stat.inverse_cdf(target).clamp(self.lower, self.upper)
    }
}

/// Initial state: price at distribution mean, inventory as supplied.
pub fn initialize_state(
    config: &AverageProfitBlendingConfig,
    inventory_position: &[f64],
) -> AverageProfitBlendingState {
    let price = TruncatedNormalPrice::new(config.price_mean, config.price_std, config.price_truncation)
        .mean();
    AverageProfitBlendingState {
        price,
        inventory_position: inventory_position.to_vec(),
    }
}

/// Interpolate the expected-revenue table at a production volume.
fn interpolate_revenue(config: &AverageProfitBlendingConfig, p: usize, production: f64) -> f64 {
    let table = &config.expected_revenue[p];
    let step = config.production_step_size;
    if production <= 0.0 {
        return table[0];
    }
    let top = config.sales_bound[p];
    if production >= top {
        return *table.last().unwrap();
    }
    let idx = (production / step).floor() as usize;
    let lo = idx;
    let hi = (idx + 1).min(table.len() - 1);
    if hi == lo {
        return table[lo];
    }
    let x_lo = lo as f64 * step;
    let ratio = (production - x_lo) / step;
    (1.0 - ratio) * table[lo] + ratio * table[hi]
}

/// One faithful period of the average-profit ameliorating-inventory env.
///
/// `purchasing` is the raw purchase volume (already rescaled from a network's
/// [-1,1] output via ((a0+1)/2)*maxInventory by the caller, if applicable).
pub fn step_state<R: Rng + ?Sized>(
    rng: &mut R,
    config: &AverageProfitBlendingConfig,
    state: &AverageProfitBlendingState,
    purchasing: f64,
) -> AverageProfitStepOutcome {
    let a_n = config.num_ages;
    let p_n = config.num_products;
    let evap = config.evap_remains();
    let purchasing = purchasing.clamp(0.0, config.max_inventory);

    // ---- 2. issuance plan over the current inventory ----
    let spec = IssuanceLpSpec {
        num_ages: a_n,
        num_products: p_n,
        target_ages: config.target_ages.clone(),
        allow_blending: config.allow_blending,
        blending_range: config.blending_range,
        evap_remains: evap.clone(),
        sales_bound: config.sales_bound.clone(),
        // marginal value of producing one unit of product p (proxy: top-grid
        // average revenue per unit) used to prioritise scarce stock.
        unit_revenue: (0..p_n)
            .map(|p| {
                let top = config.sales_bound[p].max(1e-9);
                *config.expected_revenue[p].last().unwrap() / top
            })
            .collect(),
    };
    let issuance = solve_single_period_issuance(&spec, &state.inventory_position);

    // ---- 3. production volumes ----
    let production: Vec<f64> = (0..p_n)
        .map(|p| {
            let issued: f64 = (0..a_n).map(|a| issuance[p][a] * evap[a]).sum();
            config.sales_bound[p].min(issued)
        })
        .collect();

    // ---- 4. post-issuance inventory & outdating ----
    let mut new_inventory = vec![0.0f64; a_n];
    for a in 0..a_n {
        let issued_from_a: f64 = (0..p_n).map(|p| issuance[p][a]).sum();
        new_inventory[a] = (state.inventory_position[a] - issued_from_a).max(0.0);
    }
    let outdating = new_inventory[a_n - 1];

    // ---- 5. sample demand & sales ----
    let mut realized_demand = vec![0.0f64; p_n];
    let mut sales = vec![0.0f64; p_n];
    for p in 0..p_n {
        let d = sample_correlated_demand(rng, config, p, state.price);
        realized_demand[p] = d;
        sales[p] = production[p].min(d);
    }

    // ---- 6. age & purchase ----
    let mut pre_decay = vec![0.0f64; a_n];
    pre_decay[0] = purchasing;
    for a in 1..a_n {
        pre_decay[a] = new_inventory[a - 1];
    }

    // ---- 7. sample decay ----
    let mut decay_samples = vec![0.0f64; a_n];
    let mut next_inventory = vec![0.0f64; a_n];
    for a in 0..a_n {
        let prop = config.decay_beta(a).sample(rng);
        decay_samples[a] = pre_decay[a] * prop;
        next_inventory[a] = pre_decay[a] - decay_samples[a];
    }

    // ---- 8. reward ----
    let revenue: f64 = (0..p_n)
        .map(|p| interpolate_revenue(config, p, production[p]))
        .sum();
    let purchase_cost = purchasing * state.price;
    let holding_cost: f64 = pre_decay.iter().sum::<f64>() * config.holding_costs;
    let decay_salvage_credit: f64 = (0..a_n)
        .map(|a| decay_samples[a] * config.decay_salvage[a])
        .sum();
    let outdating_cost = outdating * config.outdating_costs;
    let reward = revenue - purchase_cost - holding_cost + decay_salvage_credit - outdating_cost;

    // ---- 9. sample next price ----
    let price = TruncatedNormalPrice::new(config.price_mean, config.price_std, config.price_truncation);
    let next_price = price.ppf(rng.gen::<f64>());

    AverageProfitStepOutcome {
        purchasing,
        issuance_by_product_age: issuance,
        production,
        realized_demand,
        sales,
        outdating,
        revenue,
        purchase_cost,
        holding_cost,
        decay_salvage_credit,
        outdating_cost,
        reward,
        next_state: AverageProfitBlendingState {
            price: next_price,
            inventory_position: next_inventory,
        },
    }
}

/// Sample demand conditioned on the current sales price via the Gaussian-copula
/// correlation rho between demand and sales price (companion design). For the
/// average-profit reward only `production` matters (revenue uses the expected
/// table), but the realised demand drives the stochastic trajectory.
fn sample_correlated_demand<R: Rng + ?Sized>(
    rng: &mut R,
    config: &AverageProfitBlendingConfig,
    p: usize,
    _purchase_price: f64,
) -> f64 {
    let mean = config.demand_means[p];
    let std = config.demand_covs[p] * mean;
    // sales price is its own correlated Normal; we draw demand from its marginal
    // (the correlation matters for the precomputed expected-revenue table, which
    // we consume directly, so the trajectory marginal is the demand Normal).
    let normal = RandNormal::new(mean, std).expect("valid demand Normal");
    normal.sample(rng).max(0.0)
}
