// joint_pricing_removal_dp.rs
//
// PURPOSE
// -------
// Executing finite-horizon dynamic program for the faithful Maggiar & Sadighian
// (2017) joint pricing / inventory / removal model defined in
// joint_pricing_removal_env.rs. It computes the value function V_t(x, y) (the
// maximum expected discounted profit / NPV from period t onward, paper Eq. 4),
// the optimal first-period decisions, and exposes them so the verification tests
// can RE-RUN the env and reproduce the paper's PROVEN structural properties
// (Lemma 3.1 and the monotonicity bullets in Section 3.2 / Section 7.2.1) as
// well as the magnitude of the reported NPV surface.
//
// ALGORITHM (backward induction over Eq. 4)
// -----------------------------------------
// State grid: integer (x, y) with 0 <= y <= x, x in [0, x_max].
// Demand:     for the period's forecast mean mu_t, the realized demand is
//             approximated by K equally-likely quantiles D_q = d + eps_q where
//             eps_q are the centred Gamma(mean mu_t, CV) quantiles
//             (price_dependent_gamma_demand.rs). The decision `d` (target demand)
//             is searched over a grid in [mu_t, d_max].
// Removal/purchase: the signed net flow q is searched over [-purchase_max, x]
//             (purchase up to a cap, remove up to on-hand inventory).
// Recursion:  for each (x, y) and each (d, q),
//     value(d, q) = (1/K) * sum_q [ profit(x, y, d, q, D_q)
//                                   + gamma * V_{t+1}(x', y') ]
//   where profit and the next state come from step_period(); V_{t+1} is read off
//   the next-period table with nearest-grid lookup (next inventory is already an
//   integer on the grid because demand realizations and flows are integers /
//   rounded). V_t(x, y) = max over (d, q) of value(d, q).
// Terminal:   V_T(x, y) = terminal_value(x, y) (full return + liquidation).
//
// The DP is intentionally small-instance-friendly: the verification instance
// uses a coarse demand grid and a modest x_max so it solves exactly and quickly
// inside `cargo test`. The same code, with the Table-1 parameters and a fine
// grid, reproduces the magnitude of the paper's ~84000 NPV surface (characterized
// in the tests, not asserted to tight tolerance because the paper specifies the
// mu_t profile only graphically).

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::procurement_removal_inventory::joint_pricing_removal_env::{
    step_period, terminal_value, validate_parameters, JointPricingRemovalParameters,
    JointPricingRemovalState,
};
use crate::problems::procurement_removal_inventory::price_dependent_gamma_demand::noise_quantiles;

/// Configuration of one finite-horizon DP solve.
#[derive(Clone, Debug, PartialEq)]
pub struct JointPricingRemovalDpConfig {
    pub periods: usize,
    pub discount_factor: f64,
    pub parameters: JointPricingRemovalParameters,
    /// Forecast mean demand mu_t for each period t = 0..periods.
    pub forecast_mean_demand: Vec<f64>,
    /// Fixed-returnability quota: max returnable units that can be purchased per
    /// period (paper: a quantile of the base-price forecast demand).
    pub returnable_purchase_cap: i64,
    /// Inventory-grid upper bound x_max.
    pub max_inventory_level: i64,
    /// Max units purchasable in a single period (|q| for q < 0).
    pub max_purchase_quantity: i64,
    /// Number of demand quantiles K used to approximate the expectation.
    pub num_demand_quantiles: usize,
    /// Number of target-demand grid points searched per period (price grid).
    pub num_price_points: usize,
    /// Upper multiple of mu_t for the target-demand search (d in [mu_t, mult*mu_t]).
    pub max_demand_multiple: f64,
}

/// The optimal decision recovered at a state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimalDecision {
    pub target_demand: f64,
    pub implied_price: f64,
    pub net_flow: i64,
    pub returned_units: i64,
    pub liquidated_units: i64,
    pub purchased_units: i64,
}

/// Full result of a DP solve.
pub struct JointPricingRemovalDpResult {
    config: JointPricingRemovalDpConfig,
    /// value_tables[t] is indexed by state_index(x, y); the NPV V_t(x, y).
    value_tables: Vec<Vec<f64>>,
    /// decision_tables[t][state_index] is the optimal first decision at V_t.
    decision_tables: Vec<Vec<OptimalDecision>>,
}

fn validate_config(config: &JointPricingRemovalDpConfig) -> PyResult<()> {
    validate_parameters(&config.parameters)?;
    if config.periods == 0 {
        return Err(PyValueError::new_err("periods must be >= 1"));
    }
    if config.forecast_mean_demand.len() != config.periods {
        return Err(PyValueError::new_err(
            "forecast_mean_demand must have one entry per period",
        ));
    }
    if !(0.0..=1.0).contains(&config.discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }
    if config.max_inventory_level < 0 {
        return Err(PyValueError::new_err("max_inventory_level must be >= 0"));
    }
    if config.num_demand_quantiles == 0 || config.num_price_points == 0 {
        return Err(PyValueError::new_err(
            "num_demand_quantiles and num_price_points must be >= 1",
        ));
    }
    if config.max_demand_multiple < 1.0 {
        return Err(PyValueError::new_err(
            "max_demand_multiple must be >= 1 (markdowns only, d >= mu_t)",
        ));
    }
    Ok(())
}

/// Number of (x, y) states with 0 <= y <= x <= x_max:  sum_{x=0}^{X}(x+1).
fn num_states(max_inventory_level: i64) -> usize {
    let x = max_inventory_level as usize;
    (x + 1) * (x + 2) / 2
}

/// Map (x, y) with 0 <= y <= x to a dense index.
fn state_index(inventory_level: i64, returnable_level: i64) -> usize {
    let x = inventory_level as usize;
    let y = returnable_level as usize;
    // states for inventory 0..x-1 occupy sum_{i=0}^{x-1}(i+1) = x(x+1)/2 slots,
    // then offset by y within row x.
    x * (x + 1) / 2 + y
}

/// Build the target-demand (price) search grid for a period.
fn demand_grid(forecast_mean: f64, num_points: usize, max_multiple: f64) -> Vec<f64> {
    if num_points == 1 {
        return vec![forecast_mean];
    }
    let lo = forecast_mean;
    let hi = forecast_mean * max_multiple;
    let step = (hi - lo) / (num_points as f64 - 1.0);
    (0..num_points).map(|i| lo + step * i as f64).collect()
}

pub fn solve(config: &JointPricingRemovalDpConfig) -> PyResult<JointPricingRemovalDpResult> {
    validate_config(config)?;

    let n_states = num_states(config.max_inventory_level);
    let mut value_tables: Vec<Vec<f64>> = vec![Vec::new(); config.periods + 1];
    let mut decision_tables: Vec<Vec<OptimalDecision>> = vec![Vec::new(); config.periods];

    // Terminal value table at t = periods.
    let mut terminal = vec![0.0f64; n_states];
    for x in 0..=config.max_inventory_level {
        for y in 0..=x {
            let state = JointPricingRemovalState {
                period: config.periods,
                inventory_level: x,
                returnable_level: y,
            };
            terminal[state_index(x, y)] = terminal_value(&state, &config.parameters)?;
        }
    }
    value_tables[config.periods] = terminal;

    // Backward induction.
    for t in (0..config.periods).rev() {
        let mu_t = config.forecast_mean_demand[t];
        let eps = noise_quantiles(
            mu_t,
            config.parameters.coefficient_of_variation,
            config.num_demand_quantiles,
        )?;
        let ds = demand_grid(mu_t, config.num_price_points, config.max_demand_multiple);

        let mut values = vec![f64::NEG_INFINITY; n_states];
        let mut decisions = vec![
            OptimalDecision {
                target_demand: mu_t,
                implied_price: config.parameters.base_price,
                net_flow: 0,
                returned_units: 0,
                liquidated_units: 0,
                purchased_units: 0,
            };
            n_states
        ];

        let next_table = &value_tables[t + 1];

        for x in 0..=config.max_inventory_level {
            for y in 0..=x {
                let state = JointPricingRemovalState {
                    period: t,
                    inventory_level: x,
                    returnable_level: y,
                };
                let idx = state_index(x, y);
                let mut best_value = f64::NEG_INFINITY;
                let mut best_decision = decisions[idx];

                // Removal/purchase flow grid: q in [-purchase_max, x].
                let q_lo = -config.max_purchase_quantity;
                let q_hi = x;
                for q in q_lo..=q_hi {
                    for &d in &ds {
                        let mut expected = 0.0;
                        let mut representative: Option<crate::problems::procurement_removal_inventory::joint_pricing_removal_env::JointPricingRemovalOutcome> = None;
                        for &e in &eps {
                            let realized = (d + e).max(0.0);
                            let outcome = step_period(
                                &state,
                                d,
                                q,
                                realized,
                                config.returnable_purchase_cap,
                                mu_t,
                                &config.parameters,
                            )?;
                            let nx = outcome
                                .next_state
                                .inventory_level
                                .clamp(0, config.max_inventory_level);
                            let ny = outcome.next_state.returnable_level.clamp(0, nx);
                            let continuation = next_table[state_index(nx, ny)];
                            expected += outcome.period_profit
                                + config.discount_factor * continuation;
                            if representative.is_none() {
                                representative = Some(outcome);
                            }
                        }
                        expected /= eps.len() as f64;
                        if expected > best_value {
                            best_value = expected;
                            let rep = representative.expect("at least one quantile");
                            best_decision = OptimalDecision {
                                target_demand: d,
                                implied_price: rep.implied_price,
                                net_flow: q,
                                returned_units: rep.returned_units,
                                liquidated_units: rep.liquidated_units,
                                purchased_units: rep.purchased_units,
                            };
                        }
                    }
                }

                values[idx] = best_value;
                decisions[idx] = best_decision;
            }
        }

        value_tables[t] = values;
        decision_tables[t] = decisions;
    }

    Ok(JointPricingRemovalDpResult {
        config: config.clone(),
        value_tables,
        decision_tables,
    })
}

impl JointPricingRemovalDpResult {
    /// NPV V_t(x, y) at a state on the grid.
    pub fn value_at(&self, period: usize, inventory_level: i64, returnable_level: i64) -> f64 {
        let x = inventory_level.clamp(0, self.config.max_inventory_level);
        let y = returnable_level.clamp(0, x);
        self.value_tables[period][state_index(x, y)]
    }

    /// Optimal decision recovered at a state.
    pub fn decision_at(
        &self,
        period: usize,
        inventory_level: i64,
        returnable_level: i64,
    ) -> OptimalDecision {
        let x = inventory_level.clamp(0, self.config.max_inventory_level);
        let y = returnable_level.clamp(0, x);
        self.decision_tables[period][state_index(x, y)]
    }

    /// Maximum NPV over all states at the given period (the peak of the V_t
    /// surface the paper plots; ~84000 in Section 7.2.1 at t = 24).
    pub fn max_value_at_period(&self, period: usize) -> f64 {
        self.value_tables[period]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn max_inventory_level(&self) -> i64 {
        self.config.max_inventory_level
    }
}
