#![allow(dead_code)]

//! Exact Clark-Scarf serial multi-echelon optimizer (literature verification anchor).
//!
//! OBJECTIVE
//! ---------
//! Provide a repo-native EXACT solver for the classical periodic-review serial
//! multi-echelon inventory system (Clark and Scarf 1960). It is the analytical anchor
//! for the `serial_clark_scarf` family: it computes the exact optimal echelon
//! base-stock levels and optimal expected cost. The family's `env.rs` simulation is
//! verified to reproduce these same optima under the optimal echelon base-stock policy
//! (see `verification`), and `network_inventory` (the Pirhooshyaran general-network
//! model) reuses this solver to check the serial benchmark rows it carries.
//!
//! MODEL (Clark and Scarf 1960; Federgruen and Zipkin 1984; Chen and Zheng 1994)
//! ----------------------------------------------------------------------------
//! - N stages in series. Stage 1 is the most downstream (faces customer demand);
//!   stage N is the most upstream (replenishes from an outside source with ample
//!   stock). Internally stages are indexed downstream -> upstream, k = 0..N-1,
//!   where k = 0 is stage 1.
//! - i.i.d. per-period demand at stage 1 (Normal or Poisson here).
//! - Deterministic integer lead time L_k between consecutive stages.
//! - Linear echelon holding cost h_k charged on echelon inventory at stage k, and a
//!   linear backorder penalty p charged at stage 1.
//! - Objective: minimize the long-run average (equivalently per-period stationary)
//!   holding-plus-penalty cost. The optimal policy is an echelon base-stock policy.
//!
//! Installation (local) holding costs H_k relate to echelon holding costs by
//! H_k = h_k + h_{k+1} + ... + h_1 with the downstream stage carrying the largest
//! installation cost. The literature rows store installation costs in
//! upstream->downstream order; `solve_from_local_costs` performs the conversion.
//!
//! ALGORITHM (exact recursive newsvendor decomposition)
//! ----------------------------------------------------
//! Let D_k denote demand over the lead time L_k (the sum of L_k i.i.d. per-period
//! demands; for Normal this is Normal(mu*L_k, sigma*sqrt(L_k)), for Poisson this is
//! Poisson(mu*L_k)). Let H_tot = sum_k h_k = H_1 (downstream installation cost).
//!
//!   C_bar_{-1}(x) = (p + H_tot) * max(-x, 0)            // induced penalty seed
//!   for k = 0, 1, ..., N-1 (downstream -> upstream):
//!       C_hat_k(x) = h_k * x + C_bar_{k-1}(x)
//!       C_k(y)     = E_{D_k}[ C_hat_k(y - D_k) ]
//!       S*_k       = argmin_y C_k(y)                     // echelon base-stock level
//!       C_bar_k(x) = C_k( min(S*_k, x) )                 // induced penalty for stage k+1
//!   optimal cost  = C_{N-1}(S*_{N-1})
//!
//! `C_k` is evaluated on a truncated uniform grid in the echelon inventory level.
//! For arguments below the grid (deep backorder), `C_hat_k` is evaluated by its
//! exact asymptotic linear form `c_hat_linear_below_grid`, which is exact in the
//! all-backordered regime. The grid construction (4-sigma truncation of the
//! lead-time-demand and sum-of-lead-time-demand distributions; default 1000 inventory
//! points / 100 demand points for continuous demand; integer grid for discrete
//! demand) mirrors Snyder's `stockpyl.ssm_serial.optimize_base_stock_levels`, the
//! public reference implementation accompanying the textbook.
//!
//! SELF-CONSISTENCY
//! ----------------
//! For N = 1 the recursion collapses to the single-stage newsvendor:
//!   S*_1 = F^{-1}(p/(p+h_1)),  cost = h_1 * E[(S-D)^+] + p * E[(D-S)^+],
//! which is exactly the classical newsvendor base case. The tests here assert the
//! single-stage closed form and exact reproduction of the discrete Poisson optima from
//! Snyder's `stockpyl.ssm_serial` reference implementation.

use statrs::distribution::ContinuousCDF;
use statrs::distribution::{Discrete, Normal as StatNormal, Poisson};

const DEFAULT_TAIL_SIGMAS: f64 = 4.0;
const DEFAULT_INVENTORY_POINTS: usize = 1000;
const DEFAULT_DEMAND_POINTS: usize = 100;

/// One stage of the serial chain, in downstream -> upstream order (index 0 = most
/// downstream stage that faces customer demand).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SerialStage {
    pub echelon_holding_cost: f64,
    pub lead_time: usize,
}

/// Per-period demand at the most downstream stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SerialDemand {
    Normal { mean: f64, std: f64 },
    Poisson { mean: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridParams {
    pub tail_sigmas: f64,
    pub inventory_points: usize,
    pub demand_points: usize,
}

impl Default for GridParams {
    fn default() -> Self {
        GridParams {
            tail_sigmas: DEFAULT_TAIL_SIGMAS,
            inventory_points: DEFAULT_INVENTORY_POINTS,
            demand_points: DEFAULT_DEMAND_POINTS,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SerialClarkScarfSolution {
    /// Echelon base-stock levels in downstream -> upstream order (index 0 = stage 1).
    pub echelon_base_stock_levels: Vec<f64>,
    /// Local (installation) base-stock levels in upstream -> downstream order, to
    /// match the literature row convention.
    pub local_base_stock_levels_upstream_to_downstream: Vec<f64>,
    /// Exact optimal expected per-period (long-run average) cost.
    pub optimal_cost: f64,
}

/// A discretized lead-time-demand distribution: matching `support` and `pmf` arrays
/// whose probabilities sum to 1.
struct LeadTimeDemand {
    support: Vec<f64>,
    pmf: Vec<f64>,
}

fn normal_lead_time_demand(mean: f64, std: f64, periods: usize, grid: &GridParams) -> LeadTimeDemand {
    if periods == 0 || std <= 0.0 {
        return LeadTimeDemand {
            support: vec![mean * periods as f64],
            pmf: vec![1.0],
        };
    }
    let m = mean * periods as f64;
    let s = std * (periods as f64).sqrt();
    let dist = StatNormal::new(m, s).expect("valid lead-time normal");
    let lo = (m - grid.tail_sigmas * s).max(0.0);
    let hi = m + grid.tail_sigmas * s;
    let n = grid.demand_points;
    let delta = (hi - lo) / n as f64;
    let mut support = Vec::with_capacity(n + 1);
    let mut pmf = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let d = lo + i as f64 * delta;
        let upper = if i == n { 1.0 } else { dist.cdf(d + delta * 0.5) };
        let lower = if i == 0 { 0.0 } else { dist.cdf(d - delta * 0.5) };
        support.push(d);
        pmf.push(upper - lower);
    }
    LeadTimeDemand { support, pmf }
}

fn poisson_lead_time_demand(mean: f64, periods: usize) -> LeadTimeDemand {
    if periods == 0 || mean <= 0.0 {
        return LeadTimeDemand {
            support: vec![mean * periods as f64],
            pmf: vec![1.0],
        };
    }
    let m = mean * periods as f64;
    let dist = Poisson::new(m).expect("valid lead-time poisson");
    // Truncate where the cumulative tail is negligible (manual cumulation; statrs
    // inverse_cdf is unreliable at extreme tail probabilities).
    let hi = poisson_upper_support(&dist, m);
    let mut support = Vec::with_capacity(hi + 1);
    let mut pmf = Vec::with_capacity(hi + 1);
    let mut accumulated: f64 = 0.0;
    for k in 0..=hi {
        let p = if k == hi { (1.0 - accumulated).max(0.0) } else { dist.pmf(k as u64) };
        accumulated += dist.pmf(k as u64);
        support.push(k as f64);
        pmf.push(p);
    }
    LeadTimeDemand { support, pmf }
}

/// Smallest support upper bound whose cumulative tail mass is negligible.
fn poisson_upper_support(dist: &Poisson, mean: f64) -> usize {
    let cap = (mean + 12.0 * mean.sqrt()).ceil() as usize + 20;
    let mut cumulative = 0.0;
    for k in 0..=cap {
        cumulative += dist.pmf(k as u64);
        if cumulative >= 1.0 - 1e-12 {
            return k;
        }
    }
    cap
}

fn lead_time_demand(demand: SerialDemand, periods: usize, grid: &GridParams) -> LeadTimeDemand {
    match demand {
        SerialDemand::Normal { mean, std } => normal_lead_time_demand(mean, std, periods, grid),
        SerialDemand::Poisson { mean } => poisson_lead_time_demand(mean, periods),
    }
}

fn nearest_index(value: f64, x_lo: f64, x_delta: f64, x_num: usize) -> usize {
    let raw = ((value - x_lo) / x_delta).round();
    if raw <= 0.0 {
        0
    } else if raw as usize >= x_num {
        x_num
    } else {
        raw as usize
    }
}

/// Solve the serial system given echelon holding costs and lead times in
/// downstream -> upstream order.
pub fn solve_serial_clark_scarf(
    stages: &[SerialStage],
    penalty: f64,
    demand: SerialDemand,
    grid: GridParams,
) -> SerialClarkScarfSolution {
    let n = stages.len();
    assert!(n >= 1, "serial system needs at least one stage");

    let h: Vec<f64> = stages.iter().map(|s| s.echelon_holding_cost).collect();
    let lead_times: Vec<usize> = stages.iter().map(|s| s.lead_time).collect();
    let h_total: f64 = h.iter().sum();
    let mean = match demand {
        SerialDemand::Normal { mean, .. } => mean,
        SerialDemand::Poisson { mean } => mean,
    };
    let std = match demand {
        SerialDemand::Normal { std, .. } => std,
        SerialDemand::Poisson { mean } => mean.sqrt(),
    };
    let sum_l: usize = lead_times.iter().sum();

    // Build inventory-level grid from the sum-of-lead-time-demand distribution.
    let discrete = matches!(demand, SerialDemand::Poisson { .. });
    let (x_lo, _x_hi, x_delta, x_num) = if discrete {
        let m = mean * sum_l as f64;
        let dist = Poisson::new(m.max(1e-9)).expect("valid total poisson");
        // Upper bound covers the demand-over-all-lead-times support; the lower
        // bound extends symmetrically to host deep-backorder echelon states.
        let hi = poisson_upper_support(&dist, m) as f64;
        let x_lo = -hi;
        let x_hi = hi;
        let x_num = (x_hi - x_lo).round() as usize;
        (x_lo, x_hi, 1.0_f64, x_num)
    } else {
        let m = mean * sum_l as f64;
        let s = std * (sum_l as f64).sqrt();
        let lo = m - grid.tail_sigmas * s;
        let hi = m + grid.tail_sigmas * s;
        let x_lo = lo - hi;
        let x_hi = hi;
        let x_num = grid.inventory_points;
        let x_delta = (x_hi - x_lo) / x_num as f64;
        (x_lo, x_hi, x_delta, x_num)
    };

    let x: Vec<f64> = (0..=x_num).map(|i| x_lo + i as f64 * x_delta).collect();

    // Prefix sums of lead times (downstream -> upstream) for the linear extrapolation.
    // l_prefix[t] = L_0 + ... + L_{t-1}
    let mut l_prefix = vec![0usize; n + 1];
    for t in 0..n {
        l_prefix[t + 1] = l_prefix[t] + lead_times[t];
    }

    // Exact asymptotic linear form of C_hat_k(v) for v below the grid (all
    // demand backordered). See module docstring.
    let c_hat_linear_below_grid = |k: usize, v: f64| -> f64 {
        let sum_l_below = l_prefix[k] as f64; // L_0 + ... + L_{k-1}
        let mut value = -(penalty + h_total) * (v - mean * sum_l_below);
        for kp in 0..=k {
            let inner = (l_prefix[k] - l_prefix[kp]) as f64; // L_kp + ... + L_{k-1}
            value += h[kp] * (v - mean * inner);
        }
        value
    };

    // C_bar from the previous stage (seed = induced penalty on backorders).
    let mut c_bar_prev: Vec<f64> = x.iter().map(|xi| (penalty + h_total) * (-xi).max(0.0)).collect();

    let mut echelon_levels = vec![0.0_f64; n];
    let mut optimal_cost = 0.0_f64;

    for k in 0..n {
        let c_hat: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, xi)| h[k] * xi + c_bar_prev[i])
            .collect();

        let ltd = lead_time_demand(demand, lead_times[k], &grid);

        let chat_at = |v: f64| -> f64 {
            if v < x_lo {
                c_hat_linear_below_grid(k, v)
            } else {
                c_hat[nearest_index(v, x_lo, x_delta, x_num)]
            }
        };

        // C_k(y) for every grid point y.
        let mut c_k = vec![0.0_f64; x.len()];
        for (i, &y) in x.iter().enumerate() {
            let mut expected = 0.0;
            for (d, prob) in ltd.support.iter().zip(ltd.pmf.iter()) {
                if *prob == 0.0 {
                    continue;
                }
                expected += prob * chat_at(y - d);
            }
            c_k[i] = expected;
        }

        // Minimize over y.
        let mut best_idx = 0usize;
        let mut best_val = f64::INFINITY;
        for (i, &val) in c_k.iter().enumerate() {
            if val < best_val {
                best_val = val;
                best_idx = i;
            }
        }
        echelon_levels[k] = x[best_idx];
        optimal_cost = best_val;

        // C_bar_k(x) = C_k(min(S*_k, x)).
        let s_star = x[best_idx];
        c_bar_prev = x
            .iter()
            .map(|xi| c_k[nearest_index(xi.min(s_star), x_lo, x_delta, x_num)])
            .collect();
    }

    // Local installation base-stock levels, upstream -> downstream, from echelon
    // differences: local s_k = echelon S_k - echelon S_{k-1} (downstream -> upstream),
    // then reversed to upstream -> downstream.
    let mut local_du = vec![0.0_f64; n];
    for k in 0..n {
        local_du[k] = if k == 0 {
            echelon_levels[0]
        } else {
            echelon_levels[k] - echelon_levels[k - 1]
        };
    }
    let local_ud: Vec<f64> = local_du.into_iter().rev().collect();

    SerialClarkScarfSolution {
        echelon_base_stock_levels: echelon_levels,
        local_base_stock_levels_upstream_to_downstream: local_ud,
        optimal_cost,
    }
}

/// Convenience entry that takes installation (local) holding costs and lead times in
/// upstream -> downstream order (the literature-row convention) plus the downstream
/// penalty, converts to echelon holding costs, and solves.
pub fn solve_from_local_costs(
    local_holding_upstream_to_downstream: &[f64],
    lead_times_upstream_to_downstream: &[usize],
    downstream_penalty: f64,
    demand: SerialDemand,
    grid: GridParams,
) -> SerialClarkScarfSolution {
    let n = local_holding_upstream_to_downstream.len();
    assert_eq!(n, lead_times_upstream_to_downstream.len());

    // Reverse to downstream -> upstream installation costs / lead times.
    let local_du: Vec<f64> = local_holding_upstream_to_downstream
        .iter()
        .rev()
        .copied()
        .collect();
    let lead_du: Vec<usize> = lead_times_upstream_to_downstream
        .iter()
        .rev()
        .copied()
        .collect();

    // Echelon holding cost h_k = H_k - H_{k+1} (downstream -> upstream), H_{N} = 0.
    let stages: Vec<SerialStage> = (0..n)
        .map(|k| {
            let upstream = if k + 1 < n { local_du[k + 1] } else { 0.0 };
            SerialStage {
                echelon_holding_cost: local_du[k] - upstream,
                lead_time: lead_du[k],
            }
        })
        .collect();

    solve_serial_clark_scarf(&stages, downstream_penalty, demand, grid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn single_stage_reduces_to_newsvendor_closed_form() {
        // Normal(10, 2), L=1, h=10, p=30 -> the already-verified single-node row
        // (Pirhooshyaran/Snyder Table 1 case 2): OUL 11.35, cost 25.42.
        let sol = solve_serial_clark_scarf(
            &[SerialStage { echelon_holding_cost: 10.0, lead_time: 1 }],
            30.0,
            SerialDemand::Normal { mean: 10.0, std: 2.0 },
            GridParams::default(),
        );
        assert!(
            approx(sol.echelon_base_stock_levels[0], 11.35, 0.05),
            "OUL reproduced={} expected=11.35",
            sol.echelon_base_stock_levels[0]
        );
        assert!(
            approx(sol.optimal_cost, 25.42, 0.1),
            "cost reproduced={} expected=25.42",
            sol.optimal_cost
        );
    }

    #[test]
    fn poisson_instances_match_reference_implementation() {
        // Exact discrete (integer-grid) values from stockpyl.ssm_serial (Snyder),
        // the public reference implementation. Discrete recursion is exact.
        // 1-stage Poisson(5), L=1, h=1, p=9.
        let s1 = solve_serial_clark_scarf(
            &[SerialStage { echelon_holding_cost: 1.0, lead_time: 1 }],
            9.0,
            SerialDemand::Poisson { mean: 5.0 },
            GridParams::default(),
        );
        assert_eq!(s1.echelon_base_stock_levels[0], 8.0);
        assert!(approx(s1.optimal_cost, 4.220849, 0.01), "1-stage cost={}", s1.optimal_cost);

        // 2-stage Poisson(5), L=[1,1], p=10. stockpyl call used
        // echelon_holding_cost=[1,2] in its upstream->downstream list order, i.e.
        // downstream echelon h_1=2, upstream echelon h_2=1.
        let s2 = solve_serial_clark_scarf(
            &[
                SerialStage { echelon_holding_cost: 2.0, lead_time: 1 },
                SerialStage { echelon_holding_cost: 1.0, lead_time: 1 },
            ],
            10.0,
            SerialDemand::Poisson { mean: 5.0 },
            GridParams::default(),
        );
        assert_eq!(s2.echelon_base_stock_levels, vec![7.0, 13.0]);
        assert!(approx(s2.optimal_cost, 16.797779, 0.01), "2-stage cost={}", s2.optimal_cost);

        // 3-stage Poisson(5), L=[2,1,1] (upstream->downstream), local h=[2,4,7], p=37.12.
        let s3 = solve_from_local_costs(
            &[2.0, 4.0, 7.0],
            &[2, 1, 1],
            37.12,
            SerialDemand::Poisson { mean: 5.0 },
            GridParams::default(),
        );
        assert_eq!(s3.echelon_base_stock_levels, vec![9.0, 15.0, 26.0]);
        assert!(approx(s3.optimal_cost, 72.043543, 0.02), "3-stage poisson cost={}", s3.optimal_cost);
    }

}
