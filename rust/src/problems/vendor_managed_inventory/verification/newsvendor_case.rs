// ============================================================================
// vendor_managed_inventory / verification / newsvendor_case.rs
//
// OBJECTIVE
//   Re-derive the compound-Poisson newsvendor worked example used by the
//   vendor-managed-inventory family, so an in-crate test can reproduce it.
//
// ALGORITHM (matches the Gosavi instructor case-study derivation, which cites
// Ross 2002 / Nahmias 2001):
//   Inputs: customer arrival rate lambda; demand size d ~ UNIF(a,b); per-cycle
//   time T discrete with support/probabilities; holding cost h; penalty p.
//   1. Per-unit-time demand D = sum_{i=1..N} d_i is compound Poisson, so (Wald):
//        mu      = lambda * (a+b)/2
//        sigma^2 = lambda*(b-a)^2/12 + lambda*((a+b)/2)^2
//   2. Cycle time moments: mu_C = E[T], sigma_C^2 = Var[T].
//   3. Demand during a cycle (Nahmias 2001):
//        mu_cycle    = mu * mu_C
//        sigma_cycle^2 = mu_C*sigma^2 + mu^2*sigma_C^2
//   4. Order-up-to levels:
//        mean-demand heuristic (MDH): S = mu_cycle
//        six-sigma:                   S = mu_cycle + 3*sigma_cycle
//        newsvendor (normal approx):  critical ratio = p/(p+h),
//                                     S = mu_cycle + Phi^{-1}(p/(p+h))*sigma_cycle
//
// PROVENANCE / HONESTY (see references.rs header)
//   The numerical reference this reproduces is the Gosavi (2010) INSTRUCTOR
//   TEACHING CASE STUDY worked example, not a number printed in the
//   peer-reviewed Sui/Gosavi/Lin (2010) EMJ paper. Per the repo rule, an
//   instructor/handout number is NOT literature verification; the reference
//   carries literature_verified = false.
// ============================================================================

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::problems::vendor_managed_inventory::literature::references::NewsvendorWorkedCaseReference;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewsvendorWorkedCaseSummary {
    pub mean_demand_rate: f64,
    pub demand_variance: f64,
    pub cycle_time_mean: f64,
    pub cycle_time_variance: f64,
    pub cycle_demand_mean: f64,
    pub cycle_demand_variance: f64,
    pub cycle_demand_stddev: f64,
    pub critical_ratio: f64,
    pub k: f64,
    pub mean_demand_heuristic_order_up_to: f64,
    pub six_sigma_order_up_to: f64,
    pub newsvendor_order_up_to: f64,
}

fn validate_reference(reference: &NewsvendorWorkedCaseReference) -> PyResult<()> {
    if reference.cycle_time_support.len() != reference.cycle_time_probabilities.len() {
        return Err(PyValueError::new_err(
            "cycle_time_support and cycle_time_probabilities must have the same length",
        ));
    }
    if reference.cycle_time_support.is_empty() {
        return Err(PyValueError::new_err(
            "cycle_time_support must contain at least one value",
        ));
    }
    let sum = reference.cycle_time_probabilities.iter().sum::<f64>();
    if (sum - 1.0).abs() > 1e-12 {
        return Err(PyValueError::new_err(format!(
            "cycle_time_probabilities must sum to 1, found {sum}"
        )));
    }
    if reference.demand_size_high < reference.demand_size_low {
        return Err(PyValueError::new_err(
            "demand_size_high must be at least demand_size_low",
        ));
    }
    if reference.holding_cost_per_unit <= 0.0 {
        return Err(PyValueError::new_err(
            "holding_cost_per_unit must be positive",
        ));
    }
    if reference.stockout_cost_per_unit <= 0.0 {
        return Err(PyValueError::new_err(
            "stockout_cost_per_unit must be positive",
        ));
    }
    Ok(())
}

pub fn evaluate_newsvendor_worked_case(
    reference: &NewsvendorWorkedCaseReference,
) -> PyResult<NewsvendorWorkedCaseSummary> {
    validate_reference(reference)?;

    let mean_demand_rate = reference.customer_arrival_rate
        * (reference.demand_size_low + reference.demand_size_high)
        / 2.0;
    let demand_variance = reference.customer_arrival_rate
        * (reference.demand_size_high - reference.demand_size_low).powi(2)
        / 12.0
        + reference.customer_arrival_rate
            * ((reference.demand_size_low + reference.demand_size_high) / 2.0).powi(2);

    let cycle_time_mean = reference
        .cycle_time_support
        .iter()
        .zip(reference.cycle_time_probabilities.iter())
        .map(|(value, probability)| value * probability)
        .sum::<f64>();
    let cycle_time_variance = reference
        .cycle_time_support
        .iter()
        .zip(reference.cycle_time_probabilities.iter())
        .map(|(value, probability)| (value - cycle_time_mean).powi(2) * probability)
        .sum::<f64>();

    let cycle_demand_mean = mean_demand_rate * cycle_time_mean;
    let cycle_demand_variance =
        cycle_time_mean * demand_variance + mean_demand_rate.powi(2) * cycle_time_variance;
    let cycle_demand_stddev = cycle_demand_variance.sqrt();
    let critical_ratio = reference.stockout_cost_per_unit
        / (reference.stockout_cost_per_unit + reference.holding_cost_per_unit);
    let standard_normal =
        Normal::new(0.0, 1.0).expect("standard normal distribution must be valid");
    let k = standard_normal.inverse_cdf(critical_ratio);

    Ok(NewsvendorWorkedCaseSummary {
        mean_demand_rate,
        demand_variance,
        cycle_time_mean,
        cycle_time_variance,
        cycle_demand_mean,
        cycle_demand_variance,
        cycle_demand_stddev,
        critical_ratio,
        k,
        mean_demand_heuristic_order_up_to: cycle_demand_mean,
        six_sigma_order_up_to: cycle_demand_mean + 3.0 * cycle_demand_stddev,
        newsvendor_order_up_to: cycle_demand_mean + k * cycle_demand_stddev,
    })
}
