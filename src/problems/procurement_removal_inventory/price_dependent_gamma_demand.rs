// price_dependent_gamma_demand.rs
//
// PURPOSE
// -------
// Faithful demand model for the Maggiar & Sadighian (2017) "Joint Inventory and
// Revenue Management with Removal Decisions" environment. This module turns the
// retailer's PRICING decision into an expected demand and a randomized realized
// demand, exactly as the paper specifies in Section 7.1.1.
//
// ALGORITHM (paper Section 7.1.1)
// -------------------------------
// 1. Additive demand: D_t(p) = d_t(p) + eps_t, where d_t(p) is the price-driven
//    expected demand and eps_t is mean-zero noise.
// 2. Log-linear price response: d_t(p) = mu_t * exp(-beta * (p - p0)), where
//    mu_t is the forecast expected demand at the base price p0 (MSRP), and beta
//    is the price-sensitivity parameter. The elasticity at the base price is
//    E = -beta * p0, so beta = -E / p0. With Table-1 elasticity E = -2 and
//    p0 = 90 this gives beta = 2/90.
// 3. The retailer chooses an EXPECTED demand target `d` (equivalently a price).
//    Inverting the log-linear map gives the implied price:
//        p(d) = p0 - (1/beta) * ln(d / mu_t).
//    The per-period expected revenue is r_t(d) = d * p(d). With the pricing
//    constraint p <= p0 (markdowns only), the admissible region is d >= mu_t.
// 4. Demand noise: the paper assumes that mu_t + eps_t is Gamma-distributed with
//    mean mu_t and coefficient of variation (CV) equal to 1. A Gamma with CV = 1
//    has shape alpha = 1/CV^2 = 1 and rate = 1/(mu_t * CV^2) = 1/mu_t (i.e. an
//    Exponential(1/mu_t)). The additive noise is eps_t = (mu_t + eps_t) - mu_t,
//    whose distribution does NOT depend on the pricing decision `d` (additive
//    model). Hence a realized demand is D = d + eps_t = d + (G - mu_t) where
//    G ~ Gamma(mean mu_t, CV 1).
// 5. Demand discretization (paper Section 6.2.1): the expectation over demand is
//    approximated with K equally-likely quantiles. The q-th quantile uses the
//    midpoint probability (q - 0.5)/K of the Gamma CDF. Because the model is
//    additive, the quantiles of the noise eps_t are lifted by the chosen `d`.
//
// This module exposes:
//   - `price_at_demand`  : p(d) for the log-linear inverse map
//   - `expected_revenue` : r_t(d) = d * p(d)
//   - `noise_quantiles`  : the K mean-zero noise quantiles eps for a given mu_t
//   - `realized_demand_quantiles` : d + eps for a chosen target demand `d`
//
// All callers (the faithful environment and its DP) consume these so the demand
// process lives in exactly one place.

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use statrs::distribution::{ContinuousCDF, Gamma};

/// Price-sensitivity parameter beta from the base-price elasticity.
///
/// E = -beta * p0  =>  beta = -E / p0. The elasticity E is negative (demand
/// falls as price rises), so beta is positive.
pub fn beta_from_elasticity(base_price: f64, elasticity_at_base_price: f64) -> PyResult<f64> {
    if !base_price.is_finite() || base_price <= 0.0 {
        return Err(PyValueError::new_err("base_price must be finite and > 0"));
    }
    if !elasticity_at_base_price.is_finite() || elasticity_at_base_price >= 0.0 {
        return Err(PyValueError::new_err(
            "elasticity_at_base_price must be finite and negative",
        ));
    }
    Ok(-elasticity_at_base_price / base_price)
}

/// Inverse log-linear demand map: the price p that induces expected demand `d`
/// when the forecast expected demand at base price is `mu`.
///
/// d = mu * exp(-beta (p - p0))  =>  p = p0 - (1/beta) ln(d / mu).
pub fn price_at_demand(
    target_demand: f64,
    forecast_mean_demand: f64,
    base_price: f64,
    beta: f64,
) -> PyResult<f64> {
    if !target_demand.is_finite() || target_demand <= 0.0 {
        return Err(PyValueError::new_err("target_demand must be finite and > 0"));
    }
    if !forecast_mean_demand.is_finite() || forecast_mean_demand <= 0.0 {
        return Err(PyValueError::new_err(
            "forecast_mean_demand must be finite and > 0",
        ));
    }
    if !beta.is_finite() || beta <= 0.0 {
        return Err(PyValueError::new_err("beta must be finite and > 0"));
    }
    Ok(base_price - (1.0 / beta) * (target_demand / forecast_mean_demand).ln())
}

/// Per-period expected revenue r_t(d) = d * p(d) under the log-linear map.
pub fn expected_revenue(
    target_demand: f64,
    forecast_mean_demand: f64,
    base_price: f64,
    beta: f64,
) -> PyResult<f64> {
    let price = price_at_demand(target_demand, forecast_mean_demand, base_price, beta)?;
    Ok(target_demand * price)
}

/// Build a Gamma distribution with the given mean and coefficient of variation.
///
/// shape alpha = 1 / CV^2, rate = 1 / (mean * CV^2). For CV = 1 this is the
/// Exponential(1/mean) used by the paper.
fn gamma_with_mean_cv(mean: f64, cv: f64) -> PyResult<Gamma> {
    if !mean.is_finite() || mean <= 0.0 {
        return Err(PyValueError::new_err("demand mean must be finite and > 0"));
    }
    if !cv.is_finite() || cv <= 0.0 {
        return Err(PyValueError::new_err(
            "coefficient of variation must be finite and > 0",
        ));
    }
    let alpha = 1.0 / (cv * cv);
    let rate = 1.0 / (mean * cv * cv);
    Gamma::new(alpha, rate)
        .map_err(|err| PyValueError::new_err(format!("invalid Gamma parameters: {err}")))
}

/// The K equally-likely mean-zero noise quantiles eps_q for forecast mean `mu`
/// and coefficient of variation `cv`.
///
/// The q-th quantile (q = 1..=K) uses the midpoint probability (q - 0.5)/K of
/// the Gamma(mean mu, cv) CDF, then subtracts the mean so the noise is centred:
///     eps_q = Gamma^{-1}((q - 0.5)/K) - mu.
pub fn noise_quantiles(
    forecast_mean_demand: f64,
    coefficient_of_variation: f64,
    num_quantiles: usize,
) -> PyResult<Vec<f64>> {
    if num_quantiles == 0 {
        return Err(PyValueError::new_err("num_quantiles must be >= 1"));
    }
    let gamma = gamma_with_mean_cv(forecast_mean_demand, coefficient_of_variation)?;
    let k = num_quantiles as f64;
    Ok((1..=num_quantiles)
        .map(|q| {
            let probability = (q as f64 - 0.5) / k;
            gamma.inverse_cdf(probability) - forecast_mean_demand
        })
        .collect())
}

/// Realized demand quantiles D_q = d + eps_q for the chosen target demand `d`.
///
/// Negative realizations are clipped to zero (demand cannot be negative).
pub fn realized_demand_quantiles(
    target_demand: f64,
    forecast_mean_demand: f64,
    coefficient_of_variation: f64,
    num_quantiles: usize,
) -> PyResult<Vec<f64>> {
    let noise = noise_quantiles(forecast_mean_demand, coefficient_of_variation, num_quantiles)?;
    Ok(noise
        .into_iter()
        .map(|eps| (target_demand + eps).max(0.0))
        .collect())
}
