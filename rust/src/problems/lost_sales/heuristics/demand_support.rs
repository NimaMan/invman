// Demand-support utilities for the lost-sales heuristics.
//
// The heuristics reason about a single-product, IID per-period demand
// distribution. Two views of demand are needed:
//
//   1. A truncated, normalised probability mass function over the per-period
//      demand support `{(demand, probability)}`. This is used by the lookahead
//      cost recursions (Myopic-1 / Myopic-2), which take expectations over the
//      next period's demand.
//
//   2. A cumulative distribution function (CDF) of the demand summed over a
//      number of consecutive periods. This is used by the Standard Vector Base
//      Stock policy, whose base-stock levels are critical-fractile quantiles of
//      the lead-time demand convolution.
//
// IID Poisson and Geometric demand are supported directly; these are the
// distributions for which the lost-sales heuristics in the literature are
// defined. The truncation drops a negligible tail (`DEMAND_SUPPORT_TAIL_MASS`)
// and renormalises so the masses sum to exactly one.
//
// Markov-Modulated Poisson (MMPP2) demand is supported via its *stationary
// marginal* distribution. The MMPP2 process has two hidden regimes emitting
// Poisson(lambda_low) and Poisson(lambda_high); under stationarity the regime is
// occupied with probabilities `prob_low`/`prob_high` (computed from `demand_p00`
// / `demand_p11`). The stationary marginal per-period demand PMF is the mixture
//   P(d) = prob_low * Poisson(d; lambda_low) + prob_high * Poisson(d; lambda_high)
// truncated component-wise with the same tail cutoff as the IID Poisson support
// and renormalised. The heuristic order-quantity math then treats this marginal
// as if it were an IID demand law: the multi-period cumulative CDF used by SVBS
// is the `periods`-fold self-convolution of this marginal (i.e. assuming the
// regime is independently re-sampled each period). This is a deliberate
// approximation — it ignores the regime autocorrelation — and is what makes the
// heuristic's order quantities computable on a closed-form demand law. The
// resulting numbers are repo-computed (NOT taken from the literature, which only
// covers IID demand). Crucially, only the *order quantity* uses this marginal;
// the rollout cost is always measured on the true autocorrelated MMPP2 process,
// so the reported mean cost is a valid "heuristic-evaluated-on-true-environment"
// number.

use statrs::distribution::{DiscreteCDF, Poisson};

use crate::problems::lost_sales::demand::{LostSalesDemandConfig, LostSalesDemandKind};

/// Tail mass allowed to be dropped when truncating the demand support.
pub const DEMAND_SUPPORT_TAIL_MASS: f64 = 1e-14;

/// Build the per-period IID demand support `{(demand, probability)}` for the
/// configured demand distribution.
pub fn iid_demand_support(config: &LostSalesDemandConfig) -> Result<Vec<(usize, f64)>, String> {
    match config.kind {
        LostSalesDemandKind::Poisson => truncated_poisson_support(config.demand_rate),
        LostSalesDemandKind::Geometric => truncated_geometric_support(config.demand_rate),
        LostSalesDemandKind::MarkovModulatedPoisson2 => {
            markov_modulated_poisson2_stationary_marginal_support(config)
        }
    }
}

/// Truncated, normalised stationary marginal PMF of the MMPP2 demand process.
///
/// The stationary marginal mixes the two regime Poisson laws by their stationary
/// occupancy probabilities:
///   P(d) = prob_low * Poisson(d; lambda_low) + prob_high * Poisson(d; lambda_high)
/// Each Poisson component is built with the same truncation logic as
/// `truncated_poisson_support` (tail mass `DEMAND_SUPPORT_TAIL_MASS`), the two
/// truncated components are mixed over a common support, and the mixture is
/// renormalised exactly as the IID supports are.
pub fn markov_modulated_poisson2_stationary_marginal_support(
    config: &LostSalesDemandConfig,
) -> Result<Vec<(usize, f64)>, String> {
    let prob_low = config.stationary_prob_low()?;
    let prob_high = config.stationary_prob_high()?;

    // Build each regime's truncated Poisson PMF, then mix on a common support.
    let low_support = truncated_poisson_support(config.demand_lambda_low)?;
    let high_support = truncated_poisson_support(config.demand_lambda_high)?;

    let max_demand = low_support
        .last()
        .map(|(d, _)| *d)
        .unwrap_or(0)
        .max(high_support.last().map(|(d, _)| *d).unwrap_or(0));

    let mut mixed = vec![0.0_f64; max_demand + 1];
    for (demand, probability) in &low_support {
        mixed[*demand] += prob_low * probability;
    }
    for (demand, probability) in &high_support {
        mixed[*demand] += prob_high * probability;
    }

    let mut support: Vec<(usize, f64)> = mixed
        .into_iter()
        .enumerate()
        .filter(|(_, probability)| *probability > 0.0)
        .collect();
    normalize_support(&mut support)?;
    Ok(support)
}

/// Truncated, normalised Poisson PMF with the given mean.
pub fn truncated_poisson_support(mean: f64) -> Result<Vec<(usize, f64)>, String> {
    if !mean.is_finite() || mean < 0.0 {
        return Err(String::from("Poisson mean must be finite and non-negative"));
    }
    if mean == 0.0 {
        return Ok(vec![(0, 1.0)]);
    }

    let mut support = Vec::new();
    let mut probability = (-mean).exp();
    let mut cumulative = probability;
    support.push((0, probability));

    for demand in 1..=10_000usize {
        if cumulative >= 1.0 - DEMAND_SUPPORT_TAIL_MASS {
            break;
        }
        probability *= mean / demand as f64;
        support.push((demand, probability));
        cumulative += probability;
    }

    normalize_support(&mut support)?;
    Ok(support)
}

/// Truncated, normalised Geometric PMF (on `{0, 1, 2, ...}`) with the given mean.
pub fn truncated_geometric_support(mean: f64) -> Result<Vec<(usize, f64)>, String> {
    if !mean.is_finite() || mean < 0.0 {
        return Err(String::from(
            "Geometric mean must be finite and non-negative",
        ));
    }

    let success_probability = 1.0 / (1.0 + mean);
    let mut support = Vec::new();
    let mut probability = success_probability;
    let mut cumulative = probability;
    support.push((0, probability));

    for demand in 1..=100_000usize {
        if cumulative >= 1.0 - DEMAND_SUPPORT_TAIL_MASS {
            break;
        }
        probability *= 1.0 - success_probability;
        support.push((demand, probability));
        cumulative += probability;
    }

    normalize_support(&mut support)?;
    Ok(support)
}

fn normalize_support(support: &mut [(usize, f64)]) -> Result<(), String> {
    let total_probability = support
        .iter()
        .map(|(_, probability)| *probability)
        .sum::<f64>();
    if !total_probability.is_finite() || total_probability <= 0.0 {
        return Err(String::from(
            "demand support probabilities must sum to a positive value",
        ));
    }
    for (_, probability) in support.iter_mut() {
        *probability /= total_probability;
    }
    Ok(())
}

/// CDF of the demand summed over `periods` consecutive periods, evaluated at `k`.
pub fn cumulative_demand_cdf(
    config: &LostSalesDemandConfig,
    k: usize,
    periods: usize,
) -> Result<f64, String> {
    if periods < 1 {
        return Err(String::from("periods must be at least 1"));
    }
    match config.kind {
        LostSalesDemandKind::Poisson => {
            let distribution = Poisson::new(periods as f64 * config.demand_rate).map_err(|err| {
                format!(
                    "invalid Poisson mean {}: {err}",
                    periods as f64 * config.demand_rate
                )
            })?;
            Ok(distribution.cdf(k as u64))
        }
        LostSalesDemandKind::Geometric => {
            cumulative_geometric_sum_cdf(config.demand_rate, k, periods)
        }
        LostSalesDemandKind::MarkovModulatedPoisson2 => {
            markov_modulated_poisson2_stationary_sum_cdf(config, k, periods)
        }
    }
}

/// CDF at `k` of the sum of `periods` IID draws from the MMPP2 *stationary
/// marginal* demand law.
///
/// Consistent with the rest of the heuristic, the multi-period lead-time demand
/// is approximated as a sum of independent copies of the stationary marginal
/// PMF (`markov_modulated_poisson2_stationary_marginal_support`). The sum
/// distribution is obtained by `periods - 1` discrete self-convolutions of the
/// marginal PMF, and the CDF is the cumulative mass up to and including `k`.
fn markov_modulated_poisson2_stationary_sum_cdf(
    config: &LostSalesDemandConfig,
    k: usize,
    periods: usize,
) -> Result<f64, String> {
    let marginal = markov_modulated_poisson2_stationary_marginal_support(config)?;
    // Dense single-period PMF indexed by demand value.
    let single_max = marginal.last().map(|(d, _)| *d).unwrap_or(0);
    let mut single = vec![0.0_f64; single_max + 1];
    for (demand, probability) in &marginal {
        single[*demand] = *probability;
    }

    // Self-convolve `periods` times.
    let mut sum_pmf = single.clone();
    for _ in 1..periods {
        let mut next = vec![0.0_f64; sum_pmf.len() + single.len() - 1];
        for (i, p_sum) in sum_pmf.iter().enumerate() {
            if *p_sum == 0.0 {
                continue;
            }
            for (j, p_single) in single.iter().enumerate() {
                next[i + j] += p_sum * p_single;
            }
        }
        sum_pmf = next;
    }

    let cumulative: f64 = sum_pmf.iter().take(k + 1).sum();
    Ok(cumulative.clamp(0.0, 1.0))
}

fn cumulative_geometric_sum_cdf(mean: f64, k: usize, periods: usize) -> Result<f64, String> {
    if !mean.is_finite() || mean < 0.0 {
        return Err(String::from(
            "Geometric mean must be finite and non-negative",
        ));
    }
    let success_probability = 1.0 / (1.0 + mean);
    let failure_probability = 1.0 - success_probability;
    let mut probability = success_probability.powi(periods as i32);
    let mut cumulative = probability;

    for demand in 1..=k {
        probability *= ((periods + demand - 1) as f64 / demand as f64) * failure_probability;
        cumulative += probability;
    }

    Ok(cumulative.clamp(0.0, 1.0))
}
