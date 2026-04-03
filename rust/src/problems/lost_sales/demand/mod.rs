pub mod iid;
pub mod markov_modulated;

use rand::rngs::StdRng;

use crate::problems::lost_sales::demand::iid::{GeometricDemand, PoissonDemand};
use crate::problems::lost_sales::demand::markov_modulated::MarkovModulatedPoisson2Demand;

pub const DEFAULT_MMPP2_LAMBDA_LOW: f64 = 3.0;
pub const DEFAULT_MMPP2_LAMBDA_HIGH: f64 = 7.0;
pub const DEFAULT_MMPP2_POSITIVE_P00: f64 = 0.9;
pub const DEFAULT_MMPP2_POSITIVE_P11: f64 = 0.9;
pub const DEFAULT_MMPP2_NEGATIVE_P00: f64 = 0.1;
pub const DEFAULT_MMPP2_NEGATIVE_P11: f64 = 0.1;

#[derive(Clone, Copy)]
pub enum LostSalesDemandKind {
    Poisson,
    Geometric,
    MarkovModulatedPoisson2,
}

#[derive(Clone, Copy)]
pub struct LostSalesDemandConfig {
    pub kind: LostSalesDemandKind,
    pub demand_rate: f64,
    pub demand_lambda_low: f64,
    pub demand_lambda_high: f64,
    pub demand_p00: f64,
    pub demand_p11: f64,
}

pub enum LostSalesDemandProcess {
    Poisson(PoissonDemand),
    Geometric(GeometricDemand),
    MarkovModulatedPoisson2(MarkovModulatedPoisson2Demand),
}

impl LostSalesDemandConfig {
    pub fn stationary_prob_high(&self) -> Result<f64, String> {
        if !matches!(self.kind, LostSalesDemandKind::MarkovModulatedPoisson2) {
            return Ok(0.0);
        }
        let denominator = 2.0 - self.demand_p00 - self.demand_p11;
        if denominator <= 0.0 {
            return Err("demand_p00 + demand_p11 must be strictly less than 2".to_string());
        }
        Ok((1.0 - self.demand_p00) / denominator)
    }

    pub fn stationary_prob_low(&self) -> Result<f64, String> {
        Ok(1.0 - self.stationary_prob_high()?)
    }

    pub fn implied_mean(&self) -> Result<f64, String> {
        match self.kind {
            LostSalesDemandKind::Poisson | LostSalesDemandKind::Geometric => Ok(self.demand_rate),
            LostSalesDemandKind::MarkovModulatedPoisson2 => {
                let stationary_prob_high = self.stationary_prob_high()?;
                let stationary_prob_low = self.stationary_prob_low()?;
                Ok(stationary_prob_low * self.demand_lambda_low
                    + stationary_prob_high * self.demand_lambda_high)
            }
        }
    }

    pub fn lag_one_regime_eigenvalue(&self) -> f64 {
        if matches!(self.kind, LostSalesDemandKind::MarkovModulatedPoisson2) {
            self.demand_p00 + self.demand_p11 - 1.0
        } else {
            0.0
        }
    }

    pub fn one_period_variance(&self) -> Result<f64, String> {
        match self.kind {
            LostSalesDemandKind::Poisson => Ok(self.demand_rate),
            LostSalesDemandKind::Geometric => Ok(self.demand_rate * (1.0 + self.demand_rate)),
            LostSalesDemandKind::MarkovModulatedPoisson2 => {
                let stationary_prob_high = self.stationary_prob_high()?;
                let stationary_prob_low = 1.0 - stationary_prob_high;
                let delta = self.demand_lambda_high - self.demand_lambda_low;
                Ok(self.implied_mean()?
                    + stationary_prob_low * stationary_prob_high * delta * delta)
            }
        }
    }

    pub fn lag_k_autocovariance(&self, lag: usize) -> Result<f64, String> {
        match self.kind {
            LostSalesDemandKind::Poisson | LostSalesDemandKind::Geometric => {
                if lag == 0 {
                    self.one_period_variance()
                } else {
                    Ok(0.0)
                }
            }
            LostSalesDemandKind::MarkovModulatedPoisson2 => {
                if lag == 0 {
                    return self.one_period_variance();
                }
                let stationary_prob_high = self.stationary_prob_high()?;
                let stationary_prob_low = 1.0 - stationary_prob_high;
                let delta = self.demand_lambda_high - self.demand_lambda_low;
                Ok(stationary_prob_low
                    * stationary_prob_high
                    * delta
                    * delta
                    * self.lag_one_regime_eigenvalue().powi(lag as i32))
            }
        }
    }

    pub fn lag_k_autocorrelation(&self, lag: usize) -> Result<f64, String> {
        if lag == 0 {
            return Ok(1.0);
        }
        let variance = self.one_period_variance()?;
        if variance <= 0.0 {
            return Ok(0.0);
        }
        Ok(self.lag_k_autocovariance(lag)? / variance)
    }
}

pub fn parse_demand_kind(name: &str) -> Result<LostSalesDemandKind, String> {
    match name {
        "Poisson" => Ok(LostSalesDemandKind::Poisson),
        "Geometric" => Ok(LostSalesDemandKind::Geometric),
        "MarkovModulatedPoisson2" => Ok(LostSalesDemandKind::MarkovModulatedPoisson2),
        _ => Err(format!(
            "unsupported demand_dist_name '{name}'; expected 'Poisson', 'Geometric', or 'MarkovModulatedPoisson2'"
        )),
    }
}

fn validate_markov_modulated_mean(config: LostSalesDemandConfig) -> Result<(), String> {
    let implied_mean = config.implied_mean()?;
    if (implied_mean - config.demand_rate).abs() > 1e-6 {
        return Err(format!(
            "MarkovModulatedPoisson2 parameters imply stationary mean {implied_mean:.6}, \
but demand_rate={:.6}",
            config.demand_rate
        ));
    }
    Ok(())
}

pub fn build_demand_process(
    config: LostSalesDemandConfig,
    rng: &mut StdRng,
) -> Result<LostSalesDemandProcess, String> {
    match config.kind {
        LostSalesDemandKind::Poisson => {
            PoissonDemand::new(config.demand_rate).map(LostSalesDemandProcess::Poisson)
        }
        LostSalesDemandKind::Geometric => {
            GeometricDemand::new(config.demand_rate).map(LostSalesDemandProcess::Geometric)
        }
        LostSalesDemandKind::MarkovModulatedPoisson2 => {
            validate_markov_modulated_mean(config)?;
            MarkovModulatedPoisson2Demand::new(
                config.demand_lambda_low,
                config.demand_lambda_high,
                config.demand_p00,
                config.demand_p11,
                rng,
            )
            .map(LostSalesDemandProcess::MarkovModulatedPoisson2)
        }
    }
}

pub fn sample_demand(rng: &mut StdRng, demand_process: &mut LostSalesDemandProcess) -> i64 {
    match demand_process {
        LostSalesDemandProcess::Poisson(process) => process.sample(rng),
        LostSalesDemandProcess::Geometric(process) => process.sample(rng),
        LostSalesDemandProcess::MarkovModulatedPoisson2(process) => process.sample(rng),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positive_mean5_config() -> LostSalesDemandConfig {
        LostSalesDemandConfig {
            kind: LostSalesDemandKind::MarkovModulatedPoisson2,
            demand_rate: 5.0,
            demand_lambda_low: DEFAULT_MMPP2_LAMBDA_LOW,
            demand_lambda_high: DEFAULT_MMPP2_LAMBDA_HIGH,
            demand_p00: DEFAULT_MMPP2_POSITIVE_P00,
            demand_p11: DEFAULT_MMPP2_POSITIVE_P11,
        }
    }

    fn negative_mean5_config() -> LostSalesDemandConfig {
        LostSalesDemandConfig {
            kind: LostSalesDemandKind::MarkovModulatedPoisson2,
            demand_rate: 5.0,
            demand_lambda_low: DEFAULT_MMPP2_LAMBDA_LOW,
            demand_lambda_high: DEFAULT_MMPP2_LAMBDA_HIGH,
            demand_p00: DEFAULT_MMPP2_NEGATIVE_P00,
            demand_p11: DEFAULT_MMPP2_NEGATIVE_P11,
        }
    }

    #[test]
    fn iid_configs_have_zero_positive_lag_autocorrelation() {
        let poisson = LostSalesDemandConfig {
            kind: LostSalesDemandKind::Poisson,
            demand_rate: 5.0,
            demand_lambda_low: 0.0,
            demand_lambda_high: 0.0,
            demand_p00: 0.0,
            demand_p11: 0.0,
        };
        let geometric = LostSalesDemandConfig {
            kind: LostSalesDemandKind::Geometric,
            demand_rate: 5.0,
            demand_lambda_low: 0.0,
            demand_lambda_high: 0.0,
            demand_p00: 0.0,
            demand_p11: 0.0,
        };

        assert!((poisson.one_period_variance().unwrap() - 5.0).abs() < 1e-12);
        assert!(poisson.lag_k_autocorrelation(1).unwrap().abs() < 1e-12);
        assert!(poisson.lag_k_autocorrelation(2).unwrap().abs() < 1e-12);

        assert!((geometric.one_period_variance().unwrap() - 30.0).abs() < 1e-12);
        assert!(geometric.lag_k_autocorrelation(1).unwrap().abs() < 1e-12);
        assert!(geometric.lag_k_autocorrelation(2).unwrap().abs() < 1e-12);
    }

    #[test]
    fn positive_and_negative_mmpp2_configs_are_mean_preserving() {
        let positive = positive_mean5_config();
        let negative = negative_mean5_config();

        assert!((positive.implied_mean().unwrap() - 5.0).abs() < 1e-9);
        assert!((negative.implied_mean().unwrap() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn positive_and_negative_mmpp2_configs_have_expected_lag_one_correlation() {
        let positive = positive_mean5_config();
        let negative = negative_mean5_config();

        let positive_rho1 = positive.lag_k_autocorrelation(1).unwrap();
        let negative_rho1 = negative.lag_k_autocorrelation(1).unwrap();

        assert!((positive_rho1 - 3.2 / 9.0).abs() < 1e-12);
        assert!((negative_rho1 + 3.2 / 9.0).abs() < 1e-12);
        assert!(positive_rho1 > 0.0);
        assert!(negative_rho1 < 0.0);
    }
}
