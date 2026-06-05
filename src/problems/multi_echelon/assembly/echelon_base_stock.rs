#![allow(dead_code)]

//! Echelon base-stock policy + Monte-Carlo evaluation for the assembly env.
//!
//! The optimal echelon base-stock levels come from the Rosling serial equivalent
//! (`rosling.rs` + `serial::exact`). `simulate` runs that policy on the assembly `env.rs`
//! so the simulated long-run average cost can be checked against the analytical optimum.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RNormal, Poisson as RPoisson};

use crate::problems::multi_echelon::assembly::env::{
    consume, initialize_at_echelon_levels, replenish, AssemblyConfig,
};
use crate::problems::multi_echelon::serial::exact::SerialDemand;

#[derive(Clone, Debug, PartialEq)]
pub struct AssemblySimResult {
    pub average_cost: f64,
    pub average_holding_cost: f64,
    pub average_backorder_cost: f64,
    pub measured_periods: usize,
}

/// Simulate the assembly env under the echelon base-stock policy with levels
/// `[S_finished, S_kit]` (downstream -> upstream) and return the mean per-period cost.
pub fn simulate(
    config: &AssemblyConfig,
    demand: SerialDemand,
    echelon_levels: &[f64],
    periods: usize,
    warm_up: usize,
    seed: u64,
) -> AssemblySimResult {
    let mean = match demand {
        SerialDemand::Normal { mean, .. } => mean,
        SerialDemand::Poisson { mean } => mean,
    };
    let mut state = initialize_at_echelon_levels(config, echelon_levels, mean);

    let mut rng = StdRng::seed_from_u64(seed);
    let poisson = RPoisson::new(mean.max(1e-9)).ok();
    let normal = match demand {
        SerialDemand::Normal { mean, std } => RNormal::new(mean, std).ok(),
        _ => None,
    };

    let (mut total, mut holding, mut backorder, mut counted) = (0.0, 0.0, 0.0, 0usize);
    for t in 0..periods {
        let d = match demand {
            SerialDemand::Poisson { .. } => poisson.as_ref().unwrap().sample(&mut rng),
            SerialDemand::Normal { .. } => {
                normal.as_ref().unwrap().sample(&mut rng).round().max(0.0)
            }
        };
        let outcome = consume(config, &mut state, d);
        replenish(config, &mut state, echelon_levels);
        if t >= warm_up {
            total += outcome.period_cost;
            holding += outcome.holding_cost;
            backorder += outcome.backorder_cost;
            counted += 1;
        }
    }
    AssemblySimResult {
        average_cost: total / counted as f64,
        average_holding_cost: holding / counted as f64,
        average_backorder_cost: backorder / counted as f64,
        measured_periods: counted,
    }
}
