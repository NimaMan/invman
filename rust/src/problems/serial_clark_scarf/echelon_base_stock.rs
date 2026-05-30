#![allow(dead_code)]

//! Echelon base-stock policy and Monte-Carlo evaluation for the serial Clark-Scarf env.
//!
//! The echelon base-stock policy is the optimal policy class for the serial system
//! (Clark and Scarf 1960): each stage orders to raise its echelon inventory position to
//! a target level `echelon_levels[k]`. `exact.rs` computes the optimal `echelon_levels`
//! and the optimal cost; `simulate` runs this policy on `env.rs` so the simulated
//! long-run average cost can be checked against that analytical optimum.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RNormal, Poisson as RPoisson};

use crate::problems::serial_clark_scarf::env::{
    consume, echelon_inventory_positions, initialize_at_echelon_levels, replenish, SerialConfig,
    SerialState,
};
use crate::problems::serial_clark_scarf::exact::SerialDemand;

/// Order quantities (downstream -> upstream) that raise each stage's echelon inventory
/// position up to its echelon base-stock level.
pub fn echelon_base_stock_orders(state: &SerialState, echelon_levels: &[f64]) -> Vec<f64> {
    let ip = echelon_inventory_positions(state);
    (0..echelon_levels.len())
        .map(|k| (echelon_levels[k] - ip[k]).max(0.0))
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct SerialSimResult {
    pub average_cost: f64,
    pub average_holding_cost: f64,
    pub average_backorder_cost: f64,
    pub measured_periods: usize,
}

/// Simulate the echelon base-stock policy on the serial env and return the mean
/// per-period cost after a warm-up.
pub fn simulate(
    config: &SerialConfig,
    demand: SerialDemand,
    echelon_levels: &[f64],
    periods: usize,
    warm_up: usize,
    seed: u64,
) -> SerialSimResult {
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
        // Consume (receive + demand + cost), then decide replenishment on the resulting
        // post-demand state -- the convention that matches the literature optimum.
        let outcome = consume(config, &mut state, d);
        let orders = echelon_base_stock_orders(&state, echelon_levels);
        replenish(config, &mut state, &orders);
        if t >= warm_up {
            total += outcome.period_cost;
            holding += outcome.holding_cost;
            backorder += outcome.backorder_cost;
            counted += 1;
        }
    }
    SerialSimResult {
        average_cost: total / counted as f64,
        average_holding_cost: holding / counted as f64,
        average_backorder_cost: backorder / counted as f64,
        measured_periods: counted,
    }
}
