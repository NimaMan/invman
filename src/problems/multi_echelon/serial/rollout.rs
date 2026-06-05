#![allow(dead_code)]

//! Soft-tree population rollout for the serial Clark-Scarf env (Python call-bridge).
//!
//! OBJECTIVE
//! ---------
//! Score a CMA-ES soft-tree policy on the faithful serial multi-echelon env
//! (`env.rs`, the textbook Clark-Scarf model verified against Snyder & Shen
//! Example 6.1, optimal cost 47.65). The ROLLOUT runs entirely in Rust; this
//! module is only the bridge that decodes a flat soft-tree parameter vector into
//! the serial decision and evaluates it under paired common-random-numbers.
//!
//! ACTION GEOMETRY (the policy)
//! ----------------------------
//! The decision class for a serial system is the ECHELON BASE-STOCK policy
//! (Clark & Scarf 1960): each stage k orders to raise its echelon inventory
//! position up to a target level `S_k`. The strongest (here OPTIMAL) heuristic is
//! the exact Clark-Scarf solution, so the policy is encoded in that coordinate
//! system: the soft tree emits the N echelon base-stock LEVELS directly
//! (`direct_level`), continuous and non-negative, bounded only by a generous
//! physical ceiling. Given a per-period post-demand state, the order at each stage
//! is then `max(0, S_k - echelon_IP_k)` — exactly `echelon_base_stock_orders`.
//! Warm-starting the leaves at the exact Clark-Scarf levels makes generation 0
//! reproduce the optimum, the honest ceiling for a MATCH-only problem.
//!
//! PER-PERIOD SEQUENCE (faithful to `echelon_base_stock.rs::simulate`)
//! ------------------------------------------------------------------
//!   1. sample CONTINUOUS Normal demand (the env drops rounding; the exact solver
//!      optimises against continuous Normal, so rounding here inflates cost);
//!   2. `consume`: receive + meet demand + assess holding/backorder cost on the
//!      post-demand state;
//!   3. read the policy state (raw on-hand / pipeline / backorder), decode the N
//!      echelon levels from the soft tree, compute echelon-base-stock orders from
//!      the POST-demand state, and `replenish`.
//! The mean per-period cost after a warm-up is returned (long-run average).
//!
//! PAIRED CRN
//! ----------
//! `population_rollout` evaluates each individual on its own seed; the caller
//! drives the same seed across the population per generation for variance-reduced
//! (paired) comparison, mirroring every other soft-tree rollout binding.

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RNormal};
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_continuous_from_flat_params, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::multi_echelon::serial::env::{
    consume, echelon_inventory_positions, initialize_at_echelon_levels, raw_state_vector, replenish,
    SerialConfig,
};

#[derive(Clone)]
pub struct SerialRolloutConfig {
    pub config: SerialConfig,
    pub demand_mean: f64,
    pub demand_std: f64,
    /// Echelon base-stock levels (downstream -> upstream) used ONLY to warm-fill the
    /// initial state and pipeline at a steady-ish start; the policy decides the
    /// operating levels.
    pub warm_start_levels: Vec<f64>,
    pub depth: usize,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
    /// Per-stage continuous echelon-level bounds (downstream -> upstream).
    pub level_min: Vec<f32>,
    pub level_max: Vec<f32>,
    pub periods: usize,
    pub warm_up: usize,
}

impl SerialRolloutConfig {
    pub fn num_stages(&self) -> usize {
        self.config.num_stages()
    }
    /// Policy input dimension = raw serial state width = 2*N + 1.
    pub fn input_dim(&self) -> usize {
        2 * self.num_stages() + 1
    }
}

fn validate(config: &SerialRolloutConfig) -> PyResult<()> {
    let n = config.num_stages();
    if n < 1 {
        return Err(PyValueError::new_err("serial system needs at least one stage"));
    }
    if config.config.lead_time.len() != n {
        return Err(PyValueError::new_err("lead_time length must match num_stages"));
    }
    if config.warm_start_levels.len() != n {
        return Err(PyValueError::new_err(
            "warm_start_levels length must match num_stages",
        ));
    }
    if config.level_min.len() != n || config.level_max.len() != n {
        return Err(PyValueError::new_err(
            "level_min and level_max length must match num_stages",
        ));
    }
    if config.demand_std < 0.0 {
        return Err(PyValueError::new_err("demand_std must be non-negative"));
    }
    if config.periods < 1 {
        return Err(PyValueError::new_err("periods must be positive"));
    }
    if config.warm_up >= config.periods {
        return Err(PyValueError::new_err("warm_up must be < periods"));
    }
    Ok(())
}

/// One rollout: mean per-period cost (holding + backorder) after warm-up under the
/// decoded soft-tree echelon-base-stock policy.
pub fn rollout(flat_params: &[f32], config: &SerialRolloutConfig, seed: u64) -> PyResult<f64> {
    validate(config)?;
    let n = config.num_stages();
    let input_dim = config.input_dim();

    let mut state =
        initialize_at_echelon_levels(&config.config, &config.warm_start_levels, config.demand_mean);
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = RNormal::new(config.demand_mean, config.demand_std.max(1e-12))
        .map_err(|e| PyValueError::new_err(format!("invalid demand normal: {e}")))?;

    let (mut total, mut counted) = (0.0f64, 0usize);
    for t in 0..config.periods {
        // 1. continuous Normal demand (env drops rounding).
        let d = normal.sample(&mut rng).max(0.0);
        // 2. consume on the post-demand state, charge cost.
        let outcome = consume(&config.config, &mut state, d);
        // 3. decode echelon levels from the post-demand policy state.
        let policy_state = raw_state_vector(&state);
        if policy_state.len() != input_dim {
            return Err(PyValueError::new_err(format!(
                "policy state width {} does not match input_dim {}",
                policy_state.len(),
                input_dim
            )));
        }
        let levels = action_vector_continuous_from_flat_params(
            &policy_state,
            flat_params,
            input_dim,
            config.depth,
            config.temperature,
            config.split_type,
            config.leaf_type,
            &config.level_min,
            &config.level_max,
        )?;
        // 4. echelon-base-stock orders from the post-demand state and replenish.
        let ip = echelon_inventory_positions(&state);
        let orders: Vec<f64> = (0..n)
            .map(|k| (levels[k] as f64 - ip[k]).max(0.0))
            .collect();
        replenish(&config.config, &mut state, &orders);

        if t >= config.warm_up {
            total += outcome.period_cost;
            counted += 1;
        }
    }
    Ok(total / counted as f64)
}

/// Paired population rollout: one cost per (params, seed) pair.
pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &SerialRolloutConfig,
    seeds: &[u64],
) -> PyResult<Vec<f64>> {
    if params_batch.len() != seeds.len() {
        return Err(PyValueError::new_err(
            "params_batch and seeds must have the same length",
        ));
    }
    params_batch
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(flat_params, seed)| rollout(flat_params, config, *seed))
        .collect()
}
