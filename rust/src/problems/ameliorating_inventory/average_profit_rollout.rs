#![allow(dead_code)]

//! Soft-tree population rollout for the FAITHFUL average-profit ameliorating env
//! (`average_profit_blending_env.rs`, the Pahr & Grunow 2025 model). Python bridge.
//!
//! OBJECTIVE
//! ---------
//! Score a CMA-ES soft-tree policy on the faithful long-run AVERAGE-PROFIT
//! ameliorating-inventory env, returning per-individual mean per-period profit
//! under paired common-random-numbers. The literature anchor is the
//! perfect-information LP UPPER BOUND on average profit (`perfect_information_lp.rs`:
//! spirits_0001 = 1991.9344, port_wine = 2444.8011); the report is the GAP-TO-BOUND.
//! The ROLLOUT runs in Rust; this module is only the call-bridge that decodes a flat
//! soft-tree parameter vector into the env action and evaluates it.
//!
//! ACTION GEOMETRY (the policy)
//! ----------------------------
//! In the faithful env `step_state` the controllable decision is the scalar PURCHASE
//! volume `aP in [0, maxInventory]` (the issuance plan is solved by the per-period
//! blending LP and production is derived from it — the "3-part action" is structural,
//! only the purchase is a free control, matching the companion env where production /
//! issuance follow from the purchase + LP). The policy therefore carries a single
//! continuous purchase head over the price-augmented state. The strongest simple
//! heuristic is an ORDER-UP-TO purchase: buy toward a target total on-hand level,
//!   purchase = clip(S_target - sum(inventory_position), 0, maxInventory),
//! which a depth>=1 linear-leaf tree expresses exactly with bias = S_target and
//! per-inventory weight = -1 (and zero weight on price). Warm-starting there makes
//! generation 0 reproduce the order-up-to heuristic; the optimizer refines a
//! price-reactive purchase (buy more when the realised price is low).
//!
//! PER-PERIOD SEQUENCE
//! -------------------
//!   1. read the policy state [price, inventory_position[0..A]] (normalised by
//!      maxInventory) and decode the scalar purchase volume from the soft tree;
//!   2. `step_state(rng, config, state, purchase)` runs the faithful transition
//!      (LP issuance -> production -> outdating -> demand/sales -> age+purchase ->
//!      Beta decay -> reward) and returns the per-period profit;
//!   3. average the per-period reward after a warm-up (long-run average profit).

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::core::policies::soft_tree::{
    action_vector_continuous_from_flat_params, SoftTreeLeafType, SoftTreeSplitType,
};
use crate::problems::ameliorating_inventory::average_profit_blending_env::{
    initialize_state, step_state, AverageProfitBlendingConfig,
};

#[derive(Clone)]
pub struct AverageProfitRolloutConfig {
    pub env: AverageProfitBlendingConfig,
    /// Initial inventory position (age 0..A); typically the LP steady-state.
    pub initial_inventory: Vec<f64>,
    pub depth: usize,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
    pub periods: usize,
    pub warm_up: usize,
}

impl AverageProfitRolloutConfig {
    /// Policy input dimension = price + per-age inventory position = 1 + num_ages.
    pub fn input_dim(&self) -> usize {
        1 + self.env.num_ages
    }
}

fn validate(config: &AverageProfitRolloutConfig) -> PyResult<()> {
    if config.env.num_ages < 1 {
        return Err(PyValueError::new_err("num_ages must be at least 1"));
    }
    if config.initial_inventory.len() != config.env.num_ages {
        return Err(PyValueError::new_err(
            "initial_inventory length must match num_ages",
        ));
    }
    if config.periods < 1 {
        return Err(PyValueError::new_err("periods must be positive"));
    }
    if config.warm_up >= config.periods {
        return Err(PyValueError::new_err("warm_up must be < periods"));
    }
    Ok(())
}

/// Build the normalised policy state [price, inventory_position[0..A]] / maxInventory.
fn policy_state(
    price: f64,
    inventory_position: &[f64],
    max_inventory: f64,
) -> Vec<f32> {
    let scale = max_inventory.max(1.0);
    let mut v = Vec::with_capacity(1 + inventory_position.len());
    v.push((price / scale) as f32);
    for &inv in inventory_position {
        v.push((inv / scale) as f32);
    }
    v
}

/// One rollout: mean per-period profit after warm-up under the decoded soft-tree
/// purchase policy.
pub fn rollout(
    flat_params: &[f32],
    config: &AverageProfitRolloutConfig,
    seed: u64,
) -> PyResult<f64> {
    validate(config)?;
    let input_dim = config.input_dim();
    let max_inventory = config.env.max_inventory;

    let mut state = initialize_state(&config.env, &config.initial_inventory);
    let mut rng = StdRng::seed_from_u64(seed);

    // The purchase head spans [0, maxInventory]; for a linear leaf only the lower
    // bound (0) matters (softplus is unbounded above, clamped by step_state).
    let level_min = vec![0.0f32];
    let level_max = vec![max_inventory as f32];

    let (mut total, mut counted) = (0.0f64, 0usize);
    for t in 0..config.periods {
        let ps = policy_state(state.price, &state.inventory_position, max_inventory);
        if ps.len() != input_dim {
            return Err(PyValueError::new_err(format!(
                "policy state width {} does not match input_dim {}",
                ps.len(),
                input_dim
            )));
        }
        let action = action_vector_continuous_from_flat_params(
            &ps,
            flat_params,
            input_dim,
            config.depth,
            config.temperature,
            config.split_type,
            config.leaf_type,
            &level_min,
            &level_max,
        )?;
        let purchase = (action[0] as f64).clamp(0.0, max_inventory);
        let outcome = step_state(&mut rng, &config.env, &state, purchase);
        if t >= config.warm_up {
            total += outcome.reward;
            counted += 1;
        }
        state = outcome.next_state;
    }
    Ok(total / counted as f64)
}

/// Paired population rollout: one mean profit per (params, seed) pair.
pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &AverageProfitRolloutConfig,
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
