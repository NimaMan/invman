//! # `ppo_environment` — OWMR as a batched PPO environment
//!
//! ## Objective
//! Expose the one-warehouse-multi-retailer dynamics as a `PpoVecEnv` so the
//! reusable Rust PPO trainer (`core::ppo`) can train an actor-critic on it. The
//! scalar-cost rollout oracle (`rollout.rs`) collapses an episode into one number
//! and never surfaces per-step transitions; this wraps the SAME canonical
//! dynamics (`env::step_state` + the allocation functions + the echelon gate) as a
//! stateful, steppable, batched environment, so it is faithful by construction
//! (it calls the exact source-of-truth `step_state`, not a re-derived replica).
//!
//! ## Faithful to the reference PPO baseline
//! Matches `scripts/.../ppo_baseline/batched_env.py` + `ppo_owmr.py`:
//! - **Observation** = `observe_raw`: absolute (unnormalized) features
//!   `[wh_inv, wh_pipe.., wh_position, ret_inv.., ret_pipe(flat).., ret_positions.., remaining_periods]`
//!   (dim `1 + wh_L + 1 + K + sum(ret_L) + K + 1`; 45 for instance_14). The
//!   trainer's running normalizer standardizes it.
//! - **Action** = DIRECT orders `[warehouse_order, retailer_order_1..K]`, each a
//!   Categorical over `{0..max_values_j}` (category index == order quantity). The
//!   env rations infeasible joint retailer orders against warehouse release
//!   capacity (random-sequential while training, proportional at eval — the
//!   protocol's rule), so any sampled joint action is feasible and the actor is
//!   never penalized for over-ordering (Kaynov's RandomSequential trainability).
//! - **Reward** = `-period_cost` (the trainer divides by `reward_scale`).
//! - **Gate** (for behavior-clone warm-start) = echelon base-stock orders at
//!   `(gate_w, gate_r)` via the exact `echelon_base_stock_orders` binding.
//! - **Demand** is pre-sampled per `reset(seed)` from the instance's demand
//!   models, so a held-out seed reproduces identical paths (common random
//!   numbers); training resets use a fresh per-iteration seed.
//! - **Initial state** = mean-filled warm start (every on-hand/pipeline slot set
//!   to the rounded one-period mean demand), matching `common.benchmark_initial_state`.
//!   The effective (clip-aware) mean is estimated empirically from the instance's
//!   own `sample_demand`, so it is consistent with the demand the env generates.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::core::ppo::environment::{ActionSpec, PpoVecEnv, StepResult};
use crate::problems::one_warehouse_multi_retailer::allocation::{
    proportional_shipments, random_sequential_shipments,
};
use crate::problems::one_warehouse_multi_retailer::demand::{sample_demand, DemandModel};
use crate::problems::one_warehouse_multi_retailer::env::{
    initialize_state, step_state, CustomerBehaviorModel, OneWarehouseMultiRetailerState,
};
use crate::problems::one_warehouse_multi_retailer::heuristics::echelon_base_stock_orders;
use crate::problems::one_warehouse_multi_retailer::references::get_reference_instance;

/// instance_14 PPO constants (from `ppo_baseline/train_ppo_5seed.py`).
pub const INSTANCE_14_MAX_VALUES: [usize; 11] = [255, 85, 75, 65, 55, 45, 30, 6, 18, 43, 54];
pub const INSTANCE_14_GATE_W: usize = 440;
pub const INSTANCE_14_GATE_R: [usize; 10] = [33, 30, 28, 26, 27, 30, 2, 10, 29, 39];

/// OWMR batched PPO environment.
pub struct OwmrPpoEnv {
    num_envs: usize,
    horizon: usize,
    /// `false` = training (random-sequential rationing); `true` = eval (proportional).
    eval_mode: bool,

    demand_models: Vec<DemandModel>,
    holding_cost_warehouse: f64,
    holding_cost_retailers: Vec<f64>,
    penalty_costs_retailers: Vec<f64>,
    customer_behavior: CustomerBehaviorModel,
    emergency_shipment_probability: f64,
    num_retailers: usize,

    /// Per-head category caps; head `j` is a Categorical over `{0..max_values[j]}`.
    max_values: Vec<usize>,
    gate_w: usize,
    gate_r: Vec<usize>,

    spec: ActionSpec,
    obs_dim: usize,
    initial_state: OneWarehouseMultiRetailerState,

    // Per-episode runtime state.
    states: Vec<OneWarehouseMultiRetailerState>,
    demands: Vec<Vec<Vec<usize>>>, // [B][T][K]
    period: usize,
    alloc_rng: StdRng,
    emerg_rng: StdRng,
}

impl OwmrPpoEnv {
    /// Build the instance_14 (asymmetric 10-retailer, partial-backorder) env.
    pub fn instance_14(num_envs: usize, eval_mode: bool) -> Self {
        Self::from_reference(
            "kaynov2024_instance_14",
            num_envs,
            eval_mode,
            INSTANCE_14_MAX_VALUES.to_vec(),
            INSTANCE_14_GATE_W,
            INSTANCE_14_GATE_R.to_vec(),
        )
    }

    /// Build a PPO env from a named published reference instance plus the PPO
    /// action caps and the gate echelon base-stock levels.
    pub fn from_reference(
        reference_name: &str,
        num_envs: usize,
        eval_mode: bool,
        max_values: Vec<usize>,
        gate_w: usize,
        gate_r: Vec<usize>,
    ) -> Self {
        let reference = get_reference_instance(reference_name)
            .unwrap_or_else(|| panic!("unknown OWMR reference instance '{reference_name}'"));
        let demand_models = reference.demand_models.to_vec();
        let num_retailers = demand_models.len();
        assert_eq!(
            max_values.len(),
            num_retailers + 1,
            "max_values must be warehouse + K retailer caps"
        );
        assert_eq!(gate_r.len(), num_retailers, "gate_r must have K levels");

        let warehouse_lead_time = reference.warehouse_lead_time;
        let retailer_lead_times = reference.retailer_lead_times.to_vec();
        let horizon = reference.benchmark_periods;

        // observe_raw layout: 1 + wh_L + 1 + K + sum(ret_L) + K + 1.
        let obs_dim = 1
            + warehouse_lead_time
            + 1
            + num_retailers
            + retailer_lead_times.iter().sum::<usize>()
            + num_retailers
            + 1;

        let initial_state = mean_filled_initial_state(
            &demand_models,
            warehouse_lead_time,
            &retailer_lead_times,
        );

        let head_sizes: Vec<usize> = max_values.iter().map(|m| m + 1).collect();
        let spec = ActionSpec::MultiDiscrete { sizes: head_sizes };

        OwmrPpoEnv {
            num_envs,
            horizon,
            eval_mode,
            demand_models,
            holding_cost_warehouse: reference.holding_cost_warehouse,
            holding_cost_retailers: reference.holding_cost_retailers.to_vec(),
            penalty_costs_retailers: reference.penalty_costs_retailers.to_vec(),
            customer_behavior: reference.customer_behavior,
            emergency_shipment_probability: reference.emergency_shipment_probability,
            num_retailers,
            max_values,
            gate_w,
            gate_r,
            spec,
            obs_dim,
            initial_state,
            states: Vec::new(),
            demands: Vec::new(),
            period: 0,
            alloc_rng: StdRng::seed_from_u64(0),
            emerg_rng: StdRng::seed_from_u64(0),
        }
    }

    /// `observe_raw` features for one state.
    fn observe_raw_one(&self, s: &OneWarehouseMultiRetailerState) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.obs_dim);
        v.push(s.warehouse_inventory as f32);
        for &p in &s.warehouse_pipeline {
            v.push(p as f32);
        }
        let wh_pos = s.warehouse_inventory + s.warehouse_pipeline.iter().sum::<usize>() as i32;
        v.push(wh_pos as f32);
        for &inv in &s.retailer_inventory {
            v.push(inv as f32);
        }
        for pipe in &s.retailer_pipeline {
            for &p in pipe {
                v.push(p as f32);
            }
        }
        for (inv, pipe) in s.retailer_inventory.iter().zip(&s.retailer_pipeline) {
            let pos = inv + pipe.iter().sum::<usize>() as i32;
            v.push(pos as f32);
        }
        v.push((self.horizon - self.period) as f32);
        v
    }

    fn observe(&self) -> Vec<Vec<f32>> {
        self.states.iter().map(|s| self.observe_raw_one(s)).collect()
    }

    /// Ration retailer orders against warehouse release capacity, using the
    /// allocation rule for the current mode (eval -> proportional, train ->
    /// random-sequential).
    fn ration(&mut self, state: &OneWarehouseMultiRetailerState, retailer_orders: &[usize]) -> Vec<usize> {
        let release_capacity =
            (state.warehouse_inventory + state.warehouse_pipeline[0] as i32).max(0) as usize;
        if self.eval_mode {
            proportional_shipments(release_capacity, retailer_orders)
                .expect("proportional rationing failed")
        } else {
            random_sequential_shipments(&mut self.alloc_rng, release_capacity, retailer_orders)
                .expect("random-sequential rationing failed")
        }
    }
}

/// Estimate the effective (clip-aware) one-period mean demand for each model by
/// sampling its own `sample_demand`, then build the mean-filled warm-start state.
fn mean_filled_initial_state(
    demand_models: &[DemandModel],
    warehouse_lead_time: usize,
    retailer_lead_times: &[usize],
) -> OneWarehouseMultiRetailerState {
    const SAMPLES: usize = 100_000;
    // Fixed seed so the mean-filled warm start is deterministic across env builds.
    let mut rng = StdRng::seed_from_u64(0x4F_57_4D_52_4D_45_41_4E);
    let retailer_means: Vec<i32> = demand_models
        .iter()
        .map(|model| {
            let mut total = 0u64;
            for _ in 0..SAMPLES {
                total += sample_demand(&mut rng, model).expect("sample_demand failed") as u64;
            }
            ((total as f64 / SAMPLES as f64).round()) as i32
        })
        .collect();
    let warehouse_mean: i32 = retailer_means.iter().sum::<i32>();

    let warehouse_pipeline = vec![warehouse_mean.max(0) as usize; warehouse_lead_time];
    let retailer_pipeline: Vec<Vec<usize>> = retailer_lead_times
        .iter()
        .enumerate()
        .map(|(idx, &lead)| vec![retailer_means[idx].max(0) as usize; lead])
        .collect();

    initialize_state(
        warehouse_mean,
        &warehouse_pipeline,
        &retailer_means,
        &retailer_pipeline,
    )
    .expect("mean-filled initial state must be valid")
}

impl PpoVecEnv for OwmrPpoEnv {
    fn num_envs(&self) -> usize {
        self.num_envs
    }
    fn obs_dim(&self) -> usize {
        self.obs_dim
    }
    fn horizon(&self) -> usize {
        self.horizon
    }
    fn action_spec(&self) -> &ActionSpec {
        &self.spec
    }
    fn set_eval_mode(&mut self, eval: bool) {
        self.eval_mode = eval;
    }

    fn reset(&mut self, seed: u64) -> Vec<Vec<f32>> {
        let mut demand_rng = StdRng::seed_from_u64(seed);
        // Pre-sample demands[B][T][K] for common-random-numbers reproducibility.
        self.demands = (0..self.num_envs)
            .map(|_| {
                (0..self.horizon)
                    .map(|_| {
                        self.demand_models
                            .iter()
                            .map(|m| sample_demand(&mut demand_rng, m).expect("sample_demand"))
                            .collect::<Vec<usize>>()
                    })
                    .collect::<Vec<Vec<usize>>>()
            })
            .collect();
        self.alloc_rng = StdRng::seed_from_u64(seed ^ 0xA5A5_5A5A_A5A5_5A5A);
        self.emerg_rng = StdRng::seed_from_u64(seed ^ 0x5A5A_A5A5_5A5A_A5A5);
        self.states = vec![self.initial_state.clone(); self.num_envs];
        self.period = 0;
        self.observe()
    }

    fn step(&mut self, actions: &[Vec<i64>]) -> StepResult {
        let k = self.num_retailers;
        let mut costs = Vec::with_capacity(self.num_envs);
        let mut next_states = Vec::with_capacity(self.num_envs);
        for e in 0..self.num_envs {
            let state = self.states[e].clone();
            // Clamp the sampled categories to the valid order range per head.
            let orders: Vec<usize> = (0..=k)
                .map(|j| (actions[e][j].max(0) as usize).min(self.max_values[j]))
                .collect();
            let shipments = self.ration(&state, &orders[1..]);
            let emergency_draws: Option<Vec<bool>> =
                if self.customer_behavior == CustomerBehaviorModel::PartialBackorder {
                    Some(
                        (0..k)
                            .map(|_| self.emerg_rng.gen_bool(self.emergency_shipment_probability))
                            .collect(),
                    )
                } else {
                    None
                };
            let outcome = step_state(
                &state,
                orders[0],
                &shipments,
                &self.demands[e][self.period],
                self.holding_cost_warehouse,
                &self.holding_cost_retailers,
                &self.penalty_costs_retailers,
                self.customer_behavior,
                self.emergency_shipment_probability,
                emergency_draws.as_deref(),
            )
            .expect("step_state failed");
            costs.push(outcome.period_cost);
            next_states.push(outcome.next_state);
        }
        self.states = next_states;
        self.period += 1;
        let next_obs = self.observe();
        StepResult { costs, next_obs }
    }

    fn gate_actions(&self) -> Option<Vec<Vec<i64>>> {
        Some(
            self.states
                .iter()
                .map(|s| {
                    echelon_base_stock_orders(s, self.gate_w, &self.gate_r)
                        .expect("echelon_base_stock_orders failed")
                        .into_iter()
                        .map(|o| o as i64)
                        .collect()
                })
                .collect(),
        )
    }
}

/// Roll the echelon base-stock GATE policy through the env (eval-mode dynamics)
/// on a held-out seed and return the mean per-episode cost. This is the
/// in-protocol anchor PPO is compared against (PPO converges toward the gate),
/// and a fidelity check that the env reproduces the known Rust gate (~50,445).
pub fn gate_holdout_mean_cost(num_paths: usize, seed: u64) -> f64 {
    let mut env = OwmrPpoEnv::instance_14(num_paths, true);
    let _ = env.reset(seed);
    let mut cost_total = vec![0f64; num_paths];
    for _t in 0..env.horizon() {
        let gate = env.gate_actions().expect("gate available");
        let step = env.step(&gate);
        for e in 0..num_paths {
            cost_total[e] += step.costs[e];
        }
    }
    cost_total.iter().sum::<f64>() / num_paths as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owmr_env_shapes_are_correct() {
        let mut env = OwmrPpoEnv::instance_14(4, true);
        let obs = env.reset(900_000);
        assert_eq!(env.obs_dim(), 45, "instance_14 observe_raw is 45-dim");
        assert_eq!(obs.len(), 4);
        assert_eq!(obs[0].len(), 45);
        match env.action_spec() {
            ActionSpec::MultiDiscrete { sizes } => {
                assert_eq!(sizes.len(), 11, "K+1 = 11 heads");
                assert_eq!(sizes[0], 256, "warehouse head size = max+1");
                assert_eq!(sizes[7], 7, "retailer-7 head size = 6+1");
            }
            _ => panic!("expected multi-discrete"),
        }
        // A gate action vector has K+1 entries.
        let gate = env.gate_actions().expect("gate");
        assert_eq!(gate[0].len(), 11);
    }

    #[test]
    fn owmr_gate_holdout_cost_matches_known_rust_gate() {
        // The reference Rust gate (proportional, holdout) is ~50,445. The
        // steppable env reuses step_state, so it must reproduce that within MC
        // noise + initial-state/demand-seeding differences.
        let mean = gate_holdout_mean_cost(1024, 900_000);
        println!("OWMR gate holdout mean cost (1024 paths): {mean:.2}");
        assert!(
            (45_000.0..=56_000.0).contains(&mean),
            "gate holdout cost {mean:.1} is implausibly far from the known ~50,445"
        );
    }

    /// Short end-to-end smoke: confirms the OWMR env + candle PPO trainer run to
    /// completion and produce a sane holdout-greedy cost near the gate. Reduced
    /// sizes so it finishes quickly. Heavy for debug; run in release:
    ///   cargo test --release --features ppo -- --ignored owmr_ppo_smoke
    #[test]
    #[ignore]
    fn owmr_ppo_smoke() {
        use crate::core::ppo::ppo_trainer::{train_ppo, PpoConfig};
        let gate = gate_holdout_mean_cost(256, 900_000);
        let cfg = PpoConfig {
            iters: 4,
            train_paths: 64,
            eval_paths: 256,
            hidden: 128,
            lr: 1.2e-4,
            gamma: 1.0,
            lam: 0.95,
            clip: 0.15,
            ppo_epochs: 3,
            minibatch: 2048,
            vf_coef: 0.5,
            ent_coef: 0.001,
            max_grad_norm: 0.5,
            reward_scale: 1000.0,
            bc_epochs: 15,
            bc_paths: 128,
            bc_lr: 1e-3,
            bc_batch: 2048,
            eval_every: 2,
            seed: 0,
            train_seed_start: 600_000,
            holdout_seed_start: 900_000,
            search_seed_start: 500_000,
            verbose: true,
        };
        let make_env =
            |n: usize, eval: bool| Box::new(OwmrPpoEnv::instance_14(n, eval)) as Box<dyn PpoVecEnv>;
        let outcome = train_ppo(make_env, &cfg).expect("OWMR PPO training failed");
        println!(
            "[SMOKE] gate={gate:.1} | PPO best={:.1} final={:.1}",
            outcome.best_holdout_cost, outcome.final_holdout_cost_mean
        );
        // After BC the policy clones the gate, so cost must be near it (not diverged).
        assert!(
            outcome.best_holdout_cost < gate * 1.20,
            "PPO best {:.1} diverged far above gate {gate:.1}",
            outcome.best_holdout_cost
        );
    }

    /// BC-only convergence check: with the reference BC budget, behavior cloning
    /// should drive the greedy holdout cost down to ~the gate (~50,475), since the
    /// gate IS the BC target. Isolates BC from PPO (iters=0). Run in release:
    ///   cargo test --release --features ppo -- --ignored owmr_bc_clones_gate
    #[test]
    #[ignore]
    fn owmr_bc_clones_gate() {
        use crate::core::ppo::ppo_trainer::{train_ppo, PpoConfig};
        let gate = gate_holdout_mean_cost(1024, 900_000);
        let cfg = PpoConfig {
            iters: 0,
            train_paths: 64,
            eval_paths: 1024,
            hidden: 128,
            lr: 1.2e-4,
            gamma: 1.0,
            lam: 0.95,
            clip: 0.15,
            ppo_epochs: 5,
            minibatch: 4096,
            vf_coef: 0.5,
            ent_coef: 0.001,
            max_grad_norm: 0.5,
            reward_scale: 1000.0,
            bc_epochs: 120,
            bc_paths: 512,
            bc_lr: 1e-3,
            bc_batch: 2048,
            eval_every: 5,
            seed: 0,
            train_seed_start: 600_000,
            holdout_seed_start: 900_000,
            search_seed_start: 500_000,
            verbose: true,
        };
        let make_env =
            |n: usize, eval: bool| Box::new(OwmrPpoEnv::instance_14(n, eval)) as Box<dyn PpoVecEnv>;
        let outcome = train_ppo(make_env, &cfg).expect("BC failed");
        println!(
            "[BC-ONLY] gate={gate:.1} | after-BC greedy holdout = {:.1}",
            outcome.final_holdout_cost_mean
        );
        assert!(
            outcome.final_holdout_cost_mean < gate * 1.05,
            "BC did not clone the gate: after-BC {:.1} vs gate {gate:.1}",
            outcome.final_holdout_cost_mean
        );
    }

    /// Full validation: reproduce the in-house PyTorch PPO (~50,475) with the
    /// 5-seed hyperparameters. Heavy; run a single seed in release:
    ///   cargo test --release --features ppo -- --ignored owmr_ppo_reproduces
    #[test]
    #[ignore]
    fn owmr_ppo_reproduces_inhouse_pytorch_ppo() {
        use crate::core::ppo::ppo_trainer::{train_ppo, PpoConfig};
        let gate = gate_holdout_mean_cost(1024, 900_000);
        let cfg = PpoConfig {
            iters: 60,
            train_paths: 384,
            eval_paths: 1024,
            hidden: 128,
            lr: 1.2e-4,
            gamma: 1.0,
            lam: 0.95,
            clip: 0.15,
            ppo_epochs: 5,
            minibatch: 4096,
            vf_coef: 0.5,
            ent_coef: 0.001,
            max_grad_norm: 0.5,
            reward_scale: 1000.0,
            bc_epochs: 120,
            bc_paths: 512,
            bc_lr: 1e-3,
            bc_batch: 2048,
            eval_every: 5,
            seed: 0,
            train_seed_start: 600_000,
            holdout_seed_start: 900_000,
            search_seed_start: 500_000,
            verbose: true,
        };
        let make_env =
            |n: usize, eval: bool| Box::new(OwmrPpoEnv::instance_14(n, eval)) as Box<dyn PpoVecEnv>;
        let outcome = train_ppo(make_env, &cfg).expect("OWMR PPO training failed");
        println!(
            "[REPRODUCE] gate={gate:.1} | PPO best={:.1} final={:.1} (PyTorch PPO ref ~50,475)",
            outcome.best_holdout_cost, outcome.final_holdout_cost_mean
        );
        // PPO BC-starts at the gate and stays near it (the documented behavior);
        // a faithful trainer lands within a few % of the gate / in-house PPO.
        assert!(
            outcome.best_holdout_cost < gate * 1.10 && outcome.best_holdout_cost > gate * 0.80,
            "PPO best {:.1} not near gate {gate:.1}",
            outcome.best_holdout_cost
        );
    }
}
