//! # `ppo_environment` — lost_sales (vanilla) as a continuous PPO environment
//!
//! ## Objective
//! Demonstrate that the reusable Rust PPO trainer (`core::ppo`) works across
//! action TYPES, not just OWMR's multi-discrete: lost_sales has a single
//! continuous-scalar order action, trained by the trainer's diagonal-Gaussian
//! head. Wraps the canonical lost_sales dynamics (`vanilla::env::epoch_cost` +
//! the pipeline shift) as a stateful batched `PpoVecEnv`, faithful by
//! construction.
//!
//! ## Mapping to PpoVecEnv
//! - **Observation** = `build_pipeline_state`: `[on_hand+arriving, pipeline...]`
//!   of length `lead_time` (the trainer's running normalizer standardizes it).
//! - **Action** = continuous scalar order; the Gaussian sample is rounded and
//!   clamped to `[0, max_order]` by the env, while the log-prob uses the raw
//!   continuous sample (the standard continuous-policy-over-discretized-env
//!   recipe).
//! - **Per period:** the pipeline head arrives, the policy's order enters the
//!   pipeline tail, demand is realized, and `epoch_cost` charges holding on
//!   leftover / shortage on lost sales (vanilla: no procurement or fixed cost).
//! - **Reward** = `-period_cost` (use `reward_scale=1`; lost_sales costs are O(1)
//!   per period, unlike OWMR's hundreds).
//! - **Gate** (for BC warm-start) = a base-stock order-up-to policy:
//!   `order = clip(S - inventory_position, 0, max_order)`.
//! - **Demand** is pre-sampled per `reset(seed)` from the instance's demand
//!   process for common-random-numbers reproducibility.
//!
//! The canonical instance is `vanilla_l4_p4_poisson5` (Poisson(5), lead time 4,
//! holding 1, shortage 4; published optimal average cost 4.73, capped base-stock
//! 4.80) — so a faithful PPO should reach an average per-period cost near ~4.7-5.0.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::core::ppo::environment::{ActionSpec, PpoVecEnv, StepResult};
use crate::problems::lost_sales::demand::{
    build_demand_process, sample_demand, LostSalesDemandConfig, LostSalesDemandKind,
};
use crate::problems::lost_sales::vanilla::env::{build_pipeline_state, epoch_cost, LostSalesState};

/// lost_sales batched PPO environment (single continuous-scalar order action).
pub struct LostSalesPpoEnv {
    num_envs: usize,
    horizon: usize,
    demand_config: LostSalesDemandConfig,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    demand_mean: f64,
    max_order: usize,
    base_stock_level: usize,
    spec: ActionSpec,
    obs_dim: usize,
    states: Vec<LostSalesState>,
    demands: Vec<Vec<i64>>, // [B][T]
    period: usize,
}

impl LostSalesPpoEnv {
    /// The canonical literature instance: Poisson(5), lead time 4, holding 1,
    /// shortage 4 (`vanilla_l4_p4_poisson5`).
    pub fn vanilla_poisson5_l4(num_envs: usize) -> Self {
        let demand_config = LostSalesDemandConfig {
            kind: LostSalesDemandKind::Poisson,
            demand_rate: 5.0,
            demand_lambda_low: 0.0,
            demand_lambda_high: 0.0,
            demand_p00: 0.0,
            demand_p11: 0.0,
        };
        // base_stock_level S=24: at steady state on-hand ~ S - L*mean ~ 4, balancing
        // holding (h=1) against lost-sales shortage (p=4); near the literature base-stock.
        Self::new(demand_config, 4, 1.0, 4.0, 5.0, 200, 30, 24, num_envs)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        demand_config: LostSalesDemandConfig,
        lead_time: usize,
        holding_cost: f64,
        shortage_cost: f64,
        demand_mean: f64,
        horizon: usize,
        max_order: usize,
        base_stock_level: usize,
        num_envs: usize,
    ) -> Self {
        LostSalesPpoEnv {
            num_envs,
            horizon,
            demand_config,
            lead_time,
            holding_cost,
            shortage_cost,
            demand_mean,
            max_order,
            base_stock_level,
            spec: ActionSpec::Continuous { dim: 1 },
            obs_dim: lead_time,
            states: Vec::new(),
            demands: Vec::new(),
            period: 0,
        }
    }

    fn initial_state(&self) -> LostSalesState {
        // Start near steady state: on-hand ~ 2*mean, pipeline filled with the mean.
        LostSalesState {
            current_inventory: (2.0 * self.demand_mean).round() as i64,
            lead_time_orders: vec![self.demand_mean.round() as usize; self.lead_time],
        }
    }

    fn observe(&self) -> Vec<Vec<f32>> {
        self.states
            .iter()
            .map(|s| build_pipeline_state(s.current_inventory, &s.lead_time_orders))
            .collect()
    }
}

impl PpoVecEnv for LostSalesPpoEnv {
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

    fn reset(&mut self, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut process =
            build_demand_process(self.demand_config, &mut rng).expect("valid demand process");
        self.demands = (0..self.num_envs)
            .map(|_| {
                (0..self.horizon)
                    .map(|_| sample_demand(&mut rng, &mut process))
                    .collect::<Vec<i64>>()
            })
            .collect();
        self.states = vec![self.initial_state(); self.num_envs];
        self.period = 0;
        self.observe()
    }

    fn step(&mut self, actions: &[Vec<i64>]) -> StepResult {
        let mut costs = Vec::with_capacity(self.num_envs);
        for e in 0..self.num_envs {
            let order = (actions[e][0].max(0) as usize).min(self.max_order);
            let state = &mut self.states[e];
            // Pipeline shift: head arrives, the new order enters the tail.
            let arriving = state.lead_time_orders.remove(0);
            state.lead_time_orders.push(order);
            state.current_inventory = state.current_inventory.saturating_add(arriving as i64);
            let demand = self.demands[e][self.period];
            let cost = epoch_cost(
                &mut state.current_inventory,
                demand,
                order,
                self.holding_cost,
                self.shortage_cost,
                0.0, // procurement cost (vanilla)
                0.0, // fixed order cost (vanilla)
            );
            costs.push(cost);
        }
        self.period += 1;
        let next_obs = self.observe();
        StepResult { costs, next_obs }
    }

    fn gate_actions(&self) -> Option<Vec<Vec<i64>>> {
        // Base-stock order-up-to: order = clip(S - inventory_position, 0, max_order).
        Some(
            self.states
                .iter()
                .map(|s| {
                    let inventory_position =
                        s.current_inventory + s.lead_time_orders.iter().sum::<usize>() as i64;
                    let order = (self.base_stock_level as i64 - inventory_position)
                        .max(0)
                        .min(self.max_order as i64);
                    vec![order]
                })
                .collect(),
        )
    }
}

/// Roll the base-stock gate on a held-out seed; return the mean AVERAGE
/// per-period cost (total / horizon). The in-protocol anchor for PPO and a
/// sanity check vs the published optimal (4.73) / capped base-stock (4.80).
pub fn gate_holdout_mean_cost_per_period(num_paths: usize, seed: u64) -> f64 {
    let mut env = LostSalesPpoEnv::vanilla_poisson5_l4(num_paths);
    let _ = env.reset(seed);
    let horizon = env.horizon();
    let mut cost_total = vec![0f64; num_paths];
    for _t in 0..horizon {
        let gate = env.gate_actions().expect("gate");
        let step = env.step(&gate);
        for e in 0..num_paths {
            cost_total[e] += step.costs[e];
        }
    }
    cost_total.iter().sum::<f64>() / num_paths as f64 / horizon as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lost_sales_env_shapes_and_gate_sane() {
        let mut env = LostSalesPpoEnv::vanilla_poisson5_l4(8);
        let obs = env.reset(900_000);
        assert_eq!(env.obs_dim(), 4, "obs = pipeline of length lead_time");
        assert_eq!(obs[0].len(), 4);
        assert!(matches!(env.action_spec(), ActionSpec::Continuous { dim: 1 }));
        let gate = env.gate_actions().expect("gate");
        assert_eq!(gate[0].len(), 1);
        // Base-stock average per-period cost should be in a sane band near the
        // published optimal (4.73) / capped base-stock (4.80).
        let avg = gate_holdout_mean_cost_per_period(512, 900_000);
        println!("lost_sales base-stock gate avg per-period cost: {avg:.3}");
        assert!(
            (4.0..=7.0).contains(&avg),
            "base-stock gate avg/period {avg:.3} implausibly far from ~4.8"
        );
    }

    /// PPO (continuous Gaussian head) should learn a near-optimal ordering policy
    /// on lost_sales — average per-period cost near the published optimum (4.73).
    /// Run in release: cargo test --release --features ppo -- --ignored lost_sales_ppo
    #[test]
    #[ignore]
    fn lost_sales_ppo_reaches_near_optimal() {
        use crate::core::ppo::ppo_trainer::{train_ppo, PpoConfig};
        let gate = gate_holdout_mean_cost_per_period(1024, 900_000);
        let horizon = 200usize;
        let cfg = PpoConfig {
            iters: 80,
            train_paths: 128,
            eval_paths: 512,
            hidden: 64,
            lr: 1e-3,
            clip: 0.2,
            ppo_epochs: 4,
            minibatch: 4096,
            ent_coef: 0.0,
            vf_coef: 0.5,
            max_grad_norm: 0.5,
            reward_scale: 1.0, // lost_sales costs are O(1) per period
            bc_epochs: 40,
            bc_paths: 256,
            bc_lr: 1e-3,
            bc_batch: 4096,
            eval_every: 10,
            seed: 0,
            ..PpoConfig::default()
        };
        let make_env =
            |n: usize, _eval: bool| Box::new(LostSalesPpoEnv::vanilla_poisson5_l4(n)) as Box<dyn PpoVecEnv>;
        let outcome = train_ppo(make_env, &cfg).expect("lost_sales PPO failed");
        let ppo_avg = outcome.best_holdout_cost / horizon as f64;
        println!(
            "[lost_sales PPO] gate avg/period={gate:.3} | PPO best avg/period={ppo_avg:.3} (opt 4.73, capped-BS 4.80)"
        );
        assert!(
            ppo_avg < 5.5,
            "PPO avg/period {ppo_avg:.3} not near-optimal (opt 4.73)"
        );
    }
}
