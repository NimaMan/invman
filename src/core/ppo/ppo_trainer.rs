//! # `ppo_trainer` — the reusable PPO training loop
//!
//! ## Objective
//! Train a `PpoVecEnv` with Proximal Policy Optimization, faithfully porting the
//! validated reference algorithm (`ppo_owmr.py`). The same loop trains ANY
//! problem: it only talks to the `PpoVecEnv` trait and the candle
//! `ActorCritic`, so OWMR, lost_sales, serial, etc. all reuse it.
//!
//! ## Algorithm (per call)
//! 1. **Behavior-clone warm-start (optional).** If the env exposes a gate
//!    heuristic (`gate_actions`), roll the gate through an eval-mode env,
//!    record `(raw_obs, gate_action)` and Monte-Carlo return-to-go, fit the
//!    running normalizer, then BC the actor (cross-entropy over heads) and warm
//!    the value head (MSE to return-to-go). This starts PPO from a strong,
//!    competitive policy — fair to PPO on hard heavy-tail problems.
//! 2. **PPO iterations.** Each iteration:
//!    a. Collect a vectorized rollout: `train_paths` parallel trajectories of
//!       `horizon` steps, sampling actions from the current policy, recording
//!       `(normed_obs, action, logp, value, reward = -cost/reward_scale)`. The
//!       running normalizer is updated online during training rollouts.
//!    b. GAE(lambda) advantages + returns (finite horizon, terminal value 0).
//!    c. Normalize advantages across the whole batch.
//!    d. For `ppo_epochs` over shuffled minibatches: clipped surrogate
//!       (`min(r*A, clip(r)*A)`), clipped value loss, entropy bonus, global-norm
//!       gradient clipping, one Adam step (Adam moments persist across
//!       iterations).
//!    e. Periodically evaluate the GREEDY policy on a held-out CRN seed block
//!       under eval-mode dynamics; keep the best checkpoint.
//! 3. Restore the best checkpoint and return its held-out cost (mean/std over the
//!    parallel eval paths).
//!
//! All hyperparameters live in `PpoConfig` (defaults = the reference file
//! defaults). The OWMR caller overrides them with the validated 5-seed values.

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use super::actor_critic::ActorCritic;
use super::environment::{ActionSpec, PpoVecEnv};
use super::gae::compute_gae;
use super::multi_discrete_head::{joint_logp_entropy, sample_head};
use super::running_norm::RunningNorm;

/// PPO hyperparameters. Defaults match the reference `ppo_owmr.default_args`.
#[derive(Clone, Debug)]
pub struct PpoConfig {
    pub iters: usize,
    pub train_paths: usize,
    pub eval_paths: usize,
    pub hidden: usize,
    pub lr: f64,
    pub gamma: f32,
    pub lam: f32,
    pub clip: f32,
    pub ppo_epochs: usize,
    pub minibatch: usize,
    pub vf_coef: f32,
    pub ent_coef: f32,
    pub max_grad_norm: f32,
    pub reward_scale: f32,
    pub bc_epochs: usize,
    pub bc_paths: usize,
    pub bc_lr: f64,
    pub bc_batch: usize,
    pub eval_every: usize,
    pub seed: u64,
    pub train_seed_start: u64,
    pub holdout_seed_start: u64,
    pub search_seed_start: u64,
    pub verbose: bool,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            iters: 120,
            train_paths: 256,
            eval_paths: 1024,
            hidden: 128,
            lr: 3e-4,
            gamma: 1.0,
            lam: 0.95,
            clip: 0.2,
            ppo_epochs: 4,
            minibatch: 2048,
            vf_coef: 0.5,
            ent_coef: 0.003,
            max_grad_norm: 0.5,
            reward_scale: 1000.0,
            bc_epochs: 30,
            bc_paths: 256,
            bc_lr: 1e-3,
            bc_batch: 2048,
            eval_every: 5,
            seed: 0,
            train_seed_start: 600_000,
            holdout_seed_start: 900_000,
            search_seed_start: 500_000,
            verbose: false,
        }
    }
}

/// One point on the training curve.
#[derive(Clone, Debug)]
pub struct CurvePoint {
    pub iter: usize,
    pub phase: String,
    pub train_cost: f64,
    pub holdout_greedy_cost: f64,
}

/// Result of a PPO training run.
#[derive(Clone, Debug)]
pub struct PpoOutcome {
    pub best_holdout_cost: f64,
    pub final_holdout_cost_mean: f64,
    pub final_holdout_cost_std: f64,
    pub curve: Vec<CurvePoint>,
}

// ----------------------------- gradient clipping -----------------------------

/// Clip gradients in-place to a global L2 norm of `max_norm` (PyTorch
/// `clip_grad_norm_` semantics), then they are ready for the optimizer step.
fn clip_grad_norm(
    grads: &mut candle_core::backprop::GradStore,
    vars: &[Var],
    max_norm: f32,
) -> Result<()> {
    let mut sumsq = 0f64;
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            sumsq += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
    }
    let norm = sumsq.sqrt();
    if norm > max_norm as f64 && norm > 0.0 {
        let scale = max_norm as f64 / (norm + 1e-6);
        for v in vars {
            // Read out the scaled grad first (ending the immutable borrow) before
            // re-inserting it (mutable borrow).
            let scaled = match grads.get(v.as_tensor()) {
                Some(g) => Some(g.affine(scale, 0.0)?),
                None => None,
            };
            if let Some(s) = scaled {
                grads.insert(v.as_tensor(), s);
            }
        }
    }
    Ok(())
}

// --------------------------- best-checkpoint snapshot ------------------------

type VarSnapshot = Vec<(Vec<f32>, Vec<usize>)>;

fn snapshot_vars(vars: &[Var]) -> Result<VarSnapshot> {
    vars.iter()
        .map(|v| {
            let t = v.as_tensor();
            let shape = t.dims().to_vec();
            let data = t.flatten_all()?.to_vec1::<f32>()?;
            Ok((data, shape))
        })
        .collect()
}

fn restore_vars(vars: &[Var], snap: &VarSnapshot, device: &Device) -> Result<()> {
    for (v, (data, shape)) in vars.iter().zip(snap.iter()) {
        let t = Tensor::from_vec(data.clone(), shape.as_slice(), device)?;
        v.set(&t)?;
    }
    Ok(())
}

// ------------------------------- rollout buffer ------------------------------

struct RolloutData {
    /// Normalized observations, `T*B` rows (index `t*B + e`).
    obs: Vec<Vec<f32>>,
    /// Sampled actions, `T*B` rows of `n_heads` u32 category indices.
    actions: Vec<Vec<u32>>,
    /// Joint log-prob of each sampled action, `T*B`.
    old_logp: Vec<f32>,
    /// Value estimates, time-major `[T][B]`.
    values_tb: Vec<Vec<f32>>,
    /// Rewards (= -cost/scale), time-major `[T][B]`.
    rewards_tb: Vec<Vec<f32>>,
    /// Total per-env episode cost.
    cost_total: Vec<f64>,
    num_envs: usize,
}

/// Collect one vectorized training rollout (samples actions, updates the
/// normalizer online).
#[allow(clippy::too_many_arguments)]
fn collect_rollout(
    ac: &ActorCritic,
    norm: &mut RunningNorm,
    env: &mut dyn PpoVecEnv,
    seed: u64,
    horizon: usize,
    n_heads: usize,
    reward_scale: f32,
    rng: &mut StdRng,
) -> Result<RolloutData> {
    let b = env.num_envs();
    let mut raw = env.reset(seed);
    let mut obs = Vec::with_capacity(horizon * b);
    let mut actions = Vec::with_capacity(horizon * b);
    let mut old_logp = Vec::with_capacity(horizon * b);
    let mut values_tb = Vec::with_capacity(horizon);
    let mut rewards_tb = Vec::with_capacity(horizon);
    let mut cost_total = vec![0f64; b];

    for _t in 0..horizon {
        norm.update(&raw);
        let normed = norm.normalize_batch(&raw);
        let obs_t = ac.obs_to_tensor(&normed)?;
        let (logits, value) = ac.forward(&obs_t)?;
        let value_cpu = value.to_vec1::<f32>()?;
        let mut logits_cpu = Vec::with_capacity(n_heads);
        for lg in &logits {
            logits_cpu.push(lg.to_vec2::<f32>()?); // [head] -> (B, size_j)
        }
        let mut actions_i64 = Vec::with_capacity(b);
        for e in 0..b {
            let mut act_i64 = Vec::with_capacity(n_heads);
            let mut act_u32 = Vec::with_capacity(n_heads);
            let mut joint_logp = 0f32;
            for j in 0..n_heads {
                let u: f32 = rng.gen::<f32>();
                let (a, lp) = sample_head(&logits_cpu[j][e], false, u);
                act_i64.push(a as i64);
                act_u32.push(a as u32);
                joint_logp += lp;
            }
            obs.push(normed[e].clone());
            actions.push(act_u32);
            old_logp.push(joint_logp);
            actions_i64.push(act_i64);
        }
        let step = env.step(&actions_i64);
        let mut reward_row = Vec::with_capacity(b);
        for e in 0..b {
            cost_total[e] += step.costs[e];
            reward_row.push(-(step.costs[e] as f32) / reward_scale);
        }
        rewards_tb.push(reward_row);
        values_tb.push(value_cpu);
        raw = step.next_obs;
    }

    Ok(RolloutData {
        obs,
        actions,
        old_logp,
        values_tb,
        rewards_tb,
        cost_total,
        num_envs: b,
    })
}

/// Evaluate the GREEDY (argmax) policy on a held-out seed under eval-mode
/// dynamics, without updating the normalizer. Returns `(mean_cost, std_cost)`
/// over the parallel eval paths.
fn evaluate_greedy(
    ac: &ActorCritic,
    norm: &RunningNorm,
    env: &mut dyn PpoVecEnv,
    seed: u64,
    horizon: usize,
    n_heads: usize,
) -> Result<(f64, f64)> {
    let b = env.num_envs();
    let mut raw = env.reset(seed);
    let mut cost_total = vec![0f64; b];
    for _t in 0..horizon {
        let normed = norm.normalize_batch(&raw);
        let obs_t = ac.obs_to_tensor(&normed)?;
        let (logits, _value) = ac.forward(&obs_t)?;
        let mut logits_cpu = Vec::with_capacity(n_heads);
        for lg in &logits {
            logits_cpu.push(lg.to_vec2::<f32>()?);
        }
        let mut actions_i64 = Vec::with_capacity(b);
        for e in 0..b {
            let mut act_i64 = Vec::with_capacity(n_heads);
            for j in 0..n_heads {
                let (a, _lp) = sample_head(&logits_cpu[j][e], true, 0.0);
                act_i64.push(a as i64);
            }
            actions_i64.push(act_i64);
        }
        let step = env.step(&actions_i64);
        for e in 0..b {
            cost_total[e] += step.costs[e];
        }
        raw = step.next_obs;
    }
    let mean = cost_total.iter().sum::<f64>() / b as f64;
    let var = cost_total.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / b as f64;
    Ok((mean, var.sqrt()))
}

// ------------------------------- behavior clone ------------------------------

#[allow(clippy::too_many_arguments)]
fn behavior_clone(
    ac: &ActorCritic,
    vars: &[Var],
    norm: &mut RunningNorm,
    env: &mut dyn PpoVecEnv,
    seed: u64,
    horizon: usize,
    head_sizes: &[usize],
    cfg: &PpoConfig,
    rng: &mut StdRng,
    device: &Device,
) -> Result<()> {
    let b = env.num_envs();
    let n_heads = head_sizes.len();
    let mut raw = env.reset(seed);
    let mut obs_raw = Vec::with_capacity(horizon * b);
    let mut targets = Vec::with_capacity(horizon * b);
    let mut costs_tb = Vec::with_capacity(horizon);

    for _t in 0..horizon {
        let gate = env
            .gate_actions()
            .expect("behavior_clone requires gate_actions");
        for e in 0..b {
            obs_raw.push(raw[e].clone());
            let mut tgt = Vec::with_capacity(n_heads);
            for j in 0..n_heads {
                let a = gate[e][j].max(0).min(head_sizes[j] as i64 - 1) as u32;
                tgt.push(a);
            }
            targets.push(tgt);
        }
        let step = env.step(&gate);
        costs_tb.push(step.costs.clone());
        raw = step.next_obs;
    }

    // Monte-Carlo return-to-go (gamma=1, undiscounted protocol).
    let n = horizon * b;
    let mut rtg = vec![0f32; n];
    let mut acc = vec![0f32; b];
    for t in (0..horizon).rev() {
        for e in 0..b {
            acc[e] = -(costs_tb[t][e] as f32) / cfg.reward_scale + acc[e];
            rtg[t * b + e] = acc[e];
        }
    }

    // Fit the normalizer to all collected states, then BC.
    norm.update(&obs_raw);
    let obs_normed = norm.normalize_batch(&obs_raw);
    let mut bc_opt = AdamW::new(
        vars.to_vec(),
        ParamsAdamW {
            lr: cfg.bc_lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        },
    )?;
    for ep in 0..cfg.bc_epochs {
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(rng);
        for chunk in perm.chunks(cfg.bc_batch) {
            let mb = chunk.len();
            let mut obs_mb = Vec::with_capacity(mb);
            let mut tgt_u32 = Vec::with_capacity(mb * n_heads);
            let mut rtg_mb = Vec::with_capacity(mb);
            for &i in chunk {
                obs_mb.push(obs_normed[i].clone());
                tgt_u32.extend_from_slice(&targets[i]);
                rtg_mb.push(rtg[i]);
            }
            let obs_t = ac.obs_to_tensor(&obs_mb)?;
            let tgt_t = Tensor::from_vec(tgt_u32, (mb, n_heads), device)?;
            let rtg_t = Tensor::from_vec(rtg_mb, mb, device)?;
            let (logits, value) = ac.forward(&obs_t)?;
            let (logp, _ent) = joint_logp_entropy(&logits, &tgt_t)?;
            // Cross-entropy summed over heads = -mean(joint logp).
            let ce_loss = logp.mean_all()?.neg()?;
            let v_loss = ((&value - &rtg_t)?).sqr()?.mean_all()?.affine(0.5, 0.0)?;
            let loss = (&ce_loss + &v_loss)?;
            bc_opt.backward_step(&loss)?;
        }
        if cfg.verbose && (ep % cfg.bc_epochs.max(1).div_ceil(5) == 0 || ep == cfg.bc_epochs - 1) {
            println!("  [BC] epoch {ep}/{}", cfg.bc_epochs);
        }
    }
    Ok(())
}

// --------------------------------- PPO update --------------------------------

#[allow(clippy::too_many_arguments)]
fn ppo_update(
    ac: &ActorCritic,
    vars: &[Var],
    opt: &mut AdamW,
    data: &RolloutData,
    cfg: &PpoConfig,
    horizon: usize,
    n_heads: usize,
    rng: &mut StdRng,
    device: &Device,
) -> Result<()> {
    let b = data.num_envs;
    let (adv_tb, ret_tb) = compute_gae(&data.rewards_tb, &data.values_tb, cfg.gamma, cfg.lam);

    let n = horizon * b;
    let mut adv = Vec::with_capacity(n);
    let mut ret = Vec::with_capacity(n);
    let mut val_old = Vec::with_capacity(n);
    for t in 0..horizon {
        for e in 0..b {
            adv.push(adv_tb[t][e]);
            ret.push(ret_tb[t][e]);
            val_old.push(data.values_tb[t][e]);
        }
    }
    // Normalize advantages across the whole batch.
    let mean = adv.iter().sum::<f32>() / n as f32;
    let var = adv.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    for a in adv.iter_mut() {
        *a = (*a - mean) / (std + 1e-8);
    }

    for _epoch in 0..cfg.ppo_epochs {
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(rng);
        for chunk in perm.chunks(cfg.minibatch) {
            let mb = chunk.len();
            let mut obs_mb = Vec::with_capacity(mb);
            let mut act_u32 = Vec::with_capacity(mb * n_heads);
            let mut old_logp_mb = Vec::with_capacity(mb);
            let mut adv_mb = Vec::with_capacity(mb);
            let mut ret_mb = Vec::with_capacity(mb);
            let mut val_old_mb = Vec::with_capacity(mb);
            for &i in chunk {
                obs_mb.push(data.obs[i].clone());
                act_u32.extend_from_slice(&data.actions[i]);
                old_logp_mb.push(data.old_logp[i]);
                adv_mb.push(adv[i]);
                ret_mb.push(ret[i]);
                val_old_mb.push(val_old[i]);
            }
            let obs_t = ac.obs_to_tensor(&obs_mb)?;
            let act_t = Tensor::from_vec(act_u32, (mb, n_heads), device)?;
            let old_logp_t = Tensor::from_vec(old_logp_mb, mb, device)?;
            let adv_t = Tensor::from_vec(adv_mb, mb, device)?;
            let ret_t = Tensor::from_vec(ret_mb, mb, device)?;
            let val_old_t = Tensor::from_vec(val_old_mb, mb, device)?;

            let (logits, value) = ac.forward(&obs_t)?;
            let (logp, ent) = joint_logp_entropy(&logits, &act_t)?;

            // Clipped surrogate.
            let ratio = ((&logp - &old_logp_t)?).exp()?;
            let surr1 = (&ratio * &adv_t)?;
            let clipped = ratio.clamp(1.0 - cfg.clip as f64, 1.0 + cfg.clip as f64)?;
            let surr2 = (&clipped * &adv_t)?;
            let pg_loss = surr1.minimum(&surr2)?.mean_all()?.neg()?;

            // Clipped value loss.
            let diff = (&value - &val_old_t)?;
            let diff_clamped = diff.clamp(-(cfg.clip as f64), cfg.clip as f64)?;
            let v_clip = (&val_old_t + &diff_clamped)?;
            let vf1 = ((&value - &ret_t)?).sqr()?;
            let vf2 = ((&v_clip - &ret_t)?).sqr()?;
            let value_term = vf1
                .maximum(&vf2)?
                .mean_all()?
                .affine(0.5 * cfg.vf_coef as f64, 0.0)?;

            // Entropy bonus.
            let entropy_term = ent.mean_all()?.affine(cfg.ent_coef as f64, 0.0)?;

            let loss = ((&pg_loss + &value_term)? - &entropy_term)?;
            let mut grads = loss.backward()?;
            clip_grad_norm(&mut grads, vars, cfg.max_grad_norm)?;
            opt.step(&grads)?;
        }
    }
    Ok(())
}

// --------------------------------- top level ---------------------------------

/// Train a `PpoVecEnv` with PPO. `make_env(num_envs, eval_mode)` constructs a
/// fresh batched environment with the requested parallelism and dynamics mode.
pub fn train_ppo<F>(make_env: F, cfg: &PpoConfig) -> Result<PpoOutcome>
where
    F: Fn(usize, bool) -> Box<dyn PpoVecEnv>,
{
    let device = Device::Cpu;
    // Probe the env for its shapes/spec.
    let probe = make_env(1, false);
    let obs_dim = probe.obs_dim();
    let horizon = probe.horizon();
    let head_sizes = match probe.action_spec() {
        ActionSpec::MultiDiscrete { sizes } => sizes.clone(),
        ActionSpec::Continuous { .. } => {
            return Err(candle_core::Error::Msg(
                "ppo_trainer: Continuous action spec not yet supported (multi-discrete only)"
                    .to_string(),
            ));
        }
    };
    let n_heads = head_sizes.len();
    drop(probe);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let ac = ActorCritic::new(vb, obs_dim, &head_sizes, cfg.hidden, device.clone())?;
    let vars = varmap.all_vars();
    let mut norm = RunningNorm::new(obs_dim);
    let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_mul(2_654_435_761).wrapping_add(1));

    let mut curve = Vec::new();

    // ----- BC warm-start (only if the env exposes a gate) -----
    if cfg.bc_epochs > 0 {
        let mut bc_env = make_env(cfg.bc_paths, true);
        let has_gate = {
            let _ = bc_env.reset(cfg.search_seed_start + 50_000);
            bc_env.gate_actions().is_some()
        };
        if has_gate {
            behavior_clone(
                &ac,
                &vars,
                &mut norm,
                &mut *bc_env,
                cfg.search_seed_start + 50_000,
                horizon,
                &head_sizes,
                cfg,
                &mut rng,
                &device,
            )?;
        }
    }

    // ----- eval env (eval-mode dynamics) -----
    let mut eval_env = make_env(cfg.eval_paths, true);
    let (bc_eval_mean, _) =
        evaluate_greedy(&ac, &norm, &mut *eval_env, cfg.holdout_seed_start, horizon, n_heads)?;
    if cfg.verbose {
        println!("[after BC] greedy holdout cost {bc_eval_mean:.2}");
    }
    curve.push(CurvePoint {
        iter: 0,
        phase: "bc".to_string(),
        train_cost: f64::NAN,
        holdout_greedy_cost: bc_eval_mean,
    });

    // ----- PPO loop -----
    let mut opt = AdamW::new(
        vars.to_vec(),
        ParamsAdamW {
            lr: cfg.lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        },
    )?;
    let mut train_env = make_env(cfg.train_paths, false);

    let mut best_eval = bc_eval_mean;
    let mut best_snapshot = snapshot_vars(&vars)?;

    for it in 1..=cfg.iters {
        let seed_it = cfg.train_seed_start + (it as u64) * 1000 + cfg.seed;
        let data = collect_rollout(
            &ac,
            &mut norm,
            &mut *train_env,
            seed_it,
            horizon,
            n_heads,
            cfg.reward_scale,
            &mut rng,
        )?;
        let train_cost = data.cost_total.iter().sum::<f64>() / data.num_envs as f64;
        ppo_update(&ac, &vars, &mut opt, &data, cfg, horizon, n_heads, &mut rng, &device)?;

        if it % cfg.eval_every == 0 || it == cfg.iters {
            let (ev_mean, _ev_std) = evaluate_greedy(
                &ac,
                &norm,
                &mut *eval_env,
                cfg.holdout_seed_start,
                horizon,
                n_heads,
            )?;
            curve.push(CurvePoint {
                iter: it,
                phase: "ppo".to_string(),
                train_cost,
                holdout_greedy_cost: ev_mean,
            });
            if ev_mean < best_eval {
                best_eval = ev_mean;
                best_snapshot = snapshot_vars(&vars)?;
            }
            if cfg.verbose {
                println!(
                    "[ppo it {it}] train {train_cost:.1} holdout-greedy {ev_mean:.1} (best {best_eval:.1})"
                );
            }
        }
    }

    // Restore best checkpoint and score it.
    restore_vars(&vars, &best_snapshot, &device)?;
    let (final_mean, final_std) =
        evaluate_greedy(&ac, &norm, &mut *eval_env, cfg.holdout_seed_start, horizon, n_heads)?;

    Ok(PpoOutcome {
        best_holdout_cost: best_eval,
        final_holdout_cost_mean: final_mean,
        final_holdout_cost_std: final_std,
        curve,
    })
}

// ------------------------------------ tests ----------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ppo::environment::StepResult;

    /// Toy multi-discrete env: each step the policy observes a one-hot context
    /// `c in {0..n-1}` and must output the matching category; cost = |action - c|.
    /// Optimal greedy cost = 0. Validates the entire PPO loop (categorical head,
    /// GAE, clipped update) in isolation from any inventory env. No gate -> no BC.
    struct ToyContextBandit {
        num_envs: usize,
        n: usize,
        horizon: usize,
        spec: ActionSpec,
        rng: StdRng,
        contexts: Vec<usize>,
    }

    impl ToyContextBandit {
        fn new(num_envs: usize, n: usize, horizon: usize) -> Self {
            Self {
                num_envs,
                n,
                horizon,
                spec: ActionSpec::MultiDiscrete { sizes: vec![n] },
                rng: StdRng::seed_from_u64(0),
                contexts: vec![0; num_envs],
            }
        }
        fn one_hot(&self, c: usize) -> Vec<f32> {
            let mut v = vec![0f32; self.n];
            v[c] = 1.0;
            v
        }
        fn obs(&self) -> Vec<Vec<f32>> {
            self.contexts.iter().map(|&c| self.one_hot(c)).collect()
        }
    }

    impl PpoVecEnv for ToyContextBandit {
        fn num_envs(&self) -> usize {
            self.num_envs
        }
        fn obs_dim(&self) -> usize {
            self.n
        }
        fn horizon(&self) -> usize {
            self.horizon
        }
        fn action_spec(&self) -> &ActionSpec {
            &self.spec
        }
        fn reset(&mut self, seed: u64) -> Vec<Vec<f32>> {
            self.rng = StdRng::seed_from_u64(seed);
            self.contexts = (0..self.num_envs).map(|_| self.rng.gen_range(0..self.n)).collect();
            self.obs()
        }
        fn step(&mut self, actions: &[Vec<i64>]) -> StepResult {
            let mut costs = Vec::with_capacity(self.num_envs);
            for e in 0..self.num_envs {
                let a = actions[e][0].clamp(0, self.n as i64 - 1);
                costs.push((a - self.contexts[e] as i64).unsigned_abs() as f64);
            }
            // Redraw contexts for the next step.
            self.contexts = (0..self.num_envs).map(|_| self.rng.gen_range(0..self.n)).collect();
            let next_obs = self.obs();
            StepResult { costs, next_obs }
        }
    }

    #[test]
    fn ppo_solves_toy_context_bandit() {
        let cfg = PpoConfig {
            iters: 40,
            train_paths: 64,
            eval_paths: 256,
            hidden: 32,
            lr: 3e-3,
            gamma: 1.0,
            lam: 0.95,
            clip: 0.2,
            ppo_epochs: 4,
            minibatch: 128,
            vf_coef: 0.5,
            ent_coef: 0.01,
            max_grad_norm: 0.5,
            reward_scale: 1.0,
            bc_epochs: 0, // no gate -> no BC
            bc_paths: 64,
            bc_lr: 1e-3,
            bc_batch: 128,
            eval_every: 10,
            seed: 1,
            train_seed_start: 600_000,
            holdout_seed_start: 900_000,
            search_seed_start: 500_000,
            verbose: false,
        };
        let n = 5;
        let horizon = 8;
        let make_env =
            move |num_envs: usize, _eval: bool| -> Box<dyn PpoVecEnv> {
                Box::new(ToyContextBandit::new(num_envs, n, horizon))
            };
        let outcome = train_ppo(make_env, &cfg).expect("ppo training failed");
        // Per-episode cost of a RANDOM policy ~ horizon * E|a-c| ~ 8 * 1.6 = ~12.8.
        // PPO should solve the identity mapping -> near-zero greedy cost.
        let per_step = outcome.final_holdout_cost_mean / horizon as f64;
        assert!(
            per_step < 0.5,
            "PPO failed to solve toy bandit: per-step greedy cost {per_step:.3} (episode {:.2})",
            outcome.final_holdout_cost_mean
        );
    }
}
