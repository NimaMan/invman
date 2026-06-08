//! # `ppo_trainer` — the reusable PPO training loop
//!
//! ## Objective
//! Train a `PpoVecEnv` with Proximal Policy Optimization, faithfully porting the
//! validated reference algorithm (`ppo_owmr.py`). The same loop trains ANY
//! problem and ANY action geometry: it talks only to the `PpoVecEnv` trait and
//! the candle `ActorCritic`, and dispatches action sampling + the differentiable
//! log-prob/entropy on the policy output type (factored multi-discrete OR
//! diagonal Gaussian). OWMR (multi-discrete), lost_sales (continuous scalar),
//! serial (continuous vector), etc. all reuse it.
//!
//! ## Algorithm (per call)
//! 1. **Behavior-clone warm-start (optional).** If the env exposes a gate
//!    heuristic (`gate_actions`), roll the gate through an eval-mode env, record
//!    `(raw_obs, gate_action)` and Monte-Carlo return-to-go, fit the running
//!    normalizer, then BC the actor (maximize the gate action's log-prob =
//!    cross-entropy for discrete / Gaussian NLL for continuous) and warm the
//!    value head (MSE to return-to-go).
//! 2. **PPO iterations.** Each iteration: collect a vectorized rollout
//!    (`train_paths` x `horizon`), GAE(lambda) advantages + returns (finite
//!    horizon), normalize advantages, then `ppo_epochs` of minibatched updates:
//!    clipped surrogate + clipped value loss + entropy bonus + global-norm
//!    gradient clipping + one Adam step (moments persist across iterations).
//!    Periodically evaluate the GREEDY policy on a held-out CRN seed; keep the
//!    best checkpoint.
//! 3. Restore the best checkpoint and return its held-out cost (mean/std).
//!
//! ## Action handling
//! Actions are stored uniformly as `f32` per dimension: for multi-discrete they
//! are the sampled category indices (cast to `u32` for the gather); for
//! continuous they are the raw Gaussian samples (the env receives
//! `round(clip(·))` and the log-prob uses the raw sample — the standard
//! continuous-policy-over-discretized-env recipe).

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use super::actor_critic::{ActorCritic, PolicyOutput};
use super::environment::{ActionSpec, PpoVecEnv};
use super::gae::compute_gae;
use super::gaussian_head::{gaussian_logp_entropy, sample_gaussian};
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

// ----------------------------- action dispatch -------------------------------

/// CPU sampling from a policy output. Returns, per env, the integer ENV action,
/// the stored `f32` action (categories or raw Gaussian samples), and the joint
/// log-prob.
fn sample_actions(
    output: &PolicyOutput,
    b: usize,
    action_dim: usize,
    greedy: bool,
    rng: &mut StdRng,
) -> Result<(Vec<Vec<i64>>, Vec<Vec<f32>>, Vec<f32>)> {
    let mut env_actions = Vec::with_capacity(b);
    let mut stored = Vec::with_capacity(b);
    let mut logps = Vec::with_capacity(b);
    match output {
        PolicyOutput::MultiDiscrete(logits) => {
            let mut logits_cpu = Vec::with_capacity(logits.len());
            for lg in logits {
                logits_cpu.push(lg.to_vec2::<f32>()?); // (B, size_j)
            }
            for e in 0..b {
                let mut env_a = Vec::with_capacity(action_dim);
                let mut store_a = Vec::with_capacity(action_dim);
                let mut logp = 0f32;
                for j in 0..action_dim {
                    let u: f32 = rng.gen::<f32>();
                    let (a, lp) = sample_head(&logits_cpu[j][e], greedy, u);
                    env_a.push(a as i64);
                    store_a.push(a as f32);
                    logp += lp;
                }
                env_actions.push(env_a);
                stored.push(store_a);
                logps.push(logp);
            }
        }
        PolicyOutput::Continuous { mean, log_std } => {
            let mean_cpu = mean.to_vec2::<f32>()?; // (B, dim)
            let log_std_cpu = log_std.to_vec1::<f32>()?; // (dim,)
            for e in 0..b {
                let (a, logp) = sample_gaussian(&mean_cpu[e], &log_std_cpu, greedy, rng);
                // The env receives the rounded action and clamps to its feasible
                // range; the stored continuous sample carries the log-prob.
                let env_a: Vec<i64> = a.iter().map(|&x| x.round() as i64).collect();
                env_actions.push(env_a);
                stored.push(a);
                logps.push(logp);
            }
        }
    }
    Ok((env_actions, stored, logps))
}

/// Differentiable joint log-prob and entropy of stored actions under a policy
/// output. `actions_f32` is `(mb, action_dim)`.
fn logp_entropy_for(output: &PolicyOutput, actions_f32: &Tensor) -> Result<(Tensor, Tensor)> {
    match output {
        PolicyOutput::MultiDiscrete(logits) => {
            let actions_u32 = actions_f32.to_dtype(DType::U32)?;
            joint_logp_entropy(logits, &actions_u32)
        }
        PolicyOutput::Continuous { mean, log_std } => {
            gaussian_logp_entropy(mean, log_std, actions_f32)
        }
    }
}

// ----------------------------- gradient clipping -----------------------------

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
    obs: Vec<Vec<f32>>,         // normalized obs, T*B rows (index t*B+e)
    actions: Vec<Vec<f32>>,     // stored actions, T*B rows of action_dim f32
    old_logp: Vec<f32>,         // joint log-prob, T*B
    values_tb: Vec<Vec<f32>>,   // value estimates [T][B]
    rewards_tb: Vec<Vec<f32>>,  // rewards (= -cost/scale) [T][B]
    cost_total: Vec<f64>,       // per-env episode cost
    num_envs: usize,
}

#[allow(clippy::too_many_arguments)]
fn collect_rollout(
    ac: &ActorCritic,
    norm: &mut RunningNorm,
    env: &mut dyn PpoVecEnv,
    seed: u64,
    horizon: usize,
    action_dim: usize,
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
        let (output, value) = ac.forward(&obs_t)?;
        let value_cpu = value.to_vec1::<f32>()?;
        let (env_actions, stored, logps) = sample_actions(&output, b, action_dim, false, rng)?;
        for e in 0..b {
            obs.push(normed[e].clone());
            actions.push(stored[e].clone());
            old_logp.push(logps[e]);
        }
        let step = env.step(&env_actions);
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

fn evaluate_greedy(
    ac: &ActorCritic,
    norm: &RunningNorm,
    env: &mut dyn PpoVecEnv,
    seed: u64,
    horizon: usize,
    action_dim: usize,
    rng: &mut StdRng,
) -> Result<(f64, f64)> {
    let b = env.num_envs();
    let mut raw = env.reset(seed);
    let mut cost_total = vec![0f64; b];
    for _t in 0..horizon {
        let normed = norm.normalize_batch(&raw);
        let obs_t = ac.obs_to_tensor(&normed)?;
        let (output, _value) = ac.forward(&obs_t)?;
        let (env_actions, _stored, _logps) = sample_actions(&output, b, action_dim, true, rng)?;
        let step = env.step(&env_actions);
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
    spec: &ActionSpec,
    cfg: &PpoConfig,
    rng: &mut StdRng,
    device: &Device,
) -> Result<()> {
    let b = env.num_envs();
    let action_dim = spec.action_dim();
    // Per-dim category caps for clipping discrete targets (None for continuous).
    let sizes: Option<Vec<usize>> = match spec {
        ActionSpec::MultiDiscrete { sizes } => Some(sizes.clone()),
        ActionSpec::Continuous { .. } => None,
    };
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
            let mut tgt = Vec::with_capacity(action_dim);
            for j in 0..action_dim {
                let value = match &sizes {
                    // Clip discrete targets into the head's category range.
                    Some(sizes) => gate[e][j].max(0).min(sizes[j] as i64 - 1) as f32,
                    // Continuous: the raw gate order is the target mean.
                    None => gate[e][j] as f32,
                };
                tgt.push(value);
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
            let mut tgt_flat = Vec::with_capacity(mb * action_dim);
            let mut rtg_mb = Vec::with_capacity(mb);
            for &i in chunk {
                obs_mb.push(obs_normed[i].clone());
                tgt_flat.extend_from_slice(&targets[i]);
                rtg_mb.push(rtg[i]);
            }
            let obs_t = ac.obs_to_tensor(&obs_mb)?;
            let tgt_t = Tensor::from_vec(tgt_flat, (mb, action_dim), device)?;
            let rtg_t = Tensor::from_vec(rtg_mb, mb, device)?;
            let (output, value) = ac.forward(&obs_t)?;
            let (logp, _ent) = logp_entropy_for(&output, &tgt_t)?;
            // Maximize the gate action's log-prob = minimize -mean(logp).
            let bc_loss = logp.mean_all()?.neg()?;
            let v_loss = ((&value - &rtg_t)?).sqr()?.mean_all()?.affine(0.5, 0.0)?;
            let loss = (&bc_loss + &v_loss)?;
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
    action_dim: usize,
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
            let mut act_flat = Vec::with_capacity(mb * action_dim);
            let mut old_logp_mb = Vec::with_capacity(mb);
            let mut adv_mb = Vec::with_capacity(mb);
            let mut ret_mb = Vec::with_capacity(mb);
            let mut val_old_mb = Vec::with_capacity(mb);
            for &i in chunk {
                obs_mb.push(data.obs[i].clone());
                act_flat.extend_from_slice(&data.actions[i]);
                old_logp_mb.push(data.old_logp[i]);
                adv_mb.push(adv[i]);
                ret_mb.push(ret[i]);
                val_old_mb.push(val_old[i]);
            }
            let obs_t = ac.obs_to_tensor(&obs_mb)?;
            let act_t = Tensor::from_vec(act_flat, (mb, action_dim), device)?;
            let old_logp_t = Tensor::from_vec(old_logp_mb, mb, device)?;
            let adv_t = Tensor::from_vec(adv_mb, mb, device)?;
            let ret_t = Tensor::from_vec(ret_mb, mb, device)?;
            let val_old_t = Tensor::from_vec(val_old_mb, mb, device)?;

            let (output, value) = ac.forward(&obs_t)?;
            let (logp, ent) = logp_entropy_for(&output, &act_t)?;

            let ratio = ((&logp - &old_logp_t)?).exp()?;
            let surr1 = (&ratio * &adv_t)?;
            let clipped = ratio.clamp(1.0 - cfg.clip as f64, 1.0 + cfg.clip as f64)?;
            let surr2 = (&clipped * &adv_t)?;
            let pg_loss = surr1.minimum(&surr2)?.mean_all()?.neg()?;

            let diff = (&value - &val_old_t)?;
            let diff_clamped = diff.clamp(-(cfg.clip as f64), cfg.clip as f64)?;
            let v_clip = (&val_old_t + &diff_clamped)?;
            let vf1 = ((&value - &ret_t)?).sqr()?;
            let vf2 = ((&v_clip - &ret_t)?).sqr()?;
            let value_term = vf1
                .maximum(&vf2)?
                .mean_all()?
                .affine(0.5 * cfg.vf_coef as f64, 0.0)?;

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
    let probe = make_env(1, false);
    let obs_dim = probe.obs_dim();
    let horizon = probe.horizon();
    let spec = probe.action_spec().clone();
    let action_dim = spec.action_dim();
    drop(probe);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let ac = ActorCritic::new(vb, obs_dim, &spec, cfg.hidden, device.clone())?;
    let vars = varmap.all_vars();
    let mut norm = RunningNorm::new(obs_dim);
    let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_mul(2_654_435_761).wrapping_add(1));

    let mut curve = Vec::new();

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
                &spec,
                cfg,
                &mut rng,
                &device,
            )?;
        }
    }

    let mut eval_env = make_env(cfg.eval_paths, true);
    let (bc_eval_mean, _) = evaluate_greedy(
        &ac,
        &norm,
        &mut *eval_env,
        cfg.holdout_seed_start,
        horizon,
        action_dim,
        &mut rng,
    )?;
    if cfg.verbose {
        println!("[after BC] greedy holdout cost {bc_eval_mean:.2}");
    }
    curve.push(CurvePoint {
        iter: 0,
        phase: "bc".to_string(),
        train_cost: f64::NAN,
        holdout_greedy_cost: bc_eval_mean,
    });

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
            action_dim,
            cfg.reward_scale,
            &mut rng,
        )?;
        let train_cost = data.cost_total.iter().sum::<f64>() / data.num_envs as f64;
        ppo_update(&ac, &vars, &mut opt, &data, cfg, horizon, action_dim, &mut rng, &device)?;

        if it % cfg.eval_every == 0 || it == cfg.iters {
            let (ev_mean, _ev_std) = evaluate_greedy(
                &ac,
                &norm,
                &mut *eval_env,
                cfg.holdout_seed_start,
                horizon,
                action_dim,
                &mut rng,
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

    restore_vars(&vars, &best_snapshot, &device)?;
    let (final_mean, final_std) = evaluate_greedy(
        &ac,
        &norm,
        &mut *eval_env,
        cfg.holdout_seed_start,
        horizon,
        action_dim,
        &mut rng,
    )?;

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

    /// Toy multi-discrete env: observe a one-hot context `c`, output the matching
    /// category; cost = |action - c|. Optimal greedy cost = 0. Validates the full
    /// multi-discrete PPO loop. No gate -> no BC.
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
        fn obs(&self) -> Vec<Vec<f32>> {
            self.contexts
                .iter()
                .map(|&c| {
                    let mut v = vec![0f32; self.n];
                    v[c] = 1.0;
                    v
                })
                .collect()
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
            ppo_epochs: 4,
            minibatch: 128,
            ent_coef: 0.01,
            reward_scale: 1.0,
            bc_epochs: 0,
            eval_every: 10,
            seed: 1,
            ..PpoConfig::default()
        };
        let n = 5;
        let horizon = 8;
        let make_env = move |num_envs: usize, _eval: bool| -> Box<dyn PpoVecEnv> {
            Box::new(ToyContextBandit::new(num_envs, n, horizon))
        };
        let outcome = train_ppo(make_env, &cfg).expect("ppo training failed");
        let per_step = outcome.final_holdout_cost_mean / horizon as f64;
        assert!(
            per_step < 0.5,
            "PPO failed to solve toy bandit: per-step greedy cost {per_step:.3}"
        );
    }

    /// Toy CONTINUOUS env: observe a scalar target `c in [0, 9]`, output a real
    /// action; cost = |round(action) - c|. Validates the diagonal-Gaussian head +
    /// continuous PPO path end-to-end (the lost_sales-style scalar-order shape).
    struct ToyContinuousTarget {
        num_envs: usize,
        horizon: usize,
        spec: ActionSpec,
        rng: StdRng,
        targets: Vec<f32>,
    }

    impl ToyContinuousTarget {
        fn new(num_envs: usize, horizon: usize) -> Self {
            Self {
                num_envs,
                horizon,
                spec: ActionSpec::Continuous { dim: 1 },
                rng: StdRng::seed_from_u64(0),
                targets: vec![0.0; num_envs],
            }
        }
        fn obs(&self) -> Vec<Vec<f32>> {
            self.targets.iter().map(|&t| vec![t]).collect()
        }
        fn draw(&mut self) -> Vec<f32> {
            (0..self.num_envs).map(|_| self.rng.gen_range(0..10) as f32).collect()
        }
    }

    impl PpoVecEnv for ToyContinuousTarget {
        fn num_envs(&self) -> usize {
            self.num_envs
        }
        fn obs_dim(&self) -> usize {
            1
        }
        fn horizon(&self) -> usize {
            self.horizon
        }
        fn action_spec(&self) -> &ActionSpec {
            &self.spec
        }
        fn reset(&mut self, seed: u64) -> Vec<Vec<f32>> {
            self.rng = StdRng::seed_from_u64(seed);
            self.targets = self.draw();
            self.obs()
        }
        fn step(&mut self, actions: &[Vec<i64>]) -> StepResult {
            let mut costs = Vec::with_capacity(self.num_envs);
            for e in 0..self.num_envs {
                let a = actions[e][0].clamp(0, 9);
                costs.push((a as f32 - self.targets[e]).abs() as f64);
            }
            self.targets = self.draw();
            let next_obs = self.obs();
            StepResult { costs, next_obs }
        }
    }

    #[test]
    fn ppo_solves_toy_continuous_target() {
        let cfg = PpoConfig {
            iters: 60,
            train_paths: 64,
            eval_paths: 256,
            hidden: 32,
            lr: 3e-3,
            ppo_epochs: 4,
            minibatch: 128,
            ent_coef: 0.0,
            reward_scale: 1.0,
            bc_epochs: 0,
            eval_every: 15,
            seed: 2,
            ..PpoConfig::default()
        };
        let horizon = 8;
        let make_env = move |num_envs: usize, _eval: bool| -> Box<dyn PpoVecEnv> {
            Box::new(ToyContinuousTarget::new(num_envs, horizon))
        };
        let outcome = train_ppo(make_env, &cfg).expect("continuous ppo failed");
        let per_step = outcome.final_holdout_cost_mean / horizon as f64;
        assert!(
            per_step < 1.0,
            "continuous PPO failed to track target: per-step greedy cost {per_step:.3}"
        );
    }
}
