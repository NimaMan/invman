//! # `environment` — the reusable PPO environment seam
//!
//! ## Objective
//! Define the single trait every problem implements so the *same* PPO trainer
//! can train *any* problem. The trainer only ever talks to a `PpoVecEnv`: a
//! batched (vectorized) finite-horizon environment that exposes raw
//! observations and a step function. Each problem (OWMR, lost_sales, serial,
//! ...) wraps its existing Rust dynamics behind this trait — for OWMR that means
//! wrapping `step_state` + the demand/allocation/emergency helpers that the
//! scalar-cost rollout oracle already contains but does not expose per step.
//!
//! ## Design
//! - **Batched / vectorized.** PPO collects `num_envs` parallel trajectories of
//!   length `horizon()` per update (one "episode batch"). The env advances all
//!   parallel copies together; this mirrors the reference PyTorch baseline's
//!   `BatchedOWMR` and amortizes per-step overhead.
//! - **Raw observations.** `reset`/`step` return UNNORMALIZED features. The
//!   trainer owns a running normalizer (see `running_norm`) so normalization is
//!   identical across problems and frozen at eval — exactly as the reference.
//! - **Cost, not reward.** `step` returns the per-env immediate COST. The
//!   trainer converts to reward `= -cost / reward_scale`. Keeping cost in the
//!   env keeps the env semantics native to the inventory problem; the reward
//!   shaping (scale, sign) is a trainer concern.
//! - **Action shape via `ActionSpec`.** The env declares whether its action is
//!   factored multi-discrete (a Categorical per dimension, e.g. OWMR order
//!   quantities) or continuous (a diagonal Gaussian, e.g. serial echelon
//!   levels). The trainer's action head is chosen from this spec, so one trainer
//!   serves every action geometry.
//! - **Greedy / eval mode.** `set_eval_mode` lets a problem switch any
//!   stochastic-but-policy-independent dynamics that differ between training and
//!   evaluation (OWMR: random-sequential rationing while training, proportional
//!   while scoring — the protocol's eval rule).
//! - **Gate actions (optional).** `gate_actions` lets a problem expose its
//!   strongest heuristic's action at the current state, so the trainer can
//!   behavior-clone (warm-start) to it. Returning `None` disables BC.

/// The action geometry an environment exposes, used to pick the trainer's head.
#[derive(Clone, Debug, PartialEq)]
pub enum ActionSpec {
    /// Factored multi-discrete: one Categorical per dimension `j` over
    /// `{0, 1, ..., sizes[j]-1}`. The joint log-prob is the sum of per-dimension
    /// log-probs; the joint entropy is the sum of per-dimension entropies. For
    /// OWMR DirectOrders, `sizes[j] = max_order_j + 1` and the category index IS
    /// the order quantity.
    MultiDiscrete { sizes: Vec<usize> },
    /// Diagonal Gaussian over `dim` continuous actions (per-dim mean + learnable
    /// log-std). Integer-action problems use this by rounding+clipping the
    /// sampled action inside their own `step`; the serial problem uses it
    /// natively for fractional echelon levels.
    Continuous { dim: usize },
}

impl ActionSpec {
    /// Number of action dimensions (heads for multi-discrete, action width for
    /// continuous). Used to size action buffers.
    pub fn action_dim(&self) -> usize {
        match self {
            ActionSpec::MultiDiscrete { sizes } => sizes.len(),
            ActionSpec::Continuous { dim } => *dim,
        }
    }
}

/// Result of advancing all parallel environments by one step.
#[derive(Clone, Debug)]
pub struct StepResult {
    /// Per-env immediate cost (length `num_envs`). The trainer turns this into a
    /// reward `= -cost / reward_scale`.
    pub costs: Vec<f64>,
    /// Per-env RAW (unnormalized) observation AFTER the step (length
    /// `num_envs`, each `obs_dim`). The trainer normalizes these.
    pub next_obs: Vec<Vec<f32>>,
}

/// A batched, finite-horizon environment the PPO trainer drives. One PPO
/// "episode batch" runs `num_envs` parallel trajectories for `horizon` steps.
pub trait PpoVecEnv {
    /// Number of parallel trajectories advanced together.
    fn num_envs(&self) -> usize;
    /// Length of the raw observation vector for one env.
    fn obs_dim(&self) -> usize;
    /// Episode length (fixed finite horizon). GAE uses terminal value 0 at the
    /// last step.
    fn horizon(&self) -> usize;
    /// The action geometry, used to construct the trainer's action head.
    fn action_spec(&self) -> &ActionSpec;

    /// Switch between training (`false`) and evaluation (`true`) dynamics that
    /// are policy-independent but differ by protocol (OWMR rationing rule).
    /// Default: no-op for problems with no such distinction.
    fn set_eval_mode(&mut self, _eval: bool) {}

    /// Reset all parallel envs to the start of a fresh episode batch keyed by
    /// `seed` (which fixes the exogenous randomness, e.g. demand paths, so a
    /// held-out seed reproduces the same paths — common random numbers).
    /// Returns the initial RAW observations (`num_envs` x `obs_dim`).
    fn reset(&mut self, seed: u64) -> Vec<Vec<f32>>;

    /// Apply one action per env and advance one step. `actions[e]` is the action
    /// for env `e`: for `MultiDiscrete` it is the chosen category per dimension
    /// (as `i64`); for `Continuous` the trainer passes the rounded/native action
    /// the problem expects (the problem interprets it). Returns per-env cost and
    /// the next raw observation.
    fn step(&mut self, actions: &[Vec<i64>]) -> StepResult;

    /// Optional: the strongest heuristic's ("gate") action at the CURRENT state
    /// of each env, for behavior-cloning warm-start. `actions[e]` matches the
    /// `step` action encoding. Returning `None` disables BC for this problem.
    fn gate_actions(&self) -> Option<Vec<Vec<i64>>> {
        None
    }
}
