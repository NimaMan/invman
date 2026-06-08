//! # `ppo` — reusable in-crate PPO trainer (candle autodiff backend)
//!
//! ## Objective
//! Provide Proximal Policy Optimization as a **first-class, reusable trainer**
//! inside the Rust crate, parallel in role to the Python CMA-ES optimizer, so
//! that *every* problem family (one_warehouse_multi_retailer, lost_sales,
//! dual_sourcing, multi_echelon, ...) can be trained by a gradient method as
//! easily as by evolution strategies. The optimizer loop today lives in Python
//! (`invman/cmaes.py` + `es_mp.py`) and calls the Rust crate purely as a
//! scalar-cost rollout oracle. PPO needs per-step transitions and a
//! differentiable policy/value network, so it is implemented here in Rust with
//! the pure-Rust `candle` autodiff backend (feature-gated: `--features ppo`).
//!
//! ## Why candle / why feature-gated
//! `candle` (HuggingFace) is pure-Rust on CPU, links no system C++/CUDA libs
//! (unlike `tch-rs`/libtorch), and ships AdamW + every tensor op PPO needs.
//! Pinned to 0.9.x because 0.10+ adds a non-optional `tokenizers` dependency
//! that forces `rayon ^1.10` and conflicts with the crate's `rayon =1.7.0`
//! pin. The whole module is behind the optional `ppo` feature so the default
//! crate build (used by the rest of the test suite and the CMA-ES oracle) stays
//! candle-free and fast.
//!
//! ## Planned algorithmic structure (built incrementally)
//! 1. `PpoEnv` trait — the reusable env seam. Each problem wraps its existing
//!    (currently scalar-only) Rust rollout as a stateful `reset(seed) -> obs`,
//!    `step(action) -> (next_obs, reward, done)` environment. This is the
//!    missing primitive: the per-step `StepOutcome` already carries
//!    `reward`/`next_state`; the env loop just discards it today.
//! 2. `ActionHead` trait + four implementations — Gaussian scalar,
//!    diagonal-Gaussian vector, categorical, factored multi-discrete — each
//!    providing `sample`, `evaluate` (recompute log-prob + entropy of a stored
//!    action), and `decode_to_env` (apply the problem's order-up-to / residual
//!    adapter so PPO emits into the SAME coordinate system the soft-tree uses,
//!    not raw orders).
//! 3. `ActorCritic` — shared MLP trunk + policy head(s) + scalar value head.
//! 4. GAE(lambda) advantages + clipped surrogate + value clipping + entropy
//!    bonus + global-norm gradient clipping + Adam, optionally behavior-cloned
//!    (BC) warm-started to a problem gate.
//! 5. pyo3 binding exposing `train_ppo(...)` so Python's `es_mp.train` can
//!    dispatch `training_method="ppo"` to it for any problem.
//!
//! ## Validation discipline
//! The trainer is validated by reproducing the existing, env-faithful,
//! in-protocol PyTorch PPO baseline (OWMR instance_14: 50,475 +/- 391 over 5
//! seeds; see `scripts/one_warehouse_multi_retailer/ppo_baseline/`). Kaynov
//! 2024's published 42,835 is NOT a reproduction target: their PPO
//! hyperparameters and demand convention are not published, so their exact run
//! cannot be replicated — this is stated wherever the comparison is reported.
//! Results are always reported as mean +/- std over >= 5 training seeds.

pub mod actor_critic;
pub mod candle_backend_smoke_test;
pub mod environment;
pub mod gae;
pub mod multi_discrete_head;
pub mod ppo_trainer;
pub mod running_norm;
