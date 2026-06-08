# `src/core/ppo/` — reusable in-crate PPO trainer

This folder implements **Proximal Policy Optimization as a first-class, reusable
trainer** inside the Rust crate, parallel in role to the Python CMA-ES optimizer.
Every problem family can be trained by PPO as easily as by evolution strategies.

The whole module is behind the optional **`ppo` cargo feature** (`--features ppo`)
so the default crate build stays candle-free and fast.

## Why this exists
The optimizer loop today lives in Python (`invman/cmaes.py` + `es_mp.py`) and uses
the Rust crate only as a **scalar-cost rollout oracle**. PPO is gradient-based and
needs (a) per-step transitions and (b) a differentiable policy/value network —
neither of which the scalar oracle exposes. This module adds both in Rust using
the pure-Rust [`candle`](https://github.com/huggingface/candle) autodiff backend.

## Backend choice
- **candle 0.9.x** (pinned). Pure-Rust on CPU, no system C++/CUDA libs (unlike
  `tch-rs`/libtorch), ships `AdamW` + all needed tensor ops. 0.10+ is avoided
  because it adds a non-optional `tokenizers` dep that forces `rayon ^1.10` and
  conflicts with the crate's `rayon =1.7.0` pin; 0.9.2 needs only `rayon ^1.7.0`.

## Files (built + tested)
| file | functionality |
|---|---|
| `mod.rs` | Module root + full algorithmic overview. |
| `candle_backend_smoke_test.rs` | De-risk test: 2-layer MLP in candle, `backward()` + `AdamW`, asserts MSE drops >10x. |
| `environment.rs` | `PpoVecEnv` trait (reusable env seam) + `ActionSpec` (MultiDiscrete / Continuous) + `StepResult`. |
| `running_norm.rs` | Welford running observation normalizer (updated in training, frozen at eval). |
| `gae.rs` | GAE(lambda) advantages + returns, finite-horizon (terminal value 0). |
| `multi_discrete_head.rs` | Factored multi-discrete categorical: CPU sample/greedy + differentiable joint log-prob/entropy (candle). |
| `actor_critic.rs` | candle shared-Tanh-trunk net + per-head logits + scalar value head. |
| `ppo_trainer.rs` | The PPO loop: rollout → GAE → clipped+value-clipped+entropy update, global-norm grad clip, Adam, BC warm-start, best-checkpoint, greedy eval. `PpoConfig`/`PpoOutcome`. Toy-env test validates the whole algorithm. |

Per-problem environments live with their problem, e.g.
`src/problems/one_warehouse_multi_retailer/ppo_environment.rs` (`OwmrPpoEnv`,
implements `PpoVecEnv` by reusing the canonical `step_state`) and
`.../ppo_bindings.rs` (the `one_warehouse_multi_retailer_train_ppo` pyo3 entry).

## Status / validation
- Generic PPO core: **built + tested** (candle smoke, GAE, running-norm,
  multi-discrete head, and an end-to-end toy-context-bandit PPO test all pass).
- OWMR env: **built + fidelity-checked** — the steppable env's gate holdout cost
  is 50,453 vs the known Rust gate 50,445 (~0.02%).
- OWMR PPO reproduction (≈ in-house PyTorch PPO ~50,475): in progress.

## Still to do
- Continuous / diagonal-Gaussian action head (for scalar problems like
  lost_sales and the serial continuous-level problem) — `ActionSpec::Continuous`
  is reserved but the trainer currently supports `MultiDiscrete` only.
- Add a second problem's `PpoVecEnv` to demonstrate cross-problem reuse.
- Python `training_method="ppo"` dispatch in `es_mp.train` / `config.py`.

## Validation discipline
Validated by reproducing the existing env-faithful in-protocol PyTorch PPO
baseline (OWMR instance_14: **50,475 ± 391** over 5 seeds; see
`scripts/one_warehouse_multi_retailer/ppo_baseline/`). Kaynov 2024's published
**42,835 is not a reproduction target** — their PPO hyperparameters and demand
convention are not published, so their exact run cannot be replicated; this is
stated wherever the comparison appears. Results are reported as mean ± std over
≥ 5 training seeds.
