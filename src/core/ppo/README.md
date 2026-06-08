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
| `gaussian_head.rs` | Diagonal-Gaussian continuous head: CPU sample/greedy + differentiable log-prob/entropy (mean head + learnable log_std). |
| `actor_critic.rs` | candle shared-Tanh-trunk net; dispatches a multi-discrete OR continuous (`PolicyOutput`) head from the env's `ActionSpec` + a scalar value head. |
| `ppo_trainer.rs` | The PPO loop: rollout → GAE → clipped+value-clipped+entropy update, global-norm grad clip, Adam, BC warm-start, best-checkpoint, greedy eval. Action sampling + log-prob/entropy dispatch on the action type. `PpoConfig`/`PpoOutcome`. Toy multi-discrete AND toy continuous tests validate both paths. |

Per-problem environments live with their problem:
- `src/problems/one_warehouse_multi_retailer/ppo_environment.rs` (`OwmrPpoEnv`,
  multi-discrete, reuses the canonical `step_state`) + `ppo_bindings.rs`.
- `src/problems/lost_sales/vanilla/ppo_environment.rs` (`LostSalesPpoEnv`,
  continuous scalar order, reuses `epoch_cost`) + `ppo_bindings.rs`.

Python side: `invman.ppo_trainer.train_ppo(problem, **hp)` dispatches per-problem
bindings (registry `_PPO_BINDINGS`); adding a problem = env + binding + one row.

## Status / validation
- Generic PPO core: **built + tested** — candle smoke, GAE, running-norm, both
  action heads, and BOTH a toy multi-discrete and a toy continuous end-to-end PPO
  test pass.
- OWMR (multi-discrete): **validated** — env gate cost 50,453 vs known Rust gate
  50,445 (0.02%); full 60-iter PPO reproduces the in-house PyTorch PPO
  (after-BC 50,481 → best 49,806), matching the BC-start/climb/stabilize regime.
- lost_sales (continuous): **validated** — env base-stock gate 5.35/period; PPO
  trains end-to-end and lands ≈ the gate (best 5.48/period), the expected
  PPO≈gate behavior (opt 4.73, capped base-stock 4.80).

## Still to do (follow-on)
- Continuous-VECTOR problems (e.g. multi_echelon/serial fractional echelon
  levels) — the diagonal-Gaussian head supports `dim>1`; needs the serial env.
- Full 5-seed seed-robust OWMR reporting number (~55 min; 1-seed validated).
- The continuous BC collapses log_std (limits PPO improvement past the gate on
  lost_sales) — a known tuning item (exclude log_std from BC / entropy floor).

## Validation discipline
Validated by reproducing the existing env-faithful in-protocol PyTorch PPO
baseline (OWMR instance_14: **50,475 ± 391** over 5 seeds; see
`scripts/one_warehouse_multi_retailer/ppo_baseline/`). Kaynov 2024's published
**42,835 is not a reproduction target** — their PPO hyperparameters and demand
convention are not published, so their exact run cannot be replicated; this is
stated wherever the comparison appears. Results are reported as mean ± std over
≥ 5 training seeds.
