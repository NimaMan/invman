# OWMR `instance_14` — faithful in-protocol PPO baseline + like-for-like head-to-head

A from-scratch PPO baseline for one-warehouse-multi-retailer (Kaynov 2024) `instance_14`,
trained and scored **under our exact protocol**, so the learned soft-tree policy can be
compared to PPO **like-for-like** (rather than against the published cross-protocol scalar).

## Why this exists
The published Kaynov PPO number (42835.02) is a single scalar under that paper's own demand
convention (an unverified `N(mu, sigma)`), so being above/below it is **cross-protocol context,
not a head-to-head comparison**. This baseline retrains PPO on the *same* environment, demand
convention, and held-out CRN block we score our policy on, and scores **both policies the same
way** — the only comparison that licenses an honest "beat PPO" statement.

## What's here (all reproducible; model binaries are regenerated, not committed)
- `batched_env.py` — a **validated numpy replica** of the OWMR env (the Rust env in
  `src/problems/one_warehouse_multi_retailer/` is the source of truth; this replica reproduces
  the Rust gate to **-0.006%** and the Rust soft-tree to **+0.038%** on the holdout block — see
  `validate_batched.py`). Used so a PyTorch policy can be trained/scored without round-tripping
  the Rust oracle per step.
- `ppo_owmr.py` — a compact from-scratch PPO actor-critic: Kaynov-style **multi-discrete heads
  linear in K** (one order head per retailer + warehouse), random-rationing feasibility during
  training, GAE + clipped surrogate + value/entropy. BC-warm-started to the best tuned heuristic
  (a *fair* start at the gate).
- `train_ppo_5seed.py` — trains 5 PPO seeds; `retrain_softtree_5seed_saveparams.py` — retrains
  the soft-tree 5 seeds; `score_headtohead.py` — scores both on the same holdout CRN.
- `headtohead_result.json` — the result.

## Result (5 seeds each; same validated instrument; holdout CRN seed 900000, 2048 paths)
| policy | cost (mean ± std) |
|---|---|
| **CMA-ES soft-tree (ours)** | **43,801 ± 247** |
| faithful in-protocol PPO | 50,475 ± 391 |

- **Our soft-tree beats a faithfully-retrained PPO by ~13%** (PPO +15% more costly), robustly —
  no seed overlap, beyond 2× combined SEM.
- The PPO **BC-starts at the gate, climbs ~1.5–1.8%, then destabilizes** every seed and converges
  back to ≈ the heuristic (gate 50,484) — the known PPO instability on heavy-tail partial-backorder.
- Our retrained PPO is **+17.8% above the published 42835**, i.e. the published scalar is **not
  reproducible under our protocol** — it reflects a different demand convention.

## The two honest claims (keep them separate)
1. vs the **published cross-protocol scalar** (42835): we are +2.25% above — **not a beat** (never claimed).
2. vs a **faithfully-retrained in-protocol PPO**: we **win by ~13%** — the defensible head-to-head.

## Status / caveats
- This is **one** competent from-scratch PPO (BC-warm-start makes the comparison fair; PPO is
  tuning-sensitive). Before this becomes a **paper headline**, the PPO baseline should be hardened
  (a tuning sweep / a second reference implementation) — tracked as a **Tier-3** plan item.
- The demand-convention uncertainty moves the absolute scale and the published-scalar comparison,
  **not** the in-protocol head-to-head (both policies are on the same axis here).
