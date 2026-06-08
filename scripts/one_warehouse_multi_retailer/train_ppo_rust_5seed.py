#!/usr/bin/env python
"""Seed-robust driver for the REUSABLE RUST PPO TRAINER on OWMR instance_14.

================================ OBJECTIVE ===================================
Invoke the first-class Rust PPO trainer (core::ppo, candle autodiff) from Python
the same way CMA-ES is invoked -- as a reusable trainer -- and report the
seed-robust held-out result (mean +/- std over >= 5 training seeds), against the
in-protocol echelon base-stock GATE anchor.

The whole PPO training loop (BC warm-start to the gate, GAE, clipped surrogate,
value clipping, entropy, Adam, best-checkpoint, greedy holdout eval) runs INSIDE
Rust via `invman_rust.one_warehouse_multi_retailer_train_ppo`. No PyTorch.

HONESTY NOTE (per project decision 2026-06-08): Kaynov 2024's published PPO
number (42,835) is NOT a reproduction target -- their PPO hyperparameters and
demand convention are not published, so their exact run cannot be replicated.
This trainer is validated against the in-house in-protocol PyTorch PPO (~50,475)
and the echelon gate (~50,445); we report our own seed-robust in-protocol result.

Build the extension with the PPO feature first:
  maturin develop --release --features python-extension,ppo
Then:
  python scripts/one_warehouse_multi_retailer/train_ppo_rust_5seed.py --seeds 0 1 2 3 4 --iters 60
=============================================================================
"""
import argparse
import json
import time

import numpy as np

import invman_rust


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    results = []
    gate = None
    for seed in args.seeds:
        t0 = time.time()
        out = invman_rust.one_warehouse_multi_retailer_train_ppo(
            seed=seed,
            iters=args.iters,
            verbose=args.verbose,
        )
        dt = time.time() - t0
        gate = out["gate_holdout_cost"]
        results.append(out)
        print(
            f"[seed {seed}] gate={gate:.1f} "
            f"best={out['best_holdout_cost']:.1f} "
            f"final={out['final_holdout_cost_mean']:.1f} ({dt:.0f}s)",
            flush=True,
        )

    bests = np.array([r["best_holdout_cost"] for r in results], dtype=float)
    print("\n================ REUSABLE RUST PPO (OWMR instance_14) ================")
    print(f"gate (echelon base-stock, holdout)      : {gate:.2f}")
    print(
        f"PPO best-checkpoint holdout (mean +/- std): "
        f"{bests.mean():.2f} +/- {bests.std():.2f}  over {len(bests)} seeds"
    )
    print(f"per-seed best: {[round(float(x), 1) for x in bests]}")
    print("reference in-house PyTorch PPO           : ~50,475 (the validation target)")
    print("=====================================================================")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "gate_holdout_cost": gate,
                    "seeds": list(args.seeds),
                    "iters": args.iters,
                    "best_holdout_mean": float(bests.mean()),
                    "best_holdout_std": float(bests.std()),
                    "per_seed": results,
                },
                handle,
                indent=2,
            )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
