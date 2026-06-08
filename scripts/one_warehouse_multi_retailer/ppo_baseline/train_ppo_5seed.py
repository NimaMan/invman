#!/usr/bin/env python
"""Train 5 PPO seeds (faithful, competent, BC-warm-started) on the batched OWMR
instance_14 instrument env. Save each seed's best actor state_dict + running normalizer
+ learning curve, for paired instrument scoring against the soft-tree.

Uses the higher-LR config that demonstrably climbs off the BC warm-start (best-checkpointed
so the late PPO drift doesn't hurt the deployed policy -- standard PPO model selection on a
validation block; here the holdout-greedy cost is the selection signal).
"""
import sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, "/tmp/owmr_ppo")
import ppo_owmr as P

OUT = "/tmp/owmr_ppo"
MAX_VALUES = [255, 85, 75, 65, 55, 45, 30, 6, 18, 43, 54]
GATE_W = 440
GATE_R = [33, 30, 28, 26, 27, 30, 2, 10, 29, 39]


def run_seed(seed, iters):
    parser = P.default_args()
    a = parser.parse_args([])
    a.seed = seed
    a.iters = iters
    a.eval_every = 5
    a.bc_epochs = 120
    a.bc_paths = 512
    a.bc_lr = 1e-3
    a.train_paths = 384
    a.eval_paths = 1024
    a.lr = 1.2e-4
    a.ent_coef = 0.001
    a.clip = 0.15
    a.ppo_epochs = 5
    a.minibatch = 4096
    a.verbose = 1
    a.train_alloc = "random_sequential"
    a.max_values = MAX_VALUES
    a.gate_W = GATE_W
    a.gate_R = GATE_R
    t0 = time.time()
    ac, norm, curve, best, fmean, fstd, mv = P.train_ppo(a, log_prefix=f"[s{seed}] ")
    dt = time.time() - t0
    # save actor + normalizer
    torch.save(ac.state_dict(), f"{OUT}/ppo_actor_seed{seed}.pt")
    np.savez(f"{OUT}/ppo_norm_seed{seed}.npz", mean=norm.mean, var=norm.var, count=norm.count)
    return {
        "seed": seed,
        "best_holdout_greedy": float(best),
        "final_holdout_greedy": float(fmean),
        "curve": curve,
        "train_seconds": dt,
        "actor_path": f"{OUT}/ppo_actor_seed{seed}.pt",
        "norm_path": f"{OUT}/ppo_norm_seed{seed}.npz",
        "head_sizes": [m + 1 for m in MAX_VALUES],
        "hidden": a.hidden,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()
    results = []
    for s in args.seeds:
        print(f"\n===== PPO SEED {s} =====", flush=True)
        r = run_seed(s, args.iters)
        results.append(r)
        print(f"[s{s}] best {r['best_holdout_greedy']:.1f} final {r['final_holdout_greedy']:.1f} "
              f"({r['train_seconds']:.0f}s)", flush=True)
        # persist incrementally
        json.dump({"results": results, "max_values": MAX_VALUES, "gate_W": GATE_W,
                   "gate_R": GATE_R},
                  open(f"{OUT}/ppo_5seed_train.json", "w"), indent=2)
    bests = np.array([r["best_holdout_greedy"] for r in results])
    print(f"\n[PPO 5-seed] best-checkpoint holdout-greedy mean {bests.mean():.2f} +/- {bests.std():.2f}", flush=True)
    print(f"per-seed best: {[round(x,1) for x in bests]}", flush=True)


if __name__ == "__main__":
    main()
