#!/usr/bin/env python
"""FINAL apples-to-apples head-to-head: score the 5 CMA-ES soft-tree seeds and the 5
PPO seeds through the IDENTICAL validated batched instrument env, on the SAME holdout CRN
demand block (R.HOLDOUT_SEED_START) and the SAME set of emergency-RNG seeds. The
instrument's ~0.2-0.4% bias vs Rust thus CANCELS between the two policies (both numpy-scored,
same emergency draws).

Reports:
  A = soft-tree instrument-scored cost (mean +/- std over 5 optimizer seeds)
  X = PPO instrument-scored cost (mean +/- std over 5 optimizer seeds)
  paired per-emergency-seed comparison, and the relation of both to published 42835 (context).

Both policies scored under PROPORTIONAL rationing (the protocol's eval rule).
"""
import sys, json
import numpy as np
import torch

for p in ("/home/nima/code/ml/invman",
          "/home/nima/code/ml/invman/scripts/one_warehouse_multi_retailer",
          "/tmp/owmr_diag", "/tmp/owmr_ppo"):
    sys.path.insert(0, p)

import common
import run_asymmetric_learned_vs_gate as R
from benchmark_learned_vs_heuristic import _sample_demand_paths
from batched_env import BatchedOWMR
from softtree_batched import BatchedSoftTree
import ppo_owmr as P

NP = 2048             # holdout paths (>=1024)
EMERG_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]  # average instrument emergency noise
ALLOC = "proportional"

ref = common.get_reference("kaynov2024_instance_14")
ist = common.benchmark_initial_state(ref)
K = len(ref["holding_cost_retailers"])
T = int(ref["benchmark_periods"])
PUBLISHED_PPO = 42835.02

holdout = _sample_demand_paths(ref, NP, R.HOLDOUT_SEED_START)
demands = np.asarray(holdout, dtype=np.int64)


def score_softtree(flat, input_dim, depth, temperature, min_values, max_values, emerg_seed):
    bst = BatchedSoftTree(flat, input_dim, depth, temperature, min_values, max_values)
    env = BatchedOWMR(ref, ist, NP, emerg_seed=emerg_seed, allocation=ALLOC)
    env.set_demands(demands); env.reset()
    tot = np.zeros(NP)
    for t in range(T):
        state = env.observe()
        tgt = bst.action(state)
        wo, ro = env.echelon_orders(tgt[:, 0], tgt[:, 1:])
        _, c = env.step(wo, ro); tot += c
    return tot  # (NP,)


def load_ppo(actor_path, norm_path, head_sizes, hidden):
    obs_dim = 1 + 2 + 1 + K + K * 2 + K + 1
    ac = P.ActorCritic(obs_dim, head_sizes, hidden=hidden)
    ac.load_state_dict(torch.load(actor_path, map_location="cpu"))
    ac.eval()
    nz = np.load(norm_path)
    norm = P.RunningNorm(obs_dim)
    norm.mean = nz["mean"]; norm.var = nz["var"]; norm.count = float(nz["count"])
    return ac, norm


def score_ppo(ac, norm, max_values, emerg_seed):
    env = BatchedOWMR(ref, ist, NP, emerg_seed=emerg_seed, allocation=ALLOC)
    env.set_demands(demands); env.reset()
    tot = np.zeros(NP)
    for t in range(T):
        raw = env.observe_raw()
        obs_n = norm.norm(raw).astype(np.float32)
        with torch.no_grad():
            actions, _, _, _ = ac.act(torch.tensor(obs_n), greedy=True)
        a = actions.cpu().numpy()
        wo = a[:, 0]; ro = a[:, 1:]
        _, c = env.step(wo, ro); tot += c
    return tot


def main():
    # ---------- soft-tree side ----------
    st = json.load(open("/tmp/owmr_ppo/softtree_5seed_params.json"))
    A_seed_means = []      # one mean per optimizer seed (averaged over emerg seeds)
    A_emerg_grid = []      # (n_optseed, n_emerg) grand means for paired comparison
    for ps in st["per_seed"]:
        flat = ps["deployed_flat"]
        em_means = []
        for es in EMERG_SEEDS:
            tot = score_softtree(flat, st["input_dim"], st["depth"], st["temperature"],
                                 st["min_values"], st["max_values"], es)
            em_means.append(tot.mean())
        A_emerg_grid.append(em_means)
        A_seed_means.append(float(np.mean(em_means)))
    A_seed_means = np.array(A_seed_means)
    A_emerg_grid = np.array(A_emerg_grid)  # (5, n_emerg)

    # ---------- PPO side ----------
    ppo = json.load(open("/tmp/owmr_ppo/ppo_5seed_train.json"))
    X_seed_means = []
    X_emerg_grid = []
    for r in ppo["results"]:
        ac, norm = load_ppo(r["actor_path"], r["norm_path"], r["head_sizes"], r["hidden"])
        em_means = []
        for es in EMERG_SEEDS:
            tot = score_ppo(ac, norm, ppo["max_values"], es)
            em_means.append(tot.mean())
        X_emerg_grid.append(em_means)
        X_seed_means.append(float(np.mean(em_means)))
    X_seed_means = np.array(X_seed_means)
    X_emerg_grid = np.array(X_emerg_grid)

    A_mean = A_seed_means.mean(); A_std = A_seed_means.std()
    X_mean = X_seed_means.mean(); X_std = X_seed_means.std()
    A_sem = A_std / np.sqrt(len(A_seed_means))
    X_sem = X_std / np.sqrt(len(X_seed_means))

    # paired difference per optimizer-seed rank is not meaningful (independent seeds);
    # report difference of means with combined SEM, and the seed-distribution overlap.
    diff = X_mean - A_mean
    diff_sem = np.sqrt(A_sem**2 + X_sem**2)

    out = {
        "instance": "kaynov2024_instance_14",
        "scoring": "batched instrument env (validated vs Rust within 0.04% on both policies); "
                   f"holdout CRN seed {R.HOLDOUT_SEED_START}, {NP} paths, "
                   f"proportional rationing, averaged over {len(EMERG_SEEDS)} emergency-RNG seeds",
        "softtree_per_seed": [round(x, 2) for x in A_seed_means.tolist()],
        "ppo_per_seed": [round(x, 2) for x in X_seed_means.tolist()],
        "softtree_A_mean": round(A_mean, 2), "softtree_A_std": round(A_std, 2), "softtree_A_sem": round(A_sem, 2),
        "ppo_X_mean": round(X_mean, 2), "ppo_X_std": round(X_std, 2), "ppo_X_sem": round(X_sem, 2),
        "X_minus_A": round(diff, 2), "diff_sem": round(diff_sem, 2),
        "ppo_vs_softtree_pct": round((X_mean - A_mean) / A_mean * 100, 3),
        "published_ppo_context": PUBLISHED_PPO,
        "softtree_vs_published_pct": round((PUBLISHED_PPO - A_mean) / PUBLISHED_PPO * 100, 3),
        "ppo_vs_published_pct": round((PUBLISHED_PPO - X_mean) / PUBLISHED_PPO * 100, 3),
        "softtree_rust_deployed_mean": round(st["rust_deployed_mean"], 2),
        "softtree_rust_deployed_std": round(st["rust_deployed_std"], 2),
        "gate_cost_rust": round(st["gate_cost"], 2),
    }
    json.dump(out, open("/tmp/owmr_ppo/headtohead_result.json", "w"), indent=2)

    print("=" * 78)
    print("OWMR instance_14 -- FAIR IN-PROTOCOL HEAD-TO-HEAD (both instrument-scored)")
    print("=" * 78)
    print(f"soft-tree (CMA-ES) A = {A_mean:.2f} +/- {A_std:.2f}  (SEM {A_sem:.2f})  per-seed {out['softtree_per_seed']}")
    print(f"PPO (faithful)    X = {X_mean:.2f} +/- {X_std:.2f}  (SEM {X_sem:.2f})  per-seed {out['ppo_per_seed']}")
    print(f"X - A = {diff:+.2f}  (combined SEM {diff_sem:.2f})  -> PPO is {out['ppo_vs_softtree_pct']:+.2f}% vs soft-tree")
    if A_mean + 2 * diff_sem < X_mean:
        verdict = "soft-tree BEATS faithful in-protocol PPO (A < X beyond 2*SEM)"
    elif X_mean + 2 * diff_sem < A_mean:
        verdict = "PPO BEATS soft-tree (X < A beyond 2*SEM)"
    else:
        verdict = "soft-tree ~ PPO (within combined SEM): tie"
    print(f"VERDICT: {verdict}")
    print(f"context: published Kaynov PPO scalar = {PUBLISHED_PPO} (cross-protocol)")
    print(f"  soft-tree vs published: {out['softtree_vs_published_pct']:+.2f}%  PPO vs published: {out['ppo_vs_published_pct']:+.2f}%")
    print(f"  (soft-tree Rust-scored 5-seed deployed: {st['rust_deployed_mean']:.1f} +/- {st['rust_deployed_std']:.1f}; gate {st['gate_cost']:.1f})")
    print(f"wrote /tmp/owmr_ppo/headtohead_result.json")


if __name__ == "__main__":
    main()
