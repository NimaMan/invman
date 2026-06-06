#!/usr/bin/env python
# =============================================================================
# Adversarial verification of an OWMR PPO-scalar beat on DISJOINT held-out blocks
# =============================================================================
# OBJECTIVE
#   The campaign's deployment floor picks the best of {trained xbest, anchor, gate}
#   on ONE held-out CRN block (seed 900_000). That introduces a mild
#   selection-on-900_000 bias. This script removes it: it rebuilds the saved
#   trained policy and re-scores it on several INDEPENDENT 4096-path CRN blocks
#   (seeds 2_000_000 / 3_000_000 / 4_000_000) that were never used in training,
#   gate search, OR deployment selection. If the learned cost stays below the
#   published PPO scalar on every disjoint block, the below-PPO result is robust
#   (not block-900_000 luck).
#
#   For proportional / min_shortage allocation the rationing is deterministic given
#   the demand paths, so a disjoint demand block is a clean generalization test.
#
# ALGORITHM
#   1. Rebuild the exact model structure (depth/split/leaf/action/state) used by the
#      saved checkpoint via common.build_soft_tree_model.
#   2. Load the saved flat params (model_params.npy).
#   3. For each block seed: sample 4096 demand paths, evaluate the policy under both
#      proportional and min_shortage, report mean +/- SEM and the signed gap to PPO.
#   4. Also re-score the tuned gate base-stock (W, R) on the same blocks for the
#      paired gate-beat robustness check.
# =============================================================================
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "owmr_runner",
        str(PKG_ROOT / "scripts" / "one_warehouse_multi_retailer"
            / "run_asymmetric_learned_vs_gate.py"),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_json", required=True,
                    help="campaign result JSON of the candidate (carries structure + model npy + gate)")
    ap.add_argument("--blocks", type=int, nargs="+",
                    default=[900_000, 2_000_000, 3_000_000, 4_000_000],
                    help="held-out CRN block seed starts (900_000 = the original, as sanity check)")
    ap.add_argument("--paths", type=int, default=4096)
    args = ap.parse_args()

    R = _load_runner()
    import sys
    sys.path.insert(0, str(PKG_ROOT / "scripts" / "one_warehouse_multi_retailer"))
    import common

    d = json.loads(Path(args.result_json).read_text())
    ref = common.get_reference(d["instance"])
    action_mode = d["policy_action_mode"]
    state_mode = d["policy_state_mode"]
    ppo = float(d["published_ppo_cost"])
    npy = Path(d["trained_model_params_npy"])
    flat = np.load(npy).astype(np.float32).reshape(-1).tolist()

    model = common.build_soft_tree_model(
        ref, depth=int(d["depth"]), temperature=float(d["temperature"]),
        split_type=d["split_type"], leaf_type=d["leaf_type"],
        policy_action_mode=action_mode, policy_state_mode=state_mode,
    )
    n_model = len(model.get_model_flat_params())
    assert n_model == len(flat), f"param mismatch: model {n_model} vs npy {len(flat)}"

    allocs = (["proportional"] if action_mode == "direct_orders"
              else ["proportional", "min_shortage"])

    print(f"=== Adversarial PPO-beat verification: {d['instance']} ===")
    print(f"  structure: {action_mode}/{state_mode}/d{d['depth']}/{d['split_type']}/"
          f"{d['leaf_type']}  params={n_model}")
    print(f"  published PPO scalar (cross-protocol) = {ppo:.2f}")
    print(f"  original block-900000 deployed learned = {d['learned_cost']:.2f} "
          f"({d['learned_vs_ppo_pct']:+.3f}% vs PPO)")
    print(f"  gate (W={d['gate_warehouse_level']}, R={d['gate_retailer_levels']}) "
          f"= {d['gate_cost']:.2f}")
    print()

    # ---- learned policy on each disjoint block ----
    header = f"{'block':>10} | {'alloc':>12} | {'learned':>10} +/- SEM  | {'vs PPO %':>9} | {'verdict':>10}"
    print(header)
    print("-" * len(header))
    learned_block_best = {}
    for blk in args.blocks:
        paths = R._sample_demand_paths(ref, args.paths, blk)
        block_means = {}
        for alloc in allocs:
            costs = R._soft_tree_on_paths(ref, model, flat, alloc, action_mode,
                                          paths, R.ALLOC_SEED_HOLDOUT)
            mean = float(costs.mean())
            sem = float(costs.std() / np.sqrt(costs.size))
            vs_ppo = (ppo - mean) / ppo * 100.0
            verdict = "BEATS PPO" if mean < ppo else "above PPO"
            tag = "  (orig)" if blk == 900_000 else ""
            print(f"{blk:>10} | {alloc:>12} | {mean:>10.2f} +/- {sem:4.2f} | "
                  f"{vs_ppo:>+8.3f}% | {verdict:>10}{tag}")
            block_means[alloc] = mean
        # deployed = the better allocation (mirrors the runner's headline)
        learned_block_best[blk] = min(block_means.values())
    print()

    # ---- summary across DISJOINT blocks only (exclude the original 900000) ----
    disjoint = [b for b in args.blocks if b != 900_000]
    dj = [learned_block_best[b] for b in disjoint]
    if dj:
        arr = np.array(dj)
        print(f"DISJOINT blocks {disjoint}:")
        print(f"  learned (best alloc per block): mean={arr.mean():.2f}  "
              f"min={arr.min():.2f}  max={arr.max():.2f}")
        n_beat = int((arr < ppo).sum())
        print(f"  blocks below PPO {ppo:.2f}: {n_beat}/{len(arr)}")
        worst = arr.max()
        print(f"  WORST disjoint block vs PPO: {(ppo - worst)/ppo*100:+.3f}% "
              f"({'still beats' if worst < ppo else 'ABOVE'} PPO)")
        verdict = ("ROBUST PPO-scalar beat" if n_beat == len(arr)
                   else "NOT robust (some disjoint block above PPO)")
        print(f"  ==> {verdict}")


if __name__ == "__main__":
    main()
