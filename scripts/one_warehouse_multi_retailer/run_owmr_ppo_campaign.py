#!/usr/bin/env python
# =============================================================================
# OWMR PPO-gap campaign orchestrator
# =============================================================================
# OBJECTIVE
#   Find a one-warehouse multi-retailer (Kaynov 2024) learned soft-tree policy
#   that BEATS the published PPO cost on a currently-below-PPO instance, using
#   autoresearch (CMA-ES policy search) over the *callable* policy-design levers.
#   "Beat PPO" == held-out `learned_vs_ppo_pct > 0` beyond paired SEM, on the
#   same 4096-path CRN protocol as `run_asymmetric_learned_vs_gate.py`.
#
#   Genuine open targets (same-context Kaynov PPO rows):
#     - instance_12  partial_backorder K=3 heterogeneous : frontier 1139.34, gap -1.82%
#     - instance_13  partial_backorder K=10 symmetric hi-CV: frontier 84399,  gap -5.86%
#
# WHY THESE LEVERS (the prior 3 sessions plateaued ~1.8% short on inst 12)
#   The depth-2 axis-aligned local neighborhood is saturated and every prior
#   "trained xbest" overfit the small training batch and fell back to the
#   gate-reproducing anchor. So this campaign deliberately targets the UNTRIED
#   directions most likely to break that plateau:
#     (A) anti-overfit refine  : restart the frontier checkpoint with MANY more
#                                training paths (batch 32-64, --same_seed CRN) so
#                                a genuinely generalizing state-dependent deviation
#                                can survive the held-out re-score.
#     (B) broad multi-start    : warm-start at the gate with moderate sigma + long
#                                budget + diverse seeds to escape the single
#                                converged lineage basin (1154->...->1139.34).
#     (C) oblique gating       : linear-combination splits (untried with the rich
#                                decoupled-allocation + absolute-augmented mode).
#     (D) gentle depth-3       : depth-3 with SMALL sigma (the failed prior probe
#                                used sigma 0.30 and overfit; 0.05 stays near anchor).
#     (E) deep symmetric tree  : inst 13 is symmetric (2 controls) so a depth-3/4
#                                tree adds pure state-dependence with low overfit
#                                risk -- the highest-yield untried lever for inst 13.
#     (F/G) refine inst-13 frontier: small-sigma restarts of the 84399 checkpoint
#                                with absolute-augmented state / decoupled alloc.
#   The honest deployment floor inside the runner (deploy best of {trained xbest,
#   init anchor, gate}) guarantees NO config reports worse than the gate, so every
#   arm is downside-safe; the only question is whether any flips past PPO.
#
# ALGORITHM
#   1. Build a list of config dicts (instance x lever combination).
#   2. Map each config -> an argv for run_asymmetric_learned_vs_gate.py with a
#      unique --output_json under outputs/owmr_ppo_campaign/.
#   3. Run them concurrently with a bounded thread pool (each subprocess pinned to
#      RAYON_NUM_THREADS=OMP_NUM_THREADS=2; gate argmin is cached so no 4-worker
#      gate re-search). Per-config wall-clock timeout.
#   4. Parse each result JSON; aggregate learned_cost, learned_sem, gate_cost,
#      gap_pct_vs_gate, paired_diff, verdict, published_ppo_cost, learned_vs_ppo_pct,
#      deployed_policy, trained_model_params_npy.
#   5. Write campaign_results.tsv (one row per config) and print a ranked summary
#      (best learned_vs_ppo_pct first). Flag any config with learned_vs_ppo_pct > 0
#      as a PPO-BEAT CANDIDATE for adversarial re-verification on disjoint seeds.
#
# USAGE
#   python scripts/one_warehouse_multi_retailer/run_owmr_ppo_campaign.py \
#       --only instance_12            # or instance_13, or both (default)
#       --max_parallel 8 --timeout 2400
# =============================================================================
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PKG_ROOT / "scripts" / "one_warehouse_multi_retailer" / "run_asymmetric_learned_vs_gate.py"
OUT_DIR = PKG_ROOT / "outputs" / "owmr_ppo_campaign"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS = (PKG_ROOT / "outputs" / "one_warehouse_multi_retailer"
          / "asymmetric_learned" / "models")

# Frontier restart checkpoints (verified to reproduce the headline costs).
CKPT_I12 = MODELS / ("asym_kaynov2024_instance_12_echelon_targets_with_alloc_targets_linear"
                     "_d2_axis_aligned_t0.1_absolute_augmented_pop24_gen200_batch16"
                     "_min_shortage_crn_sig0p00125_seed721_527_200") / "model_params.npy"
CKPT_I13 = MODELS / ("asym_kaynov2024_instance_13_echelon_targets_linear_d2_axis_aligned_t0.1"
                     "_pop24_gen200_batch16_proportional_crn_sig0p25_seed735_1692_200") / "model_params.npy"


def cfg(tag, ref, **kw):
    """One campaign config. Defaults mirror the runner's full-budget protocol."""
    d = dict(
        tag=tag, ref=ref,
        action_mode="echelon_targets_with_alloc_targets",
        state_mode="absolute_augmented",
        leaf="linear", depth=2, split="axis_aligned", temp=0.10,
        sigma=0.05, train_ep=600, pop=32, batch=24, holdout=4096,
        train_alloc="min_shortage", same_seed=True, seed=0,
        init=None, warm=True,
    )
    d.update(kw)
    return d


def build_grid(only):
    g = []
    # ---------------------------------------------------------------- inst 12
    if only in ("instance_12", "both"):
        r = "kaynov2024_instance_12"
        I = str(CKPT_I12)
        # (A) anti-overfit refine from frontier: many more training paths
        g += [
            cfg("i12_A1_refine_b48_s0p002_sd811", r, init=I, sigma=0.002, batch=48, train_ep=400, seed=811),
            cfg("i12_A1_refine_b48_s0p002_sd812", r, init=I, sigma=0.002, batch=48, train_ep=400, seed=812),
            cfg("i12_A2_refine_b48_s0p005_sd814", r, init=I, sigma=0.005, batch=48, train_ep=400, seed=814),
            cfg("i12_A3_refine_b64_s0p001_sd816", r, init=I, sigma=0.001, batch=64, train_ep=500, seed=816),
        ]
        # (B) broad multi-start from gate, moderate sigma, long, diverse seeds
        g += [
            cfg("i12_B1_gate_s0p10_sd821", r, warm=True, init=None, sigma=0.10, batch=24, train_ep=800, seed=821),
            cfg("i12_B1_gate_s0p10_sd822", r, warm=True, init=None, sigma=0.10, batch=24, train_ep=800, seed=822),
            cfg("i12_B2_gate_s0p20_sd824", r, warm=True, init=None, sigma=0.20, batch=24, train_ep=800, seed=824),
        ]
        # (C) oblique depth-2 from gate
        g += [
            cfg("i12_C1_oblique_s0p10_sd831", r, split="oblique", sigma=0.10, batch=24, train_ep=800, seed=831),
        ]
        # (D) gentle depth-3 from gate, small sigma
        g += [
            cfg("i12_D1_d3_axis_s0p05_sd841", r, depth=3, sigma=0.05, batch=24, train_ep=800, seed=841),
            cfg("i12_D2_d3_oblique_s0p05_sd843", r, depth=3, split="oblique", sigma=0.05, batch=24, train_ep=800, seed=843),
        ]
    # ---------------------------------------------------------------- inst 13
    if only in ("instance_13", "both"):
        r = "kaynov2024_instance_13"
        I = str(CKPT_I13)
        # (E) deep symmetric tree + augmented state, fresh at gate (low overfit risk)
        g += [
            cfg("i13_E1_sym_d3_axis_s0p10_sd851", r, action_mode="symmetric_echelon_targets",
                depth=3, sigma=0.10, batch=16, train_ep=500, train_alloc="proportional", seed=851),
            cfg("i13_E2_sym_d4_axis_s0p10_sd853", r, action_mode="symmetric_echelon_targets",
                depth=4, sigma=0.10, batch=16, train_ep=500, train_alloc="proportional", seed=853),
            cfg("i13_E3_sym_d3_oblique_s0p10_sd854", r, action_mode="symmetric_echelon_targets",
                depth=3, split="oblique", sigma=0.10, batch=16, train_ep=500, train_alloc="proportional", seed=854),
        ]
        # (F) refine the 84399 echelon_targets frontier (augmented embed), more paths
        g += [
            cfg("i13_F1_refine_aug_s0p05_sd861", r, action_mode="echelon_targets",
                init=I, sigma=0.05, batch=32, train_ep=400, train_alloc="proportional", seed=861),
            cfg("i13_F2_refine_aug_s0p10_sd863", r, action_mode="echelon_targets",
                init=I, sigma=0.10, batch=32, train_ep=400, train_alloc="proportional", seed=863),
        ]
        # (G) decoupled alloc targets from the 84399 frontier. SINGLE embed only
        # (action echelon_targets->with_alloc, state stays normalized) -- the
        # documented/verified embed. A simultaneous action+state embed is untested.
        g += [
            cfg("i13_G1_alloc_norm_s0p05_sd871", r, action_mode="echelon_targets_with_alloc_targets",
                state_mode="normalized", init=I, sigma=0.05, batch=24, train_ep=400,
                train_alloc="min_shortage", seed=871),
        ]
    return g


def argv_for(c):
    out_json = OUT_DIR / f"{c['tag']}.json"
    a = [
        sys.executable, str(RUNNER),
        "--reference", c["ref"],
        "--budget", "full",
        "--policy_action_mode", c["action_mode"],
        "--policy_state_mode", c["state_mode"],
        "--leaf_type", c["leaf"],
        "--depth", str(c["depth"]),
        "--split_type", c["split"],
        "--temperature", str(c["temp"]),
        "--sigma_init", str(c["sigma"]),
        "--gate_search_paths", "64",
        "--training_episodes", str(c["train_ep"]),
        "--es_population", str(c["pop"]),
        "--train_seed_batch", str(c["batch"]),
        "--holdout_paths", str(c["holdout"]),
        "--train_allocation", c["train_alloc"],
        "--seed", str(c["seed"]),
        "--output_json", str(out_json),
    ]
    if c["warm"]:
        a.append("--warm_start_at_best_base_stock")
    if c["same_seed"]:
        a.append("--same_seed")
    if c["init"]:
        a += ["--init_params_npy", c["init"]]
    return a, out_json


def run_one(c, timeout):
    argv, out_json = argv_for(c)
    env = dict(os.environ, RAYON_NUM_THREADS="2", OMP_NUM_THREADS="2")
    t0 = time.time()
    log = OUT_DIR / f"{c['tag']}.log"
    try:
        with open(log, "w") as lf:
            subprocess.run(argv, env=env, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"tag": c["tag"], "status": "timeout", "seconds": time.time() - t0}
    dt = time.time() - t0
    if not out_json.exists():
        return {"tag": c["tag"], "status": "no_json", "seconds": dt}
    d = json.loads(out_json.read_text())
    return {
        "tag": c["tag"], "status": "ok", "seconds": dt,
        "instance": d["instance"], "action_mode": d["policy_action_mode"],
        "state_mode": d["policy_state_mode"], "leaf": d["leaf_type"],
        "depth": d["depth"], "split": d["split_type"], "sigma": d["sigma_init"],
        "batch": d["train_seed_batch"], "train_ep": d["training_episodes"],
        "learned_cost": d["learned_cost"], "learned_sem": d["learned_sem"],
        "gate_cost": d["gate_cost"], "gap_pct_vs_gate": d["gap_pct_vs_gate"],
        "paired_diff": d["paired_diff_mean"], "paired_sem": d["paired_diff_sem"],
        "verdict": d["verdict"], "ppo": d["published_ppo_cost"],
        "learned_vs_ppo_pct": d["learned_vs_ppo_pct"],
        "deployed": d["deployed_policy"],
        "model_npy": d.get("trained_model_params_npy"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["instance_12", "instance_13", "both"], default="both")
    ap.add_argument("--max_parallel", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()

    grid = build_grid(args.only)
    print(f"[campaign] {len(grid)} configs, max_parallel={args.max_parallel}, "
          f"timeout={args.timeout}s", flush=True)
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(run_one, c, args.timeout): c for c in grid}
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            if r["status"] == "ok":
                flag = "  <<< PPO-BEAT CANDIDATE" if (r["learned_vs_ppo_pct"] or -1) > 0 else ""
                print(f"[done {len(results)}/{len(grid)}] {r['tag']}: "
                      f"learned {r['learned_cost']:.2f} vs PPO {r['ppo']:.1f} "
                      f"({r['learned_vs_ppo_pct']:+.3f}%) gate-gap {r['gap_pct_vs_gate']:+.2f}% "
                      f"[{r['verdict']}, deployed={r['deployed']}, {r['seconds']:.0f}s]{flag}",
                      flush=True)
            else:
                print(f"[done {len(results)}/{len(grid)}] {r['tag']}: {r['status']} "
                      f"({r['seconds']:.0f}s)", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    ok.sort(key=lambda r: -(r["learned_vs_ppo_pct"] if r["learned_vs_ppo_pct"] is not None else -1e9))
    tsv = OUT_DIR / f"campaign_results_{args.only}.tsv"
    cols = ["tag", "instance", "action_mode", "state_mode", "leaf", "depth", "split",
            "sigma", "batch", "train_ep", "learned_cost", "learned_sem", "gate_cost",
            "gap_pct_vs_gate", "paired_diff", "paired_sem", "verdict", "ppo",
            "learned_vs_ppo_pct", "deployed", "seconds", "model_npy"]
    with open(tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in ok:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    print("\n===== RANKED BY learned_vs_ppo_pct (positive = BEATS PPO) =====", flush=True)
    for r in ok[:12]:
        print(f"  {r['learned_vs_ppo_pct']:+.3f}%  {r['tag']:42s} "
              f"learned={r['learned_cost']:.2f} ppo={r['ppo']:.1f} "
              f"gate-gap={r['gap_pct_vs_gate']:+.2f}% [{r['verdict']}]", flush=True)
    beats = [r for r in ok if (r["learned_vs_ppo_pct"] or -1) > 0]
    print(f"\n[campaign] PPO-beat candidates: {len(beats)}", flush=True)
    for r in beats:
        print(f"  CANDIDATE {r['tag']} -> {r['model_npy']}", flush=True)
    print(f"[campaign] wrote {tsv}", flush=True)


if __name__ == "__main__":
    main()
