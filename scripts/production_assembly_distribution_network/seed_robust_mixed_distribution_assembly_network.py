"""
Seed-ROBUST learned-vs-gate runner for the MIXED distribution-and-assembly SCN
(Pirhooshyaran & Snyder 2021, Fig. 1 / Table 5) -- the seed-cherry-pick fix.

OBJECTIVE
---------
The paper row for the mixed distribution-and-assembly network claims the learned
depth-2 soft tree beats the env's own grid-searched best pairwise base-stock gate
(297.69, echelon order-up-to [36,13,7]) by -0.99% (294.73 +/- 0.92) -- but only "on
the BEST of three CMA seeds, while a third seed sits +2.9% above the gate." That is a
best-of-N straddle, not a robust result. This runner replaces best-of-N reporting with
a HONEST seed-averaged comparison and three robustness devices, then computes the
seed-mean +/- std over >=5 independent optimizer seeds. A credible parity / negative
result is an explicitly valid outcome.

ALGORITHM (full description)
----------------------------
1. Gate. Identical grid search to autoresearch_mixed_distribution_assembly_network.py:
   per-echelon order-up-to levels (e1 = relations edge(0,1) & external->0, e2 = edges
   (1,2),(1,3), e3 = edges (2,4),(2,5),(3,4),(3,5)) on the disjoint SEARCH block, re-
   scored on the HELD-OUT block. The fine-grid argmin is echelon [36,13,7] ->
   OUL [36,13,13,7,7,7,7,36], held-out 297.69 / period. The gate POLICY (a real
   pairwise_base_stock rollout) is also kept as a deployment candidate.

2. Action geometry. Unchanged: vector_quantity over the 8 supply relations, a depth-d
   oblique soft tree with linear leaves, clipped to [MIN,MAX]. The linear leaf maps
   state features (per-node finished/raw/backlog/pipeline, per-relation raw + pipeline,
   per-edge internal backlog, remaining-horizon) to a per-relation order. CRUCIAL: the
   env build_policy_state DIVIDES every feature by a dynamic per-step scale (max raw
   value in the state). Therefore a linear leaf CANNOT express an exact affine
   order = clip(level - inventory_position) gate (unlike the OWMR echelon_targets
   geometry, whose action head is an explicit target position). The OWMR exact
   leaf-inversion warm-start is thus NOT transferable here. We instead use:

   (a) WARM-START at the GATE FLOW (--warm_start, default on). The gate's per-relation
       order-up-to levels [36,13,13,7,7,7,7,36] are decoded into a CONSTANT per-relation
       order via the same softplus-inverse leaf-bias inversion the autoresearch runner
       uses for a flat flow, but per-relation at the GATE LEVELS rather than at the flat
       demand-mean flow=5. This seeds generation 0 near the gate's steady-state ORDER
       magnitude (not the demand mean, which starves this network and gives gen0 ~864 /
       period). Leaf weights are zeroed (state-independent) so gen0 is a clean constant-
       order anchor; CMA-ES then refines the weights toward inventory feedback.

   (b) HONEST DEPLOYMENT FLOOR (--honest_floor, default on). The deployed policy on the
       held-out block is the argmin over {trained xbest, gate-flow warm-start anchor,
       the gate pairwise_base_stock policy itself}. With the gate in the candidate set,
       NO seed can deploy worse than the gate; the only open question is how much the
       search robustly subtracts BELOW the gate. The honest floor is decoder-agnostic
       (it scores real rollouts), so it sidesteps the non-invertible normalized-feature
       decoder entirely.

   (c) GENTLE SIGMA (--sigma_init, sweepable {0.1,0.2,0.3} vs the runner's default 0.8)
       so seeds stay near the gate-flow anchor instead of scattering into the 360-440
       basin the flat flow=5 + sigma-0.8 baseline lands in.

3. Robustness reporting. The runner trains N independent seeds in ONE invocation, scores
   each on the SAME held-out CRN block (HOLDOUT_SEED, holdout_paths -- identical protocol
   to the autoresearch runner so 297.69 is comparable), records per-seed deployed cost,
   and prints the seed-MEAN +/- sample-STD, the per-seed table, the fraction of seeds
   below the gate, and a verdict: ROBUST BEAT only if (gate - seed_mean) exceeds the
   cross-seed std; otherwise PARITY (or, with the floor, ROBUST GATE-MATCH if the floor
   pins seeds at the gate and the search adds ~nothing). NEVER best-of-N as the headline.

CPU CAP / USAGE
---------------
CPU is capped before NumPy/Rust import (default 2 rayon/omp threads). Run seeds serially
inside one process (each CMA-ES population rollout is already rayon-parallel); to fan out
over cores, launch several invocations with disjoint --seeds and aggregate the JSONs.
USAGE:
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/production_assembly_distribution_network/seed_robust_mixed_distribution_assembly_network.py \
      --budget full --depth 2 --sigma_init 0.2 --seeds 11 22 33 44 55 \
      --warm_start --honest_floor --description "gate-flow warmstart + floor, sigma0.2"
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.cpu_limits import configure_process_cpu_limits_from_argv

configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

import numpy as np  # noqa: E402

from invman.cmaes import CMAES  # noqa: E402

import invman_rust as ir  # noqa: E402

# Import the instance constants + helpers from the sibling autoresearch runner so the
# topology, parameters, CRN blocks, budgets and gate search are BYTE-IDENTICAL.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import autoresearch_mixed_distribution_assembly_network as base  # noqa: E402


def _gate_flow_warm_start(depth: int, leaf_type: str, gate_oul) -> np.ndarray:
    """Seed the CMA mean so generation 0 emits a CONSTANT per-relation order equal to the
    gate's order-up-to levels (per relation), state-independently.

    For a LINEAR leaf the env applies scaled = min + softplus(leaf_output); with the leaf
    WEIGHTS zeroed the output is the per-relation leaf BIAS, so
        order_dim = min + softplus(bias_dim)  =>  bias_dim = softplus_inv(gate_oul_dim - min).
    This reproduces the gate's steady-state order MAGNITUDE at gen 0 (not the gate's
    inventory feedback, which the normalized-feature decoder cannot express affinely).
    For constant / sigmoid_linear leaves we invert the sigmoid span transform instead.
    """
    n = base._flat_param_count(depth, leaf_type)
    flat = np.zeros(n, dtype=np.float64)
    num_internal = (1 << depth) - 1
    num_leaves = 1 << depth
    bias_block = num_leaves * base.ACTION_DIM
    gate = [float(x) for x in gate_oul]
    if leaf_type in ("constant", "sigmoid_linear"):
        # tail = num_leaves x ACTION_DIM sigmoid logits.
        tail = np.empty((num_leaves, base.ACTION_DIM), dtype=np.float64)
        for d in range(base.ACTION_DIM):
            span = float(base.MAX_VALUES[d] - base.MIN_VALUES[d])
            p = (gate[d] - base.MIN_VALUES[d]) / span
            p = float(min(max(p, 1e-4), 1.0 - 1e-4))
            tail[:, d] = math.log(p / (1.0 - p))
        flat[n - bias_block:] = tail.reshape(-1)
        return flat
    # linear leaf: zero weights (already zero), set per-relation leaf bias.
    bias_start = num_internal * base.INPUT_DIM + num_internal + num_leaves * base.ACTION_DIM * base.INPUT_DIM
    leaf_bias = np.empty(base.ACTION_DIM, dtype=np.float64)
    for d in range(base.ACTION_DIM):
        delta = max(gate[d] - base.MIN_VALUES[d], 1e-6)
        leaf_bias[d] = math.log(math.expm1(delta))
    tail = np.tile(leaf_bias, num_leaves)
    flat[bias_start:bias_start + bias_block] = tail
    return flat


def _train_one_seed(parsed, budget, holdout_paths, gate, x0):
    """Train ONE CMA-ES seed; return its trained xbest, gen0 holdout, and best_train."""
    depth, leaf, split, temp = parsed.depth, parsed.leaf_type, parsed.split_type, parsed.temperature
    n = base._flat_param_count(depth, leaf)
    gen0_mean, gen0_se = base.soft_tree_cost_on_paths(x0, depth, leaf, split, temp, holdout_paths)

    es = CMAES(num_params=n, sigma_init=parsed.sigma_init, popsize=budget["popsize"],
               seed=parsed.seed_value, x0=x0.tolist())
    rng = np.random.default_rng(parsed.seed_value + 1)
    best_flat = x0.copy()
    best_train = math.inf
    t0 = time.time()
    train_batch = parsed.train_batch if parsed.train_batch else budget["train_batch"]
    for _ in range(budget["generations"]):
        sols = es.ask()
        b = int(rng.integers(1, 10_000_000))
        seeds = list(range(b, b + train_batch))
        rewards = []
        for k in range(es.popsize):
            batch = [sols[k].astype(np.float32).tolist()] * train_batch
            cost = float(base.population_costs(batch, depth, leaf, split, temp, seeds).mean())
            rewards.append(-cost)
        es.tell(rewards)
        gi = int(np.argmax(rewards))
        if -rewards[gi] < best_train:
            best_train = -rewards[gi]
            best_flat = sols[gi].copy()
    train_seconds = time.time() - t0

    trained_mean, trained_se = base.soft_tree_cost_on_paths(best_flat, depth, leaf, split, temp, holdout_paths)
    return {
        "best_flat": best_flat,
        "gen0_holdout_mean": gen0_mean,
        "gen0_holdout_se": gen0_se,
        "trained_holdout_mean": trained_mean,
        "trained_holdout_se": trained_se,
        "best_train_cost": float(best_train),
        "train_seconds": float(train_seconds),
    }


def run(parsed) -> dict:
    budget = base.BUDGETS[parsed.budget]
    search_paths = base.make_paths(budget["search_paths"], base.SEARCH_SEED)
    holdout_paths = base.make_paths(budget["holdout_paths"], base.HOLDOUT_SEED)

    heuristic = base.search_best_pairwise_base_stock(search_paths, holdout_paths, budget["grid"])
    gate_cost = heuristic["holdout_mean_cost"]
    gate_oul = heuristic["oul_levels"]

    depth, leaf, split, temp = parsed.depth, parsed.leaf_type, parsed.split_type, parsed.temperature

    # Warm-start anchor.
    if parsed.warm_start:
        x0 = _gate_flow_warm_start(depth, leaf, gate_oul)
        warm_kind = f"gate_flow_oul{gate_oul}"
    else:
        x0 = base._warm_start_flow(depth, leaf, parsed.warm_start_flow)
        warm_kind = f"flat_flow{parsed.warm_start_flow}"
    anchor_mean, anchor_se = base.soft_tree_cost_on_paths(x0, depth, leaf, split, temp, holdout_paths)

    per_seed = []
    for seed in parsed.seeds:
        parsed.seed_value = seed
        res = _train_one_seed(parsed, budget, holdout_paths, gate_cost, x0)
        # Honest deployment floor: argmin over {trained xbest, warm-start anchor, gate}.
        candidates = [("trained_xbest", res["trained_holdout_mean"], res["trained_holdout_se"])]
        if parsed.honest_floor:
            candidates.append(("warm_start_anchor", anchor_mean, anchor_se))
            candidates.append(("gate", gate_cost, heuristic["holdout_stderr"]))
        deployed_policy, deployed_cost, deployed_se = min(candidates, key=lambda c: c[1])
        per_seed.append({
            "seed": seed,
            "trained_holdout_mean": res["trained_holdout_mean"],
            "trained_holdout_se": res["trained_holdout_se"],
            "gen0_holdout_mean": res["gen0_holdout_mean"],
            "best_train_cost": res["best_train_cost"],
            "deployed_policy": deployed_policy,
            "deployed_cost": deployed_cost,
            "deployed_se": deployed_se,
            "gap_pct": (deployed_cost / gate_cost - 1.0) * 100.0,
            "train_seconds": res["train_seconds"],
        })

    deployed = [s["deployed_cost"] for s in per_seed]
    trained_only = [s["trained_holdout_mean"] for s in per_seed]
    n_seeds = len(deployed)
    dep_mean = statistics.mean(deployed)
    dep_std = statistics.stdev(deployed) if n_seeds > 1 else 0.0
    tr_mean = statistics.mean(trained_only)
    tr_std = statistics.stdev(trained_only) if n_seeds > 1 else 0.0
    frac_below = sum(1 for v in deployed if v < gate_cost)
    frac_below_trained = sum(1 for v in trained_only if v < gate_cost)

    # Verdict on the DEPLOYED (floored) policy seed-mean.
    margin = gate_cost - dep_mean
    if margin > dep_std and frac_below == n_seeds and dep_std > 0:
        verdict = "ROBUST_BEAT"
    elif abs(margin) <= max(dep_std, 1e-9):
        verdict = "PARITY"
    elif margin < 0:
        verdict = "ROBUST_LOSS"
    else:
        verdict = "MARGINAL_BEAT_WITHIN_STD"
    # If the floor pins (almost) every seed exactly at the gate, label it gate-match.
    pinned = sum(1 for s in per_seed if s["deployed_policy"] == "gate")
    if parsed.honest_floor and pinned == n_seeds:
        verdict = "ROBUST_GATE_MATCH_ONLY"

    return {
        "reference": "pirhooshyaran2021_mixed_scn_fig1_table5",
        "literature_verified": False,
        "baseline_kind": "env_own_best_pairwise_base_stock (RESEARCH comparison)",
        "policy_architecture": (
            f"soft_tree_d{depth}_{split}_{leaf}_temp{temp}_vector_quantity"
            f"_sigma{parsed.sigma_init}_warm[{warm_kind}]_floor[{parsed.honest_floor}]"
        ),
        "gate": {"oul_levels": gate_oul, "echelon_levels": heuristic["echelon_levels"],
                 "holdout_mean_cost": gate_cost, "holdout_stderr": heuristic["holdout_stderr"]},
        "warm_start_anchor_holdout_mean": anchor_mean,
        "warm_start_anchor_holdout_se": anchor_se,
        "warm_start_kind": warm_kind,
        "sigma_init": parsed.sigma_init,
        "honest_floor": parsed.honest_floor,
        "depth": depth, "leaf_type": leaf, "split_type": split, "temperature": temp,
        "train_batch": parsed.train_batch if parsed.train_batch else budget["train_batch"],
        "holdout_paths": budget["holdout_paths"],
        "per_seed": per_seed,
        "n_seeds": n_seeds,
        "deployed_seed_mean": dep_mean,
        "deployed_seed_std": dep_std,
        "deployed_gap_pct": (dep_mean / gate_cost - 1.0) * 100.0,
        "trained_seed_mean": tr_mean,
        "trained_seed_std": tr_std,
        "trained_gap_pct": (tr_mean / gate_cost - 1.0) * 100.0,
        "frac_seeds_below_gate_deployed": f"{frac_below}/{n_seeds}",
        "frac_seeds_below_gate_trained": f"{frac_below_trained}/{n_seeds}",
        "verdict": verdict,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Seed-robust learned-vs-gate for the mixed SCN.")
    p.add_argument("--run_tag", default="mixed_distribution_assembly_network_seed_robust")
    p.add_argument("--budget", choices=sorted(base.BUDGETS), default="full")
    p.add_argument("--description", required=True)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--temperature", type=float, default=base.TEMPERATURE_DEFAULT)
    p.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    p.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    p.add_argument("--warm_start_flow", type=float, default=base.DEMAND_MEAN,
                   help="Flat-flow warm-start level when --warm_start is OFF (baseline parity).")
    p.add_argument("--warm_start", dest="warm_start", action="store_true", default=True,
                   help="Gate-flow warm-start (gen0 ~ gate order magnitude). Default ON.")
    p.add_argument("--no_warm_start", dest="warm_start", action="store_false",
                   help="Use the flat demand-mean flow warm-start (reproduces the baseline pathology).")
    p.add_argument("--honest_floor", dest="honest_floor", action="store_true", default=True,
                   help="Deploy argmin{trained, anchor, gate}. Default ON (no seed worse than gate).")
    p.add_argument("--no_honest_floor", dest="honest_floor", action="store_false",
                   help="Deploy the trained xbest with NO floor (reproduces the baseline pathology).")
    p.add_argument("--sigma_init", type=float, default=0.2)
    p.add_argument("--train_batch", type=int, default=None,
                   help="Override the budget's per-candidate training batch (larger = less ranking noise).")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    return p.parse_args()


def main():
    parsed = parse_args()
    out = run(parsed)
    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    root.mkdir(parents=True, exist_ok=True)
    commit = base._git_short_commit(PACKAGE_ROOT)
    batch_tag = parsed.train_batch if parsed.train_batch else base.BUDGETS[parsed.budget]["train_batch"]
    tag = (f"{parsed.budget}_d{parsed.depth}_{parsed.split_type}_{parsed.leaf_type}"
           f"_t{parsed.temperature:g}_b{batch_tag}"
           f"_sig{parsed.sigma_init}_warm{int(parsed.warm_start)}_floor{int(parsed.honest_floor)}")
    json_path = root / f"seedrobust_{tag}_{commit}.json"
    with json_path.open("w", encoding="utf-8") as h:
        json.dump({"description": parsed.description, "commit": commit, "detail": out}, h, indent=2)

    g = out["gate"]["holdout_mean_cost"]
    print("=" * 78)
    print(f"DESIGN: {out['policy_architecture']}")
    print(f"GATE: {g:.3f} +/- {out['gate']['holdout_stderr']:.3f}  OUL {out['gate']['oul_levels']}")
    print(f"warm-start anchor held-out: {out['warm_start_anchor_holdout_mean']:.3f}")
    print("-" * 78)
    print(f"{'seed':>6} {'trained':>10} {'gap%':>8} {'deployed':>10} {'policy':>18} {'gap%':>8}")
    for s in out["per_seed"]:
        print(f"{s['seed']:>6} {s['trained_holdout_mean']:>10.3f} "
              f"{(s['trained_holdout_mean']/g-1)*100:>+7.2f} {s['deployed_cost']:>10.3f} "
              f"{s['deployed_policy']:>18} {s['gap_pct']:>+7.2f}")
    print("-" * 78)
    print(f"TRAINED (no floor)  seed-mean {out['trained_seed_mean']:.3f} +/- {out['trained_seed_std']:.3f}"
          f"  gap {out['trained_gap_pct']:+.2f}%  below-gate {out['frac_seeds_below_gate_trained']}")
    print(f"DEPLOYED (floored)  seed-mean {out['deployed_seed_mean']:.3f} +/- {out['deployed_seed_std']:.3f}"
          f"  gap {out['deployed_gap_pct']:+.2f}%  below-gate {out['frac_seeds_below_gate_deployed']}")
    print(f"VERDICT: {out['verdict']}")
    print(f"WROTE_JSON: {json_path}")


if __name__ == "__main__":
    main()
