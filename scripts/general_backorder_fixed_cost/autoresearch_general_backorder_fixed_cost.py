"""
Single-policy autoresearch runner for the multi_echelon/general_backorder_fixed_cost
benchmark (Geevers, van Hezewijk & Mes 2024, CardBoard Company general network, set 1).

OBJECTIVE
---------
Produce the first LEARNED-policy result for the Geevers set-1 general network
(geevers2023_general_set1: 4 warehouses + 5 retailers, Poisson(15), unit lead times,
backorders) comparable to the four learned-policy rows already in the paper, and report it
honestly against (a) the constant node-base-stock benchmark it must beat and (b) the
published PPO best it tries to approach/beat.

This trains PURELY in Python against the already-installed `invman_rust`; it never rebuilds
the extension. The action geometry is a Python choice of `build_action_spec` arguments passed
to the rollout binding.

BASELINES (from references.rs, geevers2023_general_set1)
  - published constant node-base-stock benchmark cost = 10,467 (the row the learned policy
    must beat). The repo simulator reproduces ~10,355 (gap -1.1%) with the published levels
    [82,100,64,83,35,35,35,35,35] under random_single_connection_by_weight routing.
  - published PPO best cost = 8,714 (the DRL target to approach/beat).
The keep/discard GATE is the in-repo constant node-base-stock benchmark; the PPO row is
reported alongside for context.

ACTION GEOMETRY (the policy)
----------------------------
The rollout binding's `node_base_stock_targets` action mode interprets the soft tree's
9-dim `vector_quantity` output as the per-node ORDER-UP-TO (base-stock) target levels (4
warehouses + 5 retailers); the env's order-up-to + relative-rationing routing converts those
targets into orders. So:
  - A STATE-INDEPENDENT soft tree (split weights = 0, all leaves equal) emits a CONSTANT
    target vector == exactly a constant node-base-stock policy. Encoding the published levels
    in the leaf parameters makes generation 0 REPRODUCE the published benchmark.
  - A STATE-DEPENDENT soft tree lets the per-node order-up-to levels react to the compact
    inventory-position summary -- a richer policy class than any single constant base-stock
    vector, which is where the learned policy wins.

WARM-START (gen-0 reproduces the heuristic)
  Constant leaf: scaled = min + sigmoid(p)*(max-min)  =>  p_i = logit((L_i-min_i)/(max_i-min_i)).
  Linear   leaf: scaled = min + softplus(bias + w.state); set w=0, bias=softplus^{-1}(L_i-min_i)
                 = ln(exp(L_i-min_i)-1). Split weights/bias start at 0 (=> 50/50 gating, all
                 leaves equal => constant), so the warm vector reproduces the published levels.
  CMA-ES is seeded at this warm vector with a SMALL sigma so it refines AROUND the heuristic
  rather than wandering into the saturated-sigmoid / wild-oblique-split basin (a large sigma
  here diverges to ~30k; sigma ~0.2-0.3 converges to ~8.1k -- documented in program_*.md).

ALGORITHM
---------
1. Read the named reference instance from invman_rust (dims, levels, baselines).
2. Build the warm-start flat-param vector that encodes the published levels in the chosen
   leaf type (gen 0 == constant node-base-stock benchmark).
3. CMA-ES (cma library) over the flat params, warm-started at the encoding with small sigma.
   Fitness = mean rollout cost over a FIXED block of TRAIN base seeds, scored by
   multi_echelon_general_backorder_fixed_cost_soft_tree_population_rollout (the same binding
   used everywhere). Train seeds are fixed across generations (stationary objective) and
   DISJOINT from the held-out block.
4. Re-evaluate the warm vector and the CMA best on a DISJOINT held-out CRN block (paired:
   identical seed list for both), report mean cost +/- SEM, signed gap vs the repo heuristic
   reproduction and vs the published 10,467 / 8,714, and whether it beats / matches / loses.
5. Write outputs/autoresearch/<run_tag>/{results.tsv, <experiment>.json} and a small md note.

USAGE
-----
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/general_backorder_fixed_cost/autoresearch_general_backorder_fixed_cost.py \
      --budget screening --description "warm-start at published set-1 levels, depth-2 constant"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.cpu_limits import configure_process_cpu_limits_from_argv

# HARD CPU CAP -- must be set BEFORE importing numpy/invman_rust. Several sibling
# autoresearch agents run in parallel; default to ~2 cores each.
configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

import numpy as np

import invman_rust as ir

REFERENCE_NAME = "geevers2023_general_set1"
ACTION_MODE = "vector_quantity"
POLICY_FEATURE_MODE = "compact_summary"
POLICY_ACTION_MODE = "node_base_stock_targets"
SPLIT_TYPE = "oblique"

# Action-box caps: physical order-up-to ceilings, set comfortably above the published levels
# (warehouses up to 220 >> max published 100; retailers up to 140 >> 35) so the operating
# region (and the learned policy's higher targets) is interior to the box.
WAREHOUSE_CAP = 220
RETAILER_CAP = 140

BUDGETS = {
    # popsize, generations, n_train_seeds, n_eval_seeds, sigma_init
    "smoke": dict(popsize=8, generations=8, n_train_seeds=4, n_eval_seeds=64, sigma_init=0.25),
    "screening": dict(popsize=16, generations=40, n_train_seeds=8, n_eval_seeds=256, sigma_init=0.25),
    "full": dict(popsize=24, generations=80, n_train_seeds=12, n_eval_seeds=2000, sigma_init=0.20),
}

TRAIN_SEED_BASE = 10_000
TRAIN_SEED_STRIDE = 1_000
EVAL_SEED_BASE = 500_000
EVAL_SEED_STRIDE = 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoresearch loop for the Geevers set-1 general-network learned policy."
    )
    parser.add_argument("--run_tag", default="general_backorder_fixed_cost_autoresearch")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="screening")
    parser.add_argument("--description", required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--leaf_type", choices=["constant", "linear"], default="constant")
    parser.add_argument("--sigma_init", type=float, default=None,
                        help="Override the budget's sigma_init (small sigma keeps CMA near the warm start).")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def git_short_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(PACKAGE_ROOT), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build_action_bounds(num_warehouses: int, num_retailers: int):
    min_values = [0] * (num_warehouses + num_retailers)
    max_values = [WAREHOUSE_CAP] * num_warehouses + [RETAILER_CAP] * num_retailers
    return min_values, max_values


def _logit(frac: float) -> float:
    frac = min(max(frac, 1e-4), 1.0 - 1e-4)
    return math.log(frac / (1.0 - frac))


def _softplus_inverse(y: float) -> float:
    # numerically stable ln(exp(y)-1) for y>0
    y = max(y, 1e-4)
    return y + math.log(-math.expm1(-y))


def warm_start_flat_params(levels, min_values, max_values, depth: int, leaf_type: str):
    """Flat params encoding a STATE-INDEPENDENT constant policy at `levels`.

    Split weights/bias = 0 (50/50 gating, all leaves identical) => the soft tree emits a
    constant target vector == constant node-base-stock at `levels`, so generation 0 == the
    published benchmark policy.
    """
    action_dim = len(levels)
    num_internal = (1 << depth) - 1
    num_leaves = 1 << depth
    input_dim = len(levels) + 5  # compact_summary feature dim (nW + nR + 5)
    split_weights = [0.0] * (num_internal * input_dim)
    split_bias = [0.0] * num_internal
    leaf_params: list[float] = []
    if leaf_type == "constant":
        per_leaf = [_logit((levels[i] - min_values[i]) / (max_values[i] - min_values[i]))
                    for i in range(action_dim)]
        for _ in range(num_leaves):
            leaf_params.extend(per_leaf)
    else:  # linear: weights then biases (layout: all leaf*action*input weights, then leaf*action biases)
        leaf_weights = [0.0] * (num_leaves * action_dim * input_dim)
        leaf_biases: list[float] = []
        per_leaf_bias = [_softplus_inverse(levels[i] - min_values[i]) for i in range(action_dim)]
        for _ in range(num_leaves):
            leaf_biases.extend(per_leaf_bias)
        leaf_params = leaf_weights + leaf_biases
    return split_weights + split_bias + leaf_params, input_dim


def make_seed_block(base: int, stride: int, count: int) -> list[int]:
    return [base + k * stride for k in range(count)]


def population_costs(population, input_dim, depth, min_values, max_values, leaf_type,
                     temperature, seed_block) -> np.ndarray:
    """Mean cost per population member over the seed block, via the Rust population rollout.

    For each base seed the binding evaluates member i on seed (base + i). Base seeds are
    spaced by a wide stride so different members never collide on the same path, and the
    block is FIXED across generations -> a stationary CMA objective. Averaged over the block
    this is an unbiased common-random-number estimate.
    """
    pop = [[float(x) for x in member] for member in population]
    accum = np.zeros(len(pop), dtype=np.float64)
    for base_seed in seed_block:
        costs = ir.multi_echelon_general_backorder_fixed_cost_soft_tree_population_rollout(
            pop, input_dim, depth, min_values, max_values, ACTION_MODE, REFERENCE_NAME,
            int(base_seed), float(temperature), SPLIT_TYPE, leaf_type, None,
            POLICY_FEATURE_MODE, POLICY_ACTION_MODE,
        )
        accum += np.asarray(costs, dtype=np.float64)
    return accum / len(seed_block)


def paired_eval(flat_params, input_dim, depth, min_values, max_values, leaf_type,
                temperature, seed_block):
    """Mean cost +/- SEM for ONE policy, one rollout per seed (idx 0 => no seed offset)."""
    member = [float(x) for x in flat_params]
    costs = []
    for base_seed in seed_block:
        c = ir.multi_echelon_general_backorder_fixed_cost_soft_tree_population_rollout(
            [member], input_dim, depth, min_values, max_values, ACTION_MODE, REFERENCE_NAME,
            int(base_seed), float(temperature), SPLIT_TYPE, leaf_type, None,
            POLICY_FEATURE_MODE, POLICY_ACTION_MODE,
        )
        costs.append(float(c[0]))
    mean = float(np.mean(costs))
    sem = float(np.std(costs, ddof=1) / math.sqrt(len(costs))) if len(costs) > 1 else 0.0
    return mean, sem


def main() -> None:
    parsed = parse_args()
    budget = BUDGETS[parsed.budget]
    sigma_init = parsed.sigma_init if parsed.sigma_init is not None else budget["sigma_init"]

    ref = dict(ir.multi_echelon_general_backorder_fixed_cost_get_reference_instance(REFERENCE_NAME))
    num_warehouses = ref["num_warehouses"]
    num_retailers = ref["num_retailers"]
    levels = list(ref["benchmark_base_stock_levels"])
    published_benchmark = float(ref["published_benchmark_cost"])
    published_ppo_best = float(ref["published_ppo_best_cost"])

    min_values, max_values = build_action_bounds(num_warehouses, num_retailers)
    depth = parsed.depth
    leaf_type = parsed.leaf_type
    temperature = parsed.temperature

    x0, input_dim = warm_start_flat_params(levels, min_values, max_values, depth, leaf_type)

    train_seeds = make_seed_block(TRAIN_SEED_BASE, TRAIN_SEED_STRIDE, budget["n_train_seeds"])
    eval_seeds = make_seed_block(EVAL_SEED_BASE, EVAL_SEED_STRIDE, budget["n_eval_seeds"])
    assert not (set(train_seeds) & set(eval_seeds)), "train/eval seed blocks must be disjoint"

    print(f"[setup] reference={REFERENCE_NAME} nodes={num_warehouses + num_retailers} "
          f"input_dim={input_dim} param_dim={len(x0)} depth={depth} leaf={leaf_type} "
          f"sigma={sigma_init} budget={parsed.budget}")

    # Repo heuristic reproduction baseline (the gate) at full reps for context.
    heur_means = []
    for s in (1234, 5678, 9012):
        d = ir.multi_echelon_general_backorder_fixed_cost_simulate_base_stock(
            REFERENCE_NAME, None, int(ref["benchmark_replications"]), s, None)
        heur_means.append(float(d["mean_cost"]))
    repo_heuristic_cost = float(np.mean(heur_means))

    # Warm-start held-out cost (should track the heuristic reproduction).
    warm_mean, warm_sem = paired_eval(x0, input_dim, depth, min_values, max_values,
                                      leaf_type, temperature, eval_seeds)
    print(f"[gen0] warm-start held-out mean={warm_mean:.1f} +/- {warm_sem:.1f} "
          f"(repo heuristic {repo_heuristic_cost:.1f}, published {published_benchmark:.0f})")

    import cma
    es = cma.CMAEvolutionStrategy(
        [float(x) for x in x0], float(sigma_init),
        {"popsize": budget["popsize"], "seed": parsed.seed,
         "maxiter": budget["generations"], "verbose": -9},
    )
    start = time.time()
    gen = 0
    best_train = float("inf")
    while not es.stop():
        solutions = es.ask()
        fitness = population_costs(solutions, input_dim, depth, min_values, max_values,
                                  leaf_type, temperature, train_seeds)
        es.tell(solutions, fitness.tolist())
        gen += 1
        best_train = min(best_train, float(np.min(fitness)))
        if gen % 10 == 0 or gen == budget["generations"]:
            print(f"[gen {gen:3d}] best train cost {float(np.min(fitness)):.1f}")
    wall = time.time() - start

    best = es.result.xbest
    learned_mean, learned_sem = paired_eval(best, input_dim, depth, min_values, max_values,
                                            leaf_type, temperature, eval_seeds)

    gap_vs_heuristic = learned_mean - repo_heuristic_cost
    gap_pct_vs_heuristic = 100.0 * (learned_mean / repo_heuristic_cost - 1.0)
    gap_vs_published_benchmark = learned_mean - published_benchmark
    gap_vs_ppo_best = learned_mean - published_ppo_best

    if learned_mean < repo_heuristic_cost - 2.0 * (learned_sem + warm_sem):
        verdict = "beats"
    elif learned_mean > repo_heuristic_cost + 2.0 * (learned_sem + warm_sem):
        verdict = "loses"
    else:
        verdict = "matches"

    print(f"[result] learned held-out mean={learned_mean:.1f} +/- {learned_sem:.1f}  "
          f"({verdict} repo heuristic {repo_heuristic_cost:.1f}; "
          f"published benchmark {published_benchmark:.0f}; PPO best {published_ppo_best:.0f})")
    print(f"[result] gap vs heuristic {gap_vs_heuristic:+.1f} ({gap_pct_vs_heuristic:+.2f}%); "
          f"vs published benchmark {gap_vs_published_benchmark:+.1f}; vs PPO best {gap_vs_ppo_best:+.1f}; "
          f"wall {wall:.0f}s")

    commit = git_short_commit()
    experiment = (f"{parsed.run_tag}_{parsed.budget}_d{depth}_{leaf_type}"
                  f"_sig{sigma_init}_s{parsed.seed}")
    out_root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    out_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "commit": commit,
        "experiment": experiment,
        "reference": REFERENCE_NAME,
        "budget": parsed.budget,
        "structure": {"depth": depth, "leaf_type": leaf_type, "split_type": SPLIT_TYPE,
                      "temperature": temperature, "action_mode": ACTION_MODE,
                      "policy_action_mode": POLICY_ACTION_MODE,
                      "policy_feature_mode": POLICY_FEATURE_MODE,
                      "input_dim": input_dim, "param_dim": len(x0),
                      "min_values": min_values, "max_values": max_values},
        "cma": {"popsize": budget["popsize"], "generations": gen, "sigma_init": sigma_init,
                "seed": parsed.seed, "n_train_seeds": len(train_seeds),
                "n_eval_seeds": len(eval_seeds), "wall_seconds": wall,
                "best_train_cost": best_train},
        "baselines": {"published_benchmark_cost": published_benchmark,
                      "published_ppo_best_cost": published_ppo_best,
                      "repo_heuristic_reproduction_cost": repo_heuristic_cost,
                      "published_levels": levels},
        "warm_start_heldout": {"mean": warm_mean, "sem": warm_sem},
        "learned_heldout": {"mean": learned_mean, "sem": learned_sem},
        "gaps": {"vs_repo_heuristic": gap_vs_heuristic,
                 "vs_repo_heuristic_pct": gap_pct_vs_heuristic,
                 "vs_published_benchmark": gap_vs_published_benchmark,
                 "vs_published_ppo_best": gap_vs_ppo_best},
        "verdict_vs_repo_heuristic": verdict,
        "best_params": [float(x) for x in best],
        "description": parsed.description,
    }
    json_path = out_root / f"{experiment}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    results_tsv = out_root / "results.tsv"
    header = ["commit", "experiment", "reference", "budget", "depth", "leaf_type",
              "sigma_init", "warm_heldout", "learned_heldout", "learned_sem",
              "repo_heuristic", "published_benchmark", "published_ppo_best",
              "gap_vs_heuristic", "gap_pct_vs_heuristic", "gap_vs_ppo_best",
              "verdict", "description"]
    write_header = not results_tsv.exists()
    with results_tsv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if write_header:
            writer.writerow(header)
        writer.writerow([
            commit, experiment, REFERENCE_NAME, parsed.budget, depth, leaf_type,
            f"{sigma_init}", f"{warm_mean:.1f}", f"{learned_mean:.1f}", f"{learned_sem:.1f}",
            f"{repo_heuristic_cost:.1f}", f"{published_benchmark:.0f}", f"{published_ppo_best:.0f}",
            f"{gap_vs_heuristic:.1f}", f"{gap_pct_vs_heuristic:.2f}", f"{gap_vs_ppo_best:.1f}",
            verdict, parsed.description,
        ])

    print(json.dumps({
        "json": str(json_path),
        "results_tsv": str(results_tsv),
        "learned_heldout_mean": learned_mean,
        "repo_heuristic_cost": repo_heuristic_cost,
        "published_benchmark": published_benchmark,
        "published_ppo_best": published_ppo_best,
        "gap_pct_vs_heuristic": gap_pct_vs_heuristic,
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
