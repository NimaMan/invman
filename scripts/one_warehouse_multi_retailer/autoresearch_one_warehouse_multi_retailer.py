"""
Single-policy autoresearch runner for the one_warehouse_multi_retailer benchmark
(Kaynov et al. 2024, IJPE 267, 109088).

OBJECTIVE
---------
Train ONE soft-tree CMA-ES policy with a CLI-selected structure on a NAMED Kaynov
instance, evaluate its held-out (CRN) cost and the optimality gap vs the strongest
heuristic (grid-searched echelon base-stock under the better of {min_shortage,
proportional} allocation), and APPEND a TSV ledger row. This mirrors the repo's
dual_sourcing / multi_echelon autoresearch runners, but reuses the helpers built in
the learned-benchmark phase for THIS problem so the env, the heuristic search, and the
paired held-out evaluation are bit-identical to benchmark_learned_vs_heuristic.py.

WHY (carried from program_one_warehouse_multi_retailer.md)
----------------------------------------------------------
The learned depth-2 soft-tree LOSES to the tuned echelon base-stock + allocation
heuristic by 0.4%-1.7% on the three currently-losing instances:
  kaynov2024_instance_1  (backorder)         -1.69%
  kaynov2024_instance_6  (lost_sales)         -1.67%
  kaynov2024_instance_11 (partial_backorder)  -0.43%   <- default (closest to flipping)
The job is to search the policy/control surface to flip the sign.

SEARCH SURFACE (CLI)
--------------------
  --depth / --temperature / --split_type {oblique,axis_aligned} / --leaf_type {constant,linear}
  --policy_action_mode {symmetric_echelon_targets, direct_orders}   (action design)
  --train_allocation / heuristic best-of {proportional, min_shortage}   (allocation policy)
  --warm_start_at_best_base_stock     (seed CMA-ES x0 leaf bias at the best base-stock (W,R))

ALGORITHM (per run)
-------------------
1. Resolve the named Kaynov reference (common.get_reference).
2. Build disjoint CRN demand-path blocks (search vs held-out) with the shared sampler.
3. Strongest heuristic: grid-search echelon base-stock (W, shared R) on the search block
   for BOTH {proportional, min_shortage}, re-score the argmin on the held-out block, take
   the better allocation as the best heuristic.
4. Build the soft-tree (common.build_soft_tree_model) with the CLI structure. Optionally
   warm-start CMA-ES at the best base-stock levels: set the trailing leaf bias/constant
   block to the (W, R) target so generation-0 reproduces the strongest heuristic, and pass
   those flat params as cma_x0 so the search starts from a known-good point.
5. Train with invman.es_mp.train via the population-rollout binding (rayon-bounded; no
   Python process pool). Train allocation = --train_allocation.
6. Evaluate the trained weights on the SAME held-out block (CRN-paired) under both
   allocations; the better allocation is `trained_cost`.
6b. HONEST WARM-START FLOOR: train() returns CMA-ES `xbest`, which is the best on TRAINING
   seeds and can over-fit relative to the held-out block. When warm-started, also evaluate
   the warm-start gen-0 anchor (which reproduces the strongest heuristic exactly) on the SAME
   paired CRN block and DEPLOY the better of {trained xbest, anchor} (`deployed_policy`). The
   headline `learned_cost` therefore can never be reported worse than the heuristic-reproducing
   anchor it started from -- a spurious "loss" that is purely training-seed overfit is rejected.
7. Compute gap and gap% vs the best heuristic; append a TSV ledger row.

CPU CAP (HARD)
--------------
The shared CPU helper caps Rayon/BLAS/OpenMP before NumPy and Rust imports; mp_num_processors
is pinned to 1 (parallelism is rayon inside the population-rollout binding). The repo defaults
to ~27 cores elsewhere; this runner MUST stay capped (two sibling agents run in parallel).

USAGE (smoke)
-------------
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/one_warehouse_multi_retailer/autoresearch_one_warehouse_multi_retailer.py \
      --description "smoke" --budget screening \
      --reference kaynov2024_instance_11 --warm_start_at_best_base_stock
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
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from invman.cpu_limits import configure_process_cpu_limits_from_argv  # noqa: E402

configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

import numpy as np

from invman.es_mp import train  # noqa: E402

import common  # noqa: E402
from benchmark_learned_vs_heuristic import (  # noqa: E402  (reuse learned-benchmark helpers)
    DISCOUNT_FACTOR,
    _get_model_fitness,
    _get_population_fitness,
    _heuristic_on_paths,
    _sample_demand_paths,
    _search_best_heuristic_on_paths,
    _soft_tree_on_paths,
)

# smoke = end-to-end validation only (not a decision budget);
# screening = cheap lever-ranking pass; full = promotion budget matching the learned benchmark.
BUDGETS = {
    "smoke": {
        "training_episodes": 8,
        "es_population": 8,
        "train_seed_batch": 2,
        "search_paths": 48,
        "holdout_paths": 128,
    },
    "screening": {
        "training_episodes": 60,
        "es_population": 16,
        "train_seed_batch": 4,
        "search_paths": 96,
        "holdout_paths": 512,
    },
    "full": {
        "training_episodes": 600,
        "es_population": 32,
        "train_seed_batch": 12,
        "search_paths": 256,
        "holdout_paths": 4096,
    },
}

# Disjoint CRN seed blocks + fixed allocation-RNG anchors (same convention as the benchmark).
SEARCH_SEED_START = 500_000
HOLDOUT_SEED_START = 900_000
ALLOC_SEED_SEARCH = 700_000
ALLOC_SEED_HOLDOUT = 800_000

# Documented symmetric Poisson(3) targets (one per regime + the long-Lw lost-sales row).
# All TIE the strongest in-repo heuristic at full budget (base-stock is near-optimal here):
#   1 backorder, 6 lost_sales (Lw=1), 11 partial_backorder, 7 lost_sales (Lw=2).
DOCUMENTED_INSTANCES = {
    "kaynov2024_instance_1": "backorder",
    "kaynov2024_instance_6": "lost_sales",
    "kaynov2024_instance_7": "lost_sales",
    "kaynov2024_instance_11": "partial_backorder",
}


def parse_args():
    p = argparse.ArgumentParser(description="Autoresearch-style single-policy loop for one_warehouse_multi_retailer.")
    p.add_argument("--run_tag", default="one_warehouse_multi_retailer_autoresearch")
    p.add_argument("--budget", choices=sorted(BUDGETS), default="screening")
    p.add_argument("--description", required=True)
    p.add_argument("--reference", default="kaynov2024_instance_11")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="axis_aligned")
    p.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    p.add_argument(
        "--policy_action_mode",
        choices=["symmetric_echelon_targets", "direct_orders", "vector_quantity"],
        default=None,
        help="Action design. Default: symmetric_echelon_targets for symmetric instances, else direct_orders.",
    )
    p.add_argument(
        "--train_allocation",
        choices=["proportional", "min_shortage", "random_sequential"],
        default="proportional",
    )
    p.add_argument(
        "--warm_start_at_best_base_stock",
        action="store_true",
        help="Seed CMA-ES x0 leaf bias at the best base-stock (W,R) so gen-0 reproduces the heuristic.",
    )
    p.add_argument("--sigma_init", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=123)
    # mp_num_processors is pinned to 1 (rayon-bounded); flag kept for parity, capped on use.
    p.add_argument("--mp_num_processors", type=int, default=1)
    return p.parse_args()


def _git_short_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def _resolve_policy_action_mode(parsed, reference) -> str:
    if parsed.policy_action_mode is not None:
        return parsed.policy_action_mode
    # Default: the geometry the learned benchmark used (symmetric instances) else a raw order vector.
    return common.policy_action_mode_for_reference(reference)


def _warm_start_flat_params(model, warehouse_level: int, retailer_level: int):
    """Build a flat-param vector whose leaf block makes a discrete_grid soft-tree emit the
    best base-stock target (W, R) at every leaf, so generation 0 reproduces the strongest
    heuristic exactly. CRITICAL: the soft-tree does NOT output the raw leaf parameter; it
    passes the leaf output through a per-leaf-type transform before grid-snapping
    (`src/core/policies/soft_tree.rs::action_vector_from_flat_params`):

      - constant leaf:  scaled = min + sigmoid(leaf_param) * (max - min)
                        => to emit target T, leaf_param = logit((T - min) / (max - min)).
      - linear  leaf:   scaled = min + softplus(leaf_bias + leaf_weights . state)
                        => zero the leaf weights and set leaf_bias = softplus_inv(T - min)
                           so the leaf is state-independent and emits exactly T at gen 0.

    Writing the raw target T directly (the previous behavior) sigmoid-saturated the
    constant leaf to the grid max and offset the linear leaf by `min`, so gen 0 was a
    badly over-stocked policy, NOT the heuristic. With this inversion, gen-0 holdout cost
    matches the heuristic to the rounding of (W, R) and CMA-ES searches outward from a
    known-good point.

    Only meaningful for symmetric_echelon_targets (control_dim == 2, target = [W, R]).
    Layout (see validate_soft_tree_flat_params): split weights, split bias, then either the
    num_leaves*action_dim constant block (constant leaf) or the
    num_leaves*action_dim*input_dim leaf-weight block followed by the num_leaves*action_dim
    leaf-bias block (linear leaf). Split params stay at their initialized values.

    Returns the flat params as a python list.
    """
    flat = np.asarray(model.get_model_flat_params(), dtype=np.float32).copy()
    num_leaves = 2 ** int(model.depth)
    action_dim = int(model.control_dim)
    if action_dim != 2:
        # Unsupported geometry for a (W, R) seed; fall back to the model's own init.
        return flat.tolist()

    min_values = [float(v) for v in model.min_values]
    max_values = [float(v) for v in model.max_values]
    targets = [float(warehouse_level), float(retailer_level)]
    leaf_type = str(model.leaf_type)
    bias_block = num_leaves * action_dim

    if leaf_type == "constant":
        if bias_block > flat.size:
            return flat.tolist()
        leaf_param = np.empty(action_dim, dtype=np.float32)
        for dim in range(action_dim):
            span = max_values[dim] - min_values[dim]
            if span <= 0.0:
                leaf_param[dim] = 0.0
                continue
            p = (targets[dim] - min_values[dim]) / span
            p = float(min(max(p, 1e-4), 1.0 - 1e-4))
            leaf_param[dim] = math.log(p / (1.0 - p))  # logit
        tail = flat[flat.size - bias_block:].reshape(num_leaves, action_dim)
        tail[:, :] = leaf_param
        flat[flat.size - bias_block:] = tail.reshape(-1)
        return flat.tolist()

    # linear / sigmoid_linear: zero the leaf weights so the leaf is state-independent, then
    # set the leaf bias to the softplus-inverse so softplus(bias) == target - min.
    input_dim = int(model.input_dim)
    weights_block = num_leaves * action_dim * input_dim
    if weights_block + bias_block > flat.size:
        return flat.tolist()
    weights_start = flat.size - weights_block - bias_block
    bias_start = flat.size - bias_block
    flat[weights_start:weights_start + weights_block] = 0.0
    leaf_bias = np.empty(action_dim, dtype=np.float32)
    for dim in range(action_dim):
        delta = max(targets[dim] - min_values[dim], 1e-6)
        leaf_bias[dim] = math.log(math.expm1(delta))  # softplus_inv
    bias = flat[bias_start:].reshape(num_leaves, action_dim)
    bias[:, :] = leaf_bias
    flat[bias_start:] = bias.reshape(-1)
    return flat.tolist()


def _training_namespace(parsed, reference, budget, output_root: Path) -> SimpleNamespace:
    run_name = (
        f"{parsed.run_tag}_{parsed.budget}_{reference['name']}_d{parsed.depth}"
        f"_{parsed.split_type}_{parsed.leaf_type}_pop{budget['es_population']}"
    )
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(parsed.sigma_init),
        es_population=int(budget["es_population"]),
        training_episodes=int(budget["training_episodes"]),
        mp_num_processors=1,  # pinned: parallelism is rayon, bounded by RAYON_NUM_THREADS
        save_every=max(1, int(budget["training_episodes"])),
        save_solutions=False,
        horizon=int(reference["benchmark_periods"]),
        seed=int(parsed.seed),
        train_seed_batch=int(budget["train_seed_batch"]),
        experiment_name=run_name,
        log_dir=str(output_root / "logs"),
        trained_models_dir=str(output_root / "models"),
    )


def run(parsed) -> dict:
    budget = BUDGETS[parsed.budget]
    reference = common.get_reference(parsed.reference)
    policy_action_mode = _resolve_policy_action_mode(parsed, reference)

    # ---- disjoint CRN blocks (search vs held-out) ----
    search_paths = _sample_demand_paths(reference, budget["search_paths"], SEARCH_SEED_START)
    holdout_paths = _sample_demand_paths(reference, budget["holdout_paths"], HOLDOUT_SEED_START)

    # ---- strongest heuristic: grid-search both allocations, re-score argmin on held-out ----
    heuristics: dict[str, dict] = {}
    for allocation in ("proportional", "min_shortage"):
        searched = _search_best_heuristic_on_paths(
            reference, allocation, search_paths, ALLOC_SEED_SEARCH
        )
        holdout_costs = _heuristic_on_paths(
            reference,
            searched["warehouse_base_stock_level"],
            searched["retailer_base_stock_levels"],
            allocation,
            holdout_paths,
            ALLOC_SEED_HOLDOUT,
        )
        heuristics[allocation] = {
            "warehouse_base_stock_level": int(searched["warehouse_base_stock_level"]),
            "retailer_base_stock_levels": [int(v) for v in searched["retailer_base_stock_levels"]],
            "holdout_mean_cost": float(holdout_costs.mean()),
            "holdout_stderr": float(holdout_costs.std() / np.sqrt(holdout_costs.size)),
        }
    best_alloc = min(heuristics, key=lambda a: heuristics[a]["holdout_mean_cost"])
    best_heuristic = heuristics[best_alloc]
    best_heuristic_cost = best_heuristic["holdout_mean_cost"]

    # ---- build the soft-tree with the CLI structure ----
    model = common.build_soft_tree_model(
        reference,
        depth=parsed.depth,
        temperature=parsed.temperature,
        split_type=parsed.split_type,
        leaf_type=parsed.leaf_type,
        policy_action_mode=policy_action_mode,
    )

    train_args = _training_namespace(
        parsed,
        reference,
        budget,
        PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag,
    )

    # ---- optional CMA-ES warm start at the best base-stock (W, R) ----
    warm_started = False
    warm_start_flat = None
    if parsed.warm_start_at_best_base_stock and policy_action_mode == "symmetric_echelon_targets":
        wr = best_heuristic["warehouse_base_stock_level"]
        rr = int(round(float(np.mean(best_heuristic["retailer_base_stock_levels"]))))
        warm_start_flat = _warm_start_flat_params(model, wr, rr)
        train_args.cma_x0 = warm_start_flat
        warm_started = True

    # ---- train (population-rollout binding; no Python pool) ----
    t0 = time.time()
    trained_model, fitness_hist = train(
        model=model,
        get_model_fitness=_get_model_fitness(
            model, reference, parsed.train_allocation, policy_action_mode
        ),
        get_population_fitness=_get_population_fitness(
            model, reference, parsed.train_allocation, policy_action_mode
        ),
        args=train_args,
        same_seed=False,
    )
    train_seconds = time.time() - t0
    trained_flat = np.asarray(trained_model.get_model_flat_params(), dtype=np.float32).tolist()

    # ---- evaluate learned on the held-out block under each allocation; headline = better ----
    learned_eval: dict[str, dict] = {}
    for allocation in ("proportional", "min_shortage"):
        costs = _soft_tree_on_paths(
            reference, trained_model, trained_flat, allocation, policy_action_mode,
            holdout_paths, ALLOC_SEED_HOLDOUT,
        )
        learned_eval[allocation] = {
            "holdout_mean_cost": float(costs.mean()),
            "holdout_stderr": float(costs.std() / np.sqrt(costs.size)),
        }
    learned_best_alloc = min(learned_eval, key=lambda a: learned_eval[a]["holdout_mean_cost"])
    trained_cost = learned_eval[learned_best_alloc]["holdout_mean_cost"]

    # ---- honest warm-start floor: the gen-0 anchor exactly reproduces the heuristic, so
    # the deployed policy is the BEST of {trained xbest, warm-start anchor} under the SAME
    # paired CRN held-out block. CMA-ES returns xbest on TRAINING seeds, which can overfit
    # relative to held-out; never report a headline worse than the anchor it started from. ----
    anchor_eval: dict[str, dict] = {}
    anchor_best_alloc = None
    anchor_cost = None
    if warm_start_flat is not None:
        for allocation in ("proportional", "min_shortage"):
            costs = _soft_tree_on_paths(
                reference, model, warm_start_flat, allocation, policy_action_mode,
                holdout_paths, ALLOC_SEED_HOLDOUT,
            )
            anchor_eval[allocation] = {
                "holdout_mean_cost": float(costs.mean()),
                "holdout_stderr": float(costs.std() / np.sqrt(costs.size)),
            }
        anchor_best_alloc = min(anchor_eval, key=lambda a: anchor_eval[a]["holdout_mean_cost"])
        anchor_cost = anchor_eval[anchor_best_alloc]["holdout_mean_cost"]

    if anchor_cost is not None and anchor_cost < trained_cost:
        learned_cost = anchor_cost
        learned_best_alloc = anchor_best_alloc
        deployed_policy = "warm_start_anchor"
    else:
        learned_cost = trained_cost
        deployed_policy = "trained_xbest"

    gap = learned_cost - best_heuristic_cost
    gap_pct = (best_heuristic_cost - learned_cost) / best_heuristic_cost * 100.0
    winner = "learned" if learned_cost < best_heuristic_cost else "heuristic"

    return {
        "reference": parsed.reference,
        "customer_behavior": reference["customer_behavior"],
        "policy_action_mode": policy_action_mode,
        "policy_architecture": (
            f"soft_tree_d{parsed.depth}_{parsed.split_type}_{parsed.leaf_type}"
            f"_temp{parsed.temperature}_{policy_action_mode}"
        ),
        "warm_started": warm_started,
        "best_heuristic_allocation": best_alloc,
        "best_heuristic_cost": best_heuristic_cost,
        "best_heuristic_levels": {
            "warehouse": best_heuristic["warehouse_base_stock_level"],
            "retailers": best_heuristic["retailer_base_stock_levels"],
        },
        "learned_best_allocation": learned_best_alloc,
        "learned_cost": learned_cost,
        "deployed_policy": deployed_policy,
        "trained_cost": trained_cost,
        "anchor_cost": anchor_cost,
        "anchor_best_allocation": anchor_best_alloc,
        "anchor_by_allocation": anchor_eval,
        "learned_by_allocation": learned_eval,
        "heuristics": heuristics,
        "gap": gap,
        "gap_pct": gap_pct,
        "winner": winner,
        "train_seconds": train_seconds,
        "best_train_reward": float(np.max(fitness_hist[-1])) if len(fitness_hist) else None,
    }


def main():
    parsed = parse_args()
    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    root.mkdir(parents=True, exist_ok=True)
    results_tsv = root / "results.tsv"
    header = [
        "commit", "run_tag", "budget", "reference", "customer_behavior",
        "policy_architecture", "warm_started", "train_allocation",
        "learned_cost", "learned_allocation",
        "best_heuristic", "best_heuristic_allocation",
        "gap", "gap_pct", "winner", "train_seconds", "description",
    ]
    if not results_tsv.exists():
        with results_tsv.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, delimiter="\t").writerow(header)

    out = run(parsed)

    row = [
        _git_short_commit(PACKAGE_ROOT),
        parsed.run_tag,
        parsed.budget,
        out["reference"],
        out["customer_behavior"],
        out["policy_architecture"],
        "1" if out["warm_started"] else "0",
        parsed.train_allocation,
        f"{out['learned_cost']:.6f}",
        out["learned_best_allocation"],
        f"{out['best_heuristic_cost']:.6f}",
        out["best_heuristic_allocation"],
        f"{out['gap']:.6f}",
        f"{out['gap_pct']:.4f}",
        out["winner"],
        f"{out['train_seconds']:.1f}",
        parsed.description,
    ]
    with results_tsv.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter="\t").writerow(row)

    print(json.dumps({"ledger_tsv": str(results_tsv), "ledger_row": dict(zip(header, row)), "detail": out}, indent=2))


if __name__ == "__main__":
    main()
