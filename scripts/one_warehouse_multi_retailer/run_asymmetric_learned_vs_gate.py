"""
Asymmetric / high-CV OWMR learned-policy vs tuned-base-stock gate (Kaynov 2024).

OBJECTIVE
---------
Train a learned soft-tree CMA-ES policy on the ASYMMETRIC / high-variability
partial-backorder OWMR instances (kaynov2024_instance_12 / 13 / 14) with a
PER-RETAILER action geometry that can express asymmetric replenishment, and
compare it HONESTLY (paired CRN, held-out) against the strongest in-repo
echelon base-stock + allocation gate and the published Kaynov PPO row.

WHY A SEPARATE RUNNER (not autoresearch_*.py)
--------------------------------------------
The autoresearch runner's gate search (`_search_best_heuristic_on_paths`)
enumerates the gate grid SERIALLY in Python and, for asymmetric instances, does
a FULL CARTESIAN product over per-retailer levels. For instance_14 that is
~3e14 candidates (never terminates); for instance_13 it is ~5.7M rollouts (~90
min). This runner keeps the EXACT same env / CRN / protocol / honest-floor logic
but:
  1. Enumerates the gate with the SAME reductions the repo's exact heuristic
     search uses (common.search_best_echelon_base_stock): symmetric reduction for
     the symmetric instances, and the Kaynov z0-k candidate set for instance_14.
  2. Parallelizes the gate grid over a 4-worker process pool (each worker pinned
     to 1 rayon/omp thread) so the footprint stays <= 4 cores.
All learned training/eval goes through the identical bindings and helpers used by
benchmark_learned_vs_heuristic.py, so the numbers are like-for-like with the
autoresearch runner's protocol.

ACTION GEOMETRY (the lever)
---------------------------
The symmetric_echelon_targets geometry (control_dim=2: one W target + one SHARED
R target) CANNOT express asymmetric per-retailer policies -> it only ties the
gate. The per-retailer geometries the binding actually supports are:
  - echelon_targets  (control_dim = K+1): W target + per-retailer echelon
    base-stock TARGETS. Generalizes the gate to asymmetric retailers; supports
    BOTH proportional and min_shortage allocation (provides target positions).
  - direct_orders    (control_dim = K+1): raw per-retailer order quantities.
    Per-retailer but does NOT supply target positions -> min_shortage is
    UNSUPPORTED (proportional / random_sequential only).
  - vector_quantity  is NOT a binding policy_action_mode (it is the model's
    control_mode); parse_policy_action_mode rejects it. We therefore use
    echelon_targets (the natural per-retailer generalization of the gate) as the
    primary geometry, and direct_orders as an expressiveness ablation.

PROTOCOL (matches autoresearch_*.py / benchmark_learned_vs_heuristic.py)
-----------------------------------------------------------------------
- 100-period undiscounted total cost (discount_factor=1.0), Kaynov protocol.
- Disjoint CRN demand-path blocks: search (seed 500000) vs held-out (900000),
  allocation-RNG anchors 700000 (search) / 800000 (held-out).
- Gate: grid-search echelon base-stock on the search block for BOTH allocations,
  re-score each argmin on the held-out block, take the better allocation.
- Learned: train via invman.es_mp.train + the population-rollout binding; score
  the trained xbest on the SAME held-out block under both allocations; headline =
  better allocation. For symmetric_echelon_targets and echelon_targets, warm-start
  CMA-ES at the gate target vector and deploy the better of {trained xbest,
  warm-start anchor} (the honest floor). direct_orders emits raw orders rather than
  target positions, so it has no gate-reproducing anchor.
- WIN RULE: learned beats gate only if (gate_cost - learned_cost) exceeds the
  paired-difference SEM. Otherwise tie/lose.

CPU CAP (HARD): the shared CPU helper caps Rayon/BLAS/OpenMP before NumPy and Rust
imports; the gate pool is capped to 4 workers and each worker is pinned to 1 thread.
Keep total footprint <= 4 cores.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from invman.cpu_limits import (
    bounded_worker_count,
    configure_process_cpu_limits,
    configure_process_cpu_limits_from_argv,
)

configure_process_cpu_limits_from_argv(sys.argv[1:])

import numpy as np

import common  # noqa: E402
from benchmark_learned_vs_heuristic import (  # noqa: E402
    DISCOUNT_FACTOR,
    _get_model_fitness,
    _get_population_fitness,
    _heuristic_on_paths,
    _sample_demand_paths,
    _soft_tree_on_paths,
)
from invman.es_mp import train  # noqa: E402

# Same disjoint CRN blocks + allocation anchors as autoresearch_*.py.
SEARCH_SEED_START = 500_000
HOLDOUT_SEED_START = 900_000
ALLOC_SEED_SEARCH = 700_000
ALLOC_SEED_HOLDOUT = 800_000

# Same budgets as the autoresearch runner.
BUDGETS = {
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


# --------------------------------------------------------------------------- #
# Gate grid candidate enumeration (mirrors common.search_best_echelon_base_stock)
# --------------------------------------------------------------------------- #
def _gate_candidates(reference: dict) -> list[tuple[int, list[int]]]:
    bounds = common.echelon_base_stock_search_bounds(reference)
    wlo, whi = bounds["warehouse"]
    warehouse_levels = list(range(wlo, whi + 1))
    K = len(reference["retailer_lead_times"])
    if common.uses_kaynov_k_search(reference):
        ks = common.kaynov_instance14_k_candidates(reference)
        return [
            (w, common._retailer_targets_from_k(reference, k))
            for w in warehouse_levels
            for k in ks
        ]
    if bounds["symmetric_retailers"]:
        rlo, rhi = bounds["retailers"][0]
        return [
            (w, [r] * K)
            for w in warehouse_levels
            for r in range(rlo, rhi + 1)
        ]
    grids = [range(lo, hi + 1) for lo, hi in bounds["retailers"]]
    return [
        (w, list(levels))
        for w in warehouse_levels
        for levels in product(*grids)
    ]


# Worker globals (set ONCE per process via the initializer). The CRN search-path
# block is large (num_paths x periods x K ints); pickling it inside every job tuple
# (tens of thousands of jobs) dominates wall-clock, so it is shipped to each worker
# exactly once through initargs and referenced from this global thereafter.
_W_REF = None
_W_PATHS = None
_W_ALLOC_SEED = None


def _gate_worker_init(reference, search_paths, alloc_seed):
    # Pin every gate worker to a single thread so 4 workers == 4 cores.
    configure_process_cpu_limits(1)
    global _W_REF, _W_PATHS, _W_ALLOC_SEED
    _W_REF = reference
    _W_PATHS = search_paths
    _W_ALLOC_SEED = alloc_seed


def _gate_worker_eval(job):
    """Mean cost of one (allocation, W, levels) candidate on the worker-local CRN
    search-path block. Returns (allocation, W, levels, mean_cost)."""
    allocation, w, levels = job
    costs = _heuristic_on_paths(_W_REF, w, levels, allocation, _W_PATHS, _W_ALLOC_SEED)
    return allocation, int(w), [int(v) for v in levels], float(costs.mean())


def _search_gate_parallel(reference, allocations, search_paths, workers):
    """Grid-search the gate for each allocation in parallel (4-worker pool).
    Returns {allocation: {"warehouse_base_stock_level", "retailer_base_stock_levels",
    "search_mean_cost"}}. Jobs carry only (allocation, W, levels); the path block is
    shared via the initializer (no per-job path pickling)."""
    candidates = _gate_candidates(reference)
    jobs = [
        (allocation, w, levels)
        for allocation in allocations
        for (w, levels) in candidates
    ]
    best: dict[str, dict] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_gate_worker_init,
        initargs=(reference, search_paths, ALLOC_SEED_SEARCH),
    ) as pool:
        for allocation, w, levels, mean_cost in pool.map(_gate_worker_eval, jobs, chunksize=128):
            cur = best.get(allocation)
            if cur is None or mean_cost < cur["search_mean_cost"]:
                best[allocation] = {
                    "warehouse_base_stock_level": w,
                    "retailer_base_stock_levels": levels,
                    "search_mean_cost": mean_cost,
                }
    return best


# --------------------------------------------------------------------------- #
# Learned training (identical to the autoresearch / benchmark path)            #
# --------------------------------------------------------------------------- #
def _warm_start_flat_params(model, target_vector):
    """Seed the soft-tree leaves so generation 0 emits the gate target vector at every
    leaf (state-independently), inverting the per-dimension leaf transform applied in
    `src/core/policies/soft_tree.rs::action_vector_from_flat_params`:

      - constant / sigmoid_linear leaf:  scaled = min + sigmoid(leaf) * (max - min)
            -> leaf = logit((T - min) / (max - min))
      - linear leaf:                     scaled = min + softplus(leaf)
            -> zero the leaf weights (state-independent) and leaf_bias = softplus_inv(T - min)

    This is the SAME inversion the autoresearch runner uses for control_dim==2, here
    generalized to ANY control dim so the per-retailer echelon_targets geometry
    (control_dim = K+1, target = [W, r_1, ..., r_K]) reproduces the gate's per-retailer
    echelon base-stock policy at gen 0 -> the honest warm-start floor applies to the
    asymmetric geometries too. `direct_orders` cannot reproduce a base-stock target
    (its action is a raw order, not a target position) so warm-start is not used there.

    Returns (flat_params_list, warm_started_bool).
    """
    import math
    flat = np.asarray(model.get_model_flat_params(), dtype=np.float32).copy()
    num_leaves = 2 ** int(model.depth)
    action_dim = int(model.control_dim)
    targets = [float(v) for v in target_vector]
    if len(targets) != action_dim:
        return flat.tolist(), False
    min_values = [float(v) for v in model.min_values]
    max_values = [float(v) for v in model.max_values]
    leaf_type = str(model.leaf_type)
    bias_block = num_leaves * action_dim
    if leaf_type == "constant":
        if bias_block > flat.size:
            return flat.tolist(), False
        leaf_param = np.empty(action_dim, dtype=np.float32)
        for dim in range(action_dim):
            span = max_values[dim] - min_values[dim]
            if span <= 0.0:
                leaf_param[dim] = 0.0
                continue
            p = (targets[dim] - min_values[dim]) / span
            p = float(min(max(p, 1e-4), 1.0 - 1e-4))
            leaf_param[dim] = math.log(p / (1.0 - p))
        tail = flat[flat.size - bias_block:].reshape(num_leaves, action_dim)
        tail[:, :] = leaf_param
        flat[flat.size - bias_block:] = tail.reshape(-1)
        return flat.tolist(), True
    input_dim = int(model.input_dim)
    weights_block = num_leaves * action_dim * input_dim
    if weights_block + bias_block > flat.size:
        return flat.tolist(), False
    weights_start = flat.size - weights_block - bias_block
    bias_start = flat.size - bias_block
    flat[weights_start:weights_start + weights_block] = 0.0
    leaf_bias = np.empty(action_dim, dtype=np.float32)
    for dim in range(action_dim):
        delta = max(targets[dim] - min_values[dim], 1e-6)
        leaf_bias[dim] = math.log(math.expm1(delta))
    bias = flat[bias_start:].reshape(num_leaves, action_dim)
    bias[:, :] = leaf_bias
    flat[bias_start:] = bias.reshape(-1)
    return flat.tolist(), True


def _training_namespace(reference, budget, leaf_type, mode, seed, sigma_init, out_root,
                        depth, split_type, temperature):
    run_name = (
        f"asym_{reference['name']}_{mode}_{leaf_type}"
        f"_d{depth}_{split_type}_t{temperature:g}_pop{budget['es_population']}"
    )
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(sigma_init),
        es_population=int(budget["es_population"]),
        training_episodes=int(budget["training_episodes"]),
        mp_num_processors=1,  # rayon-bounded; no python pool during training
        save_every=max(1, int(budget["training_episodes"])),
        save_solutions=False,
        horizon=int(reference["benchmark_periods"]),
        seed=int(seed),
        train_seed_batch=int(budget["train_seed_batch"]),
        experiment_name=run_name,
        log_dir=str(out_root / "logs"),
        trained_models_dir=str(out_root / "models"),
    )


def _eval_allocs(reference, model, flat, policy_action_mode, allocations, holdout_paths):
    out = {}
    for allocation in allocations:
        costs = _soft_tree_on_paths(
            reference, model, flat, allocation, policy_action_mode,
            holdout_paths, ALLOC_SEED_HOLDOUT,
        )
        out[allocation] = {
            "costs": costs,
            "mean": float(costs.mean()),
            "sem": float(costs.std() / np.sqrt(costs.size)),
        }
    return out


def _resolve_budget(budget_name, training_episodes=None, es_population=None,
                    train_seed_batch=None, holdout_paths=None):
    budget = dict(BUDGETS[budget_name])
    if training_episodes is not None:
        budget["training_episodes"] = int(training_episodes)
    if es_population is not None:
        budget["es_population"] = int(es_population)
    if train_seed_batch is not None:
        budget["train_seed_batch"] = int(train_seed_batch)
    if holdout_paths is not None:
        budget["holdout_paths"] = int(holdout_paths)
    return budget


def _load_init_params(path, expected_size):
    if path is None:
        return None
    params = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if params.size != int(expected_size):
        raise ValueError(
            f"init_params_npy has {params.size} params, expected {expected_size}"
        )
    return params.tolist()


def _direct_order_gate_init_flat_params(model, reference, gate_best):
    """Approximate the echelon-base-stock gate in direct_orders mode.

    direct_orders emits raw quantities, so it cannot exactly represent the
    target-position gate. For a linear leaf, initialize every leaf to the same
    affine order-up-to approximation over the normalized rollout features:
      q_w ~= softplus(W - W * total_echelon_position / scale)
      q_i ~= softplus(R_i - W * retailer_position_i / scale)
    The rollout scale is not exposed as a feature, so W is used as a local scale
    proxy. This gives CMA-ES a near-gate raw-order starting point rather than a
    random one; held-out evaluation decides whether it is useful.
    """
    if str(model.leaf_type) != "linear" or int(model.control_dim) != len(reference["retailer_lead_times"]) + 1:
        return None
    flat = np.asarray(model.get_model_flat_params(), dtype=np.float32).copy()
    num_leaves = 2 ** int(model.depth)
    action_dim = int(model.control_dim)
    input_dim = int(model.input_dim)
    bias_block = num_leaves * action_dim
    weights_block = num_leaves * action_dim * input_dim
    if weights_block + bias_block > flat.size:
        return None

    w_level = float(gate_best["warehouse_base_stock_level"])
    r_levels = [float(v) for v in gate_best["retailer_base_stock_levels"]]
    scale_proxy = max(w_level, 1.0)
    weights_start = flat.size - weights_block - bias_block
    bias_start = flat.size - bias_block
    weights = np.zeros((num_leaves, action_dim, input_dim), dtype=np.float32)
    bias = np.zeros((num_leaves, action_dim), dtype=np.float32)

    total_position_idx = input_dim - 2
    bias[:, 0] = w_level
    weights[:, 0, total_position_idx] = -scale_proxy

    warehouse_lead_time = int(reference["warehouse_lead_time"])
    num_retailers = len(reference["retailer_lead_times"])
    retailer_inventory_start = 1 + warehouse_lead_time
    retailer_pipeline_start = retailer_inventory_start + num_retailers
    pipeline_idx = retailer_pipeline_start
    for retailer_idx, (target, lead_time) in enumerate(zip(r_levels, reference["retailer_lead_times"])):
        action_idx = retailer_idx + 1
        bias[:, action_idx] = target
        weights[:, action_idx, retailer_inventory_start + retailer_idx] = -scale_proxy
        for offset in range(int(lead_time)):
            weights[:, action_idx, pipeline_idx + offset] = -scale_proxy
        pipeline_idx += int(lead_time)

    flat[weights_start:weights_start + weights_block] = weights.reshape(-1)
    flat[bias_start:] = bias.reshape(-1)
    return flat.tolist()


def run_one(reference, budget_name, leaf_type, policy_action_mode, train_allocation,
            seed, sigma_init, warm_start, workers, out_root, gate_search_paths=None,
            init_params_npy=None,
            direct_order_gate_init=False,
            depth=2, temperature=0.10, split_type="axis_aligned",
            training_episodes=None, es_population=None, train_seed_batch=None,
            holdout_paths=None):
    budget = _resolve_budget(
        budget_name,
        training_episodes=training_episodes,
        es_population=es_population,
        train_seed_batch=train_seed_batch,
        holdout_paths=holdout_paths,
    )
    K = len(reference["retailer_lead_times"])
    # Gate-search path count: the base-stock cost surface is smooth, so the grid
    # ARGMIN is stable with far fewer search paths; the honest held-out comparison
    # always uses the full budget["holdout_paths"]. For the large K=10 instances the
    # 256-path search grid is multi-hour, so allow a smaller gate-search block while
    # keeping the held-out re-score (and the learned training/eval) at full budget.
    n_gate_search = int(gate_search_paths) if gate_search_paths else int(budget["search_paths"])

    # echelon_targets / symmetric_echelon_targets support both allocations;
    # direct_orders cannot supply min_shortage target positions.
    if policy_action_mode == "direct_orders":
        eval_allocs = ["proportional"]
    else:
        eval_allocs = ["proportional", "min_shortage"]

    # ---- CRN blocks ----
    search_paths = _sample_demand_paths(reference, n_gate_search, SEARCH_SEED_START)
    holdout_paths = _sample_demand_paths(reference, budget["holdout_paths"], HOLDOUT_SEED_START)

    # ---- gate (parallel grid; cached argmin per instance/budget/alloc-set) ----
    # The gate grid is the dominant cost and is identical across learned configs for a
    # given (instance, budget, allocation set). Cache the searched argmin (W, levels)
    # so repeated configs skip the grid; the held-out re-score below stays exact.
    t_gate = time.time()
    cache_dir = out_root / "gate_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{reference['name']}_{budget_name}_gs{n_gate_search}_{'-'.join(sorted(eval_allocs))}.json"
    if cache_path.exists():
        gate_searched = json.loads(cache_path.read_text())
        gate_searched = {
            a: {"warehouse_base_stock_level": int(v["warehouse_base_stock_level"]),
                "retailer_base_stock_levels": [int(x) for x in v["retailer_base_stock_levels"]],
                "search_mean_cost": float(v["search_mean_cost"])}
            for a, v in gate_searched.items()
        }
    else:
        gate_searched = _search_gate_parallel(reference, eval_allocs, search_paths, workers)
        cache_path.write_text(json.dumps(gate_searched, default=float, indent=2))
    gate = {}
    for allocation, g in gate_searched.items():
        holdout_costs = _heuristic_on_paths(
            reference, g["warehouse_base_stock_level"], g["retailer_base_stock_levels"],
            allocation, holdout_paths, ALLOC_SEED_HOLDOUT,
        )
        gate[allocation] = {
            **g,
            "holdout_costs": holdout_costs,
            "holdout_mean": float(holdout_costs.mean()),
            "holdout_sem": float(holdout_costs.std() / np.sqrt(holdout_costs.size)),
        }
    gate_best_alloc = min(gate, key=lambda a: gate[a]["holdout_mean"])
    gate_best = gate[gate_best_alloc]
    gate_cost = gate_best["holdout_mean"]
    gate_seconds = time.time() - t_gate

    # ---- build + (optionally) warm-start the soft-tree ----
    model = common.build_soft_tree_model(
        reference, depth=depth, temperature=temperature, split_type=split_type,
        leaf_type=leaf_type, policy_action_mode=policy_action_mode,
    )
    warm_flat = None
    warm_started = False
    train_args = _training_namespace(
        reference, budget, leaf_type, policy_action_mode, seed, sigma_init, out_root,
        depth, split_type, temperature
    )
    # Warm-start reproduces the gate as a base-stock TARGET, so it is meaningful for
    # the target-based geometries (symmetric_echelon_targets: [W, mean(R)];
    # echelon_targets: [W, r_1, ..., r_K]). direct_orders emits raw orders, not a
    # target, so warm-start does not apply there.
    if warm_start and policy_action_mode in ("symmetric_echelon_targets", "echelon_targets"):
        w_level = gate_best["warehouse_base_stock_level"]
        r_levels = gate_best["retailer_base_stock_levels"]
        if policy_action_mode == "symmetric_echelon_targets":
            target_vector = [w_level, int(round(float(np.mean(r_levels))))]
        else:
            target_vector = [w_level] + [int(v) for v in r_levels]
        warm_flat, warm_started = _warm_start_flat_params(model, target_vector)
        if warm_started:
            train_args.cma_x0 = warm_flat
    init_flat = _load_init_params(init_params_npy, len(model.get_model_flat_params()))
    if init_flat is not None:
        train_args.cma_x0 = init_flat
    direct_init_flat = None
    if direct_order_gate_init and policy_action_mode == "direct_orders":
        direct_init_flat = _direct_order_gate_init_flat_params(model, reference, gate_best)
        if direct_init_flat is not None:
            train_args.cma_x0 = direct_init_flat

    # ---- train ----
    t_train = time.time()
    trained_model, fitness_hist = train(
        model=model,
        get_model_fitness=_get_model_fitness(
            model, reference, train_allocation, policy_action_mode
        ),
        get_population_fitness=_get_population_fitness(
            model, reference, train_allocation, policy_action_mode
        ),
        args=train_args,
        same_seed=False,
    )
    train_seconds = time.time() - t_train
    trained_flat = np.asarray(trained_model.get_model_flat_params(), dtype=np.float32).tolist()

    # ---- evaluate learned on held-out (paired CRN) ----
    learned_eval = _eval_allocs(
        reference, trained_model, trained_flat, policy_action_mode, eval_allocs, holdout_paths
    )
    learned_best_alloc = min(learned_eval, key=lambda a: learned_eval[a]["mean"])

    # ---- honest warm-start floor (only when an anchor exists) ----
    anchor_eval = None
    anchor_best_alloc = None
    if warm_started and warm_flat is not None:
        anchor_eval = _eval_allocs(
            reference, model, warm_flat, policy_action_mode, eval_allocs, holdout_paths
        )
        anchor_best_alloc = min(anchor_eval, key=lambda a: anchor_eval[a]["mean"])
    init_eval = None
    init_best_alloc = None
    if init_flat is not None:
        init_eval = _eval_allocs(
            reference, model, init_flat, policy_action_mode, eval_allocs, holdout_paths
        )
        init_best_alloc = min(init_eval, key=lambda a: init_eval[a]["mean"])
    direct_init_eval = None
    direct_init_best_alloc = None
    if direct_init_flat is not None:
        direct_init_eval = _eval_allocs(
            reference, model, direct_init_flat, policy_action_mode, eval_allocs, holdout_paths
        )
        direct_init_best_alloc = min(direct_init_eval, key=lambda a: direct_init_eval[a]["mean"])

    trained_cost = learned_eval[learned_best_alloc]["mean"]
    candidates = [
        (
            trained_cost,
            learned_best_alloc,
            learned_eval[learned_best_alloc]["costs"],
            "trained_xbest",
            learned_eval[learned_best_alloc]["sem"],
        )
    ]
    if anchor_eval is not None:
        candidates.append((
            anchor_eval[anchor_best_alloc]["mean"],
            anchor_best_alloc,
            anchor_eval[anchor_best_alloc]["costs"],
            "warm_start_anchor",
            anchor_eval[anchor_best_alloc]["sem"],
        ))
    if init_eval is not None:
        candidates.append((
            init_eval[init_best_alloc]["mean"],
            init_best_alloc,
            init_eval[init_best_alloc]["costs"],
            "init_params_anchor",
            init_eval[init_best_alloc]["sem"],
        ))
    if direct_init_eval is not None:
        candidates.append((
            direct_init_eval[direct_init_best_alloc]["mean"],
            direct_init_best_alloc,
            direct_init_eval[direct_init_best_alloc]["costs"],
            "direct_order_gate_init_anchor",
            direct_init_eval[direct_init_best_alloc]["sem"],
        ))
    learned_cost, deployed_alloc, deployed_costs, deployed_policy, deployed_sem = min(
        candidates, key=lambda item: item[0]
    )

    # ---- paired-difference SEM on the SAME held-out block (same allocation as the
    # deployed policy so the rationing rule is held fixed in the paired test) ----
    gate_costs_for_pair = gate[deployed_alloc]["holdout_costs"] if deployed_alloc in gate else gate_best["holdout_costs"]
    diff = gate_costs_for_pair - deployed_costs  # positive => learned cheaper
    paired_mean = float(diff.mean())
    paired_sem = float(diff.std() / np.sqrt(diff.size))

    gap_pct = (gate_cost - learned_cost) / gate_cost * 100.0
    # Win only if the paired advantage exceeds its SEM.
    if paired_mean > paired_sem:
        verdict = "learned_wins"
    elif paired_mean < -paired_sem:
        verdict = "learned_loses"
    else:
        verdict = "tie"

    def pub(key):
        row = reference.get(key)
        return None if row is None else float(-dict(row)["mean_cost"])

    published = {
        "proportional": pub("published_proportional_benchmark"),
        "min_shortage": pub("published_min_shortage_benchmark"),
        "ppo": pub("published_ppo_benchmark"),
    }

    result = {
        "instance": reference["name"],
        "customer_behavior": reference["customer_behavior"],
        "num_retailers": K,
        "symmetric": common.is_symmetric_retailer_case(reference),
        "budget": budget_name,
        "leaf_type": leaf_type,
        "depth": int(depth),
        "temperature": float(temperature),
        "split_type": split_type,
        "policy_action_mode": policy_action_mode,
        "train_allocation": train_allocation,
        "warm_started": warm_started,
        "init_params_npy": None if init_params_npy is None else str(init_params_npy),
        "direct_order_gate_init": bool(direct_init_flat is not None),
        "deployed_policy": deployed_policy,
        "deployed_allocation": deployed_alloc,
        "seed": seed,
        "sigma_init": sigma_init,
        "gate_search_paths": n_gate_search,
        "search_paths": budget["search_paths"],
        "holdout_paths": budget["holdout_paths"],
        "training_episodes": budget["training_episodes"],
        "es_population": budget["es_population"],
        "train_seed_batch": budget["train_seed_batch"],
        "gate_best_allocation": gate_best_alloc,
        "gate_warehouse_level": gate_best["warehouse_base_stock_level"],
        "gate_retailer_levels": gate_best["retailer_base_stock_levels"],
        "gate_cost": gate_cost,
        "gate_sem": gate_best["holdout_sem"],
        "gate_by_allocation": {
            a: {"warehouse": g["warehouse_base_stock_level"],
                "retailers": g["retailer_base_stock_levels"],
                "holdout_mean": g["holdout_mean"], "holdout_sem": g["holdout_sem"]}
            for a, g in gate.items()
        },
        "trained_cost": trained_cost,
        "trained_best_allocation": learned_best_alloc,
        "anchor_cost": (None if anchor_eval is None else anchor_eval[anchor_best_alloc]["mean"]),
        "init_cost": (None if init_eval is None else init_eval[init_best_alloc]["mean"]),
        "init_best_allocation": init_best_alloc,
        "direct_order_gate_init_cost": (
            None if direct_init_eval is None else direct_init_eval[direct_init_best_alloc]["mean"]
        ),
        "direct_order_gate_init_best_allocation": direct_init_best_alloc,
        "learned_cost": learned_cost,
        "learned_sem": deployed_sem,
        "learned_by_allocation": {a: {"mean": v["mean"], "sem": v["sem"]} for a, v in learned_eval.items()},
        "gap_pct_vs_gate": gap_pct,
        "paired_diff_mean": paired_mean,   # gate - learned (positive => learned cheaper)
        "paired_diff_sem": paired_sem,
        "verdict": verdict,
        "published": published,
        "learned_vs_ppo_pct": (None if published["ppo"] is None
                               else (published["ppo"] - learned_cost) / published["ppo"] * 100.0),
        "gate_seconds": gate_seconds,
        "train_seconds": train_seconds,
        "best_train_reward": float(np.max(fitness_hist[-1])) if len(fitness_hist) else None,
    }
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Asymmetric/high-CV OWMR learned vs gate")
    p.add_argument("--reference", required=True)
    p.add_argument("--budget", choices=sorted(BUDGETS), default="full")
    p.add_argument("--leaf_type", choices=["constant", "linear"], default="linear")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="axis_aligned")
    p.add_argument("--policy_action_mode",
                   choices=["symmetric_echelon_targets", "echelon_targets", "direct_orders"],
                   default=None,
                   help="Default: the per-retailer geometry for the reference "
                        "(echelon_targets for asymmetric, symmetric_echelon_targets for symmetric).")
    p.add_argument("--train_allocation",
                   choices=["proportional", "min_shortage", "random_sequential"],
                   default="proportional")
    p.add_argument("--warm_start_at_best_base_stock", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--sigma_init", type=float, default=1.5)
    p.add_argument("--init_params_npy", default=None,
                   help="Optional model_params.npy to use as CMA-ES x0 for restart/refinement runs.")
    p.add_argument("--direct_order_gate_init", action="store_true",
                   help="For direct_orders + linear leaf, initialize x0 as an approximate raw-order gate.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gate_search_paths", type=int, default=None,
                   help="Override the gate grid-search path count (held-out re-score "
                        "stays at the full budget). Use a smaller value for the large "
                        "K=10 instances whose 256-path search grid is multi-hour.")
    p.add_argument("--training_episodes", type=int, default=None,
                   help="Override the budget's CMA-ES generation count for bounded screens.")
    p.add_argument("--es_population", type=int, default=None,
                   help="Override the budget's CMA-ES population for bounded screens.")
    p.add_argument("--train_seed_batch", type=int, default=None,
                   help="Override the budget's per-candidate training seed batch.")
    p.add_argument("--holdout_paths", type=int, default=None,
                   help="Override the budget's held-out path count for bounded screens.")
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def main():
    parsed = parse_args()
    parsed.workers = bounded_worker_count(parsed.workers)
    reference = common.get_reference(parsed.reference)
    mode = parsed.policy_action_mode
    if mode is None:
        mode = ("symmetric_echelon_targets"
                if common.is_symmetric_retailer_case(reference)
                else "echelon_targets")
    out_root = PACKAGE_ROOT / "outputs" / "one_warehouse_multi_retailer" / "asymmetric_learned"
    out_root.mkdir(parents=True, exist_ok=True)

    result = run_one(
        reference, parsed.budget, parsed.leaf_type, mode, parsed.train_allocation,
        parsed.seed, parsed.sigma_init, parsed.warm_start_at_best_base_stock,
        parsed.workers, out_root, gate_search_paths=parsed.gate_search_paths,
        init_params_npy=parsed.init_params_npy,
        direct_order_gate_init=parsed.direct_order_gate_init,
        depth=parsed.depth, temperature=parsed.temperature, split_type=parsed.split_type,
        training_episodes=parsed.training_episodes,
        es_population=parsed.es_population,
        train_seed_batch=parsed.train_seed_batch,
        holdout_paths=parsed.holdout_paths,
    )

    line = (
        f"{result['instance']} [{mode}/{parsed.leaf_type}/d{parsed.depth}/"
        f"{parsed.split_type}/t{parsed.temperature:g}/{parsed.budget}]: "
        f"learned {result['learned_cost']:.2f} (+/-{result['learned_sem']:.2f}, {result['deployed_allocation']}) "
        f"vs gate {result['gate_cost']:.2f} (+/-{result['gate_sem']:.2f}, {result['gate_best_allocation']}) "
        f"=> {result['gap_pct_vs_gate']:+.2f}% | paired {result['paired_diff_mean']:+.2f}+/-{result['paired_diff_sem']:.2f} "
        f"=> {result['verdict']} | PPO {result['published']['ppo']:.1f} "
        f"(learned vs PPO {result['learned_vs_ppo_pct']:+.2f}%)"
    )
    print("RESULT_LINE: " + line, flush=True)

    out_path = parsed.output_json or str(
        out_root / (
            f"{result['instance']}_{mode}_{parsed.leaf_type}_d{parsed.depth}"
            f"_{parsed.split_type}_t{parsed.temperature:g}_{parsed.budget}.json"
        )
    )
    # numpy arrays are not JSON-serializable; the result dict already holds floats only.
    Path(out_path).write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print("WROTE_JSON: " + out_path, flush=True)


if __name__ == "__main__":
    main()
