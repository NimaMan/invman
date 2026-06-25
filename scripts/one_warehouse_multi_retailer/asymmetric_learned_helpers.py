"""Helper routines for the asymmetric learned-policy vs gate runner."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from types import SimpleNamespace

import numpy as np

import common
from benchmark_learned_vs_heuristic import _heuristic_on_paths, _soft_tree_on_paths
from invman.cpu_limits import configure_process_cpu_limits

# Same disjoint CRN blocks + allocation anchors as autoresearch_*.py.
SEARCH_SEED_START = 500_000
TRAIN_SEED_START = 600_000
HOLDOUT_SEED_START = 900_000
ALLOC_SEED_SEARCH = 700_000
ALLOC_SEED_TRAIN = 750_000
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


def _gate_candidates(reference: dict) -> list[tuple[int, list[int]]]:
    bounds = common.echelon_base_stock_search_bounds(reference)
    wlo, whi = bounds["warehouse"]
    warehouse_levels = list(range(wlo, whi + 1))
    k_retailers = len(reference["retailer_lead_times"])
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
            (w, [r] * k_retailers)
            for w in warehouse_levels
            for r in range(rlo, rhi + 1)
        ]
    grids = [range(lo, hi + 1) for lo, hi in bounds["retailers"]]
    return [
        (w, list(levels))
        for w in warehouse_levels
        for levels in product(*grids)
    ]


# Worker globals (set once per process via the initializer). The CRN search-path
# block is large; shipping it through initargs avoids per-job path pickling.
_W_REF = None
_W_PATHS = None
_W_ALLOC_SEED = None


def _gate_worker_init(reference, search_paths, alloc_seed):
    configure_process_cpu_limits(1)
    global _W_REF, _W_PATHS, _W_ALLOC_SEED
    _W_REF = reference
    _W_PATHS = search_paths
    _W_ALLOC_SEED = alloc_seed


def _gate_worker_eval(job):
    """Evaluate one (allocation, W, levels) candidate on worker-local CRN paths."""
    allocation, w, levels = job
    costs = _heuristic_on_paths(_W_REF, w, levels, allocation, _W_PATHS, _W_ALLOC_SEED)
    return allocation, int(w), [int(v) for v in levels], float(costs.mean())


def _search_gate_parallel(reference, allocations, search_paths, workers):
    """Grid-search the gate for each allocation in a bounded process pool."""
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


def _warm_start_flat_params(model, target_vector, signed_tail_dims=None):
    """Seed soft-tree leaves so generation 0 emits the gate target vector.

    Inverts the per-dimension leaf transform applied in
    `src/core/policies/soft_tree.rs::action_vector_from_flat_params`.
    """
    tail_dims = set(int(d) for d in (signed_tail_dims or ()))
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
            if dim in tail_dims:
                leaf_param[dim] = 0.0
                continue
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
        if dim in tail_dims:
            leaf_bias[dim] = 0.0
            continue
        delta = max(targets[dim] - min_values[dim], 1e-6)
        leaf_bias[dim] = math.log(math.expm1(delta))
    bias = flat[bias_start:].reshape(num_leaves, action_dim)
    bias[:, :] = leaf_bias
    flat[bias_start:] = bias.reshape(-1)
    return flat.tolist(), True


def _training_namespace(
    reference,
    budget,
    leaf_type,
    mode,
    train_allocation,
    seed,
    sigma_init,
    out_root,
    depth,
    split_type,
    temperature,
    policy_state_mode="normalized",
    same_seed=False,
    train_on_fixed_paths=False,
):
    sigma_tag = f"{float(sigma_init):g}".replace(".", "p")
    same_seed_tag = "_crn" if same_seed else ""
    fixed_paths_tag = "_fixedpaths" if train_on_fixed_paths else ""
    state_tag = "" if policy_state_mode == "normalized" else f"_{policy_state_mode}"
    run_name = (
        f"asym_{reference['name']}_{mode}_{leaf_type}"
        f"_d{depth}_{split_type}_t{temperature:g}{state_tag}_pop{budget['es_population']}"
        f"_gen{budget['training_episodes']}_batch{budget['train_seed_batch']}"
        f"_{train_allocation}{same_seed_tag}{fixed_paths_tag}_sig{sigma_tag}_seed{seed}"
    )
    return SimpleNamespace(
        training_method="cma",
        sigma_init=float(sigma_init),
        es_population=int(budget["es_population"]),
        training_episodes=int(budget["training_episodes"]),
        mp_num_processors=1,
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
            reference,
            model,
            flat,
            allocation,
            policy_action_mode,
            holdout_paths,
            ALLOC_SEED_HOLDOUT,
        )
        out[allocation] = {
            "costs": costs,
            "mean": float(costs.mean()),
            "sem": float(costs.std() / np.sqrt(costs.size)),
        }
    return out


def _get_fixed_path_population_fitness(
    reference,
    model,
    allocation,
    policy_action_mode,
    train_paths,
    alloc_seed,
):
    def inner(_model, args, model_params_batch, seeds):
        del _model, args, seeds
        out = []
        for idx, params in enumerate(model_params_batch):
            costs = _soft_tree_on_paths(
                reference,
                model,
                params,
                allocation,
                policy_action_mode,
                train_paths,
                alloc_seed,
            )
            out.append((-float(costs.mean()), idx))
        return out

    return inner


def _resolve_budget(
    budget_name,
    training_episodes=None,
    es_population=None,
    train_seed_batch=None,
    holdout_paths=None,
):
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


def _expand_soft_tree_params_to_model(
    params,
    model,
    old_input_dim,
    old_action_dim,
    action_map,
    action_bias_offsets=None,
):
    """Embed an older soft-tree layout into the current model layout."""
    flat = np.asarray(params, dtype=np.float32).reshape(-1)
    num_leaves = 2 ** int(model.depth)
    num_internal = (2 ** int(model.depth)) - 1
    new_input_dim = int(model.input_dim)
    new_action_dim = int(model.control_dim)
    old_input_dim = int(old_input_dim)
    old_action_dim = int(old_action_dim)
    if len(action_map) != new_action_dim or any(idx >= old_action_dim for idx in action_map):
        return None
    if action_bias_offsets is None:
        action_bias_offsets = [0.0] * new_action_dim
    if len(action_bias_offsets) != new_action_dim:
        return None

    old_prefix = num_internal * old_input_dim + num_internal
    new_prefix = num_internal * new_input_dim + num_internal
    leaf_type = str(model.leaf_type)
    if old_input_dim > new_input_dim:
        return None

    expected_new = len(model.get_model_flat_params())
    new_flat = np.zeros(expected_new, dtype=np.float32)
    old_split_end = num_internal * old_input_dim
    new_split_end = num_internal * new_input_dim
    if flat.size < old_prefix:
        return None
    if num_internal:
        old_split = flat[:old_split_end].reshape(num_internal, old_input_dim)
        new_split = np.zeros((num_internal, new_input_dim), dtype=np.float32)
        new_split[:, :old_input_dim] = old_split
        new_flat[:new_split_end] = new_split.reshape(-1)
        new_flat[new_split_end:new_prefix] = flat[old_split_end:old_prefix]

    if leaf_type == "constant":
        expected_old = old_prefix + num_leaves * old_action_dim
        if flat.size != expected_old:
            return None
        old_tail = flat[old_prefix:].reshape(num_leaves, old_action_dim)
        new_tail = np.empty((num_leaves, new_action_dim), dtype=np.float32)
        for new_idx, old_idx in enumerate(action_map):
            new_tail[:, new_idx] = old_tail[:, old_idx]
        new_flat[new_prefix:] = new_tail.reshape(-1)
        return new_flat

    weights_len_old = num_leaves * old_action_dim * old_input_dim
    bias_len_old = num_leaves * old_action_dim
    expected_old = old_prefix + weights_len_old + bias_len_old
    if flat.size != expected_old:
        return None
    old_weights = flat[old_prefix:old_prefix + weights_len_old].reshape(
        num_leaves,
        old_action_dim,
        old_input_dim,
    )
    old_bias = flat[old_prefix + weights_len_old:].reshape(num_leaves, old_action_dim)
    new_weights = np.zeros((num_leaves, new_action_dim, new_input_dim), dtype=np.float32)
    new_bias = np.zeros((num_leaves, new_action_dim), dtype=np.float32)
    for new_idx, old_idx in enumerate(action_map):
        new_weights[:, new_idx, :old_input_dim] = old_weights[:, old_idx, :]
        new_bias[:, new_idx] = old_bias[:, old_idx] + float(action_bias_offsets[new_idx])
    new_weights_len = num_leaves * new_action_dim * new_input_dim
    new_flat[new_prefix:new_prefix + new_weights_len] = new_weights.reshape(-1)
    new_flat[new_prefix + new_weights_len:] = new_bias.reshape(-1)
    return new_flat


def _expand_echelon_params_with_alloc_targets(params, model, reference):
    """Embed older target soft trees into the 1+2K decoupled target mode."""
    expanded = _expand_init_params_for_model(
        params,
        model,
        reference,
        policy_action_mode="echelon_targets_with_alloc_targets",
    )
    return None if expanded is None else np.asarray(expanded, dtype=np.float32)


def _expand_init_params_for_model(params, model, reference, policy_action_mode):
    num_retailers = len(reference["retailer_lead_times"])
    new_input_dim = int(model.input_dim)
    new_action_dim = int(model.control_dim)
    normalized_input_dim = common.policy_state_input_dim(reference, "normalized")
    input_candidates = [new_input_dim]
    if normalized_input_dim != new_input_dim:
        input_candidates.append(normalized_input_dim)

    action_candidates = [(new_action_dim, list(range(new_action_dim)), None)]
    symmetric_min_values = None
    if str(model.leaf_type) == "linear" and common.is_symmetric_retailer_case(reference):
        symmetric_model = common.build_soft_tree_model(
            reference,
            depth=int(model.depth),
            temperature=float(model.temperature),
            split_type=str(model.split_type),
            leaf_type=str(model.leaf_type),
            policy_action_mode="symmetric_echelon_targets",
            policy_state_mode="normalized",
        )
        symmetric_min_values = [float(v) for v in symmetric_model.min_values]

    def symmetric_bias_offsets(action_map):
        if symmetric_min_values is None:
            return None
        new_min_values = [float(v) for v in model.min_values]
        return [
            symmetric_min_values[old_idx] - new_min_values[new_idx]
            for new_idx, old_idx in enumerate(action_map)
        ]

    if policy_action_mode == "echelon_targets":
        action_map = [0] + [1] * num_retailers
        action_candidates.append((2, action_map, symmetric_bias_offsets(action_map)))
    if policy_action_mode == "echelon_targets_with_alloc_targets":
        old_action_dim = num_retailers + 1
        if old_action_dim != new_action_dim:
            action_candidates.append(
                (
                    old_action_dim,
                    list(range(old_action_dim)) + list(range(1, old_action_dim)),
                    None,
                )
            )
        action_map = [0] + [1] * num_retailers + [1] * num_retailers
        action_candidates.append((2, action_map, symmetric_bias_offsets(action_map)))

    for old_input_dim in input_candidates:
        for old_action_dim, action_map, action_bias_offsets in action_candidates:
            expanded = _expand_soft_tree_params_to_model(
                params,
                model,
                old_input_dim=old_input_dim,
                old_action_dim=old_action_dim,
                action_map=action_map,
                action_bias_offsets=action_bias_offsets,
            )
            if expanded is not None and expanded.size == len(model.get_model_flat_params()):
                return expanded.tolist()
    return None


def _load_init_params(path, expected_size, model=None, reference=None, policy_action_mode=None):
    if path is None:
        return None
    params = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if params.size == int(expected_size):
        return params.tolist()
    if policy_action_mode is not None and model is not None and reference is not None:
        expanded = _expand_init_params_for_model(params, model, reference, policy_action_mode)
        if expanded is not None and len(expanded) == int(expected_size):
            return expanded
    if params.size != int(expected_size):
        raise ValueError(f"init_params_npy has {params.size} params, expected {expected_size}")
    return params.tolist()


def _direct_order_gate_init_flat_params(model, reference, gate_best):
    """Approximate the echelon-base-stock gate in direct_orders mode."""
    expected_dim = len(reference["retailer_lead_times"]) + 1
    if str(model.leaf_type) != "linear" or int(model.control_dim) != expected_dim:
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

    normalized_input_dim = common.policy_state_input_dim(reference, "normalized")
    total_position_idx = normalized_input_dim - 2
    bias[:, 0] = w_level
    weights[:, 0, total_position_idx] = -scale_proxy

    warehouse_lead_time = int(reference["warehouse_lead_time"])
    num_retailers = len(reference["retailer_lead_times"])
    retailer_inventory_start = 1 + warehouse_lead_time
    retailer_pipeline_start = retailer_inventory_start + num_retailers
    pipeline_idx = retailer_pipeline_start
    for retailer_idx, (target, lead_time) in enumerate(
        zip(r_levels, reference["retailer_lead_times"])
    ):
        action_idx = retailer_idx + 1
        bias[:, action_idx] = target
        weights[:, action_idx, retailer_inventory_start + retailer_idx] = -scale_proxy
        for offset in range(int(lead_time)):
            weights[:, action_idx, pipeline_idx + offset] = -scale_proxy
        pipeline_idx += int(lead_time)

    flat[weights_start:weights_start + weights_block] = weights.reshape(-1)
    flat[bias_start:] = bias.reshape(-1)
    return flat.tolist()
