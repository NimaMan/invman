"""
Single-policy autoresearch runner for the production_assembly_distribution_network
problem (Pirhooshyaran & Snyder 2021, arXiv:2006.05608) on the serial-shaped case3
network instance (the env's PRIMARY_REFERENCE_INSTANCE).

OBJECTIVE
---------
Train ONE soft-tree CMA-ES policy on the case3 network instance, evaluate its held-out
(paired common-random-number) mean per-period cost, and compare it to the env's OWN best
pairwise base-stock (grid-searched OUL levels). Append a TSV ledger row and dump a JSON
results artifact.

HONEST STATUS (read this first)
-------------------------------
This env is FAITHFUL to the Pirhooshyaran & Snyder (2021) network MDP (eq. 1-13, cost eq.
3, verified equation-by-equation in-crate) but is NOT literature-verified: there is NO
published optimum for THIS network env. The serial textbook optimum 47.65 is structurally
UNREACHABLE here (it is an ECHELON base-stock level applied to a LOCAL raw-position pairwise
policy -- a level-interpretation mismatch documented in the env README), and the serial
optimum's literature-verified home is the sibling `multi_echelon/serial` family, not here.
Only the single-node newsvendor row is literature-verified for this family.

Therefore the baseline here is a RESEARCH comparison, NOT a literature reproduction: the
env's own best pairwise base-stock, found by grid-searching the pairwise OUL levels on a
disjoint search block and re-scored on the held-out block. The headline is the signed gap
of the learned soft-tree vs that best pairwise base-stock under paired CRN at full reps.

ACTION DESIGN (the policy)
--------------------------
The case3 instance is a 3-node serial network: nodes 0->1->2, node 0 the only source.
Supply relations (env order = edges first, then external suppliers):
    relation 0 = edge(0->1), relation 1 = edge(1->2), relation 2 = external->node0.
The soft-tree rollout binding emits a DIRECT order quantity per supply relation
(vector_quantity action, action_dim = supply_relation_count = 3) clipped to [min,max].
This is the analogue of OWMR's weak `direct_orders` baseline: a CONSTANT leaf can only
emit a fixed order rate and cannot react to inventory, so it cannot express order-up-to
behavior. The lever is the LEAF CLASS, not the optimizer budget: a LINEAR leaf maps the
(scaled) policy-state features -- which INCLUDE per-relation raw inventory and per-relation
in-transit pipeline -- to the per-relation order, so it can express inventory-position
feedback (the q = level - max(IP,0) shape that base-stock targets), and oblique splits let
it switch behavior by inventory regime. The env owns its policy input dimension; we ask the
binding (input_dim = 30 for case3) rather than re-deriving it.

WARM START
----------
The soft-tree direct-quantity decoder is not analytically invertible into a base-stock
encoding (features are divided by a dynamic per-step scale), so we use honest
decoder-agnostic anchoring: seed the CMA mean at the steady-state FLOW rate (order the
demand mean, ~5 units, per relation each period), which is the simplest reproducible
known-good point. CMA-ES then refines OUTWARD from flow toward inventory-feedback
corrections. Generation 0 reproduces a sensible flow policy (~70/period), and the search
discovers the base-stock-beating regime.

ALGORITHM (per run)
-------------------
1. Build the case3 instance (verbatim from references.rs PRIMARY_REFERENCE_INSTANCE).
2. Build disjoint CRN demand-path blocks (search vs held-out) with a fixed sampler:
   demand only at the downstream node, N(5,1) rounded/clipped, T=10, undiscounted average
   per-period cost (matches the paper's average-cost comparison and the reproduction
   script).
3. Strongest in-env heuristic = best pairwise base-stock: grid-search the per-relation OUL
   levels on the search block (via the pairwise_base_stock policy rollout), re-score the
   argmin on the held-out block. THIS is the keep/discard gate.
4. Build the soft-tree (CLI structure), warm-start CMA-ES at the flow rate.
5. Score every candidate with the SAME Rust population-rollout binding
   (production_assembly_distribution_network_soft_tree_population_rollout) under paired CRN
   (one fresh seed block per generation, shared across the population).
6. Re-evaluate the best policy on the held-out block via the paths-based binding (CRN
   paired with the heuristic) and record learned mean cost, gap, gap%, winner.

CPU CAP (HARD)
--------------
The shared CPU helper caps Rayon/BLAS/OpenMP before NumPy and Rust imports. Parallelism is
rayon inside the population-rollout binding; there is no Python process pool. Several sibling
agents run in parallel, so this runner MUST stay capped.

USAGE (smoke)
-------------
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/production_assembly_distribution_network/autoresearch_production_assembly_distribution_network.py \
      --description "smoke" --budget smoke
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

configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

import numpy as np

from invman.cmaes import CMAES  # noqa: E402

import invman_rust as ir  # noqa: E402

# --------------------------------------------------------------------------------------
# case3 instance (verbatim from references.rs PRIMARY_REFERENCE_INSTANCE)
# --------------------------------------------------------------------------------------
NUM_NODES = 3
SOURCE_NODES = [True, False, False]
NODE_MODES = ["single", "single", "single"]
EXT_LEAD = [2, 0, 0]
EDGE_FROM = [0, 1]
EDGE_TO = [1, 2]
EDGE_LEAD = [1, 1]
HOLDING = [2.0, 4.0, 7.0]
BACKLOG = [0.0, 0.0, 37.12]
PERIODS = 10
DEMAND_MEAN = 5.0
DEMAND_STD = 1.0
DEMAND_KINDS = ["deterministic", "deterministic", "normal"]
DEMAND_P1 = [0.0, 0.0, DEMAND_MEAN]
DEMAND_P2 = [0.0, 0.0, DEMAND_STD]

# Initial state from the reference instance.
INIT_FINISHED = [10, 5, 5]
INIT_RAW = [0, 0, 0]
INIT_IBL = [0, 0]
INIT_EBL = [0, 0, 0]
INIT_PIPES = [[0], [0], [0, 0]]

# Action geometry: one direct order per supply relation (edges first, then external->0).
ACTION_DIM = 3                  # supply_relation_count(case3) = 2 edges + 1 source
INPUT_DIM = 30                  # env policy feature dim for case3 (asked, not re-derived)
MIN_VALUES = [0, 0, 0]
MAX_VALUES = [60, 60, 60]       # physical-ish cap well above the operating region (~5-25)
TEMPERATURE_DEFAULT = 0.25
DISCOUNT = 1.0                  # undiscounted average-cost comparison (matches the paper)

# Disjoint CRN blocks.
SEARCH_SEED = 500_000
HOLDOUT_SEED = 900_000

BUDGETS = {
    # smoke = end-to-end validation only (not a decision budget)
    "smoke": {"popsize": 8, "generations": 8, "train_batch": 32, "search_paths": 64, "holdout_paths": 256, "grid": "coarse"},
    "screening": {"popsize": 24, "generations": 40, "train_batch": 64, "search_paths": 256, "holdout_paths": 2000, "grid": "fine"},
    "full": {"popsize": 24, "generations": 60, "train_batch": 96, "search_paths": 256, "holdout_paths": 4000, "grid": "fine"},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Autoresearch single-policy loop for production_assembly_distribution_network (case3)."
    )
    p.add_argument("--run_tag", default="production_assembly_distribution_network_autoresearch")
    p.add_argument("--budget", choices=sorted(BUDGETS), default="screening")
    p.add_argument("--description", required=True)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    p.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    p.add_argument("--warm_start_flow", type=float, default=DEMAND_MEAN,
                   help="Seed CMA mean at this constant per-relation flow order (default = demand mean).")
    p.add_argument("--sigma_init", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def _git_short_commit(root: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                           check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return r.stdout.strip()


def make_paths(n: int, seed: int):
    """Demand only at the downstream node; N(5,1) rounded/clipped, T=10."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = np.clip(np.round(rng.normal(DEMAND_MEAN, DEMAND_STD, size=PERIODS)), 0, None).astype(int)
        out.append([[0, 0, int(d[t])] for t in range(PERIODS)])
    return out


def pairwise_base_stock_cost(oul_by_relation, paths):
    """Mean per-period cost of the pairwise base-stock policy on explicit demand paths."""
    costs = []
    levels = [float(x) for x in oul_by_relation]
    for realized in paths:
        total = ir.production_assembly_distribution_network_policy_rollout_from_paths(
            "pairwise_base_stock", levels,
            NUM_NODES, SOURCE_NODES, NODE_MODES, EXT_LEAD, EDGE_FROM, EDGE_TO, EDGE_LEAD,
            INIT_FINISHED, INIT_RAW, INIT_IBL, INIT_EBL, INIT_PIPES,
            realized, HOLDING, BACKLOG, DISCOUNT,
        )
        costs.append(total / PERIODS)
    arr = np.asarray(costs)
    return float(arr.mean()), float(arr.std() / math.sqrt(arr.size))


def search_best_pairwise_base_stock(search_paths, holdout_paths, grid: str):
    """Grid-search the per-relation OUL levels; re-score the argmin on the held-out block."""
    if grid == "coarse":
        g0, g1, g2 = range(5, 11, 2), range(5, 11, 2), range(8, 13, 2)
    else:
        g0, g1, g2 = range(4, 12), range(4, 12), range(7, 16)
    best = None
    for a in g0:
        for b in g1:
            for c in g2:
                m, _ = pairwise_base_stock_cost([a, b, c], search_paths)
                if best is None or m < best[1]:
                    best = ((a, b, c), m)
    levels, search_cost = best
    holdout_mean, holdout_se = pairwise_base_stock_cost(list(levels), holdout_paths)
    return {
        "oul_levels": [int(x) for x in levels],
        "search_mean_cost": float(search_cost),
        "holdout_mean_cost": float(holdout_mean),
        "holdout_stderr": float(holdout_se),
    }


def _flat_param_count(depth: int, leaf_type: str) -> int:
    num_internal = (1 << depth) - 1
    num_leaves = 1 << depth
    weights = num_internal * INPUT_DIM + num_internal
    if leaf_type == "constant":
        return weights + num_leaves * ACTION_DIM
    return weights + num_leaves * ACTION_DIM * INPUT_DIM + num_leaves * ACTION_DIM


def _warm_start_flow(depth: int, leaf_type: str, flow: float) -> np.ndarray:
    """Seed the CMA mean at a constant per-relation flow order (state-independent).

    constant / sigmoid_linear leaf: scaled = min + sigmoid(leaf_param)*(max-min)
        => leaf_param = logit((flow - min)/(max - min)); state weights (sigmoid_linear) set 0.
    linear leaf: scaled = min + softplus(bias + weights.state)
        => zero the leaf weights, set bias = softplus_inv(flow - min) so the leaf emits flow.
    Split params stay at zero (a balanced gate); CMA refines them.
    """
    n = _flat_param_count(depth, leaf_type)
    flat = np.zeros(n, dtype=np.float64)
    num_internal = (1 << depth) - 1
    num_leaves = 1 << depth
    bias_block = num_leaves * ACTION_DIM
    span = float(MAX_VALUES[0] - MIN_VALUES[0])
    if leaf_type == "constant":
        prob = float(min(max((flow - MIN_VALUES[0]) / span, 1e-4), 1.0 - 1e-4))
        logit = math.log(prob / (1.0 - prob))
        flat[n - bias_block:] = logit
        return flat
    if leaf_type == "sigmoid_linear":
        prob = float(min(max((flow - MIN_VALUES[0]) / span, 1e-4), 1.0 - 1e-4))
        logit = math.log(prob / (1.0 - prob))
        flat[n - bias_block:] = logit  # weights already 0 -> leaf == logit -> emits flow
        return flat
    # linear leaf
    bias_start = num_internal * INPUT_DIM + num_internal + num_leaves * ACTION_DIM * INPUT_DIM
    flat[bias_start:bias_start + bias_block] = math.log(math.expm1(max(flow - MIN_VALUES[0], 1e-6)))
    return flat


def population_costs(params_batch, depth, leaf_type, split_type, temperature, seeds):
    res = ir.production_assembly_distribution_network_soft_tree_population_rollout(
        params_batch, INPUT_DIM, depth, MIN_VALUES, MAX_VALUES, "vector_quantity",
        NUM_NODES, SOURCE_NODES, NODE_MODES, EXT_LEAD, EDGE_FROM, EDGE_TO, EDGE_LEAD,
        INIT_FINISHED, INIT_RAW, INIT_IBL, INIT_EBL, INIT_PIPES,
        PERIODS, DEMAND_KINDS, DEMAND_P1, HOLDING, BACKLOG,
        seeds, DEMAND_P2, DISCOUNT, temperature, split_type, leaf_type, None,
    )
    return np.asarray(res) / PERIODS


def soft_tree_cost_on_paths(flat, depth, leaf_type, split_type, temperature, paths):
    costs = []
    flat = list(flat)
    for realized in paths:
        total = ir.production_assembly_distribution_network_soft_tree_rollout_from_paths(
            flat, INPUT_DIM, depth, MIN_VALUES, MAX_VALUES, "vector_quantity",
            NUM_NODES, SOURCE_NODES, NODE_MODES, EXT_LEAD, EDGE_FROM, EDGE_TO, EDGE_LEAD,
            INIT_FINISHED, INIT_RAW, INIT_IBL, INIT_EBL, INIT_PIPES,
            realized, DEMAND_KINDS, DEMAND_P1, HOLDING, BACKLOG,
            DEMAND_P2, DISCOUNT, temperature, split_type, leaf_type, None,
        )
        costs.append(total / PERIODS)
    arr = np.asarray(costs)
    return float(arr.mean()), float(arr.std() / math.sqrt(arr.size))


def train_soft_tree(parsed, budget, holdout_paths):
    depth, leaf, split, temp = parsed.depth, parsed.leaf_type, parsed.split_type, parsed.temperature
    n = _flat_param_count(depth, leaf)
    x0 = _warm_start_flow(depth, leaf, parsed.warm_start_flow)

    gen0_mean, gen0_se = soft_tree_cost_on_paths(x0, depth, leaf, split, temp, holdout_paths)

    es = CMAES(num_params=n, sigma_init=parsed.sigma_init, popsize=budget["popsize"],
               seed=parsed.seed, x0=x0.tolist())
    rng = np.random.default_rng(parsed.seed + 1)
    best_flat = x0.copy()
    best_train = math.inf
    t0 = time.time()
    for _ in range(budget["generations"]):
        sols = es.ask()
        base = int(rng.integers(1, 10_000_000))
        seeds = list(range(base, base + budget["train_batch"]))  # paired CRN within the generation
        rewards = []
        for k in range(es.popsize):
            batch = [sols[k].astype(np.float32).tolist()] * budget["train_batch"]
            cost = float(population_costs(batch, depth, leaf, split, temp, seeds).mean())
            rewards.append(-cost)
        es.tell(rewards)
        gi = int(np.argmax(rewards))
        if -rewards[gi] < best_train:
            best_train = -rewards[gi]
            best_flat = sols[gi].copy()
    train_seconds = time.time() - t0

    holdout_mean, holdout_se = soft_tree_cost_on_paths(best_flat, depth, leaf, split, temp, holdout_paths)
    return {
        "flat_params": best_flat.astype(np.float32).tolist(),
        "n_params": int(n),
        "gen0_holdout_mean_cost": gen0_mean,
        "gen0_holdout_stderr": gen0_se,
        "holdout_mean_cost": holdout_mean,
        "holdout_stderr": holdout_se,
        "best_train_cost": float(best_train),
        "train_seconds": float(train_seconds),
    }


def run(parsed) -> dict:
    budget = BUDGETS[parsed.budget]
    search_paths = make_paths(budget["search_paths"], SEARCH_SEED)
    holdout_paths = make_paths(budget["holdout_paths"], HOLDOUT_SEED)

    heuristic = search_best_pairwise_base_stock(search_paths, holdout_paths, budget["grid"])
    base = heuristic["holdout_mean_cost"]

    learned = train_soft_tree(parsed, budget, holdout_paths)
    cost = learned["holdout_mean_cost"]

    gap = cost - base
    gap_pct = (cost / base - 1.0) * 100.0
    winner = "learned" if cost < base else "heuristic"

    return {
        "reference": "pirhooshyaran2021_serial_case3",
        "literature_verified": False,
        "baseline_kind": "env_own_best_pairwise_base_stock (RESEARCH comparison, NOT a literature optimum)",
        "policy_architecture": (
            f"soft_tree_d{parsed.depth}_{parsed.split_type}_{parsed.leaf_type}"
            f"_temp{parsed.temperature}_vector_quantity_warmstart_flow{parsed.warm_start_flow}"
        ),
        "best_pairwise_base_stock": heuristic,
        "learned": learned,
        "learned_cost": cost,
        "best_heuristic_cost": base,
        "gap": gap,
        "gap_pct": gap_pct,
        "winner": winner,
    }


def main():
    parsed = parse_args()
    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    root.mkdir(parents=True, exist_ok=True)
    results_tsv = root / "results.tsv"
    header = [
        "commit", "run_tag", "budget", "reference", "policy_architecture",
        "learned_cost", "learned_stderr", "best_pairwise_base_stock",
        "best_pairwise_oul", "gap", "gap_pct", "winner", "train_seconds", "description",
    ]
    if not results_tsv.exists():
        with results_tsv.open("w", newline="", encoding="utf-8") as h:
            csv.writer(h, delimiter="\t").writerow(header)

    out = run(parsed)
    commit = _git_short_commit(PACKAGE_ROOT)
    row = [
        commit, parsed.run_tag, parsed.budget, out["reference"], out["policy_architecture"],
        f"{out['learned_cost']:.6f}", f"{out['learned']['holdout_stderr']:.6f}",
        f"{out['best_heuristic_cost']:.6f}", str(out["best_pairwise_base_stock"]["oul_levels"]),
        f"{out['gap']:.6f}", f"{out['gap_pct']:.4f}", out["winner"],
        f"{out['learned']['train_seconds']:.1f}", parsed.description,
    ]
    with results_tsv.open("a", newline="", encoding="utf-8") as h:
        csv.writer(h, delimiter="\t").writerow(row)

    json_path = root / f"result_{parsed.budget}_{commit}_d{parsed.depth}_{parsed.leaf_type}.json"
    with json_path.open("w", encoding="utf-8") as h:
        json.dump({"ledger_row": dict(zip(header, row)), "detail": out}, h, indent=2)

    print(json.dumps({"ledger_tsv": str(results_tsv), "results_json": str(json_path),
                      "ledger_row": dict(zip(header, row))}, indent=2))


if __name__ == "__main__":
    main()
