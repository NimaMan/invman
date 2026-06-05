"""
Single-policy autoresearch runner for the production_assembly_distribution_network
problem on the PURE ASSEMBLY network of Pirhooshyaran & Snyder 2021 (arXiv:2006.05608),
Figure 6 (left, "assembly SCN (1)") / Supplement Table 3, instance 1.

OBJECTIVE
---------
Train ONE soft-tree CMA-ES policy on the assembly-SCN instance, evaluate its held-out
(paired common-random-number) mean per-period cost, and compare it to the env's OWN best
pairwise base-stock (grid-searched OUL levels). Append a TSV ledger row and dump a JSON
results artifact. This reuses EXACTLY the case-3 design + protocol (depth-2 oblique soft
tree, linear leaves, flow warm-start, popsize/generations/held-out eval); only the
topology constants are swapped.

HONEST STATUS (read this first)
-------------------------------
This env is FAITHFUL to the Pirhooshyaran & Snyder (2021) network MDP (eq. 1-13, cost eq.
3, verified equation-by-equation in-crate) but is NOT literature-verified: there is NO
published optimum for THIS network env. For assembly SCNs Pirhooshyaran reports only a
DNN / coordinate-descent / enumeration / DFO comparison (their Table 4, Supplement Table
3), NOT an analytical optimum. Therefore the baseline here is a RESEARCH comparison, NOT a
literature reproduction: the env's own best pairwise base-stock, grid-searched on a disjoint
search block and re-scored on the held-out block. The headline is the signed gap of the
learned soft-tree vs that best pairwise base-stock under paired CRN at full reps. It is
explicitly NOT a published-number beat.

TOPOLOGY (Pirhooshyaran & Snyder 2021, Figure 6 left / Supplement Table 3 instance 1)
-------------------------------------------------------------------------------------
7 nodes (env 0..6 == paper nodes 1..7). Sources {0,1,2,3} (paper 1..4) each fed by an
external supplier. Node 4 (paper 5) ASSEMBLY-AND from {0,1}; node 5 (paper 6) ASSEMBLY-AND
from {2,3}; node 6 (paper 7) ASSEMBLY-AND from {4,5}. External customer demand at node 6.
Internal edges (6): (0,4),(1,4),(2,5),(3,5),(4,6),(5,6). Supply relations (env order =
edges first, then external suppliers) = 6 edges + 4 external suppliers = 10 (= the paper's
"10 OUL decisions" for assembly SCN 1). ACTION_DIM = supply_relation_count = 10.

PARAMETER SOURCING (Supplement Table 3, Instance 1)
---------------------------------------------------
Per-node holding (H), shortage (S), lead time (LT), demand:
  nodes 1-4 (sources): H 0.25, S 0,  external LT 2
  nodes 5,6 (echelon-2 assembly): H 0.8, S 0,  edge LT 1
  node 7 (final assembly / customer): H 1.9, S 10, edge LT 1, demand N(13, 1.2^2)
The env charges holding PER NODE; Supplement Table 3 lists H per (predecessor,node)
relation but it is constant within each node, so the per-node map is exact: HOLDING =
[0.25,0.25,0.25,0.25, 0.8,0.8, 1.9]. Shortage 10 only at the customer node 6. External
supplier lead = 2 for the four source nodes; all internal edges lead 1. Demand N(13,1.2)
at node 6. Horizon T = 10 (matches the case-3 / paper finite-horizon protocol; the paper's
own cost in Table 4 is a separate 10x10000-period simulation, not used here -- this is the
env's own-best-heuristic gate, not a published-number reproduction).

ACTION DESIGN / WARM START / ALGORITHM / CPU CAP
------------------------------------------------
Identical to case 3. vector_quantity action of dimension ACTION_DIM = 10; depth-2 oblique
soft tree with linear leaves; flow warm-start; paired-CRN CMA-ES; held-out re-eval.
INPUT_DIM = 76 for this topology (7*7 nodes + 2*10 relations + 6 edges + 1), asked of the
binding, not re-derived. CPU capped before NumPy/Rust import (<=3 workers).
USAGE (smoke):
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/production_assembly_distribution_network/autoresearch_pure_assembly_network.py \
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
# PURE ASSEMBLY SCN (1) instance 1 (Pirhooshyaran & Snyder 2021, Fig 6 left / Supp Table 3)
# --------------------------------------------------------------------------------------
NUM_NODES = 7
SOURCE_NODES = [True, True, True, True, False, False, False]
NODE_MODES = ["single", "single", "single", "single", "assembly_and", "assembly_and", "assembly_and"]
EXT_LEAD = [2, 2, 2, 2, 0, 0, 0]                 # source external supplier lead = 2 (Supp Table 3)
EDGE_FROM = [0, 1, 2, 3, 4, 5]
EDGE_TO = [4, 4, 5, 5, 6, 6]
EDGE_LEAD = [1, 1, 1, 1, 1, 1]
HOLDING = [0.25, 0.25, 0.25, 0.25, 0.8, 0.8, 1.9]
BACKLOG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]   # shortage 10 at the customer node 6
PERIODS = 10
DEMAND_MEAN = 13.0
DEMAND_STD = 1.2
DEMAND_KINDS = ["deterministic", "deterministic", "deterministic", "deterministic",
                "deterministic", "deterministic", "normal"]
DEMAND_P1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, DEMAND_MEAN]
DEMAND_P2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, DEMAND_STD]

# Supply relations (env order): 6 edges, then 4 external suppliers. ACTION_DIM = 10.
ACTION_DIM = 10
INPUT_DIM = 76                  # env policy feature dim for assembly SCN (asked, not re-derived)

# Initial state: zero finished/raw, warm the pipelines per relation lead time.
INIT_FINISHED = [0] * NUM_NODES
INIT_RAW = [0] * ACTION_DIM
INIT_IBL = [0] * len(EDGE_FROM)
INIT_EBL = [0] * NUM_NODES
_PIPE_LEADS = EDGE_LEAD + [EXT_LEAD[0], EXT_LEAD[1], EXT_LEAD[2], EXT_LEAD[3]]
INIT_PIPES = [[0] * lead for lead in _PIPE_LEADS]

MIN_VALUES = [0] * ACTION_DIM
MAX_VALUES = [80] * ACTION_DIM   # physical-ish cap above the operating region (demand mean ~13)
TEMPERATURE_DEFAULT = 0.25
DISCOUNT = 1.0

SEARCH_SEED = 500_000
HOLDOUT_SEED = 900_000

BUDGETS = {
    "smoke": {"popsize": 8, "generations": 8, "train_batch": 32, "search_paths": 64, "holdout_paths": 256, "grid": "coarse"},
    "screening": {"popsize": 24, "generations": 40, "train_batch": 64, "search_paths": 256, "holdout_paths": 2000, "grid": "fine"},
    "full": {"popsize": 24, "generations": 60, "train_batch": 96, "search_paths": 256, "holdout_paths": 4000, "grid": "fine"},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Autoresearch single-policy loop for the pure assembly SCN (1) instance 1."
    )
    p.add_argument("--run_tag", default="pure_assembly_network_autoresearch")
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
    """Demand only at the final assembly node 6; N(13,1.2) rounded/clipped, T=10."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = np.clip(np.round(rng.normal(DEMAND_MEAN, DEMAND_STD, size=PERIODS)), 0, None).astype(int)
        out.append([[0, 0, 0, 0, 0, 0, int(d[t])] for t in range(PERIODS)])
    return out


def pairwise_base_stock_cost(oul_by_relation, paths):
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
    """Grid-search per-echelon OUL levels (tied within echelon by network symmetry, the
    same restriction Pirhooshyaran applies to its non-DNN benchmarks); re-score the argmin
    on the held-out block.
    Relation order = [ (0,4),(1,4),(2,5),(3,5),(4,6),(5,6),
                       ext->0, ext->1, ext->2, ext->3 ].
    echelon-1 = the 4 external supplier relations (sources, demand mean 13);
    echelon-2 = the 4 edges into assembly nodes 4,5;
    echelon-3 = the 2 edges into the customer node 6.
    """
    # Ranges cover the true heuristic basin (echelon argmin ~ (52,52,52), found by a
    # broad pre-sweep): the three-layer assembly-and min() production starves downstream,
    # so the pairwise base-stock needs high OUL at every relation. The grid MUST bracket
    # the argmin so the gate is the genuine env-own best, not a boundary artifact.
    if grid == "coarse":
        g1, g2, g3 = range(44, 61, 4), range(44, 61, 4), range(44, 61, 4)
    else:
        g1, g2, g3 = range(44, 61, 2), range(44, 61, 2), range(44, 61, 2)
    best = None
    for a in g1:           # external supplier level (sources)
        for b in g2:       # echelon-2 edge level
            for c in g3:   # echelon-3 (customer) edge level
                # edges: (0,4),(1,4),(2,5),(3,5),(4,6),(5,6) then ext->0..3
                levels = [b, b, b, b, c, c, a, a, a, a]
                m, _ = pairwise_base_stock_cost(levels, search_paths)
                if best is None or m < best[1]:
                    best = (tuple(levels), m, (a, b, c))
    levels, search_cost, echelon = best
    holdout_mean, holdout_se = pairwise_base_stock_cost(list(levels), holdout_paths)
    return {
        "oul_levels": [int(x) for x in levels],
        "echelon_levels": [int(x) for x in echelon],
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
    n = _flat_param_count(depth, leaf_type)
    flat = np.zeros(n, dtype=np.float64)
    num_internal = (1 << depth) - 1
    num_leaves = 1 << depth
    bias_block = num_leaves * ACTION_DIM
    span = float(MAX_VALUES[0] - MIN_VALUES[0])
    if leaf_type in ("constant", "sigmoid_linear"):
        prob = float(min(max((flow - MIN_VALUES[0]) / span, 1e-4), 1.0 - 1e-4))
        logit = math.log(prob / (1.0 - prob))
        flat[n - bias_block:] = logit
        return flat
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
        "reference": "pirhooshyaran2021_assembly_scn1_instance1_supp_table3",
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
