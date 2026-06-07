"""
Single-policy autoresearch runner for the production_assembly_distribution_network
problem on the MIXED (distribution + assembly-and) network of Pirhooshyaran & Snyder
2021 (arXiv:2006.05608), Figure 1 / Table 5.

OBJECTIVE
---------
Train ONE soft-tree CMA-ES policy on the mixed-SCN instance, evaluate its held-out
(paired common-random-number) mean per-period cost, and compare it to the env's OWN best
pairwise base-stock (grid-searched OUL levels). Append a TSV ledger row and dump a JSON
results artifact. This reuses EXACTLY the case-3 design + protocol (depth-2 oblique soft
tree, linear leaves, flow warm-start, popsize/generations/held-out eval); only the
topology constants are swapped.

HONEST STATUS (read this first)
-------------------------------
This env is FAITHFUL to the Pirhooshyaran & Snyder (2021) network MDP (eq. 1-13, cost eq.
3, verified equation-by-equation in-crate) but is NOT literature-verified: there is NO
published optimum for THIS network env. Pirhooshyaran reports for the mixed SCN only a
randomized-search / DFO / Spearmint / DNN comparison (their Tables 6-7), NOT an analytical
optimum. Therefore the baseline here is a RESEARCH comparison, NOT a literature
reproduction: the env's own best pairwise base-stock, found by grid-searching the pairwise
OUL levels on a disjoint search block and re-scored on the held-out block. The headline is
the signed gap of the learned soft-tree vs that best pairwise base-stock under paired CRN
at full reps. It is explicitly NOT a published-number beat.

TOPOLOGY (Pirhooshyaran & Snyder 2021, Figure 1 / Table 5)
----------------------------------------------------------
6 nodes (0..5). Node 0 is the sole external source; node 0 -> node 1; node 1 DISTRIBUTES
to {2,3} (proportional allocation, a `single` node with two outgoing edges); nodes 4 and 5
are ASSEMBLY-AND nodes each fed by BOTH 2 and 3. Two external customers N(5,1) at nodes 4
and 5. Internal edges (7): (0,1),(1,2),(1,3),(2,4),(2,5),(3,4),(3,5). Supply relations
(env order = edges first, then external suppliers) = 7 edges + 1 external->node0 = 8.
The paper optimizes the 7 edge OULs; the env additionally carries the external->node0
relation, so ACTION_DIM = supply_relation_count = 8.

PARAMETER SOURCING (Table 5)
----------------------------
Table 5 lists per-edge shipment lead time, holding cost, and shortage cost by echelon:
  echelon 1: edge (0,1)  holding 2,   shortage 0,      shipment lead 2
  echelon 2: edges (1,2),(1,3) holding 4, shortage 0,  shipment lead 1
  echelon 3: edges (2,4),(2,5),(3,4),(3,5) holding 7, shortage 37.12, shipment lead 1
The env charges holding PER NODE; we map the per-edge holding to the destination node
(node j inherits the rate of items arriving at j): node 1 <- 2, nodes 2,3 <- 4, nodes
4,5 <- 7. The source node 0 carries the echelon-1 rate 2 (consistent with serial case 3,
where the external-fed source carries the smallest/upstream rate). Shortage is charged
where Table 5 places 37.12 -- at the customer/assembly nodes 4 and 5. The external->node0
relation reuses the echelon-1 shipment lead time (2), the only upstream lead Table 5 gives.
Initial finished inventory is Table 5's "Initialization" column mapped to the destination
node: node 1 = 40, nodes 2,3 = 10, nodes 4,5 = 5; the source node 0 is warm-started at 40
(same as node 1) -- initialization is not a published optimum and does not affect the
research framing. Demand N(5,1) at nodes 4 and 5 (Table 5 text), horizon T = 10.

ACTION DESIGN (the policy)
--------------------------
Identical to case 3. The soft-tree rollout binding emits a DIRECT order quantity per
supply relation (vector_quantity action, ACTION_DIM = supply_relation_count = 8) clipped
to [min,max]. A LINEAR leaf maps the (scaled) policy-state features -- which include
per-relation raw inventory, per-relation in-transit pipeline, per-node finished inventory,
backlog -- to the per-relation order, so it can express inventory-position feedback; oblique
splits let it switch behaviour by inventory regime. INPUT_DIM is asked of the binding
(input_dim = 66 for the mixed SCN: 7*6 nodes + 2*8 relations + 7 edges + 1), not re-derived.

WARM START / ALGORITHM / CPU CAP / USAGE
----------------------------------------
Identical to the case-3 runner: seed the CMA mean at a per-relation flow order, score every
candidate with the SAME Rust population-rollout binding under paired CRN, re-evaluate the
best policy on the held-out block. CPU is capped before NumPy/Rust import (<=3 workers).
USAGE (smoke):
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/production_assembly_distribution_network/autoresearch_mixed_distribution_assembly_network.py \
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
# MIXED SCN instance (Pirhooshyaran & Snyder 2021, Figure 1 / Table 5)
# --------------------------------------------------------------------------------------
NUM_NODES = 6
SOURCE_NODES = [True, False, False, False, False, False]
NODE_MODES = ["single", "single", "single", "single", "assembly_and", "assembly_and"]
EXT_LEAD = [2, 0, 0, 0, 0, 0]              # external->node0 shipment lead = echelon-1 lead (Table 5)
EDGE_FROM = [0, 1, 1, 2, 2, 3, 3]
EDGE_TO = [1, 2, 3, 4, 5, 4, 5]
EDGE_LEAD = [2, 1, 1, 1, 1, 1, 1]          # Table 5 shipment lead by echelon
HOLDING = [2.0, 2.0, 4.0, 4.0, 7.0, 7.0]   # per-node, mapped from Table 5 per-edge holding
BACKLOG = [0.0, 0.0, 0.0, 0.0, 37.12, 37.12]  # shortage 37.12 at customer/assembly nodes 4,5
PERIODS = 10
DEMAND_MEAN = 5.0
DEMAND_STD = 1.0
DEMAND_KINDS = ["deterministic", "deterministic", "deterministic", "deterministic", "normal", "normal"]
DEMAND_P1 = [0.0, 0.0, 0.0, 0.0, DEMAND_MEAN, DEMAND_MEAN]
DEMAND_P2 = [0.0, 0.0, 0.0, 0.0, DEMAND_STD, DEMAND_STD]

# Supply relations (env order): 7 edges, then external->node0. ACTION_DIM = 8.
ACTION_DIM = 8
INPUT_DIM = 66                  # env policy feature dim for the mixed SCN (asked, not re-derived)

# Initial state. Finished inventory = Table 5 "Initialization" by destination node
# (node1=40, nodes2,3=10, nodes4,5=5); source node0 warm-started at 40.
INIT_FINISHED = [40, 40, 10, 10, 5, 5]
INIT_RAW = [0] * ACTION_DIM
INIT_IBL = [0] * len(EDGE_FROM)
INIT_EBL = [0] * NUM_NODES
# pipelines per relation (edges first then external->0); lengths = lead times.
_PIPE_LEADS = EDGE_LEAD + [EXT_LEAD[0]]
INIT_PIPES = [[0] * lead for lead in _PIPE_LEADS]

MIN_VALUES = [0] * ACTION_DIM
MAX_VALUES = [60] * ACTION_DIM   # physical-ish cap well above the operating region
TEMPERATURE_DEFAULT = 0.25
DISCOUNT = 1.0                   # undiscounted average-cost comparison (matches the paper)

# Disjoint CRN blocks.
SEARCH_SEED = 500_000
HOLDOUT_SEED = 900_000

BUDGETS = {
    "smoke": {"popsize": 8, "generations": 8, "train_batch": 32, "search_paths": 64, "holdout_paths": 256, "grid": "coarse"},
    "screening": {"popsize": 24, "generations": 40, "train_batch": 64, "search_paths": 256, "holdout_paths": 2000, "grid": "fine"},
    "full": {"popsize": 24, "generations": 60, "train_batch": 96, "search_paths": 256, "holdout_paths": 4000, "grid": "fine"},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Autoresearch single-policy loop for the mixed distribution+assembly SCN."
    )
    p.add_argument("--run_tag", default="mixed_distribution_assembly_network_autoresearch")
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
    """Independent N(5,1) demand at both customer nodes (4 and 5); rounded/clipped, T=10."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d4 = np.clip(np.round(rng.normal(DEMAND_MEAN, DEMAND_STD, size=PERIODS)), 0, None).astype(int)
        d5 = np.clip(np.round(rng.normal(DEMAND_MEAN, DEMAND_STD, size=PERIODS)), 0, None).astype(int)
        out.append([[0, 0, 0, 0, int(d4[t]), int(d5[t])] for t in range(PERIODS)])
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
    """Grid-search per-echelon OUL levels (tied within echelon by network symmetry, the
    same restriction Pirhooshyaran applies to its non-DNN benchmarks); re-score the argmin
    on the held-out block. Echelon grids:
      e1 (relations edge(0,1) and external->node0) ~ feeds 2 downstream demands;
      e2 (edges (1,2),(1,3)) ~ one downstream demand each;
      e3 (edges (2,4),(2,5),(3,4),(3,5)) ~ the customer relations.
    Relation order = [ (0,1),(1,2),(1,3),(2,4),(2,5),(3,4),(3,5), ext->0 ].
    """
    # Ranges cover the true heuristic basin (echelon argmin ~ (36,13,7), found by a
    # broad pre-sweep): the assembly-and min() + distribution allocation force high
    # upstream OUL. The grid MUST bracket the argmin so the gate is the genuine env-own
    # best, not a boundary artifact.
    if grid == "coarse":
        g1, g2, g3 = range(28, 45, 4), range(8, 19, 3), range(4, 11, 2)
    else:
        g1, g2, g3 = range(28, 45, 2), range(8, 19, 1), range(4, 11, 1)
    best = None
    for a in g1:           # echelon-1 level (edge 0->1 and external->0)
        for b in g2:       # echelon-2 level (edges 1->2, 1->3)
            for c in g3:   # echelon-3 level (edges 2->4,2->5,3->4,3->5)
                levels = [a, b, b, c, c, c, c, a]
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


def population_costs(
    params_batch,
    depth,
    leaf_type,
    split_type,
    temperature,
    seeds,
    action_mode="vector_quantity",
    backbone_levels=None,
    residual_group_of=None,
):
    """Mean per-period population costs via the Rust population rollout.

    ADDITIVE action-head kwargs (defaults reproduce the original vector_quantity call
    byte-identically -- backbone_levels/residual_group_of=None means the bindings build
    a plain vector_quantity action spec with no backbone, exactly as before):
      action_mode        "vector_quantity" (direct per-relation order) or
                         "residual_base_stock" (order = clamp(gate_order + round(Delta))).
      backbone_levels    gate OUL per supply relation; REQUIRED by the Rust residual head
                         when action_mode == "residual_base_stock".
      residual_group_of  optional per-relation group id to TIE the residual within a group
                         (e.g. per-echelon); averaging zeros is zero so the gate anchor at
                         the all-zero warm start is preserved.
    """
    res = ir.production_assembly_distribution_network_soft_tree_population_rollout(
        params_batch, INPUT_DIM, depth, MIN_VALUES, MAX_VALUES, action_mode,
        NUM_NODES, SOURCE_NODES, NODE_MODES, EXT_LEAD, EDGE_FROM, EDGE_TO, EDGE_LEAD,
        INIT_FINISHED, INIT_RAW, INIT_IBL, INIT_EBL, INIT_PIPES,
        PERIODS, DEMAND_KINDS, DEMAND_P1, HOLDING, BACKLOG,
        seeds, DEMAND_P2, DISCOUNT, temperature, split_type, leaf_type, None,
        backbone_levels, residual_group_of,
    )
    return np.asarray(res) / PERIODS


def soft_tree_cost_on_paths(
    flat,
    depth,
    leaf_type,
    split_type,
    temperature,
    paths,
    action_mode="vector_quantity",
    backbone_levels=None,
    residual_group_of=None,
):
    """Mean / stderr per-period held-out cost of `flat` on explicit demand paths.

    ADDITIVE action-head kwargs (see population_costs); defaults reproduce the original
    vector_quantity call byte-identically. For action_mode == "residual_base_stock" with
    the all-zero `flat`, this returns the gate cost EXACTLY (gen-0 == gate, verified
    in-crate), provided backbone_levels is the searched gate OUL per relation.
    """
    costs = []
    flat = list(flat)
    for realized in paths:
        total = ir.production_assembly_distribution_network_soft_tree_rollout_from_paths(
            flat, INPUT_DIM, depth, MIN_VALUES, MAX_VALUES, action_mode,
            NUM_NODES, SOURCE_NODES, NODE_MODES, EXT_LEAD, EDGE_FROM, EDGE_TO, EDGE_LEAD,
            INIT_FINISHED, INIT_RAW, INIT_IBL, INIT_EBL, INIT_PIPES,
            realized, DEMAND_KINDS, DEMAND_P1, HOLDING, BACKLOG,
            DEMAND_P2, DISCOUNT, temperature, split_type, leaf_type, None,
            backbone_levels, residual_group_of,
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
        "reference": "pirhooshyaran2021_mixed_scn_fig1_table5",
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
