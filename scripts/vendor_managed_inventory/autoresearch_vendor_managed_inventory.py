"""
Single-policy autoresearch runner for the vendor-managed-inventory (VMI) benchmark.

OBJECTIVE
    Train ONE soft-tree CMA-ES policy with a CLI-selected structure on a NAMED
    reduced-single-retailer VMI instance, evaluate its held-out CRN discounted
    cost + optimality gap vs the STRONGEST heuristic (tuned retailer/DC
    base-stock), and APPEND a TSV ledger row (cost, best_heuristic, gap, gap%).

    This is the autoresearch counterpart to
    scripts/dual_sourcing/autoresearch_dual_sourcing.py and
    scripts/multi_echelon/autoresearch_multi_echelon.py. It exists because the
    learned soft tree currently LOSES (or marginally ties) to the tuned
    base-stock heuristic on ~4/5 reduced single-retailer instances (see
    policy_search/programs/vendor_managed_inventory/README.md). The loop's job is to find
    a structure + warm-start that BEATS the base-stock heuristic on the losing
    instances.

REUSE
    All the env wiring, the instance set, the heuristic tuning, the held-out CRN
    seed protocol, and the summarize() helper are imported from the existing
    learned-benchmark script:
        scripts/vendor_managed_inventory/benchmark_reduced_single_retailer.py
    This runner only ADDS:
      - structure-aware soft-tree rollout wrappers (so --tree_split_type /
        --tree_leaf_type / --tree_temperature / --tree_depth / --action bounds
        flow through to the binding, which the benchmark helper hardcodes), and
      - a CMA-ES warm-start at the tuned base-stock control.

ALGORITHM
    1. Resolve the named instance from benchmark_reduced_single_retailer.INSTANCE_SET.
    2. Tune both base-stock heuristics on a TRAIN seed (grid search, reused
       tune_* helpers); the strongest = min(retailer_base_stock,
       dc_reserve_base_stock). This is the keep/discard target.
    3. Build the soft-tree param vector for the CLI structure (depth, leaf_type)
       and, if --warm_start base_stock, seed the CMA mean so the leaf outputs
       start at the tuned base-stock's per-period shipment target.
    4. Train with CMA-ES through
       vendor_managed_inventory_soft_tree_population_rollout (population rollout,
       averaged over TRAIN seeds under CRN), with the CLI structure flags.
    5. Score the trained tree and the heuristics on a disjoint HELD-OUT CRN seed
       block; compute gap = learned - best_heuristic and gap% .
    6. Append a TSV ledger row to outputs/autoresearch/<run_tag>/results.tsv.

CPU CAP
    Route everything through the binding with the shared CPU helper set before NumPy
    and Rust imports. This runner spawns NO worker processes of its own.

USAGE
    RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python \
        scripts/vendor_managed_inventory/autoresearch_vendor_managed_inventory.py \
        --description "smoke" --budget screening --instance high_penalty \
        --tree_leaf_type linear --warm_start base_stock
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
# Make the sibling benchmark module importable as a plain module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from invman.cpu_limits import configure_process_cpu_limits_from_argv

configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

import numpy as np

import invman_rust as ir

# Reuse the learned-benchmark helpers: instance definitions, heuristic tuning,
# the held-out CRN protocol, and the summarizer.
import benchmark_reduced_single_retailer as bench
from benchmark_reduced_single_retailer import (  # noqa: F401  (re-exported intent)
    Instance,
    INSTANCE_SET,
    PRIMARY,
    heuristic_held_out_samples,
    summarize,
    tune_dc_reserve_base_stock,
    tune_retailer_base_stock,
    INPUT_DIM,
    ACTION_DIM,
)


# ---------------------------------------------------------------------------
# Budgets. `full` mirrors the protocol that produced the README losing-margins
# table, so a full run is directly comparable to it; `screening` rejects fast.
# ---------------------------------------------------------------------------
BUDGETS = {
    "screening": {
        "train_seeds": 16,
        "held_out_seeds": 12,
        "soft_tree_eval_seeds": 400,
        "heuristic_reps": 400,
        "popsize": 12,
        "iters": 40,
    },
    "full": {
        "train_seeds": 64,
        "held_out_seeds": 32,
        "soft_tree_eval_seeds": 4000,
        "heuristic_reps": 1500,
        "popsize": 24,
        "iters": 200,
    },
}


# ---------------------------------------------------------------------------
# Structure-aware soft-tree param count. The benchmark helper's
# soft_tree_param_count() only covers the constant leaf; the binding also
# supports linear leaves (different param layout, see
# core::policies::soft_tree::validate_soft_tree_flat_params).
# ---------------------------------------------------------------------------
def soft_tree_param_count(depth: int, leaf_type: str) -> int:
    ni = (1 << depth) - 1
    nl = 1 << depth
    split = ni * INPUT_DIM + ni
    if leaf_type == "constant":
        leaf = nl * ACTION_DIM
    elif leaf_type in ("linear", "sigmoid_linear"):
        leaf = nl * ACTION_DIM * INPUT_DIM + nl * ACTION_DIM
    else:
        raise ValueError(f"unknown leaf_type {leaf_type!r}")
    return split + leaf


# ---------------------------------------------------------------------------
# Structure-aware rollout wrappers. Same binding the benchmark helper calls, but
# with the split_type / leaf_type / temperature / action bounds passed through
# from the CLI instead of hardcoded to oblique/constant/0.25.
# ---------------------------------------------------------------------------
def _rollout(inst: Instance, tiled_params, tiled_seeds, depth, action_min,
             action_max, temperature, split_type, leaf_type):
    return ir.vendor_managed_inventory_soft_tree_population_rollout(
        tiled_params, INPUT_DIM, depth,
        [action_min], [action_max], "scalar_quantity",
        inst.initial_dc_on_hand, inst.initial_retailer_on_hand,
        inst.initial_retailer_pipeline,
        inst.periods, "poisson", inst.demand_mean,
        inst.dc_replenishment_quantity, inst.dc_capacity,
        inst.shipment_cost_per_unit, inst.dc_holding_cost_per_unit,
        inst.retailer_holding_cost_per_unit, inst.stockout_cost_per_unit,
        inst.salvage_value_per_unit, inst.max_shipment_quantity,
        tiled_seeds, inst.discount_factor, temperature,
        split_type, leaf_type, None,
    )


def eval_population(inst, batch, depth, seeds, action_min, action_max,
                    temperature, split_type, leaf_type):
    """Mean over `seeds` for each param vector in `batch` (CRN per param)."""
    tiled_params, tiled_seeds = [], []
    for p in batch:
        for s in seeds:
            tiled_params.append(p)
            tiled_seeds.append(int(s))
    costs = _rollout(inst, tiled_params, tiled_seeds, depth, action_min,
                     action_max, temperature, split_type, leaf_type)
    n = len(seeds)
    return [sum(costs[i * n:(i + 1) * n]) / n for i in range(len(batch))]


def held_out_samples(inst, params, depth, seeds, action_min, action_max,
                     temperature, split_type, leaf_type):
    """One single-path discounted cost per held-out seed for `params`."""
    costs = _rollout(inst, [params] * len(seeds), [int(s) for s in seeds],
                     depth, action_min, action_max, temperature, split_type,
                     leaf_type)
    return list(costs)


# ---------------------------------------------------------------------------
# CMA-ES warm-start at the base-stock control. We seed the CMA mean so the
# tree's leaf outputs start at the per-period shipment a base-stock policy of
# the tuned level would issue (clamped to the shipment cap), instead of the
# all-zeros mean. Split weights/biases start at 0 (a single averaging path);
# the leaf outputs carry the base-stock anchor. The leaf->action maps are:
#   constant leaf: action = min + sigmoid(leaf) * (max - min)
#   linear  leaf : action = min + softplus(bias)         (weights start 0)
# so we invert those to place the anchor shipment q* at the leaf params.
# ---------------------------------------------------------------------------
def warm_start_mean(depth, leaf_type, anchor_shipment, action_min, action_max):
    n = soft_tree_param_count(depth, leaf_type)
    x0 = np.zeros(n, dtype=float)
    ni = (1 << depth) - 1
    nl = 1 << depth
    bias_end = ni * INPUT_DIM + ni  # split weights + split biases (all 0)
    q = float(min(max(anchor_shipment, action_min + 1e-3), action_max - 1e-3))
    if leaf_type == "constant":
        span = max(action_max - action_min, 1e-6)
        frac = min(max((q - action_min) / span, 1e-4), 1.0 - 1e-4)
        leaf_logit = math.log(frac / (1.0 - frac))  # inverse sigmoid
        for leaf_idx in range(nl):
            x0[bias_end + leaf_idx * ACTION_DIM] = leaf_logit
    else:  # linear / sigmoid_linear: leaf weights stay 0, set the bias term
        # softplus(raw) = q  =>  raw = log(exp(q) - 1); guard for large q.
        target = max(q - action_min, 1e-4)
        raw = target if target > 20 else math.log(math.expm1(target) + 1e-12)
        weights_block = nl * ACTION_DIM * INPUT_DIM
        bias_start = bias_end + weights_block
        for leaf_idx in range(nl):
            x0[bias_start + leaf_idx * ACTION_DIM] = raw
    return list(x0)


def train_soft_tree(inst, depth, train_seeds, popsize, iters, sigma0, rng_seed,
                    action_min, action_max, temperature, split_type, leaf_type,
                    x0, return_xfavorite=False):
    """Local CMA-ES train. Tracks xbest (argmin generation-best fitness, the
    historically deployed endpoint).

    ADDITIVE/REVERSIBLE (training-path audit 2026-06-06): when
    `return_xfavorite=True`, ALSO return the CMA-ES distribution MEAN
    (es.result.xfavorite == es.mean), the local-train analog of
    invman.es_mp.train's es.current_param() / result[5]. xbest overfits the
    small CRN train-seed batch; xfavorite is the less-overfit distribution
    centre. Returning it lets the caller add it to an honest best-of floor.
    The default (return_xfavorite=False) reproduces the prior 2-tuple return
    EXACTLY, so all existing callers are untouched.
    """
    import cma

    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {"popsize": popsize, "seed": rng_seed, "verbose": -9, "maxiter": iters},
    )
    best_x, best_f = None, float("inf")
    while not es.stop():
        solutions = es.ask()
        batch = [[float(v) for v in s] for s in solutions]
        fitness = eval_population(inst, batch, depth, train_seeds, action_min,
                                  action_max, temperature, split_type, leaf_type)
        es.tell(solutions, fitness)
        i = int(np.argmin(fitness))
        if fitness[i] < best_f:
            best_f, best_x = fitness[i], batch[i]
    if return_xfavorite:
        xfavorite = [float(v) for v in es.result.xfavorite]
        return best_x, best_f, xfavorite
    return best_x, best_f


# ---------------------------------------------------------------------------
def resolve_instance(name: str) -> Instance:
    for inst in INSTANCE_SET:
        if inst.name == name:
            return inst
    raise SystemExit(
        f"unknown instance {name!r}; choices: {[i.name for i in INSTANCE_SET]}"
    )


def _git_short_commit(project_root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_args():
    p = argparse.ArgumentParser(
        description="Autoresearch-style single-policy loop for the VMI benchmark.")
    p.add_argument("--run_tag", default="vmi_autoresearch")
    p.add_argument("--description", required=True)
    p.add_argument("--budget", choices=sorted(BUDGETS), default="screening")
    # Default to the WIDEST current loss (high_penalty, -2.40%): clearest signal.
    p.add_argument("--instance", default="high_penalty",
                   help="name from benchmark_reduced_single_retailer.INSTANCE_SET")
    p.add_argument("--tree_depth", type=int, default=2)
    p.add_argument("--tree_temperature", type=float, default=0.1)
    p.add_argument("--tree_split_type", choices=["oblique", "axis_aligned"],
                   default="oblique")
    p.add_argument("--tree_leaf_type", choices=["constant", "linear"],
                   default="linear")
    p.add_argument("--action_min", type=int, default=0,
                   help="lower shipment bound the leaf output is squashed into")
    p.add_argument("--action_max", type=int, default=None,
                   help="upper shipment bound (defaults to instance max_shipment_quantity)")
    p.add_argument("--warm_start", choices=["base_stock", "zero"],
                   default="base_stock",
                   help="seed the CMA mean at the tuned base-stock control, or at zero")
    # ADDITIVE honest-floor selector (training-path audit 2026-06-06), mirrors the
    # OWMR run_asymmetric_learned_vs_gate.py --deploy_endpoint flag:
    #   floor (default) -> deploy best-of {xbest, xfavorite, warm-start anchor}
    #                      on the held-out block (downside-safe vs xbest).
    #   xbest           -> deploy ONLY xbest (reproduces the historical ledger row).
    #   xfavorite       -> deploy ONLY the CMA distribution-mean endpoint.
    p.add_argument("--deploy_endpoint", choices=["floor", "xbest", "xfavorite"],
                   default="floor",
                   help="which trained endpoint(s) are deployable; floor = best-of "
                        "{xbest, xfavorite, warm-start anchor} on held-out (downside-safe)")
    p.add_argument("--sigma_init", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=12345)
    # Budgets allow explicit overrides for a tiny smoke run.
    p.add_argument("--popsize", type=int, default=None)
    p.add_argument("--iters", type=int, default=None)
    return p.parse_args()


def main():
    a = parse_args()
    inst = resolve_instance(a.instance)
    b = dict(BUDGETS[a.budget])
    if a.popsize is not None:
        b["popsize"] = a.popsize
    if a.iters is not None:
        b["iters"] = a.iters
    action_max = inst.max_shipment_quantity if a.action_max is None else a.action_max

    # Disjoint CRN seed banks (same offsets as the benchmark helper).
    train_seeds = list(range(1000, 1000 + b["train_seeds"]))
    held_out_seeds = list(range(9000, 9000 + b["held_out_seeds"]))
    soft_tree_eval_seeds = list(range(20000, 20000 + b["soft_tree_eval_seeds"]))

    # 1) strongest heuristic = min(retailer_base_stock, dc_reserve_base_stock),
    #    tuned on a fixed train seed (reused grid-search helpers).
    rbs_params, _ = tune_retailer_base_stock(inst, train_seeds[0], b["heuristic_reps"])
    dcr_params, _ = tune_dc_reserve_base_stock(inst, train_seeds[0], b["heuristic_reps"])

    # 2) warm-start anchor: per-period shipment of the tuned retailer base-stock,
    #    clamped to the shipment cap.
    anchor_level = float(rbs_params[0])
    anchor_shipment = min(anchor_level, float(action_max))
    if a.warm_start == "base_stock":
        x0 = warm_start_mean(a.tree_depth, a.tree_leaf_type, anchor_shipment,
                             a.action_min, action_max)
    else:
        x0 = list(np.zeros(soft_tree_param_count(a.tree_depth, a.tree_leaf_type)))

    # 3) train the soft tree with the CLI structure. Request xfavorite (the CMA
    #    distribution mean) for the honest best-of floor (additive; default flow
    #    still deploys xbest unless the floor finds a better held-out endpoint).
    st_params, st_train, st_xfavorite = train_soft_tree(
        inst, a.tree_depth, train_seeds, b["popsize"], b["iters"], a.sigma_init,
        a.seed, a.action_min, action_max, a.tree_temperature, a.tree_split_type,
        a.tree_leaf_type, x0, return_xfavorite=True)

    # 4) held-out CRN scoring, common seeds across policies.
    rbs_samples = heuristic_held_out_samples(
        inst, "retailer_base_stock", rbs_params, held_out_seeds, b["heuristic_reps"])
    dcr_samples = heuristic_held_out_samples(
        inst, "dc_reserve_base_stock", dcr_params, held_out_seeds, b["heuristic_reps"])

    # ---- HONEST BEST-OF FLOOR (training-path audit 2026-06-06) -------------
    # Evaluate the trained endpoints (xbest, xfavorite) and the warm-start anchor
    # on the SAME held-out soft-tree seed block, then deploy the best per
    # --deploy_endpoint. xbest overfits the small CRN train batch; xfavorite (the
    # distribution mean) often generalizes better; the warm-start anchor (the
    # tuned base-stock the CMA mean was seeded at) is the downside floor. This is
    # DOWNSIDE-SAFE under deploy_endpoint=floor: never deploys worse than xbest.
    def _score(params):
        return summarize(held_out_samples(
            inst, params, a.tree_depth, soft_tree_eval_seeds, a.action_min,
            action_max, a.tree_temperature, a.tree_split_type, a.tree_leaf_type))

    xbest_mean, _, xbest_sem = _score(st_params)
    xfav_mean, _, xfav_sem = _score(st_xfavorite)
    xbest_cand = (xbest_mean, xbest_sem, "trained_xbest", st_params)
    xfav_cand = (xfav_mean, xfav_sem, "trained_xfavorite", st_xfavorite)
    if a.deploy_endpoint == "xbest":
        candidates = [xbest_cand]
    elif a.deploy_endpoint == "xfavorite":
        candidates = [xfav_cand]
    else:  # floor: best-of {xbest, xfavorite, warm-start anchor}
        candidates = [xbest_cand, xfav_cand]
        if a.warm_start == "base_stock":
            anchor_mean, _, anchor_sem = _score(x0)
            candidates.append((anchor_mean, anchor_sem, "warm_start_anchor", x0))
    st_mean, st_sem, deployed_endpoint, deployed_params = min(
        candidates, key=lambda c: c[0])
    st_params = deployed_params  # the deployed policy for downstream scoring/print

    rbs_mean, _, rbs_sem = summarize(rbs_samples)
    dcr_mean, _, dcr_sem = summarize(dcr_samples)

    if rbs_mean <= dcr_mean:
        best_heuristic_name, best_heuristic_cost = "retailer_base_stock", rbs_mean
    else:
        best_heuristic_name, best_heuristic_cost = "dc_reserve_base_stock", dcr_mean

    gap = st_mean - best_heuristic_cost
    gap_pct = 100.0 * (st_mean / best_heuristic_cost - 1.0)
    policy_architecture = (
        f"soft_tree_{a.tree_split_type}_{a.tree_leaf_type}_d{a.tree_depth}"
        f"_t{a.tree_temperature}_ws_{a.warm_start}_deploy_{deployed_endpoint}"
    )

    # 5) append the ledger row.
    root = PACKAGE_ROOT / "outputs" / "autoresearch" / a.run_tag
    root.mkdir(parents=True, exist_ok=True)
    results_tsv = root / "results.tsv"
    header = [
        "commit", "experiment_name", "instance", "budget", "policy_architecture",
        "mean_cost", "best_heuristic", "best_heuristic_name", "heuristic_gap",
        "heuristic_gap_pct", "soft_tree_sem", "best_heuristic_sem", "description",
    ]
    if not results_tsv.exists():
        with results_tsv.open("w", newline="", encoding="utf-8") as h:
            csv.writer(h, delimiter="\t").writerow(header)
    experiment_name = f"{a.run_tag}_{a.budget}_{policy_architecture}"
    best_heuristic_sem = rbs_sem if best_heuristic_name == "retailer_base_stock" else dcr_sem
    row = [
        _git_short_commit(PACKAGE_ROOT), experiment_name, a.instance, a.budget,
        policy_architecture, f"{st_mean:.6f}", f"{best_heuristic_cost:.6f}",
        best_heuristic_name, f"{gap:.6f}", f"{gap_pct:.4f}",
        f"{st_sem:.6f}", f"{best_heuristic_sem:.6f}", a.description,
    ]
    with results_tsv.open("a", newline="", encoding="utf-8") as h:
        csv.writer(h, delimiter="\t").writerow(row)

    print(json.dumps({
        "results_tsv": str(results_tsv),
        "instance": a.instance,
        "policy_architecture": policy_architecture,
        "deploy_endpoint_flag": a.deploy_endpoint,
        "deployed_endpoint": deployed_endpoint,
        "floor_candidates": {
            "trained_xbest": xbest_mean,
            "trained_xfavorite": xfav_mean,
            **({"warm_start_anchor": candidates[-1][0]}
               if a.deploy_endpoint == "floor" and a.warm_start == "base_stock"
               else {}),
        },
        "learned_mean_cost": st_mean,
        "soft_tree_sem": st_sem,
        "best_heuristic_name": best_heuristic_name,
        "best_heuristic_cost": best_heuristic_cost,
        "best_heuristic_sem": best_heuristic_sem,
        "heuristic_gap": gap,
        "heuristic_gap_pct": gap_pct,
        "wins": gap_pct < 0,
        "ledger_row": dict(zip(header, row)),
    }, indent=2))


if __name__ == "__main__":
    main()
