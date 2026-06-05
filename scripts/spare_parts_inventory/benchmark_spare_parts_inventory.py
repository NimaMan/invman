"""Self-contained benchmark for the spare_parts_inventory problem.

OBJECTIVE
---------
Produce the three reproducible benchmark blocks that define the verification
status of `src/problems/spare_parts_inventory`, using only the already
installed `invman_rust` bindings (no Rust rebuild):

  1. Kranenburg (2006) Chapter 5, Table 5.2 LATERAL-TRANSSHIPMENT family.
     This is the literature-verified, executable, exact-analytical subfamily.
     The Rust analytical solver reproduces all 35 published rows of Table 5.2
     (situation-1 separate stock points vs situation-3 lateral transshipment,
     optimal randomized stock R* and cost C(R*), plus the cost ratio). We report
     the worst absolute deviation from the published table across all rows.

  2. Repo-native EXACT finite-horizon DP on the reduced verification instance.
     This is the repo's own MDP family (single-echelon repairable spares with
     installed-base failures, repair returns, procurement pipeline). It is NOT
     literature-verified; it is an internal self-consistency anchor: the exact
     DP must weakly dominate both carried heuristics.

  3. Learned soft-tree vs heuristics on the primary (17-period) repairable
     instance. The soft-tree weights are loaded from a previously saved CMA-ES
     training artifact and re-evaluated on a FRESH held-out seed block, so the
     comparison is out-of-sample. We compare against the two carried heuristics
     (best constant base-stock S and lead-time-mean-cover). This block depends
     on a saved `trained_flat_params` JSON; if absent it is skipped with a note.

WHY A NEW SCRIPT
----------------
The pre-existing `scripts/spare_parts_inventory/common.py` imports
`from invman.policies.soft_tree import SoftTreePolicy`, which no longer exists
(the learned-policy descriptor moved to `invman.policy.Policy`). This script is
self-contained against the current API so the benchmark is runnable today.

USAGE
-----
  python scripts/spare_parts_inventory/benchmark_spare_parts_inventory.py \
      [--soft_tree_artifact PATH] [--holdout_seeds 4096] [--holdout_seed_start 900000] \
      [--output_json PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.policy import Policy  # noqa: E402

import invman_rust as ir  # noqa: E402

DISCOUNT_FACTOR = 0.99
DEFAULT_SOFT_TREE_ARTIFACT = (
    PACKAGE_ROOT / "outputs" / "spare_parts_inventory" / "retry_d2_t010_e300_s123.json"
)


# --------------------------------------------------------------------------- #
# Block 1: Kranenburg (2006) Table 5.2 exact reproduction                     #
# --------------------------------------------------------------------------- #
def kranenburg_block() -> dict:
    refs = [dict(r) for r in ir.spare_parts_inventory_kranenburg_reference_instances()]
    rows = []
    worst = {
        "situation1_optimal_r_abs_diff": 0.0,
        "situation1_cost_abs_diff": 0.0,
        "situation3_optimal_r_abs_diff": 0.0,
        "situation3_cost_abs_diff": 0.0,
        "cost_ratio_abs_diff": 0.0,
    }
    failures = 0
    for ref in refs:
        summary = dict(ir.spare_parts_inventory_kranenburg_exact_summary(ref["name"]))
        cmp_ = dict(summary["published_table_comparison"])
        ev = dict(summary["evaluation"])
        s1 = dict(ev["situation1"])
        s3 = dict(ev["situation3"])
        rows.append(
            {
                "name": ref["name"],
                "varied_parameter": ref["varied_parameter"],
                "varied_value_label": ref["varied_value_label"],
                "computed_s1_r": s1["optimal_r"],
                "published_s1_r": ref["published_situation1_optimal_r"],
                "computed_s1_cost": s1["total_cost"],
                "published_s1_cost": ref["published_situation1_cost"],
                "computed_s3_r": s3["optimal_r"],
                "published_s3_r": ref["published_situation3_optimal_r"],
                "computed_s3_cost": s3["total_cost"],
                "published_s3_cost": ref["published_situation3_cost"],
                "all_within_tolerance": cmp_["all_within_tolerance"],
            }
        )
        for key in worst:
            worst[key] = max(worst[key], float(cmp_[key]))
        if not cmp_["all_within_tolerance"]:
            failures += 1
    return {
        "num_rows": len(rows),
        "rows_failing_tolerance": failures,
        "table_rounding_tolerance": 0.02,
        "worst_abs_diff_across_rows": worst,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Block 2: repo-native exact finite-horizon DP self-consistency               #
# --------------------------------------------------------------------------- #
def exact_dp_block() -> dict:
    dp = dict(ir.spare_parts_inventory_exact_dp_summary())
    vref = dict(dp["verification_reference"])
    return {
        "verification_source": vref["verification_source"],
        "literature_verified": vref["literature_verified"],
        "optimal_discounted_cost": dp["optimal_discounted_cost"],
        "optimal_first_action": dp["optimal_first_action"],
        "base_stock_discounted_cost": dp["base_stock_discounted_cost"],
        "lead_time_mean_cover_discounted_cost": dp["lead_time_mean_cover_discounted_cost"],
        "base_stock_gap_to_optimal": dp["base_stock_gap_to_optimal"],
        "lead_time_mean_cover_gap_to_optimal": dp["lead_time_mean_cover_gap_to_optimal"],
        "dp_dominates_base_stock": dp["base_stock_gap_to_optimal"] >= -1e-9,
        "dp_dominates_lead_time_mean_cover": dp["lead_time_mean_cover_gap_to_optimal"] >= -1e-9,
    }


# --------------------------------------------------------------------------- #
# Block 3: learned soft-tree vs heuristics on the primary instance            #
# --------------------------------------------------------------------------- #
def _primary_reference() -> dict:
    return dict(ir.spare_parts_inventory_primary_reference_instance())


def _action_cap(ref: dict) -> int:
    mc_target = int(
        ir.spare_parts_inventory_lead_time_mean_cover_target(
            installed_base=int(ref["installed_base"]),
            failure_probability=float(ref["failure_probability"]),
            procurement_lead_time=int(ref["procurement_lead_time"]),
            safety_buffer=float(ref["benchmark_lead_time_mean_cover_safety_buffer"]),
        )
    )
    return max(
        16,
        int(ref["installed_base"]),
        2 * int(ref["benchmark_base_stock_level"]),
        2 * mc_target,
    )


def _heuristic_eval(ref: dict, policy_name: str, params: list[float], reps: int, seed: int) -> dict:
    summary = dict(
        ir.spare_parts_inventory_simulate_policy(
            policy_name=str(policy_name),
            params=[float(v) for v in params],
            on_hand_inventory=int(ref["initial_on_hand_inventory"]),
            backlog=int(ref["initial_backlog"]),
            procurement_pipeline=[int(v) for v in ref["initial_procurement_pipeline"]],
            repair_pipeline=[int(v) for v in ref["initial_repair_pipeline"]],
            installed_base=int(ref["installed_base"]),
            periods=int(ref["periods"]),
            failure_probability=float(ref["failure_probability"]),
            holding_cost=float(ref["holding_cost"]),
            downtime_cost=float(ref["downtime_cost"]),
            procurement_cost=float(ref["procurement_cost"]),
            replications=int(reps),
            seed=int(seed),
            discount_factor=DISCOUNT_FACTOR,
        )
    )
    return {
        "policy": policy_name,
        "params": [float(v) for v in params],
        "mean_cost": float(summary["mean_discounted_cost"]),
        "std_cost": float(summary["std_discounted_cost"]),
    }


def _best_constant_base_stock(ref: dict, cap: int, reps: int, seed: int) -> dict:
    """Sweep S in [0, cap] to find the best constant base-stock level (fair static proxy)."""
    best = None
    sweep = []
    for level in range(0, cap + 1):
        ev = _heuristic_eval(ref, "base_stock", [float(level)], reps, seed)
        ev["base_stock_level"] = level
        sweep.append({"S": level, "mean_cost": ev["mean_cost"]})
        if best is None or ev["mean_cost"] < best["mean_cost"]:
            best = ev
    best["sweep"] = sweep
    return best


def _soft_tree_eval(ref: dict, artifact: dict, cap: int, seeds: list[int]) -> dict:
    cfg = artifact["tree_config"]
    flat = np.asarray(artifact["trained_flat_params"], dtype=np.float32)
    proc = [int(v) for v in ref["initial_procurement_pipeline"]]
    rep = [int(v) for v in ref["initial_repair_pipeline"]]
    input_dim = len(proc) + len(rep) + 7
    pol = Policy(
        backbone="soft_tree",
        input_dim=input_dim,
        control_dim=1,
        control_mode="scalar_quantity",
        min_values=(0,),
        max_values=(int(cap),),
        depth=int(cfg["depth"]),
        temperature=float(cfg["temperature"]),
        split_type=str(cfg["split_type"]),
        leaf_type=str(cfg["leaf_type"]),
        state_normalizer="identity",
    )
    if pol.num_params != flat.size:
        raise ValueError(
            f"soft-tree artifact has {flat.size} params but Policy expects {pol.num_params}"
        )
    costs = []
    flat_list = flat.tolist()
    for seed in seeds:
        cost = ir.spare_parts_inventory_soft_tree_rollout(
            flat_params=flat_list,
            input_dim=input_dim,
            depth=int(cfg["depth"]),
            min_values=[0],
            max_values=[int(cap)],
            action_mode="scalar_quantity",
            on_hand_inventory=int(ref["initial_on_hand_inventory"]),
            backlog=int(ref["initial_backlog"]),
            procurement_pipeline=proc,
            repair_pipeline=rep,
            installed_base=int(ref["installed_base"]),
            periods=int(ref["periods"]),
            failure_probability=float(ref["failure_probability"]),
            holding_cost=float(ref["holding_cost"]),
            downtime_cost=float(ref["downtime_cost"]),
            procurement_cost=float(ref["procurement_cost"]),
            seed=int(seed),
            discount_factor=DISCOUNT_FACTOR,
            temperature=float(cfg["temperature"]),
            split_type=str(cfg["split_type"]),
            leaf_type=str(cfg["leaf_type"]),
            allowed_values=None,
        )
        costs.append(float(cost))
    costs = np.asarray(costs, dtype=np.float64)
    return {
        "policy": "soft_tree",
        "tree_config": cfg,
        "mean_cost": float(costs.mean()),
        "std_cost": float(costs.std()),
        "num_samples": int(costs.size),
    }


def learned_policy_block(artifact_path: Path, holdout_start: int, holdout_seeds: int) -> dict:
    ref = _primary_reference()
    cap = _action_cap(ref)
    reps = int(holdout_seeds)
    eval_seed = int(holdout_start)

    bs_benchmark = _heuristic_eval(
        ref, "base_stock", [float(ref["benchmark_base_stock_level"])], reps, eval_seed
    )
    mc_benchmark = _heuristic_eval(
        ref,
        "lead_time_mean_cover",
        [float(ref["benchmark_lead_time_mean_cover_safety_buffer"])],
        reps,
        eval_seed,
    )
    best_bs = _best_constant_base_stock(ref, cap, reps, eval_seed)

    block = {
        "instance": ref["name"],
        "action_cap": cap,
        "holdout_seed_start": eval_seed,
        "holdout_seeds": reps,
        "benchmark_base_stock": bs_benchmark,
        "lead_time_mean_cover": mc_benchmark,
        "best_constant_base_stock": {
            "S": best_bs["base_stock_level"],
            "mean_cost": best_bs["mean_cost"],
            "std_cost": best_bs["std_cost"],
        },
    }

    if artifact_path is not None and artifact_path.exists():
        artifact = json.loads(artifact_path.read_text())
        seeds = list(range(eval_seed, eval_seed + reps))
        st = _soft_tree_eval(ref, artifact, cap, seeds)
        block["soft_tree"] = st
        block["soft_tree_artifact"] = str(artifact_path)
        best_static = best_bs["mean_cost"]
        block["soft_tree_vs_best_constant_base_stock_pct"] = (
            (best_static - st["mean_cost"]) / best_static * 100.0
        )
        block["soft_tree_vs_benchmark_base_stock_pct"] = (
            (bs_benchmark["mean_cost"] - st["mean_cost"]) / bs_benchmark["mean_cost"] * 100.0
        )
        block["soft_tree_vs_lead_time_mean_cover_pct"] = (
            (mc_benchmark["mean_cost"] - st["mean_cost"]) / mc_benchmark["mean_cost"] * 100.0
        )
    else:
        block["soft_tree"] = None
        block["soft_tree_artifact"] = (
            str(artifact_path) if artifact_path is not None else None
        )
        block["soft_tree_note"] = "no saved trained_flat_params artifact found; learned block skipped"
    return block


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--soft_tree_artifact", default=str(DEFAULT_SOFT_TREE_ARTIFACT))
    p.add_argument("--holdout_seeds", type=int, default=4096)
    p.add_argument("--holdout_seed_start", type=int, default=900000)
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def main():
    parsed = parse_args()
    artifact_path = Path(parsed.soft_tree_artifact) if parsed.soft_tree_artifact else None
    payload = {
        "kranenburg_table_5_2": kranenburg_block(),
        "repo_native_exact_dp": exact_dp_block(),
        "learned_policy_primary_instance": learned_policy_block(
            artifact_path, parsed.holdout_seed_start, parsed.holdout_seeds
        ),
    }
    if parsed.output_json:
        out = Path(parsed.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
