"""Paired-CRN optimality-gap evaluator: learned soft-tree vs the VI OPTIMUM (and MOQ) on
Vanvuchelen, Gijsbrechts & Boute (2020) joint-replenishment SETTING 5.

OBJECTIVE
---------
The four learned-policy results already in the paper are reported as a relative gap to the
strongest in-repo heuristic (MOQ). For joint-replenishment setting 5 the literature gives a
STRONGER comparator the paper itself uses in its Figure 2: the infinite-horizon discounted
VALUE-ITERATION OPTIMUM. This script computes the VI optimum's mean discounted cost on the
exact evaluation protocol (200 periods, gamma=0.99, init inventory [0,0]) under the SAME
common-random-number demand paths that score MOQ and the learned soft-tree, and reports the
learned policy's true OPTIMALITY GAP (learned / VI_optimum - 1) alongside its gap to MOQ.
This makes the joint-replenishment row directly comparable to the paper's published Figure 2
"heuristics sit 4-25% above optimal" statement.

WHY THIS IS A FAITHFUL OPTIMALITY GAP (NOT A PYTHON RE-IMPLEMENTATION DRIFT)
---------------------------------------------------------------------------
The env that scores every policy is the Rust `step_state` (Eq. 2/4 of the paper). There is no
Rust binding to simulate an arbitrary tabular (I1,I2)->(q1,q2) map, so the VI greedy policy is
rolled out in Python. To guarantee the Python rollout's cost arithmetic is byte-for-byte the
Rust env, the script ALSO rolls out MOQ in Python on the same paths and asserts it equals the
Rust `joint_replenishment_policy_rollout_from_paths` value to < 1e-6 before trusting any VI
number. The learned soft-tree is ALWAYS scored by the Rust binding
`joint_replenishment_soft_tree_rollout_from_paths` (never re-implemented in Python).

ALGORITHMIC DESCRIPTION
-----------------------
 1. VALUE ITERATION (literature anchor). Reuse benchmark_vanvuchelen_settings.value_iteration_
    setting5 (mirrors env.rs::step_state EXACTLY: order-before-demand, backorders, end-of-period
    cost c = sum_i[h_i*I+ + b_i*I- + k_i*1{q_i>0}] + M*K, aggregate order in {0, M*V}, gamma=0.99).
    It converges to the stationary discounted-optimal policy and reproduces the paper's published
    optimal action q=(0,6) at state (5,0). Tabulate the greedy action over the clamped inventory
    grid for O(1) lookup.
 2. CRN DEMAND PATHS. Build `eval_paths` fixed demand paths (numpy RandomState per seed, U[0,5]/
    U[0,3], `periods` long). The same path block scores all three policies => paired / variance-
    reduced. (numpy RNG != Rust StdRng, but identical paths across policies is a valid CRN block;
    the CRN cancellation is between policies on the SAME path, not vs the training sampler.)
 3. SCORE on the paired block: VI (Python env, validated), MOQ (Rust path binding), learned
    soft-tree (Rust path binding). Discounted cost per path, gamma=0.99.
 4. REPORT. learned vs VI optimum: signed gap and gap% = 100*(learned/VI - 1) (the optimality
    gap; >0 means above optimal). learned vs MOQ: gap% = 100*(learned/MOQ - 1) (<0 = learned
    cheaper = beats the heuristic). MOQ vs VI optimum: the heuristic's own optimality gap (the
    paper's Figure-2 number for context). Paired win-fractions and SEMs reported throughout.
    Write a JSON artifact.

CPU CAP
-------
RAYON_NUM_THREADS / OMP etc. capped to 2 before invman_rust import (sibling agents run in
parallel). VI is single-threaded numpy.

USAGE
-----
    RAYON_NUM_THREADS=2 python evaluate_setting5_vs_vi_optimum.py \
        --model_dir <.../models/<run>_<num_params>_<gen>> \
        --eval_paths 4096 --output_json <out.json>
    # --model_dir omitted => reports VI vs MOQ only (no learned policy).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _var in ("RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import invman_rust
from invman.policy import Policy

import benchmark_vanvuchelen_settings as bvs
from common import (
    get_reference,
    newsvendor_item_targets,
    soft_tree_rollout_kwargs,
)

EVAL_SEED_BASE = 1_000_000  # disjoint from any training seed block
SETTING5 = "vanvuchelen2020_small_scale_setting_5"
VI_CLAMP_LO, VI_CLAMP_HI = -12, 18  # matches value_iteration_setting5's inventory grid


def build_vi_action_table():
    """Tabulate the VI greedy optimal action over the clamped inventory grid for setting 5.

    Returns (action_table dict keyed by clamped (i1,i2), converged_iter, max_delta) and asserts
    the table reproduces the paper's published optimal action q=(0,6) at state (5,0)."""
    greedy, converged_iter, max_delta = bvs.value_iteration_setting5(
        lo=VI_CLAMP_LO, hi=VI_CLAMP_HI
    )
    table = {}
    for i1 in range(VI_CLAMP_LO, VI_CLAMP_HI + 1):
        for i2 in range(VI_CLAMP_LO, VI_CLAMP_HI + 1):
            table[(i1, i2)] = tuple(int(x) for x in greedy((i1, i2))[0])
    published = (0, 6)
    if table[(5, 0)] != published:
        raise SystemExit(
            f"VI did not reproduce the published optimal action: got {table[(5, 0)]}, "
            f"expected {published} at state (5,0)"
        )
    return table, int(converged_iter), float(max_delta)


def vi_action(table, i1, i2):
    return table[(int(np.clip(i1, VI_CLAMP_LO, VI_CLAMP_HI)),
                  int(np.clip(i2, VI_CLAMP_LO, VI_CLAMP_HI)))]


def generate_paths(reference, num_paths, periods, base=EVAL_SEED_BASE):
    """Fixed CRN demand paths: one numpy RandomState per seed, U[low,high] per item, period-long.
    The same path block is reused for every policy (paired / variance-reduced)."""
    dl = [int(x) for x in reference["demand_lows"]]
    dh = [int(x) for x in reference["demand_highs"]]
    num_items = len(dl)
    paths = []
    for s in range(int(num_paths)):
        rng = np.random.RandomState(int(base) + s)
        cols = [rng.randint(dl[i], dh[i] + 1, size=int(periods)) for i in range(num_items)]
        paths.append([list(period_demand) for period_demand in zip(*[c.tolist() for c in cols])])
    return paths


def py_rollout_tabular(reference, action_fn, paths, gamma):
    """Roll out a tabular (i1,i2)->(q1,q2) policy through a Python mirror of env.rs::step_state.

    Validated byte-for-byte against the Rust env via the MOQ cross-check in main()."""
    V = int(reference["truck_capacity"])
    K = float(reference["major_order_cost"])
    k = [float(x) for x in reference["minor_order_costs"]]
    h = [float(x) for x in reference["holding_costs"]]
    b = [float(x) for x in reference["shortage_costs"]]
    init = [int(x) for x in reference.get("initial_inventory_levels", [0, 0])]
    costs = []
    for demand_path in paths:
        i1, i2 = init
        discount, total = 1.0, 0.0
        for (d1, d2) in demand_path:
            q1, q2 = action_fn(i1, i2)
            total_q = q1 + q2
            if total_q != 0 and total_q % V != 0:
                raise ValueError(f"infeasible truck order {(q1, q2)} not a multiple of {V}")
            trucks = 0 if total_q == 0 else total_q // V
            order_cost = K * trucks + (k[0] if q1 > 0 else 0.0) + (k[1] if q2 > 0 else 0.0)
            e1, e2 = i1 + q1 - d1, i2 + q2 - d2
            holding = h[0] * max(e1, 0) + h[1] * max(e2, 0)
            shortage = b[0] * max(-e1, 0) + b[1] * max(-e2, 0)
            total += discount * (order_cost + holding + shortage)
            discount *= gamma
            i1, i2 = e1, e2
        costs.append(total)
    return np.asarray(costs, dtype=np.float64)


def moq_rust_costs(reference, targets, paths, gamma):
    dl = [int(x) for x in reference["demand_lows"]]
    dh = [int(x) for x in reference["demand_highs"]]
    init = [int(x) for x in reference.get("initial_inventory_levels", [0, 0])]
    params = [float(targets[0]), float(targets[1]), 1.0, 2.0]
    costs = []
    for demand_path in paths:
        costs.append(invman_rust.joint_replenishment_policy_rollout_from_paths(
            policy_name="minimum_order_quantity",
            params=params,
            initial_inventory_levels=init,
            demands=[list(period_demand) for period_demand in demand_path],
            demand_lows=dl,
            demand_highs=dh,
            truck_capacity=int(reference["truck_capacity"]),
            minor_order_costs=[float(x) for x in reference["minor_order_costs"]],
            major_order_cost=float(reference["major_order_cost"]),
            holding_costs=[float(x) for x in reference["holding_costs"]],
            shortage_costs=[float(x) for x in reference["shortage_costs"]],
            discount_factor=float(gamma),
        ))
    return np.asarray(costs, dtype=np.float64)


def learned_rust_costs(reference, model, paths, gamma):
    rollout_kwargs = {
        key: value
        for key, value in soft_tree_rollout_kwargs(
            reference, model, flat_params=model.get_model_flat_params()
        ).items()
        if key not in ("flat_params", "periods", "discount_factor")
    }
    flat = np.asarray(model.get_model_flat_params(), dtype=np.float32).tolist()
    costs = []
    for demand_path in paths:
        costs.append(invman_rust.joint_replenishment_soft_tree_rollout_from_paths(
            flat_params=flat,
            demands=[list(period_demand) for period_demand in demand_path],
            discount_factor=float(gamma),
            **rollout_kwargs,
        ))
    return np.asarray(costs, dtype=np.float64)


def _stats(costs):
    n = len(costs)
    return {
        "mean": float(np.mean(costs)),
        "sem": float(np.std(costs) / np.sqrt(n)) if n else 0.0,
        "n": int(n),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", default=None,
                   help="trained soft-tree save dir (with policy_artifact.json); omit for VI-vs-MOQ only")
    p.add_argument("--reference", default=SETTING5)
    p.add_argument("--periods", type=int, default=200)
    p.add_argument("--discount_factor", type=float, default=0.99)
    p.add_argument("--eval_paths", type=int, default=4096)
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def main():
    parsed = parse_args()
    if parsed.reference != SETTING5:
        raise SystemExit("the VI optimum solver in this script is specialised to setting 5")

    ref = get_reference(parsed.reference)
    ref["periods"] = int(parsed.periods)
    ref["discount_factor"] = float(parsed.discount_factor)
    ref["initial_inventory_levels"] = [0] * int(ref.get("num_items", len(ref["demand_highs"])))
    gamma = float(parsed.discount_factor)

    print("=" * 90)
    print(f"VI-OPTIMUM optimality-gap evaluator -- {parsed.reference}")
    print(f"  protocol: {parsed.periods} periods, gamma={gamma}, init inventory "
          f"{ref['initial_inventory_levels']}, {parsed.eval_paths} paired-CRN demand paths")
    print("=" * 90)

    # 1. VI optimal policy table (literature anchor; reproduces published q=(0,6) at (5,0)).
    table, vi_iter, vi_delta = build_vi_action_table()
    print(f"  VI converged at iter {vi_iter} (max delta {vi_delta:.2e}); "
          f"reproduces published optimal action q=(0,6) at state (5,0): YES")

    # 2. CRN paths.
    paths = generate_paths(ref, parsed.eval_paths, parsed.periods)

    # 3a. MOQ at the newsvendor target -- Rust path binding AND Python mirror (arithmetic check).
    targets = newsvendor_item_targets(ref)
    moq_rust = moq_rust_costs(ref, targets, paths, gamma)

    def moq_action(i1, i2):
        q = invman_rust.joint_replenishment_moq_order_quantities(
            inventory_levels=[int(i1), int(i2)], item_targets=[int(targets[0]), int(targets[1])],
            review_period=1, rounding_threshold=2.0, truck_capacity=int(ref["truck_capacity"]),
            period=0,
        )
        return int(q[0]), int(q[1])

    moq_py = py_rollout_tabular(ref, moq_action, paths, gamma)
    max_arith_diff = float(np.max(np.abs(moq_rust - moq_py)))
    if max_arith_diff > 1e-6:
        raise SystemExit(
            f"Python env arithmetic drifts from the Rust env (max|diff|={max_arith_diff}); "
            "VI numbers cannot be trusted -- aborting."
        )
    print(f"  Python-vs-Rust env arithmetic cross-check (MOQ): max|diff|={max_arith_diff:.2e} OK")

    # 3b. VI optimum -- Python rollout of the validated env.
    vi_costs = py_rollout_tabular(ref, lambda i1, i2: vi_action(table, i1, i2), paths, gamma)

    vi_stats = _stats(vi_costs)
    moq_stats = _stats(moq_rust)
    print(f"  VI optimum mean cost : {vi_stats['mean']:.3f} (SEM {vi_stats['sem']:.3f})")
    print(f"  MOQ        mean cost : {moq_stats['mean']:.3f} (SEM {moq_stats['sem']:.3f})  "
          f"newsvendor targets={list(targets)}")
    moq_opt_gap_pct = 100.0 * (moq_stats["mean"] / vi_stats["mean"] - 1.0)
    print(f"  MOQ optimality gap vs VI optimum: {moq_opt_gap_pct:+.2f}% "
          f"(paper Figure 2: heuristics 4-25% above optimal)")

    payload = {
        "reference": parsed.reference,
        "protocol": {
            "periods": parsed.periods, "discount_factor": gamma,
            "initial_inventory_levels": ref["initial_inventory_levels"],
            "eval_paths": int(parsed.eval_paths), "eval_seed_base": EVAL_SEED_BASE,
        },
        "vi": {
            "converged_iter": vi_iter, "max_delta": vi_delta,
            "reproduces_published_action_0_6_at_5_0": True,
            **vi_stats,
        },
        "moq": {"newsvendor_targets": [int(t) for t in targets],
                "optimality_gap_pct_vs_vi": moq_opt_gap_pct, **moq_stats},
        "python_vs_rust_env_max_abs_diff": max_arith_diff,
    }

    # 4. Learned soft-tree (if a model dir is given) -- Rust path binding only.
    if parsed.model_dir:
        model = Policy.load(parsed.model_dir)
        learned_costs = learned_rust_costs(ref, model, paths, gamma)
        learned_stats = _stats(learned_costs)

        learned_opt_gap_pct = 100.0 * (learned_stats["mean"] / vi_stats["mean"] - 1.0)
        learned_vs_moq_pct = 100.0 * (learned_stats["mean"] / moq_stats["mean"] - 1.0)
        win_vs_moq = int(np.sum(learned_costs < moq_rust))
        win_vs_vi = int(np.sum(learned_costs < vi_costs))

        print("-" * 90)
        print(f"  LEARNED soft-tree mean cost : {learned_stats['mean']:.3f} "
              f"(SEM {learned_stats['sem']:.3f})  [{parsed.model_dir}]")
        print(f"  learned OPTIMALITY GAP vs VI optimum : {learned_opt_gap_pct:+.2f}%  "
              f"(closes {100.0 * (1 - learned_opt_gap_pct / moq_opt_gap_pct):.1f}% of MOQ's gap)")
        print(f"  learned vs MOQ                       : {learned_vs_moq_pct:+.2f}%  "
              f"({'BEATS MOQ' if learned_vs_moq_pct < 0 else 'loses to MOQ'}; "
              f"cheaper on {win_vs_moq}/{parsed.eval_paths} paths)")
        print(f"  learned cheaper than VI optimum on   : {win_vs_vi}/{parsed.eval_paths} paths "
              f"(expected ~0; VI is the discounted optimum)")

        verdict = "beats" if learned_vs_moq_pct < -1e-9 else ("matches" if abs(learned_vs_moq_pct) <= 1e-9 else "below")
        payload["learned"] = {
            "model_dir": str(parsed.model_dir),
            "optimality_gap_pct_vs_vi": learned_opt_gap_pct,
            "gap_pct_vs_moq": learned_vs_moq_pct,
            "verdict_vs_moq": verdict,
            "fraction_of_moq_gap_closed": (
                float(1 - learned_opt_gap_pct / moq_opt_gap_pct) if moq_opt_gap_pct else None
            ),
            "win_fraction_vs_moq": win_vs_moq / int(parsed.eval_paths),
            "win_fraction_vs_vi": win_vs_vi / int(parsed.eval_paths),
            **learned_stats,
        }

    if parsed.output_json:
        out = Path(parsed.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    print()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
