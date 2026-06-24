"""
Benchmark + verification-diagnostic harness for the multi_echelon/general_backorder_fixed_cost
problem (Geevers, van Hezewijk & Mes 2024, CardBoard Company general network).

OBJECTIVE
---------
Compare the available policies against the published Geevers benchmark rows on the three
literature reference instances, and emit the verification-diagnostic numbers that root-cause the
set 2 / set 3 reproduction gap. This script only READS the already-installed invman_rust extension
(no rebuild required).

ALGORITHM
---------
1. node-base-stock heuristic (the published benchmark policy):
   - For each reference instance, simulate the published base-stock levels under the instance's
     configured routing mode, over `replications` Monte-Carlo paths and several seeds.
   - Report mean cost vs the published benchmark cost and the percent gap.
   - This is the apples-to-apples published-row comparison.

2. routing-mode + level sweep (diagnostic, sets 2/3 only):
   - For set 2/3 the configured routing mode does NOT reproduce the published 4797. This step
     sweeps every implemented routing mode at the published level, and (for the evenly-split mode)
     sweeps the retailer order-up-to level, to localise the gap:
       * the published level 30 (evenly split) gives ~92.5% customer fill and cost ~12000
       * retailer level ~36-37 reproduces BOTH cost ~4797 AND the paper's ~98% fill
     => the gap is a consistent ~6-7 unit offset in the retailer order-up-to level, i.e. a
        per-edge inventory-position / order-up-to timing convention difference in the journal's
        "order per edge" transition (exact equation only in the gated journal full text).

3. learned soft-tree policy artifacts:
   - Training/evaluation lives in `autoresearch_general_backorder_fixed_cost.py`, not in this
     diagnostic harness.
   - This script prints the tracked five-seed full-budget TSV rows for set 1 and
     Kunnumkal-Topaloglu so the verification command does not drift from the learned-policy result.

USAGE
-----
    python scripts/general_backorder_fixed_cost/benchmark_general_backorder_fixed_cost.py
"""

import csv
from pathlib import Path
import statistics as st

import invman_rust as ir

INSTANCES = [
    "geevers2023_general_set1",
    "geevers2023_general_set2",
    "geevers2023_general_set3",
]
ROUTING_MODES = [
    "random_single_connection_by_weight",
    "split_across_all_connections_by_weight",
    "split_across_all_connections_evenly",
    "duplicate_target_all_connections",
    "weighted_target_all_connections",
]
SEEDS = [1234, 5678, 9012]
REPLICATIONS = 500
REPO_ROOT = Path(__file__).resolve().parents[2]
AUTORESEARCH_TSV = (
    REPO_ROOT / "outputs" / "autoresearch" / "general_backorder_fixed_cost_autoresearch" / "results.tsv"
)


def published_row_comparison() -> None:
    print("=" * 92)
    print("1) PUBLISHED-ROW COMPARISON: node-base-stock benchmark vs Geevers published cost")
    print("   (published levels, configured routing mode, %d reps x %d seeds)" % (REPLICATIONS, len(SEEDS)))
    print("=" * 92)
    print(f"{'instance':28s} {'routing mode':42s} {'pub':>7s} {'repo':>9s} {'gap%':>8s}")
    for name in INSTANCES:
        ref = ir.multi_echelon_general_backorder_fixed_cost_get_reference_instance(name)
        pub = ref["published_benchmark_cost"]
        mode = ref["benchmark_order_routing_mode"]
        means = []
        for seed in SEEDS:
            d = ir.multi_echelon_general_backorder_fixed_cost_simulate_base_stock(
                name, None, REPLICATIONS, seed, None
            )
            means.append(d["mean_cost"])
        m = st.mean(means)
        gap = 100.0 * (m - pub) / pub
        print(f"{name:28s} {mode:42s} {pub:7.0f} {m:9.1f} {gap:+8.1f}")


def routing_mode_sweep(name: str) -> None:
    print("\n" + "=" * 92)
    print(f"2a) ROUTING-MODE SWEEP for {name} at published levels ({REPLICATIONS} reps)")
    print("=" * 92)
    ref = ir.multi_echelon_general_backorder_fixed_cost_get_reference_instance(name)
    pub = ref["published_benchmark_cost"]
    print(f"   published cost = {pub:.0f}")
    print(f"{'mode':42s} {'cost':>9s} {'hold':>8s} {'cust_bo':>9s} {'custfill':>9s} {'whfill_min':>11s}")
    for mode in ROUTING_MODES:
        a = ir.multi_echelon_general_backorder_fixed_cost_audit_base_stock(
            name, None, REPLICATIONS, 1234, mode
        )
        cf = a["customer_fill_rates"]
        ef = a["edge_fill_rates"]
        print(
            f"{mode:42s} {a['mean_cost']:9.1f} {a['mean_holding_cost']:8.1f} "
            f"{a['mean_customer_backorder_cost']:9.1f} {sum(cf)/len(cf):9.3f} {min(ef):11.3f}"
        )


def retailer_level_sweep(name: str) -> None:
    print("\n" + "=" * 92)
    print(f"2b) RETAILER-LEVEL SWEEP for {name} (evenly split, warehouse levels fixed)")
    print("    finds the retailer order-up-to that reproduces BOTH cost 4797 AND ~98% fill")
    print("=" * 92)
    ref = ir.multi_echelon_general_backorder_fixed_cost_get_reference_instance(name)
    base = list(ref["benchmark_base_stock_levels"])
    nW = ref["num_warehouses"]
    print(f"{'retailer_level':>14s} {'cost':>9s} {'custfill_mean':>14s} {'whfill_min':>11s}")
    for r in [30, 34, 36, 37, 38, 40, 42]:
        bs = base[:nW] + [r] * (len(base) - nW)
        a = ir.multi_echelon_general_backorder_fixed_cost_audit_base_stock(
            name, bs, REPLICATIONS, 1234, "split_across_all_connections_evenly"
        )
        cf = a["customer_fill_rates"]
        ef = a["edge_fill_rates"]
        print(f"{r:14d} {a['mean_cost']:9.1f} {sum(cf)/len(cf):14.3f} {min(ef):11.3f}")


def learned_policy_artifact_note() -> None:
    print("\n" + "=" * 92)
    print("3) LEARNED SOFT-TREE POLICY: seed-robust artifacts and rollout shape")
    print("=" * 92)
    print(
        "   rollout shape: depth=2 split=oblique leaf=constant param_dim=81 input_dim=14\n"
        "      action_mode=vector_quantity policy_action_mode=node_base_stock_targets"
    )

    if AUTORESEARCH_TSV.exists():
        with AUTORESEARCH_TSV.open(newline="") as fh:
            all_rows = list(csv.DictReader(fh, delimiter="\t"))
        for reference, label in [
            ("geevers2023_general_set1", "set1 CardBoard rows"),
            ("kunnumkal_topaloglu_divergent", "KT divergent rows"),
        ]:
            rows = [
                row
                for row in all_rows
                if row["reference"] == reference
                and row["budget"] == "full"
                and row["depth"] == "2"
                and row["leaf_type"] == "constant"
                and row["sigma_init"] == "0.2"
            ]
            learned = [float(row["learned_heldout"]) for row in rows]
            savings = [-float(row["gap_pct_vs_heuristic"]) for row in rows]
            if learned:
                learned_std = st.stdev(learned) if len(learned) > 1 else 0.0
                savings_std = st.stdev(savings) if len(savings) > 1 else 0.0
                gate = float(rows[0]["repo_heuristic"])
                seeds = ", ".join(row["experiment"].rsplit("_s", 1)[-1] for row in rows)
                print(
                    f"   {label}: {AUTORESEARCH_TSV.relative_to(REPO_ROOT)}\n"
                    f"      seeds {seeds}; learned mean {st.mean(learned):.1f} +/- {learned_std:.1f} "
                    f"vs gate {gate:.1f}; savings {st.mean(savings):.2f}% +/- {savings_std:.2f}%; "
                    f"{sum(float(row['learned_heldout']) < gate for row in rows)}/{len(rows)} seeds"
                )
    else:
        print(f"   autoresearch TSV missing: {AUTORESEARCH_TSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    published_row_comparison()
    for name in ["geevers2023_general_set2"]:
        routing_mode_sweep(name)
        retailer_level_sweep(name)
    learned_policy_artifact_note()
