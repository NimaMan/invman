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

3. learned soft-tree policy (NOT runnable here without a trained parameter vector):
   - The binding `multi_echelon_general_backorder_fixed_cost_soft_tree_rollout` EXISTS and runs
     (a depth-2 oblique/linear vector_quantity policy needs a 585-length flat-param vector for the
     CardBoard network), but a *trained* parameter vector is not checked into the repo. Training it
     requires running CMA-ES through the invman/ Python harness (out of scope for a read-only
     benchmark). This script prints the exact rollout call so the learned-policy benchmark can be
     wired in once a trained vector exists.

USAGE
-----
    python scripts/general_backorder_fixed_cost/benchmark_general_backorder_fixed_cost.py
"""

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


def learned_policy_blocker_note() -> None:
    print("\n" + "=" * 92)
    print("3) LEARNED SOFT-TREE POLICY: binding exists and runs, trained vector NOT available")
    print("=" * 92)
    ref = ir.multi_echelon_general_backorder_fixed_cost_get_reference_instance("geevers2023_general_set1")
    nW, nR = ref["num_warehouses"], ref["num_retailers"]
    input_dim = nW + nR + 5  # compact_summary
    action_dim = nW + nR
    depth = 2
    # depth-2 oblique/linear vector_quantity needs a 585-length flat-param vector for this network.
    print(
        "   call: multi_echelon_general_backorder_fixed_cost_soft_tree_rollout(\n"
        f"             flat_params=<len 585>, input_dim={input_dim}, depth={depth},\n"
        f"             min_values=[0]*{action_dim}, max_values=[<bound>]*{action_dim},\n"
        "             action_mode='vector_quantity', reference_name=<set>,\n"
        "             policy_feature_mode='compact_summary',\n"
        "             policy_action_mode='node_base_stock_targets')\n"
        "   BLOCKER: a trained 585-length parameter vector must first be produced by CMA-ES via the\n"
        "   invman/ Python training harness; it is not checked in. Until then only the heuristic\n"
        "   benchmark above is runnable."
    )


if __name__ == "__main__":
    published_row_comparison()
    for name in ["geevers2023_general_set2"]:
        routing_mode_sweep(name)
        retailer_level_sweep(name)
    learned_policy_blocker_note()
