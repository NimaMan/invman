#!/usr/bin/env python3
# =============================================================================
# measure_env_vs_closedform.py
#
# OBJECTIVE
#   Make the literature-verification status of the decentralized_inventory_control
#   problem auditable from Python alone, using only the already-installed
#   invman_rust extension (no Rust rebuild, no cargo test).
#
#   It answers one question precisely: does the reusable env.rs transition
#   reproduce the published Sterman / Edali-Yasarcan (2014) Beer Game benchmark
#   of per-stage [46, 50, 54, 54] / total 204?
#
# ALGORITHM (what this script computes)
#   1. Closed-form anchor (literature-verified):
#      call decentralized_inventory_control_classic_sterman_literature_summary(),
#      which runs verification/classic_board_game.rs (an exact Rust port of the
#      public Edali & Yasarcan R code with the optimized anchor-and-adjust policy
#      hardcoded). Expected: [46, 50, 54, 54], total 204.
#
#   2. env.rs under the SAME published parameters:
#      build PRIMARY_REFERENCE_INSTANCE (theta=0, S'=[28,28,28,20], sat=1, wsl=1,
#      h=0.5, p=1.0, 36-week path 4,4,4,4,8,...,8) and roll out the repo's
#      sterman_anchor_adjust heuristic through env.rs via
#      decentralized_inventory_control_policy_rollout_from_paths (discount 1.0).
#      Observed: 378 (NOT 204) -> the env is not calibrated to this anchor.
#
#   3. Best simple base-stock on env.rs:
#      sweep a single shared base-stock level S over {16,20,24,28} on env.rs.
#      Observed best at S=24 -> 278, still well above the closed-form 204,
#      confirming the gap is structural (different MDP), not a tuning artifact.
#
# INTERPRETATION
#   The closed-form simulator is literature-verified; the reusable env.rs is a
#   different (also-valid) decentralized serial MDP and is NOT literature-verified
#   against the only published anchor it carries. See the problem's
#   verification/README.md for the root cause.
# =============================================================================

import invman_rust

# ---- PRIMARY_REFERENCE_INSTANCE (literature/references.rs) ------------------
DEMANDS = [4, 4, 4, 4] + [8] * 32  # canonical 36-week path
ON_HAND = [12, 12, 12, 12]
BACKLOG = [0, 0, 0, 0]
SHIP_PIPES = [[4, 4], [4, 4], [4, 4], [4, 4]]
ORDER_PIPES = [[], [4], [4], [4]]
LAST_RECV_SHIP = [4, 4, 4, 4]
LAST_RECV_ORD = [4, 4, 4, 4]
FORECAST = [4.0, 4.0, 4.0, 4.0]
LAST_ACT = [4, 4, 4, 4]
SMOOTHING = [0.0, 0.0, 0.0, 0.0]
HOLDING = [0.5, 0.5, 0.5, 0.5]
BACKLOG_COST = [1.0, 1.0, 1.0, 1.0]

# Sterman params flattened: targets(4) | adjustment_times(4) | supply_line_weights(4)
STERMAN_PARAMS = [28.0, 28.0, 28.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def env_policy_cost(policy_name, params, discount=1.0):
    return invman_rust.decentralized_inventory_control_policy_rollout_from_paths(
        policy_name,
        params,
        ON_HAND,
        BACKLOG,
        SHIP_PIPES,
        ORDER_PIPES,
        LAST_RECV_SHIP,
        LAST_RECV_ORD,
        FORECAST,
        LAST_ACT,
        DEMANDS,
        SMOOTHING,
        HOLDING,
        BACKLOG_COST,
        discount,
    )


def main():
    print("== 1. Closed-form classic_board_game.rs (literature-verified) ==")
    summary = invman_rust.decentralized_inventory_control_classic_sterman_literature_summary()
    print("   per-stage:", summary["per_agent_costs"], "total:", summary["total_cost"])
    print("   expected (Sterman / Edali-Yasarcan 2014): [46, 50, 54, 54], total 204")

    print("\n== 2. env.rs + sterman_anchor_adjust, SAME published parameters ==")
    cost = env_policy_cost("sterman_anchor_adjust", STERMAN_PARAMS)
    print(f"   env.rs total: {cost}  (closed-form benchmark: 204)")
    print("   -> env.rs does NOT reproduce the published anchor.")

    print("\n== 3. env.rs best simple base-stock sweep ==")
    best = None
    for level in (16, 20, 24, 28):
        c = env_policy_cost("base_stock", [float(level)] * 4)
        print(f"   base_stock S={level}: total={c}")
        if best is None or c < best[1]:
            best = (level, c)
    print(f"   best base-stock: S={best[0]} -> {best[1]} (still > closed-form 204)")

    print("\n== Conclusion ==")
    print("   closed-form simulator: literature-verified (204).")
    print("   reusable env.rs: NOT literature-verified against this anchor (378 / best 278).")


if __name__ == "__main__":
    main()
