"""Executable baseline runner — `decentralized_inventory_control` (four-stage Beer Game).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the decentralized four-stage Beer Game runnable from one uniform handle over
its single canonical instance, `beer_game_classic_four_stage`: read the env
parameters, carry the PUBLISHED Sterman (1989) closed-form anchor-and-adjust cost
(total 204, per-agent [46,50,54,54]) as the literature reference, and RE-RUN both
the Sterman anchor-and-adjust policy and the best constant base-stock on the live
reusable env.rs MDP to expose the absolute costs a learned policy must beat.

The split: closed-form 204 vs. trainable env.rs (the honest caveat)
-------------------------------------------------------------------
This family is verification tier = reference, but SPLIT (per
`STERMAN_1989_CLASSIC_BENCHMARK.notes`):
  * The closed-form board-game port reproduces Sterman/Edali-Yasarcan
    [46,50,54,54]/204 EXACTLY — exposed by
    `decentralized_inventory_control_classic_sterman_literature_summary()`. This
    204 is a property of the closed-form bookkeeping ONLY.
  * The reusable, trainable `env.rs` MDP (which the heuristics + learned soft-tree
    actually run on) is a different — also valid — decentralized serial MDP whose
    pipeline/supply-line bookkeeping differs. Under the SAME parameters it does
    NOT reproduce 204: anchor-and-adjust -> 378, best simple base-stock S=24 ->
    278 (both measured live via
    `decentralized_inventory_control_policy_rollout_from_paths`).
So the published 204 (closed-form) is the literature anchor (`is_published=True`,
`is_reference=True`), while `_run_baselines` returns the closed-form 204 AND the
two env.rs costs (anchor 378, best base-stock 278, the latter the closed-form
base-stock OPTIMUM over the 36-week path, `is_optimal=True` for the env.rs MDP).
This is the cleanest honest comparison: the published board-game number plus the
runnable env's own optima.

There is NO Python accessor that returns the instance param dict, so
`_reference_dict` builds it from the canonical `PRIMARY_REFERENCE_INSTANCE`
constant in `src/problems/decentralized_inventory_control/literature/
references.rs` (transcribed into this file's `_PRIMARY_PARAMS`). One instance;
`_reference_dict` raises KeyError on anything else.

How each method serves the objective
------------------------------------
* `_reference_dict` / `_subfamily_of` — the env params (free, from the literature
  constant) tagged with the Beer-Game regime.
* `_published_baselines` — the Sterman (1989) closed-form anchor-and-adjust cost
  204 (the literature comparator the family improves over -> `is_reference=True`)
  plus its per-agent split, read from the closed-form literature summary.
* `_run_baselines` — the runnable proof: (1) re-run the closed-form board-game
  summary (reproduces 204), (2) re-run anchor-and-adjust on the live env.rs over
  the 36-week demand path (-> 378), (3) search the best constant base-stock S on
  the live env.rs (-> 278 at S=24, the env.rs closed-form path optimum,
  `is_optimal=True`). Costs are total holding+backlog COSTS, `lower_is_better=True`.

`supports_evaluate = False`: the soft-tree rollout
(`decentralized_inventory_control_soft_tree_rollout`) is NOT wired through the
uniform `build_policy`/`get_model_fitness` seam, so policy scoring is out of
scope; the params + published 204 + `run_baselines` above ARE runnable.

Verification tier: reference (closed-form 204 reference-grade; env.rs
faithful_unverified, does NOT reproduce 204 — carried honestly).
Dependencies: `invman_rust`.
================================================================================
"""

from __future__ import annotations

from typing import Optional

from invman.benchmarks.runners.base import (
    Baseline,
    EvalProtocol,
    ProblemRunner,
    ReferenceInstance,
)

# Canonical 36-week deterministic Beer-Game demand path (4,4,4,4,8,...,8),
# transcribed from CLASSIC_BEER_GAME_CUSTOMER_DEMANDS in references.rs.
_CLASSIC_DEMANDS = [4, 4, 4, 4] + [8] * 32

# Canonical single reference instance, transcribed from the Rust constant
# `PRIMARY_REFERENCE_INSTANCE` in
# src/problems/decentralized_inventory_control/literature/references.rs (no Python
# accessor returns this param dict).
_PRIMARY_INSTANCE_NAME = "beer_game_classic_four_stage"
_PRIMARY_PARAMS = {
    "name": _PRIMARY_INSTANCE_NAME,
    "source": (
        "Edali, M. & Yasarcan, H. (2014), A Mathematical Model of the Beer Game, JASSS "
        "17(4):2 (reproduces Sterman 1989 benchmark)"
    ),
    "url": "https://www.jasss.org/17/4/2.html",
    "num_agents": 4,
    "customer_demands": list(_CLASSIC_DEMANDS),
    "shipment_lead_times": [2, 2, 2, 2],
    "order_lead_times": [0, 1, 1, 1],
    "initial_on_hand_inventory": [12, 12, 12, 12],
    "initial_backlog": [0, 0, 0, 0],
    "initial_shipment_pipelines": [[4, 4], [4, 4], [4, 4], [4, 4]],
    "initial_order_pipelines": [[], [4], [4], [4]],
    "initial_last_received_shipments": [4, 4, 4, 4],
    "initial_last_received_orders": [4, 4, 4, 4],
    "initial_forecast_orders": [4.0, 4.0, 4.0, 4.0],
    "initial_last_actions": [4, 4, 4, 4],
    "holding_costs": [0.5, 0.5, 0.5, 0.5],
    "backlog_costs": [1.0, 1.0, 1.0, 1.0],
    "demand_smoothing_factors": [0.0, 0.0, 0.0, 0.0],
    "sterman_target_positions": [28.0, 28.0, 28.0, 20.0],
    "sterman_adjustment_times": [1.0, 1.0, 1.0, 1.0],
    "sterman_supply_line_weights": [1.0, 1.0, 1.0, 1.0],
    # Published closed-form Sterman benchmark (board-game bookkeeping only).
    "published_sterman_per_agent_costs": [46.0, 50.0, 54.0, 54.0],
    "published_sterman_total_cost": 204.0,
    "notes": (
        "Classic 36-week four-stage Beer Game. Sterman (1989) closed-form anchor-adjust "
        "cost 204 ([46,50,54,54]) reproduced ONLY by the board-game port; the reusable "
        "env.rs MDP yields anchor-adjust 378 / best base-stock S=24 -> 278 under "
        "identical parameters (env.rs is faithful_unverified, NOT calibrated to 204)."
    ),
}

# Candidate constant base-stock levels for the env.rs best-base-stock search; the
# published optimum on this path is S=24 (-> 278 per references.rs).
_BASE_STOCK_GRID = list(range(16, 33))


class DecentralizedInventoryControlRunner(ProblemRunner):
    """Runnable baseline driver for the four-stage Beer-Game family."""

    problem = "decentralized_inventory_control"
    # The classic instance is a single DETERMINISTIC 36-week demand path, so the
    # cost is exact (no Monte-Carlo); the seed list / horizon are carried for
    # interface uniformity but the path-based rollout ignores them (the path IS
    # the protocol). Total = undiscounted holding+backlog cost.
    published_protocol = EvalProtocol(
        seeds=(1234,), horizon=len(_CLASSIC_DEMANDS), warm_up_periods_ratio=0.0
    )
    smoke_protocol = EvalProtocol(
        seeds=(1234,), horizon=len(_CLASSIC_DEMANDS), warm_up_periods_ratio=0.0
    )
    supports_evaluate = False
    lower_is_better = True  # total holding + backlog COST

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return [_PRIMARY_INSTANCE_NAME]

    def primary_instance(self) -> str:
        return _PRIMARY_INSTANCE_NAME

    def _subfamily_of(self, name: str) -> str:
        return "classic_four_stage_beer_game"

    def _reference_dict(self, name: str) -> dict:
        if name != _PRIMARY_INSTANCE_NAME:
            raise KeyError(
                f"unknown decentralized_inventory_control instance: {name!r}. "
                f"Known: {self.list_instances()} (the reduced exact-DP verifier has no "
                f"single-contract reference dict here)."
            )
        return dict(_PRIMARY_PARAMS)

    # -- published (free) baselines: the Sterman closed-form cost ---------
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        """The Sterman (1989) closed-form anchor-and-adjust cost (literature anchor)."""
        total = inst_dict.get("published_sterman_total_cost")
        if total is None:
            return []
        return [
            Baseline(
                name="sterman_anchor_adjust_closed_form",
                mean_cost=float(total),
                source=str(inst_dict.get("source", "")),
                params={
                    "per_agent_costs": list(
                        inst_dict.get("published_sterman_per_agent_costs", [])
                    ),
                    "scope": "closed_form_board_game_only",
                },
                is_published=True,
                # The canonical literature comparator the family improves over.
                # NOT is_optimal: 204 is the optimized anchor-adjust cost, not a
                # proven MDP optimum, and is a property of the closed-form
                # bookkeeping (env.rs yields 378 under identical parameters).
                is_reference=True,
                note=(
                    "Sterman (1989) / Edali-Yasarcan (2014) closed-form board-game "
                    "benchmark; reproduced ONLY by the board-game port, NOT by env.rs"
                ),
            )
        ]

    # -- run the env (the "runnable" proof) -------------------------------
    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        p = inst.params
        out: dict[str, Baseline] = {}

        # (1) Closed-form board-game summary — reproduces the published 204.
        try:
            summary = dict(
                self._rust.decentralized_inventory_control_classic_sterman_literature_summary()
            )
            out["sterman_anchor_adjust_closed_form"] = Baseline(
                name="sterman_anchor_adjust_closed_form",
                mean_cost=float(summary["total_cost"]),
                source="recomputed:decentralized_inventory_control_classic_sterman_literature_summary",
                params={"per_agent_costs": list(summary.get("per_agent_costs", []))},
                is_published=True,
                is_reference=True,
                note="closed-form board-game port (reproduces published 204)",
            )
        except Exception as exc:  # None-safe
            out["sterman_anchor_adjust_closed_form"] = Baseline(
                name="sterman_anchor_adjust_closed_form",
                mean_cost=None,
                source=f"classic_sterman_literature_summary_failed:{type(exc).__name__}",
            )

        # Shared env.rs path-rollout kwargs (the reusable, trainable MDP).
        common = dict(
            on_hand_inventory=[int(v) for v in p["initial_on_hand_inventory"]],
            backlog=[int(v) for v in p["initial_backlog"]],
            shipment_pipelines=[[int(x) for x in row] for row in p["initial_shipment_pipelines"]],
            order_pipelines=[[int(x) for x in row] for row in p["initial_order_pipelines"]],
            last_received_shipments=[int(v) for v in p["initial_last_received_shipments"]],
            last_received_orders=[int(v) for v in p["initial_last_received_orders"]],
            forecast_orders=[float(v) for v in p["initial_forecast_orders"]],
            last_actions=[int(v) for v in p["initial_last_actions"]],
            customer_demands=[int(v) for v in p["customer_demands"]],
            demand_smoothing_factors=[float(v) for v in p["demand_smoothing_factors"]],
            holding_costs=[float(v) for v in p["holding_costs"]],
            backlog_costs=[float(v) for v in p["backlog_costs"]],
            discount_factor=1.0,  # undiscounted total cost (board-game convention)
        )

        # (2) Anchor-and-adjust on the live env.rs MDP (-> 378, the honest split).
        try:
            sterman_params = (
                [float(v) for v in p["sterman_target_positions"]]
                + [float(v) for v in p["sterman_adjustment_times"]]
                + [float(v) for v in p["sterman_supply_line_weights"]]
            )
            cost = float(
                self._rust.decentralized_inventory_control_policy_rollout_from_paths(
                    policy_name="sterman_anchor_adjust", params=sterman_params, **common
                )
            )
            out["sterman_anchor_adjust_env"] = Baseline(
                name="sterman_anchor_adjust_env",
                mean_cost=cost,
                source="recomputed:decentralized_inventory_control_policy_rollout_from_paths",
                note=(
                    "anchor-and-adjust on the reusable env.rs MDP; does NOT reproduce "
                    "the closed-form 204 (env.rs is faithful_unverified)"
                ),
            )
        except Exception as exc:  # None-safe
            out["sterman_anchor_adjust_env"] = Baseline(
                name="sterman_anchor_adjust_env",
                mean_cost=None,
                source=f"policy_rollout_from_paths_failed:{type(exc).__name__}",
            )

        # (3) Best constant base-stock S on the live env.rs MDP (-> 278 at S=24,
        #     the env.rs closed-form path optimum over the deterministic demand).
        try:
            best_cost: Optional[float] = None
            best_S: Optional[int] = None
            for S in _BASE_STOCK_GRID:
                cost = float(
                    self._rust.decentralized_inventory_control_policy_rollout_from_paths(
                        policy_name="base_stock",
                        params=[float(S)] * int(p["num_agents"]),
                        **common,
                    )
                )
                if best_cost is None or cost < best_cost:
                    best_cost, best_S = cost, S
            out["best_constant_base_stock_env"] = Baseline(
                name="best_constant_base_stock_env",
                mean_cost=best_cost,
                source="recomputed:decentralized_inventory_control_policy_rollout_from_paths",
                params={"base_stock_level": best_S, "grid": [_BASE_STOCK_GRID[0], _BASE_STOCK_GRID[-1]]},
                # The closed-form optimum over the deterministic 36-week path on the
                # env.rs MDP (the strongest static comparator a learned policy beats).
                is_optimal=True,
                note=f"best constant base-stock on env.rs over the 36-week path (S={best_S})",
            )
        except Exception as exc:  # None-safe
            out["best_constant_base_stock_env"] = Baseline(
                name="best_constant_base_stock_env",
                mean_cost=None,
                source=f"policy_rollout_from_paths_failed:{type(exc).__name__}",
            )

        return out
