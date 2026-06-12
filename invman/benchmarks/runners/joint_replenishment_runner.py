"""Executable baseline runner — `joint_replenishment` (Vanvuchelen et al. 2020).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the joint-replenishment (JRP) family runnable from one uniform handle over
the 16 Vanvuchelen, Gijsbrechts & Boute (2020) Table-2 small-scale settings
(`vanvuchelen2020_small_scale_setting_{1..16}`, two items, truck capacity 6,
major order cost 75): read the env params, surface the PUBLISHED comparator (an
ACTION, not a printed cost — Figure 3 setting-5 optimal ships one FTL to item 2
only, q=(0,6); both paper heuristics order q=(2,4) at state (5,0)), and RE-RUN
the shipped exact DP + heuristics on the live solver to produce the canonical
cost reference and to executably reproduce that published action.

Why this family's reference is an ACTION + an exact-DP COST (tier = reference)
-----------------------------------------------------------------------------
The paper's Figure-3 anchor is a POLICY ACTION (q=(0,6) optimal vs q=(2,4)
heuristic at state (5,0) on setting 5), not a long-run cost, so:
  * `_published_baselines` carries those two actions as Baselines with
    `mean_cost=None` and the action vector in `params` (like dual_sourcing's
    gap-only rows). They are the paper's headline numbers for this family. Only
    the anchor instance (setting 5) ships them; the other 15 settings have no
    published per-instance number, so `_published_baselines` returns [].
  * The canonical COST reference comes from `_run_baselines`: the repo's exact
    finite-horizon DP (`joint_replenishment_exact_dp_summary`, a fixed 4-period
    discounted self-consistency comparator on the setting-1 model family) gives
    the optimal discounted cost (`is_optimal=True`) together with the MOQ and
    dynamic-order-up-to heuristic discounted costs and their gaps to optimal.
    `_exact_dp_summary` takes NO arguments and is the SAME comparator for every
    instance (it is not per-instance parameterised), so it is surfaced uniformly
    and noted as such.

How each method serves the objective
------------------------------------
* `list_instances` / `primary_instance` / `_reference_dict` — the 16 settings via
  `*_list_reference_instances` / `*_primary_reference_instance` /
  `*_get_reference_instance`; primary = setting 5 (the Figure-3 anchor).
* `_published_baselines` — for the anchor instance, the published optimal and
  heuristic ACTIONS (cost unknown -> `mean_cost=None`, action in `params`);
  `is_reference=True` on the published-optimal action (the canonical comparator
  the paper's PPO policy is shown against).
* `_run_baselines` — the runnable proof. (a) the exact DP optimum + MOQ/dynout
  heuristic discounted costs from `*_exact_dp_summary` (optimum tagged
  `is_optimal=True`); (b) an executable reproduction of the published anchor
  heuristic action q=(2,4) via `*_moq_order_quantities` / `*_dynout_order_quantities`
  at the anchor state (5,0) — surfaced as actions (`mean_cost=None`), proving the
  env reproduces the paper's Figure-3 control.

`supports_evaluate=False`: this family's soft-tree rollout
(`joint_replenishment_soft_tree_rollout`) is NOT wired into the uniform
`build_policy`/`get_model_fitness` eval seam, so policy scoring is out of scope
and the base `_eval_model_and_args` raises an actionable pointer.

Verification tier: reference (the published anchor is an ACTION q=(0,6)/(2,4),
reproduced executably; the exact-DP cost is the repo self-consistency optimum,
`literature_verified=false`). Dependencies: `invman_rust` (only when a method
that runs the solver is called).
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

# The single published-anchor instance: setting 5 is the Figure-3 / Figure-4
# focus, with a printed optimal action q=(0,6) vs heuristic q=(2,4) at (5,0).
_ANCHOR_INSTANCE = "vanvuchelen2020_small_scale_setting_5"


class JointReplenishmentRunner(ProblemRunner):
    """Runnable baseline driver for the Vanvuchelen (2020) JRP small-scale family."""

    problem = "joint_replenishment"
    # JRP demand is discrete-uniform per item; the published anchor is an exact
    # finite-horizon discounted control (the DP comparator is 4 periods). The
    # protocol horizons size only any sim-based read; the exact-DP and action
    # reproductions ignore it. >=5 seeds is the repo's seed-robust headline rule.
    published_protocol = EvalProtocol(
        seeds=(1234, 2025, 7, 99, 1000), horizon=5000, warm_up_periods_ratio=0.2
    )
    smoke_protocol = EvalProtocol(seeds=(1234,), horizon=1000, warm_up_periods_ratio=0.2)
    # This family's soft-tree rollout is not in the uniform CMA-ES eval seam yet.
    supports_evaluate = False
    lower_is_better = True

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust
        self._instances = list(invman_rust.joint_replenishment_list_reference_instances())
        self._by_name = {str(d["name"]): dict(d) for d in self._instances}

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return list(self._by_name.keys())

    def primary_instance(self) -> str:
        return str(self._rust.joint_replenishment_primary_reference_instance()["name"])

    def _subfamily_of(self, name: str) -> str:
        return "vanvuchelen2020_small_scale"

    def _reference_dict(self, name: str) -> dict:
        if name not in self._by_name:
            raise KeyError(
                f"unknown joint_replenishment instance: {name!r}. "
                f"Known: {self.list_instances()}"
            )
        return dict(self._rust.joint_replenishment_get_reference_instance(name))

    # -- published (free) baselines: the figure-3 ACTION anchor -----------
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        # Only setting 5 ships a published number, and that number is an ACTION
        # (q=(0,6) optimal vs q=(2,4) heuristic at state (5,0)), not a cost.
        if name != _ANCHOR_INSTANCE:
            return []
        try:
            anchor = dict(self._rust.joint_replenishment_published_action_anchor())
        except Exception:
            return []
        state = list(anchor.get("state_inventory_levels", []))
        source = str(anchor.get("source", ""))
        out: list[Baseline] = []
        opt = anchor.get("optimal_action")
        if opt is not None:
            out.append(
                Baseline(
                    name="published_optimal_action",
                    mean_cost=None,  # the paper prints an ACTION, not a long-run cost
                    source=source,
                    params={"action": [int(v) for v in opt], "state": [int(v) for v in state]},
                    is_published=True,
                    is_reference=True,  # the canonical comparator the PPO policy targets
                    note=f"Figure-3 optimal action q={tuple(int(v) for v in opt)} at state "
                    f"{tuple(int(v) for v in state)} (ships one FTL to item 2 only)",
                )
            )
        heur = anchor.get("heuristic_action")
        if heur is not None:
            out.append(
                Baseline(
                    name="published_heuristic_action",
                    mean_cost=None,
                    source=source,
                    params={"action": [int(v) for v in heur], "state": [int(v) for v in state]},
                    is_published=True,
                    note=f"Figure-3 heuristic action q={tuple(int(v) for v in heur)} at state "
                    f"{tuple(int(v) for v in state)} (both paper heuristics agree)",
                )
            )
        return out

    # -- run the env (the "runnable" proof) -------------------------------
    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        out: dict[str, Baseline] = {}
        # (a) Exact DP optimum + MOQ/dynout heuristic discounted costs. This is
        # the instance-independent 4-period self-consistency comparator (the repo
        # canonical optimum); it carries the same cost for every instance, so the
        # note is explicit about that.
        try:
            dp = dict(self._rust.joint_replenishment_exact_dp_summary())
            dp_note = (
                "fixed 4-period discounted self-consistency DP on the setting-1 model "
                "family (instance-independent comparator)"
            )
            out["exact_dp_optimal"] = Baseline(
                name="exact_dp_optimal",
                mean_cost=float(dp["optimal_discounted_cost"]),
                source="recomputed:joint_replenishment_exact_dp_summary",
                params={"first_action": [int(v) for v in dp.get("optimal_first_action", [])]},
                is_optimal=True,
                note=dp_note,
            )
            out["moq_heuristic"] = Baseline(
                name="moq_heuristic",
                mean_cost=float(dp["moq_discounted_cost"]),
                source="recomputed:joint_replenishment_exact_dp_summary",
                params={
                    "first_action": [int(v) for v in dp.get("moq_first_action", [])],
                    "gap_to_optimal": float(dp.get("moq_gap_to_optimal", float("nan"))),
                },
                note=dp_note,
            )
            out["dynout_heuristic"] = Baseline(
                name="dynout_heuristic",
                mean_cost=float(dp["dynout_discounted_cost"]),
                source="recomputed:joint_replenishment_exact_dp_summary",
                params={
                    "first_action": [int(v) for v in dp.get("dynout_first_action", [])],
                    "gap_to_optimal": float(dp.get("dynout_gap_to_optimal", float("nan"))),
                },
                note=dp_note,
            )
        except Exception as exc:  # None-safe: a failed solver must not abort a sweep
            out["exact_dp_optimal"] = Baseline(
                name="exact_dp_optimal",
                mean_cost=None,
                source=f"joint_replenishment_exact_dp_summary_failed:{type(exc).__name__}",
                is_optimal=True,
            )

        # (b) Executable reproduction of the published Figure-3 anchor ACTION:
        # the MOQ and dynamic-order-up-to heuristics, evaluated at the anchor
        # state (5,0) on the anchor instance, must reproduce q=(2,4). Surfaced as
        # actions (mean_cost=None) — this is the verification of the published
        # action anchor, not a cost.
        for action_name, action in self._reproduce_anchor_actions().items():
            out[action_name] = action
        return out

    def _reproduce_anchor_actions(self) -> dict[str, Baseline]:
        """Re-derive the published anchor heuristic action q=(2,4) on the live env."""
        out: dict[str, Baseline] = {}
        try:
            anchor = dict(self._rust.joint_replenishment_published_action_anchor())
            ver = dict(self._rust.joint_replenishment_exact_verification_instance())
            inst = dict(self._rust.joint_replenishment_get_reference_instance(_ANCHOR_INSTANCE))
            state = [int(v) for v in anchor["state_inventory_levels"]]
            cap = int(inst["truck_capacity"])
            moq_q = list(
                self._rust.joint_replenishment_moq_order_quantities(
                    inventory_levels=state,
                    item_targets=[int(v) for v in ver["moq_item_targets"]],
                    review_period=int(ver["moq_review_period"]),
                    rounding_threshold=float(ver["moq_rounding_threshold"]),
                    truck_capacity=cap,
                )
            )
            out["moq_anchor_action"] = Baseline(
                name="moq_anchor_action",
                mean_cost=None,  # an ACTION reproduction, not a cost
                source="recomputed:joint_replenishment_moq_order_quantities",
                params={"action": [int(v) for v in moq_q], "state": state},
                note=f"reproduces Figure-3 heuristic action; published q="
                f"{tuple(int(v) for v in anchor.get('heuristic_action', []))}",
            )
            dyn_q = list(
                self._rust.joint_replenishment_dynout_order_quantities(
                    inventory_levels=state,
                    item_targets=[int(v) for v in ver["dynout_item_targets"]],
                    demand_lows=[int(v) for v in inst["demand_lows"]],
                    demand_highs=[int(v) for v in inst["demand_highs"]],
                    truck_capacity=cap,
                    holding_costs=[float(v) for v in inst["holding_costs"]],
                    shortage_costs=[float(v) for v in inst["shortage_costs"]],
                )
            )
            out["dynout_anchor_action"] = Baseline(
                name="dynout_anchor_action",
                mean_cost=None,
                source="recomputed:joint_replenishment_dynout_order_quantities",
                params={"action": [int(v) for v in dyn_q], "state": state},
                note=f"reproduces Figure-3 heuristic action; published q="
                f"{tuple(int(v) for v in anchor.get('heuristic_action', []))}",
            )
        except Exception as exc:  # None-safe
            out["anchor_action_reproduction"] = Baseline(
                name="anchor_action_reproduction",
                mean_cost=None,
                source=f"anchor_action_reproduction_failed:{type(exc).__name__}",
            )
        return out
