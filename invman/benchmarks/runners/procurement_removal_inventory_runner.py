"""Executable baseline runner — `procurement_removal_inventory` (Maggiar/Sadighian style).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the procurement-removal family runnable from one uniform handle: expose the
single literature-shaped reference instance (a one-item finite-horizon system
with a per-period cap on returnable purchases, explicit return + liquidation
credits, and shortage penalties), read its env params, and RE-SOLVE the exact
finite-horizon DP on the shipped reduced verification instance so a consumer has
the exact cost optimum and the two structured heuristics (interval-stock
order/remove, returnability-buffer interval-stock) as the numbers to beat.

This is a single-instance ("primary only") family: one runnable env
configuration (`procurement_removal_inventory_primary_reference_instance`) and
one exact-DP comparator (`procurement_removal_inventory_exact_dp_summary`,
computed on the smaller `exact_verification_instance` so the backward induction is
tractable). tier = faithful (repo-native; `literature_verified=false`).

Score direction (introspected)
------------------------------
`procurement_removal_inventory_exact_dp_summary` returns a *discounted COST* that
is MINIMIZED: `optimal_discounted_cost = 31.780` (lowest),
`interval_stock_discounted_cost = 34.164` (higher = worse), and the reported
`interval_stock_gap_to_optimal = +2.384` is POSITIVE (= heuristic worse than
optimal). This is a true cost minimization, so `lower_is_better = True` with the
optimum at the MINIMUM. (Verified from the actual return keys/signs.)

How each method serves the objective
------------------------------------
* `list_instances` / `primary_instance` / `_reference_dict` — one runnable
  instance (the primary); `_reference_dict` raises KeyError on any other name.
* `_published_baselines` — the instance ships no external published cost (it is
  repo-native), so this surfaces the exact-DP optimum + the two heuristic costs
  from the exact summary as recomputed-but-canonical comparators (is_published
  =False, honest provenance).
* `_run_baselines` — the runnable proof: re-solve the exact finite-horizon DP on
  the live solver and return {optimal (is_optimal=True), interval_stock,
  returnability_buffer_interval_stock}. Re-running reproduces the summary exactly
  (deterministic backward induction, tolerance 1e-6).
* `supports_evaluate = False` — the soft-tree rollout for this family is not wired
  into the uniform `build_policy`/`get_model_fitness` seam; the base
  `_eval_model_and_args` raises an actionable pointer.

Verification note: the exact DP is a deterministic backward induction on a small
discrete instance; `run_baselines` reproduces `_exact_dp_summary` to <1e-6.
Dependencies: `invman_rust` (only when an env-running method is called).
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


class ProcurementRemovalInventoryRunner(ProblemRunner):
    """Runnable baseline driver for the procurement-removal family (primary only)."""

    problem = "procurement_removal_inventory"
    # The exact-DP comparator is a deterministic backward induction; these
    # protocols only size a learned-policy rollout, which is out of scope here
    # (supports_evaluate=False). Kept for surface uniformity.
    published_protocol = EvalProtocol(
        seeds=(1234, 2025, 7, 99, 1000), horizon=16, warm_up_periods_ratio=0.0
    )
    smoke_protocol = EvalProtocol(seeds=(1234,), horizon=16, warm_up_periods_ratio=0.0)
    lower_is_better = True
    supports_evaluate = False

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust
        self._primary = dict(
            invman_rust.procurement_removal_inventory_primary_reference_instance()
        )
        self._primary_name = str(self._primary["name"])

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return [self._primary_name]

    def primary_instance(self) -> str:
        return self._primary_name

    def _subfamily_of(self, name: str) -> str:
        return "fixed_returnability_quota"

    def _reference_dict(self, name: str) -> dict:
        if str(name) != self._primary_name:
            raise KeyError(
                f"unknown procurement_removal_inventory instance: {name!r}. "
                f"Known: {self.list_instances()} (single-instance primary-only family)"
            )
        return dict(self._primary)

    # -- published (free) baselines ---------------------------------------
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        summary = dict(self._rust.procurement_removal_inventory_exact_dp_summary())
        ref = dict(summary.get("verification_reference", {}))
        source = str(ref.get("source", inst_dict.get("source", "")))
        rows = (
            ("optimal", "optimal_discounted_cost", True),
            ("interval_stock", "interval_stock_discounted_cost", False),
            (
                "returnability_buffer_interval_stock",
                "returnability_buffer_discounted_cost",
                False,
            ),
        )
        out: list[Baseline] = []
        for bname, key, is_opt in rows:
            value = summary.get(key)
            if value is None:
                continue
            out.append(
                Baseline(
                    name=bname,
                    mean_cost=float(value),
                    source=source,
                    is_published=False,  # repo exact solver, not a literature number
                    is_optimal=is_opt,
                    note="exact finite-horizon DP on the reduced verification instance",
                )
            )
        return out

    # -- run the env (the "runnable" proof): exact DP re-solve ------------
    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        try:
            summary = dict(self._rust.procurement_removal_inventory_exact_dp_summary())
        except Exception as exc:  # None-safe: a failed binding must not raise
            etype = type(exc).__name__
            return {
                name: Baseline(
                    name=name,
                    mean_cost=None,
                    source=f"recomputed:procurement_removal_inventory_exact_dp_summary:failed:{etype}",
                )
                for name in (
                    "optimal",
                    "interval_stock",
                    "returnability_buffer_interval_stock",
                )
            }
        rows = (
            ("optimal", "optimal_discounted_cost", "optimal_first_action", True),
            (
                "interval_stock",
                "interval_stock_discounted_cost",
                "interval_stock_first_action",
                False,
            ),
            (
                "returnability_buffer_interval_stock",
                "returnability_buffer_discounted_cost",
                "returnability_buffer_first_action",
                False,
            ),
        )
        out: dict[str, Baseline] = {}
        for bname, cost_key, action_key, is_opt in rows:
            value = summary.get(cost_key)
            if value is None:
                out[bname] = Baseline(
                    name=bname,
                    mean_cost=None,
                    source="recomputed:procurement_removal_inventory_exact_dp_summary:missing_key",
                )
                continue
            out[bname] = Baseline(
                name=bname,
                mean_cost=float(value),
                source="recomputed:procurement_removal_inventory_exact_dp_summary",
                params={"first_action": list(summary.get(action_key, []))},
                is_optimal=is_opt,
                note="exact backward-induction DP (discounted cost, minimized)",
            )
        return out
