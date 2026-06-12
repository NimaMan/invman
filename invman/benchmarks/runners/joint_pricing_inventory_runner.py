"""Executable baseline runner — `joint_pricing_inventory` (Zhou/Qin-style price ladder).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the joint pricing-and-inventory family runnable from one uniform handle:
expose the single literature-shaped reference instance (a one-item discrete price
ladder with price-dependent stochastic lost-sales demand), read its env params,
and RE-SOLVE the exact finite-horizon DP on the shipped reduced verification
instance so a consumer has the exact optimum and the two structured heuristics
(static-price base-stock, inventory-sensitive base-stock) as the numbers to beat.

This is a single-instance ("primary only") family: there is one runnable env
configuration (`joint_pricing_inventory_primary_reference_instance`) and one
exact-DP comparator (`joint_pricing_inventory_exact_dp_summary`, computed on the
smaller `exact_verification_instance` so the backward induction is tractable).

Score direction (CRITICAL — introspected, not assumed)
------------------------------------------------------
A price-setting newsvendor MAXIMIZES PROFIT, so one might expect a maximization.
But `joint_pricing_inventory_exact_dp_summary` returns a *discounted COST* that
the solver MINIMIZES: `optimal_discounted_cost = -33.178` is the LOWEST value,
`static_discounted_cost = -32.508` is higher (worse), and the reported
`static_gap_to_optimal = +0.670` is POSITIVE (= static is worse than optimal).
The profit shows up only as the negative sign of the cost (cost = negated net
revenue), not as a maximization of the objective. Therefore the framework
convention here is COST minimization with the optimum at the MINIMUM, and we keep
`lower_is_better = True`. (Verified from the actual return keys/signs above.)

How each method serves the objective
------------------------------------
* `list_instances` / `primary_instance` / `_reference_dict` — one runnable
  instance (the primary). `_reference_dict` builds the param dict from the
  primary accessor and raises KeyError on any other name.
* `_published_baselines` — the instance ships no external published cost (it is a
  repo-native, `literature_verified=false` interpretation), so this returns the
  exact-DP optimum + the two heuristic costs from the exact summary as
  recomputed-but-canonical comparators with honest provenance.
* `_run_baselines` — the runnable proof: re-solve the exact finite-horizon DP on
  the live solver and return {optimal (is_optimal=True), static_price_base_stock,
  inventory_sensitive_base_stock}. Re-running reproduces the summary exactly (it
  is a deterministic backward induction, tolerance 1e-6).
* `supports_evaluate = False` — the soft-tree rollout for this family is not wired
  into the uniform `build_policy`/`get_model_fitness` seam, so policy scoring is
  out of scope; the base `_eval_model_and_args` raises an actionable pointer.

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


class JointPricingInventoryRunner(ProblemRunner):
    """Runnable baseline driver for the joint pricing-inventory family (primary only)."""

    problem = "joint_pricing_inventory"
    # The exact-DP comparator is a deterministic backward induction (seed/horizon
    # independent); these protocols only size a learned-policy rollout, which is
    # out of scope here (supports_evaluate=False). Kept for surface uniformity.
    published_protocol = EvalProtocol(
        seeds=(1234, 2025, 7, 99, 1000), horizon=18, warm_up_periods_ratio=0.0
    )
    smoke_protocol = EvalProtocol(seeds=(1234,), horizon=18, warm_up_periods_ratio=0.0)
    # Price-setting newsvendor: profit shows up as NEGATIVE cost, but the exact
    # summary MINIMIZES that cost (optimum = minimum, positive gaps). Cost family.
    lower_is_better = True
    supports_evaluate = False

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust
        self._primary = dict(invman_rust.joint_pricing_inventory_primary_reference_instance())
        self._primary_name = str(self._primary["name"])

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return [self._primary_name]

    def primary_instance(self) -> str:
        return self._primary_name

    def _subfamily_of(self, name: str) -> str:
        return "discrete_price_ladder"

    def _reference_dict(self, name: str) -> dict:
        if str(name) != self._primary_name:
            raise KeyError(
                f"unknown joint_pricing_inventory instance: {name!r}. "
                f"Known: {self.list_instances()} (single-instance primary-only family)"
            )
        return dict(self._primary)

    # -- published (free) baselines ---------------------------------------
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        # The instance is repo-native (literature_verified=false) and ships no
        # external published cost; the canonical comparators are the exact-DP
        # optimum + the two heuristics, surfaced honestly as is_published=False.
        summary = dict(self._rust.joint_pricing_inventory_exact_dp_summary())
        ref = dict(summary.get("verification_reference", {}))
        source = str(ref.get("source", inst_dict.get("source", "")))
        rows = (
            ("optimal", "optimal_discounted_cost", True),
            ("static_price_base_stock", "static_discounted_cost", False),
            ("inventory_sensitive_base_stock", "inventory_sensitive_discounted_cost", False),
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
                    note="exact finite-horizon DP on the reduced verification instance "
                    "(discounted cost; profit = negated cost, minimized)",
                )
            )
        return out

    # -- run the env (the "runnable" proof): exact DP re-solve ------------
    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        try:
            summary = dict(self._rust.joint_pricing_inventory_exact_dp_summary())
        except Exception as exc:  # None-safe: a failed binding must not raise
            etype = type(exc).__name__
            return {
                name: Baseline(
                    name=name,
                    mean_cost=None,
                    source=f"recomputed:joint_pricing_inventory_exact_dp_summary:failed:{etype}",
                )
                for name in ("optimal", "static_price_base_stock", "inventory_sensitive_base_stock")
            }
        rows = (
            ("optimal", "optimal_discounted_cost", "optimal_first_action", True),
            ("static_price_base_stock", "static_discounted_cost", "static_first_action", False),
            (
                "inventory_sensitive_base_stock",
                "inventory_sensitive_discounted_cost",
                "inventory_sensitive_first_action",
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
                    source="recomputed:joint_pricing_inventory_exact_dp_summary:missing_key",
                )
                continue
            out[bname] = Baseline(
                name=bname,
                mean_cost=float(value),
                source="recomputed:joint_pricing_inventory_exact_dp_summary",
                params={"first_action": list(summary.get(action_key, []))},
                is_optimal=is_opt,
                note="exact backward-induction DP (discounted cost, minimized)",
            )
        return out
