"""Executable baseline runner — `random_yield_inventory` (Yan 2026 all-or-nothing yield).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the random-yield family runnable from one uniform handle: expose the single
literature-shaped reference instance (a one-item finite-horizon discounted system
with an all-or-nothing yield pattern and a non-zero lead time), read its env
params, RE-SOLVE the exact finite-horizon DP on the shipped reduced verification
instance for the exact optimum, and RE-SIMULATE the two structured heuristics
(weighted-newsvendor, linear-inflation / yield-inflated base-stock) on the live
env so a consumer has the numbers to beat. tier = faithful.

Instance scope (introspected `_literature_benchmark_families`)
--------------------------------------------------------------
`random_yield_inventory_literature_benchmark_families` enumerates SIX named
literature families, but every one is metadata-only for the EXECUTABLE env:
  * yan2026_small_scale_exact_dp_family / chen2018_weighted_newsvendor_family —
    `reported_numbers_available=false`, `repo_assertion_basis=
    do_not_use_for_repo_assertions` (preview/bibliographic only).
  * the four inderfurth2015_* grids — `reported_numbers_available=true` BUT
    `repo_assertion_basis=related_model_aggregate_only` and a DIFFERENT yield
    model (binomial / proportional, not the repo's all-or-nothing), with only
    grid-summary aggregates, not single comparable costs.
None is a runnable instance of THIS env with a copyable benchmark cost, so this is
a single runnable instance ("primary only"). The literature families are surfaced
as zero-cost provenance baselines (mean_cost=None) so a consumer can see the
bibliographic anchors without mistaking them for executable numbers.

Score direction (introspected)
------------------------------
`random_yield_inventory_exact_dp_summary` returns a *discounted COST* that is
MINIMIZED: `optimal_discounted_cost = 40.060` (lowest),
`linear_inflation_discounted_cost = 47.714` and
`weighted_newsvendor_discounted_cost = 60.394` (higher = worse), and both reported
gaps are POSITIVE. Cost family -> `lower_is_better = True`, optimum at the minimum.

How each method serves the objective
------------------------------------
* `list_instances` / `primary_instance` / `_reference_dict` — one runnable
  instance (the primary); `_reference_dict` raises KeyError on any other name.
* `_published_baselines` — the exact-DP optimum + the two heuristic costs from the
  exact summary (recomputed-but-canonical, is_published=False), plus the six
  literature families as provenance-only baselines (mean_cost=None).
* `_run_baselines` — the runnable proof. Two complementary pieces:
    (1) the exact finite-horizon DP optimum (is_optimal=True), deterministic,
        reproduces `_exact_dp_summary` to <1e-6; and
    (2) the weighted-newsvendor and linear-inflation heuristics RE-SIMULATED on
        the live env over the protocol seeds via
        `_policy_discounted_cost_summary` (linear-inflation parameterised by
        `_linear_inflation_parameters`), returning mean +/- std cost.
  These two pieces are on DIFFERENT instances (exact = reduced verification
  instance; simulation = the larger primary instance), so they are not equal —
  they bracket the operating region from above and below honestly.
* `supports_evaluate = False` — the soft-tree rollout is not wired into the
  uniform `build_policy`/`get_model_fitness` seam; the base raises a pointer.

Verification note: the exact DP reproduces `_exact_dp_summary` to <1e-6; the
simulated heuristics reproduce `_policy_discounted_cost_summary` exactly for the
same seeds (same binding, deterministic per seed).
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


class RandomYieldInventoryRunner(ProblemRunner):
    """Runnable baseline driver for the random-yield family (primary only)."""

    problem = "random_yield_inventory"
    # The heuristics are simulated over the seed list (>=5 = repo seed-robust
    # rule); the exact DP is seed-independent. Horizon = the instance's periods.
    published_protocol = EvalProtocol(
        seeds=(1, 2, 3, 4, 5), horizon=12, warm_up_periods_ratio=0.0
    )
    smoke_protocol = EvalProtocol(seeds=(1, 2, 3, 4, 5), horizon=12, warm_up_periods_ratio=0.0)
    lower_is_better = True
    supports_evaluate = False

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust
        self._primary = dict(invman_rust.random_yield_inventory_primary_reference_instance())
        self._primary_name = str(self._primary["name"])

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return [self._primary_name]

    def primary_instance(self) -> str:
        return self._primary_name

    def _subfamily_of(self, name: str) -> str:
        return "all_or_nothing_yield"

    def _reference_dict(self, name: str) -> dict:
        if str(name) != self._primary_name:
            raise KeyError(
                f"unknown random_yield_inventory instance: {name!r}. "
                f"Known: {self.list_instances()} (single runnable primary; the six "
                f"literature_benchmark_families are metadata-only / different yield model)"
            )
        return dict(self._primary)

    # -- published (free) baselines ---------------------------------------
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        out: list[Baseline] = []
        # (a) exact-DP optimum + the two heuristic costs (repo solver, honest).
        summary = dict(self._rust.random_yield_inventory_exact_dp_summary())
        ref = dict(summary.get("verification_reference", {}))
        source = str(ref.get("source", inst_dict.get("source", "")))
        rows = (
            ("optimal", "optimal_discounted_cost", True),
            ("linear_inflation", "linear_inflation_discounted_cost", False),
            ("weighted_newsvendor", "weighted_newsvendor_discounted_cost", False),
        )
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
        # (b) the six literature families as provenance-only anchors (no cost).
        try:
            for fam in self._rust.random_yield_inventory_literature_benchmark_families():
                fam = dict(fam)
                out.append(
                    Baseline(
                        name=f"literature_family:{fam.get('name')}",
                        mean_cost=None,  # metadata-only / different yield model
                        source=str(fam.get("source", "")),
                        is_published=True,
                        params={
                            "reported_numbers_available": bool(
                                fam.get("reported_numbers_available", False)
                            ),
                            "repo_assertion_basis": str(fam.get("repo_assertion_basis", "")),
                            "yield_model": str(fam.get("yield_model", "")),
                        },
                        note="literature provenance anchor (not an executable cost for this env)",
                    )
                )
        except Exception:
            pass
        return out

    # -- run the env (the "runnable" proof) -------------------------------
    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        out: dict[str, Baseline] = {}
        # (1) exact-DP optimum (deterministic, reduced verification instance).
        try:
            summary = dict(self._rust.random_yield_inventory_exact_dp_summary())
            out["optimal"] = Baseline(
                name="optimal",
                mean_cost=float(summary["optimal_discounted_cost"]),
                source="recomputed:random_yield_inventory_exact_dp_summary",
                params={"first_action": summary.get("optimal_first_action")},
                is_optimal=True,
                note="exact backward-induction DP (discounted cost, minimized)",
            )
        except Exception as exc:  # None-safe
            out["optimal"] = Baseline(
                name="optimal",
                mean_cost=None,
                source=f"recomputed:random_yield_inventory_exact_dp_summary:failed:{type(exc).__name__}",
            )
        # (2) heuristics re-simulated on the live primary env over the seed list.
        p = inst.params
        seeds = [int(s) for s in protocol.seeds]
        common = dict(
            initial_inventory_level=float(p["initial_inventory_level"]),
            pipeline_orders=[float(x) for x in p["initial_pipeline_orders"]],
            periods=int(p["periods"]),
            seeds=seeds,
            demand_mean=float(p["demand_mean"]),
            success_probability=float(p["success_probability"]),
            holding_cost=float(p["holding_cost"]),
            shortage_cost=float(p["shortage_cost"]),
            procurement_cost=float(p["procurement_cost"]),
            discount_factor=float(p["discount_factor"]),
        )
        # weighted-newsvendor and yield-inflated base-stock need no params; the
        # linear-inflation heuristic is parameterised by the closed-form
        # (target_stock_level, yield_inflation_factor) from the Rust helper.
        try:
            li = self._rust.random_yield_inventory_linear_inflation_parameters(
                float(p["demand_mean"]),
                float(p["success_probability"]),
                int(p["lead_time"]),
                float(p["holding_cost"]),
                float(p["shortage_cost"]),
            )
            li_params = [float(li[0]), float(li[1])]
        except Exception:
            li_params = None
        policies = (
            ("weighted_newsvendor", []),
            ("yield_inflated_base_stock", []),
            ("linear_inflation", li_params),
        )
        for pname, params in policies:
            if params is None:
                out[pname] = Baseline(
                    name=pname,
                    mean_cost=None,
                    source="recomputed:linear_inflation_parameters:failed",
                )
                continue
            try:
                s = dict(
                    self._rust.random_yield_inventory_policy_discounted_cost_summary(
                        policy_name=pname, params=[float(x) for x in params], **common
                    )
                )
                out[pname] = Baseline(
                    name=pname,
                    mean_cost=float(s["mean_cost"]),
                    std_cost=float(s.get("cost_std")) if s.get("cost_std") is not None else None,
                    source="recomputed:random_yield_inventory_policy_discounted_cost_summary",
                    params={"params": list(params), "num_samples": s.get("num_samples")},
                    note=f"live env, primary instance, seeds={seeds}",
                )
            except Exception as exc:  # None-safe: a failed policy must not abort
                out[pname] = Baseline(
                    name=pname,
                    mean_cost=None,
                    source=f"recomputed:random_yield_inventory_policy_discounted_cost_summary:"
                    f"failed:{type(exc).__name__}",
                )
        return out
