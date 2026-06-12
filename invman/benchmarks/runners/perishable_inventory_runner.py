"""Executable baseline runner — `perishable_inventory` (De Moor 2022 / Farrington 2025).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the perishable-inventory family RUNNABLE from one uniform handle over the 32
De Moor, Gijsbrechts & Boute (2022) "Scenario A" reference instances
(`de_moor2022_m{2..5}_exp{1..8}_*`, exposed by
`perishable_inventory_list_reference_instances`): read each instance's env
params, read the PUBLISHED literature numbers, and RE-RUN the shipped baselines
on the live env so the recomputed numbers reproduce the published ones on the
SAME scale.

THE SCALE IS A DISCOUNTED RETURN, NOT AN AVERAGE COST
-----------------------------------------------------
This family is evaluated on a gamma=0.99 DISCOUNTED RETURN over the published
465-period horizon (100 warm-up + 365 evaluation periods), exactly as in
Farrington et al. (2025) Table 3 — the published headline numbers are *returns*
(negative, higher = better): e.g. the primary instance has value-iteration mean
return -1457 and best base-stock mean return -1474. There is no discount-factor
field on the instance dict because gamma=0.99 is fixed inside the Rust bindings,
and the discounted-return objective is implied by the `published_scenario_a_returns`
block. So this runner is deliberately wired to the DISCOUNTED-RETURN baseline
variants (`*_discounted_return_summary`, `_exact_mdp_summary`'s
`value_iteration_mean_return`), NOT the average-cost variants.

Because the shared `ProblemRunner` contract is `lower_is_better=True` (a COST), we
report `mean_cost = -mean_return` (a positive number, e.g. 1474). Negating a
"higher is better" return gives a "lower is better" cost on a consistent scale,
so `reference_baseline`, `compare`, and the published-vs-recomputed comparison all
stay correct. The published optimality-gap % is carried verbatim as a note.

Why each method serves the objective
------------------------------------
* `_reference_dict` / `_published_baselines` — read the env params + the free
  published numbers (no simulation). The published optimum is Farrington's
  value-iteration mean return (-> cost = -return, `is_optimal=True`); the
  canonical comparator a learned policy must beat is the published best
  base-stock mean return (-> `is_reference=True`); the published optimality gap
  is attached as a note. These literals exist for ALL 32 instances.
* `_run_baselines` — the "runnable env" proof on the discounted-return scale:
    - exact optimum: `perishable_inventory_exact_mdp_summary` solves the exact MDP
      by value iteration (gamma=0.99) and returns `value_iteration_mean_return`,
      which reproduces Farrington's value-iteration return EXACTLY for the small
      instances (state_count <= 2000 => only the m2/m3-L1 instances). For the
      20 larger instances the exact solver is intentionally disabled in Rust and
      raises ValueError; we record that None-safely and rely on the published
      optimum literal instead.
    - best constant base-stock (the canonical reference): grid-search the order-
      up-to level S in [0, max_order_size] on the live simulator via
      `perishable_inventory_base_stock_search_discounted_return_summary` over the
      published 128-seed protocol; reproduces the published best base-stock
      return (-1474 -> recomputed -1474.6 on the primary, ~0.04%).
    - BSP-low-EW (optional heavier 3-parameter De Moor benchmark heuristic) via
      `perishable_inventory_bsp_low_ew_search_discounted_return_summary`.
  All three negate the return to a cost and are None-safe (a failed binding ->
  `mean_cost=None`, never raises).

The reproduction mirrors `scripts/perishable_inventory/{common,validate_against_papers}.py`
exactly (zero initial state, horizon=465, warm_up=100/465, position_upper_bound=
max_order_size, gamma=0.99, seeds=range(123, 123+N)), kept here as the uniform
runner surface so a benchmark consumer never has to reach into `scripts/`.

`supports_evaluate = False`: this family's soft-tree rollout
(`perishable_inventory_soft_tree_discounted_return`) is NOT yet wired through the
uniform `build_policy` / `get_model_fitness` seam, so policy scoring is out of
scope; the base default raises an actionable error.

Verification tier: STRICT. The exact value-iteration return reproduces
Farrington Table 3 EXACTLY on the small instances; the best base-stock search
reproduces the published best base-stock return within ~0.1% at the published
128-seed protocol (state-dependent Monte-Carlo tolerance). The De Moor (2022)
optimal-policy table + base-stock level are an exact integer match (the
`matches_published_*` flags in the exact MDP summary).
Dependencies: `invman_rust` (only when a running method is called).
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


class PerishableInventoryRunner(ProblemRunner):
    """Runnable baseline driver for the perishable_inventory family (discounted return)."""

    problem = "perishable_inventory"
    # This family's soft-tree rollout is NOT yet wired through the uniform
    # build_policy / get_model_fitness eval seam, so policy scoring is out of
    # scope; _eval_model_and_args is intentionally not overridden and the base
    # default raises an actionable error. The env params + published baselines +
    # run_baselines() below ARE runnable.
    supports_evaluate = False
    # Discounted-return objective => lower COST is better (cost = -return).
    lower_is_better = True
    # The published Farrington (2025) reproduction averages the gamma=0.99
    # discounted return over 128 seeds (seeds 123..250) at the instance's own
    # 465-period horizon / 100-period warm-up. 128 seeds is what reproduces the
    # published best base-stock return to ~0.1%; >=5 seeds is the repo headline
    # rule and is amply satisfied. _run_baselines pins horizon/warm-up to the
    # instance (the values that DEFINE the published scale) and uses these seeds.
    published_protocol = EvalProtocol(
        seeds=tuple(range(123, 123 + 128)), horizon=465, warm_up_periods_ratio=100.0 / 465.0
    )
    # Fast "does my harness run?" protocol: a 16-seed search is noisy (it can sit
    # a few % off the published return and pick the grid boundary on the largest
    # instances) but proves the env runs; use published_protocol for a faithful
    # reproduction.
    smoke_protocol = EvalProtocol(
        seeds=tuple(range(123, 123 + 16)), horizon=465, warm_up_periods_ratio=100.0 / 465.0
    )
    default_structure = {
        "depth": 2,
        "temperature": 0.25,
        "split_type": "oblique",
        "leaf_type": "linear",
    }
    #: gamma for the discounted-return objective (fixed by the Rust bindings).
    _GAMMA = 0.99
    #: top_k retained by the heuristic searches (matches scripts/.../common.py).
    _SEARCH_TOP_K = 12
    #: Include the heavier 3-parameter BSP-low-EW benchmark heuristic in
    #: _run_baselines. The base-stock search alone is the canonical comparator;
    #: BSP-low-EW adds a slower (3D grid) second De Moor benchmark.
    _RUN_BSP_LOW_EW = True

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust
        self._names = [
            str(d["name"]) for d in invman_rust.perishable_inventory_list_reference_instances()
        ]
        self._set = set(self._names)

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return list(self._names)

    def primary_instance(self) -> str:
        return str(self._rust.perishable_inventory_primary_reference_instance_name())

    def _subfamily_of(self, name: str) -> str:
        # All 32 instances are the De Moor (2022) Scenario A grid reused by
        # Farrington (2025); one constant tag suffices.
        return "de_moor2022_scenario_a"

    # -- reference dicts --------------------------------------------------
    def _reference_dict(self, name: str) -> dict:
        if name not in self._set:
            raise KeyError(
                f"unknown perishable_inventory instance: {name!r}. Known: {self.list_instances()}"
            )
        return dict(self._rust.perishable_inventory_get_reference_instance(name))

    # -- published (free) baselines ---------------------------------------
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        """Farrington (2025) Table 3 returns, mapped to costs (cost = -return).

        The published numbers are gamma=0.99 discounted RETURNS (negative, higher
        = better). We negate to a positive COST so the lower-is-better contract
        holds. Both the value-iteration optimum and the best base-stock comparator
        are carried for ALL 32 instances; the optimality gap % is a note.
        """
        out: list[Baseline] = []
        pub = inst_dict.get("published_scenario_a_returns")
        if not pub:
            return out
        source = str(pub.get("source", ""))
        gap_pct = pub.get("optimality_gap_pct")

        vi_return = pub.get("value_iteration_mean_return")
        if vi_return is not None:
            out.append(
                Baseline(
                    name="value_iteration_optimum",
                    mean_cost=-float(vi_return),  # cost = -discounted return
                    std_cost=(
                        float(pub["value_iteration_return_std"])
                        if pub.get("value_iteration_return_std") is not None
                        else None
                    ),
                    source=source,
                    params={"published_return": float(vi_return)},
                    is_published=True,
                    is_optimal=True,
                    note="published exact value-iteration mean discounted return "
                    f"(gamma=0.99); cost = -return",
                )
            )
        bs_return = pub.get("best_base_stock_mean_return")
        if bs_return is not None:
            out.append(
                Baseline(
                    name="best_base_stock",
                    mean_cost=-float(bs_return),  # cost = -discounted return
                    std_cost=(
                        float(pub["best_base_stock_return_std"])
                        if pub.get("best_base_stock_return_std") is not None
                        else None
                    ),
                    source=source,
                    params={
                        "published_return": float(bs_return),
                        "published_optimality_gap_pct": (
                            float(gap_pct) if gap_pct is not None else None
                        ),
                    },
                    is_published=True,
                    # The paper improves over the best constant base-stock policy:
                    # the canonical comparator a learned policy must beat.
                    is_reference=True,
                    note=(
                        "published best constant base-stock mean discounted return; "
                        f"optimality gap {float(gap_pct)}% vs value iteration"
                        if gap_pct is not None
                        else "published best constant base-stock mean discounted return"
                    ),
                )
            )
        return out

    # -- run the env (the "runnable" proof, discounted-return scale) ------
    def _zero_state(self, p: dict) -> tuple[list[int], list[int]]:
        """The published evaluation starts from the empty inventory / pipeline."""
        on_hand = [0] * int(p["shelf_life"])
        pipeline_orders = [0] * max(int(p["lead_time"]) - 1, 0)
        return on_hand, pipeline_orders

    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        p = inst.params
        on_hand, pipeline_orders = self._zero_state(p)
        seeds = [int(s) for s in protocol.seeds]
        # The published return scale is defined by the INSTANCE's own horizon /
        # warm-up (465 / 100), not a free protocol horizon; the protocol only
        # supplies the seed list (its horizon/warm-up match by construction but
        # we pin the instance values to stay faithful even if overridden).
        horizon = int(p["horizon"])
        warm_up = float(p["warm_up_periods_ratio"])
        max_order_size = int(p["max_order_size"])
        note = f"live env, discounted return gamma={self._GAMMA}, horizon={horizon}, seeds={len(seeds)}"

        out: dict[str, Baseline] = {}

        # 1) Exact MDP optimum (only enabled for small instances in Rust).
        try:
            summary = dict(self._rust.perishable_inventory_exact_mdp_summary(inst.name))
            vi_return = float(summary["value_iteration_mean_return"])
            out["value_iteration_optimum"] = Baseline(
                name="value_iteration_optimum",
                mean_cost=-vi_return,  # cost = -discounted return
                source="recomputed:perishable_inventory_exact_mdp_summary",
                params={
                    "recomputed_return": vi_return,
                    "best_base_stock_level": summary.get("best_base_stock_level"),
                    "matches_published_value_iteration_mean_return": summary.get(
                        "matches_published_value_iteration_mean_return"
                    ),
                },
                is_optimal=True,
                note=f"exact value iteration (gamma={self._GAMMA}); "
                f"published {summary.get('published_value_iteration_mean_return')}",
            )
        except Exception as exc:  # ValueError for state_count > 2000; None-safe
            out["value_iteration_optimum"] = Baseline(
                name="value_iteration_optimum",
                mean_cost=None,
                source=f"perishable_inventory_exact_mdp_summary_failed:{type(exc).__name__}",
                note="exact MDP disabled for this instance size; see published optimum",
            )

        # 2) Best constant base-stock (the canonical reference comparator).
        try:
            res = dict(
                self._rust.perishable_inventory_base_stock_search_discounted_return_summary(
                    on_hand=on_hand,
                    pipeline_orders=pipeline_orders,
                    horizon=horizon,
                    seeds=seeds,
                    max_order_size=max_order_size,
                    demand_mean=float(p["demand_mean"]),
                    demand_cov=float(p["demand_cov"]),
                    holding_cost=float(p["holding_cost"]),
                    shortage_cost=float(p["shortage_cost"]),
                    waste_cost=float(p["waste_cost"]),
                    position_upper_bound=max_order_size,
                    procurement_cost=float(p["procurement_cost"]),
                    warm_up_periods_ratio=warm_up,
                    gamma=self._GAMMA,
                    issuing_policy=str(p["issuing_policy"]),
                    top_k=self._SEARCH_TOP_K,
                )
            )
            best = dict(res["best"])
            mean_return = float(best["mean_return"])
            out["best_base_stock"] = Baseline(
                name="best_base_stock",
                mean_cost=-mean_return,  # cost = -discounted return
                std_cost=float(best["std_return"]),
                source="recomputed:perishable_inventory_base_stock_search_discounted_return_summary",
                params={"base_stock_level": list(best["params"]), "recomputed_return": mean_return},
                is_reference=True,
                note=note,
            )
        except Exception as exc:  # None-safe: a failed search must not abort a sweep
            out["best_base_stock"] = Baseline(
                name="best_base_stock",
                mean_cost=None,
                source=f"perishable_inventory_base_stock_search_failed:{type(exc).__name__}",
            )

        # 3) BSP-low-EW (optional heavier 3-parameter De Moor benchmark).
        if self._RUN_BSP_LOW_EW:
            try:
                res = dict(
                    self._rust.perishable_inventory_bsp_low_ew_search_discounted_return_summary(
                        on_hand=on_hand,
                        pipeline_orders=pipeline_orders,
                        horizon=horizon,
                        seeds=seeds,
                        max_order_size=max_order_size,
                        demand_mean=float(p["demand_mean"]),
                        demand_cov=float(p["demand_cov"]),
                        holding_cost=float(p["holding_cost"]),
                        shortage_cost=float(p["shortage_cost"]),
                        waste_cost=float(p["waste_cost"]),
                        position_upper_bound=max_order_size,
                        procurement_cost=float(p["procurement_cost"]),
                        warm_up_periods_ratio=warm_up,
                        gamma=self._GAMMA,
                        issuing_policy=str(p["issuing_policy"]),
                        top_k=self._SEARCH_TOP_K,
                    )
                )
                best = dict(res["best"])
                mean_return = float(best["mean_return"])
                out["bsp_low_ew"] = Baseline(
                    name="bsp_low_ew",
                    mean_cost=-mean_return,  # cost = -discounted return
                    std_cost=float(best["std_return"]),
                    source="recomputed:perishable_inventory_bsp_low_ew_search_discounted_return_summary",
                    params={"s1_s2_b": list(best["params"]), "recomputed_return": mean_return},
                    note=note,
                )
            except Exception as exc:  # None-safe
                out["bsp_low_ew"] = Baseline(
                    name="bsp_low_ew",
                    mean_cost=None,
                    source=f"perishable_inventory_bsp_low_ew_search_failed:{type(exc).__name__}",
                )

        # Carry the published optimum alongside for a side-by-side read when the
        # live exact solver could not run (the larger instances).
        if not out["value_iteration_optimum"].available:
            for b in inst.published_baselines:
                if b.name == "value_iteration_optimum" and b.available:
                    out["value_iteration_optimum_published"] = b
        return out
