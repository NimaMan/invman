"""Executable baseline runner — `multi_echelon` (divergent one-warehouse / N-retailer).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the divergent multi-echelon family runnable from one handle over the 5
reference instances exposed by `multi_echelon_list_reference_instances`:

  * van_roy1997_simple_problem / case_study1 / case_study2  — Van Roy et al.
    (1997) one-warehouse one/N-retailer (van_roy_1997 dynamics; reproduction
    targets — they publish absolute constant-base-stock + best-NDP costs).
  * gijsbrechts2022_setting1 / setting2  — the two Van Roy settings reused by
    Gijsbrechts et al. (2022) Table 3 (gijs_2022 dynamics; the FAITHFUL search
    targets — the paper reports A3C's % savings over constant base-stock, not an
    absolute cost).

(The `multi_echelon` umbrella also has serial / assembly / general-backorder-
fixed-cost / PADN subfamilies, each with its own exact-solver or base-stock
accessors and a different baseline contract; those are out of scope for this
single-contract divergent runner — see the README.)

Why divergent baselines mix published + recomputed
--------------------------------------------------
* `_published_baselines` carries the free published numbers where they exist:
  the constant-base-stock cost (the CANONICAL comparator the paper improves over
  -> `is_reference=True`) and the best-NDP cost, plus the A3C %-savings as a note
  (a relative improvement, not a cost). For the gijs settings these are null
  (only %-savings are published), so the reference comes from recomputation.
* `_run_baselines` is the runnable proof: grid-search the best CONSTANT
  base-stock over the instance's published warehouse/retailer level grids on the
  live env (`multi_echelon_search_stationary_policy`), reproducing the published
  constant-base-stock cost for the Van Roy cases and supplying it for the gijs
  settings. This is the number a learned policy must beat.

How each method serves the objective
------------------------------------
* `_eval_model_and_args` — build the CMA-ES eval seam. multi_echelon is
  soft_tree-only; the default action design `direct_level` directly estimates
  the warehouse/retailer order-up-to LEVELS bounded by the physical inventory
  caps (the design that reaches the ~330 operating region); `grid` restricts to
  the published reduced level grid.

Verification tier: mixed (serial/divergent strict, gijs settings faithful search
targets). Dependencies: `invman_rust`, the `invman.*` optimizer layer.
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


class MultiEchelonRunner(ProblemRunner):
    """Runnable baseline driver for the divergent multi-echelon family."""

    problem = "multi_echelon"
    # The faithful search horizon/replications live on each instance
    # (benchmark_search_horizon / benchmark_replications, typically 10000 x 100);
    # this protocol is a moderate default — pass the instance values for a fully
    # faithful reproduction. evaluate() of a learned policy uses the seed list.
    published_protocol = EvalProtocol(
        seeds=(123, 2025, 7, 99, 1000), horizon=10000, warm_up_periods_ratio=0.0, replications=30
    )
    smoke_protocol = EvalProtocol(seeds=(123,), horizon=2000, warm_up_periods_ratio=0.0, replications=3)
    default_structure = {
        "depth": 2,
        "temperature": 0.25,
        "split_type": "oblique",
        "leaf_type": "linear",
        "multi_action_design": "direct_level",
    }

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust
        self._names = [
            str(d["name"]) for d in invman_rust.multi_echelon_list_reference_instances()
        ]
        self._set = set(self._names)

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return list(self._names)

    def primary_instance(self) -> str:
        return str(self._rust.multi_echelon_primary_reference_instance()["name"])

    def _subfamily_of(self, name: str) -> str:
        return "divergent_special_delivery"

    def _reference_dict(self, name: str) -> dict:
        if name not in self._set:
            raise KeyError(
                f"unknown multi_echelon instance: {name!r}. Known: {self.list_instances()} "
                f"(serial/assembly/gbk/padn subfamilies use their own accessors)"
            )
        return dict(self._rust.multi_echelon_get_reference_instance(name))

    # -- published (free) baselines ---------------------------------------
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        out: list[Baseline] = []
        cbs = inst_dict.get("published_constant_base_stock_mean_cost")
        if cbs is not None:
            out.append(
                Baseline(
                    name="constant_base_stock_published",
                    mean_cost=float(cbs),
                    source=str(inst_dict.get("source", "")),
                    params={"levels": list(inst_dict.get("published_constant_base_stock_levels", []))},
                    is_published=True,
                    is_reference=True,  # the paper's canonical comparator
                )
            )
        ndp = inst_dict.get("published_van_roy_best_ndp_mean_cost")
        if ndp is not None:
            out.append(
                Baseline(
                    name="van_roy_best_ndp_published",
                    mean_cost=float(ndp),
                    source=str(inst_dict.get("source", "")),
                    is_published=True,
                    note="best published neuro-DP policy (target to approach)",
                )
            )
        a3c = inst_dict.get("published_a3c_savings_pct")
        if a3c is not None:
            out.append(
                Baseline(
                    name="a3c",
                    mean_cost=None,  # a %-savings over constant base-stock, not a cost
                    source=str(inst_dict.get("source", "")),
                    params={
                        "published_savings_pct": float(a3c),
                        "confidence_half_width_pct": inst_dict.get(
                            "published_a3c_confidence_half_width_pct"
                        ),
                    },
                    is_published=True,
                    note=f"published A3C savings {float(a3c)}% over constant base-stock",
                )
            )
        return out

    def _search_grid(self, p: dict) -> tuple[list[int], list[int]]:
        """Warehouse/retailer level grids for the constant-base-stock search.

        Default = the instance's shipped reference grid (the gijs reduced search
        grid, which is the faithful comparator for the gijs settings). For the
        Van Roy reproduction instances the published optimum (e.g. warehouse 330)
        lies OUTSIDE that reduced grid, so the search would be starved at the grid
        boundary; when a published constant-base-stock policy is shipped and falls
        outside the reduced grid, widen to span the operating region and pin the
        published levels into the grid so the search can actually reach it.
        """
        w = [int(v) for v in p["benchmark_warehouse_levels"]]
        r = [int(v) for v in p["benchmark_retailer_levels"]]
        pub = [int(v) for v in (p.get("published_constant_base_stock_levels") or [])]
        if len(pub) == 2 and (pub[0] > max(w) or pub[1] > max(r)):
            w_top = int(max(max(w), pub[0]) * 1.15)
            r_top = int(max(max(r), pub[1]) * 1.15)
            w = sorted(set(list(range(0, w_top + 1, max(1, w_top // 40))) + [pub[0]]))
            r = sorted(set(list(range(0, r_top + 1, max(1, r_top // 40))) + [pub[1]]))
        return w, r

    # -- run the env (the "runnable" proof): best constant base-stock -----
    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        p = inst.params
        warehouse_levels, retailer_levels = self._search_grid(p)
        result = self._rust.multi_echelon_search_stationary_policy(
            policy_kind="constant_base_stock",
            allocation_mode=str(p.get("policy_allocation_mode", "min_shortage")),
            warehouse_levels=warehouse_levels,
            retailer_levels=retailer_levels,
            warehouse_lead_time=int(p["warehouse_lead_time"]),
            retailer_lead_time=int(p["retailer_lead_time"]),
            num_retailers=int(p["num_retailers"]),
            warehouse_holding_cost=float(p["warehouse_holding_cost"]),
            retailer_holding_cost=float(p["retailer_holding_cost"]),
            warehouse_expedited_cost=float(p["warehouse_expedited_cost"]),
            warehouse_lost_sale_cost=float(p["warehouse_lost_sale_cost"]),
            expedited_service_prob=float(p["expedited_service_prob"]),
            warehouse_capacity=int(p["warehouse_capacity"]),
            warehouse_inventory_cap=int(p["warehouse_inventory_cap"]),
            retailer_inventory_cap=int(p["retailer_inventory_cap"]),
            inventory_dynamics_mode=str(p["inventory_dynamics_mode"]),
            demand_distribution=str(p["demand_distribution"]),
            demand_mean=float(p["demand_mean"]),
            demand_std=float(p["demand_std"]),
            horizon=int(protocol.horizon),
            replications=int(protocol.replications),
            seed=int(protocol.seeds[0]),
            warm_up_periods_ratio=float(protocol.warm_up_periods_ratio),
            objective=str(p.get("rollout_objective", "average_cost_after_warmup")),
            top_k=3,
        )
        best = dict(result["best_result"])
        out: dict[str, Baseline] = {
            "constant_base_stock": Baseline(
                name="constant_base_stock",
                mean_cost=float(best["mean_cost"]),
                std_cost=float(best["cost_std"]),
                source="recomputed:multi_echelon_search_stationary_policy",
                params={
                    "warehouse_level": int(best["warehouse_level"]),
                    "retailer_level": int(best["retailer_level"]),
                },
                is_reference=True,
                note=f"live env, horizon={protocol.horizon}, reps={protocol.replications}, "
                f"seed={protocol.seeds[0]}, grid={len(warehouse_levels)}x{len(retailer_levels)}",
            )
        }
        # Carry the published comparators alongside for a side-by-side read.
        for b in inst.published_baselines:
            if b.available and b.name not in out:
                out[b.name] = b
        return out

    # -- eval seam --------------------------------------------------------
    def _eval_model_and_args(
        self, inst: ReferenceInstance, structure: dict, protocol: EvalProtocol
    ):
        from invman.config import get_config
        from invman.policy_build import build_policy
        from invman.policy_registry import apply_policy_name, make_soft_tree_policy_name

        s = self._structure_with_defaults(structure)
        p = inst.params
        args = get_config([])
        args.problem = "multi_echelon"
        args.horizon = int(protocol.horizon)
        args.warm_up_periods_ratio = float(p.get("warm_up_periods_ratio", protocol.warm_up_periods_ratio))
        args.warehouse_lead_time = int(p["warehouse_lead_time"])
        args.retailer_lead_time = int(p["retailer_lead_time"])
        args.num_retailers = int(p["num_retailers"])
        args.warehouse_holding_cost = float(p["warehouse_holding_cost"])
        args.retailer_holding_cost = float(p["retailer_holding_cost"])
        args.warehouse_expedited_cost = float(p["warehouse_expedited_cost"])
        args.warehouse_lost_sale_cost = float(p["warehouse_lost_sale_cost"])
        args.expedited_service_prob = float(p["expedited_service_prob"])
        args.warehouse_capacity = int(p["warehouse_capacity"])
        args.warehouse_inventory_cap = int(p["warehouse_inventory_cap"])
        args.retailer_inventory_cap = int(p["retailer_inventory_cap"])
        args.inventory_dynamics_mode = str(p["inventory_dynamics_mode"])
        args.demand_distribution = str(p["demand_distribution"])
        args.multi_demand_mean = float(p["demand_mean"])
        args.multi_demand_std = float(p["demand_std"])
        args.rollout_objective = str(p.get("rollout_objective", "average_cost_after_warmup"))
        args.warehouse_base_stock_levels = [int(v) for v in p["benchmark_warehouse_levels"]]
        args.retailer_base_stock_levels = [int(v) for v in p["benchmark_retailer_levels"]]
        args.multi_action_design = str(s.get("multi_action_design", "direct_level"))
        args.policy_name = make_soft_tree_policy_name(
            depth=int(s["depth"]),
            temperature=float(s["temperature"]),
            split_type=str(s["split_type"]),
            leaf_type=str(s["leaf_type"]),
        )
        apply_policy_name(args)
        return build_policy(args), args
