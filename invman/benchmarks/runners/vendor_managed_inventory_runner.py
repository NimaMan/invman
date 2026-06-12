"""Executable baseline runner — `vendor_managed_inventory` (Sui, Gosavi & Lin 2010 consignment VMI).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the consignment-VMI family runnable from one uniform handle over its single
repo-native reference instance, `sui_gosavi_lin_2010_style_single_retailer`: read
the env parameters, surface the only OPEN literature artifact (the Gosavi
instructor worked-newsvendor case — order-up-to LEVELS, not a cost), and RE-RUN
the shipped base-stock shipment heuristics on the live env to obtain the absolute
reference cost a learned policy must beat.

Why this family has NO published COST and is single-instance
------------------------------------------------------------
The peer-reviewed Sui/Gosavi/Lin (2010, EMJ 22(4):44-53) results table (RL vs.
newsvendor PROFIT per case) is paywalled and not openly reproducible, so the repo
carries NO number printed in the paper (`SUI_GOSAVI_LIN_2010_REFERENCE.
literature_verified = false`). The one open executable anchor is the Gosavi
instructor TEACHING CASE STUDY worked newsvendor example, exposed by
`vendor_managed_inventory_newsvendor_worked_case_summary()`. That binding returns
ORDER-UP-TO LEVELS (mean_demand=15, six_sigma=31.53, newsvendor=26.99 vs the
displayed 26.96), NOT a long-run cost — so it is carried as a published baseline
with `mean_cost=None` and the levels in `params` (honest: a handout, an
order-up-to figure, not a reproducible paper cost). There is exactly one runnable
reference instance (`PRIMARY_REFERENCE_INSTANCE`, a repo-chosen reduced
single-retailer slice); the truck-dispatch 8-case family and the exact-DP verifier
have no single-contract param dict + no published cost and are out of scope here.

There is NO Python accessor that returns the instance param dict, so
`_reference_dict` builds it from the canonical `PRIMARY_REFERENCE_INSTANCE`
constant in `src/problems/vendor_managed_inventory/literature/references.rs`
(read directly into this file's `_PRIMARY_PARAMS`). `list_instances()` returns the
one known name; `_reference_dict` raises KeyError on anything else.

How each method serves the objective
------------------------------------
* `_reference_dict` / `_subfamily_of` — the env params (free, from the literature
  constant) tagged with the paper regime.
* `_published_baselines` — the Gosavi worked-newsvendor order-up-to levels
  (mean_cost=None; the only open literature artifact, a handout not a paper row).
* `_run_baselines` — the runnable proof: simulate the two shipped base-stock
  shipment heuristics (`retailer_base_stock` level=4, `dc_reserve_base_stock`
  level=5/reserve=2) on the live discounted-cost env via
  `vendor_managed_inventory_simulate_policy`. The cheaper of the two
  (`dc_reserve_base_stock`) is the canonical heuristic comparator (`is_reference`)
  — the absolute cost a learned soft-tree must beat. Cost is a 24-period
  discounted COST (`mean_discounted_cost`), so `lower_is_better = True`.

`supports_evaluate = False`: the VMI soft-tree rollout
(`vendor_managed_inventory_soft_tree_rollout`) is NOT wired through the uniform
`build_policy`/`get_model_fitness` seam, so policy scoring is out of scope; the
params + published levels + `run_baselines` above ARE runnable.

Verification tier: faithful (no peer-reviewed paper number is reproduced; the
runnable proof is the base-stock heuristic cost on the live env, and the open
literature anchor is an instructor worked example). Dependencies: `invman_rust`.
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

# Canonical single reference instance, transcribed from the Rust constant
# `PRIMARY_REFERENCE_INSTANCE` in
# src/problems/vendor_managed_inventory/literature/references.rs (no Python
# accessor returns this param dict). A repo-chosen reduced single-retailer slice
# of the Sui/Gosavi/Lin (2010) consignment-VMI structure; NOT a published table.
_PRIMARY_INSTANCE_NAME = "sui_gosavi_lin_2010_style_single_retailer"
_PRIMARY_PARAMS = {
    "name": _PRIMARY_INSTANCE_NAME,
    "source": (
        "Sui, Z., A. Gosavi, and L. Lin (2010), A Reinforcement Learning Approach for "
        "Inventory Replenishment in Vendor-Managed Inventory Systems With Consignment "
        "Inventory, Engineering Management Journal 22(4): 44-53"
    ),
    "url": "https://doi.org/10.1080/10429247.2010.11431878",
    "periods": 24,
    "demand_distribution_kind": "poisson",
    "demand_mean": 2.5,
    "initial_dc_on_hand": 8,
    "initial_retailer_on_hand": 2,
    "initial_retailer_pipeline": 1,
    "dc_replenishment_quantity": 3,
    "dc_capacity": 10,
    "shipment_cost_per_unit": 0.4,
    "dc_holding_cost_per_unit": 0.25,
    "retailer_holding_cost_per_unit": 0.6,
    "stockout_cost_per_unit": 5.0,
    "salvage_value_per_unit": 0.2,
    "max_shipment_quantity": 5,
    "discount_factor": 0.99,
    "benchmark_retailer_base_stock_level": 4,
    "benchmark_dc_reserve_base_stock_level": 5,
    "benchmark_dc_reserve_quantity": 2,
    "notes": (
        "Repo-native reduced single-retailer consignment-VMI slice (Sui, Gosavi & Lin "
        "2010 structure). Repo-chosen parameters; NOT a published table; no published "
        "cost reproduced. literature_verified=false."
    ),
}


class VendorManagedInventoryRunner(ProblemRunner):
    """Runnable baseline driver for the Sui/Gosavi/Lin (2010) consignment-VMI family."""

    problem = "vendor_managed_inventory"
    # The env is a finite-horizon 24-period DISCOUNTED cost; the published profile
    # is per-replication. >=5 seeds is the repo seed-robust rule but here the
    # discounted-cost mean is taken over `replications` internal demand paths per
    # seed (the env's own MC), so a single seed with many replications reproduces
    # the heuristic cost tightly.
    published_protocol = EvalProtocol(
        seeds=(1234, 2025, 7, 99, 1000), horizon=24, warm_up_periods_ratio=0.0, replications=2000
    )
    smoke_protocol = EvalProtocol(seeds=(1234,), horizon=24, warm_up_periods_ratio=0.0, replications=500)
    supports_evaluate = False
    lower_is_better = True  # 24-period discounted COST (mean_discounted_cost)

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return [_PRIMARY_INSTANCE_NAME]

    def primary_instance(self) -> str:
        return _PRIMARY_INSTANCE_NAME

    def _subfamily_of(self, name: str) -> str:
        return "sui_gosavi_lin_2010_consignment"

    def _reference_dict(self, name: str) -> dict:
        if name != _PRIMARY_INSTANCE_NAME:
            raise KeyError(
                f"unknown vendor_managed_inventory instance: {name!r}. "
                f"Known: {self.list_instances()} (the truck-dispatch 8-case family and "
                f"the exact-DP verifier have no single-contract reference dict here)."
            )
        return dict(_PRIMARY_PARAMS)

    # -- published (free) baselines: the Gosavi worked-newsvendor LEVELS --
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        """The only OPEN literature artifact: instructor worked-newsvendor order-up-to.

        These are order-up-to LEVELS (not a cost), and they come from a teaching
        HANDOUT, not the peer-reviewed paper, so `mean_cost=None` (no published
        cost to carry) and `is_published=False` (handout != literature). The
        levels are placed in `params` so a consumer can read them.
        """
        try:
            summary = dict(self._rust.vendor_managed_inventory_newsvendor_worked_case_summary())
        except Exception:
            return []
        out: list[Baseline] = []
        for key in (
            "mean_demand_heuristic_order_up_to",
            "six_sigma_order_up_to",
            "newsvendor_order_up_to",
        ):
            level = summary.get(key)
            if level is None:
                continue
            out.append(
                Baseline(
                    name=key,
                    mean_cost=None,  # an order-up-to LEVEL from a teaching handout, not a cost
                    source=str(summary.get("source", "")),
                    params={
                        "order_up_to_level": float(level),
                        "displayed_newsvendor_order_up_to": summary.get(
                            "displayed_newsvendor_order_up_to"
                        ),
                    },
                    is_published=False,  # instructor handout, NOT the peer-reviewed paper
                    note=(
                        "Gosavi instructor worked-newsvendor order-up-to LEVEL (teaching "
                        "handout based on Sui/Gosavi/Lin 2010); not a published cost"
                    ),
                )
            )
        return out

    # -- run the env (the "runnable" proof): base-stock heuristic costs ----
    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        p = inst.params
        seed = int(protocol.seeds[0])
        periods = int(p["periods"])  # the env is finite-horizon; ignore protocol.horizon
        replications = int(protocol.replications)
        common = dict(
            dc_on_hand=int(p["initial_dc_on_hand"]),
            retailer_on_hand=int(p["initial_retailer_on_hand"]),
            retailer_pipeline=int(p["initial_retailer_pipeline"]),
            periods=periods,
            replications=replications,
            seed=seed,
            demand_mean=float(p["demand_mean"]),
            demand_kind=str(p["demand_distribution_kind"]),
            dc_replenishment_quantity=int(p["dc_replenishment_quantity"]),
            dc_capacity=int(p["dc_capacity"]),
            shipment_cost_per_unit=float(p["shipment_cost_per_unit"]),
            dc_holding_cost_per_unit=float(p["dc_holding_cost_per_unit"]),
            retailer_holding_cost_per_unit=float(p["retailer_holding_cost_per_unit"]),
            stockout_cost_per_unit=float(p["stockout_cost_per_unit"]),
            max_shipment_quantity=int(p["max_shipment_quantity"]),
            discount_factor=float(p["discount_factor"]),
            salvage_value_per_unit=float(p["salvage_value_per_unit"]),
        )
        # The two shipped base-stock shipment heuristics; dc_reserve is the
        # cheaper, canonical comparator (a DC-reserve guard atop retailer fill).
        heuristics = {
            "retailer_base_stock": {
                "params": [float(p["benchmark_retailer_base_stock_level"])],
                "is_reference": False,
            },
            "dc_reserve_base_stock": {
                "params": [
                    float(p["benchmark_dc_reserve_base_stock_level"]),
                    float(p["benchmark_dc_reserve_quantity"]),
                ],
                "is_reference": True,
            },
        }
        out: dict[str, Baseline] = {}
        for hname, spec in heuristics.items():
            try:
                summary = dict(
                    self._rust.vendor_managed_inventory_simulate_policy(
                        policy_name=hname, params=spec["params"], **common
                    )
                )
                out[hname] = Baseline(
                    name=hname,
                    mean_cost=float(summary["mean_discounted_cost"]),
                    std_cost=float(summary.get("std_discounted_cost", 0.0)),
                    source="recomputed:vendor_managed_inventory_simulate_policy",
                    params={"policy_params": spec["params"]},
                    is_reference=bool(spec["is_reference"]),
                    note=f"live env, periods={periods}, reps={replications}, seed={seed}",
                )
            except Exception as exc:  # None-safe: a failed sim must not abort a sweep
                out[hname] = Baseline(
                    name=hname,
                    mean_cost=None,
                    source=f"vendor_managed_inventory_simulate_policy_failed:{type(exc).__name__}",
                )
        return out
