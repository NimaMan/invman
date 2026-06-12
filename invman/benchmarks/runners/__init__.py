"""Executable baseline runners — registry + uniform entry points.

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
This package is the EXECUTABLE half of the benchmark surface (the catalog is the
metadata half). It turns a catalog problem name into a runnable handle so a
consumer can, with one import, load a literature instance, re-run its baseline,
and score their own policy. See `base.py` for the shared abstractions and each
`*_runner.py` for the family specifics.

`get_runner(problem)` resolves a problem name to a cached `ProblemRunner`
instance (the runner imports `invman_rust` on construction). `load_instance(
problem, name)` is the one-call shortcut a user reaches for first. The registry
intentionally covers only the families with an executable runner today
(lost_sales incl. fixed-order-cost, dual_sourcing, multi_echelon-divergent); a
problem in the catalog but not here raises a clear, listing error rather than a
silent miss — the remaining families' accessors exist in `invman_rust` and are
the next runners to add.

This module imports lazily: importing `invman.benchmarks.catalog` does NOT import
any runner (so the pure-metadata catalog stays stdlib-only and importable without
the compiled extension); a runner is constructed only when `get_runner` is called.
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

# Problem name -> "module:ClassName" (resolved lazily so importing this package
# does not import invman_rust until a runner is actually requested).
_RUNNER_REGISTRY: dict[str, str] = {
    # Full runners (params + published baselines + run_baselines + evaluate seam).
    "lost_sales": "invman.benchmarks.runners.lost_sales_runner:LostSalesRunner",
    "dual_sourcing": "invman.benchmarks.runners.dual_sourcing_runner:DualSourcingRunner",
    "multi_echelon": "invman.benchmarks.runners.multi_echelon_runner:MultiEchelonRunner",
    # Metadata + run_baselines runners (supports_evaluate=False — policy scoring is
    # not yet in the uniform CMA-ES seam for these; load/baselines/compare work).
    "one_warehouse_multi_retailer": "invman.benchmarks.runners.one_warehouse_multi_retailer_runner:OneWarehouseMultiRetailerRunner",
    "perishable_inventory": "invman.benchmarks.runners.perishable_inventory_runner:PerishableInventoryRunner",
    "joint_replenishment": "invman.benchmarks.runners.joint_replenishment_runner:JointReplenishmentRunner",
    "spare_parts_inventory": "invman.benchmarks.runners.spare_parts_inventory_runner:SparePartsInventoryRunner",
    "joint_pricing_inventory": "invman.benchmarks.runners.joint_pricing_inventory_runner:JointPricingInventoryRunner",
    "procurement_removal_inventory": "invman.benchmarks.runners.procurement_removal_inventory_runner:ProcurementRemovalInventoryRunner",
    "random_yield_inventory": "invman.benchmarks.runners.random_yield_inventory_runner:RandomYieldInventoryRunner",
    "nonstationary_lot_sizing": "invman.benchmarks.runners.nonstationary_lot_sizing_runner:NonstationaryLotSizingRunner",
    "ameliorating_inventory": "invman.benchmarks.runners.ameliorating_inventory_runner:AmelioratingInventoryRunner",
    "vendor_managed_inventory": "invman.benchmarks.runners.vendor_managed_inventory_runner:VendorManagedInventoryRunner",
    "decentralized_inventory_control": "invman.benchmarks.runners.decentralized_inventory_control_runner:DecentralizedInventoryControlRunner",
}

_runner_cache: dict[str, ProblemRunner] = {}


def verification_tier(problem: str) -> str:
    """The family's honest verification tier (from the manifest, via the catalog).

    'strict' / 'reference' / 'mixed' / 'faithful'. The catalog is the single
    source of truth; this is a thin convenience so the runner layer can filter by
    it without each caller importing the catalog.
    """
    from invman.benchmarks import catalog

    return catalog.get(problem).verification_tier


def is_literature_verified(problem: str) -> bool:
    """True iff the family reproduces a real literature anchor (tier != faithful).

    The 5 `faithful` families (one_warehouse_multi_retailer, joint_pricing_inventory,
    procurement_removal_inventory, random_yield_inventory, vendor_managed_inventory)
    are repo-native '<author>_style' / paywalled instances solved by the repo's own
    DP — NOT a reproduction of any published number. See
    docs/benchmarks/LITERATURE_VERIFICATION_AUDIT_2026_06_12.md.
    """
    return verification_tier(problem) != "faithful"


def available_runners(include_unverified: bool = False) -> list[str]:
    """Problem names with an executable runner.

    By default returns ONLY the literature-verified families (the honest benchmark
    surface). Pass `include_unverified=True` to also get the 5 `faithful` families,
    which remain fully usable via `get_runner` / `load_instance` but are hidden
    from the default listing.
    """
    names = list(_RUNNER_REGISTRY.keys())
    if include_unverified:
        return names
    return [n for n in names if is_literature_verified(n)]


def has_runner(problem: str) -> bool:
    return str(problem).strip() in _RUNNER_REGISTRY


def get_runner(problem: str) -> ProblemRunner:
    """Return the cached `ProblemRunner` for `problem` (constructs once).

    Fails loudly (KeyError) for a problem without a runner, listing the ones that
    do exist — so a typo or an unmigrated family is obvious, never silent.
    """
    key = str(problem).strip()
    if key in _runner_cache:
        return _runner_cache[key]
    if key not in _RUNNER_REGISTRY:
        raise KeyError(
            f"no executable runner for problem {problem!r}; available: "
            f"{available_runners()}"
        )
    module_path, class_name = _RUNNER_REGISTRY[key].split(":")
    import importlib

    runner_cls = getattr(importlib.import_module(module_path), class_name)
    runner = runner_cls()
    _runner_cache[key] = runner
    return runner


def load_instance(problem: str, name: Optional[str] = None) -> ReferenceInstance:
    """One-call shortcut: the `ReferenceInstance` for (`problem`, `name`)."""
    return get_runner(problem).load_instance(name)


__all__ = [
    "Baseline",
    "EvalProtocol",
    "ProblemRunner",
    "ReferenceInstance",
    "available_runners",
    "has_runner",
    "get_runner",
    "load_instance",
    "is_literature_verified",
    "verification_tier",
]
