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
    "lost_sales": "invman.benchmarks.runners.lost_sales_runner:LostSalesRunner",
    "dual_sourcing": "invman.benchmarks.runners.dual_sourcing_runner:DualSourcingRunner",
    "multi_echelon": "invman.benchmarks.runners.multi_echelon_runner:MultiEchelonRunner",
}

_runner_cache: dict[str, ProblemRunner] = {}


def available_runners() -> list[str]:
    """Catalog problem names that currently have an executable runner."""
    return list(_RUNNER_REGISTRY.keys())


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
]
