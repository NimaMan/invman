"""Benchmark consumer API for the inventory-control ImageNet.

Two layers, one source of truth (`docs/benchmarks/BENCHMARK_MANIFEST.json`):

* `catalog` — dependency-light METADATA over the manifest: list problems, get a
  structured `ProblemCard`, render per-problem BENCHMARK_CARDs. Pure stdlib;
  importable without the compiled extension. See `catalog.py`.
* `runners` — the EXECUTABLE layer: `runners.load_instance(problem, name)` (or
  `catalog.get(problem).load_instance(name)`) returns a runnable
  `ReferenceInstance` from which a consumer reads the env params + published
  baselines, re-runs those baselines on the live env (`run_baselines`), and
  scores their own soft-tree policy (`evaluate`). Imports `invman_rust` lazily —
  only when a runner is actually requested. See `runners/`.
* `verify` — README-local verification-target contract parser and CLI helper.
"""

from invman.benchmarks import catalog

__all__ = ["catalog", "runners", "verify"]


def __getattr__(name):
    # Lazy attribute access so `from invman.benchmarks import runners` works while
    # `import invman.benchmarks` (or using only `catalog`) never imports the
    # runners package / invman_rust. Use importlib with the FULL path so the
    # submodule is bound directly (a relative `from . import runners` would route
    # back through this __getattr__ and recurse).
    if name in {"runners", "verify"}:
        import importlib

        return importlib.import_module(f"invman.benchmarks.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
