"""CPU concurrency guards for benchmark and training entrypoints.

The Rust rollout bindings use Rayon for population rollouts, while numerical
Python dependencies may use BLAS/OpenMP thread pools.  These environment
variables must be bounded before those runtimes initialize.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import MutableMapping

DEFAULT_MAX_CPU_WORKERS = 4

CPU_THREAD_ENV_VARS = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def bounded_worker_count(
    value: int | str | None,
    *,
    default: int = DEFAULT_MAX_CPU_WORKERS,
    upper_bound: int = DEFAULT_MAX_CPU_WORKERS,
) -> int:
    try:
        requested = int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        requested = int(default)
    return max(1, min(requested, int(upper_bound)))


def requested_worker_count_from_argv(
    argv: list[str] | None = None,
    *,
    default: int = DEFAULT_MAX_CPU_WORKERS,
    upper_bound: int = DEFAULT_MAX_CPU_WORKERS,
) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mp_num_processors", type=int, default=default)
    parsed, _ = parser.parse_known_args(argv)
    return bounded_worker_count(parsed.mp_num_processors, default=default, upper_bound=upper_bound)


def configure_process_cpu_limits(
    worker_count: int | str | None,
    *,
    environ: MutableMapping[str, str] | None = None,
    upper_bound: int = DEFAULT_MAX_CPU_WORKERS,
) -> int:
    limit = bounded_worker_count(worker_count, upper_bound=upper_bound)
    target = os.environ if environ is None else environ
    for name in CPU_THREAD_ENV_VARS:
        current = target.get(name)
        try:
            current_value = int(current) if current not in (None, "") else None
        except (TypeError, ValueError):
            current_value = None
        if current_value is None or current_value < 1 or current_value > limit:
            target[name] = str(limit)
    return limit


def configure_process_cpu_limits_from_argv(
    argv: list[str] | None = None,
    *,
    default: int = DEFAULT_MAX_CPU_WORKERS,
    upper_bound: int = DEFAULT_MAX_CPU_WORKERS,
) -> int:
    return configure_process_cpu_limits(
        requested_worker_count_from_argv(argv, default=default, upper_bound=upper_bound),
        upper_bound=upper_bound,
    )


def normalize_args_cpu_limits(
    args,
    *,
    attr: str = "mp_num_processors",
    upper_bound: int = DEFAULT_MAX_CPU_WORKERS,
) -> int:
    limit = configure_process_cpu_limits(
        getattr(args, attr, DEFAULT_MAX_CPU_WORKERS),
        upper_bound=upper_bound,
    )
    setattr(args, attr, limit)
    return limit


def cpu_limited_environ(
    worker_count: int | str | None,
    *,
    base_environ: MutableMapping[str, str] | None = None,
    upper_bound: int = DEFAULT_MAX_CPU_WORKERS,
) -> dict[str, str]:
    env = dict(os.environ if base_environ is None else base_environ)
    configure_process_cpu_limits(worker_count, environ=env, upper_bound=upper_bound)
    return env
