"""
Single source of truth for the optimizer-seed robustness standard.

OBJECTIVE
=========
Every HEADLINE learned-policy result in the paper must be robust to CMA-ES
optimization randomness. There are two legitimate ways to establish that
robustness, and which one applies is a property OF THE PROBLEM, not a choice made
per run. This module declares, for every problem family, which mechanism it uses
and pins the shared aggregation/verdict logic so that every seed-robust runner
reports the same way (no per-script drift in seed counts or std conventions).

THE TWO ROBUSTNESS MECHANISMS
=============================
1. BREADTH (mode="breadth"). For problems benchmarked over a LARGE grid of
   heterogeneous instances, robustness is evidenced by breadth: a single CMA-ES
   optimizer run per instance, reported across the whole grid. Because optimizer
   randomness is independent from one instance to the next, a verdict sustained
   across dozens of independent instances (e.g. learned policy instance-best in
   22/24 and 47/48 cells) cannot be the artifact of one lucky seed. Only the two
   lost-sales surfaces (24- and 48-instance grids) qualify; nothing else in the
   current problem set has a grid large enough for the breadth argument.

2. DEPTH / SEEDS (mode="seeds"). For problems with only a handful of instances,
   breadth is unavailable, so robustness must come from depth: re-run CMA-ES with
   >= MIN_OPTIMIZER_SEEDS independent optimizer seeds on the same instance and
   report the cross-seed mean +/- sample standard deviation, plus the fraction of
   seeds that clear the same-protocol comparator. This is the default; any
   problem not explicitly registered as "breadth" is held to the seeds standard
   (fail-closed).

ALGORITHM (what this module provides)
=====================================
- OPTIMIZER_SEED_POLICY: problem_id -> {mode, n_optimizer_seeds, default_seeds,
  note}. policy_for() resolves it with a fail-closed default of seeds/>=5.
- seeds_for(problem_id, override): the canonical optimizer-seed list for a
  problem; validates an override has >= the required count for seeds-mode.
- summarize_values(values): cross-seed mean + SAMPLE std (n-1; the convention the
  manuscript reports) + count. One implementation so every runner matches.
- verdict_label(savings_mean, savings_std, frac_pos, n): the shared
  ROBUST_BEAT / BEAT_WITHIN_STD / PARITY / ROBUST_LOSS rule, lifted verbatim from
  the multi-echelon seed-robust runner so all problems share one verdict rule.
- build_seed_robust_summary(per_seed, ...): assembles the standardized summary
  block (learned/gate seed-mean+/-std, savings mean+/-std, frac beating, verdict,
  n_optimizer_seeds) and self-checks it against the policy.
- assert_seed_policy(problem_id, n_seeds) / assert_breadth_grid(...): the guards
  that fail LOUDLY when a result does not satisfy its declared mechanism. These
  back the generate-time check in paper/generate_results_tables.py, turning "I
  think it is 5 seeds" into a checked invariant.

This module is pure-stdlib (statistics only) so it can be imported by any runner,
the table generator, or a test without pulling in the Rust bindings or CMA-ES.
"""

from __future__ import annotations

import statistics
from typing import Callable, Iterable, Sequence

# Minimum independent CMA-ES optimizer seeds for a "seeds"-mode headline result.
MIN_OPTIMIZER_SEEDS = 5

# Canonical optimizer-seed list reused across problems (matches the historical
# multi-echelon / PADN seed-robust defaults so re-runs reproduce prior runs).
CANONICAL_SEEDS_5 = (9001, 9002, 9003, 9004, 9005)

# Per-problem declaration. mode is "breadth" or "seeds". A problem absent from
# this table is treated as seeds/MIN_OPTIMIZER_SEEDS (fail-closed) by policy_for.
OPTIMIZER_SEED_POLICY: dict[str, dict] = {
    # --- BREADTH: large heterogeneous instance grids -------------------------
    "lost_sales": {
        "mode": "breadth",
        "n_optimizer_seeds": 1,
        "default_seeds": (42,),
        "grid_instances": 24,
        "note": "24-instance vanilla surface (3 demand families x 4 lead times x 2 penalties); "
                "single optimizer run per cell, robustness from breadth across the grid.",
    },
    "lost_sales_fixed_order_cost": {
        "mode": "breadth",
        "n_optimizer_seeds": 1,
        "default_seeds": (42,),
        "grid_instances": 48,
        "note": "48-instance fixed-cost surface (adds 2 setup costs); single optimizer run per "
                "cell, robustness from breadth across the grid.",
    },
    # --- SEEDS: few-instance problems, robustness from >=5 optimizer seeds ----
    "dual_sourcing": {
        "mode": "seeds",
        "n_optimizer_seeds": MIN_OPTIMIZER_SEEDS,
        "default_seeds": CANONICAL_SEEDS_5,
        "note": "Gijsbrechts Fig-9 rows; learned soft-tree vs capped-dual-index proxy, "
                "paired CRN; seed-mean +/- std over >=5 optimizer seeds.",
    },
    "multi_echelon": {
        "mode": "seeds",
        "n_optimizer_seeds": MIN_OPTIMIZER_SEEDS,
        "default_seeds": CANONICAL_SEEDS_5,
        "note": "One-warehouse R-retailer divergent special-delivery; vs same-protocol gate.",
    },
    "one_warehouse_multi_retailer": {
        "mode": "seeds",
        "n_optimizer_seeds": MIN_OPTIMIZER_SEEDS,
        "default_seeds": CANONICAL_SEEDS_5,
        "note": "Kaynov OWMR instances; vs best tuned same-protocol heuristic.",
    },
    "perishable_inventory": {
        "mode": "seeds",
        "n_optimizer_seeds": MIN_OPTIMIZER_SEEDS,
        "default_seeds": CANONICAL_SEEDS_5,
        "note": "FIFO perishable; vs best base-stock.",
    },
    "general_backorder_fixed_cost": {
        "mode": "seeds",
        "n_optimizer_seeds": MIN_OPTIMIZER_SEEDS,
        "default_seeds": CANONICAL_SEEDS_5,
        "note": "General-network backorder (Pirhooshyaran-style); vs gate.",
    },
    "multi_echelon_serial": {
        "mode": "seeds",
        "n_optimizer_seeds": MIN_OPTIMIZER_SEEDS,
        "default_seeds": CANONICAL_SEEDS_5,
        "note": "Serial Clark-Scarf; vs echelon base-stock optimum / gate.",
    },
    "ameliorating_inventory": {
        "mode": "seeds",
        "n_optimizer_seeds": MIN_OPTIMIZER_SEEDS,
        "default_seeds": CANONICAL_SEEDS_5,
        "note": "Ameliorating inventory (average profit); vs comparator.",
    },
    "production_assembly_distribution_network": {
        "mode": "seeds",
        "n_optimizer_seeds": MIN_OPTIMIZER_SEEDS,
        "default_seeds": CANONICAL_SEEDS_5,
        "note": "PADN serial / pure-assembly / mixed topologies; vs the environment's own "
                "pairwise base-stock. NOTE: serial & pure-assembly rows must reach >=5 seeds "
                "before a 'robust' claim (the mixed topology already does).",
    },
}

# Verdict labels (shared rule, lifted from the multi-echelon seed-robust runner).
VERDICT_ROBUST_BEAT = "ROBUST_BEAT_VS_GATE"
VERDICT_BEAT_WITHIN_STD = "BEAT_WITHIN_STD"
VERDICT_PARITY = "PARITY"
VERDICT_ROBUST_LOSS = "ROBUST_LOSS_VS_GATE"


def policy_for(problem_id: str) -> dict:
    """Resolve the seed policy for a problem. Fail-closed: unknown -> seeds/>=5."""
    pol = OPTIMIZER_SEED_POLICY.get(problem_id)
    if pol is not None:
        return dict(pol)
    return {
        "mode": "seeds",
        "n_optimizer_seeds": MIN_OPTIMIZER_SEEDS,
        "default_seeds": CANONICAL_SEEDS_5,
        "note": f"UNREGISTERED problem '{problem_id}': defaulted to seeds/>={MIN_OPTIMIZER_SEEDS} "
                f"(fail-closed). Register it in OPTIMIZER_SEED_POLICY.",
    }


def is_breadth(problem_id: str) -> bool:
    return policy_for(problem_id)["mode"] == "breadth"


def required_seed_count(problem_id: str) -> int:
    return int(policy_for(problem_id)["n_optimizer_seeds"])


def seeds_for(problem_id: str, override: Sequence[int] | None = None) -> list[int]:
    """Canonical optimizer-seed list for a problem.

    With an override, validate it carries enough distinct seeds for seeds-mode;
    breadth-mode problems are allowed their single declared seed.
    """
    pol = policy_for(problem_id)
    if override is not None:
        seeds = [int(s) for s in override]
        if pol["mode"] == "seeds" and len(set(seeds)) < pol["n_optimizer_seeds"]:
            raise ValueError(
                f"{problem_id}: seeds-mode requires >= {pol['n_optimizer_seeds']} distinct "
                f"optimizer seeds, got {len(set(seeds))} ({seeds})."
            )
        return seeds
    return [int(s) for s in pol["default_seeds"]]


def summarize_values(values: Sequence[float]) -> dict:
    """Cross-seed mean + SAMPLE std (n-1) + count. One shared implementation.

    Sample std (statistics.stdev) is the manuscript's reported convention; a
    single value yields std 0.0.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        raise ValueError("summarize_values: empty values")
    return {
        "seed_mean": statistics.mean(vals),
        "seed_std": statistics.stdev(vals) if n > 1 else 0.0,
        "n": n,
    }


def verdict_label(savings_mean: float, savings_std: float, frac_pos: int, n: int) -> str:
    """Shared verdict rule (identical thresholds across all seed-robust runners)."""
    if savings_mean > savings_std and frac_pos == n and savings_std >= 0:
        return VERDICT_ROBUST_BEAT
    if abs(savings_mean) <= max(savings_std, 1e-9):
        return VERDICT_PARITY
    if savings_mean < 0:
        return VERDICT_ROBUST_LOSS
    return VERDICT_BEAT_WITHIN_STD


def build_seed_robust_summary(
    per_seed: Sequence[dict],
    *,
    problem_id: str,
    learned_key: str = "best_learned_cost",
    gate_key: str = "gate_cost",
    savings_key: str | None = "savings_pct_vs_gate",
    enforce_policy: bool = True,
) -> dict:
    """Assemble the standardized seed-robust summary block from per-seed records.

    `per_seed` is a list of dicts each carrying at least `learned_key` and
    `gate_key` (and `savings_key` if savings is precomputed; otherwise it is
    derived as 100*(gate-learned)/gate per seed). Returns the canonical summary:
    learned/gate seed-mean+/-std, savings seed-mean+/-std, frac beating, verdict,
    n_optimizer_seeds. With enforce_policy=True (default) it asserts the seed
    count satisfies the problem's declared policy.
    """
    learned = [float(s[learned_key]) for s in per_seed]
    gates = [float(s[gate_key]) for s in per_seed]
    n = len(per_seed)
    if savings_key is not None and all(savings_key in s for s in per_seed):
        sav = [float(s[savings_key]) for s in per_seed]
    else:
        sav = [100.0 * (g - l) / g for g, l in zip(gates, learned)]
    frac_pos = sum(1 for v in sav if v > 0)
    learned_s = summarize_values(learned)
    gate_s = summarize_values(gates)
    sav_s = summarize_values(sav)

    if enforce_policy:
        assert_seed_policy(problem_id, n)

    return {
        "problem_id": problem_id,
        "n_optimizer_seeds": n,
        "learned_seed_mean": learned_s["seed_mean"],
        "learned_seed_std": learned_s["seed_std"],
        "gate_seed_mean": gate_s["seed_mean"],
        "gate_seed_std": gate_s["seed_std"],
        "savings_pct_seed_mean": sav_s["seed_mean"],
        "savings_pct_seed_std": sav_s["seed_std"],
        "frac_seeds_beating_gate": f"{frac_pos}/{n}",
        "verdict_vs_same_protocol_gate": verdict_label(sav_s["seed_mean"], sav_s["seed_std"], frac_pos, n),
    }


# ----------------------------------------------------------------------------
# Guards: fail loudly when a result does not satisfy its declared mechanism.
# ----------------------------------------------------------------------------
class SeedRobustnessError(AssertionError):
    """Raised when a result violates its declared optimizer-seed policy."""


def assert_seed_policy(problem_id: str, n_seeds_present: int, *, context: str = "") -> None:
    """For seeds-mode problems, require >= the declared optimizer-seed count."""
    pol = policy_for(problem_id)
    where = f" [{context}]" if context else ""
    if pol["mode"] != "seeds":
        return
    if int(n_seeds_present) < int(pol["n_optimizer_seeds"]):
        raise SeedRobustnessError(
            f"{problem_id}{where}: seeds-mode headline requires >= "
            f"{pol['n_optimizer_seeds']} optimizer seeds but only {n_seeds_present} present. "
            f"Re-run with the full seed set or do not report this as a headline. "
            f"Policy note: {pol['note']}"
        )


def assert_breadth_grid(problem_id: str, n_instances_present: int, *, context: str = "") -> None:
    """For breadth-mode problems, require the (near-)complete instance grid."""
    pol = policy_for(problem_id)
    where = f" [{context}]" if context else ""
    if pol["mode"] != "breadth":
        raise SeedRobustnessError(
            f"{problem_id}{where}: assert_breadth_grid called on a non-breadth problem."
        )
    expected = int(pol.get("grid_instances", 0))
    if expected and int(n_instances_present) < expected:
        raise SeedRobustnessError(
            f"{problem_id}{where}: breadth-mode requires the full {expected}-instance grid "
            f"but only {n_instances_present} instances are present; the breadth robustness "
            f"argument needs the whole surface."
        )


def assert_problem_result(
    problem_id: str,
    *,
    n_optimizer_seeds: int | None = None,
    n_instances: int | None = None,
    context: str = "",
) -> None:
    """Dispatch to the right guard given a problem's declared mode."""
    if is_breadth(problem_id):
        if n_instances is None:
            raise SeedRobustnessError(
                f"{problem_id} [{context}]: breadth-mode check needs n_instances."
            )
        assert_breadth_grid(problem_id, n_instances, context=context)
    else:
        if n_optimizer_seeds is None:
            raise SeedRobustnessError(
                f"{problem_id} [{context}]: seeds-mode check needs n_optimizer_seeds."
            )
        assert_seed_policy(problem_id, n_optimizer_seeds, context=context)


def run_over_seeds(
    problem_id: str,
    train_one_seed: Callable[[int], dict],
    *,
    seeds: Sequence[int] | None = None,
    learned_key: str = "best_learned_cost",
    gate_key: str = "gate_cost",
    savings_key: str | None = "savings_pct_vs_gate",
) -> dict:
    """Generic optimizer-seed driver shared by every seeds-mode runner.

    Resolves the seed list from the registry (enforcing >= the required count),
    calls the problem-specific `train_one_seed(seed) -> per-seed record` for each,
    and returns {"per_seed": [...], **standardized summary}. Problem scripts keep
    only their `train_one_seed` closure; the loop, aggregation, verdict, and
    policy enforcement live here.
    """
    seed_list = seeds_for(problem_id, seeds)
    per_seed = [train_one_seed(int(seed)) for seed in seed_list]
    summary = build_seed_robust_summary(
        per_seed,
        problem_id=problem_id,
        learned_key=learned_key,
        gate_key=gate_key,
        savings_key=savings_key,
    )
    return {"seeds": list(seed_list), "per_seed": list(per_seed), **summary}
