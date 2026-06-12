"""Executable baseline runner — `one_warehouse_multi_retailer` (Kaynov 2024 OWMR).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the one-warehouse / multi-retailer (OWMR) family runnable from one uniform
handle over the 14 published Kaynov et al. (2024, IJPE 267, 109088) reference
instances (`kaynov2024_instance_1..14`): read the env params, read the PUBLISHED
echelon-base-stock + PPO benchmark numbers, RE-RUN the echelon base-stock policy
on the live simulator to recover the absolute reference costs, and (for the small
repo-native verifier) RE-SOLVE the exact finite-horizon DP optimum. Policy
scoring of a user's soft-tree is intentionally OUT of scope here
(`supports_evaluate=False`): this family's soft-tree rollout is not wired through
the uniform CMA-ES `build_policy`/`get_model_fitness` seam, so `evaluate()` fails
loudly with a pointer rather than pretending to score.

Verification tier = FAITHFUL (approx-only). There is NO exact optimum for the
full 100-period Kaynov instances; the only exact solver is a reduced 2-retailer /
binary-demand / 2-period self-consistency DP on a repo-native verification
instance (optimum 8.485, `literature_verified=false` — an anchor, NOT a published
number). For the 14 published instances the runnable reference is the echelon
base-stock heuristic re-simulated on the live env, which reproduces the published
rows to ~1% (symmetric) and ~5% (asymmetric K=10), in the SAME direction.

Cost-sign convention (important)
--------------------------------
The PAPER reports a negative REWARD; the repo (and `simulate_policy`) reports a
POSITIVE COST = -reward. `lower_is_better=True`. So every published benchmark
`mean_cost` (negative) is negated here into a positive cost, and the re-simulated
`simulate_policy` cost is already positive and directly comparable.

How each method serves the objective
------------------------------------
* `list_instances` / `primary_instance` / `_reference_dict` — the 14 Kaynov rows
  via `*_list_reference_instances` / `*_primary_reference_instance` /
  `*_get_reference_instance`; `_reference_dict` raises KeyError on an unknown name.
* `_published_baselines` — the free published numbers carried on each instance:
  echelon base-stock + proportional allocation, echelon base-stock + min_shortage
  allocation, and PPO. Each carried as a positive cost (negated reward) with its
  standard error and published relative-gap-%. The CHEAPEST published echelon
  base-stock allocation is tagged `is_reference=True` (the canonical comparator a
  learned policy must beat — the "gate"); PPO is carried as cross-protocol
  context (it is the learner the paper benchmarks, not the static comparator).
  None is `is_optimal` — there is no exact optimum for these full instances.
* `_run_baselines` — the runnable proof, two parts:
    (a) the EXACT DP on the reduced verification instance
        (`*_exact_dp_summary`): the optimum (`is_optimal=True`) plus the two
        allocations it dominates. This is the only exact, byte-stable number.
    (b) the echelon base-stock heuristic RE-SIMULATED on the actual instance for
        BOTH allocations via `*_simulate_policy`, using the "mean-filled pipeline
        warm start" initial state and a moment-derived base-stock search:
        symmetric instances collapse to a 1-D retailer grid x warehouse grid
        (cheap, ~1% reproduction); asymmetric instances use bounded coordinate
        descent over the per-retailer levels (cheap, ~5% reproduction for K=10).
        These reproduce the published proportional / min_shortage rows.
  None-safe: any failed binding yields a `Baseline(mean_cost=None, source=
  "...failed:<ExcType>")` rather than aborting the sweep.

This mirrors `scripts/one_warehouse_multi_retailer/run_heuristic_published_
benchmark.py` (same moments, same mean-filled warm start, same simulate call),
kept here as an importable library surface so a consumer never reaches into
`scripts/`.

Verification note: instance_7 (lost_sales) min_shortage re-simulates to ~1394.82
vs published 1408.08 (gap -0.94%); instance_11 (partial_backorder) proportional
to ~1113.17 vs published 1111.76 (gap +0.13%); exact-DP optimum 8.485 dominates
both heuristics at 9.2225. Dependencies: `invman_rust` (lazy import in __init__).
================================================================================
"""

from __future__ import annotations

import math
from typing import Optional

from invman.benchmarks.runners.base import (
    Baseline,
    EvalProtocol,
    ProblemRunner,
    ReferenceInstance,
)


class OneWarehouseMultiRetailerRunner(ProblemRunner):
    """Runnable baseline driver for the Kaynov 2024 OWMR family (approx-only tier)."""

    problem = "one_warehouse_multi_retailer"
    # The published evaluation is 100 periods x 1000 replications (discount 1.0).
    # `_run_baselines` re-simulates at the protocol's `replications`/`seeds[0]`;
    # the >=5 seeds carried here are the repo's seed-robust headline rule (used by
    # a learned-policy evaluation once this family is wired into the eval seam).
    published_protocol = EvalProtocol(
        seeds=(2222, 2025, 7, 99, 1000), horizon=100, warm_up_periods_ratio=0.0, replications=1000
    )
    # Smoke: a short re-simulation with a small search-replication budget.
    smoke_protocol = EvalProtocol(seeds=(2222,), horizon=100, warm_up_periods_ratio=0.0, replications=60)
    # policy scoring is not wired into the uniform CMA-ES eval seam for this family.
    supports_evaluate = False
    lower_is_better = True

    # Re-evaluation replication budget for the recomputed reference (after search).
    _EVAL_REPLICATIONS = 1000
    _EVAL_SEED = 2222
    # Bounded coordinate-descent passes for the asymmetric (per-retailer) search.
    _COORD_PASSES = 3

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust
        self._names = [
            str(d["name"])
            for d in invman_rust.one_warehouse_multi_retailer_list_reference_instances()
        ]
        self._set = set(self._names)

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return list(self._names)

    def primary_instance(self) -> str:
        return str(
            self._rust.one_warehouse_multi_retailer_primary_reference_instance()["name"]
        )

    def _subfamily_of(self, name: str) -> str:
        # One catalog entry spans 3 customer-behaviour regimes; tag the regime.
        try:
            cb = str(self._reference_dict(name).get("customer_behavior", ""))
        except KeyError:
            cb = ""
        return f"kaynov2024_{cb}" if cb else "kaynov2024"

    def _reference_dict(self, name: str) -> dict:
        if name not in self._set:
            raise KeyError(
                f"unknown one_warehouse_multi_retailer instance: {name!r}. "
                f"Known: {self.list_instances()}"
            )
        return dict(self._rust.one_warehouse_multi_retailer_get_reference_instance(name))

    # -- published (free) baselines ---------------------------------------
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        """The free published Kaynov rows (positive cost = -published reward).

        The cheapest published echelon base-stock allocation is the canonical
        comparator (`is_reference=True`); PPO is carried as cross-protocol context.
        """
        rows = [
            ("echelon_base_stock_proportional", inst_dict.get("published_proportional_benchmark")),
            ("echelon_base_stock_min_shortage", inst_dict.get("published_min_shortage_benchmark")),
            ("ppo", inst_dict.get("published_ppo_benchmark")),
        ]
        source = str(inst_dict.get("source", ""))
        out: list[Baseline] = []
        # Pick the cheapest echelon base-stock allocation as the canonical gate.
        ebs_costs = {}
        for tag, bench in rows[:2]:
            if bench is not None and bench.get("mean_cost") is not None:
                ebs_costs[tag] = -float(bench["mean_cost"])  # negate reward -> cost
        cheapest_ebs = min(ebs_costs, key=ebs_costs.get) if ebs_costs else None
        for tag, bench in rows:
            if bench is None or bench.get("mean_cost") is None:
                continue
            out.append(
                Baseline(
                    name=tag,
                    mean_cost=-float(bench["mean_cost"]),  # paper reports negative reward
                    std_cost=(
                        float(bench["standard_error"])
                        if bench.get("standard_error") is not None
                        else None
                    ),
                    source=source,
                    params={
                        "allocation_policy": bench.get("allocation_policy"),
                        "published_relative_gap_percent": bench.get("relative_gap_percent"),
                    },
                    is_published=True,
                    # No exact optimum exists for the full 100-period instances.
                    is_optimal=False,
                    is_reference=(tag == cheapest_ebs),
                    note=(
                        "published cost = -reward; "
                        f"rel.gap {bench.get('relative_gap_percent')}% "
                        f"(PPO is cross-protocol context, not a static comparator)"
                        if tag == "ppo"
                        else f"published cost = -reward; rel.gap {bench.get('relative_gap_percent')}%"
                    ),
                )
            )
        return out

    # -- demand moments / state / search (mirrors run_heuristic_published_benchmark) --
    @staticmethod
    def _normal_cdf(x: float, mean: float, std: float) -> float:
        if std <= 0.0:
            return 1.0 if x >= mean else 0.0
        return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2.0))))

    def _rounded_normal_moments(self, mean: float, std: float) -> tuple[float, float]:
        if std <= 0.0:
            return float(max(int(round(mean)), 0)), 0.0
        probs = [max(0.0, min(1.0, self._normal_cdf(0.5, mean, std)))]
        support = [0]
        k = 1
        cumulative = probs[0]
        while cumulative < 1.0 - 1e-12 and k < 10_000:
            prob = max(0.0, self._normal_cdf(k + 0.5, mean, std) - self._normal_cdf(k - 0.5, mean, std))
            if prob > 1e-15:
                probs.append(prob)
                support.append(k)
                cumulative += prob
            k += 1
        if cumulative < 1.0:
            probs[-1] += 1.0 - cumulative
        sup = [float(s) for s in support]
        mean_value = sum(s * p for s, p in zip(sup, probs))
        variance = sum(((s - mean_value) ** 2) * p for s, p in zip(sup, probs))
        return mean_value, math.sqrt(max(variance, 0.0))

    def _demand_moments(self, p: dict) -> tuple[list[float], list[float]]:
        means: list[float] = []
        stds: list[float] = []
        for kind, p1, p2 in zip(p["demand_kinds"], p["demand_param1"], p["demand_param2"]):
            if kind == "poisson":
                means.append(float(p1))
                stds.append(math.sqrt(float(p1)))
            elif kind == "discrete_uniform":
                low = int(round(p1))
                high = int(round(p2))
                n = high - low + 1
                means.append(0.5 * (low + high))
                stds.append(math.sqrt((n * n - 1) / 12.0))
            elif kind == "rounded_normal":
                m, s = self._rounded_normal_moments(float(p1), float(p2))
                means.append(m)
                stds.append(s)
            else:  # deterministic
                means.append(float(p1))
                stds.append(0.0)
        return means, stds

    def _initial_state(self, p: dict):
        """Mean-filled pipeline warm start (matches the reproduction script)."""
        means, _ = self._demand_moments(p)
        warehouse_mean = int(round(sum(means)))
        retailer_inventory = [int(round(m)) for m in means]
        warehouse_pipeline = [warehouse_mean] * int(p["warehouse_lead_time"])
        retailer_pipeline = [
            [retailer_inventory[i]] * int(lt) for i, lt in enumerate(p["retailer_lead_times"])
        ]
        return warehouse_mean, warehouse_pipeline, retailer_inventory, retailer_pipeline

    def _is_symmetric(self, p: dict) -> bool:
        return (
            len(set(p["retailer_lead_times"])) == 1
            and len(set(p["holding_cost_retailers"])) == 1
            and len(set(p["penalty_costs_retailers"])) == 1
            and all(
                p["demand_kinds"][i] == p["demand_kinds"][0]
                and p["demand_param1"][i] == p["demand_param1"][0]
                and p["demand_param2"][i] == p["demand_param2"][0]
                for i in range(1, len(p["retailer_lead_times"]))
            )
        )

    def _search_bounds(self, p: dict):
        means, stds = self._demand_moments(p)
        retailer_bounds = []
        for mean, std, lead_time in zip(means, stds, p["retailer_lead_times"]):
            lead_periods = int(lead_time) + 1
            lower = max(0, int(math.floor(mean * lead_periods)))
            upper = max(0, int(math.ceil(mean * lead_periods + 3.0 * std * math.sqrt(lead_periods))))
            retailer_bounds.append((lower, upper))
        cumulative = (
            int(p["warehouse_lead_time"]) + max(int(v) for v in p["retailer_lead_times"]) + 1
        )
        system_mean = sum(means)
        system_variance = sum(s * s for s in stds)
        warehouse_lower = max(0, int(math.floor(system_mean * cumulative)))
        warehouse_upper = max(
            0, int(math.ceil(system_mean * cumulative + 3.0 * math.sqrt(system_variance * cumulative)))
        )
        return (warehouse_lower, warehouse_upper), retailer_bounds

    def _simulate(self, p, warehouse_level, retailer_levels, allocation, replications, seed) -> float:
        wi, wp, ri, rp = self._initial_state(p)
        mean_cost, _ = self._rust.one_warehouse_multi_retailer_simulate_policy(
            policy_name="echelon_base_stock",
            params=[float(warehouse_level)] + [float(x) for x in retailer_levels],
            initial_warehouse_inventory=wi,
            initial_warehouse_pipeline=wp,
            initial_retailer_inventory=ri,
            initial_retailer_pipeline=rp,
            periods=int(p["benchmark_periods"]),
            replications=int(replications),
            seed=int(seed),
            demand_kinds=[str(k) for k in p["demand_kinds"]],
            demand_param1=[float(x) for x in p["demand_param1"]],
            demand_param2=[float(x) for x in p["demand_param2"]],
            holding_cost_warehouse=float(p["holding_cost_warehouse"]),
            holding_cost_retailers=[float(x) for x in p["holding_cost_retailers"]],
            penalty_costs_retailers=[float(x) for x in p["penalty_costs_retailers"]],
            customer_behavior=str(p["customer_behavior"]),
            emergency_shipment_probability=float(p["emergency_shipment_probability"]),
            discount_factor=1.0,
            allocation_policy=str(allocation),
        )
        return float(mean_cost)

    def _search_best_base_stock(self, p, allocation, replications, seed):
        """Best echelon base-stock levels for `allocation` on the live env.

        Symmetric -> 1-D retailer grid x warehouse grid (cheap, tight reproduction).
        Asymmetric -> bounded coordinate descent over per-retailer + warehouse
        levels (cheap, ~5% reproduction for the K=10 instances). Returns
        (warehouse_level, retailer_levels, search_mean_cost).
        """
        (wlo, whi), rbounds = self._search_bounds(p)
        num_retailers = len(p["retailer_lead_times"])
        if self._is_symmetric(p):
            rlo, rhi = rbounds[0]
            best = None
            for w in range(wlo, whi + 1):
                for r in range(rlo, rhi + 1):
                    cost = self._simulate(p, w, [r] * num_retailers, allocation, replications, seed)
                    if best is None or cost < best[2]:
                        best = (w, [r] * num_retailers, cost)
            return best
        # Asymmetric: coordinate descent.
        levels = [(lo + hi) // 2 for lo, hi in rbounds]
        warehouse = (wlo + whi) // 2
        best_cost = self._simulate(p, warehouse, levels, allocation, replications, seed)
        for _ in range(self._COORD_PASSES):
            for i in range(num_retailers):
                lo, hi = rbounds[i]
                best_level = levels[i]
                for cand in range(lo, hi + 1):
                    levels[i] = cand
                    cost = self._simulate(p, warehouse, levels, allocation, replications, seed)
                    if cost < best_cost:
                        best_cost = cost
                        best_level = cand
                levels[i] = best_level
            best_w = warehouse
            for cand in range(wlo, whi + 1):
                cost = self._simulate(p, cand, levels, allocation, replications, seed)
                if cost < best_cost:
                    best_cost = cost
                    best_w = cand
            warehouse = best_w
        return warehouse, list(levels), best_cost

    # -- run the env (the "runnable" proof) -------------------------------
    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        out: dict[str, Baseline] = {}
        p = inst.params
        search_reps = int(protocol.replications)
        search_seed = int(protocol.seeds[0])

        # (a) EXACT DP on the reduced repo-native verification instance — the only
        # exact, byte-stable number this family has (NOT a published number).
        try:
            dp = dict(self._rust.one_warehouse_multi_retailer_exact_dp_summary())
            out["verification_dp_optimal"] = Baseline(
                name="verification_dp_optimal",
                mean_cost=float(dp["optimal_discounted_cost"]),
                source="recomputed:one_warehouse_multi_retailer_exact_dp_summary",
                is_optimal=True,
                note="exact finite-horizon DP on the 2-retailer/binary/2-period "
                "verification anchor (literature_verified=false; NOT a Kaynov row)",
            )
            out["verification_dp_proportional"] = Baseline(
                name="verification_dp_proportional",
                mean_cost=float(dp["proportional_discounted_cost"]),
                source="recomputed:one_warehouse_multi_retailer_exact_dp_summary",
                note="echelon base-stock + proportional on the verification anchor",
            )
            out["verification_dp_min_shortage"] = Baseline(
                name="verification_dp_min_shortage",
                mean_cost=float(dp["min_shortage_discounted_cost"]),
                source="recomputed:one_warehouse_multi_retailer_exact_dp_summary",
                note="echelon base-stock + min_shortage on the verification anchor",
            )
        except Exception as exc:  # None-safe: never abort the sweep
            out["verification_dp_optimal"] = Baseline(
                name="verification_dp_optimal",
                mean_cost=None,
                source=f"exact_dp_failed:{type(exc).__name__}",
            )

        # (b) Echelon base-stock RE-SIMULATED on the actual instance, both
        # allocations: search the levels, then re-evaluate the argmin at a fresh
        # high-replication seed (matches the published 1000-rep / seed 2222 eval).
        cheapest = None
        for tag, allocation in (
            ("echelon_base_stock_proportional", "proportional"),
            ("echelon_base_stock_min_shortage", "min_shortage"),
        ):
            try:
                w, levels, _ = self._search_best_base_stock(p, allocation, search_reps, search_seed)
                eval_cost = self._simulate(
                    p, w, levels, allocation, self._EVAL_REPLICATIONS, self._EVAL_SEED
                )
                pub = next(
                    (b for b in inst.published_baselines if b.name == tag and b.available), None
                )
                gap = (
                    100.0 * (eval_cost - pub.mean_cost) / pub.mean_cost
                    if pub is not None and pub.mean_cost
                    else None
                )
                out[tag] = Baseline(
                    name=tag,
                    mean_cost=float(eval_cost),
                    source="recomputed:one_warehouse_multi_retailer_simulate_policy",
                    params={
                        "warehouse_base_stock_level": int(w),
                        "retailer_base_stock_levels": [int(x) for x in levels],
                        "allocation_policy": allocation,
                    },
                    note=(
                        f"live env, periods={p['benchmark_periods']}, "
                        f"eval_reps={self._EVAL_REPLICATIONS}, eval_seed={self._EVAL_SEED}, "
                        f"search_reps={search_reps}/seed={search_seed}"
                        + (f", gap vs published {gap:+.2f}%" if gap is not None else "")
                    ),
                )
                if cheapest is None or eval_cost < cheapest[1]:
                    cheapest = (tag, eval_cost)
            except Exception as exc:  # None-safe
                out[tag] = Baseline(
                    name=tag,
                    mean_cost=None,
                    source=f"simulate_failed:{type(exc).__name__}",
                )
        # Tag the cheapest re-simulated echelon base-stock as the canonical gate.
        if cheapest is not None:
            t, _ = cheapest
            b = out[t]
            out[t] = Baseline(
                name=b.name,
                mean_cost=b.mean_cost,
                source=b.source,
                std_cost=b.std_cost,
                params=b.params,
                is_published=b.is_published,
                is_optimal=b.is_optimal,
                is_reference=True,
                note=b.note,
            )
        return out
