"""Executable baseline runner — `dual_sourcing` (Gijsbrechts 2022 Figure-9 family).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the dual-sourcing family runnable from one handle over the 6 published
Gijsbrechts et al. (2022) Section 6.2 / Figure-9 instances
(`dual_l{2,3,4}_ce{105,110}`): read the env params, read the PUBLISHED
optimality gaps, re-run the four shipped heuristics on the live env to get the
absolute reference costs, and score a user's soft-tree policy on the same
instance.

Why dual-sourcing baselines are GAP-based, not absolute
-------------------------------------------------------
The paper reports each policy's RELATIVE optimality gap (%), not an absolute
long-run cost: capped_dual_index 0.0%, tailored_base_surge 0.06%,
dual_index 0.11%, single_index 0.56%, A3C 0.52%. So:
  * `_published_baselines` carries those gaps (cost unknown -> `mean_cost=None`,
    `published_gap_pct` in `params`). They are the paper's headline numbers.
  * `_run_baselines` produces the ABSOLUTE costs by grid-searching each heuristic
    on one fixed demand path via the Rust `*_search_from_demands` bindings
    (single / dual / capped-dual index + tailored base surge). capped_dual_index
    is tagged the optimal-proxy: at ~0% published gap it is the strongest static
    policy and the practical "number to beat".
This mirrors `scripts/dual_sourcing/dual_sourcing_benchmark_lib.py` exactly (same
fixed-path, same searches), kept here as an importable library surface so a
consumer never has to reach into `scripts/`.

How each method serves the objective
------------------------------------
* `_reference_dict` / `_published_baselines` — env params + published gaps, free.
* `_run_baselines` — the runnable proof: simulate the four heuristics, return
  absolute costs; the cheapest available is the optimal proxy. The bounded
  average-cost DP optimum is intentionally NOT on this path (it is slow on the
  l_r in {3,4} rows; the capped-dual-index proxy is ~0% from it).
* `_eval_model_and_args` — build the CMA-ES eval seam. Dual sourcing is
  soft_tree-only; the default `action_adapter='identity'` learns raw two-source
  orders, and a user can pass `action_adapter='capped_dual_index_targets'` (etc.)
  via the structure to search over the published control geometry.

Verification tier: strict (the env's l_r=2 row has an executing reproduction of
the Gijs Figure-9 gap). Dependencies: `invman_rust`, numpy, the `invman.*` layer.
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

# The four published Gijsbrechts heuristics and the param names their search
# bindings return (best = [..., cost]).
_GIJS_GAP_KEYS = ("single_index", "dual_index", "capped_dual_index", "tailored_base_surge", "a3c")


class DualSourcingRunner(ProblemRunner):
    """Runnable baseline driver for the dual-sourcing Gijs Figure-9 family."""

    problem = "dual_sourcing"
    # The published evaluation is a long single path; >=5 seeds is the repo rule.
    published_protocol = EvalProtocol(
        seeds=(123, 2025, 7, 99, 1000), horizon=10000, warm_up_periods_ratio=0.2
    )
    smoke_protocol = EvalProtocol(seeds=(123,), horizon=3000, warm_up_periods_ratio=0.2)
    default_structure = {
        "depth": 2,
        "temperature": 0.25,
        "split_type": "oblique",
        "leaf_type": "constant",
        "action_adapter": "identity",
    }

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust
        self._instances = list(invman_rust.dual_sourcing_list_reference_instances())
        self._by_name = {str(d["name"]): dict(d) for d in self._instances}

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return list(self._by_name.keys())

    def primary_instance(self) -> str:
        return str(self._rust.dual_sourcing_primary_reference_instance_name())

    def _subfamily_of(self, name: str) -> str:
        return "linear_cost"

    def _reference_dict(self, name: str) -> dict:
        if name not in self._by_name:
            raise KeyError(
                f"unknown dual_sourcing instance: {name!r}. Known: {self.list_instances()}"
            )
        return dict(self._by_name[name])

    # -- published (free) baselines: the optimality GAPS ------------------
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        gaps = dict(inst_dict.get("published_optimality_gap_pct", {}) or {})
        out: list[Baseline] = []
        for key in _GIJS_GAP_KEYS:
            if key not in gaps:
                continue
            out.append(
                Baseline(
                    name=key,
                    mean_cost=None,  # the paper reports a gap, not an absolute cost
                    source=str(inst_dict.get("source", "")),
                    params={"published_gap_pct": float(gaps[key])},
                    is_published=True,
                    note=f"published optimality gap {float(gaps[key])}%",
                )
            )
        return out

    # -- run the env (the "runnable" proof): absolute heuristic costs ------
    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        import numpy as np

        p = inst.params
        seed = int(protocol.seeds[0])
        horizon = int(protocol.horizon)
        rng = np.random.RandomState(seed)
        demands = rng.randint(int(p["demand_low"]), int(p["demand_high"]) + 1, size=horizon)
        demands = demands.astype(int).tolist()
        state = [int(v) for v in p["initial_state"]]
        mean_demand = 0.5 * (int(p["demand_low"]) + int(p["demand_high"]))
        target_upper_bound = max(
            int(p["expedited_max_order_size"]),
            min(24, int(round((int(p["regular_lead_time"]) + 2) * mean_demand
                              + 2 * int(p["expedited_max_order_size"])))),
        )
        common = dict(
            state=state,
            demands=demands,
            regular_max_order_size=int(p["regular_max_order_size"]),
            expedited_max_order_size=int(p["expedited_max_order_size"]),
            regular_order_cost=float(p["regular_order_cost"]),
            expedited_order_cost=float(p["expedited_order_cost"]),
            holding_cost=float(p["holding_cost"]),
            shortage_cost=float(p["shortage_cost"]),
            warm_up_periods_ratio=float(protocol.warm_up_periods_ratio),
            top_k=3,
            target_upper_bound=target_upper_bound,
        )
        searches = {
            "single_index": (
                self._rust.dual_sourcing_single_index_search_from_demands,
                lambda b: {"s_e": int(b[0]), "s_r": int(b[1])},
            ),
            "dual_index": (
                self._rust.dual_sourcing_dual_index_search_from_demands,
                lambda b: {"s_e": int(b[0]), "s_r": int(b[1])},
            ),
            "capped_dual_index": (
                self._rust.dual_sourcing_capped_dual_index_search_from_demands,
                lambda b: {"s_e": int(b[0]), "s_r": int(b[1]), "cap_r": int(b[2])},
            ),
            "tailored_base_surge": (
                self._rust.dual_sourcing_tailored_base_surge_search_from_demands,
                lambda b: {"surge_level": int(b[0]), "regular_qty": int(b[1])},
            ),
        }
        out: dict[str, Baseline] = {}
        for hname, (search_fn, params_of) in searches.items():
            try:
                best, _top = search_fn(**common)
                out[hname] = Baseline(
                    name=hname,
                    mean_cost=float(best[-1]),
                    source="recomputed:rust_search_from_demands",
                    params=params_of(best),
                    # capped_dual_index is the published ~0%-gap optimal proxy.
                    is_optimal=(hname == "capped_dual_index"),
                    note=f"live env, horizon={horizon}, seed={seed}",
                )
            except Exception as exc:  # None-safe: a failed search must not abort a sweep
                out[hname] = Baseline(
                    name=hname,
                    mean_cost=None,
                    source=f"rust_search_failed:{type(exc).__name__}",
                )
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
        args.problem = "dual_sourcing"
        args.horizon = int(protocol.horizon)
        args.warm_up_periods_ratio = float(protocol.warm_up_periods_ratio)
        args.regular_lead_time = int(p["regular_lead_time"])
        args.expedited_lead_time = int(p["expedited_lead_time"])
        args.regular_order_cost = float(p["regular_order_cost"])
        args.expedited_order_cost = float(p["expedited_order_cost"])
        args.holding_cost = float(p["holding_cost"])
        args.shortage_cost = float(p["shortage_cost"])
        args.regular_max_order_size = int(p["regular_max_order_size"])
        args.expedited_max_order_size = int(p["expedited_max_order_size"])
        args.dual_demand_low = int(p["demand_low"])
        args.dual_demand_high = int(p["demand_high"])
        args.policy_name = make_soft_tree_policy_name(
            depth=int(s["depth"]),
            temperature=float(s["temperature"]),
            split_type=str(s["split_type"]),
            leaf_type=str(s["leaf_type"]),
            action_adapter=str(s.get("action_adapter", "identity")),
        )
        apply_policy_name(args)
        return build_policy(args), args
