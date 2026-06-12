"""Executable baseline runner — `lost_sales` (vanilla + fixed_order_cost).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the lost-sales family RUNNABLE from one uniform handle: list the literature
instances, read their published baselines, re-run those baselines on the live
env, and score a user's soft-tree policy on the same instance. The catalog
problem `lost_sales` covers TWO subfamilies (the manifest lists both under one
entry), so this one runner composes them:

  * vanilla            — Zipkin (2008) lost-sales grid. 33 reference instances
                         (`lost_sales_reference_instance_names`). Published
                         baselines: optimal / myopic1 / myopic2 / svbs /
                         capped_base_stock (`lost_sales_reference_costs`).
  * fixed_order_cost   — Bijvank, Bhulai & Huh (2015) lost-sales with a fixed
                         order cost K. Reference instance(s) via
                         `lost_sales_fixed_order_cost_list_reference_instances`;
                         published baselines: optimal_dp / (s,S) / (s,nQ) /
                         modified (s,S,q).

Both subfamilies use the SAME scalar-order soft-tree rollout
(`get_model_fitness` dispatches both to `_lost_sales_single`), so the only
difference here is which Rust accessor supplies the params + baselines and
whether `fixed_order_cost > 0`.

How each method serves the objective
------------------------------------
* `_reference_dict` / `_published_baselines` — read the env params + the free
  published numbers (no simulation). For vanilla the costs come from the
  reference-cost config (literature where the `source` says so, repo-computed
  otherwise — surfaced honestly via `is_published`). For fixed-cost the
  published optimal + the three heuristic rows (with their (s,S)/(s,nQ)/(s,S,q)
  params) come straight off the reference instance.
* `_run_baselines` — the "runnable env" proof. Vanilla re-evaluates
  myopic1/myopic2/svbs on the live simulator (`lost_sales_heuristics_all`);
  fixed-cost re-solves the exact average-cost DP and re-evaluates the published
  (s,S)/(s,nQ)/(s,S,q) policies (`lost_sales_fixed_order_cost_exact_literature_
  summary`), reproducing the published costs within tolerance.
* `_eval_model_and_args` — build the CMA-ES eval seam (config -> env fields ->
  policy) so a user's trained soft-tree is scored by identical code to training.

Verification tier: vanilla `fixed`/`Poisson` rows reproduce Zipkin; the fixed
instance reproduces Bijvank Table 1 (strict). See the manifest / BENCHMARK cards.
Dependencies: `invman_rust` + the `invman.*` optimizer layer (only when running).
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

# Map the reference-cost demand token to the rollout's demand-distribution name.
_DEMAND_DIST_NAME = {"poisson": "Poisson", "geometric": "Geometric"}


class LostSalesRunner(ProblemRunner):
    """Runnable baseline driver for the lost_sales family (vanilla + fixed-cost)."""

    problem = "lost_sales"
    # Zipkin/Bijvank-style long-run average cost: a long single-seed horizon is
    # the literature setting; >=5 seeds is the repo's seed-robust headline rule.
    published_protocol = EvalProtocol(
        seeds=(1234, 2025, 7, 99, 1000), horizon=20000, warm_up_periods_ratio=0.2
    )
    smoke_protocol = EvalProtocol(seeds=(1234,), horizon=4000, warm_up_periods_ratio=0.2)
    default_structure = {
        "depth": 2,
        "temperature": 0.25,
        "split_type": "oblique",
        "leaf_type": "linear",
    }
    # Per-period order cap for the soft-tree policy on the fixed-cost subfamily
    # (matches the exact solver's inventory-position cap; only the user's policy
    # uses it, not the published-baseline reproduction).
    _FIXED_MAX_ORDER_SIZE = 24

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust
        self._vanilla_names = list(invman_rust.lost_sales_reference_instance_names())
        self._fixed_names = list(
            invman_rust.lost_sales_fixed_order_cost_list_reference_instances()
        )
        self._fixed_set = set(self._fixed_names)

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return list(self._vanilla_names) + list(self._fixed_names)

    def primary_instance(self) -> str:
        # The canonical literature-verified Poisson L=4 row.
        return "lit_poisson_p4_l4" if "lit_poisson_p4_l4" in self._vanilla_names else self._vanilla_names[0]

    def _is_fixed(self, name: str) -> bool:
        return name in self._fixed_set

    def _subfamily_of(self, name: str) -> str:
        return "fixed_order_cost" if self._is_fixed(name) else "vanilla"

    # -- reference dicts --------------------------------------------------
    def _reference_dict(self, name: str) -> dict:
        if self._is_fixed(name):
            inst = self._rust.lost_sales_fixed_order_cost_get_reference_instance(name)
            if inst is None:
                raise KeyError(f"unknown lost_sales fixed-cost instance: {name!r}")
            return dict(inst)
        inst = self._rust.lost_sales_reference_costs(name)
        if inst is None:
            raise KeyError(
                f"unknown lost_sales instance: {name!r}. Known: {self.list_instances()}"
            )
        return dict(inst)

    # -- published (free) baselines ---------------------------------------
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        if self._is_fixed(name):
            return self._fixed_published_baselines(inst_dict)
        return self._vanilla_published_baselines(inst_dict)

    def _vanilla_published_baselines(self, inst_dict: dict) -> list[Baseline]:
        costs = dict(inst_dict.get("costs", {}))
        is_lit = str(inst_dict.get("source", "")).strip().lower() == "literature"
        source = f"reference_config:{inst_dict.get('source', 'computed')}"
        out: list[Baseline] = []
        for key in ("optimal", "myopic1", "myopic2", "svbs", "capped_base_stock"):
            value = costs.get(key)
            if value is None:
                continue
            out.append(
                Baseline(
                    name=key,
                    mean_cost=float(value),
                    source=source,
                    is_published=is_lit,
                    is_optimal=(key == "optimal"),
                )
            )
        return out

    def _fixed_published_baselines(self, inst_dict: dict) -> list[Baseline]:
        out: list[Baseline] = []
        opt = inst_dict.get("published_optimal_cost")
        if opt is not None:
            out.append(
                Baseline(
                    name="optimal_dp",
                    mean_cost=float(opt),
                    source=str(inst_dict.get("source", "")),
                    is_published=True,
                    is_optimal=True,
                )
            )
        for row in inst_dict.get("published_heuristic_rows", []):
            out.append(
                Baseline(
                    name=str(row.get("policy_name")),
                    mean_cost=float(row.get("mean_cost")),
                    source=str(inst_dict.get("source", "")),
                    params={"params": list(row.get("params", []))},
                    is_published=True,
                )
            )
        return out

    # -- run the env (the "runnable" proof) -------------------------------
    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        if inst.subfamily == "fixed_order_cost":
            return self._run_fixed_baselines(inst)
        return self._run_vanilla_baselines(inst, protocol)

    def _run_vanilla_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        p = inst.params
        seed = int(protocol.seeds[0])
        recomputed = dict(
            self._rust.lost_sales_heuristics_all(
                demand_kind=str(p["demand_kind"]),
                demand_rate=float(p["demand_rate"]),
                demand_lambda_low=float(p.get("demand_lambda_low", 0.0)),
                demand_lambda_high=float(p.get("demand_lambda_high", 0.0)),
                demand_p00=float(p.get("demand_p00", 0.0)),
                demand_p11=float(p.get("demand_p11", 0.0)),
                lead_time=int(p["lead_time"]),
                holding_cost=float(p["holding_cost"]),
                shortage_cost=float(p["shortage_cost"]),
                procurement_cost=0.0,
                fixed_order_cost=0.0,
                horizon=int(protocol.horizon),
                seed=seed,
                warm_up_periods_ratio=float(protocol.warm_up_periods_ratio),
                order_search_upper_bound=50,
                heuristic_discount_factor=0.99,
            )
        )
        out: dict[str, Baseline] = {}
        for key, value in recomputed.items():
            out[key] = Baseline(
                name=key,
                mean_cost=float(value),
                source="recomputed:lost_sales_heuristics_all",
                note=f"live env, horizon={protocol.horizon}, seed={seed}",
            )
        # The exact optimum + capped-base-stock come free from the config.
        for b in inst.published_baselines:
            if b.name in ("optimal", "capped_base_stock") and b.name not in out:
                out[b.name] = b
        return out

    def _run_fixed_baselines(self, inst: ReferenceInstance) -> dict[str, Baseline]:
        summary = dict(
            self._rust.lost_sales_fixed_order_cost_exact_literature_summary(
                inst.name, self._FIXED_MAX_ORDER_SIZE
            )
        )
        return {
            "optimal_dp": Baseline(
                name="optimal_dp",
                mean_cost=float(summary["optimal_average_cost"]),
                source="recomputed:exact_average_cost_value_iteration",
                is_optimal=True,
                note=f"published {summary.get('published_optimal_cost')}",
            ),
            "s_s": Baseline(
                name="s_s",
                mean_cost=float(summary["s_s_average_cost"]),
                source="recomputed:exact_policy_evaluation",
                note=f"published {summary.get('published_s_s_cost')}",
            ),
            "s_nq": Baseline(
                name="s_nq",
                mean_cost=float(summary["s_nq_average_cost"]),
                source="recomputed:exact_policy_evaluation",
                note=f"published {summary.get('published_s_nq_cost')}",
            ),
            "modified_s_s_q": Baseline(
                name="modified_s_s_q",
                mean_cost=float(summary["modified_s_s_q_average_cost"]),
                source="recomputed:exact_policy_evaluation",
                note=f"published {summary.get('published_modified_s_s_q_cost')}",
            ),
        }

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
        args.horizon = int(protocol.horizon)
        args.warm_up_periods_ratio = float(protocol.warm_up_periods_ratio)

        if inst.subfamily == "fixed_order_cost":
            args.problem = "lost_sales_fixed_order_cost"
            args.demand_rate = float(p["demand_mean_per_review_period"])
            args.demand_dist_name = _DEMAND_DIST_NAME.get(
                str(p.get("demand_distribution", "poisson")).lower(), "Poisson"
            )
            args.lead_time = int(p["lead_time"])
            args.holding_cost = float(p["holding_cost"])
            args.shortage_cost = float(p["shortage_cost"])
            args.fixed_order_cost = float(p["fixed_order_cost"])
            args.max_order_size = self._FIXED_MAX_ORDER_SIZE
        else:
            args.problem = "lost_sales"
            args.demand_rate = float(p["demand_rate"])
            args.demand_dist_name = str(p["demand_kind"])
            args.demand_lambda_low = float(p.get("demand_lambda_low", 0.0))
            args.demand_lambda_high = float(p.get("demand_lambda_high", 0.0))
            args.demand_p00 = float(p.get("demand_p00", 0.0))
            args.demand_p11 = float(p.get("demand_p11", 0.0))
            args.lead_time = int(p["lead_time"])
            args.holding_cost = float(p["holding_cost"])
            args.shortage_cost = float(p["shortage_cost"])
            args.fixed_order_cost = 0.0

        args.policy_name = make_soft_tree_policy_name(
            depth=int(s["depth"]),
            temperature=float(s["temperature"]),
            split_type=str(s["split_type"]),
            leaf_type=str(s["leaf_type"]),
        )
        apply_policy_name(args)
        return build_policy(args), args
