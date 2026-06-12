"""Executable baseline runner — `spare_parts_inventory` (Kranenburg 2006 Table 5.2).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the spare-parts family RUNNABLE from one uniform handle: list the literature
instances, read their published reference costs, and RE-SOLVE the exact optimum on
the live solver so a consumer can verify the env is runnable and reproduces the
published numbers — without parsing Rust or markdown.

HONESTY LEDGER — two DIFFERENT models share the catalog name `spare_parts_inventory`
-----------------------------------------------------------------------------------
The repo carries two distinct spare-parts artifacts under one catalog name, and
they are NOT the same model:

  (A) The KRANENBURG (2006) analytical lateral-transshipment module — a single-item
      multi-location system with symmetric local warehouses, Poisson demand,
      emergency replenishment and optional lateral transshipment, solved EXACTLY in
      closed form (an O(R) marginal-analysis sweep over the randomized stock R).
      35 literature-verified reference rows from PhD-thesis Table 5.2
      (`spare_parts_inventory_kranenburg_reference_instances`), each with a published
      optimal R and cost for Situation 1 (no lateral transshipment) and Situation 3
      (at-most-one-per-warehouse). `spare_parts_inventory_kranenburg_exact_summary`
      re-derives those numbers and the repo verifies them within tol=0.02.

  (B) The repo-native single-echelon PERIODIC-REVIEW repairable env that the soft-tree
      policy actually TRAINS on (`spare_parts_inventory_primary_reference_instance` =
      `single_echelon_repairable_operational_spares`). This env is explicitly
      `literature_verified=false` (no paper publishes a matching numeric cost); its
      only exact check is a reduced finite-horizon DP self-consistency verifier
      (`spare_parts_inventory_exact_dp_summary`, on a 3-installed-base / 4-period
      instance) proving the optimal DP dominates the carried heuristics.

This runner uses model (A) — the Kranenburg analytical instances — as the family's
reference surface, because that is where the executable, literature-verified
optimum lives (verification tier = reference). EVERY baseline this runner emits is
tagged with a `note` stating that the trainable soft-tree env is model (B), a
DIFFERENT model, so a consumer never conflates the analytical optimum with a policy
score on the trainable env. `supports_evaluate=False`: this family's soft-tree
rollout (`spare_parts_inventory_soft_tree_rollout`) is NOT in the uniform
`build_policy`/`get_model_fitness` seam, so policy scoring is intentionally out of
scope here (the base class raises an actionable error if `evaluate()` is called).

How each method serves the objective
------------------------------------
* `list_instances` / `primary_instance` / `_reference_dict` — drive off
  `spare_parts_inventory_kranenburg_reference_instances` (the per-instance param
  list); `_reference_dict` raises KeyError on an unknown name.
* `_subfamily_of` — one constant tag, `kranenburg2006_table5_2` (all 35 rows are
  Table-5.2 sensitivity sweeps of the same base case).
* `_published_baselines` — the free published numbers: Situation 1 and Situation 3
  optimal costs (`published_situation{1,3}_cost`) straight off the row. Situation 1
  (the headline "no lateral transshipment" optimum the thesis tabulates) is tagged
  `is_optimal=True, is_reference=True`; Situation 3 is the published variant.
* `_run_baselines` — the runnable proof: call `kranenburg_exact_summary(name)` to
  RE-SOLVE the exact analytical optimum on the live solver, returning the
  recomputed Situation 1 / Situation 3 total costs (which reproduce the published
  costs within the repo's tol=0.02). None-safe: a failed binding yields a
  `mean_cost=None` Baseline tagged `...failed:<ExcType>`, never raises.

Verification tier: reference (analytical exact, literature-verified within tol=0.02).
Dependencies: `invman_rust` (lazy import in `__init__`).
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

# Standing caveat attached to every baseline: the executable/verified optimum is the
# Kranenburg analytical lateral-transshipment model, a DIFFERENT model from the
# repo-native periodic-review repairable env the soft-tree policy trains on.
_MODEL_CAVEAT = (
    "Kranenburg (2006) analytical lateral-transshipment optimum; a DIFFERENT model "
    "from the trainable periodic-review repairable env "
    "(single_echelon_repairable_operational_spares, literature_verified=false)."
)


class SparePartsInventoryRunner(ProblemRunner):
    """Runnable baseline driver for spare_parts_inventory (Kranenburg Table 5.2)."""

    problem = "spare_parts_inventory"
    # The Kranenburg optimum is a CLOSED-FORM analytical solve, not a simulation, so
    # the protocol horizons/seeds are nominal — `_run_baselines` ignores them. They
    # remain populated for the uniform surface (>=5 seeds is the repo headline rule).
    published_protocol = EvalProtocol(
        seeds=(1234, 2025, 7, 99, 1000), horizon=2000, warm_up_periods_ratio=0.2
    )
    smoke_protocol = EvalProtocol(seeds=(1234,), horizon=1000, warm_up_periods_ratio=0.2)
    # This family's soft-tree rollout is NOT wired into the uniform CMA-ES eval seam.
    supports_evaluate = False
    lower_is_better = True

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust
        self._instances = [
            dict(d)
            for d in invman_rust.spare_parts_inventory_kranenburg_reference_instances()
        ]
        self._by_name = {str(d["name"]): d for d in self._instances}

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return list(self._by_name.keys())

    def primary_instance(self) -> str:
        # The Kranenburg Table 5.2 base case is the canonical analytical reference;
        # the no-arg exact summary defaults to it.
        return str(
            self._rust.spare_parts_inventory_kranenburg_exact_summary()[
                "reference_instance"
            ]["name"]
        )

    def _subfamily_of(self, name: str) -> str:
        return "kranenburg2006_table5_2"

    # -- reference dicts --------------------------------------------------
    def _reference_dict(self, name: str) -> dict:
        inst = self._by_name.get(name)
        if inst is None:
            raise KeyError(
                f"unknown spare_parts_inventory instance: {name!r}. "
                f"Known: {self.list_instances()}"
            )
        return dict(inst)

    # -- published (free) baselines ---------------------------------------
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        source = str(inst_dict.get("source", ""))
        is_lit = bool(inst_dict.get("literature_verified", False))
        out: list[Baseline] = []
        # Situation 1 (no lateral transshipment) is the thesis's headline optimum
        # and the canonical comparator -> reference + optimal.
        s1_cost = inst_dict.get("published_situation1_cost")
        if s1_cost is not None:
            out.append(
                Baseline(
                    name="kranenburg_situation1_optimal",
                    mean_cost=float(s1_cost),
                    source=source,
                    params={"optimal_r": inst_dict.get("published_situation1_optimal_r")},
                    is_published=is_lit,
                    is_optimal=True,
                    is_reference=True,
                    note=f"published Situation 1 (no lateral transshipment) optimum; {_MODEL_CAVEAT}",
                )
            )
        # Situation 3 (at most one item per local warehouse) — published variant.
        s3_cost = inst_dict.get("published_situation3_cost")
        if s3_cost is not None:
            out.append(
                Baseline(
                    name="kranenburg_situation3_optimal",
                    mean_cost=float(s3_cost),
                    source=source,
                    params={"optimal_r": inst_dict.get("published_situation3_optimal_r")},
                    is_published=is_lit,
                    is_optimal=True,
                    note=f"published Situation 3 (<=1 per local warehouse) optimum; {_MODEL_CAVEAT}",
                )
            )
        return out

    # -- run the env (the "runnable" proof): re-solve the exact optimum ---
    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        try:
            summary = dict(
                self._rust.spare_parts_inventory_kranenburg_exact_summary(
                    instance_name=inst.name
                )
            )
        except Exception as exc:  # None-safe: a failed solve must not abort a sweep.
            failed = Baseline(
                name="kranenburg_situation1_optimal",
                mean_cost=None,
                source=f"recomputed:kranenburg_exact_summary_failed:{type(exc).__name__}",
                note=_MODEL_CAVEAT,
            )
            return {failed.name: failed}

        evaluation = dict(summary.get("evaluation", {}) or {})
        ref = dict(summary.get("reference_instance", {}) or {})
        comparison = dict(summary.get("published_table_comparison", {}) or {})
        within = bool(comparison.get("all_within_tolerance", False))
        tol = comparison.get("tolerance")
        out: dict[str, Baseline] = {}

        s1 = evaluation.get("situation1")
        if isinstance(s1, dict) and s1.get("total_cost") is not None:
            out["kranenburg_situation1_optimal"] = Baseline(
                name="kranenburg_situation1_optimal",
                mean_cost=float(s1["total_cost"]),
                source="recomputed:spare_parts_inventory_kranenburg_exact_summary",
                params={
                    "optimal_r": s1.get("optimal_r"),
                    "emergency_probability": s1.get("emergency_probability"),
                    "mean_waiting_time": s1.get("mean_waiting_time"),
                },
                is_optimal=True,
                is_reference=True,
                note=(
                    f"re-solved exact optimum; published {ref.get('published_situation1_cost')}, "
                    f"all_within_tol={within} (tol={tol}); {_MODEL_CAVEAT}"
                ),
            )

        s3 = evaluation.get("situation3")
        if isinstance(s3, dict) and s3.get("total_cost") is not None:
            out["kranenburg_situation3_optimal"] = Baseline(
                name="kranenburg_situation3_optimal",
                mean_cost=float(s3["total_cost"]),
                source="recomputed:spare_parts_inventory_kranenburg_exact_summary",
                params={
                    "optimal_r": s3.get("optimal_r"),
                    "emergency_probability": s3.get("emergency_probability"),
                    "mean_waiting_time": s3.get("mean_waiting_time"),
                },
                is_optimal=True,
                note=(
                    f"re-solved exact optimum; published {ref.get('published_situation3_cost')}, "
                    f"all_within_tol={within} (tol={tol}); {_MODEL_CAVEAT}"
                ),
            )
        return out
