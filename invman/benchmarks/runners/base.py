"""Executable baseline layer — shared abstractions (the "runnable env + reference cost").

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
The catalog (`invman.benchmarks.catalog`) answers "what are the reference
problems and their published numbers?" purely from the manifest — it never runs
anything. This module is the EXECUTABLE counterpart: given a problem family and a
named reference instance, it returns a `ReferenceInstance` from which a consumer
can, with minimal effort,
  (1) read the env parameters of the literature instance,
  (2) read the published / literature reference baselines (the numbers to beat),
  (3) RE-RUN the shipped baseline on the live env (`run_baselines`) — proving the
      env is runnable and reproducing the reference numbers, and
  (4) drop in THEIR OWN soft-tree policy and score it on the SAME instance under
      the SAME eval protocol (`evaluate`) so they can compare apples-to-apples.

This is `PROPER_REPO_BUILD_PLAN/README.md` workstream (a): one uniform Python surface so
a benchmark consumer never has to parse Rust or markdown. Each problem family has
a different env contract (lost-sales scalar order vs dual-sourcing two-source
vector vs multi-echelon allocation), so the family specifics live in per-family
`ProblemRunner` subclasses; this base file fixes the SHARED vocabulary and the
shared evaluation seam.

Why these pieces (each requirement maps to the objective)
---------------------------------------------------------
* `EvalProtocol` — the eval contract (optimizer/eval seeds, horizon, warm-up,
  replications). The repo mandate is mean over >=5 seeds (seed-robust reporting);
  a `published_protocol` reproduces the literature setting, a `smoke_protocol`
  is a fast "does my harness run?" setting. Carrying the protocol explicitly is
  what makes "compare under the SAME protocol" enforceable rather than implicit.
* `Baseline` — one comparator number with provenance (published vs recomputed),
  whether it is the exact optimum, and the policy params that achieve it. The
  honesty discipline of the repo requires never conflating a published number
  with a recomputed one; the `source` + `is_published` fields keep that explicit.
* `ReferenceInstance` — the object a consumer holds. It bundles the env params
  (free, from the Rust accessor), the published baselines (free), and BOUND
  methods that actually run the env. `reference_baseline`/`reference_cost` pick
  the single canonical number to beat (the exact optimum if one exists, else the
  cheapest available heuristic). `compare(my_cost)` reports the signed gap so a
  user gets a verdict, not just two numbers.
* `ProblemRunner` — the abstract per-family driver. Subclasses implement the
  family-specific Rust calls; the base class implements the family-INDEPENDENT
  logic (load, multi-seed averaging in `evaluate`, param-count discovery) so each
  runner stays small and the uniform behaviour is written once.

Evaluation seam (how `evaluate` stays identical to training)
------------------------------------------------------------
`evaluate` does NOT re-implement a rollout. It reuses the exact CMA-ES training
seam: `invman.config.get_config` -> set env fields from the instance ->
`apply_policy_name` -> `invman.policy_build.build_policy` ->
`invman.rollout_fitness.get_model_fitness`. So a policy scored here is scored by
byte-identical code to the optimizer's fitness — there is no second, drifting
evaluator. The per-family subclass only supplies `_eval_model_and_args`; the base
class loops the protocol seeds and averages (lower cost = better).

Dependencies: numpy + the in-repo `invman.*` optimizer layer + `invman_rust`
(only when a method that actually runs the env is called; constructing a
`ReferenceInstance` and reading its published baselines needs neither).
================================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


# ---------------------------------------------------------------------------
# Eval protocol — the comparison contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalProtocol:
    """How a policy / baseline is scored on an instance.

    `seeds` are the (optimizer or CRN eval) seeds the mean is taken over — the
    repo mandate is >=5 for a seed-robust headline. `replications` is consumed
    by search-based baselines (multi-echelon constant base-stock search) that
    average over internal demand replications rather than a seed list.
    """

    seeds: tuple[int, ...] = (1234,)
    horizon: int = 2000
    warm_up_periods_ratio: float = 0.2
    replications: int = 1
    label: str = ""

    def with_overrides(
        self,
        *,
        seeds: Optional[Sequence[int]] = None,
        horizon: Optional[int] = None,
        warm_up_periods_ratio: Optional[float] = None,
        replications: Optional[int] = None,
    ) -> "EvalProtocol":
        """Return a copy with selected fields overridden (None = keep)."""
        return EvalProtocol(
            seeds=tuple(int(s) for s in seeds) if seeds is not None else self.seeds,
            horizon=self.horizon if horizon is None else int(horizon),
            warm_up_periods_ratio=(
                self.warm_up_periods_ratio
                if warm_up_periods_ratio is None
                else float(warm_up_periods_ratio)
            ),
            replications=self.replications if replications is None else int(replications),
            label=self.label,
        )


# ---------------------------------------------------------------------------
# Baseline — one comparator number with provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Baseline:
    """One comparator cost (lower is better) for a reference instance."""

    name: str
    mean_cost: Optional[float]
    source: str
    std_cost: Optional[float] = None
    params: Optional[dict] = None
    is_published: bool = False
    is_optimal: bool = False
    # The family's CANONICAL comparator (e.g. multi-echelon's "beat constant
    # base-stock"), used as the reference even when it is not the cheapest.
    is_reference: bool = False
    note: str = ""

    @property
    def available(self) -> bool:
        return self.mean_cost is not None


# ---------------------------------------------------------------------------
# ReferenceInstance — the object a consumer holds
# ---------------------------------------------------------------------------


@dataclass
class ReferenceInstance:
    """A literature instance: env params + published baselines + bound runners."""

    problem: str
    name: str
    subfamily: str
    params: dict
    source: str
    notes: str
    published_baselines: list[Baseline]
    runner: "ProblemRunner" = field(repr=False, default=None)

    # -- free reads (no env execution) ------------------------------------
    @property
    def published_costs(self) -> dict[str, float]:
        """{baseline_name: cost} for every available published baseline."""
        return {b.name: b.mean_cost for b in self.published_baselines if b.available}

    @property
    def _lower_is_better(self) -> bool:
        """Score direction; profit families (ameliorating) maximize, so False."""
        return getattr(self.runner, "lower_is_better", True)

    def _best(self, baselines: list[Baseline]) -> Baseline:
        key = (lambda b: b.mean_cost) if self._lower_is_better else (lambda b: -b.mean_cost)
        return min(baselines, key=key)

    @property
    def reference_baseline(self) -> Optional[Baseline]:
        """The single canonical number to beat.

        Preference order: a baseline the family explicitly declared the canonical
        comparator (`is_reference`, e.g. multi-echelon's constant base-stock) >
        the exact optimum (`is_optimal`) > the best available published baseline
        (the strongest heuristic). "Best" = cheapest for cost families, highest
        for profit families (`runner.lower_is_better`). None if nothing available.
        """
        available = [b for b in self.published_baselines if b.available]
        if not available:
            return None
        declared = [b for b in available if b.is_reference]
        if declared:
            return self._best(declared)
        optima = [b for b in available if b.is_optimal]
        if optima:
            return self._best(optima)
        return self._best(available)

    @property
    def reference_cost(self) -> Optional[float]:
        ref = self.reference_baseline
        return None if ref is None else ref.mean_cost

    # -- literature-verification self-report (from the manifest) ----------
    @property
    def verification_tier(self) -> str:
        """The honest tier ('strict'|'reference'|'faithful'|'mixed') of this family."""
        return self.runner.verification_tier

    @property
    def literature_verified(self) -> bool:
        """True unless the family is `faithful` (repo-native, no public anchor)."""
        return self.runner.literature_verified

    # -- env-running methods (require invman_rust) ------------------------
    def run_baselines(self, protocol: Optional[EvalProtocol] = None) -> dict[str, Baseline]:
        """Re-run the shipped baseline(s) on the live env and return the costs.

        This is the "runnable env" proof: it actually simulates, reproducing the
        published reference numbers (within Monte-Carlo / solver tolerance).
        """
        return self.runner._run_baselines(self, protocol or self.runner.smoke_protocol)

    def evaluate(
        self,
        flat_params: Sequence[float],
        *,
        protocol: Optional[EvalProtocol] = None,
        seeds: Optional[Sequence[int]] = None,
        horizon: Optional[int] = None,
        warm_up_periods_ratio: Optional[float] = None,
        **structure: Any,
    ) -> float:
        """Score a soft-tree policy on THIS instance; return mean cost (lower=better).

        `flat_params` is the trained soft-tree weight vector (length must equal
        `policy_param_count(**structure)`). `structure` are the tree hyper-params
        (depth / temperature / split_type / leaf_type and the family's action
        design); each runner fills sensible defaults. The mean is taken over the
        protocol's seeds, scored by the SAME code the CMA-ES optimizer uses.
        """
        proto = (protocol or self.runner.published_protocol).with_overrides(
            seeds=seeds, horizon=horizon, warm_up_periods_ratio=warm_up_periods_ratio
        )
        costs = [
            self.runner._evaluate_single(self, flat_params, structure, int(seed), proto)
            for seed in proto.seeds
        ]
        return float(sum(costs) / len(costs))

    def policy_param_count(self, **structure: Any) -> int:
        """Expected length of `flat_params` for a soft-tree policy of `structure`."""
        return self.runner._policy_param_count(self, structure)

    def compare(self, my_cost: float, *, against: Optional[str] = None) -> dict:
        """Compare a user's cost to the reference (or a named published baseline).

        Returns the signed gap (positive = the user is WORSE, since lower cost is
        better) in absolute and percentage terms, plus a boolean `beats`.
        """
        if against is not None:
            target = next(
                (b for b in self.published_baselines if b.name == against and b.available),
                None,
            )
            if target is None:
                raise KeyError(
                    f"no available published baseline {against!r} for {self.name!r}; "
                    f"have {sorted(self.published_costs)}"
                )
            ref_name, ref_cost = target.name, target.mean_cost
        else:
            ref = self.reference_baseline
            if ref is None:
                raise ValueError(f"{self.name!r} has no published reference cost to compare against")
            ref_name, ref_cost = ref.name, ref.mean_cost
        # Signed so a POSITIVE gap always means "worse than the reference",
        # regardless of whether the family minimizes cost or maximizes profit.
        raw_delta = float(my_cost) - float(ref_cost)
        gap_abs = raw_delta if self._lower_is_better else -raw_delta
        gap_pct = 100.0 * gap_abs / abs(float(ref_cost)) if ref_cost else float("nan")
        return {
            "instance": self.name,
            "reference": ref_name,
            "reference_cost": float(ref_cost),
            "my_cost": float(my_cost),
            "gap_abs": gap_abs,
            "gap_pct": gap_pct,
            "beats": gap_abs < 0.0,
        }


# ---------------------------------------------------------------------------
# ProblemRunner — abstract per-family driver
# ---------------------------------------------------------------------------


class ProblemRunner(ABC):
    """One inventory family's executable baseline driver.

    Subclasses implement the family-specific Rust calls (`_reference_dict`,
    `_published_baselines`, `_run_baselines`, `_eval_model_and_args`); the base
    class implements the family-independent surface (`load_instance`, multi-seed
    `_evaluate_single`, `_policy_param_count`).
    """

    #: Catalog problem name this runner serves (e.g. "lost_sales").
    problem: str = ""
    #: Literature-faithful eval protocol (reproduces the published setting).
    published_protocol: EvalProtocol = EvalProtocol()
    #: Fast "does my harness run?" protocol.
    smoke_protocol: EvalProtocol = EvalProtocol(seeds=(1234,), horizon=1000)
    #: True only when this family's soft-tree rollout is wired into the uniform
    #: CMA-ES eval seam (build_policy + get_model_fitness) so `evaluate()` works.
    #: Metadata-only runners (params + published baselines + run_baselines, but no
    #: in-seam policy scoring yet) set this False; `evaluate()` then fails loudly
    #: with a pointer rather than pretending to score.
    supports_evaluate: bool = True
    #: Score direction. Cost families minimize (True); profit families
    #: (ameliorating_inventory) maximize — set False so `reference_baseline` picks
    #: the highest and `compare` keeps "positive gap = worse".
    lower_is_better: bool = True

    # -- abstract family hooks --------------------------------------------
    @abstractmethod
    def list_instances(self) -> list[str]:
        """All reference-instance names for this family (manifest/Rust order)."""

    @abstractmethod
    def primary_instance(self) -> str:
        """The canonical instance returned when `load_instance(None)` is called."""

    @abstractmethod
    def _reference_dict(self, name: str) -> dict:
        """Raw Rust reference-instance dict for `name` (raises on unknown)."""

    @abstractmethod
    def _subfamily_of(self, name: str) -> str:
        """Subfamily tag for `name` (e.g. 'vanilla' vs 'fixed_order_cost')."""

    @abstractmethod
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        """The free published / literature baselines for `name`."""

    @abstractmethod
    def _run_baselines(self, inst: ReferenceInstance, protocol: EvalProtocol) -> dict[str, Baseline]:
        """Re-simulate the shipped baseline(s) on the live env."""

    def _eval_model_and_args(
        self, inst: ReferenceInstance, structure: dict, protocol: EvalProtocol
    ):
        """Build `(model, args)` for the CMA-ES eval seam on this instance.

        Override in a runner whose soft-tree rollout is wired through
        `build_policy` + `get_model_fitness` (and set `supports_evaluate=True`).
        The default raises — a metadata-only runner does not score policies yet.
        """
        raise NotImplementedError(
            f"policy evaluation is not yet wired into the uniform runner for "
            f"{self.problem!r}. The env params + published baselines + "
            f"run_baselines() above ARE runnable; the family's soft-tree rollout "
            f"is `invman_rust.{self.problem}_soft_tree_rollout` (kwargs differ per "
            f"family — see scripts/{self.problem}/). Wiring it through "
            f"invman.policy_build.build_policy + invman.rollout_fitness."
            f"get_model_fitness is the next increment."
        )

    # -- literature-verification (single source of truth = the manifest) --
    @property
    def verification_tier(self) -> str:
        """The family's honest tier, read from the manifest via the catalog.

        'strict' (re-runs a peer-reviewed printed number) / 'reference'
        (companion-code / closed-form / published-action) / 'mixed' (umbrella
        with verified sub-families) / 'faithful' (repo-native, NO public anchor).
        """
        from invman.benchmarks import catalog

        return catalog.get(self.problem).verification_tier

    @property
    def literature_verified(self) -> bool:
        """True iff the family reproduces a real literature anchor.

        Derived from the manifest tier (the single source of truth): everything
        except `faithful` is literature-anchored. This matches, family-for-family,
        the adversarial audit in
        docs/benchmarks/LITERATURE_VERIFICATION_AUDIT_2026_06_12/README.md.
        """
        return self.verification_tier != "faithful"

    # -- concrete shared surface ------------------------------------------
    def load_instance(self, name: Optional[str] = None) -> ReferenceInstance:
        """Return the `ReferenceInstance` for `name` (or the primary instance)."""
        resolved = self.primary_instance() if name is None else str(name)
        inst_dict = self._reference_dict(resolved)
        return ReferenceInstance(
            problem=self.problem,
            name=resolved,
            subfamily=self._subfamily_of(resolved),
            params=dict(inst_dict),
            source=str(inst_dict.get("source", "")),
            notes=str(inst_dict.get("notes", "")),
            published_baselines=self._published_baselines(resolved, inst_dict),
            runner=self,
        )

    def load_all(self) -> list[ReferenceInstance]:
        """Every reference instance for this family."""
        return [self.load_instance(name) for name in self.list_instances()]

    def _evaluate_single(
        self,
        inst: ReferenceInstance,
        flat_params: Sequence[float],
        structure: dict,
        seed: int,
        protocol: EvalProtocol,
    ) -> float:
        """Score `flat_params` on `inst` for ONE seed via the CMA-ES eval seam."""
        import numpy as np

        from invman.rollout_fitness import get_model_fitness

        if not self.supports_evaluate:
            # Triggers the actionable NotImplementedError below.
            self._eval_model_and_args(inst, structure, protocol)
        model, args = self._eval_model_and_args(inst, structure, protocol)
        params = np.asarray(flat_params, dtype=np.float32)
        if params.size != model.num_params:
            raise ValueError(
                f"flat_params length {params.size} != expected {model.num_params} for a "
                f"{self.problem} soft-tree of structure {self._structure_with_defaults(structure)}; "
                f"use ReferenceInstance.policy_param_count(**structure) to size it"
            )
        neg_cost, _ = get_model_fitness(model, args, model_params=params, seed=int(seed))
        return -float(neg_cost)

    def _policy_param_count(self, inst: ReferenceInstance, structure: dict) -> int:
        """Expected soft-tree `flat_params` length for `structure` on `inst`."""
        model, _ = self._eval_model_and_args(inst, structure, self.smoke_protocol)
        return int(model.num_params)

    # -- structure defaults (overridable per family) ----------------------
    #: Default soft-tree structure for this family's `evaluate`.
    default_structure: dict = {
        "depth": 2,
        "temperature": 0.25,
        "split_type": "oblique",
        "leaf_type": "linear",
    }

    def _structure_with_defaults(self, structure: dict) -> dict:
        merged = dict(self.default_structure)
        merged.update({k: v for k, v in (structure or {}).items() if v is not None})
        return merged
