"""Executable baseline runner — `ameliorating_inventory` (Pahr & Grunow 2025 blending).

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Make the ameliorating-inventory family (Pahr & Grunow 2025, Production and
Operations Management 35(5), DOI 10.1177/10591478251387795; companion repo
amelioratinginventory/ameliorating_inventory) runnable from one uniform handle
over its 2 literature reference instances. This is a PROFIT-MAXIMIZING family:
the controllable action is the per-period purchase volume of an ameliorating
(value-gaining-with-age) good, and the objective is long-run AVERAGE PROFIT, so
HIGHER is better. For each instance a consumer can:
  (1) read the env params (age classes, products, target ages, capacity,
      evaporation, holding cost, blending flag) from references.rs;
  (2) read the PUBLISHED perfect-information LP UPPER BOUND on average profit
      (companion `upper_bound.json`); and
  (3) RE-SOLVE that LP bound on the live solver (`run_baselines`), reproducing
      the published bound to < 1e-7.

This family ships NO Rust list/get reference accessor in the Python module, so
the instance names and params are derived by reading
`src/problems/ameliorating_inventory/references.rs` directly: the two
`REFERENCE_INSTANCES` rows — `pahr_grunow2025_spirits_0001` (the companion
default spirits config: 10 ages, 3 products, target ages 2/4/6, capacity 50,
evaporation 0.03, holding 2.5, no blending) and `pahr_grunow2025_port_wine` (the
industry port-wine case study: 25 ages, 2 products, target ages 9/19, capacity
50, evaporation 0.02, holding 1.0, blending enabled).

CRITICAL — this is a BOUND, not an achievable optimum
-----------------------------------------------------
The perfect-information LP solves the relaxation where all demand / sales-price
realizations are known in advance, so its average profit is an UPPER BOUND that
no causal policy can exceed; it is NOT the optimal achievable profit. Every
baseline is tagged accordingly (`is_optimal=False`, the note says "upper bound").
`reference_baseline` selects it as the canonical number to approach-from-below
(`is_reference=True`), and because `lower_is_better=False` the base class's
"best = highest" / "positive gap = worse" logic stays correct.

Why these pieces serve the objective
------------------------------------
* `_reference_dict` — builds the per-instance param dict from the references.rs
  constants embedded below (no Rust accessor exists), including the published LP
  bound (`published_max_reward`) for the free baseline.
* `_published_baselines` — surfaces the companion-published LP upper bound on
  average profit (`is_published=True`, `is_reference=True`, `is_optimal=False`;
  the note flags it as a BOUND). These are values printed in the companion
  `upper_bound.json` and reproduced to < 1e-3 by the in-crate solver.
* `_run_baselines` — the runnable proof: re-solve the perfect-information LP via
  `ameliorating_inventory_perfect_info_lp_bound_summary(reference_name=...)`,
  returning the freshly re-solved `upper_bound_max_reward` (and carrying the
  published anchor + gap in the note). The simulate-a-heuristic-profit path is
  intentionally NOT taken: the heuristic-rollout bindings
  (`*_simulate_policy` / `*_average_profit_soft_tree_rollout`) require the full
  demand / price / sales / salvage process fields that live in the LP DATASET
  files, not in references.rs, so re-running them faithfully from this metadata
  surface is out of scope — the LP bound is the cleanly-runnable comparator.

`supports_evaluate=False`: the family's faithful average-profit soft-tree rollout
(`ameliorating_inventory_average_profit_soft_tree_rollout`) is NOT wired through
the uniform `build_policy` / `get_model_fitness` seam (and needs the dataset
process fields), so policy scoring is out of scope here; the base
`_eval_model_and_args` raises an actionable pointer.

`lower_is_better=False` (PROFIT family; maximize long-run average profit).

Verification note: `run_baselines` re-solves the perfect-information LP and
reproduces the companion-published `published_max_reward` to < 1e-7 for both
instances (verified: spirits_0001 1991.9344293930808 vs published
1991.9344293376805; port_wine 2444.801064407908 vs published 2444.8010643781136).
Dependencies: `invman_rust` (only inside `_run_baselines`).
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

# ---------------------------------------------------------------------------
# Constants transcribed from src/problems/ameliorating_inventory/references.rs
# (REFERENCE_INSTANCES = [PRIMARY_REFERENCE_INSTANCE, PORT_WINE_REFERENCE_INSTANCE]).
# No Rust accessor exposes these to Python, so they are the source of truth here.
# `lp_reference_name` is the key the perfect-info LP binding accepts.
# ---------------------------------------------------------------------------

_SOURCE = (
    "Pahr and Grunow (2025), Production and Operations Management 35(5), "
    "DOI 10.1177/10591478251387795"
)
_REPO_URL = "https://github.com/amelioratinginventory/ameliorating_inventory"

_REFERENCE_INSTANCES = {
    "pahr_grunow2025_spirits_0001": dict(
        name="pahr_grunow2025_spirits_0001",
        lp_reference_name="pahr_grunow2025_spirits_0001",
        dataset_file="spirits_0001_perfect_information_lp.txt",
        num_ages=10,
        num_products=3,
        target_ages=[2, 4, 6],
        max_inventory=50.0,
        evaporation=0.03,
        holding_cost=2.5,
        allow_blending=False,
        published_max_reward=1991.9344293376805,
        notes=(
            "Companion default spirits instance. published_max_reward is the "
            "perfect-information LP average-profit UPPER BOUND from "
            "problem_configurations/spirits_0001/upper_bound.json (a bound, NOT an "
            "achievable optimum)."
        ),
    ),
    "pahr_grunow2025_port_wine": dict(
        name="pahr_grunow2025_port_wine",
        lp_reference_name="pahr_grunow2025_port_wine",
        dataset_file="port_wine_perfect_information_lp.txt",
        num_ages=25,
        num_products=2,
        target_ages=[9, 19],
        max_inventory=50.0,
        evaporation=0.02,
        holding_cost=1.0,
        allow_blending=True,
        published_max_reward=2444.8010643781136,
        notes=(
            "Port-wine industry case study. published_max_reward is the "
            "perfect-information LP average-profit UPPER BOUND from "
            "problem_configurations/port_wine/upper_bound.json (a bound, NOT an "
            "achievable optimum)."
        ),
    ),
}

# references.rs::PRIMARY_REFERENCE_INSTANCE
_PRIMARY = "pahr_grunow2025_spirits_0001"

_BOUND_NOTE = (
    "perfect-information LP UPPER BOUND on long-run average profit "
    "(known-future relaxation; a bound, NOT an achievable optimum)"
)


class AmelioratingInventoryRunner(ProblemRunner):
    """Runnable baseline driver for the Pahr & Grunow 2025 ameliorating-inventory family."""

    problem = "ameliorating_inventory"
    # The perfect-information LP bound is an exact solve (seed/horizon-independent);
    # the protocols matter only for a LEARNED policy's seed-robust headline (>=5).
    published_protocol = EvalProtocol(
        seeds=(1234, 2025, 7, 99, 1000), horizon=2000, warm_up_periods_ratio=0.2
    )
    smoke_protocol = EvalProtocol(seeds=(1234,), horizon=1000, warm_up_periods_ratio=0.2)
    supports_evaluate = False
    # PROFIT family: maximize. Keeps reference_baseline = highest and
    # compare()'s "positive gap = worse" correct.
    lower_is_better = False

    def __init__(self) -> None:
        import invman_rust

        self._rust = invman_rust

    # -- listing ----------------------------------------------------------
    def list_instances(self) -> list[str]:
        return list(_REFERENCE_INSTANCES.keys())

    def primary_instance(self) -> str:
        return _PRIMARY

    def _subfamily_of(self, name: str) -> str:
        return "pahr_grunow2025_average_profit_blending"

    # -- reference dict (built from references.rs constants) --------------
    def _reference_dict(self, name: str) -> dict:
        if name not in _REFERENCE_INSTANCES:
            raise KeyError(
                f"unknown ameliorating_inventory instance: {name!r}. "
                f"Known: {self.list_instances()}"
            )
        d = dict(_REFERENCE_INSTANCES[name])
        d["source"] = _SOURCE
        d["url"] = _REPO_URL
        return d

    # -- published (free) baselines: the LP upper bound on profit ---------
    def _published_baselines(self, name: str, inst_dict: dict) -> list[Baseline]:
        bound = inst_dict.get("published_max_reward")
        if bound is None:
            return []
        return [
            Baseline(
                name="perfect_information_lp_bound",
                mean_cost=float(bound),  # "cost" slot holds the average-profit bound
                source=_SOURCE,
                is_published=True,
                # A BOUND, not the achievable optimum.
                is_optimal=False,
                is_reference=True,
                note=_BOUND_NOTE
                + " — companion upper_bound.json (reproduced to <1e-3 in-crate)",
            )
        ]

    # -- run the env (the "runnable" proof): re-solve the LP bound --------
    def _run_baselines(
        self, inst: ReferenceInstance, protocol: EvalProtocol
    ) -> dict[str, Baseline]:
        p = inst.params
        try:
            summary = dict(
                self._rust.ameliorating_inventory_perfect_info_lp_bound_summary(
                    reference_name=str(p["lp_reference_name"])
                )
            )
            return {
                "perfect_information_lp_bound": Baseline(
                    name="perfect_information_lp_bound",
                    mean_cost=float(summary["upper_bound_max_reward"]),
                    source="recomputed:ameliorating_inventory_perfect_info_lp_bound_summary",
                    params={
                        "purchasing": float(summary.get("upper_bound_purchasing", float("nan"))),
                    },
                    is_optimal=False,
                    is_reference=True,
                    note=(
                        _BOUND_NOTE
                        + f" — re-solved; published {summary.get('published_max_reward')}, "
                        f"gap {summary.get('max_reward_gap_to_published')}"
                    ),
                )
            }
        except Exception as exc:  # None-safe: never abort the sweep
            return {
                "perfect_information_lp_bound": Baseline(
                    name="perfect_information_lp_bound",
                    mean_cost=None,
                    source=f"recomputed_failed:{type(exc).__name__}",
                )
            }
