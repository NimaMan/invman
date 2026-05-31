"""
Compute the lost-sales heuristic costs (Myopic-1, Myopic-2, SVBS) that are
missing from the literature config, using the Rust heuristic binding
(`invman_rust.lost_sales_heuristic_mean_cost`).

Objective
=========
Fill the heuristic performance numbers the config does not have:
  - Geometric instances (all lead times): myopic1, svbs  (myopic2 is in literature)
  - Poisson L10 instances:                myopic1, svbs  (myopic2 is in literature)
  - MMPP2 positive/negative instances:    myopic1, myopic2, svbs (no literature)

It also cross-validates the Rust evaluator against literature on a few IID
instances (where myopic2 is known) so the newly computed numbers can be trusted.

These numbers are repo-computed via the Rust IID/stationary-marginal heuristics,
NOT literature.

Streaming: each (instance, heuristic) result is appended as one JSON line to the
path given by $OUT_JSONL (default /tmp/missing_heuristics.jsonl) and flushed, so
partial progress is readable while the sweep runs. Instances are processed
cheapest-first (ascending lead time) so the costly L10 nested lookups come last.

Run:
  PYTHONPATH=/home/nima/code/ml/invman env -u VIRTUAL_ENV \
    /home/nima/miniconda3/bin/python \
    scripts/lost_sales/compute_missing_heuristics_via_rust.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import invman_rust as rust

from invman.problems.lost_sales.reference_instances import REFERENCE_INSTANCES

HORIZON = int(os.environ.get("HEUR_HORIZON", "100000"))
SEED = 123
WARMUP = 0.2
ORDER_SEARCH_UPPER_BOUND = 200
DISCOUNT = 0.995
OUT_JSONL = os.environ.get("OUT_JSONL", "/tmp/missing_heuristics.jsonl")

CROSSCHECK_NAMES = {
    "lit_poisson_p4_l4",
    "lit_geometric_p4_l4",
    "lit_geometric_p19_l4",
}


def _rust_args(params: dict) -> dict:
    return dict(
        demand_kind=str(params["demand_dist_name"]),
        demand_rate=float(params.get("demand_rate", 5.0)),
        demand_lambda_low=float(params.get("demand_lambda_low", 0.0)),
        demand_lambda_high=float(params.get("demand_lambda_high", 0.0)),
        demand_p00=float(params.get("demand_p00", 0.0)),
        demand_p11=float(params.get("demand_p11", 0.0)),
        lead_time=int(params["lead_time"]),
        holding_cost=float(params.get("holding_cost", 1.0)),
        shortage_cost=float(params["shortage_cost"]),
        procurement_cost=float(params.get("procurement_cost", 0.0)),
        fixed_order_cost=float(params.get("fixed_order_cost", 0.0)),
        horizon=HORIZON,
        seed=SEED,
        warm_up_periods_ratio=WARMUP,
        order_search_upper_bound=ORDER_SEARCH_UPPER_BOUND,
        heuristic_discount_factor=DISCOUNT,
    )


def main() -> int:
    out = open(OUT_JSONL, "w", buffering=1)  # line-buffered

    def emit(record: dict) -> None:
        out.write(json.dumps(record) + "\n")
        out.flush()
        sys.stderr.write(json.dumps(record) + "\n")
        sys.stderr.flush()

    instances = [(n, i) for n, i in REFERENCE_INSTANCES.items() if n != "vanilla_l4_p4_poisson5"]
    # cheapest first: ascending lead time, then name
    instances.sort(key=lambda ni: (int(ni[1].params["lead_time"]), ni[0]))

    for name, inst in instances:
        params = inst.params
        demand = str(params["demand_dist_name"])
        is_mmpp2 = demand == "MarkovModulatedPoisson2"
        exp = inst.expected_costs
        lead_time = int(params["lead_time"])
        shortage_cost = int(round(float(params["shortage_cost"])))

        todo = []  # (heuristic, role)
        for heur, lit_key in (("myopic1", "myopic1"), ("svbs", "svbs")):
            if exp.get(lit_key) is None:
                todo.append((heur, "fill"))
        if is_mmpp2:
            todo.append(("myopic2", "fill"))
        elif name in CROSSCHECK_NAMES:
            todo.append(("myopic2", "crosscheck"))

        for heur, role in todo:
            t = time.time()
            try:
                value = round(float(rust.lost_sales_heuristic_mean_cost(heuristic=heur, **_rust_args(params))), 5)
                err = None
            except Exception as exc:  # noqa: BLE001
                value, err = None, str(exc)
            emit({
                "instance": name, "demand": demand, "lead_time": lead_time,
                "shortage_cost": shortage_cost, "heuristic": heur, "role": role,
                "value": value, "literature": exp.get(heur), "error": err,
                "seconds": round(time.time() - t, 1),
            })

    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
