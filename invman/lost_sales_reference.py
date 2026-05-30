"""
Heuristic baseline lookup for lost-sales experiment results.

After the Python-cleanup migration, the heuristic reference costs live in Rust
(crate module ``problems::lost_sales::reference_costs``) and are exposed via
``invman_rust.lost_sales_reference_costs``. This thin wrapper maps an experiment
``args`` object onto the matching benchmark-grid instance and returns the
heuristic baselines in the result-payload shape used by ``experiment_runner``.

Only the vanilla lost-sales problem (``problem == "lost_sales"`` with no fixed
order cost) is covered; the fixed-order-cost problem uses a different heuristic
family ((s,S), (s,nQ), modified (s,S,q)) and is handled elsewhere.
"""

from __future__ import annotations

import invman_rust

_DEMAND_TOKEN = {"Poisson": "poisson", "Geometric": "geometric"}


def reference_instance_name(args) -> str | None:
    """Canonical benchmark-grid instance name for `args`, or None if unmatched."""
    if str(getattr(args, "problem", "lost_sales")) != "lost_sales":
        return None
    if float(getattr(args, "fixed_order_cost", 0.0) or 0.0) != 0.0:
        return None
    demand = str(getattr(args, "demand_dist_name", ""))
    lead_time = int(getattr(args, "lead_time", 0))
    shortage_cost = int(round(float(getattr(args, "shortage_cost", 0.0))))
    if demand in _DEMAND_TOKEN:
        token = _DEMAND_TOKEN[demand]
    elif demand == "MarkovModulatedPoisson2":
        # positive vs negative autocorrelation is set by the transition probs
        token = "mmpp2_pos" if float(getattr(args, "demand_p00", 0.0)) >= 0.5 else "mmpp2_neg"
    else:
        return None
    return f"lit_{token}_p{shortage_cost}_l{lead_time}"


def heuristic_baselines_for(args) -> dict:
    """Heuristic baselines for `args` from the Rust reference-cost config.

    Returns a dict shaped like the old ``evaluation.heuristics`` block:
    ``{heuristic: {"mean_cost": float, "source": str}}`` for the available
    myopic1/myopic2/svbs costs plus optimal/capped-base-stock references. Empty
    when the instance is not in the config (e.g. fixed-cost or unknown demand).
    """
    name = reference_instance_name(args)
    if name is None:
        return {}
    reference = invman_rust.lost_sales_reference_costs(name)
    if reference is None:
        return {}
    costs = reference["costs"]
    source = f"reference_config:{reference['source']}"
    baselines: dict[str, dict] = {}
    for heuristic in ("myopic1", "myopic2", "svbs"):
        if costs.get(heuristic) is not None:
            baselines[heuristic] = {"mean_cost": float(costs[heuristic]), "source": source}
    for payload_key, cost_key in (
        ("optimal_reference", "optimal"),
        ("capped_base_stock_reference", "capped_base_stock"),
    ):
        if costs.get(cost_key) is not None:
            baselines[payload_key] = {"mean_cost": float(costs[cost_key]), "source": source}
    baselines["reference_instance"] = name
    return baselines
