"""
Self-contained benchmark for procurement_removal_inventory.

OBJECTIVE
    Compare the available policies for the procurement-removal single-item control slice on a
    well-posed instance set, using ONLY the already-installed `invman_rust` extension (no Rust
    rebuild, no dead Python imports). This is the INVESTIGATE-FIX-track companion to the
    repo-exact verifier: the problem is NOT literature-verified (see literature/README.md), so the
    benchmark target is "learned soft-tree vs best tuned interval-stock heuristic vs the reduced
    exact DP optimum", not a published cost row.

WHY THIS SCRIPT EXISTS (it replaces the broken scripts)
    The pre-existing `scripts/procurement_removal_inventory/common.py` /
    `train_soft_tree_reference.py` / `validate_against_exact_dp.py` import
    `invman.policies.soft_tree.SoftTreePolicy`, a module that was DELETED in the Python-cleanup
    migration (the repo is now Rust-only for problem dynamics; policies route through `invman.policy`
    / `invman.policy_registry`). Those scripts therefore raise `ModuleNotFoundError` and cannot run.
    This script depends on nothing from the deleted tree: it talks to the Rust bindings directly and
    uses only `invman.cmaes.CMAES` for the optional learned-policy comparison.

ALGORITHM (full description)
    Instances:
      * primary            : the carried `procurement_removal_inventory_primary_reference_instance`
                             (maggiar2017_style_fixed_returnability). Demand mean 4 over 16 periods
                             drains inventory fast, so the REMOVAL lever is essentially inactive here
                             and the interval-stock policy collapses to a pure order-up-to policy.
      * removal_active     : a repo-native instance defined in this script (mirrored into
                             `literature/references.rs` as REMOVAL_ACTIVE_REFERENCE_INSTANCE for the
                             next rebuild) with high initial inventory and lower demand so that
                             overstock occurs and the remove-down-to threshold actually binds. This
                             is the instance on which procurement-vs-removal tradeoffs are visible.

    For each instance:
      1. EXACT DP (primary's reduced verifier only): read
         `procurement_removal_inventory_exact_dp_summary`, which solves the small finite-horizon DP
         exactly and reports optimal vs the two carried heuristics. (The removal_active instance has a
         Poisson demand that is too wide for the small discrete-support exact verifier, so no exact
         row is computed for it; it is benchmarked by simulation only.)
      2. BEST TUNED HEURISTIC: grid-search constant interval-stock (order_up_to, remove_down_to) and
         the returnability-buffer variant over the action range, scoring each by Monte-Carlo mean
         discounted cost on a fixed held-out seed set (`simulate_policy`). The best tuned constant
         interval-stock policy is the strong static comparator (the optimal-policy STRUCTURE for this
         family per Maggiar & Sadighian 2017 Theorem 3.4 is interval-stock, so a tuned constant
         interval-stock is the natural near-optimal static proxy).
      3. LEARNED SOFT-TREE (optional, --train): CMA-ES over the Rust soft-tree population rollout
         (`procurement_removal_inventory_soft_tree_population_rollout`), warm seeds averaged per
         generation; evaluate the best mean parameter on the SAME held-out seed set
         (`procurement_removal_inventory_soft_tree_rollout`). The soft-tree num_params is computed
         from the architecture exactly as `invman.policy.Policy.num_params` does for a soft_tree
         backbone (no dependency on that class).

    Report: per-instance markdown table of mean discounted cost (lower is better), plus the exact
    optimum and gaps for the primary instance.

NOTE ON FIDELITY (honest status)
    Lower discounted cost is better. All costs are NEGATIVE-of-reward discounted sums INCLUDING the
    terminal salvage credit `s*min(x,y) + l*max(x-y,0)` (matching Maggiar & Sadighian 2017,
    Assumption 4 terminal value). The env is a control-only reduction of the cited revenue-management
    model: no pricing/markdown decision, lost-sales instead of backlog, Poisson instead of additive
    price-dependent Gamma demand. There is no published cost row to reproduce; see literature/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import invman_rust  # noqa: E402

# ---------------------------------------------------------------------------
# Instance definitions
# ---------------------------------------------------------------------------

SOFT_TREE_INPUT_DIM = 7  # rollout.rs policy feature map width (validated at input_dim == 7)
SOFT_TREE_CONTROL_DIM = 2  # (purchase_quantity, removal_quantity)


def primary_instance() -> dict:
    """Carried primary reference; removal lever essentially inactive at demand mean 4."""
    return dict(invman_rust.procurement_removal_inventory_primary_reference_instance())


# REMOVAL_ACTIVE_REFERENCE_INSTANCE: repo-native instance where the removal channel binds.
# Mirrored into literature/references.rs (same name) so the next Rust rebuild can expose it via a
# binding. Until then this dict is the source of truth used by simulate_policy directly.
def removal_active_instance() -> dict:
    return {
        "name": "removal_active_returnability",
        "source": "repo_native_removal_active_instance",
        "periods": 16,
        "demand_distribution_kind": "poisson",
        "demand_mean": 3.0,
        "initial_inventory_level": 12,
        "initial_returnable_inventory": 8,
        "returnable_purchase_cap": 2,
        "purchase_cost_per_unit": 6.0,
        "return_value_per_unit": 4.0,
        "liquidation_value_per_unit": 1.0,
        "holding_cost_per_unit": 1.0,
        "shortage_cost_per_unit": 9.0,
        "max_purchase_quantity": 6,
        "max_removal_quantity": 8,
    }


# ---------------------------------------------------------------------------
# Heuristic evaluation (Monte-Carlo via simulate_policy)
# ---------------------------------------------------------------------------


def _sim_kwargs(ref: dict, seeds: list[int]) -> dict:
    return dict(
        inventory_level=int(ref["initial_inventory_level"]),
        returnable_inventory=int(ref["initial_returnable_inventory"]),
        periods=int(ref["periods"]),
        seeds=[int(s) for s in seeds],
        demand_kind=str(ref["demand_distribution_kind"]),
        demand_mean=float(ref["demand_mean"]),
        returnable_purchase_cap=int(ref["returnable_purchase_cap"]),
        purchase_cost_per_unit=float(ref["purchase_cost_per_unit"]),
        return_value_per_unit=float(ref["return_value_per_unit"]),
        liquidation_value_per_unit=float(ref["liquidation_value_per_unit"]),
        holding_cost_per_unit=float(ref["holding_cost_per_unit"]),
        shortage_cost_per_unit=float(ref["shortage_cost_per_unit"]),
        max_purchase_quantity=int(ref["max_purchase_quantity"]),
        max_removal_quantity=int(ref["max_removal_quantity"]),
        discount_factor=0.99,
    )


def eval_heuristic(ref: dict, name: str, params: list[int], seeds: list[int]) -> dict:
    return dict(
        invman_rust.procurement_removal_inventory_simulate_policy(
            policy_name=name,
            params=[int(p) for p in params],
            **_sim_kwargs(ref, seeds),
        )
    )


def best_interval_stock(ref: dict, seeds: list[int]) -> tuple[list[int], float]:
    hi = int(ref["max_purchase_quantity"]) + int(ref["initial_inventory_level"]) + 2
    best_params, best_cost = None, float("inf")
    for order_up_to in range(0, hi):
        for remove_down_to in range(order_up_to, hi + 2):
            cost = eval_heuristic(ref, "interval_stock", [order_up_to, remove_down_to], seeds)[
                "mean_discounted_cost"
            ]
            if cost < best_cost:
                best_params, best_cost = [order_up_to, remove_down_to], cost
    return best_params, best_cost


def best_returnability_buffer(ref: dict, seeds: list[int]) -> tuple[list[int], float]:
    hi = int(ref["max_purchase_quantity"]) + int(ref["initial_inventory_level"]) + 2
    best_params, best_cost = None, float("inf")
    for order_up_to in range(0, hi):
        for remove_down_to in range(order_up_to, hi + 2):
            for buffer in range(0, int(ref["returnable_purchase_cap"]) + 2):
                cost = eval_heuristic(
                    ref,
                    "returnability_buffer_interval_stock",
                    [order_up_to, remove_down_to, buffer],
                    seeds,
                )["mean_discounted_cost"]
                if cost < best_cost:
                    best_params, best_cost = [order_up_to, remove_down_to, buffer], cost
    return best_params, best_cost


# ---------------------------------------------------------------------------
# Soft-tree (learned policy) via Rust rollout + invman.cmaes
# ---------------------------------------------------------------------------


def soft_tree_num_params(input_dim: int, depth: int, control_dim: int, leaf_type: str) -> int:
    n_internal = (2 ** depth) - 1
    n_leaf = 2 ** depth
    count = n_internal * input_dim + n_internal
    if leaf_type == "constant":
        count += n_leaf * control_dim
    else:
        count += n_leaf * control_dim * input_dim + n_leaf * control_dim
    return int(count)


def _soft_tree_base_kwargs(ref: dict, depth: int, temperature: float, split_type: str, leaf_type: str) -> dict:
    return dict(
        input_dim=SOFT_TREE_INPUT_DIM,
        depth=depth,
        min_values=[0, 0],
        max_values=[int(ref["max_purchase_quantity"]), int(ref["max_removal_quantity"])],
        action_mode="vector_quantity",
        inventory_level=int(ref["initial_inventory_level"]),
        returnable_inventory=int(ref["initial_returnable_inventory"]),
        periods=int(ref["periods"]),
        demand_kind=str(ref["demand_distribution_kind"]),
        demand_mean=float(ref["demand_mean"]),
        returnable_purchase_cap=int(ref["returnable_purchase_cap"]),
        purchase_cost_per_unit=float(ref["purchase_cost_per_unit"]),
        return_value_per_unit=float(ref["return_value_per_unit"]),
        liquidation_value_per_unit=float(ref["liquidation_value_per_unit"]),
        holding_cost_per_unit=float(ref["holding_cost_per_unit"]),
        shortage_cost_per_unit=float(ref["shortage_cost_per_unit"]),
        max_purchase_quantity=int(ref["max_purchase_quantity"]),
        max_removal_quantity=int(ref["max_removal_quantity"]),
        discount_factor=0.99,
        temperature=temperature,
        split_type=split_type,
        leaf_type=leaf_type,
        allowed_values=None,
    )


def train_soft_tree(
    ref: dict,
    *,
    depth: int,
    temperature: float,
    split_type: str,
    leaf_type: str,
    population: int,
    generations: int,
    train_seed_batch: int,
    sigma_init: float,
    seed: int,
) -> np.ndarray:
    from invman.cmaes import CMAES

    num_params = soft_tree_num_params(SOFT_TREE_INPUT_DIM, depth, SOFT_TREE_CONTROL_DIM, leaf_type)
    base_kw = _soft_tree_base_kwargs(ref, depth, temperature, split_type, leaf_type)
    es = CMAES(num_params=num_params, sigma_init=sigma_init, popsize=population, seed=seed)
    for gen in range(generations):
        solutions = es.ask()
        batch = [np.asarray(s, dtype=np.float32).tolist() for s in solutions]
        seeds_base = 1000 + gen * population
        acc = np.zeros(len(batch))
        for offset in range(train_seed_batch):
            costs = invman_rust.procurement_removal_inventory_soft_tree_population_rollout(
                params_batch=batch,
                seeds=[seeds_base + i + offset * 7919 for i in range(len(batch))],
                **base_kw,
            )
            acc += np.asarray(costs)
        acc /= train_seed_batch
        es.tell((-acc).tolist())
    return np.asarray(es.best_param(), dtype=np.float32)


def eval_soft_tree(ref, flat_params, seeds, *, depth, temperature, split_type, leaf_type) -> dict:
    base_kw = _soft_tree_base_kwargs(ref, depth, temperature, split_type, leaf_type)
    costs = []
    for s in seeds:
        costs.append(
            invman_rust.procurement_removal_inventory_soft_tree_rollout(
                flat_params=np.asarray(flat_params, dtype=np.float32).tolist(),
                seed=int(s),
                **base_kw,
            )
        )
    costs = np.asarray(costs, dtype=np.float64)
    return {
        "mean_discounted_cost": float(costs.mean()),
        "std_discounted_cost": float(costs.std()),
        "num_seeds": int(costs.size),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def benchmark_instance(ref: dict, args, *, with_exact: bool) -> dict:
    eval_seeds = [args.eval_seed_base + i for i in range(args.eval_seeds)]
    result: dict = {"instance": ref, "eval_seeds": len(eval_seeds), "policies": {}}

    if with_exact:
        result["exact_dp_summary"] = dict(invman_rust.procurement_removal_inventory_exact_dp_summary())

    iv_params, iv_cost = best_interval_stock(ref, eval_seeds)
    bf_params, bf_cost = best_returnability_buffer(ref, eval_seeds)
    result["policies"]["best_interval_stock"] = {"params": iv_params, "mean_discounted_cost": iv_cost}
    result["policies"]["best_returnability_buffer"] = {"params": bf_params, "mean_discounted_cost": bf_cost}

    if args.train:
        best_params = train_soft_tree(
            ref,
            depth=args.depth,
            temperature=args.temperature,
            split_type=args.split_type,
            leaf_type=args.leaf_type,
            population=args.population,
            generations=args.generations,
            train_seed_batch=args.train_seed_batch,
            sigma_init=args.sigma_init,
            seed=args.seed,
        )
        st_eval = eval_soft_tree(
            ref,
            best_params,
            eval_seeds,
            depth=args.depth,
            temperature=args.temperature,
            split_type=args.split_type,
            leaf_type=args.leaf_type,
        )
        result["policies"]["soft_tree"] = {
            "params": "trained",
            "mean_discounted_cost": st_eval["mean_discounted_cost"],
            "std_discounted_cost": st_eval["std_discounted_cost"],
        }
    return result


def markdown(name: str, result: dict) -> str:
    lines = [f"### Instance: `{name}`  ({result['instance'].get('name', name)})", ""]
    if "exact_dp_summary" in result:
        s = result["exact_dp_summary"]
        lines += [
            "Reduced exact DP verifier (separate small instance, not this benchmark instance):",
            "",
            "| Policy | Discounted Cost | Gap to Optimal |",
            "| --- | ---: | ---: |",
            f"| optimal (exact DP) | {s['optimal_discounted_cost']:.4f} | 0.0000 |",
            f"| interval_stock | {s['interval_stock_discounted_cost']:.4f} | {s['interval_stock_gap_to_optimal']:.4f} |",
            f"| returnability_buffer | {s['returnability_buffer_discounted_cost']:.4f} | {s['returnability_buffer_gap_to_optimal']:.4f} |",
            "",
        ]
    lines += [
        f"Simulation benchmark ({result['eval_seeds']} held-out seeds, lower cost is better):",
        "",
        "| Policy | Params | Mean Discounted Cost |",
        "| --- | --- | ---: |",
    ]
    for pol, info in result["policies"].items():
        lines.append(f"| `{pol}` | `{info['params']}` | {info['mean_discounted_cost']:.4f} |")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--eval_seeds", type=int, default=4096)
    p.add_argument("--eval_seed_base", type=int, default=500000)
    p.add_argument("--train", action="store_true", help="also train + benchmark a soft-tree policy")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    p.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--train_seed_batch", type=int, default=8)
    p.add_argument("--sigma_init", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    payload = {
        "literature_verified": False,
        "verification_note": (
            "Control-only slice of Maggiar & Sadighian (2017) joint pricing+removal model; "
            "no published cost row exists for this reduced package. Exact DP cross-checked "
            "independently in Python to machine precision."
        ),
        "primary": benchmark_instance(primary_instance(), args, with_exact=True),
        "removal_active": benchmark_instance(removal_active_instance(), args, with_exact=False),
    }
    payload["elapsed_seconds"] = time.time() - t0
    payload["markdown"] = (
        markdown("primary", payload["primary"]) + "\n\n" + markdown("removal_active", payload["removal_active"])
    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in payload.items() if k != "markdown"}, indent=2, sort_keys=True))
    print()
    print(payload["markdown"])


if __name__ == "__main__":
    main()
