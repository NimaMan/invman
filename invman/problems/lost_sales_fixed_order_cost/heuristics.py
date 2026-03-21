from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np

from invman.env.lost_sales import build_env_from_args


@dataclass
class PolicySearchResult:
    policy_name: str
    params: dict[str, int]
    mean_cost: float
    search_seed: int
    search_horizon: int

    def to_dict(self):
        return {
            "policy_name": self.policy_name,
            "params": dict(self.params),
            "mean_cost": float(self.mean_cost),
            "search_seed": int(self.search_seed),
            "search_horizon": int(self.search_horizon),
        }


@dataclass
class PolicySearchSummary:
    best_result: PolicySearchResult
    top_results: list[PolicySearchResult]
    evaluated_candidates: int

    def to_dict(self):
        return {
            "best_result": self.best_result.to_dict(),
            "top_results": [result.to_dict() for result in self.top_results],
            "evaluated_candidates": int(self.evaluated_candidates),
        }


def get_review_period_demand_variance(demand_dist_name: str, demand_rate: float) -> float:
    if demand_dist_name == "Poisson":
        return float(demand_rate)
    if demand_dist_name == "Geometric":
        return float(demand_rate * (1.0 + demand_rate))
    raise NotImplementedError(f"Unsupported demand distribution: {demand_dist_name}")


def get_default_position_upper_bound(args) -> int:
    review_variance = get_review_period_demand_variance(args.demand_dist_name, args.demand_rate)
    protection_mean = (args.lead_time + 1) * args.demand_rate
    protection_std = sqrt((args.lead_time + 1) * review_variance)
    upper_bound = int(ceil(protection_mean + 4.0 * protection_std))
    return max(1, min(int(args.max_order_size), upper_bound))


def get_s_s_order_quantity(inventory_position: int, s: int, S: int, max_order_size: int) -> int:
    if inventory_position > s:
        return 0
    return min(max(0, int(S) - int(inventory_position)), int(max_order_size))


def get_s_nq_order_quantity(inventory_position: int, s: int, q: int, max_order_size: int) -> int:
    if q <= 0:
        raise ValueError("q must be positive")
    if inventory_position > s:
        return 0
    batches = int(ceil((int(s) + 1 - int(inventory_position)) / int(q)))
    return min(max(0, batches * int(q)), int(max_order_size))


def get_modified_s_s_q_order_quantity(
    inventory_position: int,
    s: int,
    S: int,
    q: int,
    max_order_size: int,
) -> int:
    if q <= 0:
        raise ValueError("q must be positive")
    if inventory_position > s:
        return 0
    return min(int(max_order_size), int(q), max(0, int(S) - int(inventory_position)))


def get_paper_q_heuristic(args, s: int, S: int) -> int:
    if S <= s:
        raise ValueError("S must be greater than s")

    demand_mean = float(args.demand_rate)
    demand_variance = get_review_period_demand_variance(args.demand_dist_name, args.demand_rate)
    average_undershoot = (demand_variance + demand_mean**2) / (2.0 * demand_mean)
    average_cycle_length = max(1.0, (S - s) / demand_mean + average_undershoot / demand_mean)
    q = int(round(S * average_cycle_length / (args.lead_time + average_cycle_length)))
    q = max(int(S - s), q)
    return min(int(args.max_order_size), q)


def _policy_action(policy_name: str, inventory_position: int, params: dict[str, int], max_order_size: int) -> int:
    if policy_name == "s_s":
        return get_s_s_order_quantity(
            inventory_position=inventory_position,
            s=params["s"],
            S=params["S"],
            max_order_size=max_order_size,
        )
    if policy_name == "s_nq":
        return get_s_nq_order_quantity(
            inventory_position=inventory_position,
            s=params["s"],
            q=params["q"],
            max_order_size=max_order_size,
        )
    if policy_name == "modified_s_s_q":
        return get_modified_s_s_q_order_quantity(
            inventory_position=inventory_position,
            s=params["s"],
            S=params["S"],
            q=params["q"],
            max_order_size=max_order_size,
        )
    raise NotImplementedError(f"Unknown policy '{policy_name}'")


def evaluate_policy_cost(
    args,
    policy_name: str,
    params: dict[str, int],
    seed: int = 1234,
    horizon: int | None = None,
    track_demand: bool = True,
):
    np.random.seed(seed)
    env = build_env_from_args(args, horizon=horizon, track_demand=track_demand)

    while not env.is_done():
        order_quantity = _policy_action(
            policy_name=policy_name,
            inventory_position=env.inventory_position,
            params=params,
            max_order_size=env.max_order_size,
        )
        env.step(order_quantity=order_quantity)

    return env.avg_total_cost


def evaluate_policy_across_seeds(
    args,
    policy_name: str,
    params: dict[str, int],
    num_seeds: int = 3,
    seed: int | None = None,
    horizon: int | None = None,
    track_demand: bool = True,
):
    base_seed = getattr(args, "seed", 1234) if seed is None else seed
    costs = []
    for seed_offset in range(num_seeds):
        costs.append(
            evaluate_policy_cost(
                args=args,
                policy_name=policy_name,
                params=params,
                seed=base_seed + seed_offset,
                horizon=horizon,
                track_demand=track_demand,
            )
        )

    return {
        "params": dict(params),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "num_seeds": int(num_seeds),
    }


def _search_candidates(
    args,
    policy_name: str,
    candidates: list[dict[str, int]],
    seed: int,
    horizon: int,
    top_k: int = 10,
):
    results = []
    for params in candidates:
        mean_cost = evaluate_policy_cost(
            args=args,
            policy_name=policy_name,
            params=params,
            seed=seed,
            horizon=horizon,
            track_demand=True,
        )
        results.append(
            PolicySearchResult(
                policy_name=policy_name,
                params=dict(params),
                mean_cost=float(mean_cost),
                search_seed=seed,
                search_horizon=horizon,
            )
        )

    results.sort(key=lambda result: result.mean_cost)
    return PolicySearchSummary(
        best_result=results[0],
        top_results=results[:top_k],
        evaluated_candidates=len(results),
    )


def _get_search_upper_bound(args, position_upper_bound: int | None) -> int:
    if position_upper_bound is not None:
        return int(position_upper_bound)
    return get_default_position_upper_bound(args)


def search_best_s_s_policy(
    args,
    seed: int | None = None,
    horizon: int | None = None,
    position_upper_bound: int | None = None,
    top_k: int = 12,
):
    search_seed = getattr(args, "seed", 1234) if seed is None else int(seed)
    search_horizon = args.horizon if horizon is None else int(horizon)
    upper_bound = _get_search_upper_bound(args, position_upper_bound)

    candidates = []
    for s in range(upper_bound):
        for S in range(s + 1, upper_bound + 1):
            candidates.append({"s": s, "S": S})

    return _search_candidates(
        args=args,
        policy_name="s_s",
        candidates=candidates,
        seed=search_seed,
        horizon=search_horizon,
        top_k=top_k,
    )


def search_best_s_nq_policy(
    args,
    seed: int | None = None,
    horizon: int | None = None,
    position_upper_bound: int | None = None,
    top_k: int = 12,
):
    search_seed = getattr(args, "seed", 1234) if seed is None else int(seed)
    search_horizon = args.horizon if horizon is None else int(horizon)
    upper_bound = _get_search_upper_bound(args, position_upper_bound)

    candidates = []
    for s in range(upper_bound):
        for q in range(1, upper_bound + 1):
            candidates.append({"s": s, "q": q})

    return _search_candidates(
        args=args,
        policy_name="s_nq",
        candidates=candidates,
        seed=search_seed,
        horizon=search_horizon,
        top_k=top_k,
    )


def _iter_modified_candidates(args, s_s_summary: PolicySearchSummary, q_window: int):
    seen = set()
    upper_bound = int(args.max_order_size)

    for result in s_s_summary.top_results:
        base_s = result.params["s"]
        base_S = result.params["S"]
        for delta_s in (-1, 0, 1):
            for delta_S in (-1, 0, 1):
                s = base_s + delta_s
                S = base_S + delta_S
                if s < 0 or S <= s or S > upper_bound:
                    continue

                q_center = get_paper_q_heuristic(args=args, s=s, S=S)
                q_candidates = {
                    max(1, S - s),
                    min(upper_bound, q_center),
                    min(upper_bound, S),
                    upper_bound,
                }
                q_min = max(1, q_center - q_window)
                q_max = min(upper_bound, q_center + q_window)
                for q in range(q_min, q_max + 1):
                    q_candidates.add(q)

                for q in sorted(q_candidates):
                    key = (s, S, q)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {"s": s, "S": S, "q": q}


def search_best_modified_s_s_q_policy(
    args,
    seed: int | None = None,
    horizon: int | None = None,
    position_upper_bound: int | None = None,
    top_k_s_s_pairs: int = 12,
    q_window: int = 8,
    s_s_summary: PolicySearchSummary | None = None,
):
    if s_s_summary is None:
        s_s_summary = search_best_s_s_policy(
            args=args,
            seed=seed,
            horizon=horizon,
            position_upper_bound=position_upper_bound,
            top_k=top_k_s_s_pairs,
        )
    search_seed = getattr(args, "seed", 1234) if seed is None else int(seed)
    search_horizon = args.horizon if horizon is None else int(horizon)

    candidates = list(_iter_modified_candidates(args=args, s_s_summary=s_s_summary, q_window=q_window))
    summary = _search_candidates(
        args=args,
        policy_name="modified_s_s_q",
        candidates=candidates,
        seed=search_seed,
        horizon=search_horizon,
        top_k=min(10, len(candidates)),
    )
    return {
        "search_basis": s_s_summary,
        "modified_policy": summary,
    }
