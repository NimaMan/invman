from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np

from invman.problems.lost_sales.demand import (
    DEFAULT_MMPP2_LAMBDA_HIGH,
    DEFAULT_MMPP2_LAMBDA_LOW,
    DEFAULT_MMPP2_POSITIVE_P00,
    DEFAULT_MMPP2_POSITIVE_P11,
    build_demand_config,
    build_demand_config_from_args,
)
from invman.problems.lost_sales_fixed_order_cost.env import build_env_from_args


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


@dataclass(frozen=True)
class FixedDemandPath:
    current_inventory: int
    lead_time_orders: tuple[int, ...]
    demands: tuple[int, ...]
    horizon: int
    seed: int

    def to_dict(self):
        return {
            "current_inventory": int(self.current_inventory),
            "lead_time_orders": list(self.lead_time_orders),
            "demands": list(self.demands),
            "horizon": int(self.horizon),
            "seed": int(self.seed),
        }


def get_review_period_demand_variance(
    demand_dist_name: str,
    demand_rate: float,
    *,
    demand_lambda_low: float = DEFAULT_MMPP2_LAMBDA_LOW,
    demand_lambda_high: float = DEFAULT_MMPP2_LAMBDA_HIGH,
    demand_p00: float = DEFAULT_MMPP2_POSITIVE_P00,
    demand_p11: float = DEFAULT_MMPP2_POSITIVE_P11,
) -> float:
    config = build_demand_config(
        demand_dist_name=demand_dist_name,
        demand_rate=demand_rate,
        demand_lambda_low=demand_lambda_low,
        demand_lambda_high=demand_lambda_high,
        demand_p00=demand_p00,
        demand_p11=demand_p11,
    )
    return float(config.one_period_variance)


def get_protection_period_demand_variance(args, periods: int) -> float:
    config = build_demand_config_from_args(args)
    return float(config.cumulative_variance(periods))


def get_default_position_upper_bound(args) -> int:
    protection_mean = (args.lead_time + 1) * args.demand_rate
    protection_variance = get_protection_period_demand_variance(args, args.lead_time + 1)
    protection_std = sqrt(protection_variance)
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
    demand_variance = build_demand_config_from_args(args).one_period_variance
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


def build_fixed_demand_path(
    args,
    seed: int = 1234,
    horizon: int | None = None,
):
    path_horizon = args.horizon if horizon is None else int(horizon)
    np.random.seed(int(seed))
    env = build_env_from_args(args, horizon=path_horizon, track_demand=True)
    return FixedDemandPath(
        current_inventory=int(env.current_inventory_level),
        lead_time_orders=tuple(int(order) for order in env.lead_time_orders),
        demands=tuple(int(demand) for demand in env.horizon_demand.tolist()),
        horizon=path_horizon,
        seed=int(seed),
    )


def evaluate_policy_cost_on_path(
    args,
    policy_name: str,
    params: dict[str, int],
    fixed_path: FixedDemandPath,
):
    current_inventory = int(fixed_path.current_inventory)
    lead_time_orders = list(fixed_path.lead_time_orders)
    epoch_costs = []

    for demand in fixed_path.demands:
        inventory_position = current_inventory + sum(lead_time_orders)
        order_quantity = _policy_action(
            policy_name=policy_name,
            inventory_position=inventory_position,
            params=params,
            max_order_size=int(args.max_order_size),
        )
        arriving_order = int(lead_time_orders.pop(0))
        lead_time_orders.append(int(order_quantity))
        current_inventory += arriving_order

        epoch_cost = float(getattr(args, "procurement_cost", 0.0)) * int(order_quantity)
        if order_quantity > 0:
            epoch_cost += float(getattr(args, "fixed_order_cost", 0.0))

        if int(demand) < current_inventory:
            current_inventory -= int(demand)
            epoch_cost += current_inventory * float(args.holding_cost)
        else:
            lost_sales = int(demand) - current_inventory
            epoch_cost += float(args.shortage_cost) * lost_sales
            current_inventory = 0

        epoch_costs.append(epoch_cost)

    warm_up_ratio = float(getattr(args, "warm_up_periods_ratio", 0.2))
    warm_up_periods = min(len(epoch_costs), int(warm_up_ratio * len(epoch_costs)))
    active_costs = epoch_costs[warm_up_periods:] if warm_up_periods < len(epoch_costs) else epoch_costs
    return float(np.mean(active_costs))


def evaluate_policy_cost(
    args,
    policy_name: str,
    params: dict[str, int],
    seed: int = 1234,
    horizon: int | None = None,
    track_demand: bool = True,
    fixed_path: FixedDemandPath | None = None,
):
    if fixed_path is not None:
        return evaluate_policy_cost_on_path(
            args=args,
            policy_name=policy_name,
            params=params,
            fixed_path=fixed_path,
        )

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


def _build_candidate_result(policy_name: str, params: dict[str, int], mean_cost: float, seed: int, horizon: int):
    return PolicySearchResult(
        policy_name=policy_name,
        params=dict(params),
        mean_cost=float(mean_cost),
        search_seed=int(seed),
        search_horizon=int(horizon),
    )


def _sorted_summary(results: list[PolicySearchResult], top_k: int):
    results.sort(key=lambda result: result.mean_cost)
    return PolicySearchSummary(
        best_result=results[0],
        top_results=results[:top_k],
        evaluated_candidates=len(results),
    )


def _search_candidates(
    args,
    policy_name: str,
    candidates: list[dict[str, int]],
    seed: int,
    horizon: int,
    top_k: int = 10,
    backend: str = "python",
    position_upper_bound: int | None = None,
):
    fixed_path = build_fixed_demand_path(args=args, seed=seed, horizon=horizon)
    upper_bound = _get_search_upper_bound(args, position_upper_bound)

    if backend == "python":
        results = []
        for params in candidates:
            mean_cost = evaluate_policy_cost(
                args=args,
                policy_name=policy_name,
                params=params,
                seed=seed,
                horizon=horizon,
                track_demand=True,
                fixed_path=fixed_path,
            )
            results.append(
                _build_candidate_result(
                    policy_name=policy_name,
                    params=params,
                    mean_cost=mean_cost,
                    seed=seed,
                    horizon=horizon,
                )
            )
        return _sorted_summary(results, top_k=top_k)

    if backend == "rust":
        try:
            import invman_rust
        except ImportError as exc:  # pragma: no cover - exercised in integration tests
            raise ImportError("Rust backend requested for heuristic search, but invman_rust is unavailable.") from exc

        if policy_name == "s_s":
            best, top_results = invman_rust.lost_sales_fixed_s_s_search_from_demands(
                current_inventory=fixed_path.current_inventory,
                lead_time_orders=list(fixed_path.lead_time_orders),
                demands=list(fixed_path.demands),
                max_order_size=int(args.max_order_size),
                position_upper_bound=upper_bound,
                holding_cost=float(args.holding_cost),
                shortage_cost=float(args.shortage_cost),
                procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
                fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
                warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
                top_k=int(top_k),
            )
            results = [
                _build_candidate_result("s_s", {"s": int(s), "S": int(S)}, float(mean_cost), seed, horizon)
                for s, S, mean_cost in top_results
            ]
            return PolicySearchSummary(
                best_result=_build_candidate_result(
                    "s_s",
                    {"s": int(best[0]), "S": int(best[1])},
                    float(best[2]),
                    seed,
                    horizon,
                ),
                top_results=results,
                evaluated_candidates=int(upper_bound * (upper_bound + 1) // 2),
            )
        if policy_name == "s_nq":
            best, top_results = invman_rust.lost_sales_fixed_s_nq_search_from_demands(
                current_inventory=fixed_path.current_inventory,
                lead_time_orders=list(fixed_path.lead_time_orders),
                demands=list(fixed_path.demands),
                max_order_size=int(args.max_order_size),
                position_upper_bound=upper_bound,
                holding_cost=float(args.holding_cost),
                shortage_cost=float(args.shortage_cost),
                procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
                fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
                warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
                top_k=int(top_k),
            )
            results = [
                _build_candidate_result("s_nq", {"s": int(s), "q": int(q)}, float(mean_cost), seed, horizon)
                for s, q, mean_cost in top_results
            ]
            return PolicySearchSummary(
                best_result=_build_candidate_result(
                    "s_nq",
                    {"s": int(best[0]), "q": int(best[1])},
                    float(best[2]),
                    seed,
                    horizon,
                ),
                top_results=results,
                evaluated_candidates=int(upper_bound * upper_bound),
            )
        if policy_name == "modified_s_s_q":
            best, top_results, evaluated_candidates = invman_rust.lost_sales_fixed_modified_s_s_q_search_from_demands(
                current_inventory=fixed_path.current_inventory,
                lead_time_orders=list(fixed_path.lead_time_orders),
                demands=list(fixed_path.demands),
                max_order_size=int(args.max_order_size),
                position_upper_bound=upper_bound,
                holding_cost=float(args.holding_cost),
                shortage_cost=float(args.shortage_cost),
                procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
                fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
                warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
                top_k=int(top_k),
            )
            results = [
                _build_candidate_result(
                    "modified_s_s_q",
                    {"s": int(s), "S": int(S), "q": int(q)},
                    float(mean_cost),
                    seed,
                    horizon,
                )
                for s, S, q, mean_cost in top_results
            ]
            return PolicySearchSummary(
                best_result=_build_candidate_result(
                    "modified_s_s_q",
                    {"s": int(best[0]), "S": int(best[1]), "q": int(best[2])},
                    float(best[3]),
                    seed,
                    horizon,
                ),
                top_results=results,
                evaluated_candidates=int(evaluated_candidates),
            )
        raise NotImplementedError(f"Rust backend does not support policy '{policy_name}'")

    raise ValueError(f"Unknown heuristic search backend '{backend}'")


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
    backend: str = "python",
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
        backend=backend,
        position_upper_bound=upper_bound,
    )


def search_best_s_nq_policy(
    args,
    seed: int | None = None,
    horizon: int | None = None,
    position_upper_bound: int | None = None,
    top_k: int = 12,
    backend: str = "python",
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
        backend=backend,
        position_upper_bound=upper_bound,
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
    search_mode: str = "guided",
    backend: str = "python",
):
    search_seed = getattr(args, "seed", 1234) if seed is None else int(seed)
    search_horizon = args.horizon if horizon is None else int(horizon)
    upper_bound = _get_search_upper_bound(args, position_upper_bound)

    if search_mode == "exhaustive":
        candidates = []
        for s in range(upper_bound):
            for S in range(s + 1, upper_bound + 1):
                for q in range(1, int(args.max_order_size) + 1):
                    candidates.append({"s": s, "S": S, "q": q})
        summary = _search_candidates(
            args=args,
            policy_name="modified_s_s_q",
            candidates=candidates,
            seed=search_seed,
            horizon=search_horizon,
            top_k=min(10, len(candidates)),
            backend=backend,
            position_upper_bound=upper_bound,
        )
        return {
            "search_basis": None,
            "modified_policy": summary,
        }

    if search_mode != "guided":
        raise ValueError(f"Unknown modified-policy search mode '{search_mode}'")

    if s_s_summary is None:
        s_s_summary = search_best_s_s_policy(
            args=args,
            seed=seed,
            horizon=horizon,
            position_upper_bound=position_upper_bound,
            top_k=top_k_s_s_pairs,
            backend=backend,
        )

    candidates = list(_iter_modified_candidates(args=args, s_s_summary=s_s_summary, q_window=q_window))
    summary = _search_candidates(
        args=args,
        policy_name="modified_s_s_q",
        candidates=candidates,
        seed=search_seed,
        horizon=search_horizon,
        top_k=min(10, len(candidates)),
        backend=backend,
        position_upper_bound=upper_bound,
    )
    return {
        "search_basis": s_s_summary,
        "modified_policy": summary,
    }
