from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from invman.problems.dual_sourcing.env import DualSourcingFixedPath, build_env_from_args


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


def get_single_index_action(regular_inventory_position: int, s_e: int, s_r: int, max_regular: int, max_expedited: int):
    expedited = min(max(0, int(s_e) - int(regular_inventory_position)), int(max_expedited))
    regular = min(max(0, int(s_r) - int(regular_inventory_position) - int(expedited)), int(max_regular))
    return regular, expedited


def get_dual_index_action(expedited_inventory_position: int, regular_inventory_position: int, s_e: int, s_r: int, max_regular: int, max_expedited: int):
    expedited = min(max(0, int(s_e) - int(expedited_inventory_position)), int(max_expedited))
    regular = min(max(0, int(s_r) - int(regular_inventory_position) - int(expedited)), int(max_regular))
    return regular, expedited


def get_capped_dual_index_action(expedited_inventory_position: int, regular_inventory_position: int, s_e: int, s_r: int, cap_r: int, max_regular: int, max_expedited: int):
    expedited = min(max(0, int(s_e) - int(expedited_inventory_position)), int(max_expedited))
    desired_regular = max(0, int(s_r) - int(regular_inventory_position) - int(expedited))
    regular = min(desired_regular, int(cap_r), int(max_regular))
    return regular, expedited


def get_tailored_base_surge_action(expedited_inventory_position: int, surge_level: int, regular_qty: int, max_regular: int, max_expedited: int):
    expedited = min(max(0, int(surge_level) - int(expedited_inventory_position)), int(max_expedited))
    regular = min(max(0, int(regular_qty)), int(max_regular))
    return regular, expedited


def _policy_action(policy_name: str, state: tuple[int, ...], params: dict[str, int], args):
    expedited_ip = int(state[0])
    regular_ip = int(sum(state))
    max_regular = int(args.regular_max_order_size)
    max_expedited = int(args.expedited_max_order_size)
    if policy_name == "single_index":
        return get_single_index_action(
            regular_inventory_position=regular_ip,
            s_e=params["s_e"],
            s_r=params["s_r"],
            max_regular=max_regular,
            max_expedited=max_expedited,
        )
    if policy_name == "dual_index":
        return get_dual_index_action(
            expedited_inventory_position=expedited_ip,
            regular_inventory_position=regular_ip,
            s_e=params["s_e"],
            s_r=params["s_r"],
            max_regular=max_regular,
            max_expedited=max_expedited,
        )
    if policy_name == "capped_dual_index":
        return get_capped_dual_index_action(
            expedited_inventory_position=expedited_ip,
            regular_inventory_position=regular_ip,
            s_e=params["s_e"],
            s_r=params["s_r"],
            cap_r=params["cap_r"],
            max_regular=max_regular,
            max_expedited=max_expedited,
        )
    if policy_name == "tailored_base_surge":
        return get_tailored_base_surge_action(
            expedited_inventory_position=expedited_ip,
            surge_level=params["surge_level"],
            regular_qty=params["regular_qty"],
            max_regular=max_regular,
            max_expedited=max_expedited,
        )
    raise NotImplementedError(f"Unknown dual-sourcing policy '{policy_name}'")


def build_fixed_demand_path(args, seed: int = 1234, horizon: int | None = None):
    path_horizon = int(args.horizon if horizon is None else horizon)
    np.random.seed(int(seed))
    env = build_env_from_args(args, horizon=path_horizon, track_demand=True)
    return DualSourcingFixedPath(
        state=tuple(int(value) for value in env.state),
        demands=tuple(int(demand) for demand in env.horizon_demand.tolist()),
        horizon=path_horizon,
        seed=int(seed),
    )


def evaluate_policy_cost_on_path(args, policy_name: str, params: dict[str, int], fixed_path: DualSourcingFixedPath):
    state = list(fixed_path.state)
    epoch_costs = []
    for demand in fixed_path.demands:
        regular_order, expedited_order = _policy_action(policy_name, tuple(state), params, args)
        available_inventory = int(state[0]) + int(expedited_order)
        end_inventory = available_inventory - int(demand)
        epoch_cost = (
            float(args.regular_order_cost) * int(regular_order)
            + float(args.expedited_order_cost) * int(expedited_order)
            + float(args.holding_cost) * max(end_inventory, 0)
            + float(args.shortage_cost) * max(-end_inventory, 0)
        )
        epoch_costs.append(float(epoch_cost))
        if len(state) == 1:
            state = [end_inventory + int(regular_order)]
        else:
            state = [end_inventory + int(state[1])] + [int(value) for value in state[2:]] + [int(regular_order)]

    warm_up_ratio = float(getattr(args, "warm_up_periods_ratio", 0.2))
    warm_up_periods = min(len(epoch_costs), int(warm_up_ratio * len(epoch_costs)))
    active_costs = epoch_costs[warm_up_periods:] if warm_up_periods < len(epoch_costs) else epoch_costs
    return float(np.mean(active_costs))


def evaluate_policy_cost(args, policy_name: str, params: dict[str, int], seed: int = 1234, horizon: int | None = None, track_demand: bool = True, fixed_path: DualSourcingFixedPath | None = None):
    if fixed_path is not None:
        return evaluate_policy_cost_on_path(args, policy_name, params, fixed_path)

    env = build_env_from_args(args, horizon=horizon, track_demand=track_demand)
    while not env.is_done():
        env.step(_policy_action(policy_name, tuple(env.state), params, args))
    return env.avg_total_cost


def evaluate_policy_across_seeds(args, policy_name: str, params: dict[str, int], num_seeds: int = 3, seed: int | None = None, horizon: int | None = None, track_demand: bool = True):
    base_seed = int(getattr(args, "seed", 1234) if seed is None else seed)
    costs = [
        evaluate_policy_cost(
            args=args,
            policy_name=policy_name,
            params=params,
            seed=base_seed + seed_offset,
            horizon=horizon,
            track_demand=track_demand,
        )
        for seed_offset in range(num_seeds)
    ]
    return {
        "params": dict(params),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "num_seeds": int(num_seeds),
    }


def _get_target_upper_bound(args):
    mean_demand = 0.5 * (int(args.dual_demand_low) + int(args.dual_demand_high))
    upper = int(round((int(args.regular_lead_time) + 2) * mean_demand + 2 * int(args.expedited_max_order_size)))
    return max(int(args.expedited_max_order_size), min(24, upper))


def _sorted_summary(results: list[PolicySearchResult], top_k: int):
    results.sort(key=lambda result: result.mean_cost)
    return PolicySearchSummary(
        best_result=results[0],
        top_results=results[:top_k],
        evaluated_candidates=len(results),
    )


def _search_candidates(args, policy_name: str, candidates: list[dict[str, int]], seed: int, horizon: int, top_k: int, backend: str):
    fixed_path = build_fixed_demand_path(args, seed=seed, horizon=horizon)
    if backend == "python":
        results = [
            PolicySearchResult(
                policy_name=policy_name,
                params=dict(params),
                mean_cost=evaluate_policy_cost_on_path(args, policy_name, params, fixed_path),
                search_seed=int(seed),
                search_horizon=int(horizon),
            )
            for params in candidates
        ]
        return _sorted_summary(results, top_k)

    if backend == "rust":
        try:
            import invman_rust
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Rust backend requested for dual-sourcing heuristic search, but invman_rust is unavailable.") from exc

        common_kwargs = {
            "state": list(fixed_path.state),
            "demands": list(fixed_path.demands),
            "regular_max_order_size": int(args.regular_max_order_size),
            "expedited_max_order_size": int(args.expedited_max_order_size),
            "regular_order_cost": float(args.regular_order_cost),
            "expedited_order_cost": float(args.expedited_order_cost),
            "holding_cost": float(args.holding_cost),
            "shortage_cost": float(args.shortage_cost),
            "warm_up_periods_ratio": float(getattr(args, "warm_up_periods_ratio", 0.2)),
            "top_k": int(top_k),
            "target_upper_bound": int(_get_target_upper_bound(args)),
        }
        if policy_name == "single_index":
            best, top_results = invman_rust.dual_sourcing_single_index_search_from_demands(**common_kwargs)
            results = [
                PolicySearchResult("single_index", {"s_e": int(s_e), "s_r": int(s_r)}, float(cost), int(seed), int(horizon))
                for s_e, s_r, cost in top_results
            ]
            return PolicySearchSummary(
                best_result=PolicySearchResult("single_index", {"s_e": int(best[0]), "s_r": int(best[1])}, float(best[2]), int(seed), int(horizon)),
                top_results=results,
                evaluated_candidates=len(candidates),
            )
        if policy_name == "dual_index":
            best, top_results = invman_rust.dual_sourcing_dual_index_search_from_demands(**common_kwargs)
            results = [
                PolicySearchResult("dual_index", {"s_e": int(s_e), "s_r": int(s_r)}, float(cost), int(seed), int(horizon))
                for s_e, s_r, cost in top_results
            ]
            return PolicySearchSummary(
                best_result=PolicySearchResult("dual_index", {"s_e": int(best[0]), "s_r": int(best[1])}, float(best[2]), int(seed), int(horizon)),
                top_results=results,
                evaluated_candidates=len(candidates),
            )
        if policy_name == "capped_dual_index":
            best, top_results = invman_rust.dual_sourcing_capped_dual_index_search_from_demands(**common_kwargs)
            results = [
                PolicySearchResult("capped_dual_index", {"s_e": int(s_e), "s_r": int(s_r), "cap_r": int(cap_r)}, float(cost), int(seed), int(horizon))
                for s_e, s_r, cap_r, cost in top_results
            ]
            return PolicySearchSummary(
                best_result=PolicySearchResult("capped_dual_index", {"s_e": int(best[0]), "s_r": int(best[1]), "cap_r": int(best[2])}, float(best[3]), int(seed), int(horizon)),
                top_results=results,
                evaluated_candidates=len(candidates),
            )
        if policy_name == "tailored_base_surge":
            best, top_results = invman_rust.dual_sourcing_tailored_base_surge_search_from_demands(**common_kwargs)
            results = [
                PolicySearchResult("tailored_base_surge", {"surge_level": int(surge_level), "regular_qty": int(regular_qty)}, float(cost), int(seed), int(horizon))
                for surge_level, regular_qty, cost in top_results
            ]
            return PolicySearchSummary(
                best_result=PolicySearchResult("tailored_base_surge", {"surge_level": int(best[0]), "regular_qty": int(best[1])}, float(best[2]), int(seed), int(horizon)),
                top_results=results,
                evaluated_candidates=len(candidates),
            )
    raise NotImplementedError(f"Unsupported backend/policy combination: {backend}/{policy_name}")


def search_best_single_index_policy(args, seed: int = 1234, horizon: int | None = None, top_k: int = 10, backend: str = "python"):
    path_horizon = int(args.horizon if horizon is None else horizon)
    upper = _get_target_upper_bound(args)
    candidates = [{"s_e": s_e, "s_r": s_r} for s_e in range(upper + 1) for s_r in range(s_e, upper + 1)]
    return _search_candidates(args, "single_index", candidates, seed, path_horizon, top_k, backend)


def search_best_dual_index_policy(args, seed: int = 1234, horizon: int | None = None, top_k: int = 10, backend: str = "python"):
    path_horizon = int(args.horizon if horizon is None else horizon)
    upper = _get_target_upper_bound(args)
    candidates = [{"s_e": s_e, "s_r": s_r} for s_e in range(upper + 1) for s_r in range(s_e, upper + 1)]
    return _search_candidates(args, "dual_index", candidates, seed, path_horizon, top_k, backend)


def search_best_capped_dual_index_policy(args, seed: int = 1234, horizon: int | None = None, top_k: int = 10, backend: str = "python"):
    path_horizon = int(args.horizon if horizon is None else horizon)
    upper = _get_target_upper_bound(args)
    candidates = [
        {"s_e": s_e, "s_r": s_r, "cap_r": cap_r}
        for s_e in range(upper + 1)
        for s_r in range(s_e, upper + 1)
        for cap_r in range(int(args.regular_max_order_size) + 1)
    ]
    return _search_candidates(args, "capped_dual_index", candidates, seed, path_horizon, top_k, backend)


def search_best_tailored_base_surge_policy(args, seed: int = 1234, horizon: int | None = None, top_k: int = 10, backend: str = "python"):
    path_horizon = int(args.horizon if horizon is None else horizon)
    upper = _get_target_upper_bound(args)
    candidates = [
        {"surge_level": surge_level, "regular_qty": regular_qty}
        for surge_level in range(upper + 1)
        for regular_qty in range(int(args.regular_max_order_size) + 1)
    ]
    return _search_candidates(args, "tailored_base_surge", candidates, seed, path_horizon, top_k, backend)
