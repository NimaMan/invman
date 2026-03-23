from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from invman.problems.multi_echelon.env import MultiEchelonFixedPath, build_env_from_args


@dataclass
class PolicySearchResult:
    params: dict[str, int]
    mean_cost: float
    search_seed: int
    search_horizon: int

    def to_dict(self):
        return {
            "params": dict(self.params),
            "mean_cost": float(self.mean_cost),
            "search_seed": int(self.search_seed),
            "search_horizon": int(self.search_horizon),
        }


def build_fixed_demand_path(args, seed: int = 1234, horizon: int | None = None):
    path_horizon = int(args.horizon if horizon is None else horizon)
    np.random.seed(int(seed))
    env = build_env_from_args(args, horizon=path_horizon, track_demand=True)
    return MultiEchelonFixedPath(
        warehouse_inventory=int(env.warehouse_inventory),
        warehouse_pipeline=tuple(int(value) for value in env.warehouse_pipeline),
        retailer_inventory=tuple(int(value) for value in env.retailer_inventory.tolist()),
        retailer_pipeline=tuple(tuple(int(value) for value in row.tolist()) for row in env.retailer_pipeline),
        demands=tuple(tuple(int(value) for value in row.tolist()) for row in env.horizon_demands),
        expedite_uniforms=tuple(
            tuple(tuple(float(value) for value in row_unit.tolist()) for row_unit in row)
            for row in env.horizon_expedite_uniforms
        ),
        horizon=path_horizon,
        seed=int(seed),
    )


def _evaluate_on_path(args, params: dict[str, int], fixed_path: MultiEchelonFixedPath):
    env = build_env_from_args(args, horizon=fixed_path.horizon, track_demand=False)
    env.warehouse_inventory = int(fixed_path.warehouse_inventory)
    env.warehouse_pipeline = [int(value) for value in fixed_path.warehouse_pipeline]
    env.retailer_inventory = np.asarray(fixed_path.retailer_inventory, dtype=np.int64)
    env.retailer_pipeline = np.asarray(fixed_path.retailer_pipeline, dtype=np.int64)
    env.horizon_demands = np.asarray(fixed_path.demands, dtype=np.int64)
    env.horizon_expedite_uniforms = np.asarray(fixed_path.expedite_uniforms, dtype=np.float64)
    env.track_demand = True
    while not env.is_done():
        env.step((int(params["warehouse_level"]), int(params["retailer_level"])))
    return env.avg_total_cost


def evaluate_constant_base_stock_policy(args, params: dict[str, int], seed: int = 1234, horizon: int | None = None, fixed_path: MultiEchelonFixedPath | None = None):
    if fixed_path is not None:
        return _evaluate_on_path(args, params, fixed_path)
    np.random.seed(int(seed))
    env = build_env_from_args(args, horizon=horizon, track_demand=True)
    while not env.is_done():
        env.step((int(params["warehouse_level"]), int(params["retailer_level"])))
    return env.avg_total_cost


def evaluate_constant_base_stock_policy_across_seeds(args, params: dict[str, int], num_seeds: int = 3, seed: int | None = None, horizon: int | None = None):
    base_seed = int(getattr(args, "seed", 1234) if seed is None else seed)
    costs = [
        evaluate_constant_base_stock_policy(
            args,
            params=params,
            seed=base_seed + seed_offset,
            horizon=horizon,
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


def search_best_constant_base_stock_policy(args, seed: int = 1234, horizon: int | None = None, backend: str = "python"):
    path_horizon = int(args.horizon if horizon is None else horizon)
    fixed_path = build_fixed_demand_path(args, seed=seed, horizon=path_horizon)
    warehouse_levels = [int(value) for value in getattr(args, "warehouse_base_stock_levels", [50, 60, 70, 80, 90, 100])]
    retailer_levels = [int(value) for value in getattr(args, "retailer_base_stock_levels", [0, 5, 10, 15, 20, 25, 30, 35, 40])]

    if backend == "rust":
        try:
            import invman_rust
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Rust backend requested for multi-echelon heuristic search, but invman_rust is unavailable.") from exc

        best, top_results = invman_rust.multi_echelon_constant_base_stock_search_from_demands(
            warehouse_inventory=int(fixed_path.warehouse_inventory),
            warehouse_pipeline=list(fixed_path.warehouse_pipeline),
            retailer_inventory=list(fixed_path.retailer_inventory),
            retailer_pipeline=[list(row) for row in fixed_path.retailer_pipeline],
            demands=[list(row) for row in fixed_path.demands],
            expedite_uniforms=[
                [list(units) for units in period]
                for period in fixed_path.expedite_uniforms
            ],
            warehouse_levels=warehouse_levels,
            retailer_levels=retailer_levels,
            warehouse_holding_cost=float(args.warehouse_holding_cost),
            retailer_holding_cost=float(args.retailer_holding_cost),
            warehouse_expedited_cost=float(args.warehouse_expedited_cost),
            warehouse_lost_sale_cost=float(args.warehouse_lost_sale_cost),
            expedited_service_prob=float(args.expedited_service_prob),
            warehouse_capacity=int(args.warehouse_capacity),
            warehouse_inventory_cap=int(args.warehouse_inventory_cap),
            retailer_inventory_cap=int(args.retailer_inventory_cap),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            top_k=10,
        )
        return {
            "best_result": PolicySearchResult(
                params={"warehouse_level": int(best[0]), "retailer_level": int(best[1])},
                mean_cost=float(best[2]),
                search_seed=int(seed),
                search_horizon=int(path_horizon),
            ).to_dict(),
            "top_results": [
                PolicySearchResult(
                    params={"warehouse_level": int(row[0]), "retailer_level": int(row[1])},
                    mean_cost=float(row[2]),
                    search_seed=int(seed),
                    search_horizon=int(path_horizon),
                ).to_dict()
                for row in top_results
            ],
        }

    results = []
    for warehouse_level in warehouse_levels:
        for retailer_level in retailer_levels:
            params = {"warehouse_level": int(warehouse_level), "retailer_level": int(retailer_level)}
            results.append(
                PolicySearchResult(
                    params=params,
                    mean_cost=_evaluate_on_path(args, params, fixed_path),
                    search_seed=int(seed),
                    search_horizon=int(path_horizon),
                )
            )
    results.sort(key=lambda result: result.mean_cost)
    return {
        "best_result": results[0].to_dict(),
        "top_results": [result.to_dict() for result in results[:10]],
    }
