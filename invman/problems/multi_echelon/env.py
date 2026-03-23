from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from invman.policies.common import normalize_action_spec


@dataclass(frozen=True)
class MultiEchelonFixedPath:
    warehouse_inventory: int
    warehouse_pipeline: tuple[int, ...]
    retailer_inventory: tuple[int, ...]
    retailer_pipeline: tuple[tuple[int, ...], ...]
    demands: tuple[tuple[int, ...], ...]
    expedite_uniforms: tuple[tuple[tuple[float, ...], ...], ...]
    horizon: int
    seed: int


class MultiEchelonEnv:
    def __init__(
        self,
        warehouse_lead_time: int = 2,
        retailer_lead_time: int = 2,
        num_retailers: int = 10,
        warehouse_holding_cost: float = 3.0,
        retailer_holding_cost: float = 3.0,
        warehouse_expedited_cost: float = 0.0,
        warehouse_lost_sale_cost: float = 60.0,
        expedited_service_prob: float = 0.8,
        warehouse_capacity: int = 100,
        warehouse_inventory_cap: int = 1000,
        retailer_inventory_cap: int = 100,
        demand_mean: float = 5.0,
        demand_std: float = 14.0,
        warehouse_base_stock_levels: list[int] | None = None,
        retailer_base_stock_levels: list[int] | None = None,
        horizon: int = 4000,
        track_demand: bool = True,
        warm_up_periods_ratio: float = 0.2,
    ):
        self.warehouse_lead_time = int(warehouse_lead_time)
        self.retailer_lead_time = int(retailer_lead_time)
        self.num_retailers = int(num_retailers)
        self.warehouse_holding_cost = float(warehouse_holding_cost)
        self.retailer_holding_cost = float(retailer_holding_cost)
        self.warehouse_expedited_cost = float(warehouse_expedited_cost)
        self.warehouse_lost_sale_cost = float(warehouse_lost_sale_cost)
        self.expedited_service_prob = float(expedited_service_prob)
        self.warehouse_capacity = int(warehouse_capacity)
        self.warehouse_inventory_cap = int(warehouse_inventory_cap)
        self.retailer_inventory_cap = int(retailer_inventory_cap)
        self.demand_mean = float(demand_mean)
        self.demand_std = float(demand_std)
        self.horizon = int(horizon)
        self.track_demand = bool(track_demand)
        self.warm_up_periods = int(float(warm_up_periods_ratio) * self.horizon)
        self.warehouse_base_stock_levels = list(
            warehouse_base_stock_levels if warehouse_base_stock_levels is not None else [50, 60, 70, 80, 90, 100]
        )
        self.retailer_base_stock_levels = list(
            retailer_base_stock_levels if retailer_base_stock_levels is not None else [0, 5, 10, 15, 20, 25, 30, 35, 40]
        )
        self.action_spec = normalize_action_spec(
            {
                "action_dim": 2,
                "action_mode": "discrete_grid",
                "allowed_values": [
                    [int(value) for value in self.warehouse_base_stock_levels],
                    [int(value) for value in self.retailer_base_stock_levels],
                ],
            }
        )
        self.state_space_dim = self.warehouse_lead_time + self.num_retailers * self.retailer_lead_time
        self.action_space_dim = len(self.warehouse_base_stock_levels) * len(self.retailer_base_stock_levels)
        self.max_order_size = int(max(self.warehouse_base_stock_levels))
        self.reset()

    def _sample_demands(self, size: int):
        sampled = np.rint(np.random.normal(loc=self.demand_mean, scale=self.demand_std, size=(size, self.num_retailers))).astype(np.int64)
        return np.maximum(sampled, 0)

    def _sample_expedite_uniforms(self, size: int, max_units: int):
        return np.random.random(size=(size, self.num_retailers, max_units))

    def reset(self):
        self.current_epoch = 0
        self.done = False
        self.total_cost = 0.0
        self.epoch_costs = []
        max_units = max(1, int(np.ceil(max(self.demand_mean + 6.0 * self.demand_std, 20.0))))
        self.horizon_demands = self._sample_demands(self.horizon) if self.track_demand else None
        self.horizon_expedite_uniforms = self._sample_expedite_uniforms(self.horizon, max_units) if self.track_demand else None
        self.warehouse_inventory = int(round(self.num_retailers * max(self.demand_mean, 1.0)))
        self.warehouse_pipeline = [int(np.random.choice(self.warehouse_base_stock_levels)) for _ in range(self.warehouse_lead_time)]
        self.retailer_inventory = np.full(self.num_retailers, int(round(max(self.demand_mean, 1.0))), dtype=np.int64)
        self.retailer_pipeline = np.zeros((self.num_retailers, self.retailer_lead_time), dtype=np.int64)
        for retailer_idx in range(self.num_retailers):
            self.retailer_pipeline[retailer_idx, :] = np.random.choice(self.retailer_base_stock_levels, size=self.retailer_lead_time)
        return self.policy_state

    def _decision_state(self):
        warehouse_available = int(self.warehouse_inventory + self.warehouse_pipeline[0])
        retailer_available = self.retailer_inventory + self.retailer_pipeline[:, 0]
        warehouse_future = self.warehouse_pipeline[1:]
        retailer_future = self.retailer_pipeline[:, 1:]
        return warehouse_available, retailer_available.astype(np.int64), warehouse_future, retailer_future.astype(np.int64)

    @property
    def policy_state(self):
        warehouse_available, retailer_available, warehouse_future, retailer_future = self._decision_state()
        scale = float(max(1, self.warehouse_inventory_cap))
        pieces = [np.asarray([warehouse_available], dtype=np.float32) / scale]
        if warehouse_future:
            pieces.append(np.asarray(warehouse_future, dtype=np.float32) / scale)
        pieces.append(retailer_available.astype(np.float32) / float(max(1, self.retailer_inventory_cap)))
        if retailer_future.size:
            pieces.append(retailer_future.astype(np.float32).reshape(-1) / float(max(1, self.retailer_inventory_cap)))
        return np.concatenate(pieces).astype(np.float32, copy=False)

    @property
    def avg_total_cost(self):
        if not self.epoch_costs:
            return 0.0
        if self.warm_up_periods < len(self.epoch_costs):
            return float(np.mean(self.epoch_costs[self.warm_up_periods:]))
        return float(np.mean(self.epoch_costs))

    def _current_demands(self):
        if self.horizon_demands is not None:
            return self.horizon_demands[self.current_epoch]
        return self._sample_demands(1)[0]

    def _current_expedite_uniforms(self):
        if self.horizon_expedite_uniforms is not None:
            return np.asarray(self.horizon_expedite_uniforms[self.current_epoch], dtype=np.float64)
        max_units = max(1, int(np.ceil(max(self.demand_mean + 6.0 * self.demand_std, 20.0))))
        return self._sample_expedite_uniforms(1, max_units)[0]

    def is_done(self):
        return self.done

    def step(self, base_stock_levels):
        warehouse_target, retailer_target = [int(value) for value in base_stock_levels]
        warehouse_target = int(min(max(warehouse_target, 0), self.warehouse_inventory_cap))
        retailer_target = int(min(max(retailer_target, 0), self.retailer_inventory_cap))

        warehouse_available, retailer_available, warehouse_future, retailer_future = self._decision_state()
        self.warehouse_pipeline = list(warehouse_future)
        self.retailer_pipeline = retailer_future.copy() if retailer_future.size else np.zeros((self.num_retailers, 0), dtype=np.int64)

        warehouse_ip = warehouse_available + int(sum(self.warehouse_pipeline))
        retailer_ip = retailer_available + (self.retailer_pipeline.sum(axis=1) if self.retailer_pipeline.size else 0)

        warehouse_order = min(self.warehouse_capacity, max(0, warehouse_target - warehouse_ip))
        desired_retail_orders = np.maximum(0, retailer_target - retailer_ip)
        desired_retail_orders = np.minimum(desired_retail_orders, self.retailer_inventory_cap).astype(np.int64)

        shipped_retail_orders = np.zeros(self.num_retailers, dtype=np.int64)
        remaining_warehouse_inventory = int(warehouse_available)
        for retailer_idx in range(self.num_retailers):
            shipped = min(int(desired_retail_orders[retailer_idx]), remaining_warehouse_inventory)
            shipped_retail_orders[retailer_idx] = shipped
            remaining_warehouse_inventory -= shipped

        self.warehouse_pipeline.append(int(warehouse_order))
        shipped_retail_orders_col = shipped_retail_orders.reshape(self.num_retailers, 1)
        if self.retailer_pipeline.size:
            self.retailer_pipeline = np.concatenate([self.retailer_pipeline, shipped_retail_orders_col], axis=1)
        else:
            self.retailer_pipeline = shipped_retail_orders_col

        demands = self._current_demands()
        retailer_on_hand = retailer_available.astype(np.int64)
        served_at_retailers = np.minimum(retailer_on_hand, demands)
        unmet = demands - served_at_retailers
        retailer_end_inventory = retailer_on_hand - served_at_retailers

        expedite_uniforms = self._current_expedite_uniforms()
        accepted_same_day = np.zeros(self.num_retailers, dtype=np.int64)
        for retailer_idx in range(self.num_retailers):
            units = int(unmet[retailer_idx])
            if units > 0:
                accepted_same_day[retailer_idx] = int(np.sum(expedite_uniforms[retailer_idx, :units] < self.expedited_service_prob))

        total_accepted = int(np.sum(accepted_same_day))
        expedited_shipped = min(total_accepted, remaining_warehouse_inventory)
        remaining_warehouse_inventory -= expedited_shipped
        lost_at_retailer = int(np.sum(unmet - accepted_same_day))
        lost_at_warehouse = int(total_accepted - expedited_shipped)

        self.warehouse_inventory = int(remaining_warehouse_inventory)
        self.retailer_inventory = retailer_end_inventory.astype(np.int64)

        epoch_cost = (
            float(self.warehouse_holding_cost) * max(self.warehouse_inventory, 0)
            + float(self.retailer_holding_cost) * float(np.sum(np.maximum(self.retailer_inventory, 0)))
            + float(self.warehouse_expedited_cost) * expedited_shipped
            + float(self.warehouse_lost_sale_cost) * (lost_at_retailer + lost_at_warehouse)
        )
        self.epoch_costs.append(float(epoch_cost))
        self.total_cost += float(epoch_cost)
        self.current_epoch += 1
        if self.current_epoch >= self.horizon:
            self.done = True
        return self.policy_state, float(epoch_cost), self.done


def build_env_from_args(args, horizon=None, track_demand=False):
    return MultiEchelonEnv(
        warehouse_lead_time=int(args.warehouse_lead_time),
        retailer_lead_time=int(args.retailer_lead_time),
        num_retailers=int(args.num_retailers),
        warehouse_holding_cost=float(args.warehouse_holding_cost),
        retailer_holding_cost=float(args.retailer_holding_cost),
        warehouse_expedited_cost=float(args.warehouse_expedited_cost),
        warehouse_lost_sale_cost=float(args.warehouse_lost_sale_cost),
        expedited_service_prob=float(args.expedited_service_prob),
        warehouse_capacity=int(args.warehouse_capacity),
        warehouse_inventory_cap=int(args.warehouse_inventory_cap),
        retailer_inventory_cap=int(args.retailer_inventory_cap),
        demand_mean=float(args.multi_demand_mean),
        demand_std=float(args.multi_demand_std),
        warehouse_base_stock_levels=list(getattr(args, "warehouse_base_stock_levels", [50, 60, 70, 80, 90, 100])),
        retailer_base_stock_levels=list(getattr(args, "retailer_base_stock_levels", [0, 5, 10, 15, 20, 25, 30, 35, 40])),
        horizon=int(args.horizon if horizon is None else horizon),
        track_demand=bool(track_demand),
        warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
    )


def _should_use_rust_soft_tree_rollout(model, args, track_demand=False, return_env=False):
    if return_env or track_demand:
        return False
    if getattr(args, "rollout_backend", "python") != "rust":
        return False
    return type(model).__name__ == "SoftTreePolicy"


def get_model_fitness(
    model,
    args,
    model_params=None,
    seed=1234,
    indiv_idx=-1,
    return_env=False,
    track_demand=False,
    verbose=False,
):
    import torch

    use_rust_rollout = _should_use_rust_soft_tree_rollout(model, args, track_demand=track_demand, return_env=return_env)
    if model_params is not None and not use_rust_rollout:
        model.set_model_params(model_params)

    if use_rust_rollout:
        import invman_rust

        flat_params = model_params if model_params is not None else model.get_model_flat_params()
        avg_cost = invman_rust.multi_echelon_soft_tree_rollout(
            flat_params=np.asarray(flat_params, dtype=np.float32).tolist(),
            input_dim=int(model.input_dim),
            depth=int(model.depth),
            min_values=[int(value) for value in model.min_values],
            max_values=[int(value) for value in model.max_values],
            allowed_values=model.action_spec["allowed_values"],
            split_type=str(getattr(model, "split_type", "oblique")),
            leaf_type=str(getattr(model, "leaf_type", "constant")),
            action_mode=str(getattr(model, "action_mode", "discrete_grid")),
            warehouse_lead_time=int(args.warehouse_lead_time),
            retailer_lead_time=int(args.retailer_lead_time),
            num_retailers=int(args.num_retailers),
            warehouse_holding_cost=float(args.warehouse_holding_cost),
            retailer_holding_cost=float(args.retailer_holding_cost),
            warehouse_expedited_cost=float(args.warehouse_expedited_cost),
            warehouse_lost_sale_cost=float(args.warehouse_lost_sale_cost),
            expedited_service_prob=float(args.expedited_service_prob),
            warehouse_capacity=int(args.warehouse_capacity),
            warehouse_inventory_cap=int(args.warehouse_inventory_cap),
            retailer_inventory_cap=int(args.retailer_inventory_cap),
            demand_mean=float(args.multi_demand_mean),
            demand_std=float(args.multi_demand_std),
            horizon=int(args.horizon),
            seed=int(seed),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            temperature=float(model.temperature),
        )
        if verbose:
            print(f"Seed {seed}: avg cost {avg_cost:.4f}")
        return -float(avg_cost), indiv_idx

    np.random.seed(seed)
    torch.manual_seed(seed)
    env = build_env_from_args(args, track_demand=track_demand)
    state = env.policy_state
    done = False
    while not done:
        action = model(torch.as_tensor(state, dtype=torch.float32))
        state, _, done = env.step(action)
    if verbose:
        print(f"Seed {seed}: avg cost {env.avg_total_cost:.4f}")
    if return_env:
        return -env.avg_total_cost, env
    return -env.avg_total_cost, indiv_idx


def get_population_fitness(model, args, model_params_batch, seeds):
    if not _should_use_rust_soft_tree_rollout(model, args, track_demand=False, return_env=False):
        return None

    import invman_rust

    params_batch = [np.asarray(model_params, dtype=np.float32).tolist() for model_params in model_params_batch]
    costs = invman_rust.multi_echelon_soft_tree_population_rollout(
        params_batch=params_batch,
        input_dim=int(model.input_dim),
        depth=int(model.depth),
        min_values=[int(value) for value in model.min_values],
        max_values=[int(value) for value in model.max_values],
        allowed_values=model.action_spec["allowed_values"],
        split_type=str(getattr(model, "split_type", "oblique")),
        leaf_type=str(getattr(model, "leaf_type", "constant")),
        action_mode=str(getattr(model, "action_mode", "discrete_grid")),
        warehouse_lead_time=int(args.warehouse_lead_time),
        retailer_lead_time=int(args.retailer_lead_time),
        num_retailers=int(args.num_retailers),
        warehouse_holding_cost=float(args.warehouse_holding_cost),
        retailer_holding_cost=float(args.retailer_holding_cost),
        warehouse_expedited_cost=float(args.warehouse_expedited_cost),
        warehouse_lost_sale_cost=float(args.warehouse_lost_sale_cost),
        expedited_service_prob=float(args.expedited_service_prob),
        warehouse_capacity=int(args.warehouse_capacity),
        warehouse_inventory_cap=int(args.warehouse_inventory_cap),
        retailer_inventory_cap=int(args.retailer_inventory_cap),
        demand_mean=float(args.multi_demand_mean),
        demand_std=float(args.multi_demand_std),
        horizon=int(args.horizon),
        seeds=[int(seed) for seed in seeds],
        warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
        temperature=float(model.temperature),
    )
    return [(-float(cost), indiv_idx) for indiv_idx, cost in enumerate(costs)]
