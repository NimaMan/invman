from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from invman.policies.common import normalize_action_spec


@dataclass(frozen=True)
class DualSourcingFixedPath:
    state: tuple[int, ...]
    demands: tuple[int, ...]
    horizon: int
    seed: int


class DualSourcingEnv:
    def __init__(
        self,
        regular_lead_time: int = 2,
        expedited_lead_time: int = 0,
        regular_order_cost: float = 100.0,
        expedited_order_cost: float = 105.0,
        holding_cost: float = 5.0,
        shortage_cost: float = 495.0,
        regular_max_order_size: int = 12,
        expedited_max_order_size: int = 12,
        demand_low: int = 0,
        demand_high: int = 4,
        horizon: int = 6000,
        track_demand: bool = True,
        warm_up_periods_ratio: float = 0.2,
    ):
        if expedited_lead_time != 0:
            raise NotImplementedError("The current dual-sourcing implementation supports only expedited lead time 0.")
        if regular_lead_time < 1:
            raise ValueError("regular_lead_time must be at least 1")
        if demand_high < demand_low:
            raise ValueError("demand_high must be >= demand_low")

        self.regular_lead_time = int(regular_lead_time)
        self.expedited_lead_time = int(expedited_lead_time)
        self.regular_order_cost = float(regular_order_cost)
        self.expedited_order_cost = float(expedited_order_cost)
        self.holding_cost = float(holding_cost)
        self.shortage_cost = float(shortage_cost)
        self.regular_max_order_size = int(regular_max_order_size)
        self.expedited_max_order_size = int(expedited_max_order_size)
        self.demand_low = int(demand_low)
        self.demand_high = int(demand_high)
        self.horizon = int(horizon)
        self.track_demand = bool(track_demand)
        self.warm_up_periods = int(float(warm_up_periods_ratio) * self.horizon)
        self.state_space_dim = self.regular_lead_time
        self.action_space_dim = (self.regular_max_order_size + 1) * (self.expedited_max_order_size + 1)
        self.action_spec = normalize_action_spec(
            {
                "action_dim": 2,
                "action_mode": "vector_quantity",
                "min_values": [0, 0],
                "max_values": [self.regular_max_order_size, self.expedited_max_order_size],
                "allowed_values": None,
            }
        )
        self.reset()

    @property
    def mean_demand(self):
        return 0.5 * (self.demand_low + self.demand_high)

    def _sample_demand_vector(self, size: int):
        return np.random.randint(self.demand_low, self.demand_high + 1, size=size)

    def reset(self):
        self.current_epoch = 0
        self.done = False
        self.total_cost = 0.0
        self.epoch_costs = []
        self.horizon_demand = self._sample_demand_vector(self.horizon) if self.track_demand else None
        mean_demand = self.mean_demand
        future_pipeline = [
            int(np.random.randint(0, self.regular_max_order_size + 1))
            for _ in range(max(0, self.regular_lead_time - 1))
        ]
        first_coordinate = int(round((self.regular_lead_time + 1) * mean_demand))
        self.state = [first_coordinate] + future_pipeline
        return self.policy_state

    @property
    def expedited_inventory_position(self) -> int:
        return int(self.state[0])

    @property
    def regular_inventory_position(self) -> int:
        return int(sum(self.state))

    @property
    def policy_state(self):
        scale = float(max(1, self.regular_max_order_size + self.expedited_max_order_size))
        return np.asarray(self.state, dtype=np.float32) / scale

    @property
    def avg_total_cost(self):
        if not self.epoch_costs:
            return 0.0
        if self.warm_up_periods < len(self.epoch_costs):
            return float(np.mean(self.epoch_costs[self.warm_up_periods:]))
        return float(np.mean(self.epoch_costs))

    def get_realized_demand(self):
        if self.horizon_demand is not None:
            return int(self.horizon_demand[self.current_epoch])
        return int(self._sample_demand_vector(1)[0])

    def is_done(self):
        return self.done

    def step(self, order_quantity):
        regular_order, expedited_order = [int(value) for value in order_quantity]
        if not (0 <= regular_order <= self.regular_max_order_size):
            raise ValueError(f"regular order {regular_order} outside 0..{self.regular_max_order_size}")
        if not (0 <= expedited_order <= self.expedited_max_order_size):
            raise ValueError(f"expedited order {expedited_order} outside 0..{self.expedited_max_order_size}")

        demand = self.get_realized_demand()
        available_inventory = int(self.state[0]) + expedited_order
        end_inventory = available_inventory - demand
        holding = self.holding_cost * max(end_inventory, 0)
        backlog = self.shortage_cost * max(-end_inventory, 0)
        procurement = self.regular_order_cost * regular_order + self.expedited_order_cost * expedited_order
        epoch_cost = float(holding + backlog + procurement)
        self.epoch_costs.append(epoch_cost)
        self.total_cost += epoch_cost

        if self.regular_lead_time == 1:
            next_state = [end_inventory + regular_order]
        else:
            next_state = [end_inventory + int(self.state[1])] + [int(value) for value in self.state[2:]] + [regular_order]
        self.state = next_state

        self.current_epoch += 1
        if self.current_epoch >= self.horizon:
            self.done = True
        return self.policy_state, epoch_cost, self.done


def build_env_from_args(args, horizon=None, track_demand=False):
    return DualSourcingEnv(
        regular_lead_time=int(args.regular_lead_time),
        expedited_lead_time=int(getattr(args, "expedited_lead_time", 0)),
        regular_order_cost=float(getattr(args, "regular_order_cost", 100.0)),
        expedited_order_cost=float(getattr(args, "expedited_order_cost", 105.0)),
        holding_cost=float(args.holding_cost),
        shortage_cost=float(args.shortage_cost),
        regular_max_order_size=int(getattr(args, "regular_max_order_size", 12)),
        expedited_max_order_size=int(getattr(args, "expedited_max_order_size", 12)),
        demand_low=int(getattr(args, "dual_demand_low", 0)),
        demand_high=int(getattr(args, "dual_demand_high", 4)),
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
    use_rust_rollout = _should_use_rust_soft_tree_rollout(
        model,
        args,
        track_demand=track_demand,
        return_env=return_env,
    )
    if model_params is not None and not use_rust_rollout:
        model.set_model_params(model_params)

    if use_rust_rollout:
        import invman_rust

        flat_params = model_params if model_params is not None else model.get_model_flat_params()
        avg_cost = invman_rust.dual_sourcing_soft_tree_rollout(
            flat_params=np.asarray(flat_params, dtype=np.float32).tolist(),
            input_dim=int(model.input_dim),
            depth=int(model.depth),
            min_values=[int(value) for value in model.min_values],
            max_values=[int(value) for value in model.max_values],
            allowed_values=model.control_spec["allowed_values"],
            split_type=str(getattr(model, "split_type", "oblique")),
            leaf_type=str(getattr(model, "leaf_type", "constant")),
            action_mode=str(getattr(model, "control_mode", "vector_quantity")),
            action_adapter=str(getattr(model, "action_adapter", "identity")),
            regular_lead_time=int(args.regular_lead_time),
            regular_order_cost=float(args.regular_order_cost),
            expedited_order_cost=float(args.expedited_order_cost),
            holding_cost=float(args.holding_cost),
            shortage_cost=float(args.shortage_cost),
            regular_max_order_size=int(args.regular_max_order_size),
            expedited_max_order_size=int(args.expedited_max_order_size),
            demand_low=int(args.dual_demand_low),
            demand_high=int(args.dual_demand_high),
            horizon=int(args.horizon),
            seed=int(seed),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            temperature=float(model.temperature),
        )
        if verbose:
            print(f"Seed {seed}: avg cost {avg_cost:.4f}")
        return -float(avg_cost), indiv_idx

    np.random.seed(seed)
    env = build_env_from_args(args, track_demand=track_demand)
    state = env.policy_state
    done = False
    while not done:
        action = model(state)
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
    costs = invman_rust.dual_sourcing_soft_tree_population_rollout(
        params_batch=params_batch,
        input_dim=int(model.input_dim),
        depth=int(model.depth),
        min_values=[int(value) for value in model.min_values],
        max_values=[int(value) for value in model.max_values],
        allowed_values=model.control_spec["allowed_values"],
        split_type=str(getattr(model, "split_type", "oblique")),
        leaf_type=str(getattr(model, "leaf_type", "constant")),
        action_mode=str(getattr(model, "control_mode", "vector_quantity")),
        action_adapter=str(getattr(model, "action_adapter", "identity")),
        regular_lead_time=int(args.regular_lead_time),
        regular_order_cost=float(args.regular_order_cost),
        expedited_order_cost=float(args.expedited_order_cost),
        holding_cost=float(args.holding_cost),
        shortage_cost=float(args.shortage_cost),
        regular_max_order_size=int(args.regular_max_order_size),
        expedited_max_order_size=int(args.expedited_max_order_size),
        demand_low=int(args.dual_demand_low),
        demand_high=int(args.dual_demand_high),
        horizon=int(args.horizon),
        seeds=[int(seed) for seed in seeds],
        warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
        temperature=float(model.temperature),
    )
    return [(-float(cost), indiv_idx) for indiv_idx, cost in enumerate(costs)]
