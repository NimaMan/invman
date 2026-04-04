from collections import deque

import numpy as np

from invman.problems.lost_sales.demand import (
    DEFAULT_MMPP2_LAMBDA_HIGH,
    DEFAULT_MMPP2_LAMBDA_LOW,
    DEFAULT_MMPP2_POSITIVE_P00,
    DEFAULT_MMPP2_POSITIVE_P11,
    build_demand_config,
    build_demand_config_from_args,
    build_demand_process,
    get_cumulative_demand_cdf,
    get_demand_prob_vector,
)


def _normalize_state_features(state_features: str) -> str:
    aliases = {
        "pipeline": "pipeline",
        "pipeline_plus_summary": "pipeline_plus_summary",
        "augmented": "pipeline_plus_summary",
        "feature_augmented": "pipeline_plus_summary",
        "raw_pipeline": "raw_pipeline",
        "pipeline_raw": "raw_pipeline",
        "raw_pipeline_plus_summary": "raw_pipeline_plus_summary",
        "pipeline_plus_summary_raw": "raw_pipeline_plus_summary",
        "raw_augmented": "raw_pipeline_plus_summary",
    }
    normalized = aliases.get(state_features)
    if normalized is None:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown state feature mode '{state_features}'. Expected one of: {valid}")
    return normalized


class LostSalesEnv:
    def __init__(
        self,
        demand_rate: float,
        lead_time: int = 2,
        max_order_size: int = 25,
        one_hot_inventory_upper_bound: int = 200,
        holding_cost: float = 1.0,
        shortage_cost: float = 4.0,
        horizon: int = int(1e5),
        procurement_cost: float = 0.0,
        fixed_order_cost: float = 0.0,
        demand_dist_name: str = "Poisson",
        demand_lambda_low: float = DEFAULT_MMPP2_LAMBDA_LOW,
        demand_lambda_high: float = DEFAULT_MMPP2_LAMBDA_HIGH,
        demand_p00: float = DEFAULT_MMPP2_POSITIVE_P00,
        demand_p11: float = DEFAULT_MMPP2_POSITIVE_P11,
        track_demand: bool = True,
        warm_up_periods_ratio: float = 0.2,
        state_features: str = "pipeline",
    ):
        if lead_time < 1:
            raise ValueError("lead_time must be at least 1")

        self.demand_rate = float(demand_rate)
        self.demand_dist_name = demand_dist_name
        self.holding_cost = float(holding_cost)
        self.shortage_cost = float(shortage_cost)
        self.procurement_cost = float(procurement_cost)
        self.fixed_order_cost = float(fixed_order_cost)
        del max_order_size
        self.one_hot_inventory_upper_bound = int(one_hot_inventory_upper_bound)
        self.lead_time = int(lead_time)
        self.state_features = _normalize_state_features(state_features)
        self.state_space_dim = self.get_state_dim()
        self.lead_time_orders = deque(maxlen=self.lead_time)
        self.current_epoch = 0
        self.done = False
        self.horizon = int(horizon)
        self.warm_up_periods = int(warm_up_periods_ratio * self.horizon)
        self.gamma = 0.995
        self.track_demand = track_demand
        self.demand_lambda_low = float(demand_lambda_low)
        self.demand_lambda_high = float(demand_lambda_high)
        self.demand_p00 = float(demand_p00)
        self.demand_p11 = float(demand_p11)
        self.demand_config = build_demand_config(
            demand_dist_name=self.demand_dist_name,
            demand_rate=self.demand_rate,
            demand_lambda_low=self.demand_lambda_low,
            demand_lambda_high=self.demand_lambda_high,
            demand_p00=self.demand_p00,
            demand_p11=self.demand_p11,
        )
        self.demand_probs, self.demand_lb, self.demand_ub = get_demand_prob_vector(self.demand_config)
        self._runtime_demand_process = None
        self.reset()

    def initialize_env(self):
        self.current_inventory_level = int(round(2 * self.demand_rate))
        self.lead_time_orders.clear()
        initial_order_quantity = int(round(self.demand_mean))
        demand_process = (
            self._runtime_demand_process
            if not self.track_demand
            else build_demand_process(self.demand_config, rng=np.random)
        )
        for _ in range(self.lead_time):
            self.lead_time_orders.append(initial_order_quantity)
            epoch_demand = int(demand_process.sample())
            self.current_inventory_level = max(0, self.current_inventory_level - int(epoch_demand))

    def reset(self):
        self.total_cost = 0.0
        self.epoch_costs = []
        self.current_epoch = 0
        self.done = False
        self.arriving_order = 0
        if self.track_demand:
            self.horizon_demand = self.get_demand()
            self._runtime_demand_process = None
        else:
            self.horizon_demand = None
            self._runtime_demand_process = build_demand_process(self.demand_config, rng=np.random)
        self.initialize_env()
        return self.norm_state

    def is_valid_action(self, action):
        return int(action) >= 0

    def _sample_single_demand(self):
        if self._runtime_demand_process is None:
            self._runtime_demand_process = build_demand_process(self.demand_config, rng=np.random)
        return int(self._runtime_demand_process.sample())

    def get_demand(self):
        demand_process = build_demand_process(self.demand_config, rng=np.random)
        return demand_process.sample_path(self.horizon)

    def get_realized_demand(self):
        if self.horizon_demand is not None:
            return int(self.horizon_demand[self.current_epoch])
        return self._sample_single_demand()

    def get_state_dim(self):
        if self.state_features in {"pipeline", "raw_pipeline"}:
            return self.lead_time
        if self.state_features in {"pipeline_plus_summary", "raw_pipeline_plus_summary"}:
            return self.lead_time + 3
        raise NotImplementedError(f"Unknown state feature mode: {self.state_features}")

    def is_done(self):
        return self.done

    def get_epoch_cost(self, epoch_demand, order_quantity):
        epoch_cost = self.procurement_cost * int(order_quantity)
        if order_quantity > 0:
            epoch_cost += self.fixed_order_cost

        if epoch_demand < self.current_inventory_level:
            self.current_inventory_level -= int(epoch_demand)
            epoch_cost += self.current_inventory_level * self.holding_cost
        else:
            lost_sales = int(epoch_demand) - self.current_inventory_level
            epoch_cost += self.shortage_cost * lost_sales
            self.current_inventory_level = 0

        return float(epoch_cost)

    def update_lead_time_orders(self, order_quantity):
        self.arriving_order = int(self.lead_time_orders.popleft())
        self.lead_time_orders.append(int(order_quantity))
        return self.arriving_order

    @property
    def state(self):
        state = list(self.lead_time_orders)
        state[0] += self.current_inventory_level
        return state

    @property
    def norm_state(self):
        return self.policy_state

    @property
    def policy_state(self):
        scale = float(max(1.0, self.demand_mean))
        pipeline_state = np.asarray(self.state, dtype=np.float32) / scale
        raw_pipeline_state = np.asarray(self.state, dtype=np.float32)
        if self.state_features == "pipeline":
            return pipeline_state
        if self.state_features == "raw_pipeline":
            return raw_pipeline_state
        if self.state_features == "pipeline_plus_summary":
            position_scale = float(max(1.0, (self.lead_time + 1) * self.demand_mean))
            summary = np.asarray(
                [
                    self.current_inventory_level / scale,
                    sum(self.lead_time_orders) / position_scale,
                    self.inventory_position / position_scale,
                ],
                dtype=np.float32,
            )
            return np.concatenate([pipeline_state, summary]).astype(np.float32, copy=False)
        if self.state_features == "raw_pipeline_plus_summary":
            raw_summary = np.asarray(
                [
                    self.current_inventory_level,
                    sum(self.lead_time_orders),
                    self.inventory_position,
                ],
                dtype=np.float32,
            )
            return np.concatenate([raw_pipeline_state, raw_summary]).astype(np.float32, copy=False)
        raise NotImplementedError(f"Unknown state feature mode: {self.state_features}")

    @property
    def inventory_position(self):
        return self.current_inventory_level + sum(self.lead_time_orders)

    @property
    def demand_mean(self):
        return float(self.demand_config.stationary_mean)

    def step(self, order_quantity):
        order_quantity = int(order_quantity)
        if not self.is_valid_action(order_quantity):
            raise ValueError(f"Invalid order quantity {order_quantity}; expected a non-negative integer")

        arriving_orders = self.update_lead_time_orders(order_quantity)
        self.current_inventory_level += arriving_orders
        epoch_demand = self.get_realized_demand()
        epoch_cost = self.get_epoch_cost(epoch_demand=epoch_demand, order_quantity=order_quantity)

        self.epoch_costs.append(epoch_cost)
        self.total_cost += epoch_cost
        self.current_epoch += 1
        if self.current_epoch >= self.horizon:
            self.done = True

        return self.norm_state, epoch_cost, self.done

    def get_one_hot_encoded_state(self, state):
        trailing_bound = 1 + max(0, int(state[1])) if len(state) > 1 else 1
        d = self.one_hot_inventory_upper_bound + trailing_bound
        s = np.zeros((1, d))
        s[0, state[0]] = 1
        s[0, self.one_hot_inventory_upper_bound + state[1]] = 1
        return s

    @property
    def avg_total_cost(self, after_warmup=True):
        if not self.epoch_costs:
            return 0.0
        if after_warmup and self.warm_up_periods < len(self.epoch_costs):
            return float(np.mean(self.epoch_costs[self.warm_up_periods:]))
        return float(np.mean(self.epoch_costs))

    def get_demand_lower_upper_bound_Poisson(self, eps=1e-14):
        _, lb, ub = get_demand_prob_vector(
            build_demand_config(demand_dist_name="Poisson", demand_rate=self.demand_rate),
            eps=eps,
        )
        return lb, ub

    def get_demand_prob_vector_Poisson(self):
        return get_demand_prob_vector(
            build_demand_config(demand_dist_name="Poisson", demand_rate=self.demand_rate)
        )

    def get_Geometric_demand_probs_lower_upper_bound(self, eps=1e-14):
        probs = []
        success_prob = 1.0 / (1.0 + self.demand_rate)
        cumulative = 0.0
        k = 0
        while cumulative < 1 - eps:
            prob = success_prob * ((1 - success_prob) ** k)
            probs.append(prob)
            cumulative += prob
            k += 1
        probs = np.asarray(probs, dtype=np.float64)
        probs /= probs.sum()
        return probs, 0, len(probs) - 1

    def get_demand_prob_vector_Geometric(self):
        return get_demand_prob_vector(
            build_demand_config(demand_dist_name="Geometric", demand_rate=self.demand_rate)
        )

    def get_demand_prob_vector(self):
        return get_demand_prob_vector(self.demand_config)

    def get_cumulative_demand_l_L(self, k, l=0):
        periods = self.lead_time - l + 1
        return get_cumulative_demand_cdf(self.demand_config, k=k, periods=periods)

    def get_critical_fractile(self):
        return (self.procurement_cost + self.holding_cost) / (self.holding_cost + self.shortage_cost)

    @property
    def critical_fractile(self):
        return self.get_critical_fractile()

    def get_order_pipeline_partial_sum(self, state, l):
        if l == self.lead_time:
            return 0
        return int(sum(list(state)[l:]))


def build_env_from_args(args, horizon=None, track_demand=False):
    return LostSalesEnv(
        demand_rate=args.demand_rate,
        lead_time=args.lead_time,
        horizon=args.horizon if horizon is None else horizon,
        max_order_size=args.max_order_size,
        one_hot_inventory_upper_bound=getattr(args, "one_hot_inventory_upper_bound", 200),
        holding_cost=args.holding_cost,
        shortage_cost=args.shortage_cost,
        procurement_cost=getattr(args, "procurement_cost", 0.0),
        fixed_order_cost=getattr(args, "fixed_order_cost", 0.0),
        demand_dist_name=args.demand_dist_name,
        demand_lambda_low=float(getattr(args, "demand_lambda_low", DEFAULT_MMPP2_LAMBDA_LOW)),
        demand_lambda_high=float(getattr(args, "demand_lambda_high", DEFAULT_MMPP2_LAMBDA_HIGH)),
        demand_p00=float(getattr(args, "demand_p00", DEFAULT_MMPP2_POSITIVE_P00)),
        demand_p11=float(getattr(args, "demand_p11", DEFAULT_MMPP2_POSITIVE_P11)),
        track_demand=track_demand,
        warm_up_periods_ratio=getattr(args, "warm_up_periods_ratio", 0.2),
        state_features=getattr(args, "state_features", "pipeline"),
    )


def _rust_demand_kwargs(args):
    config = build_demand_config_from_args(args)
    return {
        "demand_lambda_low": float(config.demand_lambda_low),
        "demand_lambda_high": float(config.demand_lambda_high),
        "demand_p00": float(config.demand_p00),
        "demand_p11": float(config.demand_p11),
    }

def _rust_lost_sales_policy_mode(model, args, track_demand=False, return_env=False):
    if return_env or track_demand:
        return None
    if getattr(args, "rollout_backend", "python") != "rust":
        return None
    if getattr(args, "state_features", "pipeline") != "pipeline":
        return None

    model_name = type(model).__name__
    if model_name == "SoftTreePolicy":
        return "soft_tree" if str(getattr(model, "leaf_type", "constant")) == "linear" else None
    dense_rust_heads = {
        "categorical_quantity",
        "direct_quantity",
        "capped_direct_quantity",
        "sigmoid_direct_quantity",
        "soft_gated_direct_quantity",
        "gated_sigmoid_direct_quantity",
        "hard_gated_direct_quantity",
        "soft_gated_ordinal_quantity",
        "hard_gated_ordinal_quantity",
    }
    if model_name == "LinearPolicyNet" and getattr(model, "action_output_mode", None) in dense_rust_heads:
        return "linear"
    if model_name == "PolicyNet" and getattr(model, "action_output_mode", None) in dense_rust_heads:
        return "nn"
    return None


def _dense_policy_output_dim(model):
    return int(getattr(model, "policy_output_dim", getattr(model, "output_dim")))


def _rust_policy_max_quantity(model):
    bounded_dense_heads = {
        "categorical_quantity",
        "capped_direct_quantity",
        "sigmoid_direct_quantity",
        "soft_gated_direct_quantity",
        "gated_sigmoid_direct_quantity",
        "hard_gated_direct_quantity",
        "soft_gated_ordinal_quantity",
        "hard_gated_ordinal_quantity",
    }
    action_output_mode = str(getattr(model, "action_output_mode", ""))
    if action_output_mode not in bounded_dense_heads:
        return None

    policy_max_quantity = getattr(model, "max_order_size", None)
    if policy_max_quantity is None:
        raise ValueError(f"{action_output_mode} requires a policy-side quantity cap")
    return int(policy_max_quantity)


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

    rust_mode = _rust_lost_sales_policy_mode(
        model,
        args,
        track_demand=track_demand,
        return_env=return_env,
    )
    if model_params is not None and rust_mode is None:
        model.set_model_params(model_params)

    if rust_mode == "soft_tree":
        import invman_rust

        flat_params = model_params if model_params is not None else model.get_model_flat_params()
        avg_cost = invman_rust.lost_sales_soft_tree_rollout(
            flat_params=np.asarray(flat_params, dtype=np.float32).tolist(),
            input_dim=int(model.input_dim),
            depth=int(model.depth),
            split_type=str(getattr(model, "split_type", "oblique")),
            leaf_type=str(getattr(model, "leaf_type", "constant")),
            demand_rate=float(args.demand_rate),
            demand_dist_name=str(getattr(args, "demand_dist_name", "Poisson")),
            lead_time=int(args.lead_time),
            holding_cost=float(args.holding_cost),
            shortage_cost=float(args.shortage_cost),
            procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
            fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
            horizon=int(args.horizon),
            seed=int(seed),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            temperature=float(model.temperature),
            **_rust_demand_kwargs(args),
        )
        if verbose:
            print(f"Seed {seed}: avg cost {avg_cost:.4f}")
        return -float(avg_cost), indiv_idx

    if rust_mode == "linear":
        import invman_rust

        flat_params = model_params if model_params is not None else model.get_model_flat_params()
        policy_head = str(getattr(model, "action_output_mode", "categorical_quantity"))
        avg_cost = invman_rust.lost_sales_linear_rollout(
            flat_params=np.asarray(flat_params, dtype=np.float32).tolist(),
            input_dim=int(model.input_dim),
            output_dim=_dense_policy_output_dim(model),
            policy_max_quantity=_rust_policy_max_quantity(model),
            policy_head=policy_head,
            demand_rate=float(args.demand_rate),
            demand_dist_name=str(getattr(args, "demand_dist_name", "Poisson")),
            lead_time=int(args.lead_time),
            holding_cost=float(args.holding_cost),
            shortage_cost=float(args.shortage_cost),
            procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
            fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
            horizon=int(args.horizon),
            seed=int(seed),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            **_rust_demand_kwargs(args),
        )
        if verbose:
            print(f"Seed {seed}: avg cost {avg_cost:.4f}")
        return -float(avg_cost), indiv_idx

    if rust_mode == "nn":
        import invman_rust

        flat_params = model_params if model_params is not None else model.get_model_flat_params()
        policy_head = str(getattr(model, "action_output_mode", "categorical_quantity"))
        avg_cost = invman_rust.lost_sales_nn_rollout(
            flat_params=np.asarray(flat_params, dtype=np.float32).tolist(),
            input_dim=int(model.input_dim),
            hidden_dims=[int(width) for width in model.hidden_dim],
            output_dim=_dense_policy_output_dim(model),
            policy_max_quantity=_rust_policy_max_quantity(model),
            policy_head=policy_head,
            activation=str(getattr(model, "activation_name", "selu")),
            demand_rate=float(args.demand_rate),
            demand_dist_name=str(getattr(args, "demand_dist_name", "Poisson")),
            lead_time=int(args.lead_time),
            holding_cost=float(args.holding_cost),
            shortage_cost=float(args.shortage_cost),
            procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
            fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
            horizon=int(args.horizon),
            seed=int(seed),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            **_rust_demand_kwargs(args),
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
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        order_quantity = model(state_tensor)
        state, _, done = env.step(order_quantity=order_quantity)
    if verbose:
        print(f"Seed {seed}: avg cost {env.avg_total_cost:.4f}")
    if return_env:
        return -env.avg_total_cost, env
    return -env.avg_total_cost, indiv_idx


def get_population_fitness(model, args, model_params_batch, seeds):
    rust_mode = _rust_lost_sales_policy_mode(model, args, track_demand=False, return_env=False)
    if rust_mode is None:
        return None

    import invman_rust

    params_batch = [
        np.asarray(model_params, dtype=np.float32).tolist() for model_params in model_params_batch
    ]
    if rust_mode == "soft_tree":
        costs = invman_rust.lost_sales_soft_tree_population_rollout(
            params_batch=params_batch,
            input_dim=int(model.input_dim),
            depth=int(model.depth),
            split_type=str(getattr(model, "split_type", "oblique")),
            leaf_type=str(getattr(model, "leaf_type", "constant")),
            demand_rate=float(args.demand_rate),
            seeds=[int(seed) for seed in seeds],
            demand_dist_name=str(getattr(args, "demand_dist_name", "Poisson")),
            lead_time=int(args.lead_time),
            holding_cost=float(args.holding_cost),
            shortage_cost=float(args.shortage_cost),
            procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
            fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
            horizon=int(args.horizon),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            temperature=float(model.temperature),
            **_rust_demand_kwargs(args),
        )
    elif rust_mode == "linear":
        policy_head = str(getattr(model, "action_output_mode", "categorical_quantity"))
        costs = invman_rust.lost_sales_linear_population_rollout(
            params_batch=params_batch,
            input_dim=int(model.input_dim),
            output_dim=_dense_policy_output_dim(model),
            policy_max_quantity=_rust_policy_max_quantity(model),
            policy_head=policy_head,
            demand_rate=float(args.demand_rate),
            seeds=[int(seed) for seed in seeds],
            demand_dist_name=str(getattr(args, "demand_dist_name", "Poisson")),
            lead_time=int(args.lead_time),
            holding_cost=float(args.holding_cost),
            shortage_cost=float(args.shortage_cost),
            procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
            fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
            horizon=int(args.horizon),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            **_rust_demand_kwargs(args),
        )
    else:
        policy_head = str(getattr(model, "action_output_mode", "categorical_quantity"))
        costs = invman_rust.lost_sales_nn_population_rollout(
            params_batch=params_batch,
            input_dim=int(model.input_dim),
            hidden_dims=[int(width) for width in model.hidden_dim],
            output_dim=_dense_policy_output_dim(model),
            policy_max_quantity=_rust_policy_max_quantity(model),
            policy_head=policy_head,
            activation=str(getattr(model, "activation_name", "selu")),
            demand_rate=float(args.demand_rate),
            seeds=[int(seed) for seed in seeds],
            demand_dist_name=str(getattr(args, "demand_dist_name", "Poisson")),
            lead_time=int(args.lead_time),
            holding_cost=float(args.holding_cost),
            shortage_cost=float(args.shortage_cost),
            procurement_cost=float(getattr(args, "procurement_cost", 0.0)),
            fixed_order_cost=float(getattr(args, "fixed_order_cost", 0.0)),
            horizon=int(args.horizon),
            warm_up_periods_ratio=float(getattr(args, "warm_up_periods_ratio", 0.2)),
            **_rust_demand_kwargs(args),
        )
    return [(-float(cost), indiv_idx) for indiv_idx, cost in enumerate(costs)]
