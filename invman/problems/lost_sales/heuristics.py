"""
Reference heuristics for the classic lost-sales problem:
- Myopic-1
- Myopic-2
- Standard vector base-stock (SVBS)
"""

import numpy as np

from invman.problems.lost_sales.env import LostSalesEnv, build_env_from_args


class LostSalesHeuristicPolicies:
    def __init__(self, env: LostSalesEnv):
        self.env = env
        self.one_period_cache = {}
        self.q_l_cache = {}
        self.q_L_cache = {}
        self.m2_cache = {}
        self._svbs_levels = None

    @property
    def demand_support(self):
        return np.arange(self.env.demand_lb, self.env.demand_ub + 1)

    def get_x_plus_l_1(self, x, demand, l):
        next_x = list(x[:-1])
        next_x[0] = max(0, int(x[0]) - int(demand)) + int(x[1])
        for i in range(1, l):
            next_x[i] = int(x[i + 1])
        return next_x

    def get_one_period_cost(self, y):
        if y in self.one_period_cache:
            return self.one_period_cache[y]

        support = self.demand_support
        overage = np.maximum(y - support, 0)
        underage = np.maximum(support - y, 0)
        expected_overage = np.dot(self.env.demand_probs, overage)
        expected_underage = np.dot(self.env.demand_probs, underage)
        total_cost = (
            self.env.procurement_cost * y
            + self.env.holding_cost * expected_overage
            + self.env.shortage_cost * expected_underage
        )
        self.one_period_cache[y] = float(total_cost)
        return float(total_cost)

    def get_q_l(self, x, l):
        key = (l, *x)
        if key in self.q_l_cache:
            return self.q_l_cache[key]

        if l == 0:
            value = self.get_one_period_cost(int(x[0]))
        else:
            value = 0.0
            for idx, demand in enumerate(self.demand_support):
                x_plus_l_1 = self.get_x_plus_l_1(x=x, demand=demand, l=l)
                value += self.get_q_l(x=x_plus_l_1, l=l - 1) * self.env.demand_probs[idx]
            value = self.env.gamma * value

        self.q_l_cache[key] = float(value)
        return float(value)

    def get_Q_L_x_L_from_state(self, state, order_quantity):
        key = (order_quantity, *state)
        if key in self.q_L_cache:
            return self.q_L_cache[key]

        x_L = list(state) + [int(order_quantity)]
        q_value = 0.0
        for idx, demand in enumerate(self.demand_support):
            x_plus_l_1 = self.get_x_plus_l_1(x=x_L, demand=demand, l=self.env.lead_time)
            q_value += self.get_q_l(x=x_plus_l_1, l=self.env.lead_time - 1) * self.env.demand_probs[idx]

        q_value = float(self.env.gamma * q_value)
        self.q_L_cache[key] = q_value
        return q_value

    def _best_quantity(self, evaluator, state, return_value=False):
        best_quantity = 0
        current_value = evaluator(state, 0)
        previous_value = np.inf

        while best_quantity + 1 < self.env.action_space_dim and previous_value > current_value:
            best_quantity += 1
            previous_value = current_value
            current_value = evaluator(state, best_quantity)

        if previous_value > current_value:
            chosen_quantity = best_quantity
            chosen_value = current_value
        else:
            chosen_quantity = max(0, best_quantity - 1)
            chosen_value = previous_value

        if return_value:
            return int(chosen_quantity), float(chosen_value)
        return int(chosen_quantity)

    def get_myopic_1_order_quantity(self, state, return_qhat=False):
        return self._best_quantity(self.get_Q_L_x_L_from_state, state, return_value=return_qhat)

    def get_myopic_2_q_L_x_L(self, state, order_quantity):
        key = (order_quantity, *state)
        if key in self.m2_cache:
            return self.m2_cache[key]

        x_L = list(state) + [int(order_quantity)]
        q_hat_z = self.get_Q_L_x_L_from_state(state, order_quantity)
        future_value = 0.0
        for idx, demand in enumerate(self.demand_support):
            x_plus = self.get_x_plus_l_1(x=x_L, demand=demand, l=self.env.lead_time)
            _, q_hat = self.get_myopic_1_order_quantity(x_plus, return_qhat=True)
            future_value += q_hat * self.env.demand_probs[idx]

        value = float(q_hat_z + self.env.gamma * future_value)
        self.m2_cache[key] = value
        return value

    def get_myopic_2_order_quantity(self, state, return_qhat=False):
        return self._best_quantity(self.get_myopic_2_q_L_x_L, state, return_value=return_qhat)

    def get_order_pipeline_partial_sum(self, l, state):
        if l == self.env.lead_time:
            return 0
        return int(sum(list(state)[l:]))

    def get_standard_vector_base_stock_policy(self):
        if self._svbs_levels is not None:
            return self._svbs_levels

        sbar = np.zeros(self.env.lead_time + 1, dtype=int)
        for l in range(self.env.lead_time + 1):
            s = 0
            while (1 - self.env.get_cumulative_demand_l_L(k=s, l=l)) >= self.env.critical_fractile:
                s += 1
            sbar[l] = s

        self._svbs_levels = sbar
        return sbar

    def get_standard_vector_base_stock_policy_order_quantity(self, state):
        z_x = np.zeros(self.env.lead_time + 1, dtype=int)
        sbar = self.get_standard_vector_base_stock_policy()

        for l in range(self.env.lead_time + 1):
            v_l = self.get_order_pipeline_partial_sum(l=l, state=state)
            z_x[l] = int(sbar[l] - v_l)

        order_quantity = max(0, int(np.min(z_x)))
        return min(order_quantity, self.env.max_order_size)


def get_heuristic_policy_cost(args, env=None, heuristic="myopic1", seed=1234):
    np.random.seed(getattr(args, "seed", seed))

    if env is None:
        env = build_env_from_args(args, track_demand=getattr(args, "track_demand", False))
    elif not isinstance(env, LostSalesEnv):
        raise TypeError("env must be a LostSalesEnv instance")

    policy = LostSalesHeuristicPolicies(env=env)
    heuristic_name = heuristic.lower()
    done = False
    state_action = {}
    while not done:
        state_key = tuple(env.state)
        if state_key not in state_action:
            if heuristic_name == "myopic1":
                state_action[state_key] = policy.get_myopic_1_order_quantity(env.state)
            elif heuristic_name == "myopic2":
                state_action[state_key] = policy.get_myopic_2_order_quantity(env.state)
            elif heuristic_name in {"svbs", "standard_vector_base_stock"}:
                state_action[state_key] = policy.get_standard_vector_base_stock_policy_order_quantity(env.state)
            else:
                raise NotImplementedError(f"Unknown heuristic: {heuristic}")

        _, _, done = env.step(order_quantity=state_action[state_key])

    return env, -env.avg_total_cost, state_action
