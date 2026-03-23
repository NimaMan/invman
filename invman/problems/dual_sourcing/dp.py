from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class BoundedDPResult:
    average_cost: float
    best_action_by_state: dict[tuple[int, ...], tuple[int, int]]
    inventory_bounds: tuple[int, int]
    iterations: int
    max_action: tuple[int, int]

    def to_dict(self):
        return {
            "average_cost": float(self.average_cost),
            "inventory_bounds": [int(self.inventory_bounds[0]), int(self.inventory_bounds[1])],
            "iterations": int(self.iterations),
            "max_action": [int(self.max_action[0]), int(self.max_action[1])],
        }


def solve_bounded_dp(
    args,
    inventory_lower: int = -40,
    inventory_upper: int = 60,
    tolerance: float = 1e-6,
    max_iterations: int = 200,
):
    regular_lead_time = int(args.regular_lead_time)
    max_regular = int(args.regular_max_order_size)
    max_expedited = int(args.expedited_max_order_size)
    demand_values = list(range(int(args.dual_demand_low), int(args.dual_demand_high) + 1))
    demand_prob = 1.0 / len(demand_values)

    state_space = list(
        product(
            range(int(inventory_lower), int(inventory_upper) + 1),
            *[range(max_regular + 1) for _ in range(max(0, regular_lead_time - 1))],
        )
    )
    state_to_idx = {state: idx for idx, state in enumerate(state_space)}
    values = np.zeros(len(state_space), dtype=np.float64)
    policy = {}
    reference_idx = state_to_idx[state_space[0]]

    for iteration in range(1, max_iterations + 1):
        new_values = np.empty_like(values)
        new_policy = {}
        max_delta = 0.0
        for state_idx, state in enumerate(state_space):
            best_cost = float("inf")
            best_action = (0, 0)
            for regular_order in range(max_regular + 1):
                for expedited_order in range(max_expedited + 1):
                    expected_cost = 0.0
                    for demand in demand_values:
                        end_inventory = int(state[0]) + int(expedited_order) - int(demand)
                        next_first = end_inventory + (int(state[1]) if regular_lead_time > 1 else int(regular_order))
                        next_state = (
                            [next_first] + [int(value) for value in state[2:]] + ([int(regular_order)] if regular_lead_time > 1 else [])
                        )
                        next_state = tuple(
                            max(int(inventory_lower), min(int(inventory_upper), int(next_state[0])))
                            if idx == 0
                            else max(0, min(max_regular, int(next_state[idx])))
                            for idx in range(len(next_state))
                        )
                        next_idx = state_to_idx[next_state]
                        epoch_cost = (
                            float(args.regular_order_cost) * int(regular_order)
                            + float(args.expedited_order_cost) * int(expedited_order)
                            + float(args.holding_cost) * max(end_inventory, 0)
                            + float(args.shortage_cost) * max(-end_inventory, 0)
                        )
                        expected_cost += demand_prob * (epoch_cost + values[next_idx])
                    if expected_cost < best_cost:
                        best_cost = expected_cost
                        best_action = (int(regular_order), int(expedited_order))
            new_values[state_idx] = best_cost
            new_policy[state] = best_action

        baseline = float(new_values[reference_idx])
        new_values -= baseline
        max_delta = float(np.max(np.abs(new_values - values)))
        values = new_values
        policy = new_policy
        if max_delta < tolerance:
            break

    initial_state = tuple([int(round((regular_lead_time + 1) * 0.5 * (args.dual_demand_low + args.dual_demand_high)))] + [0] * max(0, regular_lead_time - 1))
    if initial_state not in policy:
        initial_state = state_space[0]
    regular_order, expedited_order = policy[initial_state]
    average_cost = 0.0
    for demand in demand_values:
        end_inventory = int(initial_state[0]) + int(expedited_order) - int(demand)
        average_cost += demand_prob * (
            float(args.regular_order_cost) * int(regular_order)
            + float(args.expedited_order_cost) * int(expedited_order)
            + float(args.holding_cost) * max(end_inventory, 0)
            + float(args.shortage_cost) * max(-end_inventory, 0)
        )
    return BoundedDPResult(
        average_cost=float(average_cost),
        best_action_by_state=policy,
        inventory_bounds=(int(inventory_lower), int(inventory_upper)),
        iterations=int(iteration),
        max_action=(int(max_regular), int(max_expedited)),
    )
