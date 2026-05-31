#!/usr/bin/env python3
"""Independent (non-Rust) verification of the multi_echelon/assembly problem.

OBJECTIVE
---------
The assembly problem's correctness rests on two claims:
  (1) Rosling (1989): an equal-component-lead-time assembly system is EXACTLY a
      2-stage serial system (kit -> finished), so the exact optimum comes from the
      literature-verified Clark-Scarf serial solver.
  (2) The assembly env.rs simulation, driven by those optimal echelon base-stock
      levels, reproduces that exact serial optimum (within Monte-Carlo error).

The repo verifies these in Rust unit tests (assembly/verification.rs) that this
agent is not allowed to run (cargo test forbidden, no Python binding exists for
assembly). This script INDEPENDENTLY reimplements both the Clark-Scarf exact solver
(faithful to serial/exact.rs) and the assembly env (faithful to assembly/env.rs),
in pure Python/NumPy, and runs the three verification instances from
assembly/verification.rs plus the published serial anchors. If this independent
reimplementation reproduces the published numbers and the env matches the exact
optimum, the assembly verification claim holds independently of the Rust tests.

It does NOT import invman_rust for assembly (no such binding exists); it is a
from-scratch check of the math.
"""

import math
import numpy as np
from scipy.stats import norm, poisson


# ---------------------------------------------------------------------------
# Faithful reimplementation of serial/exact.rs (Clark-Scarf recursive newsvendor)
# ---------------------------------------------------------------------------

DEFAULT_TAIL_SIGMAS = 4.0
DEFAULT_INVENTORY_POINTS = 1000
DEFAULT_DEMAND_POINTS = 100


def poisson_upper_support(mean):
    cap = math.ceil(mean + 12.0 * math.sqrt(mean)) + 20
    cumulative = 0.0
    for k in range(0, cap + 1):
        cumulative += poisson.pmf(k, mean)
        if cumulative >= 1.0 - 1e-12:
            return k
    return cap


def normal_lead_time_demand(mean, std, periods, tail_sigmas, demand_points):
    if periods == 0 or std <= 0.0:
        return np.array([mean * periods]), np.array([1.0])
    m = mean * periods
    s = std * math.sqrt(periods)
    lo = max(m - tail_sigmas * s, 0.0)
    hi = m + tail_sigmas * s
    n = demand_points
    delta = (hi - lo) / n
    support = []
    pmf = []
    for i in range(0, n + 1):
        d = lo + i * delta
        upper = 1.0 if i == n else norm.cdf(d + delta * 0.5, m, s)
        lower = 0.0 if i == 0 else norm.cdf(d - delta * 0.5, m, s)
        support.append(d)
        pmf.append(upper - lower)
    return np.array(support), np.array(pmf)


def poisson_lead_time_demand(mean, periods):
    if periods == 0 or mean <= 0.0:
        return np.array([mean * periods]), np.array([1.0])
    m = mean * periods
    hi = poisson_upper_support(m)
    support = []
    pmf = []
    accumulated = 0.0
    for k in range(0, hi + 1):
        p = max(1.0 - accumulated, 0.0) if k == hi else poisson.pmf(k, m)
        accumulated += poisson.pmf(k, m)
        support.append(float(k))
        pmf.append(p)
    return np.array(support), np.array(pmf)


def lead_time_demand(demand, periods, tail_sigmas, demand_points):
    if demand[0] == "normal":
        _, mean, std = demand
        return normal_lead_time_demand(mean, std, periods, tail_sigmas, demand_points)
    else:
        _, mean = demand
        return poisson_lead_time_demand(mean, periods)


def nearest_index(value, x_lo, x_delta, x_num):
    raw = round((value - x_lo) / x_delta)
    if raw <= 0:
        return 0
    elif raw >= x_num:
        return x_num
    else:
        return int(raw)


def solve_serial_clark_scarf(echelon_h, lead_times, penalty, demand,
                             tail_sigmas=DEFAULT_TAIL_SIGMAS,
                             inventory_points=DEFAULT_INVENTORY_POINTS,
                             demand_points=DEFAULT_DEMAND_POINTS):
    """echelon_h, lead_times in downstream -> upstream order. Mirrors exact.rs."""
    n = len(echelon_h)
    h = list(echelon_h)
    h_total = sum(h)
    if demand[0] == "normal":
        mean = demand[1]; std = demand[2]
    else:
        mean = demand[1]; std = math.sqrt(demand[1])
    sum_l = sum(lead_times)
    discrete = demand[0] == "poisson"

    if discrete:
        m = mean * sum_l
        hi = float(poisson_upper_support(max(m, 1e-9)))
        x_lo = -hi
        x_hi = hi
        x_delta = 1.0
        x_num = int(round(x_hi - x_lo))
    else:
        m = mean * sum_l
        s = std * math.sqrt(sum_l)
        lo = m - tail_sigmas * s
        hi = m + tail_sigmas * s
        x_lo = lo - hi
        x_hi = hi
        x_num = inventory_points
        x_delta = (x_hi - x_lo) / x_num

    x = np.array([x_lo + i * x_delta for i in range(0, x_num + 1)])

    l_prefix = [0] * (n + 1)
    for t in range(n):
        l_prefix[t + 1] = l_prefix[t] + lead_times[t]

    def c_hat_linear_below_grid(k, v):
        sum_l_below = float(l_prefix[k])
        value = -(penalty + h_total) * (v - mean * sum_l_below)
        for kp in range(0, k + 1):
            inner = float(l_prefix[k] - l_prefix[kp])
            value += h[kp] * (v - mean * inner)
        return value

    c_bar_prev = np.array([(penalty + h_total) * max(-xi, 0.0) for xi in x])

    echelon_levels = [0.0] * n
    optimal_cost = 0.0

    for k in range(n):
        c_hat = h[k] * x + c_bar_prev
        ltd_support, ltd_pmf = lead_time_demand(demand, lead_times[k], tail_sigmas, demand_points)

        def chat_at(v):
            if v < x_lo:
                return c_hat_linear_below_grid(k, v)
            else:
                return c_hat[nearest_index(v, x_lo, x_delta, x_num)]

        c_k = np.zeros(len(x))
        for i, y in enumerate(x):
            expected = 0.0
            for d, prob in zip(ltd_support, ltd_pmf):
                if prob == 0.0:
                    continue
                expected += prob * chat_at(y - d)
            c_k[i] = expected

        best_idx = int(np.argmin(c_k))
        echelon_levels[k] = float(x[best_idx])
        optimal_cost = float(c_k[best_idx])

        s_star = x[best_idx]
        c_bar_prev = np.array(
            [c_k[nearest_index(min(xi, s_star), x_lo, x_delta, x_num)] for xi in x]
        )

    return echelon_levels, optimal_cost


def solve_from_local_costs(local_ud, lead_ud, penalty, demand, **kw):
    """local holding + lead times in upstream->downstream order. Mirrors exact.rs."""
    n = len(local_ud)
    local_du = list(reversed(local_ud))
    lead_du = list(reversed(lead_ud))
    echelon_h = []
    for k in range(n):
        upstream = local_du[k + 1] if k + 1 < n else 0.0
        echelon_h.append(local_du[k] - upstream)
    return solve_serial_clark_scarf(echelon_h, lead_du, penalty, demand, **kw)


# ---------------------------------------------------------------------------
# Faithful reimplementation of assembly/env.rs + rosling.rs + echelon_base_stock.rs
# ---------------------------------------------------------------------------

class AssemblyConfig:
    def __init__(self, component_holding_costs, component_lead_time,
                 finished_holding_cost, finished_lead_time, penalty):
        self.component_holding_costs = list(component_holding_costs)
        self.component_lead_time = component_lead_time
        self.finished_holding_cost = finished_holding_cost
        self.finished_lead_time = finished_lead_time
        self.penalty = penalty

    @property
    def num_components(self):
        return len(self.component_holding_costs)

    @property
    def kit_holding_cost(self):
        return sum(self.component_holding_costs)


def reduce_equal_lead_time(config):
    """rosling.rs: assembly -> equivalent serial. Returns (local_ud, lead_ud, penalty)."""
    kit_holding = config.kit_holding_cost
    assert config.finished_holding_cost + 1e-12 >= kit_holding, \
        f"finished holding {config.finished_holding_cost} must be >= kit holding {kit_holding}"
    local_holding_ud = [kit_holding, config.finished_holding_cost]
    lead_ud = [config.component_lead_time, config.finished_lead_time]
    return local_holding_ud, lead_ud, config.penalty


class AssemblyState:
    def __init__(self, config, echelon_levels, demand_mean):
        s_finished = echelon_levels[0]
        s_kit = echelon_levels[1]
        kit_local = max(s_kit - s_finished, 0.0)
        m = config.num_components
        self.component_on_hand = [kit_local] * m
        from collections import deque
        self.component_pipeline = [deque([demand_mean] * config.component_lead_time) for _ in range(m)]
        self.finished_on_hand = s_finished
        self.finished_pipeline = deque([demand_mean] * config.finished_lead_time)
        self.backorder = 0.0


def consume(config, state, demand):
    """env.rs consume: receive, demand, cost (post-demand, pre-replenish)."""
    m = config.num_components
    for k in range(m):
        arrival = state.component_pipeline[k].popleft() if state.component_pipeline[k] else 0.0
        state.component_on_hand[k] += arrival
    finished_arrival = state.finished_pipeline.popleft() if state.finished_pipeline else 0.0
    state.finished_on_hand += finished_arrival

    need = demand + state.backorder
    shipped = min(state.finished_on_hand, need)
    state.finished_on_hand -= shipped
    state.backorder = need - shipped

    holding = config.finished_holding_cost * max(state.finished_on_hand, 0.0)
    for k in range(m):
        holding += config.component_holding_costs[k] * max(state.component_on_hand[k], 0.0)
    backorder_cost = config.penalty * state.backorder
    return holding, backorder_cost, holding + backorder_cost


def replenish(config, state, echelon_levels):
    """env.rs replenish: echelon base-stock with levels [S_finished, S_kit]."""
    s_finished = echelon_levels[0]
    s_kit = echelon_levels[1]
    m = config.num_components

    finished_in_transit = sum(state.finished_pipeline)
    finished_ip = state.finished_on_hand + finished_in_transit - state.backorder
    component_ip = []
    for k in range(m):
        comp_in_transit = sum(state.component_pipeline[k])
        component_ip.append(
            state.component_on_hand[k] + comp_in_transit + state.finished_on_hand
            + finished_in_transit - state.backorder
        )

    desired_assembly = max(s_finished - finished_ip, 0.0)
    kits_available = min(state.component_on_hand)
    assembled = min(desired_assembly, kits_available)
    for k in range(m):
        state.component_on_hand[k] -= assembled
    state.finished_pipeline.append(assembled)

    for k in range(m):
        order = max(s_kit - component_ip[k], 0.0)
        state.component_pipeline[k].append(order)


def simulate_assembly(config, demand, echelon_levels, periods, warm_up, seed):
    """echelon_base_stock.rs simulate."""
    if demand[0] == "normal":
        mean = demand[1]; std = demand[2]
    else:
        mean = demand[1]
    state = AssemblyState(config, echelon_levels, mean)
    rng = np.random.default_rng(seed)

    total = holding_t = backorder_t = 0.0
    counted = 0
    for t in range(periods):
        if demand[0] == "poisson":
            d = float(rng.poisson(max(mean, 1e-9)))
        else:
            d = max(round(rng.normal(mean, std)), 0.0)
        holding, backorder_cost, period_cost = consume(config, state, d)
        replenish(config, state, echelon_levels)
        if t >= warm_up:
            total += period_cost
            holding_t += holding
            backorder_t += backorder_cost
            counted += 1
    return total / counted, holding_t / counted, backorder_t / counted, counted


def rel_err(a, b):
    return abs(a - b) / b


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_serial_anchor():
    print("=== Serial Clark-Scarf solver anchor (must reproduce published numbers) ===")
    # Snyder & Shen Example 6.1: local h [2,4,7] up->down, lead [2,1,1], p=37.12, Normal(5,1)
    lv, cost = solve_from_local_costs([2.0, 4.0, 7.0], [2, 1, 1], 37.12, ("normal", 5.0, 1.0))
    print(f"Example 6.1 (Normal): cost={cost:.4f} (published 47.65), levels={['%.2f'%x for x in lv]}, relerr={rel_err(cost,47.65):.4%}")
    # stockpyl Poisson 1-stage: echelon h=1, L=1, p=9 -> C*=4.220849, S*=8
    lv1, c1 = solve_serial_clark_scarf([1.0], [1], 9.0, ("poisson", 5.0))
    print(f"Poisson 1-stage: cost={c1:.6f} (stockpyl 4.220849), S*={lv1}")
    # stockpyl Poisson 2-stage: echelon h=[2,1] down->up, L=[1,1], p=10 -> 16.797779, S*=[7,13]
    lv2, c2 = solve_serial_clark_scarf([2.0, 1.0], [1, 1], 10.0, ("poisson", 5.0))
    print(f"Poisson 2-stage: cost={c2:.6f} (stockpyl 16.797779), S*={lv2}")
    # stockpyl Poisson 3-stage: local [2,4,7] up->down, L=[2,1,1], p=37.12 -> 72.043543, S*=[9,15,26]
    lv3, c3 = solve_from_local_costs([2.0, 4.0, 7.0], [2, 1, 1], 37.12, ("poisson", 5.0))
    print(f"Poisson 3-stage: cost={c3:.6f} (stockpyl 72.043543), S*={lv3}")
    print()


def check_assembly_instance(name, config, demand, seed, periods=400_000, warm_up=5_000):
    assert config.finished_lead_time == 1, "verified scope: finished lead time 1"
    local_ud, lead_ud, penalty = reduce_equal_lead_time(config)
    levels, exact_cost = solve_from_local_costs(local_ud, lead_ud, penalty, demand)
    avg, hold, back, counted = simulate_assembly(config, demand, levels, periods, warm_up, seed)
    err = rel_err(avg, exact_cost)
    ok = err < 0.02
    print(f"[{name}] serial-equiv local_ud={local_ud} lead_ud={lead_ud} p={penalty}")
    print(f"    exact optimal cost = {exact_cost:.4f}, levels [S_fin,S_kit] = {['%.2f'%x for x in levels]}")
    print(f"    env sim avg cost   = {avg:.4f} (holding {hold:.3f} + backorder {back:.3f})")
    print(f"    rel error          = {err:.4%}  -> {'PASS (<2%)' if ok else 'FAIL'}")
    print()
    return ok


def main():
    check_serial_anchor()
    print("=== Assembly env reproduces Rosling serial optimum (verification.rs instances) ===")
    results = []
    # Test 1: 2 components (h=1 each -> kit 2), L_c=1; finished h=3, L_a=1; p=10; Poisson(5)
    c1 = AssemblyConfig([1.0, 1.0], 1, 3.0, 1, 10.0)
    results.append(check_assembly_instance("2-comp Poisson", c1, ("poisson", 5.0), 3))
    # Test 2: 3 components (kit 3), L_c=2; finished h=7, L_a=1; p=37.12; Poisson(5)
    c2 = AssemblyConfig([1.0, 1.0, 1.0], 2, 7.0, 1, 37.12)
    results.append(check_assembly_instance("3-comp L_c=2 Poisson", c2, ("poisson", 5.0), 7))
    # Test 3: heterogeneous components [0.5,1.5] -> kit 2, L_c=2; finished h=4, L_a=1; p=20; Poisson(4)
    c3 = AssemblyConfig([0.5, 1.5], 2, 4.0, 1, 20.0)
    results.append(check_assembly_instance("heterogeneous comp Poisson", c3, ("poisson", 4.0), 11))

    print("=== SUMMARY ===")
    print(f"Serial anchor: see numbers above (should match published to ~machine precision)")
    print(f"Assembly env-vs-exact instances: {sum(results)}/{len(results)} PASS")
    if all(results):
        print("RESULT: Assembly verification claim INDEPENDENTLY CONFIRMED.")
    else:
        print("RESULT: At least one assembly instance FAILED independent reproduction.")


if __name__ == "__main__":
    main()
