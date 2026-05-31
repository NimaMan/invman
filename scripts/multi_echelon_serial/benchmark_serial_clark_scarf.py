"""
Benchmark the serial Clark-Scarf multi-echelon problem
(rust/src/problems/multi_echelon/serial).

OBJECTIVE
---------
Compare inventory policies on the literature-verified serial multi-echelon instance
set, against the exact Clark-Scarf optimum:

  1. OPTIMAL  : the exact echelon base-stock policy (Clark & Scarf 1960), whose levels
                and cost come from the exact recursive newsvendor decomposition. This is
                the known-optimal policy class for the serial system.
  2. HEURISTIC: simpler base-stock heuristics that a learned policy / practitioner would
                be compared against:
                  - "lead_time_mean_cover": each echelon orders up to its own lead-time
                    demand mean (echelon level = mean * cumulative lead time). A naive,
                    no-safety-stock target.
                  - "newsvendor_per_echelon": each echelon sets its echelon base-stock to
                    the single-echelon newsvendor quantile of its lead-time demand using
                    the GLOBAL critical ratio p/(p+H_tot) (ignores the Clark-Scarf
                    induced-penalty coupling between stages).

WHY THIS IS A FAITHFUL PORT (not the installed Rust)
----------------------------------------------------
The serial env (env.rs) is NOT exposed to Python in the installed invman_rust build
(there is no `serial_*` binding and serial/bindings.rs does not exist; the
`multi_echelon_*` functions belong to production_assembly_distribution_network). So a
learned soft-tree rollout on the serial env cannot be run without rebuilding Rust.

This script therefore re-implements env.rs (`consume`/`replenish` + the echelon
base-stock evaluator) and exact.rs (the recursive newsvendor decomposition) in Python,
faithfully, line-for-line. It was validated to reproduce the Rust solver / env:
  - exact optima reproduce stockpyl.ssm_serial AND the repo exact.rs test values
    (Poisson N=1/2/3: 4.220849, 16.797779, 72.043543; Normal Ex6.1: 47.6654 ~ 47.65);
  - the env simulation under the optimal levels reproduces those optima to <=0.18%
    (Poisson) and exactly (Normal, with continuous demand).

KNOWN ENV LIMITATION (carried from env.rs docstring, independently confirmed here)
---------------------------------------------------------------------------------
The multi-stage env reproduces the exact optimum only when the demand-facing (most
downstream) stage has lead time 1 -- true for ALL carried verification instances. For a
demand-facing stage with lead time >= 2 the env UNDER-COUNTS cost (e.g. 2-stage,
downstream L=2: sim ~20.1 vs exact ~25.1, a ~20% under-count), because the env charges
installation holding on physical on-hand only and does not charge the downstream-echelon
in-transit pipeline. Instances with downstream L>=2 are therefore excluded from the
benchmarked set and flagged here.

NORMAL-DEMAND EVALUATOR NOTE
----------------------------
The repo evaluator (echelon_base_stock.rs::simulate) ROUNDS Normal demand to integers,
while the exact solver uses continuous Normal. That rounding biases the simulated cost of
the Normal Ex6.1 instance up ~1.6% (48.4 vs 47.67). This script reports BOTH the rounded
(repo-faithful) and continuous simulations so the bias is explicit.

USAGE
-----
  python scripts/multi_echelon_serial/benchmark_serial_clark_scarf.py

There are no external dependencies beyond numpy/scipy. Set --periods / --seeds to change
the Monte-Carlo budget.
"""

import argparse
import math
from collections import deque

import numpy as np
from scipy.stats import norm, poisson


# --------------------------------------------------------------------------------------
# Exact Clark-Scarf recursive newsvendor decomposition (faithful port of exact.rs).
# Convention: stages indexed downstream -> upstream, k = 0..N-1; echelon holding h[k].
# --------------------------------------------------------------------------------------
TAIL_SIGMAS = 4.0
INVENTORY_POINTS = 1000


def _normal_ltd(mean, std, periods, demand_points=100):
    if periods == 0 or std <= 0.0:
        return np.array([mean * periods]), np.array([1.0])
    m = mean * periods
    s = std * math.sqrt(periods)
    lo = max(m - TAIL_SIGMAS * s, 0.0)
    hi = m + TAIL_SIGMAS * s
    n = demand_points
    delta = (hi - lo) / n
    support = np.array([lo + i * delta for i in range(n + 1)])
    pmf = np.empty(n + 1)
    for i in range(n + 1):
        upper = 1.0 if i == n else norm.cdf(support[i] + delta * 0.5, m, s)
        lower = 0.0 if i == 0 else norm.cdf(support[i] - delta * 0.5, m, s)
        pmf[i] = upper - lower
    return support, pmf


def _poisson_ltd(mean, periods):
    if periods == 0 or mean <= 0.0:
        return np.array([mean * periods]), np.array([1.0])
    m = mean * periods
    hi = _poisson_upper(m)
    support = np.arange(hi + 1, dtype=float)
    pmf = poisson.pmf(np.arange(hi + 1), m)
    pmf[hi] = max(1.0 - poisson.cdf(hi - 1, m), 0.0)
    return support, pmf


def _poisson_upper(mean):
    cap = int(math.ceil(mean + 12.0 * math.sqrt(mean))) + 20
    c = 0.0
    for k in range(cap + 1):
        c += poisson.pmf(k, mean)
        if c >= 1.0 - 1e-12:
            return k
    return cap


def solve_serial_clark_scarf(echelon_holding, lead_times, penalty, demand_kind,
                             demand_mean, demand_std):
    """Exact solver. echelon_holding/lead_times in downstream->upstream order.
    Returns (echelon_levels[d->u], optimal_cost)."""
    n = len(echelon_holding)
    h = list(echelon_holding)
    h_total = sum(h)
    mean = demand_mean
    std = demand_std if demand_kind == "normal" else math.sqrt(demand_mean)
    sum_l = sum(lead_times)
    discrete = demand_kind == "poisson"

    if discrete:
        m = mean * sum_l
        hi = float(_poisson_upper(max(m, 1e-9)))
        x_lo, x_delta, x_num = -hi, 1.0, int(round(2 * hi))
    else:
        m = mean * sum_l
        s = std * math.sqrt(sum_l)
        lo = m - TAIL_SIGMAS * s
        hi = m + TAIL_SIGMAS * s
        x_lo = lo - hi
        x_num = INVENTORY_POINTS
        x_delta = (hi - x_lo) / x_num
    x = np.array([x_lo + i * x_delta for i in range(x_num + 1)])

    l_prefix = [0] * (n + 1)
    for t in range(n):
        l_prefix[t + 1] = l_prefix[t] + lead_times[t]

    def chat_linear_below(k, v):
        sum_l_below = l_prefix[k]
        val = -(penalty + h_total) * (v - mean * sum_l_below)
        for kp in range(k + 1):
            inner = l_prefix[k] - l_prefix[kp]
            val += h[kp] * (v - mean * inner)
        return val

    def nearest(value):
        raw = round((value - x_lo) / x_delta)
        return int(min(max(raw, 0), x_num))

    c_bar_prev = (penalty + h_total) * np.maximum(-x, 0.0)
    echelon_levels = [0.0] * n
    optimal_cost = 0.0

    for k in range(n):
        c_hat = h[k] * x + c_bar_prev
        support, pmf = (_poisson_ltd(mean, lead_times[k]) if discrete
                        else _normal_ltd(mean, std, lead_times[k]))

        def chat_at(v):
            if v < x_lo:
                return chat_linear_below(k, v)
            return c_hat[nearest(v)]

        c_k = np.empty(len(x))
        for i, y in enumerate(x):
            exp = 0.0
            for d, pr in zip(support, pmf):
                if pr == 0.0:
                    continue
                exp += pr * chat_at(y - d)
            c_k[i] = exp

        best_idx = int(np.argmin(c_k))
        echelon_levels[k] = float(x[best_idx])
        optimal_cost = float(c_k[best_idx])
        s_star = x[best_idx]
        c_bar_prev = np.array([c_k[nearest(min(xi, s_star))] for xi in x])

    return echelon_levels, optimal_cost


# --------------------------------------------------------------------------------------
# Env port (faithful to env.rs): receive -> demand -> cost(post-demand) -> replenish.
# holding/lead in downstream->upstream order; echelon_levels downstream->upstream.
# --------------------------------------------------------------------------------------
def _echelon_ip(on_hand, pipeline, backorder):
    n = len(on_hand)
    ip = [0.0] * n
    po = pp = 0.0
    for k in range(n):
        po += on_hand[k]
        pp += sum(pipeline[k])
        ip[k] = po + pp - backorder
    return ip


def simulate_echelon_base_stock(holding, lead, penalty, demand_kind, demand_mean,
                                demand_std, echelon_levels, periods, warm_up, seed,
                                round_normal=True):
    rng = np.random.default_rng(seed)
    n = len(holding)
    on_hand = [echelon_levels[0] if k == 0 else echelon_levels[k] - echelon_levels[k - 1]
               for k in range(n)]
    pipeline = [deque([demand_mean] * lead[k]) for k in range(n)]
    backorder = 0.0
    tot = hold_acc = bo_acc = 0.0
    cnt = 0
    for t in range(periods):
        if demand_kind == "poisson":
            d = float(rng.poisson(demand_mean))
        else:
            raw = rng.normal(demand_mean, demand_std)
            d = max(round(raw), 0.0) if round_normal else max(raw, 0.0)
        # consume
        for k in range(n):
            on_hand[k] += pipeline[k].popleft() if pipeline[k] else 0.0
        need = d + backorder
        shipped = min(on_hand[0], need)
        on_hand[0] -= shipped
        backorder = need - shipped
        hold = sum(holding[k] * max(on_hand[k], 0.0) for k in range(n))
        bo = penalty * backorder
        period_cost = hold + bo
        # replenish (echelon base-stock orders from post-demand state)
        ip = _echelon_ip(on_hand, pipeline, backorder)
        orders = [max(echelon_levels[k] - ip[k], 0.0) for k in range(n)]
        for k in reversed(range(n)):
            if k == n - 1:
                ship = max(orders[k], 0.0)
            else:
                ship = min(max(orders[k], 0.0), on_hand[k + 1])
                on_hand[k + 1] -= ship
            pipeline[k].append(ship)
        if t >= warm_up:
            tot += period_cost
            hold_acc += hold
            bo_acc += bo
            cnt += 1
    return tot / cnt, hold_acc / cnt, bo_acc / cnt


# --------------------------------------------------------------------------------------
# Heuristic echelon-level designs (the policies the optimal is benchmarked against).
# --------------------------------------------------------------------------------------
def heuristic_lead_time_mean_cover(echelon_holding, lead_times, demand_mean):
    """Echelon level = mean demand over cumulative lead time (no safety stock)."""
    n = len(echelon_holding)
    cum_l = 0
    levels = []
    for k in range(n):
        cum_l += lead_times[k]
        levels.append(demand_mean * cum_l)
    return levels


def heuristic_newsvendor_per_echelon(echelon_holding, lead_times, penalty, demand_kind,
                                     demand_mean, demand_std):
    """Each echelon's level = newsvendor quantile of its OWN cumulative-lead-time demand
    using the global critical ratio p/(p+H_tot). Ignores Clark-Scarf coupling."""
    n = len(echelon_holding)
    h_total = sum(echelon_holding)
    cr = penalty / (penalty + h_total)
    std = demand_std if demand_kind == "normal" else math.sqrt(demand_mean)
    cum_l = 0
    levels = []
    for k in range(n):
        cum_l += lead_times[k]
        if demand_kind == "poisson":
            levels.append(float(poisson.ppf(cr, demand_mean * cum_l)))
        else:
            m = demand_mean * cum_l
            s = std * math.sqrt(cum_l)
            levels.append(float(norm.ppf(cr, m, s)))
    return levels


# --------------------------------------------------------------------------------------
# Benchmark instance set: the carried verification instances (downstream L=1 only).
# echelon_holding / lead_times are downstream->upstream.
# --------------------------------------------------------------------------------------
INSTANCES = [
    dict(name="poisson_N1", kind="poisson", mean=5.0, std=0.0,
         echelon_holding=[1.0], lead=[1], penalty=9.0, published=4.220849),
    dict(name="poisson_N2", kind="poisson", mean=5.0, std=0.0,
         echelon_holding=[2.0, 1.0], lead=[1, 1], penalty=10.0, published=16.797779),
    dict(name="poisson_N3", kind="poisson", mean=5.0, std=0.0,
         echelon_holding=[3.0, 2.0, 2.0], lead=[1, 1, 2], penalty=37.12,
         published=72.043543),
    dict(name="normal_ex6_1", kind="normal", mean=5.0, std=1.0,
         echelon_holding=[3.0, 2.0, 2.0], lead=[1, 1, 2], penalty=37.12,
         published=47.65),
]


def echelon_to_installation(echelon_holding):
    """Installation (local) holding down->up from echelon holding down->up.

    exact.rs uses echelon h_k = H_k - H_{k+1} (H_N = 0), so inversely the installation
    holding at stage k is the echelon cost at stage k plus all UPSTREAM echelon costs:
        H[k] = h[k] + h[k+1] + ... + h[n-1]   (downstream->upstream indices).
    Verified: echelon [2,1] -> [3,1]; [3,2,2] -> [7,4,2]; [1] -> [1]."""
    n = len(echelon_holding)
    return [sum(echelon_holding[k:]) for k in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", type=int, default=400_000)
    ap.add_argument("--warmup", type=int, default=5_000)
    ap.add_argument("--seeds", type=int, nargs="*", default=[3, 17, 21])
    args = ap.parse_args()

    print("=" * 96)
    print("SERIAL CLARK-SCARF BENCHMARK  (faithful Python port of env.rs + exact.rs)")
    print(f"periods={args.periods}  warmup={args.warmup}  seeds={args.seeds}")
    print("=" * 96)

    for inst in INSTANCES:
        h = inst["echelon_holding"]
        lead = inst["lead"]
        p = inst["penalty"]
        kind = inst["kind"]
        mean = inst["mean"]
        std = inst["std"]
        installation = echelon_to_installation(h)

        # Exact optimum.
        opt_levels, opt_cost = solve_serial_clark_scarf(h, lead, p, kind, mean, std)

        # Heuristic level designs.
        heur = {
            "OPTIMAL (Clark-Scarf)": opt_levels,
            "newsvendor_per_echelon": heuristic_newsvendor_per_echelon(
                h, lead, p, kind, mean, std),
            "lead_time_mean_cover": heuristic_lead_time_mean_cover(h, lead, mean),
        }

        print(f"\n[{inst['name']}] {kind} demand mean={mean} std={std}; "
              f"echelon h(d->u)={h} lead(d->u)={lead} penalty={p}")
        print(f"  exact optimum: C* = {opt_cost:.4f}  "
              f"(published {inst['published']})  "
              f"S*(d->u) = {[round(x, 3) for x in opt_levels]}")
        rounded = (kind == "normal")
        print(f"  {'policy':24s} {'levels(d->u)':28s} {'sim_cost':>10s} {'gap_vs_opt%':>12s}")
        for label, levels in heur.items():
            costs = [simulate_echelon_base_stock(
                installation, lead, p, kind, mean, std, levels,
                args.periods, args.warmup, s, round_normal=rounded)[0]
                for s in args.seeds]
            mc = float(np.mean(costs))
            gap = (mc - opt_cost) / opt_cost * 100.0
            lv = "[" + ",".join(f"{x:.2f}" for x in levels) + "]"
            print(f"  {label:24s} {lv:28s} {mc:10.4f} {gap:+11.2f}%")
        if rounded:
            # Show the continuous (un-rounded) optimal-policy sim to expose the
            # evaluator rounding bias against the continuous-Normal exact optimum.
            cont = [simulate_echelon_base_stock(
                installation, lead, p, kind, mean, std, opt_levels,
                args.periods, args.warmup, s, round_normal=False)[0]
                for s in args.seeds]
            mc = float(np.mean(cont))
            print(f"  {'OPTIMAL (continuous dem)':24s} "
                  f"{'[same as OPTIMAL]':28s} {mc:10.4f} "
                  f"{(mc - opt_cost) / opt_cost * 100.0:+11.2f}%   "
                  f"<- un-rounded; matches exact, isolates the evaluator rounding bias")

    print("\n" + "=" * 96)
    print("EXCLUDED (env KNOWN LIMITATION): instances with most-downstream lead time >= 2.")
    print("  The env under-counts cost there (e.g. 2-stage downstream L=2: sim ~20.1 vs")
    print("  exact ~25.1, ~20% under). Resolve the in-transit-holding convention first.")
    print("=" * 96)

    print("\nTO ADD A LEARNED SOFT-TREE COMPARISON (requires Rust rebuild + new binding):")
    print("  blocker: no `serial_*` Python binding exists in the installed invman_rust;")
    print("  serial/bindings.rs is absent and serial is not registered in")
    print("  multi_echelon/bindings.rs. Once a `serial_soft_tree_rollout_from_demands`")
    print("  (mirroring lost_sales_soft_tree_rollout_from_demands) is added and the")
    print("  extension is rebuilt, drop it into the `heur` dict above keyed by the")
    print("  trained params and re-run; the exact optimum here is the reference floor.")


if __name__ == "__main__":
    main()
