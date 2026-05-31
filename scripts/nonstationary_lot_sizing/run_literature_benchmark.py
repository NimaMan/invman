"""Literature benchmark for the nonstationary_lot_sizing problem family.

OBJECTIVE
---------
This problem is LITERATURE-VERIFIED against Dehaybe, Catanzaro & Chevalier (2024),
"Deep Reinforcement Learning for inventory optimization with non-stationary
uncertain demand", EJOR 314(2):433-445, DOI 10.1016/j.ejor.2023.10.007, and the
author code/data at HenriDeh/DRL_MMULS (single-item branch). The eight reference
rows carried in `references.rs` are byte-for-byte the author-repo testbed CSVs
(`scarf_testbed_simple_lostsales.csv` and `scarf_testbed_DP_lostsales.csv`) for
the canonical slice leadtime=2, shortage=5, setup=10, lostsales, CV=0.2,
horizon=32. This script BENCHMARKS the repo's policies against those published
rows on the eight forecast instances.

ALGORITHM (what this script computes)
-------------------------------------
For each of the eight forecast instances (constant_5/10/15, seasonal_1/2/4,
growth, decline):

  1. simple_s_s        -> CV-Normal demand (cv=0.2). Author "simple" baseline.
                          Levels: s = quantile_Normal(LTDmean, LTDstd, b/(b+h)),
                          S = s + EOQ, EOQ = sqrt(2*mean(forecast)*K/h),
                          LTDmean = sum(forecast[0..=L]), LTDstd = sqrt(sum((f_i*cv)^2)).
  2. rolling_dp_s_s    -> Poisson demand. Author DP baseline (the paper's strong
                          dynamic-programming comparator). Finite-horizon backward
                          DP on inventory position over an augmented forecast with
                          a stationary tail; first-period (s,S) levels replayed.
  3. lead_time_base_stock -> CV-Normal demand. Repo heuristic (no fixed-cost EOQ
                          batching); reported as an additional comparator.

  Each policy is simulated with `--replications` Monte-Carlo replications and
  compared to the author's published mean cost and lost-sales (shortage) rate.

  The DP baseline is the strongest available comparator (no exact optimum is
  available for the rolling-forecast path); gaps of every other policy are
  reported relative to it.

  Optionally (`--learned`), a depth-`--tree_depth` soft tree is trained per
  instance with CMA-ES against the EXPOSED Rust rollout binding
  `nonstationary_lot_sizing_soft_tree_population_rollout`, using the read-only
  `invman.cmaes.CMAES`. The soft tree maps the normalized policy state
  (forecast window, net inventory, pipeline) to a scalar order quantity
  (action_mode='scalar_quantity', clipped to [0, --action_cap]). This is the
  learned-policy comparison; it is fully self-contained and does NOT touch the
  shared training harness (which has no nonstationary_lot_sizing branch yet --
  see the README "Learned-policy blocker").

NOTES ON FIDELITY
-----------------
- Order of events per period (matches env.rs::step_state and the paper's
  Section 4.2 worked transition, reward -130):
  place order -> oldest pipeline order arrives -> demand realizes ->
  charge fixed cost K (if ordered) + holding*max(end_inv,0) + penalty*unmet.
- simple_s_s uses CV-Normal demand; rolling_dp_s_s uses Poisson demand, exactly
  as in the author testbed (the "simple" and "DP" CSVs use different demand
  models).

USAGE
-----
  python scripts/nonstationary_lot_sizing/run_literature_benchmark.py
  python scripts/nonstationary_lot_sizing/run_literature_benchmark.py --replications 25000
  python scripts/nonstationary_lot_sizing/run_literature_benchmark.py --learned --instances constant_10 seasonal_2 growth
"""

import argparse
import json
import math
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import invman_rust as ir

# ---------------------------------------------------------------------------
# Canonical slice (mirrors references.rs LOST_SALES_FORECAST_BENCHMARKS)
# leadtime=2, shortage=5, setup=10, lostsales, CV=0.2, horizon=32, periods=104.
# ---------------------------------------------------------------------------
PERIODS = 104
FORECAST_HORIZON = 32
LEAD_TIME = 2
HOLDING_COST = 1.0
SHORTAGE_COST = 5.0
FIXED_ORDER_COST = 10.0
PROCUREMENT_COST = 0.0
INITIAL_NET_INVENTORY = 20.0
DEMAND_CV = 0.2
ROLLING_DP_DISCOUNT = 0.99
ROLLING_DP_TAIL = 32

# Published author-repo rows (verbatim from references.rs / DRL_MMULS CSVs):
# forecast_id -> (name, simple_cost, simple_shortrate, dp_cost, dp_shortrate)
INSTANCES = {
    1: ("constant_5", 1252.4885126630645, 0.002257224822374979, 1215.264, 0.08371429560108733),
    2: ("constant_10", 1832.9142436489014, 0.0029443487165113735, 1711.741, 0.04793465748308879),
    3: ("constant_15", 2369.6265719327503, 0.010798230024562525, 2072.164, 0.03265778250574352),
    4: ("seasonal_1", 1824.9849305221624, 0.005102263384820955, 1675.81, 0.04499945105003535),
    5: ("seasonal_2", 1869.9015804632895, 0.00556035793112148, 1680.512, 0.04560985552056054),
    6: ("seasonal_4", 1858.1096981637254, 0.0068329782121353015, 1687.426, 0.045789060677398144),
    7: ("growth", 1754.7650626733312, 0.0016976563165351682, 1603.741, 0.05073870776319464),
    8: ("decline", 1964.4606533055787, 0.011555343257297896, 1840.866, 0.05170177110825886),
}
NAME_TO_ID = {name: fid for fid, (name, *_rest) in INSTANCES.items()}


def build_forecast_path(forecast_id: int, length: int):
    """Mirror references.rs::build_forecast_path exactly."""
    values = []
    for period in range(length):
        t = period + 1.0
        if forecast_id == 1:
            v = 5.0
        elif forecast_id == 2:
            v = 10.0
        elif forecast_id == 3:
            v = 15.0
        elif forecast_id == 4:
            v = 10.0 + 5.0 * math.sin(2.0 * math.pi * t / 104.0)
        elif forecast_id == 5:
            v = 10.0 + 5.0 * math.sin(2.0 * math.pi * 2.0 * t / 104.0)
        elif forecast_id == 6:
            v = 10.0 + 5.0 * math.sin(2.0 * math.pi * 4.0 * t / 104.0)
        elif forecast_id == 7:
            v = 5.0 + 10.0 * period / max(length - 1, 1)
        elif forecast_id == 8:
            v = 15.0 - 10.0 * period / max(length - 1, 1)
        else:
            raise ValueError(f"unknown forecast_id {forecast_id}")
        values.append(v)
    return values


def soft_tree_param_count(input_dim: int, depth: int, leaf_type: str, action_dim: int = 1) -> int:
    """Mirror core/policies/soft_tree.rs::validate_soft_tree_flat_params layout."""
    num_internal = (1 << depth) - 1
    num_leaves = 1 << depth
    weights = num_internal * input_dim
    biases = num_internal
    if leaf_type == "constant":
        leaf = num_leaves * action_dim
    elif leaf_type in ("linear", "sigmoid_linear"):
        leaf = num_leaves * action_dim * input_dim + num_leaves * action_dim
    else:
        raise ValueError(f"unknown leaf_type {leaf_type}")
    return weights + biases + leaf


def simulate_simple(forecast, replications, seed):
    return ir.nonstationary_lot_sizing_simulate_policy(
        "simple_s_s", [], forecast, FORECAST_HORIZON, INITIAL_NET_INVENTORY, [0.0] * LEAD_TIME,
        PERIODS, replications, seed, HOLDING_COST, SHORTAGE_COST, FIXED_ORDER_COST,
        "cv_normal", DEMAND_CV, PROCUREMENT_COST, True,
    )


def simulate_lead_time_base_stock(forecast, replications, seed):
    return ir.nonstationary_lot_sizing_simulate_policy(
        "lead_time_base_stock", [], forecast, FORECAST_HORIZON, INITIAL_NET_INVENTORY, [0.0] * LEAD_TIME,
        PERIODS, replications, seed, HOLDING_COST, SHORTAGE_COST, FIXED_ORDER_COST,
        "cv_normal", DEMAND_CV, PROCUREMENT_COST, True,
    )


def simulate_rolling_dp(forecast, replications, seed):
    return ir.nonstationary_lot_sizing_simulate_rolling_dp_policy(
        forecast, FORECAST_HORIZON, INITIAL_NET_INVENTORY, [0.0] * LEAD_TIME, PERIODS,
        replications, seed, HOLDING_COST, SHORTAGE_COST, FIXED_ORDER_COST,
        "poisson", 0.0, PROCUREMENT_COST, True, ROLLING_DP_DISCOUNT, ROLLING_DP_TAIL,
    )


def train_soft_tree(forecast, depth, leaf_type, action_cap, generations, popsize, seed):
    """Self-contained CMA-ES soft-tree training against the exposed rollout binding.

    Returns (best_params, train_mean_cost). Uses CV-Normal demand to match the
    'simple' author column (so the learned policy and simple baseline are on the
    same demand model). Trains on the FULL 104-period rolling-forecast path
    (no warm-up truncation) to mirror the benchmark objective.
    """
    from invman.cmaes import CMAES

    input_dim = FORECAST_HORIZON + 1 + LEAD_TIME
    num_params = soft_tree_param_count(input_dim, depth, leaf_type)
    min_values = [0]
    max_values = [int(action_cap)]

    optimizer = CMAES(num_params=num_params, sigma_init=0.5, popsize=popsize, seed=seed)
    best_params = None
    best_cost = float("inf")
    for generation in range(generations):
        population = optimizer.ask()  # shape (popsize, num_params)
        params_batch = [row.astype("float32").tolist() for row in population]
        # Common-random-number seeds: one fixed seed per candidate keeps the
        # objective comparable across the batch within a generation.
        seeds = [seed + 1 + i for i in range(len(params_batch))]
        costs = ir.nonstationary_lot_sizing_soft_tree_population_rollout(
            params_batch, input_dim, depth, min_values, max_values, "scalar_quantity",
            forecast, FORECAST_HORIZON, INITIAL_NET_INVENTORY, [0.0] * LEAD_TIME,
            HOLDING_COST, SHORTAGE_COST, FIXED_ORDER_COST, PERIODS, seeds,
            "cv_normal", DEMAND_CV, PROCUREMENT_COST, True, 0.0, 0.25, "oblique", leaf_type, None,
        )
        # CMAES.tell maximizes the supplied reward; rollout returns mean PERIOD
        # cost (lower is better), so hand it the negative.
        optimizer.tell([-c for c in costs])
        gen_best = min(costs)
        if gen_best < best_cost:
            best_cost = gen_best
            best_idx = costs.index(gen_best)
            best_params = params_batch[best_idx]
    return best_params, best_cost


def evaluate_soft_tree(forecast, params, depth, leaf_type, action_cap, replications, seed):
    """Total-cost evaluation (mean PERIOD cost * PERIODS) over replications."""
    input_dim = FORECAST_HORIZON + 1 + LEAD_TIME
    min_values = [0]
    max_values = [int(action_cap)]
    total = 0.0
    for r in range(replications):
        mean_period_cost = ir.nonstationary_lot_sizing_soft_tree_rollout(
            params, input_dim, depth, min_values, max_values, "scalar_quantity",
            forecast, FORECAST_HORIZON, INITIAL_NET_INVENTORY, [0.0] * LEAD_TIME,
            HOLDING_COST, SHORTAGE_COST, FIXED_ORDER_COST, PERIODS, seed + r,
            "cv_normal", DEMAND_CV, PROCUREMENT_COST, True, 0.0, 0.25, "oblique", leaf_type, None,
        )
        total += mean_period_cost * PERIODS
    return total / replications


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=25000,
                        help="Monte-Carlo replications for the literature reproduction (author used 25000 for simple).")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--instances", nargs="*", default=None,
                        help="Subset of instance names (default: all 8).")
    parser.add_argument("--learned", action="store_true",
                        help="Also train+evaluate a CMA-ES soft tree per instance.")
    parser.add_argument("--tree_depth", type=int, default=2)
    parser.add_argument("--leaf_type", default="linear", choices=["constant", "linear", "sigmoid_linear"])
    parser.add_argument("--action_cap", type=float, default=80.0)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--learned_replications", type=int, default=2000)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    selected = args.instances if args.instances else [name for _, (name, *_r) in sorted(INSTANCES.items())]
    rows = []
    print(f"# nonstationary_lot_sizing literature benchmark "
          f"(L={LEAD_TIME}, b={SHORTAGE_COST}, K={FIXED_ORDER_COST}, h={HOLDING_COST}, "
          f"cv={DEMAND_CV}, H={FORECAST_HORIZON}, T={PERIODS}, reps={args.replications})")
    header = (f"{'instance':<13}{'simple':>9}{'pub_smpl':>9}{'Δ%':>7} | "
              f"{'dp(cmp)':>9}{'pub_dp':>9}{'Δ%':>7} | {'ltbs':>9}{'gap_dp%':>8}")
    if args.learned:
        header += f" | {'learned':>9}{'gap_dp%':>8}"
    print(header)

    for name in selected:
        fid = NAME_TO_ID[name]
        _n, pub_simple, pub_simple_sr, pub_dp, pub_dp_sr = INSTANCES[fid]
        forecast = build_forecast_path(fid, PERIODS + FORECAST_HORIZON)

        s_cost, s_std, s_sr = simulate_simple(forecast, args.replications, args.seed)
        dp_cost, dp_std, dp_sr = simulate_rolling_dp(forecast, args.replications, args.seed)
        lt_cost, lt_std, lt_sr = simulate_lead_time_base_stock(forecast, args.replications, args.seed)

        simple_dpct = 100.0 * (s_cost - pub_simple) / pub_simple
        dp_dpct = 100.0 * (dp_cost - pub_dp) / pub_dp
        lt_gap_dp = 100.0 * (lt_cost - dp_cost) / dp_cost

        row = {
            "instance": name,
            "simple_repro_cost": s_cost, "simple_pub_cost": pub_simple, "simple_pct_diff": simple_dpct,
            "simple_repro_shortrate": s_sr, "simple_pub_shortrate": pub_simple_sr,
            "dp_repro_cost": dp_cost, "dp_pub_cost": pub_dp, "dp_pct_diff": dp_dpct,
            "dp_repro_shortrate": dp_sr, "dp_pub_shortrate": pub_dp_sr,
            "lead_time_base_stock_cost": lt_cost, "lead_time_base_stock_gap_vs_dp_pct": lt_gap_dp,
        }
        line = (f"{name:<13}{s_cost:>9.1f}{pub_simple:>9.1f}{simple_dpct:>7.2f} | "
                f"{dp_cost:>9.1f}{pub_dp:>9.1f}{dp_dpct:>7.2f} | {lt_cost:>9.1f}{lt_gap_dp:>8.2f}")

        if args.learned:
            params, train_cost = train_soft_tree(
                forecast, args.tree_depth, args.leaf_type, args.action_cap,
                args.generations, args.popsize, args.seed)
            learned_cost = evaluate_soft_tree(
                forecast, params, args.tree_depth, args.leaf_type, args.action_cap,
                args.learned_replications, args.seed + 99)
            learned_gap_dp = 100.0 * (learned_cost - dp_cost) / dp_cost
            row["learned_soft_tree_cost"] = learned_cost
            row["learned_soft_tree_gap_vs_dp_pct"] = learned_gap_dp
            row["learned_soft_tree_train_period_cost"] = train_cost
            line += f" | {learned_cost:>9.1f}{learned_gap_dp:>8.2f}"

        print(line)
        rows.append(row)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
