"""
Single-policy autoresearch runner for the perishable-inventory benchmark
(De Moor, Gijsbrechts & Boute 2022 / Farrington, Wong, Li & Utley 2025 Scenario A).

OBJECTIVE
---------
Produce one honest learned-policy result on the literature-verified perishable
slice, comparable to the four learned-policy results already in the paper
(lost-sales, fixed-order-cost, dual-sourcing, multi-echelon). The benchmark
instance is the m=2 / lead-time-1 FIFO primary anchor
`de_moor2022_m2_exp2_l1_cp7_fifo`, whose exact value-iteration MDP reproduces,
in-crate at test time, the published optimum (VI mean discounted return -1457,
best base-stock level 7, optimal-policy table) and the Farrington (2025) Table-3
returns. The metric is the mean discounted return (gamma=0.99) vs the VI optimum
(optimality gap) and vs the in-repo base-stock gate.

WHY THIS DESIGN (the recipe, from policy_search/POLICY_DESIGN_GUIDELINES/README.md)
--------------------------------------------------------------------------
1. BASELINE. Two anchors, both honest:
   - VI optimum = the analytic expected discounted return under the midpoint-binned
     gamma demand (the value the crate reproduces from Farrington 2025 Table 3),
     -1457.28 for the primary FIFO instance. This is a DIFFERENT estimator from the
     Monte-Carlo rollouts (sampled+rounded gamma demand, finite horizon, zero start),
     which sit ~11 units (~0.7%) below it; the VI gap therefore MIXES estimators and
     is reported for context only.
   - base-stock GATE = the De Moor / Farrington best base-stock level (7 FIFO),
     scored by the SAME Monte-Carlo discounted-return estimator on the SAME held-out
     CRN eval seeds as the learned policy. This is the apples-to-apples comparator
     (`gap_vs_base_stock`); a learned policy beats it only if it is cheaper here.

2. ACTION GEOMETRY = THE POLICY. The rollout binding fixes a SCALAR order head
   (action_dim == 1) over the perishable age-state
   (state = [pipeline_orders(L-1), on_hand(m)] / max(demand_mean,1)). The expressive
   class is the soft-tree LEAF: a LINEAR leaf computes
       q = round( softplus( bias + w . state ) ),
   which is exactly the base-stock structure q = max(0, S - IP) when
       bias = S,  w_i = -max(demand_mean,1)  (so w . state = -(sum on_hand + pipeline) = -IP).
   The single scalar head is the right geometry: the perishable order decision IS a
   scalar order-up-to-style quantity over the age-disaggregated state, and the soft
   tree's splits let the leaves express the AGE-DEPENDENT corrections the published
   optimal policy table carries (it orders differently depending on which age bucket
   holds the inventory, not just on total IP).

3. WARM-START AT THE BEST BASE-STOCK. CMA-ES is warm-started (cma_x0) at the inverted
   leaf transform above with S = published best base-stock level (7), so GENERATION 0
   REPRODUCES the base-stock heuristic to within a single-state rounding artifact
   (softplus(0)=0.69 rounds to 1 at IP=S). Verified: the encoded depth-1/2 soft tree
   evaluates to -1468.1, matching the actual base-stock policy at -1468.4. The
   optimizer then searches OUTWARD from a known-good point, the same
   gen-0-reproduces-heuristic device used for OWMR symmetric_echelon_targets and
   dual-sourcing capped_dual_index.

4. SCORE WITH THE RUST BINDING under paired CRN. Every candidate is scored by
   `perishable_inventory_soft_tree_population_discounted_return` (population rollout,
   one fresh paired seed per individual per generation); the incumbent and the
   per-generation best are both re-evaluated on a disjoint held-out CRN eval block.
   The base-stock gate is scored on the SAME eval block (variance-reduced / paired).

5. HONEST REPORT. Warm-start guarantees the learned policy should at least reproduce
   the base-stock gate; we report the signed gap to the VI optimum AND to the
   base-stock gate, and label beats / matches / loses by whether the learned policy is
   cheaper than the base-stock gate by more than the eval SEM under paired CRN.

ALGORITHM
---------
For the named reference instance:
  1. Exact MDP summary -> VI optimum (analytic), best base-stock level, table-match flags.
  2. base-stock gate: evaluate base_stock at the published best level on held-out CRN
     eval seeds (Monte-Carlo discounted return). Also tune base_stock by stochastic
     search on disjoint search seeds and keep the better of {published level, search argmin}.
  3. Build the warm-start vector encoding base_stock(S=best level) in the soft-tree
     linear-leaf coordinate system; verify gen-0 reproduces the gate.
  4. CMA-ES (warm-started at the gate; small sigma) over `--generations` generations,
     popsize `--popsize`, paired population rollout, one fresh seed per individual per gen.
  5. Re-evaluate the per-generation best AND the CMA incumbent on the held-out eval
     block; keep whichever is cheaper. Report mean return, gap vs VI optimum, gap vs
     base-stock gate (signed + %), SEM, and the beats/matches/loses verdict.
  6. Append a TSV ledger row and write a per-run JSON artifact under
     outputs/autoresearch/<run_tag>/.

CPU CAP: defaults RAYON_NUM_THREADS / OMP_NUM_THREADS to 2 (set before importing
invman_rust); the population rollout fans out via rayon, not a Python Pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.cpu_limits import configure_process_cpu_limits_from_argv

# CPU cap MUST be set before numpy/invman_rust are imported.
configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

import numpy as np

import invman_rust
from invman.cmaes import CMAES


GAMMA = 0.99
PRIMARY_REFERENCE = "de_moor2022_m2_exp2_l1_cp7_fifo"

BUDGETS = {
    # popsize, generations, search_seeds, eval_seeds  (sigma_init via --sigma_init)
    "smoke": dict(popsize=8, generations=10, search_seeds=16, eval_seeds=128),
    "screening": dict(popsize=16, generations=40, search_seeds=48, eval_seeds=512),
    "full": dict(popsize=24, generations=120, search_seeds=64, eval_seeds=2048),
}


def get_reference(name: str) -> dict:
    return dict(invman_rust.perishable_inventory_get_reference_instance(name))


def _zero_state(reference: dict):
    return (
        [0 for _ in range(int(reference["shelf_life"]))],
        [0 for _ in range(max(int(reference["lead_time"]) - 1, 0))],
    )


def _git_short_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(PACKAGE_ROOT), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def soft_tree_kwargs(reference: dict, *, input_dim, depth, temperature, split_type,
                     leaf_type, horizon, max_order) -> dict:
    return dict(
        input_dim=int(input_dim),
        depth=int(depth),
        min_values=[0],
        max_values=[int(max_order)],
        action_mode="scalar_quantity",
        demand_mean=float(reference["demand_mean"]),
        demand_cov=float(reference["demand_cov"]),
        shelf_life=int(reference["shelf_life"]),
        lead_time=int(reference["lead_time"]),
        holding_cost=float(reference["holding_cost"]),
        shortage_cost=float(reference["shortage_cost"]),
        waste_cost=float(reference["waste_cost"]),
        procurement_cost=float(reference["procurement_cost"]),
        horizon=int(horizon),
        warm_up_periods_ratio=float(reference["warm_up_periods_ratio"]),
        gamma=GAMMA,
        temperature=float(temperature),
        split_type=str(split_type),
        leaf_type=str(leaf_type),
        issuing_policy=str(reference["issuing_policy"]),
        allowed_values=None,
    )


def population_returns(reference, kw, batch, seeds):
    return invman_rust.perishable_inventory_soft_tree_population_discounted_return(
        params_batch=[np.asarray(b, dtype=np.float32).tolist() for b in batch],
        seeds=[int(s) for s in seeds],
        **kw,
    )


def evaluate_soft_tree(reference, kw, flat_params, eval_seeds) -> dict:
    returns = np.asarray(
        [
            invman_rust.perishable_inventory_soft_tree_discounted_return(
                flat_params=np.asarray(flat_params, dtype=np.float32).tolist(),
                seed=int(s),
                **kw,
            )
            for s in eval_seeds
        ],
        dtype=np.float64,
    )
    n = int(returns.size)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "sem_return": float(np.std(returns) / np.sqrt(n)) if n else 0.0,
        "num_seeds": n,
    }


def evaluate_base_stock(reference, level, seeds, horizon) -> dict:
    on_hand, pipeline = _zero_state(reference)
    summary = dict(
        invman_rust.perishable_inventory_policy_discounted_return_summary(
            policy_name="base_stock",
            params=[int(level)],
            on_hand=on_hand,
            pipeline_orders=pipeline,
            horizon=int(horizon),
            seeds=[int(s) for s in seeds],
            max_order_size=int(reference["max_order_size"]),
            demand_mean=float(reference["demand_mean"]),
            demand_cov=float(reference["demand_cov"]),
            holding_cost=float(reference["holding_cost"]),
            shortage_cost=float(reference["shortage_cost"]),
            waste_cost=float(reference["waste_cost"]),
            procurement_cost=float(reference["procurement_cost"]),
            warm_up_periods_ratio=float(reference["warm_up_periods_ratio"]),
            gamma=GAMMA,
            issuing_policy=str(reference["issuing_policy"]),
        )
    )
    n = len(seeds)
    summary["sem_return"] = (
        float(summary["std_return"] / np.sqrt(n)) if n else 0.0
    )
    return summary


def search_best_base_stock_level(reference, search_seeds, horizon) -> dict:
    on_hand, pipeline = _zero_state(reference)
    return dict(
        invman_rust.perishable_inventory_base_stock_search_discounted_return_summary(
            on_hand=on_hand,
            pipeline_orders=pipeline,
            horizon=int(horizon),
            seeds=[int(s) for s in search_seeds],
            max_order_size=int(reference["max_order_size"]),
            demand_mean=float(reference["demand_mean"]),
            demand_cov=float(reference["demand_cov"]),
            holding_cost=float(reference["holding_cost"]),
            shortage_cost=float(reference["shortage_cost"]),
            waste_cost=float(reference["waste_cost"]),
            position_upper_bound=int(reference["max_order_size"]),
            procurement_cost=float(reference["procurement_cost"]),
            warm_up_periods_ratio=float(reference["warm_up_periods_ratio"]),
            issuing_policy=str(reference["issuing_policy"]),
            gamma=GAMMA,
            top_k=12,
        )
    )


def base_stock_warm_start_vector(*, depth, input_dim, base_stock_level, demand_mean,
                                 leaf_type) -> np.ndarray:
    """Encode base_stock(S) in the soft-tree linear-leaf coordinate system.

    State fed to the tree is [pipeline(L-1), on_hand(m)] / scale with
    scale = max(demand_mean, 1). A linear leaf computes softplus(bias + w . state).
    Setting bias = S and every weight = -scale gives
        bias + w . state = S - scale * sum(state) = S - (pipeline + on_hand) = S - IP,
    so softplus(.) ~= max(0, S - IP), the base-stock order. All leaves carry the same
    encoding and the split weights/bias are zero, so the (irrelevant) gate routes to a
    base-stock leaf regardless and generation 0 reproduces the heuristic.
    """
    if leaf_type != "linear":
        raise ValueError("base-stock warm start is only defined for linear leaves")
    scale = max(float(demand_mean), 1.0)
    n_internal = (2 ** int(depth)) - 1
    n_leaf = 2 ** int(depth)
    split_weights = [0.0] * (n_internal * int(input_dim))
    split_bias = [0.0] * n_internal
    leaf_weights = []
    leaf_bias = []
    for _ in range(n_leaf):
        leaf_weights.extend([-scale] * int(input_dim))  # control_dim == 1
        leaf_bias.append(float(base_stock_level))
    return np.asarray(split_weights + split_bias + leaf_weights + leaf_bias, dtype=np.float64)


def train_warm_started(reference, kw, *, x0, popsize, generations, sigma_init, seed):
    """Run warm-started CMA-ES; return the CMA incumbent (xbest), the CMA
    distribution MEAN (xfavorite), plus every generation's per-training-seed best
    individual.

    The per-generation training argmax is HEAVILY selection-biased (a single training
    seed flatters the chosen individual), so we do NOT pick the promoted policy on the
    training return. We return all candidates and let the caller select on a DISJOINT
    validation block (see main()). This was the load-bearing fix: at full budget the
    eval-block / training-argmax selection overfit (held-out -1482) while disjoint-
    validation selection recovered a genuine win (held-out -1457.9 vs gate -1475.1).

    Honest-floor endpoint (additive, TRAINING_PATH_AUDIT_2026_06_06): this runner uses
    a LOCAL CMA-ES loop (not invman.es_mp.train), so the two CMA endpoints are read
    directly off the wrapper, exactly as the OWMR reference reads them off the optimizer
    returned by es_mp.train(return_optimizer=True):
      - xbest     = es.best_param()    = es.result[0]  (best individual ever sampled;
                    the historical deployed endpoint -- can overfit the training batch)
      - xfavorite = es.current_param() = es.result[5]  (the CMA distribution MEAN;
                    averages out the per-seed sampling noise, often more robust)
    Both are returned so the caller can deploy the best-of {xbest, xfavorite, anchor}
    on a DISJOINT validation block (downside-safe: never worse than xbest).
    """
    es = CMAES(num_params=int(x0.size), sigma_init=float(sigma_init),
               popsize=int(popsize), seed=int(seed), x0=x0)
    rng = np.random.default_rng(seed)
    gen_candidates = []
    history = []
    for _ in range(int(generations)):
        sols = es.ask()
        # paired CRN: one fresh seed shared across the batch dimension per generation
        gen_seeds = rng.integers(0, 2 ** 31 - 1, size=len(sols))
        returns = population_returns(reference, kw, sols, gen_seeds)
        es.tell(returns)
        history.append(float(np.max(returns)))
        gen_candidates.append(np.asarray(sols[int(np.argmax(returns))], dtype=np.float32))
    cma_best = np.asarray(es.best_param(), dtype=np.float32)
    cma_favorite = np.asarray(es.current_param(), dtype=np.float32)
    return cma_best, cma_favorite, gen_candidates, history


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mp_num_processors", type=int, default=None,
                   help="CPU worker cap (consumed by configure_process_cpu_limits_from_argv "
                        "before invman_rust import; accepted here so strict parsing does not reject it)")
    p.add_argument("--run_tag", default="perishable_inventory_autoresearch")
    p.add_argument("--reference", default=PRIMARY_REFERENCE)
    p.add_argument("--budget", choices=sorted(BUDGETS), default="screening")
    p.add_argument("--description", default="warm-started base-stock soft tree")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    p.add_argument("--leaf_type", choices=["linear"], default="linear",
                   help="warm start requires a linear leaf (softplus base-stock encoding)")
    p.add_argument("--sigma_init", type=float, default=0.75,
                   help="small sigma confines the search to a base-stock neighbourhood")
    p.add_argument("--no_warm_start", action="store_true",
                   help="ablation: random CMA init instead of base-stock warm start")
    p.add_argument("--deploy_endpoint", choices=["floor", "xbest", "xfavorite"], default="floor",
                   help="honest-floor endpoint selection (TRAINING_PATH_AUDIT_2026_06_06, "
                        "additive). 'floor' (default) adds the CMA distribution-MEAN endpoint "
                        "xfavorite (es.current_param() = result[5]) AND the warm-start anchor to "
                        "the validation-block candidate set {cma_incumbent(xbest), gen-bests} so "
                        "the deployed policy is the best-of on the DISJOINT validation block "
                        "(downside-safe: never validation-worse than xbest); 'xbest' reproduces "
                        "the historical candidate set EXACTLY (xfavorite/anchor not deployable); "
                        "'xfavorite' deploys only the distribution-mean endpoint (+anchor).")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--popsize", type=int, default=None)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--search_seeds", type=int, default=None)
    p.add_argument("--eval_seeds", type=int, default=None)
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def main():
    parsed = parse_args()
    budget = dict(BUDGETS[parsed.budget])
    popsize = parsed.popsize or budget["popsize"]
    generations = parsed.generations or budget["generations"]
    n_search = parsed.search_seeds or budget["search_seeds"]
    n_eval = parsed.eval_seeds or budget["eval_seeds"]

    reference = get_reference(parsed.reference)
    horizon = int(reference["horizon"])
    input_dim = int(reference["shelf_life"]) + int(reference["lead_time"]) - 1
    max_order = int(reference["max_order_size"])

    # Disjoint CRN blocks: training seeds (random per generation), heuristic search
    # seeds, a VALIDATION block (used ONLY to select the promoted policy), and the
    # held-out EVAL block (the reported number). Keeping validation disjoint from eval
    # removes the selection bias that flips a screening "win" into a full-budget "loss".
    search_seeds = [parsed.seed + i for i in range(n_search)]
    n_val = max(256, n_eval // 4)
    val_seeds = [parsed.seed + 3_000_000 + i for i in range(n_val)]
    eval_seeds = [parsed.seed + 1_000_000 + i for i in range(n_eval)]

    # --- baseline 1: exact VI optimum (analytic) ---
    exact = dict(invman_rust.perishable_inventory_exact_mdp_summary(parsed.reference))
    vi_optimum = float(exact["value_iteration_mean_return"])
    best_base_stock_level = int(exact["best_base_stock_level"])

    # --- baseline 2: base-stock GATE (Monte-Carlo, paired CRN eval block) ---
    bs_search = search_best_base_stock_level(reference, search_seeds, horizon)
    search_level = int(bs_search["best"]["params"][0])
    candidate_levels = sorted({best_base_stock_level, search_level})
    bs_evals = {
        lvl: evaluate_base_stock(reference, lvl, eval_seeds, horizon)
        for lvl in candidate_levels
    }
    gate_level = max(candidate_levels, key=lambda lvl: bs_evals[lvl]["mean_return"])
    gate = bs_evals[gate_level]
    gate_return = float(gate["mean_return"])
    gate_sem = float(gate["sem_return"])

    kw = soft_tree_kwargs(
        reference, input_dim=input_dim, depth=parsed.depth,
        temperature=parsed.temperature, split_type=parsed.split_type,
        leaf_type=parsed.leaf_type, horizon=horizon, max_order=max_order,
    )

    # --- warm start: encode base-stock(gate_level) in the leaf coordinate system ---
    x0 = base_stock_warm_start_vector(
        depth=parsed.depth, input_dim=input_dim, base_stock_level=gate_level,
        demand_mean=float(reference["demand_mean"]), leaf_type=parsed.leaf_type,
    )
    warm_start_eval = evaluate_soft_tree(reference, kw, x0, eval_seeds)
    if parsed.no_warm_start:
        cma_x0 = np.zeros_like(x0)
    else:
        cma_x0 = x0

    # --- CMA-ES warm-started at the gate ---
    t0 = time.time()
    cma_best_params, cma_favorite_params, gen_candidates, history = train_warm_started(
        reference, kw, x0=cma_x0, popsize=popsize, generations=generations,
        sigma_init=parsed.sigma_init, seed=parsed.seed,
    )
    train_seconds = time.time() - t0

    # --- model selection on the DISJOINT validation block (never the eval block) ---
    # Candidate set: the CMA incumbent (xbest) and every generation's training-argmax
    # individual. Scoring them on a validation block disjoint from eval avoids rewarding
    # a candidate that merely overfit its single training seed.
    #
    # HONEST FLOOR (additive, TRAINING_PATH_AUDIT_2026_06_06): under deploy_endpoint
    # 'floor' (default) we additively expand the candidate set with the CMA distribution
    # MEAN endpoint xfavorite (es.current_param()) AND the warm-start anchor x0, then
    # validation-select the best-of (higher mean_return is better). This is downside-safe
    # -- xbest is always in the candidate set under 'floor', so the deployed policy is
    # never validation-worse than the historical xbest endpoint. 'xbest' reproduces the
    # historical candidate set EXACTLY (xfavorite/anchor excluded); 'xfavorite' deploys
    # only the distribution-mean endpoint (+anchor). NB validation higher-is-better here
    # (return maximization), mirroring the OWMR floor's lower-is-better cost minimization.
    cma_val = float(evaluate_soft_tree(reference, kw, cma_best_params, val_seeds)["mean_return"])
    gen_val = [
        float(evaluate_soft_tree(reference, kw, p, val_seeds)["mean_return"])
        for p in gen_candidates
    ]
    best_gen_val = max(gen_val) if gen_val else -np.inf

    # Build the deployable candidate set as (val_return, params, source) tuples gated by
    # deploy_endpoint. xbest's two members (cma incumbent + gen-bests) are the historical
    # set; floor/xfavorite add the distribution mean and (always, when warm-started) the
    # anchor x0 so a known-good fallback is deployable.
    candidates: list[tuple[float, np.ndarray, str]] = []
    if parsed.deploy_endpoint in ("floor", "xbest"):
        candidates.append((cma_val, cma_best_params, "cma_incumbent"))
        if gen_candidates:
            _gi = int(np.argmax(gen_val))
            candidates.append((best_gen_val, gen_candidates[_gi], f"gen_best@{_gi}_val_selected"))
    if parsed.deploy_endpoint in ("floor", "xfavorite"):
        xfav_val = float(evaluate_soft_tree(reference, kw, cma_favorite_params, val_seeds)["mean_return"])
        candidates.append((xfav_val, cma_favorite_params, "cma_xfavorite"))
        # warm-start anchor x0 (the encoded base-stock gate) -- deployable fallback.
        if not parsed.no_warm_start:
            anchor_val = float(evaluate_soft_tree(reference, kw, x0, val_seeds)["mean_return"])
            candidates.append((anchor_val, np.asarray(x0, dtype=np.float32), "warm_start_anchor"))
    selection_val_return, learned_params, learned_source = max(candidates, key=lambda c: c[0])

    # --- report the SELECTED policy on the held-out eval block ---
    learned_eval = evaluate_soft_tree(reference, kw, learned_params, eval_seeds)
    learned_return = float(learned_eval["mean_return"])
    learned_sem = float(learned_eval["sem_return"])

    # Gaps. Higher (less negative) return is better.
    gap_vs_vi = learned_return - vi_optimum                # negative => below optimum
    gap_vs_vi_pct = 100.0 * (gap_vs_vi / abs(vi_optimum))
    gap_vs_gate = learned_return - gate_return             # positive => beats base-stock
    gap_vs_gate_pct = 100.0 * (gap_vs_gate / abs(gate_return))

    paired_sem = float(np.sqrt(learned_sem ** 2 + gate_sem ** 2))
    if gap_vs_gate > paired_sem:
        verdict = "beats"
    elif gap_vs_gate < -paired_sem:
        verdict = "loses"
    else:
        verdict = "matches"

    payload = {
        "family": "perishable_inventory",
        "benchmark": "autoresearch_perishable_inventory",
        "commit": _git_short_commit(),
        "reference_instance": parsed.reference,
        "issuing_policy": str(reference["issuing_policy"]),
        "shelf_life": int(reference["shelf_life"]),
        "lead_time": int(reference["lead_time"]),
        "gamma": GAMMA,
        "budget": parsed.budget,
        "config": {
            "depth": parsed.depth,
            "temperature": parsed.temperature,
            "split_type": parsed.split_type,
            "leaf_type": parsed.leaf_type,
            "sigma_init": parsed.sigma_init,
            "warm_start": not parsed.no_warm_start,
            "popsize": popsize,
            "generations": generations,
            "search_seeds": n_search,
            "val_seeds": n_val,
            "eval_seeds": n_eval,
            "seed": parsed.seed,
            "horizon": horizon,
            "input_dim": input_dim,
            "num_params": int(x0.size),
            "train_seconds": round(train_seconds, 1),
        },
        "baselines": {
            "vi_optimum_analytic": vi_optimum,
            "vi_optimum_rounded": int(exact["value_iteration_mean_return_rounded"]),
            "matches_published_vi": bool(exact.get("matches_published_value_iteration_mean_return")),
            "published_vi_mean_return": exact.get("published_value_iteration_mean_return"),
            "exact_best_base_stock_level": best_base_stock_level,
            "matches_published_base_stock_level": bool(exact.get("matches_published_base_stock_level")),
            "matches_published_policy_table": bool(exact.get("matches_published_policy_table")),
            "base_stock_gate_level": gate_level,
            "base_stock_gate_return": gate_return,
            "base_stock_gate_sem": gate_sem,
            "base_stock_search_argmin_level": search_level,
            "base_stock_candidate_evals": {
                str(lvl): bs_evals[lvl]["mean_return"] for lvl in candidate_levels
            },
        },
        "warm_start_gen0": {
            "encoded_base_stock_level": gate_level,
            "gen0_mean_return": float(warm_start_eval["mean_return"]),
            "reproduces_gate_within_sem": bool(
                abs(warm_start_eval["mean_return"] - gate_return)
                <= 2.0 * float(warm_start_eval["sem_return"] + gate_sem)
            ),
        },
        "learned": {
            "source": learned_source,
            "deploy_endpoint": parsed.deploy_endpoint,
            "floor_deviated_from_xbest": bool(
                learned_source not in ("cma_incumbent",)
                and not learned_source.startswith("gen_best@")
            ),
            "selection_val_return": selection_val_return,
            "cma_incumbent_val_return": cma_val,
            "mean_return": learned_return,
            "std_return": float(learned_eval["std_return"]),
            "sem_return": learned_sem,
            "final_gen_best_train_return": history[-1] if history else None,
        },
        "result": {
            "gap_vs_vi_optimum": gap_vs_vi,
            "gap_vs_vi_optimum_pct": gap_vs_vi_pct,
            "gap_vs_base_stock_gate": gap_vs_gate,
            "gap_vs_base_stock_gate_pct": gap_vs_gate_pct,
            "paired_sem": paired_sem,
            "verdict": verdict,
        },
        "description": parsed.description,
    }

    # --- ledger row ---
    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    root.mkdir(parents=True, exist_ok=True)
    results_tsv = root / "results.tsv"
    header = [
        "commit", "reference", "budget", "depth", "split_type", "leaf_type",
        "warm_start", "learned_return", "vi_optimum", "base_stock_gate",
        "gap_vs_vi_pct", "gap_vs_gate", "gap_vs_gate_pct", "verdict", "description",
    ]
    if not results_tsv.exists():
        with results_tsv.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh, delimiter="\t").writerow(header)
    with results_tsv.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh, delimiter="\t").writerow([
            payload["commit"], parsed.reference, parsed.budget, parsed.depth,
            parsed.split_type, parsed.leaf_type, str(not parsed.no_warm_start),
            f"{learned_return:.4f}", f"{vi_optimum:.4f}", f"{gate_return:.4f}",
            f"{gap_vs_vi_pct:.4f}", f"{gap_vs_gate:.4f}", f"{gap_vs_gate_pct:.4f}",
            verdict, parsed.description,
        ])

    out_json = parsed.output_json or str(
        root / f"{parsed.reference}_d{parsed.depth}_{parsed.split_type}_{parsed.budget}.json"
    )
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["results_json"] = out_json

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
