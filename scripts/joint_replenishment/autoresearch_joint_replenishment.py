"""Single-policy autoresearch runner for the joint-replenishment benchmark.

OBJECTIVE
---------
Train ONE soft-tree CMA-ES policy on a NAMED Vanvuchelen et al. (2020) Table-2
joint-replenishment instance and log its held-out cost + optimality gap vs the
strongest heuristic (MOQ, `minimum_order_quantity`, at the per-item newsvendor
target). DYN-OUT is dominated by MOQ on all 16 settings, so MOQ is the sole gap
target. This is the joint-replenishment counterpart of
scripts/dual_sourcing/autoresearch_dual_sourcing.py and
scripts/multi_echelon/autoresearch_multi_echelon.py, and it REUSES the
learned-benchmark helpers in scripts/joint_replenishment/common.py (binding
`joint_replenishment_soft_tree_rollout` / `..._population_rollout`).

It defaults to a currently-LOSING instance (setting 4, the -18.13% worst loss from
the learned-benchmark phase) so a single run lands on the hardest target.

ALGORITHMIC DESCRIPTION
-----------------------
 1. Reference -> rollout config (common.build_soft_tree_model semantics). The 16
    Table-2 references carry only cost/demand structure; we add the simulation knobs
    used everywhere else for this problem: horizon = `periods` (default 200), initial
    inventory = zeros, discount gamma = 0.99, action box per item = 2*truck_capacity.
 2. Soft-tree policy with CLI-selected structure (depth / temperature / split_type /
    leaf_type) over the `vector_quantity` action box. backbone='soft_tree',
    input_dim = num_items + 2 (per-item inventory + total-inventory + remaining-horizon
    fraction), control_dim = num_items.
 3. OPTIONAL CMA-ES WARM-START AT MOQ (--warm_start_moq). The soft-tree decoder lives
    in Rust and is not analytically invertible into tree params, so we cannot encode MOQ
    exactly. Instead we seed the CMA mean (args.cma_x0) with the BEST of a small candidate
    set -- the zero vector plus a few small randn probes -- scored on a few training
    seeds. This is honest, decoder-agnostic anchoring near a vetted starting point rather
    than a blind random mean. (When the action-design lever adds a base-stock-anchored
    adapter, the zero vector itself becomes the MOQ-anchored start.)
 4. CMA-ES training via invman.es_mp.train (population path). Each generation samples
    `es_population` candidates; the Rust population rollout scores each on its paired
    seed (CRN within the generation), averaged over `train_seed_batch` seed offsets,
    returning negative discounted cost as fitness. Seeds advance each generation.
 5. HELD-OUT EVALUATION. Training seeds start at `seed`; evaluation uses a DISJOINT block
    of `eval_seeds` seeds from EVAL_SEED_BASE (1_000_000). The SAME eval block scores the
    learned soft-tree and MOQ (paired / CRN), so the gap is variance-reduced.
 6. LEDGER. Append a TSV row: commit, experiment, reference, budget, structure,
    learned mean_cost, MOQ cost (best_heuristic), best_heuristic_name, gap, gap%, winner,
    description. Negative gap%/gap = learned cheaper than MOQ = a win on a losing setting.

CPU CAP
-------
HARD 2-CORE CAP (two sibling autoresearch agents run in parallel). Every native layer is
capped BEFORE numpy / invman_rust import: RAYON_NUM_THREADS (Rust population rollout),
OPENBLAS/OMP/MKL/NUMEXPR (numpy/CMA-ES eigendecomposition). mp_num_processors is forced to
1 (the population path bypasses the multiprocessing Pool; rayon is the only fan-out, capped
at RAYON_NUM_THREADS). Lower externally-exported thread values are preserved; higher values
are capped by the shared CPU helper. The scripts' ~27-core default is overridden.

USAGE
-----
  # smoke (wiring check, tiny budget)
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python autoresearch_joint_replenishment.py \
      --budget smoke --description "smoke: wiring" \
      --reference vanvuchelen2020_small_scale_setting_4

  # screening pass on a losing setting with MOQ warm-start
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 python autoresearch_joint_replenishment.py \
      --budget screening --warm_start_moq --depth 3 \
      --reference vanvuchelen2020_small_scale_setting_12 \
      --description "screening: depth3 + MOQ warm-start on high-cost loser"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from invman.cpu_limits import configure_process_cpu_limits_from_argv

# HARD 2-CORE CAP -- set BEFORE numpy / invman_rust pull in their native libs.
configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

import numpy as np

from invman.es_mp import train

from common import (
    build_soft_tree_model,
    evaluate_heuristic_policy,
    evaluate_soft_tree_policy,
    get_reference,
    newsvendor_item_targets,
    soft_tree_rollout_kwargs,
)

import invman_rust


EVAL_SEED_BASE = 1_000_000  # disjoint from any training seed block

# Per-budget knobs. `full` defaults to depth 3 -- the high-cost-setting recovery budget.
BUDGETS = {
    "smoke":     {"es_population": 8,  "training_episodes": 8,   "train_seed_batch": 2,  "eval_seeds": 64,   "depth": 2},
    "screening": {"es_population": 16, "training_episodes": 80,  "train_seed_batch": 4,  "eval_seeds": 512,  "depth": 2},
    "full":      {"es_population": 24, "training_episodes": 300, "train_seed_batch": 12, "eval_seeds": 2048, "depth": 3},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Autoresearch-style loop for the joint-replenishment benchmark.")
    parser.add_argument("--run_tag", default="joint_replenishment_autoresearch")
    parser.add_argument("--budget", choices=sorted(BUDGETS), default="screening")
    parser.add_argument("--description", required=True)
    # Default to the worst currently-LOSING setting (-18.13% vs MOQ in the learned phase).
    parser.add_argument("--reference", default="vanvuchelen2020_small_scale_setting_4")
    parser.add_argument("--periods", type=int, default=200)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    # Soft-tree structure (the search surface). depth defaults to the budget's depth.
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    parser.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    parser.add_argument("--sigma_init", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    # Action design (the high-cost-setting recovery lever). "wide" = 2*truck_capacity per
    # item (the default rounded-action box); "basestock" = newsvendor target + cap_slack
    # per item, a base-stock-anchored action box that finens decode resolution around the
    # optimal order for the high-cost h=5,b=95 family. See common._max_order_quantities.
    parser.add_argument("--action_box", choices=["wide", "basestock"], default="wide")
    parser.add_argument("--cap_slack", type=int, default=1)
    # Budget overrides (otherwise taken from the --budget preset).
    parser.add_argument("--es_population", type=int, default=None)
    parser.add_argument("--training_episodes", type=int, default=None)
    parser.add_argument("--train_seed_batch", type=int, default=None)
    parser.add_argument("--eval_seeds", type=int, default=None)
    # CMA-ES warm-start at MOQ (decoder-agnostic best-of-candidates seed of the CMA mean).
    parser.add_argument("--warm_start_moq", action="store_true",
                        help="Seed the CMA mean (cma_x0) with the best of a small candidate set, anchoring near MOQ.")
    parser.add_argument("--warm_start_candidates", type=int, default=8)
    return parser.parse_args()


def _git_short_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def _full_reference(ref: dict, parsed) -> dict:
    """Attach the simulation knobs the 16 Table-2 references omit (same as the benchmark)."""
    full = dict(ref)
    full["periods"] = int(parsed.periods)
    full["discount_factor"] = float(parsed.discount_factor)
    full["initial_inventory_levels"] = [0] * int(ref.get("num_items", len(ref["demand_highs"])))
    return full


def _get_model_fitness(model, reference):
    def inner(_model, args, model_params=None, seed=1234, indiv_idx=-1,
              return_env=False, track_demand=False, verbose=False):
        del _model, return_env, track_demand, verbose
        flat_params = model.get_model_flat_params() if model_params is None else model_params
        costs = [
            float(invman_rust.joint_replenishment_soft_tree_rollout(
                seed=int(seed) + offset,
                **soft_tree_rollout_kwargs(reference, model, flat_params=flat_params),
            ))
            for offset in range(int(getattr(args, "train_seed_batch", 1)))
        ]
        return -float(np.mean(costs)), indiv_idx

    return inner


def _get_population_fitness(model, reference):
    def inner(_model, args, model_params_batch, seeds):
        del _model
        params_batch = [np.asarray(p, dtype=np.float32).tolist() for p in model_params_batch]
        rollout_kwargs = {
            key: value
            for key, value in soft_tree_rollout_kwargs(
                reference, model, flat_params=model.get_model_flat_params()
            ).items()
            if key != "flat_params"
        }
        batch_costs = [
            invman_rust.joint_replenishment_soft_tree_population_rollout(
                params_batch=params_batch,
                seeds=[int(s) + offset for s in seeds],
                **rollout_kwargs,
            )
            for offset in range(int(getattr(args, "train_seed_batch", 1)))
        ]
        costs = np.mean(np.asarray(batch_costs, dtype=np.float64), axis=0)
        return [(-float(c), i) for i, c in enumerate(costs.tolist())]

    return inner


def _moq_warm_start(model, reference, *, num_candidates: int, seed: int, train_seed_batch: int):
    """Decoder-agnostic warm-start: among the zero vector and a few small randn probes,
    return the candidate flat-param vector with the lowest mean training-seed cost. Used
    to seed the CMA mean near a vetted point (MOQ behaviour) instead of a blind random
    vector. The soft-tree action decoder lives in Rust and is not invertible into tree
    params, so this is an honest best-of-candidates anchor, not an exact MOQ encoding."""
    rng = np.random.RandomState(int(seed))
    n = int(model.num_params)
    candidates = [np.zeros(n, dtype=np.float32)]
    for _ in range(max(0, int(num_candidates) - 1)):
        candidates.append((0.5 * rng.randn(n)).astype(np.float32))
    probe_seeds = [int(seed) + offset for offset in range(max(1, int(train_seed_batch)))]
    best_vec, best_cost = None, float("inf")
    for vec in candidates:
        costs = [
            float(invman_rust.joint_replenishment_soft_tree_rollout(
                seed=ps, **soft_tree_rollout_kwargs(reference, model, flat_params=vec.tolist()),
            ))
            for ps in probe_seeds
        ]
        mean_cost = float(np.mean(costs))
        if mean_cost < best_cost:
            best_vec, best_cost = vec, mean_cost
    return best_vec.astype(np.float64).tolist(), best_cost


def main():
    parsed = parse_args()
    budget = BUDGETS[parsed.budget]
    depth = parsed.depth if parsed.depth is not None else budget["depth"]
    es_population = parsed.es_population if parsed.es_population is not None else budget["es_population"]
    training_episodes = parsed.training_episodes if parsed.training_episodes is not None else budget["training_episodes"]
    train_seed_batch = parsed.train_seed_batch if parsed.train_seed_batch is not None else budget["train_seed_batch"]
    eval_seeds = parsed.eval_seeds if parsed.eval_seeds is not None else budget["eval_seeds"]

    reference = _full_reference(get_reference(parsed.reference), parsed)
    model = build_soft_tree_model(
        reference,
        depth=int(depth),
        temperature=float(parsed.temperature),
        split_type=str(parsed.split_type),
        leaf_type=str(parsed.leaf_type),
        action_box=str(parsed.action_box),
        cap_slack=int(parsed.cap_slack),
    )

    box_tag = "wide" if parsed.action_box == "wide" else f"bs{parsed.cap_slack}"
    structure = f"d{depth}_{parsed.split_type}_{parsed.leaf_type}_t{parsed.temperature}_{box_tag}"
    experiment_name = (
        f"{parsed.run_tag}_{parsed.budget}_{parsed.reference}_{structure}"
        f"{'_moqws' if parsed.warm_start_moq else ''}_s{parsed.seed}"
    )

    root = PACKAGE_ROOT / "outputs" / "autoresearch" / parsed.run_tag
    root.mkdir(parents=True, exist_ok=True)
    results_tsv = root / "results.tsv"
    if not results_tsv.exists():
        with results_tsv.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, delimiter="\t").writerow([
                "commit", "experiment_name", "reference", "budget", "policy_architecture",
                "mean_cost", "best_heuristic", "best_heuristic_name",
                "heuristic_gap", "heuristic_gap_pct", "winner", "description",
            ])

    cma_x0 = None
    warm_start_cost = None
    if parsed.warm_start_moq:
        cma_x0, warm_start_cost = _moq_warm_start(
            model, reference,
            num_candidates=int(parsed.warm_start_candidates),
            seed=int(parsed.seed),
            train_seed_batch=int(train_seed_batch),
        )

    output_root = root / experiment_name
    train_args = SimpleNamespace(
        training_method="cma",
        sigma_init=float(parsed.sigma_init),
        es_population=int(es_population),
        training_episodes=int(training_episodes),
        mp_num_processors=1,  # population path bypasses the Pool; keep the fallback single-process too
        save_every=max(1, int(training_episodes)),
        save_solutions=False,
        horizon=int(reference["periods"]),
        seed=int(parsed.seed),
        train_seed_batch=int(train_seed_batch),
        cma_x0=cma_x0,
        experiment_name=experiment_name,
        log_dir=str(output_root / "logs"),
        trained_models_dir=str(output_root / "models"),
    )

    trained_model, _ = train(
        model=model,
        get_model_fitness=_get_model_fitness(model, reference),
        get_population_fitness=_get_population_fitness(model, reference),
        args=train_args,
        same_seed=False,
    )

    held_out = [EVAL_SEED_BASE + offset for offset in range(int(eval_seeds))]
    learned = evaluate_soft_tree_policy(reference, trained_model, held_out)
    learned_cost = float(learned["mean_cost"])

    targets = newsvendor_item_targets(reference)
    moq = evaluate_heuristic_policy(
        reference, "minimum_order_quantity",
        [float(targets[0]), float(targets[1]), 1.0, 2.0],
        replications=int(eval_seeds), seed=EVAL_SEED_BASE,
    )
    best_heuristic_name = "MOQ"
    best_heuristic_cost = float(moq["mean_cost"])
    gap = learned_cost - best_heuristic_cost                 # < 0 => learned cheaper (win)
    gap_pct = 100.0 * (learned_cost / best_heuristic_cost - 1.0)
    winner = "learned" if learned_cost < best_heuristic_cost - 1e-9 else best_heuristic_name

    with results_tsv.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter="\t").writerow([
            _git_short_commit(PACKAGE_ROOT),
            experiment_name,
            parsed.reference,
            parsed.budget,
            structure,
            f"{learned_cost:.6f}",
            f"{best_heuristic_cost:.6f}",
            best_heuristic_name,
            f"{gap:.6f}",
            f"{gap_pct:.4f}",
            winner,
            parsed.description,
        ])

    print(json.dumps({
        "ledger": str(results_tsv),
        "reference": parsed.reference,
        "budget": parsed.budget,
        "policy_architecture": structure,
        "warm_start_moq": bool(parsed.warm_start_moq),
        "warm_start_probe_cost": warm_start_cost,
        "learned_mean_cost": learned_cost,
        "best_heuristic_name": best_heuristic_name,
        "best_heuristic_cost": best_heuristic_cost,
        "heuristic_gap": gap,
        "heuristic_gap_pct": gap_pct,
        "winner": winner,
        "num_eval_seeds": int(eval_seeds),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
    }, indent=2))


if __name__ == "__main__":
    main()
