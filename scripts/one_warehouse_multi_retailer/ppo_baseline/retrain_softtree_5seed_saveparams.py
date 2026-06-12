#!/usr/bin/env python
"""Retrain the incumbent OWMR instance_14 soft-tree (explore_best_constant_oblique
spec, T=0.25) over 5 optimizer seeds and SAVE each seed's trained flat params, so
they can be re-scored by the Python instrument on the same holdout CRN block as the
faithful in-protocol PPO. This is the A-side of the fair head-to-head.

Reproduces evaluate_policy_spec.evaluate_spec EXACTLY for instance 14 (the 43820
5-seed headline), but persists per-seed trained_flat + the Rust holdout cost so we
can later instrument-score both PPO and the soft-tree identically.

Outputs /tmp/owmr_ppo/softtree_5seed_params.json :
  per_seed: [{seed, rust_holdout_cost, trained_flat, deployed_flat, deployed_cost}]
  plus gate levels, anchor (warm) flat, model config.
"""
import sys, json, time
from pathlib import Path
import numpy as np

PKG = "/home/nima/code/ml/invman"
APS = PKG + "/policy_search/agentic"
OWMR = PKG + "/scripts/one_warehouse_multi_retailer"
for p in (PKG, APS, OWMR):
    if p not in sys.path:
        sys.path.insert(0, p)

import common
import evaluate_policy_spec as E
import run_asymmetric_learned_vs_gate as R
from invman.es_mp import train
from policy_spec_compiler import compile_policy_spec, attach_gate_warm_start
from benchmark_learned_vs_heuristic import _get_model_fitness, _get_population_fitness

SPEC_PATH = APS + "/specs/explore_best_constant_oblique.json"
# headline 5-seed run used temperature 0.25 (see echelon_baseline_full_5seed.json)
HEADLINE_TEMPERATURE = 0.25
INSTANCE = 14
BUDGET = "full"
N_SEEDS = 5
SIGMA_INIT = 0.1
WORKERS = 4

OUT = Path("/tmp/owmr_ppo")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    spec = json.loads(Path(SPEC_PATH).read_text())
    spec["temperature"] = HEADLINE_TEMPERATURE  # match the 43820 headline run
    reference = common.get_reference(f"kaynov2024_instance_{INSTANCE}")
    budget = E.BUDGETS[BUDGET]
    out_root = OUT / "evaluate_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    compiled = compile_policy_spec(spec, reference)
    eval_allocs = compiled.eval_allocations

    gate, gate_best_alloc, holdout_paths = E._gate_cost_and_levels(
        reference, budget, WORKERS, out_root, eval_allocs
    )
    gate_best = gate[gate_best_alloc]
    gate_cost = float(gate_best["holdout_mean"])
    print(f"[gate] best_alloc={gate_best_alloc} cost={gate_cost:.2f} W={gate_best['warehouse_base_stock_level']}", flush=True)

    compiled = attach_gate_warm_start(
        compiled, reference, spec.get("warm_start", "gate_invertible"),
        gate_best["warehouse_base_stock_level"], gate_best["retailer_base_stock_levels"],
    )
    anchor_cost = None
    if compiled.warm_started and compiled.warm_flat is not None:
        anchor_alloc_costs = E._eval_policy_on_holdout(
            reference, compiled.model, compiled.warm_flat,
            compiled.policy_action_mode, eval_allocs, holdout_paths,
        )
        anchor_cost = min(anchor_alloc_costs.values())
    print(f"[anchor] warm cost={anchor_cost:.2f}", flush=True)

    train_allocation = gate_best_alloc if gate_best_alloc in eval_allocs else eval_allocs[0]
    seed_base = 700_000
    per_seed = []
    for s in range(N_SEEDS):
        seed = seed_base + 101 * s
        t0 = time.time()
        model = common.build_soft_tree_model(
            reference, depth=compiled.depth, temperature=compiled.temperature,
            split_type=compiled.split_type, leaf_type=compiled.leaf_type,
            policy_action_mode=compiled.policy_action_mode,
            policy_state_mode=compiled.policy_state_mode,
        )
        train_args = E._training_namespace(reference, budget, compiled, seed, SIGMA_INIT, out_root)
        if compiled.warm_flat is not None:
            train_args.cma_x0 = compiled.warm_flat
        trained_model, _hist = train(
            model=model,
            get_model_fitness=_get_model_fitness(model, reference, train_allocation, compiled.policy_action_mode),
            get_population_fitness=_get_population_fitness(model, reference, train_allocation, compiled.policy_action_mode),
            args=train_args, same_seed=True,
        )
        trained_flat = np.asarray(trained_model.get_model_flat_params(), dtype=np.float32).tolist()
        trained_alloc_costs = E._eval_policy_on_holdout(
            reference, trained_model, trained_flat, compiled.policy_action_mode, eval_allocs, holdout_paths,
        )
        trained_cost = min(trained_alloc_costs.values())
        deployed_flat = trained_flat
        deployed_cost = trained_cost
        if anchor_cost is not None and anchor_cost < trained_cost:
            deployed_flat = list(compiled.warm_flat)
            deployed_cost = float(anchor_cost)
        per_seed.append({
            "seed": seed,
            "rust_trained_cost": float(trained_cost),
            "rust_trained_alloc_costs": {k: float(v) for k, v in trained_alloc_costs.items()},
            "deployed_cost": float(deployed_cost),
            "trained_flat": trained_flat,
            "deployed_flat": deployed_flat,
        })
        print(f"[seed {s} ={seed}] trained_cost={trained_cost:.2f} deployed={deployed_cost:.2f} ({time.time()-t0:.0f}s)", flush=True)

    payload = {
        "instance": INSTANCE,
        "spec_path": SPEC_PATH,
        "temperature": HEADLINE_TEMPERATURE,
        "budget": BUDGET,
        "n_seeds": N_SEEDS,
        "sigma_init": SIGMA_INIT,
        "gate_best_allocation": gate_best_alloc,
        "gate_cost": gate_cost,
        "gate_warehouse_level": int(gate_best["warehouse_base_stock_level"]),
        "gate_retailer_levels": [int(v) for v in gate_best["retailer_base_stock_levels"]],
        "anchor_cost": anchor_cost,
        "anchor_flat": list(compiled.warm_flat) if compiled.warm_flat is not None else None,
        "policy_action_mode": compiled.policy_action_mode,
        "policy_state_mode": compiled.policy_state_mode,
        "depth": compiled.depth,
        "temperature_compiled": compiled.temperature,
        "split_type": compiled.split_type,
        "leaf_type": compiled.leaf_type,
        "input_dim": int(compiled.model.input_dim),
        "min_values": [int(v) for v in compiled.model.min_values],
        "max_values": [int(v) for v in compiled.model.max_values],
        "train_allocation": train_allocation,
        "per_seed": per_seed,
        "rust_deployed_mean": float(np.mean([p["deployed_cost"] for p in per_seed])),
        "rust_deployed_std": float(np.std([p["deployed_cost"] for p in per_seed])),
    }
    (OUT / "softtree_5seed_params.json").write_text(json.dumps(payload, indent=2))
    print(f"[done] rust deployed mean={payload['rust_deployed_mean']:.2f} +/- {payload['rust_deployed_std']:.2f}", flush=True)
    print(f"wrote {OUT/'softtree_5seed_params.json'}", flush=True)


if __name__ == "__main__":
    main()
