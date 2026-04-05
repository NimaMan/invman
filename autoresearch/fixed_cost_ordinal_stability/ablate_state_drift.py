#!/usr/bin/env python3
"""Ablate state-semantics drift for the archived good fixed-cost ordinal checkpoint."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from invman.policies.linear import LinearPolicyNet
from invman.problems.lost_sales.env import build_env_from_args as build_current_env
from invman.problems.lost_sales_fixed_order_cost.reference_instances import build_reference_args


MODEL_DIR = (
    REPO_ROOT
    / "outputs"
    / "benchmarks"
    / "fixed_cost_canonical_suite_5k_seed42"
    / "models"
    / "fixed_cost_canonical_suite_5k_seed42_linear_gated_ordinal_quantity_255_5000"
)


def load_checkpoint() -> LinearPolicyNet:
    cfg = json.loads((MODEL_DIR / "model_config.json").read_text())
    state = torch.load(MODEL_DIR / "model_params.torch", map_location="cpu")
    model = LinearPolicyNet(**cfg["init_kwargs"])
    model.linear_weight[...] = (
        state["linear_app.weight"].detach().cpu().numpy().astype(np.float32)
    )
    model.linear_bias[...] = (
        state["linear_app.bias"].detach().cpu().numpy().astype(np.float32)
    )
    model.eval()
    return model


def load_old_env_module():
    old_src = subprocess.check_output(
        ["git", "show", "60d833d:invman/env/lost_sales.py"],
        cwd=REPO_ROOT,
        text=True,
    )
    module = types.ModuleType("old_lost_sales_env")
    exec(old_src, module.__dict__)
    return module


def eval_current(model, args, horizon: int, seed: int, *, use_old_scale: bool, use_old_init: bool) -> float:
    np.random.seed(seed)
    env = build_current_env(args, horizon=horizon, track_demand=False)
    if use_old_init:
        env.current_inventory_level = int(round(2 * args.demand_rate))
        env.lead_time_orders.clear()
        for _ in range(args.lead_time):
            q = (
                0
                if int(args.max_order_size) == 0
                else np.random.randint(1, int(args.max_order_size) + 1)
            )
            env.lead_time_orders.append(int(q))
            demand = env._sample_single_demand()
            env.current_inventory_level = max(0, env.current_inventory_level - int(demand))

    while not env.is_done():
        obs = (
            env.norm_state
            if not use_old_scale
            else np.asarray(env.state, dtype=np.float32) / float(max(1, args.max_order_size))
        )
        env.step(model(obs))
    return float(env.avg_total_cost)


def eval_old(model, old_env_module, args, horizon: int, seed: int) -> float:
    np.random.seed(seed)
    env = old_env_module.build_env_from_args(args, horizon=horizon, track_demand=False)
    while not env.is_done():
        env.step(model(env.norm_state))
    return float(env.avg_total_cost)


def summarize(label: str, costs: list[float]) -> None:
    print(
        label,
        {
            "mean": round(float(np.mean(costs)), 6),
            "std": round(float(np.std(costs)), 6),
            "min": round(float(np.min(costs)), 6),
            "max": round(float(np.max(costs)), 6),
            "costs": [round(float(cost), 6) for cost in costs],
        },
    )


def main() -> None:
    model = load_checkpoint()
    old_env_module = load_old_env_module()
    args = build_reference_args("lit_pois_mu5_l4_p4_k5")
    horizon = 20_000
    seeds = [42, 43, 44]

    variants = {
        "current": lambda seed: eval_current(
            model, args, horizon, seed, use_old_scale=False, use_old_init=False
        ),
        "old_scale_only": lambda seed: eval_current(
            model, args, horizon, seed, use_old_scale=True, use_old_init=False
        ),
        "old_init_only": lambda seed: eval_current(
            model, args, horizon, seed, use_old_scale=False, use_old_init=True
        ),
        "old_scale_old_init": lambda seed: eval_current(
            model, args, horizon, seed, use_old_scale=True, use_old_init=True
        ),
        "old_env": lambda seed: eval_old(model, old_env_module, args, horizon, seed),
    }

    for label, fn in variants.items():
        summarize(label, [fn(seed) for seed in seeds])


if __name__ == "__main__":
    main()
