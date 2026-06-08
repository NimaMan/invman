"""Reusable PPO trainer entry — the gradient-based counterpart to CMA-ES.

OBJECTIVE
---------
Expose the first-class Rust PPO trainer (``core::ppo``, candle autodiff) as a
single reusable Python entry, parallel in role to ``invman.es_mp.train`` for
CMA-ES. Where CMA-ES optimizes a flat policy-parameter vector on the Rust
scalar-cost oracle (gradient-free), PPO trains a neural actor-critic by rolling
the problem's environment per step (gradient-based). Both are trainers; this
module is the PPO dispatch.

ARCHITECTURE
------------
The whole PPO loop runs INSIDE Rust (BC warm-start, GAE, clipped surrogate,
value clipping, entropy, Adam, best-checkpoint, greedy holdout eval). Each
problem that wants PPO implements a ``PpoVecEnv`` in Rust and exposes a
``<problem>_train_ppo`` pyo3 binding; this module routes ``problem`` -> binding.
Adding a new problem to PPO = add its env + binding + one row in ``_PPO_BINDINGS``
(no change here otherwise), exactly mirroring how new problems plug into the
shared rollout oracle.

Build the extension with the PPO feature before use::

    maturin develop --release --features python-extension,ppo

Then::

    from invman.ppo_trainer import train_ppo
    result = train_ppo("one_warehouse_multi_retailer", seed=0, iters=60)
    # result: {gate_holdout_cost, best_holdout_cost, final_holdout_cost_mean,
    #          final_holdout_cost_std, curve}
"""
from __future__ import annotations

from typing import Any

import invman_rust

# problem name -> the pyo3 binding that trains it with PPO.
_PPO_BINDINGS = {
    "one_warehouse_multi_retailer": "one_warehouse_multi_retailer_train_ppo",
}


def available_problems() -> list[str]:
    """Problems with a wired PPO trainer."""
    return sorted(_PPO_BINDINGS)


def train_ppo(problem: str, **hyperparameters: Any) -> dict:
    """Train ``problem``'s policy with the reusable Rust PPO trainer.

    Parameters mirror the binding's keyword arguments (seed, iters, lr, clip,
    ppo_epochs, bc_epochs, ...); see the binding's signature for defaults (the
    validated 5-seed config). Returns the binding's result dict.
    """
    binding_name = _PPO_BINDINGS.get(problem)
    if binding_name is None:
        raise NotImplementedError(
            f"PPO trainer is not yet wired for problem '{problem}'. "
            f"Implement a Rust PpoVecEnv + a <problem>_train_ppo binding "
            f"(see one_warehouse_multi_retailer/ppo_environment.rs + ppo_bindings.rs) "
            f"and register it in invman.ppo_trainer._PPO_BINDINGS. "
            f"Currently available: {available_problems()}."
        )
    binding = getattr(invman_rust, binding_name, None)
    if binding is None:
        raise RuntimeError(
            f"invman_rust.{binding_name} is missing — rebuild the extension with the "
            f"PPO feature: `maturin develop --release --features python-extension,ppo`."
        )
    return binding(**hyperparameters)
