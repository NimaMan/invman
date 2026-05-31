"""CMA-ES optimizer wrapper used by the training loop.

Algorithm:
  Thin wrapper around ``cma.CMAEvolutionStrategy``. The optimizer searches in a
  param-scale-normalized coordinate system: ``ask()`` samples candidates from the
  CMA distribution and multiplies by ``param_scales`` to return raw parameters;
  ``tell()`` divides the evaluated solutions back by ``param_scales`` before
  updating the distribution.

  Initialization of the CMA mean:
    - default: a random vector ``rng.randn(num_params)`` (unchanged behavior).
    - optional warm start: pass ``x0`` (raw parameter units) to seed the mean at
      a known-good solution; it is divided by ``param_scales`` to enter the
      normalized search space. Combined with a small ``sigma_init`` this confines
      the search to a neighborhood of x0 -- used to seed dual-sourcing soft-tree
      policies at the encoded capped-dual-index (CDI) optimum so CMA-ES refines
      around the verified static optimum instead of risking a worse basin.
"""

from __future__ import annotations

import numpy as np


def compute_weight_decay(weight_decay: float, model_param_list) -> np.ndarray:
    model_param_grid = np.asarray(model_param_list, dtype=np.float64)
    return -weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class CMAES:
    """Thin wrapper around ``cma.CMAEvolutionStrategy``."""

    def __init__(
        self,
        num_params: int,
        sigma_init: float = 0.10,
        popsize: int = 255,
        weight_decay: float = 0.00,
        param_scales=None,
        seed: int | None = None,
        x0=None,
    ) -> None:
        self.num_params = num_params
        self.sigma_init = sigma_init
        self.base_popsize = int(popsize)
        self.popsize = self.base_popsize
        self.weight_decay = weight_decay
        self.solutions = None
        self.seed = None if seed is None else int(seed)
        self.param_scales = (
            np.ones(self.num_params, dtype=np.float64)
            if param_scales is None
            else np.asarray(param_scales, dtype=np.float64)
        )

        import cma

        rng = np.random.RandomState(self.seed)
        # Optional warm start: seed the CMA mean at a known-good solution (e.g. an
        # encoded heuristic control) instead of a random vector. CMA optimizes in
        # the param_scales-normalized space, so the supplied x0 (in raw parameter
        # units) is divided by param_scales before being handed to the library.
        if x0 is None:
            initial = rng.randn(self.num_params)
        else:
            initial = np.asarray(x0, dtype=np.float64) / self.param_scales
        self.es = cma.CMAEvolutionStrategy(
            initial,
            self.sigma_init,
            {"popsize": self.popsize, "seed": self.seed},
        )

    def rms_stdev(self) -> float:
        sigma = self.es.result[6]
        return float(np.sqrt(np.mean(sigma * sigma)))

    def ask(self, popsize: int | None = None) -> np.ndarray:
        """Return a population of candidate parameters."""
        active_popsize = self.popsize if popsize is None else int(popsize)
        if active_popsize <= 0:
            raise ValueError(f"Population size must be positive; got {active_popsize}")
        if active_popsize != self.es.sp.popsize:
            # Recompute the CMA strategy weights for the requested population size
            # before sampling. This avoids per-iteration warnings from the library
            # and keeps the internal recombination weights aligned with the batch.
            self.es.sp.set(self.es.opts, active_popsize, verbose=False)
        self.popsize = active_popsize
        self.solutions = np.asarray(self.es.ask(number=active_popsize), dtype=np.float64)
        self.solutions *= self.param_scales[None, :]
        return self.solutions

    def tell(self, reward_table_result) -> None:
        reward_table = -np.asarray(reward_table_result, dtype=np.float64)
        if self.weight_decay > 0:
            reward_table += compute_weight_decay(self.weight_decay, self.solutions)
        self.es.tell(
            self.solutions / self.param_scales[None, :],
            reward_table.tolist(),
        )

    def current_param(self) -> np.ndarray:
        return np.asarray(self.es.result[5], dtype=np.float64) * self.param_scales

    def set_mu(self, mu) -> None:  # pragma: no cover - retained for compatibility
        del mu

    def best_param(self) -> np.ndarray:
        return np.asarray(self.es.result[0], dtype=np.float64) * self.param_scales

    def result(self):
        r = self.es.result
        return (r[0], -r[1], -r[1], r[6])
