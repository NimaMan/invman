"""CMA-ES optimizer wrapper used by the training loop."""

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
    ) -> None:
        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.solutions = None
        self.param_scales = (
            np.ones(self.num_params, dtype=np.float64)
            if param_scales is None
            else np.asarray(param_scales, dtype=np.float64)
        )

        import cma

        self.es = cma.CMAEvolutionStrategy(
            np.random.randn(self.num_params),
            self.sigma_init,
            {"popsize": self.popsize},
        )

    def rms_stdev(self) -> float:
        sigma = self.es.result[6]
        return float(np.sqrt(np.mean(sigma * sigma)))

    def ask(self) -> np.ndarray:
        """Return a population of candidate parameters."""
        self.solutions = np.asarray(self.es.ask(), dtype=np.float64)
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
