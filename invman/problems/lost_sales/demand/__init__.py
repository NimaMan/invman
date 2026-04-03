from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


DEFAULT_MMPP2_LAMBDA_LOW = 3.0
DEFAULT_MMPP2_LAMBDA_HIGH = 7.0
DEFAULT_MMPP2_POSITIVE_P00 = 0.9
DEFAULT_MMPP2_POSITIVE_P11 = 0.9
DEFAULT_MMPP2_NEGATIVE_P00 = 0.1
DEFAULT_MMPP2_NEGATIVE_P11 = 0.1

SUPPORTED_DEMAND_NAMES = ("Poisson", "Geometric", "MarkovModulatedPoisson2")


def normalize_demand_dist_name(name: str) -> str:
    aliases = {
        "poisson": "Poisson",
        "geometric": "Geometric",
        "markovmodulatedpoisson2": "MarkovModulatedPoisson2",
        "mmpp2": "MarkovModulatedPoisson2",
    }
    normalized = aliases.get(str(name).replace("_", "").replace("-", "").lower())
    if normalized is None:
        valid = ", ".join(SUPPORTED_DEMAND_NAMES)
        raise ValueError(f"Unknown demand distribution '{name}'. Expected one of: {valid}")
    return normalized


@dataclass(frozen=True)
class DemandConfig:
    demand_dist_name: str
    demand_rate: float
    demand_lambda_low: float = DEFAULT_MMPP2_LAMBDA_LOW
    demand_lambda_high: float = DEFAULT_MMPP2_LAMBDA_HIGH
    demand_p00: float = DEFAULT_MMPP2_POSITIVE_P00
    demand_p11: float = DEFAULT_MMPP2_POSITIVE_P11

    def __post_init__(self):
        object.__setattr__(self, "demand_dist_name", normalize_demand_dist_name(self.demand_dist_name))
        object.__setattr__(self, "demand_rate", float(self.demand_rate))
        object.__setattr__(self, "demand_lambda_low", float(self.demand_lambda_low))
        object.__setattr__(self, "demand_lambda_high", float(self.demand_lambda_high))
        object.__setattr__(self, "demand_p00", float(self.demand_p00))
        object.__setattr__(self, "demand_p11", float(self.demand_p11))

        if self.demand_rate < 0:
            raise ValueError("demand_rate must be non-negative")
        if self.demand_dist_name != "MarkovModulatedPoisson2":
            return
        if self.demand_lambda_low < 0 or self.demand_lambda_high < 0:
            raise ValueError("demand_lambda_low and demand_lambda_high must be non-negative")
        if not 0.0 <= self.demand_p00 <= 1.0:
            raise ValueError("demand_p00 must be in [0, 1]")
        if not 0.0 <= self.demand_p11 <= 1.0:
            raise ValueError("demand_p11 must be in [0, 1]")
        if self.demand_p00 + self.demand_p11 >= 2.0:
            raise ValueError("demand_p00 + demand_p11 must be strictly less than 2")
        implied_mean = self.stationary_mean
        if abs(implied_mean - self.demand_rate) > 1e-6:
            raise ValueError(
                "MarkovModulatedPoisson2 parameters imply stationary mean "
                f"{implied_mean:.6f}, but demand_rate={self.demand_rate:.6f}"
            )

    @property
    def is_iid(self) -> bool:
        return self.demand_dist_name in {"Poisson", "Geometric"}

    @property
    def stationary_prob_high(self) -> float:
        denominator = 2.0 - self.demand_p00 - self.demand_p11
        return (1.0 - self.demand_p00) / denominator

    @property
    def stationary_prob_low(self) -> float:
        return 1.0 - self.stationary_prob_high

    @property
    def stationary_mean(self) -> float:
        if self.demand_dist_name != "MarkovModulatedPoisson2":
            return self.demand_rate
        return (
            self.stationary_prob_low * self.demand_lambda_low
            + self.stationary_prob_high * self.demand_lambda_high
        )

    @property
    def lag_one_regime_eigenvalue(self) -> float:
        return self.demand_p00 + self.demand_p11 - 1.0

    @property
    def one_period_variance(self) -> float:
        if self.demand_dist_name == "Poisson":
            return self.demand_rate
        if self.demand_dist_name == "Geometric":
            return self.demand_rate * (1.0 + self.demand_rate)
        delta = self.demand_lambda_high - self.demand_lambda_low
        return self.stationary_mean + self.stationary_prob_low * self.stationary_prob_high * (delta**2)

    def lag_k_autocovariance(self, lag: int) -> float:
        if lag < 0:
            raise ValueError("lag must be non-negative")
        if self.demand_dist_name != "MarkovModulatedPoisson2":
            return 0.0 if lag > 0 else self.one_period_variance
        if lag == 0:
            return self.one_period_variance
        delta = self.demand_lambda_high - self.demand_lambda_low
        return (
            self.stationary_prob_low
            * self.stationary_prob_high
            * (delta**2)
            * (self.lag_one_regime_eigenvalue**lag)
        )

    def lag_k_autocorrelation(self, lag: int) -> float:
        if lag < 0:
            raise ValueError("lag must be non-negative")
        if lag == 0:
            return 1.0
        variance = self.one_period_variance
        if variance <= 0.0:
            return 0.0
        return self.lag_k_autocovariance(lag) / variance

    def cumulative_variance(self, periods: int) -> float:
        if periods < 1:
            raise ValueError("periods must be at least 1")
        if self.demand_dist_name != "MarkovModulatedPoisson2":
            return periods * self.one_period_variance
        variance = periods * self.one_period_variance
        for lag in range(1, periods):
            variance += 2.0 * (periods - lag) * self.lag_k_autocovariance(lag)
        return variance


def build_demand_config(
    *,
    demand_dist_name: str,
    demand_rate: float,
    demand_lambda_low: float = DEFAULT_MMPP2_LAMBDA_LOW,
    demand_lambda_high: float = DEFAULT_MMPP2_LAMBDA_HIGH,
    demand_p00: float = DEFAULT_MMPP2_POSITIVE_P00,
    demand_p11: float = DEFAULT_MMPP2_POSITIVE_P11,
) -> DemandConfig:
    return DemandConfig(
        demand_dist_name=demand_dist_name,
        demand_rate=demand_rate,
        demand_lambda_low=demand_lambda_low,
        demand_lambda_high=demand_lambda_high,
        demand_p00=demand_p00,
        demand_p11=demand_p11,
    )


def build_demand_config_from_args(args) -> DemandConfig:
    return build_demand_config(
        demand_dist_name=getattr(args, "demand_dist_name", "Poisson"),
        demand_rate=float(getattr(args, "demand_rate", 5.0)),
        demand_lambda_low=float(getattr(args, "demand_lambda_low", DEFAULT_MMPP2_LAMBDA_LOW)),
        demand_lambda_high=float(getattr(args, "demand_lambda_high", DEFAULT_MMPP2_LAMBDA_HIGH)),
        demand_p00=float(getattr(args, "demand_p00", DEFAULT_MMPP2_POSITIVE_P00)),
        demand_p11=float(getattr(args, "demand_p11", DEFAULT_MMPP2_POSITIVE_P11)),
    )


class _BaseDemandProcess:
    def sample(self) -> int:
        raise NotImplementedError

    def sample_path(self, size: int) -> np.ndarray:
        return np.asarray([self.sample() for _ in range(int(size))], dtype=np.int64)


class _PoissonDemandProcess(_BaseDemandProcess):
    def __init__(self, config: DemandConfig, rng):
        self._rate = float(config.demand_rate)
        self._rng = rng

    def sample(self) -> int:
        return int(self._rng.poisson(lam=self._rate))

    def sample_path(self, size: int) -> np.ndarray:
        return np.asarray(self._rng.poisson(lam=self._rate, size=int(size)), dtype=np.int64)


class _GeometricDemandProcess(_BaseDemandProcess):
    def __init__(self, config: DemandConfig, rng):
        self._success_prob = 1.0 / (1.0 + float(config.demand_rate))
        self._rng = rng

    def sample(self) -> int:
        return int(self._rng.geometric(p=self._success_prob) - 1)

    def sample_path(self, size: int) -> np.ndarray:
        draws = self._rng.geometric(p=self._success_prob, size=int(size)) - 1
        return np.asarray(draws, dtype=np.int64)


class _MarkovModulatedPoisson2DemandProcess(_BaseDemandProcess):
    def __init__(self, config: DemandConfig, rng):
        self._lambda_low = float(config.demand_lambda_low)
        self._lambda_high = float(config.demand_lambda_high)
        self._p00 = float(config.demand_p00)
        self._p11 = float(config.demand_p11)
        self._rng = rng
        self._state = 1 if float(self._rng.random()) < config.stationary_prob_high else 0

    def _transition(self):
        if self._state == 0:
            self._state = 0 if float(self._rng.random()) < self._p00 else 1
        else:
            self._state = 1 if float(self._rng.random()) < self._p11 else 0

    def sample(self) -> int:
        rate = self._lambda_high if self._state == 1 else self._lambda_low
        demand = int(self._rng.poisson(lam=rate))
        self._transition()
        return demand


def build_demand_process(config: DemandConfig, rng=np.random) -> _BaseDemandProcess:
    if config.demand_dist_name == "Poisson":
        return _PoissonDemandProcess(config, rng)
    if config.demand_dist_name == "Geometric":
        return _GeometricDemandProcess(config, rng)
    if config.demand_dist_name == "MarkovModulatedPoisson2":
        return _MarkovModulatedPoisson2DemandProcess(config, rng)
    raise NotImplementedError(f"Unsupported demand distribution: {config.demand_dist_name}")


def get_demand_prob_vector(config: DemandConfig, eps: float = 1e-14) -> tuple[np.ndarray, int, int]:
    if config.demand_dist_name == "Poisson":
        lb, ub = poisson.interval(1 - eps, mu=config.demand_rate)
        support = np.arange(int(lb), int(ub) + 1)
        probs = poisson.pmf(support, config.demand_rate)
        probs /= probs.sum()
        return probs.astype(np.float64), int(lb), int(ub)
    if config.demand_dist_name == "Geometric":
        success_prob = 1.0 / (1.0 + config.demand_rate)
        probs = []
        cumulative = 0.0
        k = 0
        while cumulative < 1 - eps:
            prob = success_prob * ((1 - success_prob) ** k)
            probs.append(prob)
            cumulative += prob
            k += 1
        values = np.asarray(probs, dtype=np.float64)
        values /= values.sum()
        return values, 0, len(values) - 1
    if config.demand_dist_name == "MarkovModulatedPoisson2":
        _, ub_low = poisson.interval(1 - eps, mu=config.demand_lambda_low)
        _, ub_high = poisson.interval(1 - eps, mu=config.demand_lambda_high)
        ub = int(max(ub_low, ub_high))
        support = np.arange(ub + 1)
        probs = (
            config.stationary_prob_low * poisson.pmf(support, config.demand_lambda_low)
            + config.stationary_prob_high * poisson.pmf(support, config.demand_lambda_high)
        )
        probs /= probs.sum()
        return probs.astype(np.float64), 0, ub
    raise NotImplementedError(f"Unsupported demand distribution: {config.demand_dist_name}")


def get_cumulative_demand_cdf(config: DemandConfig, k: int, periods: int) -> float:
    if periods < 1:
        raise ValueError("periods must be at least 1")
    if config.demand_dist_name == "Poisson":
        return float(poisson.cdf(k=k, mu=periods * config.demand_rate))
    if config.demand_dist_name == "Geometric":
        success_prob = 1.0 / (1.0 + config.demand_rate)
        from scipy.stats import nbinom

        return float(nbinom.cdf(k=k, n=periods, p=success_prob))
    if config.demand_dist_name == "MarkovModulatedPoisson2":
        if k < 0:
            return 0.0
        total_ub = int(poisson.interval(1 - 1e-14, mu=periods * max(config.demand_lambda_low, config.demand_lambda_high))[1])
        if k >= total_ub:
            return 1.0

        transition = np.asarray(
            [
                [config.demand_p00, 1.0 - config.demand_p00],
                [1.0 - config.demand_p11, config.demand_p11],
            ],
            dtype=np.float64,
        )
        stationary = np.asarray(
            [config.stationary_prob_low, config.stationary_prob_high],
            dtype=np.float64,
        )
        emission_low = poisson.pmf(np.arange(total_ub + 1), config.demand_lambda_low)
        emission_high = poisson.pmf(np.arange(total_ub + 1), config.demand_lambda_high)

        state_mass = np.zeros((2, total_ub + 1), dtype=np.float64)
        state_mass[0, 0] = stationary[0]
        state_mass[1, 0] = stationary[1]

        for _ in range(periods):
            next_mass = np.zeros_like(state_mass)
            for state_idx, emission in enumerate((emission_low, emission_high)):
                totals = np.nonzero(state_mass[state_idx])[0]
                for total in totals:
                    mass = state_mass[state_idx, total]
                    max_emit = total_ub - total
                    emitted = mass * emission[: max_emit + 1]
                    next_mass[0, total : total + max_emit + 1] += emitted * transition[state_idx, 0]
                    next_mass[1, total : total + max_emit + 1] += emitted * transition[state_idx, 1]
            state_mass = next_mass

        pmf = state_mass.sum(axis=0)
        pmf /= pmf.sum()
        return float(pmf[: k + 1].sum())
    raise NotImplementedError(f"Unsupported demand distribution: {config.demand_dist_name}")


MMPP2_POSITIVE_MEAN5 = {
    "demand_dist_name": "MarkovModulatedPoisson2",
    "demand_rate": 5.0,
    "demand_lambda_low": 3.0,
    "demand_lambda_high": 7.0,
    "demand_p00": DEFAULT_MMPP2_POSITIVE_P00,
    "demand_p11": DEFAULT_MMPP2_POSITIVE_P11,
}

MMPP2_NEGATIVE_MEAN5 = {
    "demand_dist_name": "MarkovModulatedPoisson2",
    "demand_rate": 5.0,
    "demand_lambda_low": 3.0,
    "demand_lambda_high": 7.0,
    "demand_p00": DEFAULT_MMPP2_NEGATIVE_P00,
    "demand_p11": DEFAULT_MMPP2_NEGATIVE_P11,
}


__all__ = [
    "DEFAULT_MMPP2_LAMBDA_HIGH",
    "DEFAULT_MMPP2_LAMBDA_LOW",
    "DEFAULT_MMPP2_NEGATIVE_P00",
    "DEFAULT_MMPP2_NEGATIVE_P11",
    "DEFAULT_MMPP2_POSITIVE_P00",
    "DEFAULT_MMPP2_POSITIVE_P11",
    "DemandConfig",
    "MMPP2_NEGATIVE_MEAN5",
    "MMPP2_POSITIVE_MEAN5",
    "SUPPORTED_DEMAND_NAMES",
    "build_demand_config",
    "build_demand_config_from_args",
    "build_demand_process",
    "get_cumulative_demand_cdf",
    "get_demand_prob_vector",
    "normalize_demand_dist_name",
]
