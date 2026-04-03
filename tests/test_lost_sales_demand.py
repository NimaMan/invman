import numpy as np
import pytest

from invman.problems.lost_sales.demand import (
    build_demand_config,
    build_demand_process,
    get_cumulative_demand_cdf,
    get_demand_prob_vector,
)


def _lag_one_corr(samples: np.ndarray) -> float:
    return float(np.corrcoef(samples[:-1], samples[1:])[0, 1])


def test_iid_demand_configs_have_zero_analytic_autocorrelation():
    poisson_config = build_demand_config(demand_dist_name="Poisson", demand_rate=5.0)
    geometric_config = build_demand_config(demand_dist_name="Geometric", demand_rate=5.0)

    assert poisson_config.one_period_variance == pytest.approx(5.0)
    assert poisson_config.lag_k_autocorrelation(1) == pytest.approx(0.0)
    assert poisson_config.lag_k_autocorrelation(2) == pytest.approx(0.0)

    assert geometric_config.one_period_variance == pytest.approx(30.0)
    assert geometric_config.lag_k_autocorrelation(1) == pytest.approx(0.0)
    assert geometric_config.lag_k_autocorrelation(2) == pytest.approx(0.0)


def test_mmpp2_configs_have_expected_analytic_lag_one_autocorrelation():
    positive_config = build_demand_config(
        demand_dist_name="MarkovModulatedPoisson2",
        demand_rate=5.0,
        demand_lambda_low=3.0,
        demand_lambda_high=7.0,
        demand_p00=0.9,
        demand_p11=0.9,
    )
    negative_config = build_demand_config(
        demand_dist_name="MarkovModulatedPoisson2",
        demand_rate=5.0,
        demand_lambda_low=3.0,
        demand_lambda_high=7.0,
        demand_p00=0.1,
        demand_p11=0.1,
    )

    expected_rho1 = 3.2 / 9.0

    assert positive_config.stationary_mean == pytest.approx(5.0)
    assert positive_config.one_period_variance == pytest.approx(9.0)
    assert positive_config.lag_k_autocorrelation(1) == pytest.approx(expected_rho1)

    assert negative_config.stationary_mean == pytest.approx(5.0)
    assert negative_config.one_period_variance == pytest.approx(9.0)
    assert negative_config.lag_k_autocorrelation(1) == pytest.approx(-expected_rho1)


def test_sampled_demand_paths_match_expected_correlation_regimes():
    configs = {
        "Poisson": (
            build_demand_config(demand_dist_name="Poisson", demand_rate=5.0),
            (-0.03, 0.03),
        ),
        "Geometric": (
            build_demand_config(demand_dist_name="Geometric", demand_rate=5.0),
            (-0.03, 0.03),
        ),
        "MMPP2 positive": (
            build_demand_config(
                demand_dist_name="MarkovModulatedPoisson2",
                demand_rate=5.0,
                demand_lambda_low=3.0,
                demand_lambda_high=7.0,
                demand_p00=0.9,
                demand_p11=0.9,
            ),
            (0.28, 0.43),
        ),
        "MMPP2 negative": (
            build_demand_config(
                demand_dist_name="MarkovModulatedPoisson2",
                demand_rate=5.0,
                demand_lambda_low=3.0,
                demand_lambda_high=7.0,
                demand_p00=0.1,
                demand_p11=0.1,
            ),
            (-0.43, -0.28),
        ),
    }

    for seed, (config, bounds) in enumerate(configs.values(), start=123):
        rng = np.random.RandomState(seed)
        process = build_demand_process(config, rng=rng)
        samples = process.sample_path(50000)
        rho1 = _lag_one_corr(samples)
        assert bounds[0] <= rho1 <= bounds[1]


def test_mmpp2_cumulative_demand_cdf_matches_one_period_stationary_mixture():
    config = build_demand_config(
        demand_dist_name="MarkovModulatedPoisson2",
        demand_rate=5.0,
        demand_lambda_low=3.0,
        demand_lambda_high=7.0,
        demand_p00=0.9,
        demand_p11=0.9,
    )
    probs, _, _ = get_demand_prob_vector(config)
    cdf_from_pmf = float(np.cumsum(probs)[10])
    cdf_from_helper = get_cumulative_demand_cdf(config, k=10, periods=1)
    assert cdf_from_helper == pytest.approx(cdf_from_pmf)
