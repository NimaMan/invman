use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use statrs::distribution::{Discrete, Poisson};

use crate::problems::nonstationary_lot_sizing::demand::{
    sample_demand, DemandDistributionKind,
};
use crate::problems::nonstationary_lot_sizing::env::{
    inventory_position, step_state, validate_state, NonstationaryLotSizingState,
};
use super::{s_s_order_quantity, PolicySimulationSummary};

const DEMAND_TAIL_MASS: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RollingDpPolicyLevels {
    pub reorder_point: i32,
    pub order_up_to: i32,
}

fn validate_forecast_window(forecast_window: &[f64]) -> PyResult<()> {
    if forecast_window.is_empty() {
        return Err(PyValueError::new_err(
            "forecast_window must contain at least one period",
        ));
    }
    if forecast_window
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "forecast_window must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_forecast_path(
    forecast_means: &[f64],
    periods: usize,
    forecast_horizon: usize,
) -> PyResult<()> {
    let required_len = periods + forecast_horizon;
    if forecast_means.len() < required_len {
        return Err(PyValueError::new_err(format!(
            "forecast path length {} is smaller than required {}",
            forecast_means.len(),
            required_len
        )));
    }
    if forecast_means
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "forecast_means must be finite and non-negative",
        ));
    }
    Ok(())
}

fn poisson_support(mean: f64) -> PyResult<Vec<(i32, f64)>> {
    if !mean.is_finite() || mean < 0.0 {
        return Err(PyValueError::new_err(
            "Poisson mean must be finite and non-negative",
        ));
    }
    if mean == 0.0 {
        return Ok(vec![(0, 1.0)]);
    }

    let poisson = Poisson::new(mean).map_err(|err| {
        PyValueError::new_err(format!("invalid Poisson mean {mean}: {err}"))
    })?;
    let mut support = Vec::new();
    let mut cumulative_probability = 0.0;
    let mut probability = (-mean).exp();
    support.push((0, probability));
    cumulative_probability += probability;

    for demand in 1..=10_000 {
        probability *= mean / demand as f64;
        let tail_probability = (1.0 - cumulative_probability - probability).max(0.0);
        if tail_probability <= DEMAND_TAIL_MASS {
            support.push((demand, (1.0 - cumulative_probability).max(0.0)));
            break;
        }
        support.push((demand, probability));
        cumulative_probability += probability;
    }

    let total_probability = support.iter().map(|(_, p)| *p).sum::<f64>();
    if (total_probability - 1.0).abs() > 1e-10 {
        return Err(PyValueError::new_err(format!(
            "Poisson support for mean {mean} summed to {total_probability}, expected 1"
        )));
    }
    if support.is_empty() {
        let fallback_tail = poisson.pmf(0);
        return Ok(vec![(0, fallback_tail)]);
    }
    Ok(support)
}

fn build_augmented_forecast(
    forecast_window: &[f64],
    stationary_tail_periods: usize,
) -> Vec<f64> {
    let mean_forecast = forecast_window.iter().sum::<f64>() / forecast_window.len() as f64;
    let mut augmented = forecast_window.to_vec();
    augmented.extend(std::iter::repeat(mean_forecast).take(stationary_tail_periods));
    augmented
}

fn implied_inventory_bounds(
    augmented_forecast: &[f64],
    lead_time: usize,
    holding_cost: f64,
    fixed_order_cost: f64,
) -> PyResult<(i32, i32)> {
    let max_single_support = augmented_forecast
        .iter()
        .map(|mean| poisson_support(*mean))
        .collect::<PyResult<Vec<_>>>()?
        .iter()
        .map(|dist| dist.last().map(|(demand, _)| *demand).unwrap_or(0))
        .max()
        .unwrap_or(0);
    let max_cumulative_mean = augmented_forecast
        .windows(lead_time + 1)
        .map(|window| window.iter().sum::<f64>())
        .fold(0.0, f64::max);
    let max_cumulative_support = poisson_support(max_cumulative_mean)?
        .last()
        .map(|(demand, _)| *demand)
        .unwrap_or(0);
    let mean_forecast = augmented_forecast.iter().sum::<f64>() / augmented_forecast.len() as f64;
    let eoq = if holding_cost > 0.0 {
        (2.0 * mean_forecast * fixed_order_cost.max(0.0) / holding_cost).sqrt()
    } else {
        0.0
    };

    let min_inventory_position = -((2 * max_cumulative_support).max(4 * max_single_support).max(32));
    let max_inventory_position =
        (max_cumulative_support as f64 + eoq + (2 * max_single_support) as f64).ceil() as i32;

    Ok((min_inventory_position, max_inventory_position.max(32)))
}

fn expected_stage_cost(
    order_up_to: i32,
    cumulative_support: &[(i32, f64)],
    holding_cost: f64,
    shortage_cost: f64,
) -> f64 {
    cumulative_support
        .iter()
        .map(|(demand, probability)| {
            let inventory_delta = order_up_to - demand;
            let holding = inventory_delta.max(0) as f64;
            let shortage = (-inventory_delta).max(0) as f64;
            probability * (holding_cost * holding + shortage_cost * shortage)
        })
        .sum()
}

fn expected_future_cost(
    next_values: &[f64],
    order_up_to: i32,
    single_period_support: &[(i32, f64)],
    min_inventory_position: i32,
    max_inventory_position: i32,
    holding_cost: f64,
    shortage_cost: f64,
    discount_factor: f64,
) -> f64 {
    let min_index = 0usize;
    let max_index = next_values.len() - 1;

    single_period_support
        .iter()
        .map(|(demand, probability)| {
            let next_inventory_position = order_up_to - demand;
            let continuation_cost = if next_inventory_position < min_inventory_position {
                let extra_shortage_units = (min_inventory_position - next_inventory_position) as f64;
                next_values[min_index] + shortage_cost * extra_shortage_units / (1.0 - discount_factor)
            } else if next_inventory_position > max_inventory_position {
                let extra_inventory_units = (next_inventory_position - max_inventory_position) as f64;
                next_values[max_index] + holding_cost * extra_inventory_units / (1.0 - discount_factor)
            } else {
                next_values[(next_inventory_position - min_inventory_position) as usize]
            };
            probability * continuation_cost
        })
        .sum()
}

pub fn rolling_dp_s_s_levels(
    forecast_window: &[f64],
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    demand_kind: DemandDistributionKind,
    discount_factor: f64,
    stationary_tail_periods: usize,
) -> PyResult<RollingDpPolicyLevels> {
    validate_forecast_window(forecast_window)?;
    if lead_time == 0 {
        return Err(PyValueError::new_err(
            "rolling_dp_s_s requires lead_time >= 1",
        ));
    }
    if demand_kind != DemandDistributionKind::Poisson {
        return Err(PyValueError::new_err(
            "rolling_dp_s_s currently supports only Poisson demand",
        ));
    }
    if !(0.0..1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err(
            "discount_factor must lie in [0, 1)",
        ));
    }

    let augmented_forecast = build_augmented_forecast(forecast_window, stationary_tail_periods);
    if augmented_forecast.len() <= lead_time {
        return Err(PyValueError::new_err(
            "augmented forecast must be longer than the lead time",
        ));
    }

    let (min_inventory_position, max_inventory_position) =
        implied_inventory_bounds(&augmented_forecast, lead_time, holding_cost, fixed_order_cost)?;
    let states = (min_inventory_position..=max_inventory_position).collect::<Vec<_>>();
    let decision_periods = augmented_forecast.len() - lead_time;
    let single_period_supports = augmented_forecast[..decision_periods]
        .iter()
        .map(|mean| poisson_support(*mean))
        .collect::<PyResult<Vec<_>>>()?;
    let cumulative_supports = (0..decision_periods)
        .map(|period| poisson_support(augmented_forecast[period..=period + lead_time].iter().sum()))
        .collect::<PyResult<Vec<_>>>()?;

    let mut value_function = vec![vec![0.0; states.len()]; decision_periods + 1];
    let mut optimal_order_up_to = vec![vec![0i32; states.len()]; decision_periods];

    for period in (0..decision_periods).rev() {
        let (current_and_past, future) = value_function.split_at_mut(period + 1);
        let current_values = &mut current_and_past[period];
        let continuation_values = &future[0];
        let single_period_support = &single_period_supports[period];
        let cumulative_support = &cumulative_supports[period];

        let stage_costs = states
            .iter()
            .map(|inventory_position| {
                expected_stage_cost(
                    *inventory_position,
                    cumulative_support,
                    holding_cost,
                    shortage_cost,
                )
            })
            .collect::<Vec<_>>();

        for (state_index, inventory_position) in states.iter().enumerate() {
            let mut best_value = f64::INFINITY;
            let mut best_order_up_to = *inventory_position;

            for order_up_to in *inventory_position..=max_inventory_position {
                let order_cost = if order_up_to > *inventory_position {
                    fixed_order_cost
                } else {
                    0.0
                };
                let continuation_cost = expected_future_cost(
                    continuation_values,
                    order_up_to,
                    single_period_support,
                    min_inventory_position,
                    max_inventory_position,
                    holding_cost,
                    shortage_cost,
                    discount_factor,
                );
                let total_cost = order_cost
                    + stage_costs[(order_up_to - min_inventory_position) as usize]
                    + discount_factor * continuation_cost;

                if total_cost < best_value - 1e-12 {
                    best_value = total_cost;
                    best_order_up_to = order_up_to;
                }
            }

            current_values[state_index] = best_value;
            optimal_order_up_to[period][state_index] = best_order_up_to;
        }
    }

    let first_period_orders = &optimal_order_up_to[0];
    let ordering_states = states
        .iter()
        .zip(first_period_orders.iter())
        .filter_map(|(inventory_position, order_up_to)| {
            if order_up_to > inventory_position {
                Some((*inventory_position, *order_up_to))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if ordering_states.is_empty() {
        return Err(PyValueError::new_err(
            "rolling DP did not find an ordering region",
        ));
    }

    let order_up_to = ordering_states[0].1;
    if ordering_states
        .iter()
        .any(|(_, candidate_order_up_to)| *candidate_order_up_to != order_up_to)
    {
        return Err(PyValueError::new_err(
            "rolling DP first-period policy is not an (s,S) rule on the computed grid",
        ));
    }

    let reorder_point = ordering_states
        .iter()
        .map(|(inventory_position, _)| *inventory_position)
        .max()
        .unwrap_or(order_up_to);

    Ok(RollingDpPolicyLevels {
        reorder_point,
        order_up_to,
    })
}

pub fn rolling_dp_s_s_sequence(
    forecast_means: &[f64],
    periods: usize,
    forecast_horizon: usize,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    demand_kind: DemandDistributionKind,
    discount_factor: f64,
    stationary_tail_periods: usize,
) -> PyResult<Vec<RollingDpPolicyLevels>> {
    validate_forecast_path(forecast_means, periods, forecast_horizon)?;
    let mut cache = HashMap::new();
    let mut sequence = Vec::with_capacity(periods);

    for period in 0..periods {
        let forecast_window = &forecast_means[period..period + forecast_horizon];
        let cache_key = forecast_window
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let levels = if let Some(cached) = cache.get(&cache_key) {
            *cached
        } else {
            let solved = rolling_dp_s_s_levels(
                forecast_window,
                lead_time,
                holding_cost,
                shortage_cost,
                fixed_order_cost,
                demand_kind,
                discount_factor,
                stationary_tail_periods,
            )?;
            cache.insert(cache_key, solved);
            solved
        };
        sequence.push(levels);
    }

    Ok(sequence)
}

pub fn simulate_periodic_s_s_policy(
    levels: &[RollingDpPolicyLevels],
    initial_state: &NonstationaryLotSizingState,
    forecast_means: &[f64],
    replications: usize,
    seed: u64,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    lost_sales: bool,
    demand_cv: f64,
    demand_kind: DemandDistributionKind,
) -> PyResult<PolicySimulationSummary> {
    validate_state(
        initial_state,
        initial_state.forecast_window.len(),
        initial_state.pipeline_orders.len(),
    )?;
    validate_forecast_path(forecast_means, levels.len(), initial_state.forecast_window.len())?;
    if replications == 0 {
        return Err(PyValueError::new_err("replications must be at least 1"));
    }
    if levels.iter().any(|level| level.order_up_to < level.reorder_point) {
        return Err(PyValueError::new_err(
            "periodic (s,S) levels must satisfy order_up_to >= reorder_point",
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut returns = Vec::with_capacity(replications);
    let mut total_shortage = 0.0;
    let mut total_demand = 0.0;

    for _ in 0..replications {
        let mut state = initial_state.clone();
        let mut total_cost = 0.0;

        for (period, level) in levels.iter().enumerate() {
            let realized_demand =
                sample_demand(&mut rng, forecast_means[period], demand_cv, demand_kind)?;
            let order_quantity = s_s_order_quantity(
                inventory_position(&state),
                level.reorder_point as f64,
                level.order_up_to as f64,
            );
            let next_forecast_mean = forecast_means[period + state.forecast_window.len()];
            let outcome = step_state(
                &state,
                order_quantity,
                realized_demand,
                next_forecast_mean,
                holding_cost,
                shortage_cost,
                procurement_cost,
                fixed_order_cost,
                lost_sales,
            )?;
            total_cost += outcome.period_cost;
            total_shortage += outcome.unmet_demand;
            total_demand += realized_demand;
            state = outcome.next_state;
        }

        returns.push(total_cost);
    }

    let mean_cost = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|value| (value - mean_cost).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    Ok(PolicySimulationSummary {
        mean_cost,
        cost_std: variance.sqrt(),
        shortage_rate: if total_demand > 0.0 {
            total_shortage / total_demand
        } else {
            0.0
        },
    })
}
