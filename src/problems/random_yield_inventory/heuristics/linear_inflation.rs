use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use statrs::distribution::{DiscreteCDF, Poisson};

use crate::problems::random_yield_inventory::env::{
    expected_inventory_position, round_order_quantity, RandomYieldInventoryState,
};

pub fn lead_time_target_stock_level(
    demand_mean: f64,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
) -> PyResult<f64> {
    if !demand_mean.is_finite() || demand_mean < 0.0 {
        return Err(PyValueError::new_err(
            "demand_mean must be finite and non-negative",
        ));
    }
    if holding_cost < 0.0 || shortage_cost < 0.0 {
        return Err(PyValueError::new_err(
            "holding_cost and shortage_cost must be non-negative",
        ));
    }
    let critical_ratio = if holding_cost + shortage_cost == 0.0 {
        0.5
    } else {
        shortage_cost / (holding_cost + shortage_cost)
    };
    let lead_time_mean = demand_mean * (lead_time as f64 + 1.0);
    if lead_time_mean == 0.0 {
        return Ok(0.0);
    }
    let distribution = Poisson::new(lead_time_mean).map_err(|err| {
        PyValueError::new_err(format!("invalid Poisson mean {lead_time_mean}: {err}"))
    })?;
    Ok(distribution.inverse_cdf(critical_ratio.clamp(0.0, 1.0)) as f64)
}

pub fn linear_inflation_order_quantity(
    state: &RandomYieldInventoryState,
    success_probability: f64,
    target_stock_level: f64,
    yield_inflation_factor: f64,
) -> PyResult<f64> {
    if !yield_inflation_factor.is_finite() || yield_inflation_factor < 0.0 {
        return Err(PyValueError::new_err(
            "yield_inflation_factor must be finite and non-negative",
        ));
    }
    let expected_position = expected_inventory_position(state, success_probability)?;
    let raw_gap = (target_stock_level - expected_position).max(0.0);
    Ok(round_order_quantity(yield_inflation_factor * raw_gap))
}

pub fn yield_inflated_base_stock_parameters(
    demand_mean: f64,
    success_probability: f64,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
) -> PyResult<(f64, f64)> {
    if !success_probability.is_finite() || !(0.0..=1.0).contains(&success_probability) {
        return Err(PyValueError::new_err(
            "success_probability must lie in [0, 1]",
        ));
    }
    if success_probability == 0.0 {
        return Err(PyValueError::new_err(
            "success_probability must be strictly positive for linear inflation",
        ));
    }
    Ok((
        lead_time_target_stock_level(demand_mean, lead_time, holding_cost, shortage_cost)?,
        1.0 / success_probability,
    ))
}

pub fn yield_inflated_base_stock_order_quantity(
    state: &RandomYieldInventoryState,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
) -> PyResult<f64> {
    let (target_stock_level, yield_inflation_factor) = yield_inflated_base_stock_parameters(
        demand_mean,
        success_probability,
        state.pipeline_orders.len(),
        holding_cost,
        shortage_cost,
    )?;
    linear_inflation_order_quantity(
        state,
        success_probability,
        target_stock_level,
        yield_inflation_factor,
    )
}
