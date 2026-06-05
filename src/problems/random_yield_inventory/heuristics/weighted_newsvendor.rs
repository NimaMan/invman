use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use statrs::distribution::{Discrete, Poisson};

use crate::problems::random_yield_inventory::env::{
    round_order_quantity, RandomYieldInventoryState,
};
use crate::problems::random_yield_inventory::heuristics::lead_time_target_stock_level;

fn enumerate_pipeline_arrival_scenarios(
    pipeline_orders: &[f64],
    success_probability: f64,
) -> PyResult<Vec<(f64, f64)>> {
    if pipeline_orders.len() > 20 {
        return Err(PyValueError::new_err(
            "weighted_newsvendor currently supports lead times up to 20 for exact scenario enumeration",
        ));
    }
    let mut scenarios = Vec::with_capacity(1usize << pipeline_orders.len());
    for mask in 0..(1usize << pipeline_orders.len()) {
        let mut probability = 1.0;
        let mut realized_pipeline = 0.0;
        for (index, order_quantity) in pipeline_orders.iter().enumerate() {
            let succeeds = ((mask >> index) & 1) == 1;
            if succeeds {
                probability *= success_probability;
                realized_pipeline += *order_quantity;
            } else {
                probability *= 1.0 - success_probability;
            }
        }
        scenarios.push((probability, realized_pipeline));
    }
    Ok(scenarios)
}

pub fn weighted_newsvendor_order_quantity(
    state: &RandomYieldInventoryState,
    demand_mean: f64,
    success_probability: f64,
    holding_cost: f64,
    shortage_cost: f64,
) -> PyResult<f64> {
    if !success_probability.is_finite() || !(0.0..=1.0).contains(&success_probability) {
        return Err(PyValueError::new_err(
            "success_probability must lie in [0, 1]",
        ));
    }
    let lead_time = state.pipeline_orders.len();
    let target_stock_level =
        lead_time_target_stock_level(demand_mean, lead_time, holding_cost, shortage_cost)?;
    let pipeline_scenarios =
        enumerate_pipeline_arrival_scenarios(&state.pipeline_orders, success_probability)?;
    let cumulative_demand_mean = demand_mean * lead_time as f64;
    let demand_distribution = Poisson::new(cumulative_demand_mean.max(1e-12)).map_err(|err| {
        PyValueError::new_err(format!(
            "invalid cumulative demand mean {cumulative_demand_mean}: {err}"
        ))
    })?;
    let upper_demand = (cumulative_demand_mean + 10.0 * cumulative_demand_mean.sqrt() + 10.0)
        .ceil()
        .max(0.0) as u64;
    let mut expected_gap = 0.0;
    let mut probability_mass = 0.0f64;

    for demand in 0..=upper_demand {
        let demand_probability = if demand == upper_demand {
            (1.0 - probability_mass).max(0.0)
        } else {
            demand_distribution.pmf(demand)
        };
        probability_mass += demand_probability;
        for (scenario_probability, realized_pipeline) in &pipeline_scenarios {
            let projected_inventory = state.inventory_level + realized_pipeline - demand as f64;
            expected_gap += scenario_probability
                * demand_probability
                * (target_stock_level - projected_inventory).max(0.0);
        }
    }

    Ok(round_order_quantity(expected_gap))
}
