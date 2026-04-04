mod base_stock;
mod bsp_low_ew;

pub use base_stock::base_stock_order_quantity;
pub use bsp_low_ew::bsp_low_ew_order_quantity;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::perishable_inventory::env::{step_state, IssuingPolicy, PerishableState};

fn mean_after_warmup(epoch_costs: &[f64], warm_up_periods_ratio: f64) -> PyResult<f64> {
    if epoch_costs.is_empty() {
        return Err(PyValueError::new_err("epoch_costs must be non-empty"));
    }
    if !(0.0..=1.0).contains(&warm_up_periods_ratio) {
        return Err(PyValueError::new_err(
            "warm_up_periods_ratio must be in [0, 1]",
        ));
    }
    let horizon = epoch_costs.len();
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let active_costs = if warm_up_periods < epoch_costs.len() {
        &epoch_costs[warm_up_periods..]
    } else {
        epoch_costs
    };
    Ok(active_costs.iter().sum::<f64>() / active_costs.len() as f64)
}

pub fn policy_rollout_from_demands(
    policy_name: &str,
    params: &[usize],
    initial_state: &PerishableState,
    demands: &[usize],
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: IssuingPolicy,
) -> PyResult<f64> {
    let mut state = initial_state.clone();
    let mut epoch_costs = Vec::with_capacity(demands.len());

    for demand in demands.iter().copied() {
        let order_quantity = match policy_name {
            "base_stock" => {
                if params.len() != 1 {
                    return Err(PyValueError::new_err("base_stock expects params [S]"));
                }
                base_stock_order_quantity(&state, params[0], max_order_size)
            }
            "bsp_low_ew" => {
                if params.len() != 3 {
                    return Err(PyValueError::new_err(
                        "bsp_low_ew expects params [S1, S2, b]",
                    ));
                }
                bsp_low_ew_order_quantity(
                    &state,
                    params[0],
                    params[1],
                    params[2],
                    max_order_size,
                    lead_time,
                    demand_mean,
                    issuing_policy,
                )
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown perishable-inventory policy '{policy_name}'"
                )))
            }
        };

        let outcome = step_state(
            &state,
            order_quantity,
            demand,
            holding_cost,
            shortage_cost,
            waste_cost,
            procurement_cost,
            issuing_policy,
        );
        epoch_costs.push(outcome.cost);
        state = outcome.next_state;
    }

    mean_after_warmup(&epoch_costs, warm_up_periods_ratio)
}

pub fn search_base_stock_from_demands(
    initial_state: &PerishableState,
    demands: &[usize],
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: IssuingPolicy,
    position_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, f64), Vec<(usize, f64)>)> {
    if top_k == 0 {
        return Err(PyValueError::new_err("top_k must be positive"));
    }
    let mut results = Vec::new();
    for level in 0..=position_upper_bound {
        let mean_cost = policy_rollout_from_demands(
            "base_stock",
            &[level],
            initial_state,
            demands,
            lead_time,
            max_order_size,
            demand_mean,
            holding_cost,
            shortage_cost,
            waste_cost,
            procurement_cost,
            warm_up_periods_ratio,
            issuing_policy,
        )?;
        results.push((level, mean_cost));
    }
    results.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    Ok((results[0], results.into_iter().take(top_k).collect()))
}

pub fn search_bsp_low_ew_from_demands(
    initial_state: &PerishableState,
    demands: &[usize],
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: IssuingPolicy,
    position_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, usize, usize, f64), Vec<(usize, usize, usize, f64)>)> {
    if top_k == 0 {
        return Err(PyValueError::new_err("top_k must be positive"));
    }
    let mut results = Vec::new();
    for low_inventory_level in 0..=position_upper_bound {
        for high_inventory_level in 0..=low_inventory_level {
            for threshold in 0..=position_upper_bound {
                let mean_cost = policy_rollout_from_demands(
                    "bsp_low_ew",
                    &[low_inventory_level, high_inventory_level, threshold],
                    initial_state,
                    demands,
                    lead_time,
                    max_order_size,
                    demand_mean,
                    holding_cost,
                    shortage_cost,
                    waste_cost,
                    procurement_cost,
                    warm_up_periods_ratio,
                    issuing_policy,
                )?;
                results.push((
                    low_inventory_level,
                    high_inventory_level,
                    threshold,
                    mean_cost,
                ));
            }
        }
    }
    results.sort_by(|left, right| {
        left.3
            .total_cmp(&right.3)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    Ok((results[0], results.into_iter().take(top_k).collect()))
}
