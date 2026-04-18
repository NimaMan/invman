use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::decentralized_inventory_control::env::{
    on_order_items, DecentralizedInventoryControlState,
};

fn round_half_away_from_zero(value: f64) -> f64 {
    if value >= 0.0 {
        (value + 0.5).floor()
    } else {
        (value - 0.5).ceil()
    }
}

pub fn sterman_anchor_adjust_orders(
    state: &DecentralizedInventoryControlState,
    current_received_orders: &[usize],
    target_positions: &[f64],
    adjustment_times: &[f64],
    supply_line_weights: &[f64],
) -> PyResult<Vec<usize>> {
    let num_agents = state.on_hand_inventory.len();
    if current_received_orders.len() != num_agents
        || target_positions.len() != num_agents
        || adjustment_times.len() != num_agents
        || supply_line_weights.len() != num_agents
    {
        return Err(PyValueError::new_err(
            "current_received_orders and all Sterman parameter vectors must match the number of agents",
        ));
    }
    if adjustment_times
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(PyValueError::new_err(
            "adjustment_times must be finite and strictly positive",
        ));
    }

    let on_order = on_order_items(state)?;
    Ok((0..num_agents)
        .map(|agent_idx| {
            let effective_inventory =
                state.on_hand_inventory[agent_idx] as f64 - state.backlog[agent_idx] as f64;
            let raw_order = current_received_orders[agent_idx] as f64
                + (target_positions[agent_idx]
                    - effective_inventory
                    - supply_line_weights[agent_idx] * on_order[agent_idx] as f64)
                    / adjustment_times[agent_idx];
            round_half_away_from_zero(raw_order).max(0.0) as usize
        })
        .collect())
}
