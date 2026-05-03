use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::joint_replenishment::demand::{support, DemandRange};
use crate::problems::joint_replenishment::env::{total_inventory, JointReplenishmentState};

fn expected_single_item_cost(
    post_order_inventory: i32,
    demand_range: DemandRange,
    holding_cost: f64,
    shortage_cost: f64,
) -> PyResult<f64> {
    Ok(support(demand_range)?
        .iter()
        .map(|(demand, probability)| {
            let ending_inventory = post_order_inventory - *demand as i32;
            probability
                * (holding_cost * ending_inventory.max(0) as f64
                    + shortage_cost * (-ending_inventory).max(0) as f64)
        })
        .sum())
}

fn myopic_allocation(
    state: &JointReplenishmentState,
    item_targets: &[usize],
    total_quantity: usize,
    demand_ranges: &[DemandRange],
    holding_costs: &[f64],
    shortage_costs: &[f64],
) -> PyResult<Vec<usize>> {
    let num_items = state.inventory_levels.len();
    let mut value = vec![vec![f64::INFINITY; total_quantity + 1]; num_items + 1];
    let mut deviation = vec![vec![u64::MAX; total_quantity + 1]; num_items + 1];
    let mut choice = vec![vec![0usize; total_quantity + 1]; num_items];
    value[0][0] = 0.0;
    deviation[0][0] = 0;

    for item_idx in 0..num_items {
        for assigned_quantity in 0..=total_quantity {
            if !value[item_idx][assigned_quantity].is_finite() {
                continue;
            }
            for item_quantity in 0..=total_quantity - assigned_quantity {
                let cost = expected_single_item_cost(
                    state.inventory_levels[item_idx] + item_quantity as i32,
                    demand_ranges[item_idx],
                    holding_costs[item_idx],
                    shortage_costs[item_idx],
                )?;
                let candidate = value[item_idx][assigned_quantity] + cost;
                let post_order_inventory = state.inventory_levels[item_idx] + item_quantity as i32;
                let target_gap = post_order_inventory - item_targets[item_idx] as i32;
                let candidate_deviation = deviation[item_idx][assigned_quantity]
                    + (target_gap as i64 * target_gap as i64) as u64;
                if candidate < value[item_idx + 1][assigned_quantity + item_quantity] - 1e-12
                    || ((candidate - value[item_idx + 1][assigned_quantity + item_quantity]).abs()
                        < 1e-12
                        && candidate_deviation
                            < deviation[item_idx + 1][assigned_quantity + item_quantity])
                {
                    value[item_idx + 1][assigned_quantity + item_quantity] = candidate;
                    deviation[item_idx + 1][assigned_quantity + item_quantity] =
                        candidate_deviation;
                    choice[item_idx][assigned_quantity + item_quantity] = item_quantity;
                }
            }
        }
    }

    let mut remaining = total_quantity;
    let mut allocation = vec![0usize; num_items];
    for item_idx in (0..num_items).rev() {
        let quantity = choice[item_idx][remaining];
        allocation[item_idx] = quantity;
        remaining -= quantity;
    }
    Ok(allocation)
}

pub fn dynamic_order_up_to_order_quantities(
    state: &JointReplenishmentState,
    item_targets: &[usize],
    truck_capacity: usize,
    demand_ranges: &[DemandRange],
    holding_costs: &[f64],
    shortage_costs: &[f64],
) -> PyResult<Vec<usize>> {
    let num_items = state.inventory_levels.len();
    if item_targets.len() != num_items
        || demand_ranges.len() != num_items
        || holding_costs.len() != num_items
        || shortage_costs.len() != num_items
    {
        return Err(PyValueError::new_err(
            "all item-wise arrays must match the number of items",
        ));
    }
    if truck_capacity == 0 {
        return Err(PyValueError::new_err(
            "truck_capacity must be strictly positive",
        ));
    }

    let aggregate_target = item_targets.iter().sum::<usize>() as i32;
    let aggregate_gap = aggregate_target - total_inventory(state);
    let trucks = ((aggregate_gap as f64) / truck_capacity as f64)
        .round()
        .max(0.0) as usize;
    let total_quantity = trucks * truck_capacity;
    if total_quantity == 0 {
        return Ok(vec![0; num_items]);
    }

    myopic_allocation(
        state,
        item_targets,
        total_quantity,
        demand_ranges,
        holding_costs,
        shortage_costs,
    )
}
