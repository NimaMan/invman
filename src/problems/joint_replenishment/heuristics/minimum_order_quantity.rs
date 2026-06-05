use std::cmp::Ordering;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::joint_replenishment::env::JointReplenishmentState;

fn allocate_proportionally(total_quantity: usize, deficits: &[usize]) -> Vec<usize> {
    if total_quantity == 0 || deficits.iter().all(|value| *value == 0) {
        return vec![0; deficits.len()];
    }

    let total_deficit = deficits.iter().sum::<usize>() as f64;
    let mut base = vec![0usize; deficits.len()];
    let mut remainders = Vec::with_capacity(deficits.len());
    let mut assigned = 0usize;

    for (index, deficit) in deficits.iter().copied().enumerate() {
        let exact_share = total_quantity as f64 * deficit as f64 / total_deficit;
        let floor_share = exact_share.floor() as usize;
        base[index] = floor_share;
        assigned += floor_share;
        remainders.push((index, exact_share - floor_share as f64));
    }

    remainders.sort_by(|lhs, rhs| {
        rhs.1
            .partial_cmp(&lhs.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.0.cmp(&rhs.0))
    });

    for (index, _) in remainders
        .into_iter()
        .take(total_quantity.saturating_sub(assigned))
    {
        base[index] += 1;
    }
    base
}

pub fn minimum_order_quantity_order_quantities(
    state: &JointReplenishmentState,
    item_targets: &[usize],
    review_period: usize,
    rounding_threshold: f64,
    truck_capacity: usize,
) -> PyResult<Vec<usize>> {
    if item_targets.len() != state.inventory_levels.len() {
        return Err(PyValueError::new_err(
            "item_targets length must match the number of items",
        ));
    }
    if review_period == 0 {
        return Err(PyValueError::new_err(
            "review_period must be strictly positive",
        ));
    }
    if truck_capacity == 0 {
        return Err(PyValueError::new_err(
            "truck_capacity must be strictly positive",
        ));
    }
    if !rounding_threshold.is_finite() || rounding_threshold < 0.0 {
        return Err(PyValueError::new_err(
            "rounding_threshold must be finite and non-negative",
        ));
    }

    if state.period % review_period != 0 {
        return Ok(vec![0; item_targets.len()]);
    }

    let deficits = item_targets
        .iter()
        .zip(state.inventory_levels.iter())
        .map(|(target, inventory)| (*target as i32 - *inventory).max(0) as usize)
        .collect::<Vec<_>>();
    let raw_gap = deficits.iter().sum::<usize>();
    if raw_gap == 0 {
        return Ok(vec![0; item_targets.len()]);
    }

    let full_trucks = raw_gap / truck_capacity;
    let remainder = raw_gap % truck_capacity;
    let trucks_to_order = if remainder as f64 > rounding_threshold {
        full_trucks + 1
    } else {
        full_trucks
    };
    let total_quantity = trucks_to_order * truck_capacity;
    Ok(allocate_proportionally(total_quantity, &deficits))
}
