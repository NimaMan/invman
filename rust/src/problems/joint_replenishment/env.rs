use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Debug, PartialEq)]
pub struct JointReplenishmentState {
    pub period: usize,
    pub inventory_levels: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JointReplenishmentStepOutcome {
    pub next_state: JointReplenishmentState,
    pub order_quantities: Vec<usize>,
    pub realized_demands: Vec<usize>,
    pub trucks_used: usize,
    pub total_order_quantity: usize,
    pub order_cost: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub period_cost: f64,
    pub reward: f64,
}

fn validate_cost_vector(name: &str, values: &[f64], expected_len: usize) -> PyResult<()> {
    if values.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "{name} length {} does not match expected {}",
            values.len(),
            expected_len
        )));
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(format!(
            "{name} must contain finite non-negative values",
        )));
    }
    Ok(())
}

pub fn validate_state(state: &JointReplenishmentState) -> PyResult<()> {
    if state.inventory_levels.is_empty() {
        return Err(PyValueError::new_err(
            "joint_replenishment state must contain at least one item",
        ));
    }
    Ok(())
}

pub fn initialize_state(initial_inventory_levels: &[i32]) -> PyResult<JointReplenishmentState> {
    let state = JointReplenishmentState {
        period: 0,
        inventory_levels: initial_inventory_levels.to_vec(),
    };
    validate_state(&state)?;
    Ok(state)
}

pub fn total_inventory(state: &JointReplenishmentState) -> i32 {
    state.inventory_levels.iter().sum()
}

pub fn build_raw_state(state: &JointReplenishmentState) -> PyResult<Vec<f32>> {
    validate_state(state)?;
    let mut raw_state = state
        .inventory_levels
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    raw_state.push(state.period as f32);
    Ok(raw_state)
}

pub fn trucks_required(order_quantities: &[usize], truck_capacity: usize) -> PyResult<usize> {
    if truck_capacity == 0 {
        return Err(PyValueError::new_err(
            "truck_capacity must be strictly positive",
        ));
    }
    let total_order_quantity = order_quantities.iter().sum::<usize>();
    if total_order_quantity == 0 {
        Ok(0)
    } else if total_order_quantity % truck_capacity != 0 {
        Err(PyValueError::new_err(format!(
            "total order quantity {total_order_quantity} must be zero or an exact multiple of truck_capacity {truck_capacity}",
        )))
    } else {
        Ok(total_order_quantity / truck_capacity)
    }
}

pub fn step_state(
    state: &JointReplenishmentState,
    order_quantities: &[usize],
    realized_demands: &[usize],
    truck_capacity: usize,
    minor_order_costs: &[f64],
    major_order_cost: f64,
    holding_costs: &[f64],
    shortage_costs: &[f64],
) -> PyResult<JointReplenishmentStepOutcome> {
    validate_state(state)?;
    let num_items = state.inventory_levels.len();
    if order_quantities.len() != num_items {
        return Err(PyValueError::new_err(format!(
            "order_quantities length {} does not match num_items {}",
            order_quantities.len(),
            num_items
        )));
    }
    if realized_demands.len() != num_items {
        return Err(PyValueError::new_err(format!(
            "realized_demands length {} does not match num_items {}",
            realized_demands.len(),
            num_items
        )));
    }
    if !major_order_cost.is_finite() || major_order_cost < 0.0 {
        return Err(PyValueError::new_err(
            "major_order_cost must be finite and non-negative",
        ));
    }
    validate_cost_vector("minor_order_costs", minor_order_costs, num_items)?;
    validate_cost_vector("holding_costs", holding_costs, num_items)?;
    validate_cost_vector("shortage_costs", shortage_costs, num_items)?;

    let trucks_used = trucks_required(order_quantities, truck_capacity)?;
    let total_order_quantity = order_quantities.iter().sum::<usize>();
    let order_cost = major_order_cost * trucks_used as f64
        + order_quantities
            .iter()
            .zip(minor_order_costs.iter())
            .map(|(quantity, cost)| if *quantity > 0 { *cost } else { 0.0 })
            .sum::<f64>();

    let mut next_inventory_levels = Vec::with_capacity(num_items);
    let mut holding_cost = 0.0;
    let mut shortage_cost = 0.0;

    for item_idx in 0..num_items {
        let ending_inventory = state.inventory_levels[item_idx] + order_quantities[item_idx] as i32
            - realized_demands[item_idx] as i32;
        holding_cost += holding_costs[item_idx] * ending_inventory.max(0) as f64;
        shortage_cost += shortage_costs[item_idx] * (-ending_inventory).max(0) as f64;
        next_inventory_levels.push(ending_inventory);
    }

    let period_cost = order_cost + holding_cost + shortage_cost;
    Ok(JointReplenishmentStepOutcome {
        next_state: JointReplenishmentState {
            period: state.period + 1,
            inventory_levels: next_inventory_levels,
        },
        order_quantities: order_quantities.to_vec(),
        realized_demands: realized_demands.to_vec(),
        trucks_used,
        total_order_quantity,
        order_cost,
        holding_cost,
        shortage_cost,
        period_cost,
        reward: -period_cost,
    })
}
