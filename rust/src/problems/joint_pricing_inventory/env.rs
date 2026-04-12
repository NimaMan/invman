use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::joint_pricing_inventory::demand::validate_price_ladder;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JointPricingInventoryState {
    pub period: usize,
    pub inventory_level: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JointPricingInventoryStepOutcome {
    pub next_state: JointPricingInventoryState,
    pub order_quantity: usize,
    pub price_index: usize,
    pub selling_price: f64,
    pub realized_demand: usize,
    pub inventory_after_order: usize,
    pub sales: usize,
    pub lost_sales: usize,
    pub revenue: f64,
    pub procurement_cost: f64,
    pub holding_cost: f64,
    pub stockout_cost: f64,
    pub period_cost: f64,
    pub reward: f64,
}

pub fn validate_costs(
    procurement_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    salvage_value_per_unit: f64,
) -> PyResult<()> {
    let values = [
        procurement_cost_per_unit,
        holding_cost_per_unit,
        stockout_cost_per_unit,
        salvage_value_per_unit,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "all costs and salvage values must be finite and non-negative",
        ));
    }
    Ok(())
}

pub fn initialize_state(inventory_level: usize) -> PyResult<JointPricingInventoryState> {
    Ok(JointPricingInventoryState {
        period: 0,
        inventory_level,
    })
}

pub fn build_raw_state(state: &JointPricingInventoryState) -> Vec<f32> {
    vec![state.inventory_level as f32, state.period as f32]
}

pub fn clip_action(
    order_quantity: usize,
    price_index: usize,
    max_order_quantity: usize,
    num_prices: usize,
) -> PyResult<(usize, usize)> {
    if num_prices == 0 {
        return Err(PyValueError::new_err("num_prices must be at least 1"));
    }
    Ok((
        order_quantity.min(max_order_quantity),
        price_index.min(num_prices - 1),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn step_state(
    state: &JointPricingInventoryState,
    order_quantity: usize,
    price_index: usize,
    realized_demand: usize,
    price_levels: &[f64],
    procurement_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
) -> PyResult<JointPricingInventoryStepOutcome> {
    validate_price_ladder(price_levels, &vec![0.0; price_levels.len()])?;
    validate_costs(
        procurement_cost_per_unit,
        holding_cost_per_unit,
        stockout_cost_per_unit,
        0.0,
    )?;
    if price_index >= price_levels.len() {
        return Err(PyValueError::new_err(format!(
            "price_index {price_index} out of range for {} price levels",
            price_levels.len()
        )));
    }

    let inventory_after_order = state.inventory_level + order_quantity;
    let sales = inventory_after_order.min(realized_demand);
    let lost_sales = realized_demand - sales;
    let next_inventory_level = inventory_after_order - sales;
    let selling_price = price_levels[price_index];

    let revenue = selling_price * sales as f64;
    let procurement_cost = procurement_cost_per_unit * order_quantity as f64;
    let holding_cost = holding_cost_per_unit * next_inventory_level as f64;
    let stockout_cost = stockout_cost_per_unit * lost_sales as f64;
    let period_cost = procurement_cost + holding_cost + stockout_cost - revenue;

    Ok(JointPricingInventoryStepOutcome {
        next_state: JointPricingInventoryState {
            period: state.period + 1,
            inventory_level: next_inventory_level,
        },
        order_quantity,
        price_index,
        selling_price,
        realized_demand,
        inventory_after_order,
        sales,
        lost_sales,
        revenue,
        procurement_cost,
        holding_cost,
        stockout_cost,
        period_cost,
        reward: -period_cost,
    })
}

pub fn terminal_salvage_credit(
    state: &JointPricingInventoryState,
    salvage_value_per_unit: f64,
) -> PyResult<f64> {
    validate_costs(0.0, 0.0, 0.0, salvage_value_per_unit)?;
    Ok(salvage_value_per_unit * state.inventory_level as f64)
}
