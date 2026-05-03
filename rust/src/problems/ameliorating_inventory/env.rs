use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::ameliorating_inventory::issuance::{
    optimize_average_age_blending, IssuancePlan,
};

#[derive(Clone, Debug, PartialEq)]
pub struct AmelioratingInventoryState {
    pub period: usize,
    pub inventory_by_age: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AmelioratingInventoryStepOutcome {
    pub next_state: AmelioratingInventoryState,
    pub purchase_quantity: usize,
    pub realized_demands: Vec<usize>,
    pub shipments_by_product_age: Vec<Vec<usize>>,
    pub shipped_by_product: Vec<usize>,
    pub lost_sales_by_product: Vec<usize>,
    pub decayed_units_by_age: Vec<usize>,
    pub revenue: f64,
    pub purchase_cost: f64,
    pub holding_cost: f64,
    pub salvage_credit: f64,
    pub period_cost: f64,
    pub reward: f64,
}

pub fn validate_state(state: &AmelioratingInventoryState, num_ages: usize) -> PyResult<()> {
    if num_ages == 0 {
        return Err(PyValueError::new_err("num_ages must be at least 1"));
    }
    if state.inventory_by_age.len() != num_ages {
        return Err(PyValueError::new_err(format!(
            "inventory_by_age length {} does not match num_ages {}",
            state.inventory_by_age.len(),
            num_ages
        )));
    }
    Ok(())
}

pub fn validate_problem_spec(
    num_ages: usize,
    target_ages: &[usize],
    product_prices: &[f64],
    age_retention: &[f64],
    decay_salvage_values: &[f64],
) -> PyResult<()> {
    if target_ages.is_empty() {
        return Err(PyValueError::new_err(
            "ameliorating_inventory requires at least one product",
        ));
    }
    if target_ages.len() != product_prices.len() {
        return Err(PyValueError::new_err(
            "target_ages and product_prices must have the same length",
        ));
    }
    if age_retention.len() != num_ages || decay_salvage_values.len() != num_ages {
        return Err(PyValueError::new_err(
            "age_retention and decay_salvage_values must match num_ages",
        ));
    }
    if target_ages.iter().any(|age| *age >= num_ages) {
        return Err(PyValueError::new_err(
            "all target ages must be valid inventory age indices",
        ));
    }
    if product_prices
        .iter()
        .any(|price| !price.is_finite() || *price < 0.0)
    {
        return Err(PyValueError::new_err(
            "product_prices must be finite and non-negative",
        ));
    }
    if age_retention
        .iter()
        .any(|retention| !retention.is_finite() || !(0.0..=1.0).contains(retention))
    {
        return Err(PyValueError::new_err(
            "age_retention values must lie in [0, 1]",
        ));
    }
    if decay_salvage_values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "decay_salvage_values must be finite and non-negative",
        ));
    }
    Ok(())
}

pub fn initialize_state(inventory_by_age: &[usize]) -> PyResult<AmelioratingInventoryState> {
    let state = AmelioratingInventoryState {
        period: 0,
        inventory_by_age: inventory_by_age.to_vec(),
    };
    validate_state(&state, inventory_by_age.len())?;
    Ok(state)
}

pub fn total_inventory(state: &AmelioratingInventoryState) -> usize {
    state.inventory_by_age.iter().sum()
}

pub fn build_policy_state(
    state: &AmelioratingInventoryState,
    expected_demands: &[f64],
    total_periods: usize,
) -> PyResult<Vec<f32>> {
    validate_state(state, state.inventory_by_age.len())?;
    if expected_demands
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "expected_demands must be finite and non-negative",
        ));
    }

    let total_inventory = total_inventory(state) as f64;
    let scale = total_inventory
        .max(state.inventory_by_age.iter().copied().max().unwrap_or(0) as f64)
        .max(expected_demands.iter().copied().fold(0.0_f64, f64::max))
        .max(1.0) as f32;

    let mut features =
        Vec::with_capacity(state.inventory_by_age.len() + expected_demands.len() + 2);
    features.extend(
        state
            .inventory_by_age
            .iter()
            .map(|inventory| *inventory as f32 / scale),
    );
    features.push(total_inventory as f32 / scale);
    features.extend(expected_demands.iter().map(|demand| *demand as f32 / scale));
    let remaining_fraction = if total_periods == 0 {
        0.0
    } else {
        (total_periods.saturating_sub(state.period) as f32) / total_periods as f32
    };
    features.push(remaining_fraction);
    Ok(features)
}

pub fn step_state(
    state: &AmelioratingInventoryState,
    purchase_quantity: usize,
    realized_demands: &[usize],
    target_ages: &[usize],
    product_prices: &[f64],
    age_retention: &[f64],
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: &[f64],
) -> PyResult<AmelioratingInventoryStepOutcome> {
    let num_ages = state.inventory_by_age.len();
    validate_state(state, num_ages)?;
    validate_problem_spec(
        num_ages,
        target_ages,
        product_prices,
        age_retention,
        decay_salvage_values,
    )?;
    if realized_demands.len() != target_ages.len() {
        return Err(PyValueError::new_err(
            "realized_demands length must match the number of products",
        ));
    }
    if !purchase_cost_per_unit.is_finite()
        || !holding_cost_per_unit.is_finite()
        || purchase_cost_per_unit < 0.0
        || holding_cost_per_unit < 0.0
    {
        return Err(PyValueError::new_err(
            "purchase_cost_per_unit and holding_cost_per_unit must be finite and non-negative",
        ));
    }

    let mut pre_demand_inventory = state.inventory_by_age.clone();
    pre_demand_inventory[0] += purchase_quantity;
    let issuance_plan: IssuancePlan = optimize_average_age_blending(
        &pre_demand_inventory,
        realized_demands,
        target_ages,
        product_prices,
    );

    let mut remaining_inventory = pre_demand_inventory.clone();
    for age in 0..num_ages {
        let shipped_from_age = issuance_plan
            .shipments_by_product_age
            .iter()
            .map(|shipments| shipments[age])
            .sum::<usize>();
        remaining_inventory[age] -= shipped_from_age;
    }

    let mut next_inventory_by_age = vec![0usize; num_ages];
    let mut decayed_units_by_age = vec![0usize; num_ages];
    for age in 0..num_ages {
        let survivors = ((remaining_inventory[age] as f64) * age_retention[age]).round() as usize;
        let survivors = survivors.min(remaining_inventory[age]);
        decayed_units_by_age[age] = remaining_inventory[age] - survivors;
        let destination_age = if age + 1 < num_ages { age + 1 } else { age };
        next_inventory_by_age[destination_age] += survivors;
    }

    let next_state = AmelioratingInventoryState {
        period: state.period + 1,
        inventory_by_age: next_inventory_by_age,
    };
    let purchase_cost = purchase_cost_per_unit * purchase_quantity as f64;
    let holding_cost = holding_cost_per_unit * total_inventory(&next_state) as f64;
    let salvage_credit = decayed_units_by_age
        .iter()
        .zip(decay_salvage_values.iter())
        .map(|(quantity, value)| *quantity as f64 * *value)
        .sum::<f64>();
    let period_cost = purchase_cost + holding_cost - issuance_plan.revenue - salvage_credit;

    Ok(AmelioratingInventoryStepOutcome {
        next_state,
        purchase_quantity,
        realized_demands: realized_demands.to_vec(),
        shipments_by_product_age: issuance_plan.shipments_by_product_age,
        shipped_by_product: issuance_plan.shipped_by_product,
        lost_sales_by_product: issuance_plan.lost_sales_by_product,
        decayed_units_by_age,
        revenue: issuance_plan.revenue,
        purchase_cost,
        holding_cost,
        salvage_credit,
        period_cost,
        reward: -period_cost,
    })
}
