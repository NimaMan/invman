use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProcurementRemovalState {
    pub period: usize,
    pub inventory_level: usize,
    pub returnable_inventory: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProcurementRemovalStepOutcome {
    pub next_state: ProcurementRemovalState,
    pub purchase_quantity: usize,
    pub removal_quantity: usize,
    pub returned_units: usize,
    pub liquidated_units: usize,
    pub realized_demand: usize,
    pub sales: usize,
    pub shortage: usize,
    pub purchase_cost: f64,
    pub removal_credit: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub period_cost: f64,
    pub reward: f64,
}

pub fn validate_costs(
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
) -> PyResult<()> {
    let values = [
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
    ];
    if values.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(PyValueError::new_err(
            "all costs and credits must be finite and non-negative",
        ));
    }
    if purchase_cost_per_unit <= return_value_per_unit {
        return Err(PyValueError::new_err(
            "purchase_cost_per_unit must exceed return_value_per_unit",
        ));
    }
    if return_value_per_unit < liquidation_value_per_unit {
        return Err(PyValueError::new_err(
            "return_value_per_unit must be at least liquidation_value_per_unit",
        ));
    }
    Ok(())
}

pub fn validate_state(state: &ProcurementRemovalState) -> PyResult<()> {
    if state.returnable_inventory > state.inventory_level {
        return Err(PyValueError::new_err(format!(
            "returnable_inventory {} cannot exceed inventory_level {}",
            state.returnable_inventory, state.inventory_level
        )));
    }
    Ok(())
}

pub fn initialize_state(
    inventory_level: usize,
    returnable_inventory: usize,
) -> PyResult<ProcurementRemovalState> {
    let state = ProcurementRemovalState {
        period: 0,
        inventory_level,
        returnable_inventory,
    };
    validate_state(&state)?;
    Ok(state)
}

pub fn build_policy_state(
    state: &ProcurementRemovalState,
    expected_demand: f64,
    periods: usize,
    returnable_purchase_cap: usize,
) -> PyResult<Vec<f32>> {
    validate_state(state)?;
    if !expected_demand.is_finite() || expected_demand < 0.0 {
        return Err(PyValueError::new_err(
            "expected_demand must be finite and non-negative",
        ));
    }

    let non_returnable_inventory = state.inventory_level - state.returnable_inventory;
    let scale = state
        .inventory_level
        .max(returnable_purchase_cap)
        .max(expected_demand.ceil() as usize)
        .max(1) as f32;
    let remaining_fraction = if periods == 0 {
        0.0
    } else {
        (periods.saturating_sub(state.period) as f32) / periods as f32
    };
    Ok(vec![
        state.inventory_level as f32 / scale,
        state.returnable_inventory as f32 / scale,
        non_returnable_inventory as f32 / scale,
        if state.inventory_level > 0 {
            state.returnable_inventory as f32 / state.inventory_level as f32
        } else {
            0.0
        },
        expected_demand as f32 / scale,
        returnable_purchase_cap as f32 / scale,
        remaining_fraction,
    ])
}

pub fn clip_action(
    state: &ProcurementRemovalState,
    purchase_quantity: usize,
    removal_quantity: usize,
    max_purchase_quantity: usize,
    max_removal_quantity: usize,
) -> PyResult<(usize, usize)> {
    validate_state(state)?;
    let purchase_quantity = purchase_quantity.min(max_purchase_quantity);
    let removal_limit = max_removal_quantity.min(state.inventory_level + purchase_quantity);
    Ok((purchase_quantity, removal_quantity.min(removal_limit)))
}

#[allow(clippy::too_many_arguments)]
pub fn step_state(
    state: &ProcurementRemovalState,
    purchase_quantity: usize,
    removal_quantity: usize,
    realized_demand: usize,
    returnable_purchase_cap: usize,
    purchase_cost_per_unit: f64,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
    holding_cost_per_unit: f64,
    shortage_cost_per_unit: f64,
) -> PyResult<ProcurementRemovalStepOutcome> {
    validate_state(state)?;
    validate_costs(
        purchase_cost_per_unit,
        return_value_per_unit,
        liquidation_value_per_unit,
        holding_cost_per_unit,
        shortage_cost_per_unit,
    )?;

    let inventory_after_purchase = state.inventory_level + purchase_quantity;
    let returnable_after_purchase =
        state.returnable_inventory + purchase_quantity.min(returnable_purchase_cap);
    if removal_quantity > inventory_after_purchase {
        return Err(PyValueError::new_err(format!(
            "removal_quantity {} cannot exceed available inventory {} after purchase",
            removal_quantity, inventory_after_purchase
        )));
    }

    let returned_units = removal_quantity.min(returnable_after_purchase);
    let liquidated_units = removal_quantity - returned_units;
    let inventory_before_demand = inventory_after_purchase - removal_quantity;
    let returnable_before_demand = returnable_after_purchase - returned_units;

    let sales = realized_demand.min(inventory_before_demand);
    let shortage = realized_demand - sales;
    let next_inventory_level = inventory_before_demand - sales;
    let next_returnable_inventory = returnable_before_demand.min(next_inventory_level);

    let next_state = ProcurementRemovalState {
        period: state.period + 1,
        inventory_level: next_inventory_level,
        returnable_inventory: next_returnable_inventory,
    };

    let purchase_cost = purchase_cost_per_unit * purchase_quantity as f64;
    let removal_credit = return_value_per_unit * returned_units as f64
        + liquidation_value_per_unit * liquidated_units as f64;
    let holding_cost = holding_cost_per_unit * next_inventory_level as f64;
    let shortage_cost = shortage_cost_per_unit * shortage as f64;
    let period_cost = purchase_cost + holding_cost + shortage_cost - removal_credit;

    Ok(ProcurementRemovalStepOutcome {
        next_state,
        purchase_quantity,
        removal_quantity,
        returned_units,
        liquidated_units,
        realized_demand,
        sales,
        shortage,
        purchase_cost,
        removal_credit,
        holding_cost,
        shortage_cost,
        period_cost,
        reward: -period_cost,
    })
}

pub fn terminal_salvage_credit(
    state: &ProcurementRemovalState,
    return_value_per_unit: f64,
    liquidation_value_per_unit: f64,
) -> PyResult<f64> {
    validate_state(state)?;
    validate_costs(
        return_value_per_unit + 1.0,
        return_value_per_unit,
        liquidation_value_per_unit,
        0.0,
        0.0,
    )?;
    let non_returnable_inventory = state.inventory_level - state.returnable_inventory;
    Ok(return_value_per_unit * state.returnable_inventory as f64
        + liquidation_value_per_unit * non_returnable_inventory as f64)
}
