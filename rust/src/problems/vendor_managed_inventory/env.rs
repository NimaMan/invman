use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VendorManagedInventoryState {
    pub period: usize,
    pub dc_on_hand: usize,
    pub retailer_on_hand: usize,
    pub retailer_pipeline: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VendorManagedInventoryStepOutcome {
    pub next_state: VendorManagedInventoryState,
    pub shipment_quantity: usize,
    pub realized_demand: usize,
    pub arrivals_to_retailer: usize,
    pub sales: usize,
    pub lost_sales: usize,
    pub dc_replenishment: usize,
    pub shipment_cost: f64,
    pub dc_holding_cost: f64,
    pub retailer_holding_cost: f64,
    pub stockout_cost: f64,
    pub period_cost: f64,
    pub reward: f64,
}

pub fn validate_costs(
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    salvage_value_per_unit: f64,
) -> PyResult<()> {
    let values = [
        shipment_cost_per_unit,
        dc_holding_cost_per_unit,
        retailer_holding_cost_per_unit,
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

pub fn validate_state(state: &VendorManagedInventoryState, dc_capacity: usize) -> PyResult<()> {
    if state.dc_on_hand > dc_capacity {
        return Err(PyValueError::new_err(format!(
            "dc_on_hand {} cannot exceed dc_capacity {}",
            state.dc_on_hand, dc_capacity
        )));
    }
    Ok(())
}

pub fn initialize_state(
    dc_on_hand: usize,
    retailer_on_hand: usize,
    retailer_pipeline: usize,
    dc_capacity: usize,
) -> PyResult<VendorManagedInventoryState> {
    let state = VendorManagedInventoryState {
        period: 0,
        dc_on_hand,
        retailer_on_hand,
        retailer_pipeline,
    };
    validate_state(&state, dc_capacity)?;
    Ok(state)
}

pub fn retailer_inventory_position(state: &VendorManagedInventoryState) -> usize {
    state.retailer_on_hand + state.retailer_pipeline
}

pub fn build_policy_state(
    state: &VendorManagedInventoryState,
    expected_demand: f64,
    periods: usize,
    dc_capacity: usize,
    dc_replenishment_quantity: usize,
) -> PyResult<Vec<f32>> {
    validate_state(state, dc_capacity)?;
    if !expected_demand.is_finite() || expected_demand < 0.0 {
        return Err(PyValueError::new_err(
            "expected_demand must be finite and non-negative",
        ));
    }
    let scale = dc_capacity
        .max(dc_replenishment_quantity)
        .max(expected_demand.ceil() as usize)
        .max(1) as f32;
    let remaining_fraction = if periods == 0 {
        0.0
    } else {
        (periods.saturating_sub(state.period) as f32) / periods as f32
    };
    Ok(vec![
        state.dc_on_hand as f32 / scale,
        state.retailer_on_hand as f32 / scale,
        state.retailer_pipeline as f32 / scale,
        retailer_inventory_position(state) as f32 / scale,
        expected_demand as f32 / scale,
        dc_replenishment_quantity as f32 / scale,
        remaining_fraction,
    ])
}

pub fn clip_action(
    state: &VendorManagedInventoryState,
    shipment_quantity: usize,
    dc_capacity: usize,
    max_shipment_quantity: usize,
) -> PyResult<usize> {
    validate_state(state, dc_capacity)?;
    Ok(shipment_quantity
        .min(max_shipment_quantity)
        .min(state.dc_on_hand))
}

#[allow(clippy::too_many_arguments)]
pub fn step_state(
    state: &VendorManagedInventoryState,
    shipment_quantity: usize,
    realized_demand: usize,
    dc_replenishment_quantity: usize,
    dc_capacity: usize,
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
) -> PyResult<VendorManagedInventoryStepOutcome> {
    validate_state(state, dc_capacity)?;
    validate_costs(
        shipment_cost_per_unit,
        dc_holding_cost_per_unit,
        retailer_holding_cost_per_unit,
        stockout_cost_per_unit,
        0.0,
    )?;

    if shipment_quantity > state.dc_on_hand {
        return Err(PyValueError::new_err(format!(
            "shipment_quantity {} cannot exceed dc_on_hand {}",
            shipment_quantity, state.dc_on_hand
        )));
    }

    let arrivals_to_retailer = state.retailer_pipeline;
    let retailer_available = state.retailer_on_hand + arrivals_to_retailer;
    let sales = retailer_available.min(realized_demand);
    let lost_sales = realized_demand - sales;
    let next_retailer_on_hand = retailer_available - sales;
    let next_retailer_pipeline = shipment_quantity;

    let dc_after_shipment = state.dc_on_hand - shipment_quantity;
    let dc_replenishment = dc_replenishment_quantity.min(dc_capacity - dc_after_shipment);
    let next_dc_on_hand = dc_after_shipment + dc_replenishment;

    let next_state = VendorManagedInventoryState {
        period: state.period + 1,
        dc_on_hand: next_dc_on_hand,
        retailer_on_hand: next_retailer_on_hand,
        retailer_pipeline: next_retailer_pipeline,
    };

    let shipment_cost = shipment_cost_per_unit * shipment_quantity as f64;
    let dc_holding_cost = dc_holding_cost_per_unit * next_dc_on_hand as f64;
    let retailer_holding_cost = retailer_holding_cost_per_unit * next_retailer_on_hand as f64;
    let stockout_cost = stockout_cost_per_unit * lost_sales as f64;
    let period_cost = shipment_cost + dc_holding_cost + retailer_holding_cost + stockout_cost;

    Ok(VendorManagedInventoryStepOutcome {
        next_state,
        shipment_quantity,
        realized_demand,
        arrivals_to_retailer,
        sales,
        lost_sales,
        dc_replenishment,
        shipment_cost,
        dc_holding_cost,
        retailer_holding_cost,
        stockout_cost,
        period_cost,
        reward: -period_cost,
    })
}

pub fn terminal_salvage_credit(
    state: &VendorManagedInventoryState,
    dc_capacity: usize,
    salvage_value_per_unit: f64,
) -> PyResult<f64> {
    validate_state(state, dc_capacity)?;
    validate_costs(0.0, 0.0, 0.0, 0.0, salvage_value_per_unit)?;
    Ok(salvage_value_per_unit
        * (state.dc_on_hand + state.retailer_on_hand + state.retailer_pipeline) as f64)
}
