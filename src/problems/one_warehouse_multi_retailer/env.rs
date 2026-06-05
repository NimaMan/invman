use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CustomerBehaviorModel {
    LostSales,
    Backorder,
    PartialBackorder,
}

pub fn parse_customer_behavior_model(model: &str) -> PyResult<CustomerBehaviorModel> {
    match model {
        "lost_sales" | "lost" => Ok(CustomerBehaviorModel::LostSales),
        "backorder" | "backlog" | "complete_backorder" => Ok(CustomerBehaviorModel::Backorder),
        "partial_backorder" | "partial" => Ok(CustomerBehaviorModel::PartialBackorder),
        _ => Err(PyValueError::new_err(format!(
            "unknown customer behavior '{model}'; expected 'lost_sales', 'backorder', or 'partial_backorder'"
        ))),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OneWarehouseMultiRetailerState {
    pub period: usize,
    pub warehouse_inventory: i32,
    pub warehouse_pipeline: Vec<usize>,
    pub retailer_inventory: Vec<i32>,
    pub retailer_pipeline: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OneWarehouseMultiRetailerStepOutcome {
    pub next_state: OneWarehouseMultiRetailerState,
    pub warehouse_arrival: usize,
    pub retailer_arrivals: Vec<usize>,
    pub retailer_shipments: Vec<usize>,
    pub emergency_shipments: Vec<usize>,
    pub unmet_demand: Vec<usize>,
    pub warehouse_ending_inventory: i32,
    pub retailer_ending_inventory: Vec<i32>,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub period_cost: f64,
    pub reward: f64,
}

pub fn validate_state(state: &OneWarehouseMultiRetailerState) -> PyResult<()> {
    if state.warehouse_pipeline.is_empty() {
        return Err(PyValueError::new_err(
            "warehouse_pipeline must have positive lead time length",
        ));
    }
    if state.retailer_inventory.is_empty() {
        return Err(PyValueError::new_err(
            "retailer_inventory must contain at least one retailer",
        ));
    }
    if state.retailer_pipeline.len() != state.retailer_inventory.len() {
        return Err(PyValueError::new_err(
            "retailer_pipeline length must match retailer_inventory length",
        ));
    }
    if state
        .retailer_pipeline
        .iter()
        .any(|pipeline| pipeline.is_empty())
    {
        return Err(PyValueError::new_err(
            "each retailer pipeline must have positive lead time length",
        ));
    }
    Ok(())
}

pub fn initialize_state(
    warehouse_inventory: i32,
    warehouse_pipeline: &[usize],
    retailer_inventory: &[i32],
    retailer_pipeline: &[Vec<usize>],
) -> PyResult<OneWarehouseMultiRetailerState> {
    let state = OneWarehouseMultiRetailerState {
        period: 0,
        warehouse_inventory,
        warehouse_pipeline: warehouse_pipeline.to_vec(),
        retailer_inventory: retailer_inventory.to_vec(),
        retailer_pipeline: retailer_pipeline.to_vec(),
    };
    validate_state(&state)?;
    Ok(state)
}

pub fn retailer_inventory_positions(state: &OneWarehouseMultiRetailerState) -> PyResult<Vec<i32>> {
    validate_state(state)?;
    Ok(state
        .retailer_inventory
        .iter()
        .zip(state.retailer_pipeline.iter())
        .map(|(inventory, pipeline)| *inventory + pipeline.iter().sum::<usize>() as i32)
        .collect())
}

pub fn warehouse_echelon_inventory_position(
    state: &OneWarehouseMultiRetailerState,
) -> PyResult<i32> {
    validate_state(state)?;
    let retailer_positions = retailer_inventory_positions(state)?;
    Ok(state.warehouse_inventory
        + state.warehouse_pipeline.iter().sum::<usize>() as i32
        + retailer_positions.iter().sum::<i32>())
}

pub fn build_raw_state(state: &OneWarehouseMultiRetailerState) -> PyResult<Vec<f32>> {
    validate_state(state)?;
    let mut raw_state = Vec::with_capacity(
        1 + state.warehouse_pipeline.len()
            + state.retailer_inventory.len()
            + state
                .retailer_pipeline
                .iter()
                .map(|pipeline| pipeline.len())
                .sum::<usize>()
            + 1,
    );
    raw_state.push(state.warehouse_inventory as f32);
    raw_state.extend(state.warehouse_pipeline.iter().map(|value| *value as f32));
    raw_state.extend(state.retailer_inventory.iter().map(|value| *value as f32));
    for pipeline in state.retailer_pipeline.iter() {
        raw_state.extend(pipeline.iter().map(|value| *value as f32));
    }
    raw_state.push(state.period as f32);
    Ok(raw_state)
}

pub fn step_state(
    state: &OneWarehouseMultiRetailerState,
    warehouse_order: usize,
    retailer_shipments: &[usize],
    realized_demands: &[usize],
    holding_cost_warehouse: f64,
    holding_cost_retailers: &[f64],
    penalty_costs_retailers: &[f64],
    customer_behavior: CustomerBehaviorModel,
    emergency_shipment_probability: f64,
    emergency_draws: Option<&[bool]>,
) -> PyResult<OneWarehouseMultiRetailerStepOutcome> {
    validate_state(state)?;
    let num_retailers = state.retailer_inventory.len();
    if retailer_shipments.len() != num_retailers
        || realized_demands.len() != num_retailers
        || holding_cost_retailers.len() != num_retailers
        || penalty_costs_retailers.len() != num_retailers
    {
        return Err(PyValueError::new_err(
            "all retailer-wise vectors must match the number of retailers",
        ));
    }
    if !holding_cost_warehouse.is_finite() || holding_cost_warehouse < 0.0 {
        return Err(PyValueError::new_err(
            "holding_cost_warehouse must be finite and non-negative",
        ));
    }
    if holding_cost_retailers
        .iter()
        .chain(penalty_costs_retailers.iter())
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "retailer costs must be finite and non-negative",
        ));
    }
    if !emergency_shipment_probability.is_finite()
        || !(0.0..=1.0).contains(&emergency_shipment_probability)
    {
        return Err(PyValueError::new_err(
            "emergency_shipment_probability must lie in [0, 1]",
        ));
    }
    if customer_behavior == CustomerBehaviorModel::PartialBackorder
        && emergency_draws.map(|draws| draws.len()) != Some(num_retailers)
    {
        return Err(PyValueError::new_err(
            "partial_backorder requires emergency_draws for each retailer",
        ));
    }

    let warehouse_arrival = state.warehouse_pipeline[0];
    let retailer_arrivals = state
        .retailer_pipeline
        .iter()
        .map(|pipeline| pipeline[0])
        .collect::<Vec<_>>();
    let available_warehouse_inventory = state.warehouse_inventory + warehouse_arrival as i32;
    let total_shipments = retailer_shipments.iter().sum::<usize>() as i32;
    if total_shipments > available_warehouse_inventory {
        return Err(PyValueError::new_err(format!(
            "retailer_shipments total {} exceeds available warehouse inventory {}",
            total_shipments, available_warehouse_inventory
        )));
    }

    let mut next_warehouse_pipeline = state.warehouse_pipeline[1..].to_vec();
    next_warehouse_pipeline.push(warehouse_order);
    let mut next_retailer_pipeline = Vec::with_capacity(num_retailers);
    for (pipeline, shipment) in state
        .retailer_pipeline
        .iter()
        .zip(retailer_shipments.iter().copied())
    {
        let mut next_pipeline = pipeline[1..].to_vec();
        next_pipeline.push(shipment);
        next_retailer_pipeline.push(next_pipeline);
    }

    let mut warehouse_ending_inventory = available_warehouse_inventory - total_shipments;
    // Warehouse holding is charged on the POST-emergency on-hand inventory: Kaynov et al. (2024)
    // Eq. 6 reduces I_0(t+1) by the emergency shipments, so holding must be assessed after the
    // retailer loop below deducts them. For lost-sales and backorder there are no emergency
    // shipments, so warehouse_ending_inventory is unchanged and this matches the prior behavior.
    let mut holding_cost = 0.0;
    let mut shortage_cost = 0.0;
    let mut emergency_shipments = vec![0usize; num_retailers];
    let mut unmet_demand = vec![0usize; num_retailers];
    let mut retailer_ending_inventory = Vec::with_capacity(num_retailers);

    for retailer_idx in 0..num_retailers {
        let retailer_available =
            state.retailer_inventory[retailer_idx] + retailer_arrivals[retailer_idx] as i32;
        let demand = realized_demands[retailer_idx] as i32;
        let (ending_inventory, unmet_units) = match customer_behavior {
            CustomerBehaviorModel::LostSales => {
                let ending_inventory = (retailer_available - demand).max(0);
                let unmet_units = (demand - retailer_available).max(0) as usize;
                (ending_inventory, unmet_units)
            }
            CustomerBehaviorModel::Backorder => {
                let ending_inventory = retailer_available - demand;
                let unmet_units = (-ending_inventory).max(0) as usize;
                (ending_inventory, unmet_units)
            }
            CustomerBehaviorModel::PartialBackorder => {
                let shortage_before_emergency = (demand - retailer_available).max(0) as usize;
                let can_trigger_emergency =
                    emergency_draws.expect("validated above for partial_backorder")[retailer_idx];
                let emergency_shipment = if shortage_before_emergency > 0 && can_trigger_emergency {
                    shortage_before_emergency.min(warehouse_ending_inventory.max(0) as usize)
                } else {
                    0
                };
                emergency_shipments[retailer_idx] = emergency_shipment;
                warehouse_ending_inventory -= emergency_shipment as i32;
                let after_emergency_inventory = retailer_available + emergency_shipment as i32;
                let ending_inventory = (after_emergency_inventory - demand).max(0);
                let unmet_units = (demand - after_emergency_inventory).max(0) as usize;
                (ending_inventory, unmet_units)
            }
        };
        unmet_demand[retailer_idx] = unmet_units;
        holding_cost += holding_cost_retailers[retailer_idx] * ending_inventory.max(0) as f64;
        shortage_cost += match customer_behavior {
            CustomerBehaviorModel::Backorder => {
                penalty_costs_retailers[retailer_idx] * (-ending_inventory).max(0) as f64
            }
            CustomerBehaviorModel::LostSales | CustomerBehaviorModel::PartialBackorder => {
                penalty_costs_retailers[retailer_idx] * unmet_units as f64
            }
        };
        retailer_ending_inventory.push(ending_inventory);
    }

    // Post-emergency warehouse holding (see the note where holding_cost is initialized).
    holding_cost += holding_cost_warehouse * warehouse_ending_inventory.max(0) as f64;

    let period_cost = holding_cost + shortage_cost;
    Ok(OneWarehouseMultiRetailerStepOutcome {
        next_state: OneWarehouseMultiRetailerState {
            period: state.period + 1,
            warehouse_inventory: warehouse_ending_inventory,
            warehouse_pipeline: next_warehouse_pipeline,
            retailer_inventory: retailer_ending_inventory.clone(),
            retailer_pipeline: next_retailer_pipeline,
        },
        warehouse_arrival,
        retailer_arrivals,
        retailer_shipments: retailer_shipments.to_vec(),
        emergency_shipments,
        unmet_demand,
        warehouse_ending_inventory,
        retailer_ending_inventory,
        holding_cost,
        shortage_cost,
        period_cost,
        reward: -period_cost,
    })
}
