use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetailConnectionEdge {
    pub warehouse_idx: usize,
    pub retailer_idx: usize,
    pub connection_weight: f64,
    pub lead_time: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneralBackorderFixedCostNetwork {
    pub num_suppliers: usize,
    pub num_warehouses: usize,
    pub num_retailers: usize,
    pub supplier_lead_times: Vec<usize>,
    pub retail_edges: Vec<RetailConnectionEdge>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneralBackorderFixedCostState {
    pub period: usize,
    pub warehouse_inventory: Vec<usize>,
    pub retailer_inventory: Vec<usize>,
    pub supplier_orders_due: Vec<usize>,
    pub retailer_orders_due: Vec<usize>,
    pub supplier_deliveries_due: Vec<usize>,
    pub retailer_deliveries_due: Vec<usize>,
    pub supplier_in_transit: Vec<usize>,
    pub retailer_in_transit: Vec<usize>,
    pub retailer_backorders: Vec<usize>,
    pub customer_backorders: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionStateOutcome {
    pub decision_state: GeneralBackorderFixedCostState,
    pub realized_demands: Vec<usize>,
    pub received_supplier_deliveries: Vec<usize>,
    pub received_retail_deliveries: Vec<usize>,
    pub fulfilled_current_retail_orders: Vec<usize>,
    pub fulfilled_customer_demands: Vec<usize>,
    pub period_cost: f64,
    pub holding_cost: f64,
    pub warehouse_backorder_cost: f64,
    pub customer_backorder_cost: f64,
    pub reward: f64,
}

pub fn validate_network(network: &GeneralBackorderFixedCostNetwork) -> PyResult<()> {
    if network.num_suppliers == 0 {
        return Err(PyValueError::new_err(
            "general_backorder_fixed_cost requires at least one supplier",
        ));
    }
    if network.num_warehouses == 0 {
        return Err(PyValueError::new_err(
            "general_backorder_fixed_cost requires at least one warehouse",
        ));
    }
    if network.num_retailers == 0 {
        return Err(PyValueError::new_err(
            "general_backorder_fixed_cost requires at least one retailer",
        ));
    }
    if network.supplier_lead_times.len() != network.num_warehouses {
        return Err(PyValueError::new_err(
            "supplier_lead_times length must match num_warehouses",
        ));
    }
    if network
        .supplier_lead_times
        .iter()
        .any(|lead_time| *lead_time != 1)
    {
        return Err(PyValueError::new_err(
            "the current Geevers benchmark implementation only supports supplier lead time 1",
        ));
    }
    if network.retail_edges.is_empty() {
        return Err(PyValueError::new_err(
            "general_backorder_fixed_cost requires at least one warehouse-to-retailer edge",
        ));
    }
    for (edge_idx, edge) in network.retail_edges.iter().enumerate() {
        if edge.warehouse_idx >= network.num_warehouses
            || edge.retailer_idx >= network.num_retailers
        {
            return Err(PyValueError::new_err(format!(
                "retail edge {edge_idx} is out of bounds for {} warehouses and {} retailers",
                network.num_warehouses, network.num_retailers
            )));
        }
        if edge.lead_time != 1 {
            return Err(PyValueError::new_err(
                "the current Geevers benchmark implementation only supports warehouse-to-retailer lead time 1",
            ));
        }
        if !edge.connection_weight.is_finite() || edge.connection_weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "retail edge {edge_idx} must have a positive finite connection weight"
            )));
        }
    }
    for retailer_idx in 0..network.num_retailers {
        let total_weight = incoming_retail_edge_indices(network, retailer_idx)
            .iter()
            .map(|edge_idx| network.retail_edges[*edge_idx].connection_weight)
            .sum::<f64>();
        if total_weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "retailer {retailer_idx} must have at least one upstream edge"
            )));
        }
        if (total_weight - 1.0).abs() > 1e-9 {
            return Err(PyValueError::new_err(format!(
                "retailer {retailer_idx} incoming connection weights must sum to 1.0; got {total_weight}"
            )));
        }
    }
    Ok(())
}

pub fn incoming_retail_edge_indices(
    network: &GeneralBackorderFixedCostNetwork,
    retailer_idx: usize,
) -> Vec<usize> {
    network
        .retail_edges
        .iter()
        .enumerate()
        .filter_map(|(edge_idx, edge)| (edge.retailer_idx == retailer_idx).then_some(edge_idx))
        .collect()
}

pub fn outgoing_retail_edge_indices(
    network: &GeneralBackorderFixedCostNetwork,
    warehouse_idx: usize,
) -> Vec<usize> {
    network
        .retail_edges
        .iter()
        .enumerate()
        .filter_map(|(edge_idx, edge)| (edge.warehouse_idx == warehouse_idx).then_some(edge_idx))
        .collect()
}

pub fn validate_state(
    network: &GeneralBackorderFixedCostNetwork,
    state: &GeneralBackorderFixedCostState,
) -> PyResult<()> {
    validate_network(network)?;
    if state.warehouse_inventory.len() != network.num_warehouses
        || state.retailer_inventory.len() != network.num_retailers
        || state.supplier_orders_due.len() != network.num_warehouses
        || state.retailer_orders_due.len() != network.retail_edges.len()
        || state.supplier_deliveries_due.len() != network.num_warehouses
        || state.retailer_deliveries_due.len() != network.retail_edges.len()
        || state.supplier_in_transit.len() != network.num_warehouses
        || state.retailer_in_transit.len() != network.retail_edges.len()
        || state.retailer_backorders.len() != network.retail_edges.len()
        || state.customer_backorders.len() != network.num_retailers
    {
        return Err(PyValueError::new_err(
            "state dimensions must match the network dimensions",
        ));
    }
    Ok(())
}

pub fn initialize_zero_state(
    network: &GeneralBackorderFixedCostNetwork,
) -> PyResult<GeneralBackorderFixedCostState> {
    let state = GeneralBackorderFixedCostState {
        period: 0,
        warehouse_inventory: vec![0usize; network.num_warehouses],
        retailer_inventory: vec![0usize; network.num_retailers],
        supplier_orders_due: vec![0usize; network.num_warehouses],
        retailer_orders_due: vec![0usize; network.retail_edges.len()],
        supplier_deliveries_due: vec![0usize; network.num_warehouses],
        retailer_deliveries_due: vec![0usize; network.retail_edges.len()],
        supplier_in_transit: vec![0usize; network.num_warehouses],
        retailer_in_transit: vec![0usize; network.retail_edges.len()],
        retailer_backorders: vec![0usize; network.retail_edges.len()],
        customer_backorders: vec![0usize; network.num_retailers],
    };
    validate_state(network, &state)?;
    Ok(state)
}

pub fn build_raw_state(
    network: &GeneralBackorderFixedCostNetwork,
    state: &GeneralBackorderFixedCostState,
) -> PyResult<Vec<f32>> {
    validate_state(network, state)?;
    let mut raw = Vec::with_capacity(
        network.num_warehouses * 4 + network.num_retailers * 3 + network.retail_edges.len() * 4,
    );
    raw.extend(state.warehouse_inventory.iter().map(|value| *value as f32));
    raw.extend(state.retailer_inventory.iter().map(|value| *value as f32));
    raw.extend(state.supplier_orders_due.iter().map(|value| *value as f32));
    raw.extend(state.retailer_orders_due.iter().map(|value| *value as f32));
    raw.extend(
        state
            .supplier_deliveries_due
            .iter()
            .map(|value| *value as f32),
    );
    raw.extend(
        state
            .retailer_deliveries_due
            .iter()
            .map(|value| *value as f32),
    );
    raw.extend(state.supplier_in_transit.iter().map(|value| *value as f32));
    raw.extend(state.retailer_in_transit.iter().map(|value| *value as f32));
    raw.extend(state.retailer_backorders.iter().map(|value| *value as f32));
    raw.extend(state.customer_backorders.iter().map(|value| *value as f32));
    Ok(raw)
}

pub fn warehouse_inventory_positions(
    network: &GeneralBackorderFixedCostNetwork,
    state: &GeneralBackorderFixedCostState,
) -> PyResult<Vec<i32>> {
    validate_state(network, state)?;
    Ok((0..network.num_warehouses)
        .map(|warehouse_idx| {
            let outstanding_backorders = outgoing_retail_edge_indices(network, warehouse_idx)
                .iter()
                .map(|edge_idx| state.retailer_backorders[*edge_idx] as i32)
                .sum::<i32>();
            state.warehouse_inventory[warehouse_idx] as i32
                + state.supplier_in_transit[warehouse_idx] as i32
                - outstanding_backorders
        })
        .collect())
}

pub fn retailer_total_inventory_positions(
    network: &GeneralBackorderFixedCostNetwork,
    state: &GeneralBackorderFixedCostState,
) -> PyResult<Vec<i32>> {
    validate_state(network, state)?;
    Ok((0..network.num_retailers)
        .map(|retailer_idx| {
            let inbound = incoming_retail_edge_indices(network, retailer_idx)
                .iter()
                .map(|edge_idx| state.retailer_in_transit[*edge_idx] as i32)
                .sum::<i32>();
            state.retailer_inventory[retailer_idx] as i32 + inbound
                - state.customer_backorders[retailer_idx] as i32
        })
        .collect())
}

pub fn retailer_selected_edge_inventory_position(
    network: &GeneralBackorderFixedCostNetwork,
    state: &GeneralBackorderFixedCostState,
    retail_edge_idx: usize,
) -> PyResult<i32> {
    validate_state(network, state)?;
    if retail_edge_idx >= network.retail_edges.len() {
        return Err(PyValueError::new_err(format!(
            "retail_edge_idx {retail_edge_idx} is out of bounds"
        )));
    }
    let edge = network.retail_edges[retail_edge_idx];
    Ok(state.retailer_inventory[edge.retailer_idx] as i32
        + state.retailer_in_transit[retail_edge_idx] as i32
        - state.customer_backorders[edge.retailer_idx] as i32)
}

fn ship_to_warehouse(
    state: &mut GeneralBackorderFixedCostState,
    warehouse_idx: usize,
    quantity: usize,
) {
    if quantity == 0 {
        return;
    }
    state.supplier_deliveries_due[warehouse_idx] += quantity;
    state.supplier_in_transit[warehouse_idx] += quantity;
}

fn ship_to_retailer(
    state: &mut GeneralBackorderFixedCostState,
    network: &GeneralBackorderFixedCostNetwork,
    retail_edge_idx: usize,
    quantity: usize,
) {
    if quantity == 0 {
        return;
    }
    let edge = network.retail_edges[retail_edge_idx];
    state.warehouse_inventory[edge.warehouse_idx] -= quantity;
    state.retailer_deliveries_due[retail_edge_idx] += quantity;
    state.retailer_in_transit[retail_edge_idx] += quantity;
}

fn fulfill_current_retail_orders_for_warehouse(
    state: &mut GeneralBackorderFixedCostState,
    network: &GeneralBackorderFixedCostNetwork,
    warehouse_idx: usize,
    fulfilled_current_retail_orders: &mut [usize],
) {
    let mut open_edges = outgoing_retail_edge_indices(network, warehouse_idx)
        .into_iter()
        .filter(|edge_idx| state.retailer_orders_due[*edge_idx] > 0)
        .collect::<Vec<_>>();
    if open_edges.is_empty() {
        return;
    }
    let total_order = open_edges
        .iter()
        .map(|edge_idx| state.retailer_orders_due[*edge_idx])
        .sum::<usize>();
    if state.warehouse_inventory[warehouse_idx] < total_order {
        open_edges.sort_by(|lhs, rhs| {
            let lhs_retailer = network.retail_edges[*lhs].retailer_idx;
            let rhs_retailer = network.retail_edges[*rhs].retailer_idx;
            let lhs_priority = state.retailer_inventory[lhs_retailer] as i64
                - state.customer_backorders[lhs_retailer] as i64;
            let rhs_priority = state.retailer_inventory[rhs_retailer] as i64
                - state.customer_backorders[rhs_retailer] as i64;
            lhs_priority.cmp(&rhs_priority).then(lhs.cmp(rhs))
        });
    }
    for edge_idx in open_edges {
        let requested = state.retailer_orders_due[edge_idx];
        let fulfilled = requested.min(state.warehouse_inventory[warehouse_idx]);
        if fulfilled > 0 {
            ship_to_retailer(state, network, edge_idx, fulfilled);
            fulfilled_current_retail_orders[edge_idx] += fulfilled;
        }
        let remaining = requested - fulfilled;
        if remaining > 0 {
            state.retailer_backorders[edge_idx] += remaining;
        }
        state.retailer_orders_due[edge_idx] = 0;
    }
}

fn fulfill_existing_backorders(
    state: &mut GeneralBackorderFixedCostState,
    network: &GeneralBackorderFixedCostNetwork,
) {
    for retail_edge_idx in 0..network.retail_edges.len() {
        let edge = network.retail_edges[retail_edge_idx];
        if state.warehouse_inventory[edge.warehouse_idx] == 0
            || state.retailer_backorders[retail_edge_idx] == 0
        {
            continue;
        }
        let fulfilled = state.warehouse_inventory[edge.warehouse_idx]
            .min(state.retailer_backorders[retail_edge_idx]);
        ship_to_retailer(state, network, retail_edge_idx, fulfilled);
        state.retailer_backorders[retail_edge_idx] -= fulfilled;
    }
    for retailer_idx in 0..network.num_retailers {
        if state.retailer_inventory[retailer_idx] == 0
            || state.customer_backorders[retailer_idx] == 0
        {
            continue;
        }
        let fulfilled =
            state.retailer_inventory[retailer_idx].min(state.customer_backorders[retailer_idx]);
        state.retailer_inventory[retailer_idx] -= fulfilled;
        state.customer_backorders[retailer_idx] -= fulfilled;
    }
}

pub fn advance_to_decision_state(
    network: &GeneralBackorderFixedCostNetwork,
    state: &GeneralBackorderFixedCostState,
    realized_demands: &[usize],
    warehouse_holding_costs: &[f64],
    retailer_holding_costs: &[f64],
    warehouse_backorder_costs: &[f64],
    retailer_backorder_costs: &[f64],
) -> PyResult<DecisionStateOutcome> {
    validate_state(network, state)?;
    if realized_demands.len() != network.num_retailers {
        return Err(PyValueError::new_err(
            "realized_demands length must match num_retailers",
        ));
    }
    if warehouse_holding_costs.len() != network.num_warehouses
        || retailer_holding_costs.len() != network.num_retailers
        || warehouse_backorder_costs.len() != network.num_warehouses
        || retailer_backorder_costs.len() != network.num_retailers
    {
        return Err(PyValueError::new_err(
            "cost vector lengths must match the network dimensions",
        ));
    }
    let mut decision_state = state.clone();
    let received_supplier_deliveries = decision_state.supplier_deliveries_due.clone();
    let received_retail_deliveries = decision_state.retailer_deliveries_due.clone();
    let mut fulfilled_current_retail_orders = vec![0usize; network.retail_edges.len()];
    let mut fulfilled_customer_demands = vec![0usize; network.num_retailers];

    for warehouse_idx in 0..network.num_warehouses {
        let received = decision_state.supplier_deliveries_due[warehouse_idx];
        decision_state.warehouse_inventory[warehouse_idx] += received;
        decision_state.supplier_in_transit[warehouse_idx] -= received;
        decision_state.supplier_deliveries_due[warehouse_idx] = 0;
    }
    for retail_edge_idx in 0..network.retail_edges.len() {
        let received = decision_state.retailer_deliveries_due[retail_edge_idx];
        let retailer_idx = network.retail_edges[retail_edge_idx].retailer_idx;
        decision_state.retailer_inventory[retailer_idx] += received;
        decision_state.retailer_in_transit[retail_edge_idx] -= received;
        decision_state.retailer_deliveries_due[retail_edge_idx] = 0;
    }

    for warehouse_idx in 0..network.num_warehouses {
        let due = decision_state.supplier_orders_due[warehouse_idx];
        if due > 0 {
            ship_to_warehouse(&mut decision_state, warehouse_idx, due);
            decision_state.supplier_orders_due[warehouse_idx] = 0;
        }
    }

    for warehouse_idx in 0..network.num_warehouses {
        fulfill_current_retail_orders_for_warehouse(
            &mut decision_state,
            network,
            warehouse_idx,
            &mut fulfilled_current_retail_orders,
        );
    }

    for retailer_idx in 0..network.num_retailers {
        let demand = realized_demands[retailer_idx];
        let fulfilled = demand.min(decision_state.retailer_inventory[retailer_idx]);
        decision_state.retailer_inventory[retailer_idx] -= fulfilled;
        decision_state.customer_backorders[retailer_idx] += demand - fulfilled;
        fulfilled_customer_demands[retailer_idx] = fulfilled;
    }

    fulfill_existing_backorders(&mut decision_state, network);

    let holding_cost = (0..network.num_warehouses)
        .map(|warehouse_idx| {
            warehouse_holding_costs[warehouse_idx]
                * decision_state.warehouse_inventory[warehouse_idx] as f64
        })
        .sum::<f64>()
        + (0..network.num_retailers)
            .map(|retailer_idx| {
                retailer_holding_costs[retailer_idx]
                    * decision_state.retailer_inventory[retailer_idx] as f64
            })
            .sum::<f64>();
    let warehouse_backorder_cost = (0..network.num_warehouses)
        .map(|warehouse_idx| {
            let total_backorders = outgoing_retail_edge_indices(network, warehouse_idx)
                .iter()
                .map(|edge_idx| decision_state.retailer_backorders[*edge_idx] as f64)
                .sum::<f64>();
            warehouse_backorder_costs[warehouse_idx] * total_backorders
        })
        .sum::<f64>();
    let customer_backorder_cost = (0..network.num_retailers)
        .map(|retailer_idx| {
            retailer_backorder_costs[retailer_idx]
                * decision_state.customer_backorders[retailer_idx] as f64
        })
        .sum::<f64>();
    let period_cost = holding_cost + warehouse_backorder_cost + customer_backorder_cost;

    Ok(DecisionStateOutcome {
        decision_state,
        realized_demands: realized_demands.to_vec(),
        received_supplier_deliveries,
        received_retail_deliveries,
        fulfilled_current_retail_orders,
        fulfilled_customer_demands,
        period_cost,
        holding_cost,
        warehouse_backorder_cost,
        customer_backorder_cost,
        reward: -period_cost,
    })
}

pub fn apply_next_orders(
    network: &GeneralBackorderFixedCostNetwork,
    decision_state: &GeneralBackorderFixedCostState,
    warehouse_orders_next: &[usize],
    retailer_orders_next: &[usize],
) -> PyResult<GeneralBackorderFixedCostState> {
    validate_state(network, decision_state)?;
    if warehouse_orders_next.len() != network.num_warehouses {
        return Err(PyValueError::new_err(
            "warehouse_orders_next length must match num_warehouses",
        ));
    }
    if retailer_orders_next.len() != network.retail_edges.len() {
        return Err(PyValueError::new_err(
            "retailer_orders_next length must match the number of retail edges",
        ));
    }
    let mut next_state = decision_state.clone();
    next_state.period += 1;
    next_state.supplier_orders_due = warehouse_orders_next.to_vec();
    next_state.retailer_orders_due = retailer_orders_next.to_vec();
    validate_state(network, &next_state)?;
    Ok(next_state)
}
