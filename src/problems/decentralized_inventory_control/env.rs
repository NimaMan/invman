use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Debug, PartialEq)]
pub struct DecentralizedInventoryControlState {
    pub period: usize,
    pub on_hand_inventory: Vec<usize>,
    pub backlog: Vec<usize>,
    pub shipment_pipelines: Vec<Vec<usize>>,
    pub order_pipelines: Vec<Vec<usize>>,
    pub last_received_shipments: Vec<usize>,
    pub last_received_orders: Vec<usize>,
    pub forecast_orders: Vec<f64>,
    pub last_actions: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecentralizedInventoryControlStepOutcome {
    pub next_state: DecentralizedInventoryControlState,
    pub realized_customer_demand: usize,
    pub received_shipments: Vec<usize>,
    pub received_orders: Vec<usize>,
    pub downstream_shipments: Vec<usize>,
    pub ending_on_hand_inventory: Vec<usize>,
    pub ending_backlog: Vec<usize>,
    pub holding_cost: f64,
    pub backlog_cost: f64,
    pub period_cost: f64,
    pub reward: f64,
}

pub fn validate_state(state: &DecentralizedInventoryControlState) -> PyResult<()> {
    let num_agents = state.on_hand_inventory.len();
    if num_agents < 2 {
        return Err(PyValueError::new_err(
            "decentralized_inventory_control requires at least two agents",
        ));
    }
    if state.backlog.len() != num_agents
        || state.shipment_pipelines.len() != num_agents
        || state.order_pipelines.len() != num_agents
        || state.last_received_shipments.len() != num_agents
        || state.last_received_orders.len() != num_agents
        || state.forecast_orders.len() != num_agents
        || state.last_actions.len() != num_agents
    {
        return Err(PyValueError::new_err(
            "all state vectors must have the same length as on_hand_inventory",
        ));
    }
    if !state.order_pipelines[0].is_empty() {
        return Err(PyValueError::new_err(
            "retailer order pipeline must be empty; customer demand is exogenous",
        ));
    }
    if state
        .shipment_pipelines
        .iter()
        .any(|pipeline| pipeline.is_empty())
    {
        return Err(PyValueError::new_err(
            "all shipment pipelines must have strictly positive lead time",
        ));
    }
    if state
        .order_pipelines
        .iter()
        .enumerate()
        .skip(1)
        .any(|(_, pipeline)| pipeline.is_empty())
    {
        return Err(PyValueError::new_err(
            "all non-retailer order pipelines must have strictly positive lead time",
        ));
    }
    if state
        .forecast_orders
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "forecast_orders must be finite and non-negative",
        ));
    }
    Ok(())
}

pub fn initialize_state(
    on_hand_inventory: &[usize],
    backlog: &[usize],
    shipment_pipelines: &[Vec<usize>],
    order_pipelines: &[Vec<usize>],
    last_received_shipments: &[usize],
    last_received_orders: &[usize],
    forecast_orders: &[f64],
    last_actions: &[usize],
) -> PyResult<DecentralizedInventoryControlState> {
    let state = DecentralizedInventoryControlState {
        period: 0,
        on_hand_inventory: on_hand_inventory.to_vec(),
        backlog: backlog.to_vec(),
        shipment_pipelines: shipment_pipelines.to_vec(),
        order_pipelines: order_pipelines.to_vec(),
        last_received_shipments: last_received_shipments.to_vec(),
        last_received_orders: last_received_orders.to_vec(),
        forecast_orders: forecast_orders.to_vec(),
        last_actions: last_actions.to_vec(),
    };
    validate_state(&state)?;
    Ok(state)
}

pub fn shipment_pipeline_items(state: &DecentralizedInventoryControlState) -> PyResult<Vec<usize>> {
    validate_state(state)?;
    Ok(state
        .shipment_pipelines
        .iter()
        .map(|pipeline| pipeline.iter().sum::<usize>())
        .collect())
}

pub fn current_received_shipments(
    state: &DecentralizedInventoryControlState,
) -> PyResult<Vec<usize>> {
    validate_state(state)?;
    Ok(state
        .shipment_pipelines
        .iter()
        .map(|pipeline| pipeline[0])
        .collect())
}

pub fn current_received_orders(
    state: &DecentralizedInventoryControlState,
    realized_customer_demand: usize,
) -> PyResult<Vec<usize>> {
    validate_state(state)?;
    let num_agents = state.on_hand_inventory.len();
    Ok((0..num_agents)
        .map(|agent_idx| {
            if agent_idx == 0 {
                realized_customer_demand
            } else {
                state.order_pipelines[agent_idx][0]
            }
        })
        .collect())
}

pub fn on_order_items(state: &DecentralizedInventoryControlState) -> PyResult<Vec<usize>> {
    validate_state(state)?;
    let num_agents = state.on_hand_inventory.len();
    Ok((0..num_agents)
        .map(|agent_idx| {
            let inbound_shipments = state.shipment_pipelines[agent_idx].iter().sum::<usize>();
            let inbound_orders = if agent_idx + 1 < num_agents {
                state.order_pipelines[agent_idx + 1].iter().sum::<usize>()
            } else {
                0
            };
            inbound_shipments + inbound_orders
        })
        .collect())
}

pub fn inventory_positions(state: &DecentralizedInventoryControlState) -> PyResult<Vec<i32>> {
    let on_order = on_order_items(state)?;
    Ok(state
        .on_hand_inventory
        .iter()
        .zip(on_order.iter())
        .zip(state.backlog.iter())
        .map(|((on_hand, on_order), backlog)| *on_hand as i32 + *on_order as i32 - *backlog as i32)
        .collect())
}

pub fn build_local_policy_state(
    state: &DecentralizedInventoryControlState,
    agent_idx: usize,
    total_periods: usize,
    holding_costs: &[f64],
    backlog_costs: &[f64],
    realized_customer_demand: usize,
) -> PyResult<Vec<f32>> {
    validate_state(state)?;
    let num_agents = state.on_hand_inventory.len();
    if agent_idx >= num_agents {
        return Err(PyValueError::new_err(format!(
            "agent_idx {agent_idx} out of bounds for {num_agents} agents"
        )));
    }
    if holding_costs.len() != num_agents || backlog_costs.len() != num_agents {
        return Err(PyValueError::new_err(
            "holding_costs and backlog_costs must match the number of agents",
        ));
    }

    let received_shipments = current_received_shipments(state)?;
    let received_orders = current_received_orders(state, realized_customer_demand)?;
    let on_order = on_order_items(state)?[agent_idx] as f64;
    let on_hand = state.on_hand_inventory[agent_idx] as f64;
    let backlog = state.backlog[agent_idx] as f64;
    let net_inventory = on_hand - backlog;
    let received_shipment = received_shipments[agent_idx] as f64;
    let received_order = received_orders[agent_idx] as f64;
    let forecast = state.forecast_orders[agent_idx];
    let last_action = state.last_actions[agent_idx] as f64;
    let inventory_scale = on_hand
        .max(backlog)
        .max(on_order)
        .max(received_shipment)
        .max(received_order)
        .max(forecast)
        .max(last_action)
        .max(net_inventory.abs())
        .max(1.0) as f32;
    let cost_scale = holding_costs[agent_idx]
        .abs()
        .max(backlog_costs[agent_idx].abs())
        .max(1.0) as f32;
    let period_fraction = if total_periods == 0 {
        0.0
    } else {
        (total_periods.saturating_sub(state.period) as f32) / total_periods as f32
    };
    let stage_fraction = if num_agents == 1 {
        0.0
    } else {
        agent_idx as f32 / (num_agents - 1) as f32
    };

    Ok(vec![
        period_fraction,
        stage_fraction,
        on_hand as f32 / inventory_scale,
        backlog as f32 / inventory_scale,
        net_inventory as f32 / inventory_scale,
        on_order as f32 / inventory_scale,
        received_order as f32 / inventory_scale,
        received_shipment as f32 / inventory_scale,
        forecast as f32 / inventory_scale,
        last_action as f32 / inventory_scale,
        holding_costs[agent_idx] as f32 / cost_scale,
        backlog_costs[agent_idx] as f32 / cost_scale,
    ])
}

pub fn step_state(
    state: &DecentralizedInventoryControlState,
    actions: &[usize],
    realized_customer_demand: usize,
    demand_smoothing_factors: &[f64],
    holding_costs: &[f64],
    backlog_costs: &[f64],
) -> PyResult<DecentralizedInventoryControlStepOutcome> {
    validate_state(state)?;
    let num_agents = state.on_hand_inventory.len();
    if actions.len() != num_agents
        || demand_smoothing_factors.len() != num_agents
        || holding_costs.len() != num_agents
        || backlog_costs.len() != num_agents
    {
        return Err(PyValueError::new_err(
            "actions, demand_smoothing_factors, holding_costs, and backlog_costs must match the number of agents",
        ));
    }
    if demand_smoothing_factors
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(PyValueError::new_err(
            "demand_smoothing_factors must be finite and lie in [0, 1]",
        ));
    }

    let received_shipments = current_received_shipments(state)?;
    let mut next_shipment_pipelines = vec![Vec::new(); num_agents];
    for agent_idx in 0..num_agents {
        next_shipment_pipelines[agent_idx] = state.shipment_pipelines[agent_idx][1..].to_vec();
    }

    let received_orders = current_received_orders(state, realized_customer_demand)?;
    let mut next_order_pipelines = vec![Vec::new(); num_agents];
    for agent_idx in 1..num_agents {
        next_order_pipelines[agent_idx] = state.order_pipelines[agent_idx][1..].to_vec();
    }

    let mut next_on_hand_inventory = state.on_hand_inventory.clone();
    let mut next_backlog = state.backlog.clone();
    for agent_idx in 0..num_agents {
        next_on_hand_inventory[agent_idx] += received_shipments[agent_idx];
    }

    let mut downstream_shipments = vec![0usize; num_agents];
    for agent_idx in 0..num_agents {
        let total_orders = next_backlog[agent_idx] + received_orders[agent_idx];
        let shipped = next_on_hand_inventory[agent_idx].min(total_orders);
        downstream_shipments[agent_idx] = shipped;
        next_on_hand_inventory[agent_idx] -= shipped;
        next_backlog[agent_idx] = total_orders - shipped;
    }

    for agent_idx in 1..num_agents {
        next_shipment_pipelines[agent_idx - 1].push(downstream_shipments[agent_idx]);
    }
    next_shipment_pipelines[num_agents - 1].push(actions[num_agents - 1]);

    for agent_idx in 1..num_agents {
        next_order_pipelines[agent_idx].push(actions[agent_idx - 1]);
    }

    let next_forecast_orders = state
        .forecast_orders
        .iter()
        .zip(demand_smoothing_factors.iter())
        .zip(received_orders.iter())
        .map(|((forecast, smoothing), observed_order)| {
            forecast + smoothing * (*observed_order as f64 - forecast)
        })
        .collect::<Vec<_>>();

    let holding_cost = holding_costs
        .iter()
        .zip(next_on_hand_inventory.iter())
        .map(|(cost, inventory)| cost * *inventory as f64)
        .sum::<f64>();
    let backlog_cost = backlog_costs
        .iter()
        .zip(next_backlog.iter())
        .map(|(cost, backlog)| cost * *backlog as f64)
        .sum::<f64>();
    let period_cost = holding_cost + backlog_cost;

    Ok(DecentralizedInventoryControlStepOutcome {
        next_state: DecentralizedInventoryControlState {
            period: state.period + 1,
            on_hand_inventory: next_on_hand_inventory.clone(),
            backlog: next_backlog.clone(),
            shipment_pipelines: next_shipment_pipelines,
            order_pipelines: next_order_pipelines,
            last_received_shipments: received_shipments.clone(),
            last_received_orders: received_orders.clone(),
            forecast_orders: next_forecast_orders,
            last_actions: actions.to_vec(),
        },
        realized_customer_demand,
        received_shipments,
        received_orders,
        downstream_shipments,
        ending_on_hand_inventory: next_on_hand_inventory,
        ending_backlog: next_backlog,
        holding_cost,
        backlog_cost,
        period_cost,
        reward: -period_cost,
    })
}
