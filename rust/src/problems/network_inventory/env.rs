use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NetworkEdge {
    pub from: usize,
    pub to: usize,
    pub lead_time: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NetworkInventoryGraph {
    pub num_nodes: usize,
    pub source_nodes: Vec<bool>,
    pub edges: Vec<NetworkEdge>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetworkInventoryState {
    pub period: usize,
    pub on_hand_inventory: Vec<usize>,
    pub backlog: Vec<usize>,
    pub edge_pipelines: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetworkInventoryStepOutcome {
    pub next_state: NetworkInventoryState,
    pub realized_demands: Vec<usize>,
    pub received_shipments_by_node: Vec<usize>,
    pub received_shipments_by_edge: Vec<usize>,
    pub shipments_on_edges: Vec<usize>,
    pub ending_on_hand_inventory: Vec<usize>,
    pub ending_backlog: Vec<usize>,
    pub holding_cost: f64,
    pub backlog_cost: f64,
    pub period_cost: f64,
    pub reward: f64,
}

fn validate_graph(graph: &NetworkInventoryGraph) -> PyResult<()> {
    if graph.num_nodes == 0 {
        return Err(PyValueError::new_err(
            "network_inventory requires at least one node",
        ));
    }
    if graph.source_nodes.len() != graph.num_nodes {
        return Err(PyValueError::new_err(
            "source_nodes length must match num_nodes",
        ));
    }
    if graph.edges.is_empty() {
        return Err(PyValueError::new_err(
            "network_inventory requires at least one directed edge",
        ));
    }
    for edge in graph.edges.iter() {
        if edge.from >= graph.num_nodes || edge.to >= graph.num_nodes {
            return Err(PyValueError::new_err(format!(
                "edge ({}, {}) is out of bounds for {} nodes",
                edge.from, edge.to, graph.num_nodes
            )));
        }
        if edge.from == edge.to {
            return Err(PyValueError::new_err(
                "self-loops are not supported in network_inventory",
            ));
        }
        if edge.lead_time == 0 {
            return Err(PyValueError::new_err(
                "all edge lead times must be at least 1",
            ));
        }
    }
    Ok(())
}

pub fn incoming_edge_indices(graph: &NetworkInventoryGraph, node_idx: usize) -> Vec<usize> {
    graph
        .edges
        .iter()
        .enumerate()
        .filter_map(|(edge_idx, edge)| (edge.to == node_idx).then_some(edge_idx))
        .collect()
}

pub fn outgoing_edge_indices(graph: &NetworkInventoryGraph, node_idx: usize) -> Vec<usize> {
    graph
        .edges
        .iter()
        .enumerate()
        .filter_map(|(edge_idx, edge)| (edge.from == node_idx).then_some(edge_idx))
        .collect()
}

pub fn validate_state(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
) -> PyResult<()> {
    validate_graph(graph)?;
    if state.on_hand_inventory.len() != graph.num_nodes || state.backlog.len() != graph.num_nodes {
        return Err(PyValueError::new_err(
            "on_hand_inventory and backlog must match num_nodes",
        ));
    }
    if state.edge_pipelines.len() != graph.edges.len() {
        return Err(PyValueError::new_err(
            "edge_pipelines length must match the number of edges",
        ));
    }
    for (edge_idx, pipeline) in state.edge_pipelines.iter().enumerate() {
        if pipeline.len() != graph.edges[edge_idx].lead_time {
            return Err(PyValueError::new_err(format!(
                "edge_pipelines[{edge_idx}] length {} does not match lead time {}",
                pipeline.len(),
                graph.edges[edge_idx].lead_time
            )));
        }
    }
    Ok(())
}

pub fn initialize_state(
    graph: &NetworkInventoryGraph,
    on_hand_inventory: &[usize],
    backlog: &[usize],
    edge_pipelines: &[Vec<usize>],
) -> PyResult<NetworkInventoryState> {
    let state = NetworkInventoryState {
        period: 0,
        on_hand_inventory: on_hand_inventory.to_vec(),
        backlog: backlog.to_vec(),
        edge_pipelines: edge_pipelines.to_vec(),
    };
    validate_state(graph, &state)?;
    Ok(state)
}

pub fn inbound_pipeline_totals(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
) -> PyResult<Vec<usize>> {
    validate_state(graph, state)?;
    let mut totals = vec![0usize; graph.num_nodes];
    for (edge_idx, edge) in graph.edges.iter().enumerate() {
        totals[edge.to] += state.edge_pipelines[edge_idx].iter().sum::<usize>();
    }
    Ok(totals)
}

pub fn inventory_positions(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
) -> PyResult<Vec<i32>> {
    let inbound = inbound_pipeline_totals(graph, state)?;
    Ok(state
        .on_hand_inventory
        .iter()
        .zip(state.backlog.iter())
        .zip(inbound.iter())
        .map(|((on_hand, backlog), inbound)| *on_hand as i32 + *inbound as i32 - *backlog as i32)
        .collect())
}

pub fn build_policy_state(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
    demand_means: &[f64],
    total_periods: usize,
) -> PyResult<Vec<f32>> {
    validate_state(graph, state)?;
    if demand_means.len() != graph.num_nodes {
        return Err(PyValueError::new_err(
            "demand_means length must match num_nodes",
        ));
    }
    if demand_means
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "demand_means must be finite and non-negative",
        ));
    }

    let inbound = inbound_pipeline_totals(graph, state)?;
    let mut scale = 1.0f32;
    for value in state
        .on_hand_inventory
        .iter()
        .chain(state.backlog.iter())
        .chain(inbound.iter())
    {
        scale = scale.max(*value as f32);
    }
    for value in state.edge_pipelines.iter().flat_map(|pipeline| pipeline.iter()) {
        scale = scale.max(*value as f32);
    }
    for value in demand_means.iter() {
        scale = scale.max(*value as f32);
    }

    let mut features = Vec::with_capacity(4 * graph.num_nodes + graph.edges.len() + 1);
    for node_idx in 0..graph.num_nodes {
        features.push(state.on_hand_inventory[node_idx] as f32 / scale);
        features.push(state.backlog[node_idx] as f32 / scale);
        features.push(inbound[node_idx] as f32 / scale);
        features.push(demand_means[node_idx] as f32 / scale);
    }
    for edge_idx in 0..graph.edges.len() {
        features.push(state.edge_pipelines[edge_idx].iter().sum::<usize>() as f32 / scale);
    }
    let remaining_fraction = if total_periods == 0 {
        0.0
    } else {
        (total_periods.saturating_sub(state.period) as f32) / total_periods as f32
    };
    features.push(remaining_fraction);
    Ok(features)
}

fn proportional_shipments(total_available: usize, requests: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let total_requested = requests.iter().map(|(_, quantity)| *quantity).sum::<usize>();
    if total_requested <= total_available {
        return requests.to_vec();
    }
    if total_requested == 0 || total_available == 0 {
        return requests.iter().map(|(idx, _)| (*idx, 0usize)).collect();
    }

    let mut allocated = requests
        .iter()
        .map(|(edge_idx, request)| (*edge_idx, request.saturating_mul(total_available) / total_requested))
        .collect::<Vec<_>>();
    let mut used = allocated.iter().map(|(_, quantity)| *quantity).sum::<usize>();
    let mut remainders = requests
        .iter()
        .map(|(edge_idx, request)| {
            (
                *edge_idx,
                request.saturating_mul(total_available) % total_requested,
            )
        })
        .collect::<Vec<_>>();
    remainders.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    for (edge_idx, _) in remainders.into_iter() {
        if used >= total_available {
            break;
        }
        let request = requests
            .iter()
            .find(|(idx, _)| *idx == edge_idx)
            .map(|(_, quantity)| *quantity)
            .unwrap_or(0);
        let entry = allocated
            .iter_mut()
            .find(|(idx, _)| *idx == edge_idx)
            .expect("edge index must exist in allocated");
        if entry.1 < request {
            entry.1 += 1;
            used += 1;
        }
    }
    allocated
}

pub fn step_state(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
    edge_requests: &[usize],
    realized_demands: &[usize],
    holding_costs: &[f64],
    backlog_costs: &[f64],
) -> PyResult<NetworkInventoryStepOutcome> {
    validate_state(graph, state)?;
    if edge_requests.len() != graph.edges.len()
        || realized_demands.len() != graph.num_nodes
        || holding_costs.len() != graph.num_nodes
        || backlog_costs.len() != graph.num_nodes
    {
        return Err(PyValueError::new_err(
            "edge_requests, realized_demands, holding_costs, and backlog_costs must match the graph dimensions",
        ));
    }

    let mut on_hand_inventory = state.on_hand_inventory.clone();
    let mut backlog = state.backlog.clone();

    let mut received_shipments_by_edge = vec![0usize; graph.edges.len()];
    let mut received_shipments_by_node = vec![0usize; graph.num_nodes];
    let mut next_edge_pipelines = vec![Vec::new(); graph.edges.len()];
    for (edge_idx, edge) in graph.edges.iter().enumerate() {
        let arrival = state.edge_pipelines[edge_idx][0];
        received_shipments_by_edge[edge_idx] = arrival;
        received_shipments_by_node[edge.to] += arrival;
        on_hand_inventory[edge.to] += arrival;
        next_edge_pipelines[edge_idx] = state.edge_pipelines[edge_idx][1..].to_vec();
    }

    let mut shipments_on_edges = vec![0usize; graph.edges.len()];
    for node_idx in 0..graph.num_nodes {
        let outgoing = outgoing_edge_indices(graph, node_idx);
        if outgoing.is_empty() {
            continue;
        }
        if graph.source_nodes[node_idx] {
            for edge_idx in outgoing.into_iter() {
                shipments_on_edges[edge_idx] = edge_requests[edge_idx];
            }
            continue;
        }
        let outgoing_requests = outgoing
            .iter()
            .map(|edge_idx| (*edge_idx, edge_requests[*edge_idx]))
            .collect::<Vec<_>>();
        let allocated = proportional_shipments(on_hand_inventory[node_idx], &outgoing_requests);
        let total_shipped = allocated.iter().map(|(_, quantity)| *quantity).sum::<usize>();
        for (edge_idx, quantity) in allocated.into_iter() {
            shipments_on_edges[edge_idx] = quantity;
        }
        on_hand_inventory[node_idx] = on_hand_inventory[node_idx].saturating_sub(total_shipped);
    }

    for edge_idx in 0..graph.edges.len() {
        next_edge_pipelines[edge_idx].push(shipments_on_edges[edge_idx]);
    }

    for node_idx in 0..graph.num_nodes {
        let total_demand = backlog[node_idx] + realized_demands[node_idx];
        let served = on_hand_inventory[node_idx].min(total_demand);
        on_hand_inventory[node_idx] -= served;
        backlog[node_idx] = total_demand - served;
    }

    let holding_cost = holding_costs
        .iter()
        .zip(on_hand_inventory.iter())
        .map(|(cost, inventory)| cost * *inventory as f64)
        .sum::<f64>();
    let backlog_cost = backlog_costs
        .iter()
        .zip(backlog.iter())
        .map(|(cost, backlog)| cost * *backlog as f64)
        .sum::<f64>();
    let period_cost = holding_cost + backlog_cost;

    Ok(NetworkInventoryStepOutcome {
        next_state: NetworkInventoryState {
            period: state.period + 1,
            on_hand_inventory: on_hand_inventory.clone(),
            backlog: backlog.clone(),
            edge_pipelines: next_edge_pipelines,
        },
        realized_demands: realized_demands.to_vec(),
        received_shipments_by_node,
        received_shipments_by_edge,
        shipments_on_edges,
        ending_on_hand_inventory: on_hand_inventory,
        ending_backlog: backlog,
        holding_cost,
        backlog_cost,
        period_cost,
        reward: -period_cost,
    })
}
