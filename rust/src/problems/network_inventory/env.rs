use std::collections::VecDeque;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NetworkNodeMode {
    Single,
    AssemblyAnd,
    AssemblyOr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NetworkEdge {
    pub from: usize,
    pub to: usize,
    pub lead_time: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SupplyRelation {
    pub predecessor_node: Option<usize>,
    pub successor_node: usize,
    pub lead_time: usize,
    pub internal_edge_idx: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NetworkInventoryGraph {
    pub num_nodes: usize,
    pub source_nodes: Vec<bool>,
    pub node_modes: Vec<NetworkNodeMode>,
    pub external_supplier_lead_times: Vec<usize>,
    pub edges: Vec<NetworkEdge>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NetworkInventoryState {
    pub period: usize,
    pub finished_inventory: Vec<usize>,
    pub raw_inventory_by_relation: Vec<usize>,
    pub internal_backlog_by_edge: Vec<usize>,
    pub external_backlog: Vec<usize>,
    pub supply_pipelines: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetworkInventoryStepOutcome {
    pub next_state: NetworkInventoryState,
    pub realized_external_demands: Vec<usize>,
    pub supply_requests: Vec<usize>,
    pub received_shipments_by_relation: Vec<usize>,
    pub produced_finished_goods: Vec<usize>,
    pub shipped_on_internal_edges: Vec<usize>,
    pub shipped_to_external_customer: Vec<usize>,
    pub holding_cost: f64,
    pub backlog_cost: f64,
    pub period_cost: f64,
    pub reward: f64,
}

fn topological_order(graph: &NetworkInventoryGraph) -> PyResult<Vec<usize>> {
    let mut indegree = vec![0usize; graph.num_nodes];
    let mut outgoing = vec![Vec::new(); graph.num_nodes];
    for edge in graph.edges.iter() {
        indegree[edge.to] += 1;
        outgoing[edge.from].push(edge.to);
    }
    let mut queue = VecDeque::new();
    for node in 0..graph.num_nodes {
        if indegree[node] == 0 {
            queue.push_back(node);
        }
    }
    let mut order = Vec::with_capacity(graph.num_nodes);
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for successor in outgoing[node].iter().copied() {
            indegree[successor] -= 1;
            if indegree[successor] == 0 {
                queue.push_back(successor);
            }
        }
    }
    if order.len() != graph.num_nodes {
        return Err(PyValueError::new_err(
            "network_inventory graph must be acyclic",
        ));
    }
    Ok(order)
}

pub fn supply_relations(graph: &NetworkInventoryGraph) -> Vec<SupplyRelation> {
    let mut relations = graph
        .edges
        .iter()
        .enumerate()
        .map(|(edge_idx, edge)| SupplyRelation {
            predecessor_node: Some(edge.from),
            successor_node: edge.to,
            lead_time: edge.lead_time,
            internal_edge_idx: Some(edge_idx),
        })
        .collect::<Vec<_>>();
    for node in 0..graph.num_nodes {
        if graph.source_nodes[node] {
            relations.push(SupplyRelation {
                predecessor_node: None,
                successor_node: node,
                lead_time: graph.external_supplier_lead_times[node],
                internal_edge_idx: None,
            });
        }
    }
    relations
}

pub fn supply_relation_count(graph: &NetworkInventoryGraph) -> usize {
    graph.edges.len() + graph.source_nodes.iter().filter(|flag| **flag).count()
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

pub fn incoming_supply_relation_indices(
    graph: &NetworkInventoryGraph,
    node_idx: usize,
) -> Vec<usize> {
    supply_relations(graph)
        .iter()
        .enumerate()
        .filter_map(|(relation_idx, relation)| {
            (relation.successor_node == node_idx).then_some(relation_idx)
        })
        .collect()
}

pub fn external_supplier_relation_index(
    graph: &NetworkInventoryGraph,
    node_idx: usize,
) -> Option<usize> {
    let mut relation_idx = graph.edges.len();
    for candidate in 0..graph.num_nodes {
        if graph.source_nodes[candidate] {
            if candidate == node_idx {
                return Some(relation_idx);
            }
            relation_idx += 1;
        }
    }
    None
}

fn total_outgoing_internal_backlog(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
    node_idx: usize,
) -> usize {
    outgoing_edge_indices(graph, node_idx)
        .iter()
        .map(|edge_idx| state.internal_backlog_by_edge[*edge_idx])
        .sum()
}

pub fn total_inbound_pipeline_by_node(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
) -> PyResult<Vec<usize>> {
    validate_state(graph, state)?;
    let relations = supply_relations(graph);
    let mut totals = vec![0usize; graph.num_nodes];
    for (relation_idx, relation) in relations.iter().enumerate() {
        totals[relation.successor_node] +=
            state.supply_pipelines[relation_idx].iter().sum::<usize>();
    }
    Ok(totals)
}

pub fn total_raw_inventory_by_node(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
) -> PyResult<Vec<usize>> {
    validate_state(graph, state)?;
    let relations = supply_relations(graph);
    let mut totals = vec![0usize; graph.num_nodes];
    for (relation_idx, relation) in relations.iter().enumerate() {
        totals[relation.successor_node] += state.raw_inventory_by_relation[relation_idx];
    }
    Ok(totals)
}

pub fn aggregate_inventory_positions(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
) -> PyResult<Vec<i32>> {
    let raw = total_raw_inventory_by_node(graph, state)?;
    let inbound = total_inbound_pipeline_by_node(graph, state)?;
    Ok((0..graph.num_nodes)
        .map(|node_idx| {
            state.finished_inventory[node_idx] as i32
                + raw[node_idx] as i32
                + inbound[node_idx] as i32
                - total_outgoing_internal_backlog(graph, state, node_idx) as i32
                - state.external_backlog[node_idx] as i32
        })
        .collect())
}

pub fn validate_graph(graph: &NetworkInventoryGraph) -> PyResult<()> {
    if graph.num_nodes == 0 {
        return Err(PyValueError::new_err(
            "network_inventory requires at least one node",
        ));
    }
    if graph.source_nodes.len() != graph.num_nodes
        || graph.node_modes.len() != graph.num_nodes
        || graph.external_supplier_lead_times.len() != graph.num_nodes
    {
        return Err(PyValueError::new_err(
            "source_nodes, node_modes, and external_supplier_lead_times must match num_nodes",
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
    }
    let _ = topological_order(graph)?;
    for node_idx in 0..graph.num_nodes {
        let has_internal_predecessor = graph.edges.iter().any(|edge| edge.to == node_idx);
        if graph.source_nodes[node_idx] && has_internal_predecessor {
            return Err(PyValueError::new_err(format!(
                "node {node_idx} cannot have both an external supplier and an internal predecessor"
            )));
        }
        if !graph.source_nodes[node_idx] && graph.external_supplier_lead_times[node_idx] != 0 {
            return Err(PyValueError::new_err(format!(
                "non-source node {node_idx} must have external_supplier_lead_times[{node_idx}] = 0"
            )));
        }
    }
    Ok(())
}

pub fn validate_state(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
) -> PyResult<()> {
    validate_graph(graph)?;
    let relation_count = supply_relation_count(graph);
    if state.finished_inventory.len() != graph.num_nodes
        || state.external_backlog.len() != graph.num_nodes
        || state.raw_inventory_by_relation.len() != relation_count
        || state.supply_pipelines.len() != relation_count
        || state.internal_backlog_by_edge.len() != graph.edges.len()
    {
        return Err(PyValueError::new_err(
            "state vectors do not match the graph dimensions",
        ));
    }
    let relations = supply_relations(graph);
    for (relation_idx, relation) in relations.iter().enumerate() {
        if state.supply_pipelines[relation_idx].len() != relation.lead_time {
            return Err(PyValueError::new_err(format!(
                "supply_pipelines[{relation_idx}] length {} does not match lead time {}",
                state.supply_pipelines[relation_idx].len(),
                relation.lead_time
            )));
        }
    }
    Ok(())
}

pub fn initialize_state(
    graph: &NetworkInventoryGraph,
    finished_inventory: &[usize],
    raw_inventory_by_relation: &[usize],
    internal_backlog_by_edge: &[usize],
    external_backlog: &[usize],
    supply_pipelines: &[Vec<usize>],
) -> PyResult<NetworkInventoryState> {
    let state = NetworkInventoryState {
        period: 0,
        finished_inventory: finished_inventory.to_vec(),
        raw_inventory_by_relation: raw_inventory_by_relation.to_vec(),
        internal_backlog_by_edge: internal_backlog_by_edge.to_vec(),
        external_backlog: external_backlog.to_vec(),
        supply_pipelines: supply_pipelines.to_vec(),
    };
    validate_state(graph, &state)?;
    Ok(state)
}

pub fn build_policy_state(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
    demand_means: &[f64],
    current_external_demands: &[usize],
    total_periods: usize,
) -> PyResult<Vec<f32>> {
    validate_state(graph, state)?;
    if demand_means.len() != graph.num_nodes || current_external_demands.len() != graph.num_nodes {
        return Err(PyValueError::new_err(
            "demand_means and current_external_demands must match num_nodes",
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
    let raw_by_node = total_raw_inventory_by_node(graph, state)?;
    let inbound_pipeline = total_inbound_pipeline_by_node(graph, state)?;

    let mut scale = 1.0f32;
    for value in state
        .finished_inventory
        .iter()
        .chain(state.raw_inventory_by_relation.iter())
        .chain(state.internal_backlog_by_edge.iter())
        .chain(state.external_backlog.iter())
        .chain(inbound_pipeline.iter())
        .chain(raw_by_node.iter())
    {
        scale = scale.max(*value as f32);
    }
    for value in state
        .supply_pipelines
        .iter()
        .flat_map(|pipeline| pipeline.iter())
    {
        scale = scale.max(*value as f32);
    }
    for value in demand_means.iter() {
        scale = scale.max(*value as f32);
    }

    let relation_count = supply_relation_count(graph);
    let mut features =
        Vec::with_capacity(7 * graph.num_nodes + 2 * relation_count + graph.edges.len() + 1);
    for node_idx in 0..graph.num_nodes {
        features.push(state.finished_inventory[node_idx] as f32 / scale);
        features.push(raw_by_node[node_idx] as f32 / scale);
        features.push(total_outgoing_internal_backlog(graph, state, node_idx) as f32 / scale);
        features.push(state.external_backlog[node_idx] as f32 / scale);
        features.push(inbound_pipeline[node_idx] as f32 / scale);
        features.push(demand_means[node_idx] as f32 / scale);
        features.push(current_external_demands[node_idx] as f32 / scale);
    }
    for relation_idx in 0..relation_count {
        features.push(state.raw_inventory_by_relation[relation_idx] as f32 / scale);
        features.push(state.supply_pipelines[relation_idx].iter().sum::<usize>() as f32 / scale);
    }
    for edge_idx in 0..graph.edges.len() {
        features.push(state.internal_backlog_by_edge[edge_idx] as f32 / scale);
    }
    let remaining_fraction = if total_periods == 0 {
        0.0
    } else {
        (total_periods.saturating_sub(state.period) as f32) / total_periods as f32
    };
    features.push(remaining_fraction);
    Ok(features)
}

fn proportional_allocation(
    total_available: usize,
    total_requirements: &[usize],
    current_demands: &[usize],
) -> Vec<usize> {
    if total_requirements.is_empty() || total_available == 0 {
        return vec![0; total_requirements.len()];
    }
    let total_required = total_requirements.iter().sum::<usize>();
    if total_required <= total_available {
        return total_requirements.to_vec();
    }

    let weights = if current_demands.iter().sum::<usize>() > 0 {
        current_demands
    } else {
        total_requirements
    };
    let weight_sum = weights.iter().sum::<usize>().max(1);

    let mut allocated = weights
        .iter()
        .enumerate()
        .map(|(idx, weight)| {
            let share = weight.saturating_mul(total_available) / weight_sum;
            (idx, share.min(total_requirements[idx]))
        })
        .collect::<Vec<_>>();
    let mut used = allocated.iter().map(|(_, qty)| *qty).sum::<usize>();
    let mut remainders = weights
        .iter()
        .enumerate()
        .map(|(idx, weight)| (idx, weight.saturating_mul(total_available) % weight_sum))
        .collect::<Vec<_>>();
    remainders.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    for (idx, _) in remainders.into_iter() {
        if used >= total_available {
            break;
        }
        let entry = allocated
            .iter_mut()
            .find(|(candidate_idx, _)| *candidate_idx == idx)
            .expect("allocated entry must exist");
        if entry.1 < total_requirements[idx] {
            entry.1 += 1;
            used += 1;
        }
    }

    allocated.sort_by_key(|(idx, _)| *idx);
    allocated.into_iter().map(|(_, qty)| qty).collect()
}

pub fn step_state(
    graph: &NetworkInventoryGraph,
    state: &NetworkInventoryState,
    supply_requests: &[usize],
    realized_external_demands: &[usize],
    holding_costs: &[f64],
    backlog_costs: &[f64],
) -> PyResult<NetworkInventoryStepOutcome> {
    validate_state(graph, state)?;
    let relations = supply_relations(graph);
    if supply_requests.len() != relations.len()
        || realized_external_demands.len() != graph.num_nodes
        || holding_costs.len() != graph.num_nodes
        || backlog_costs.len() != graph.num_nodes
    {
        return Err(PyValueError::new_err(
            "supply_requests, realized_external_demands, holding_costs, and backlog_costs must match the graph dimensions",
        ));
    }

    let order = topological_order(graph)?;
    let reverse_order = order.iter().rev().copied().collect::<Vec<_>>();

    let mut next_state = state.clone();
    next_state.period += 1;

    let mut received_shipments_by_relation = vec![0usize; relations.len()];
    let mut produced_finished_goods = vec![0usize; graph.num_nodes];
    let mut shipped_on_internal_edges = vec![0usize; graph.edges.len()];
    let mut shipped_to_external_customer = vec![0usize; graph.num_nodes];

    for node_idx in reverse_order.into_iter() {
        let _ = node_idx;
        // Phase 1 is implicit for direct-request policies: successor order quantities are already
        // provided in `supply_requests`, and external demands are already realized.
    }

    for node_idx in order.into_iter() {
        for relation_idx in incoming_supply_relation_indices(graph, node_idx) {
            let relation = relations[relation_idx];
            let arrival = if relation.lead_time == 0 {
                match relation.predecessor_node {
                    Some(_) => {
                        shipped_on_internal_edges
                            [relation.internal_edge_idx.expect("edge idx exists")]
                    }
                    None => supply_requests[relation_idx],
                }
            } else {
                let pipeline = &mut next_state.supply_pipelines[relation_idx];
                let arrival = pipeline.first().copied().unwrap_or(0);
                if !pipeline.is_empty() {
                    pipeline.remove(0);
                }
                arrival
            };
            received_shipments_by_relation[relation_idx] = arrival;
            next_state.raw_inventory_by_relation[relation_idx] += arrival;
        }

        if let Some(relation_idx) = external_supplier_relation_index(graph, node_idx) {
            let relation = relations[relation_idx];
            if relation.lead_time > 0 {
                next_state.supply_pipelines[relation_idx].push(supply_requests[relation_idx]);
            }
        }

        let incoming_relations = incoming_supply_relation_indices(graph, node_idx);
        let produced = match graph.node_modes[node_idx] {
            NetworkNodeMode::AssemblyAnd => incoming_relations
                .iter()
                .map(|relation_idx| next_state.raw_inventory_by_relation[*relation_idx])
                .min()
                .unwrap_or(0),
            NetworkNodeMode::AssemblyOr | NetworkNodeMode::Single => incoming_relations
                .iter()
                .map(|relation_idx| next_state.raw_inventory_by_relation[*relation_idx])
                .sum(),
        };
        produced_finished_goods[node_idx] = produced;
        match graph.node_modes[node_idx] {
            NetworkNodeMode::AssemblyAnd => {
                for relation_idx in incoming_relations.iter().copied() {
                    next_state.raw_inventory_by_relation[relation_idx] -= produced;
                }
            }
            NetworkNodeMode::AssemblyOr | NetworkNodeMode::Single => {
                for relation_idx in incoming_relations.iter().copied() {
                    next_state.raw_inventory_by_relation[relation_idx] = 0;
                }
            }
        }
        next_state.finished_inventory[node_idx] += produced;

        let outgoing_edges = outgoing_edge_indices(graph, node_idx);
        let mut total_requirements = Vec::with_capacity(outgoing_edges.len() + 1);
        let mut current_demands = Vec::with_capacity(outgoing_edges.len() + 1);

        for edge_idx in outgoing_edges.iter().copied() {
            total_requirements
                .push(next_state.internal_backlog_by_edge[edge_idx] + supply_requests[edge_idx]);
            current_demands.push(supply_requests[edge_idx]);
        }
        total_requirements
            .push(next_state.external_backlog[node_idx] + realized_external_demands[node_idx]);
        current_demands.push(realized_external_demands[node_idx]);

        let available = next_state.finished_inventory[node_idx];
        let shipments = proportional_allocation(available, &total_requirements, &current_demands);
        let total_shipped = shipments.iter().sum::<usize>();
        next_state.finished_inventory[node_idx] =
            next_state.finished_inventory[node_idx].saturating_sub(total_shipped);

        for (offset, edge_idx) in outgoing_edges.iter().copied().enumerate() {
            shipped_on_internal_edges[edge_idx] = shipments[offset];
            next_state.internal_backlog_by_edge[edge_idx] =
                total_requirements[offset].saturating_sub(shipments[offset]);
            let relation = relations[edge_idx];
            if relation.lead_time > 0 {
                next_state.supply_pipelines[edge_idx].push(shipments[offset]);
            }
        }
        shipped_to_external_customer[node_idx] = *shipments.last().unwrap_or(&0);
        next_state.external_backlog[node_idx] = total_requirements
            .last()
            .copied()
            .unwrap_or(0)
            .saturating_sub(*shipments.last().unwrap_or(&0));
    }

    let inbound_counts = (0..graph.num_nodes)
        .map(|node_idx| incoming_supply_relation_indices(graph, node_idx).len())
        .collect::<Vec<_>>();
    let raw_by_node = total_raw_inventory_by_node(graph, &next_state)?;

    let mut holding_cost = 0.0;
    for node_idx in 0..graph.num_nodes {
        let outgoing_in_transit = outgoing_edge_indices(graph, node_idx)
            .iter()
            .map(|edge_idx| next_state.supply_pipelines[*edge_idx].iter().sum::<usize>())
            .sum::<usize>();
        holding_cost += holding_costs[node_idx]
            * (raw_by_node[node_idx] as f64
                + inbound_counts[node_idx] as f64
                    * (next_state.finished_inventory[node_idx] + outgoing_in_transit) as f64);
    }

    let mut backlog_cost = 0.0;
    for node_idx in 0..graph.num_nodes {
        let internal_backlog = outgoing_edge_indices(graph, node_idx)
            .iter()
            .map(|edge_idx| next_state.internal_backlog_by_edge[*edge_idx])
            .sum::<usize>();
        backlog_cost += backlog_costs[node_idx]
            * (internal_backlog + next_state.external_backlog[node_idx]) as f64;
    }
    let period_cost = holding_cost + backlog_cost;

    Ok(NetworkInventoryStepOutcome {
        next_state,
        realized_external_demands: realized_external_demands.to_vec(),
        supply_requests: supply_requests.to_vec(),
        received_shipments_by_relation,
        produced_finished_goods,
        shipped_on_internal_edges,
        shipped_to_external_customer,
        holding_cost,
        backlog_cost,
        period_cost,
        reward: -period_cost,
    })
}
