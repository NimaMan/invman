use crate::problems::core::flownet::{FlowNetInstance, FlowNetParameter};
use crate::problems::network_inventory::demand::{DemandDistributionKind, DemandModel};
use crate::problems::network_inventory::env::NetworkEdge;
use crate::problems::network_inventory::flownet::formulation::NETWORK_INVENTORY_FLOWNET_NAME;
use crate::problems::network_inventory::references::{
    ExactVerificationReference, NetworkInventoryReferenceInstance, PRIMARY_REFERENCE_INSTANCE,
    VERIFICATION_PROBLEM_INSTANCE,
};

fn demand_model_description(model: &DemandModel) -> String {
    match model.kind {
        DemandDistributionKind::Deterministic => {
            format!("deterministic({:.3})", model.param1)
        }
        DemandDistributionKind::Poisson => format!("poisson({:.3})", model.param1),
    }
}

fn edge_description(edge: &NetworkEdge) -> String {
    format!("{}->{}@{}", edge.from, edge.to, edge.lead_time)
}

fn nested_edge_pipelines(rows: &[&[usize]]) -> String {
    let formatted = rows
        .iter()
        .map(|row| format!("{row:?}"))
        .collect::<Vec<_>>();
    format!("[{}]", formatted.join(", "))
}

pub fn instance_from_reference(reference: &NetworkInventoryReferenceInstance) -> FlowNetInstance {
    FlowNetInstance {
        name: String::from(reference.name),
        flownet_name: String::from(NETWORK_INVENTORY_FLOWNET_NAME),
        parameters: vec![
            FlowNetParameter {
                name: String::from("num_nodes"),
                value: reference.num_nodes.to_string(),
            },
            FlowNetParameter {
                name: String::from("source_nodes"),
                value: format!("{:?}", reference.source_nodes),
            },
            FlowNetParameter {
                name: String::from("edges"),
                value: format!(
                    "[{}]",
                    reference
                        .edges
                        .iter()
                        .map(edge_description)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            },
            FlowNetParameter {
                name: String::from("demand_models"),
                value: format!(
                    "[{}]",
                    reference
                        .demand_models
                        .iter()
                        .map(demand_model_description)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            },
            FlowNetParameter {
                name: String::from("holding_costs"),
                value: format!("{:?}", reference.holding_costs),
            },
            FlowNetParameter {
                name: String::from("backlog_costs"),
                value: format!("{:?}", reference.backlog_costs),
            },
            FlowNetParameter {
                name: String::from("base_stock_levels"),
                value: format!("{:?}", reference.base_stock_levels),
            },
            FlowNetParameter {
                name: String::from("initial_on_hand_inventory"),
                value: format!("{:?}", reference.initial_on_hand_inventory),
            },
            FlowNetParameter {
                name: String::from("initial_backlog"),
                value: format!("{:?}", reference.initial_backlog),
            },
            FlowNetParameter {
                name: String::from("initial_edge_pipelines"),
                value: nested_edge_pipelines(reference.initial_edge_pipelines),
            },
        ],
        horizon_periods: None,
        notes: vec![String::from(reference.notes)],
    }
}

pub fn primary_reference_instance() -> FlowNetInstance {
    instance_from_reference(&PRIMARY_REFERENCE_INSTANCE)
}

pub fn verification_instance_from_reference(
    reference: &ExactVerificationReference,
) -> FlowNetInstance {
    FlowNetInstance {
        name: String::from("network_inventory_exact_verification_reference"),
        flownet_name: String::from(NETWORK_INVENTORY_FLOWNET_NAME),
        parameters: vec![
            FlowNetParameter {
                name: String::from("num_nodes"),
                value: reference.num_nodes.to_string(),
            },
            FlowNetParameter {
                name: String::from("source_nodes"),
                value: format!("{:?}", reference.source_nodes),
            },
            FlowNetParameter {
                name: String::from("edges"),
                value: format!(
                    "[{}]",
                    reference
                        .edges
                        .iter()
                        .map(edge_description)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            },
            FlowNetParameter {
                name: String::from("periods"),
                value: reference.periods.to_string(),
            },
            FlowNetParameter {
                name: String::from("discount_factor"),
                value: format!("{:.2}", reference.discount_factor),
            },
            FlowNetParameter {
                name: String::from("max_edge_requests"),
                value: format!("{:?}", reference.max_edge_requests),
            },
            FlowNetParameter {
                name: String::from("base_stock_levels"),
                value: format!("{:?}", reference.base_stock_levels),
            },
        ],
        horizon_periods: Some(reference.periods),
        notes: vec![String::from(reference.notes)],
    }
}

pub fn exact_verification_instance() -> FlowNetInstance {
    verification_instance_from_reference(&VERIFICATION_PROBLEM_INSTANCE)
}
