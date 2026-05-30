#![allow(dead_code)]

use crate::problems::core::flownet::{FlowNetInstance, FlowNetParameter};
use crate::problems::multi_echelon::general_network::demand::{DemandDistributionKind, DemandModel};
use crate::problems::multi_echelon::general_network::env::{NetworkEdge, NetworkNodeMode};
use crate::problems::multi_echelon::general_network::flownet::formulation::GENERAL_NETWORK_FLOWNET_NAME;
use crate::problems::multi_echelon::general_network::literature::{
    ExactVerificationReference, NetworkInventoryReferenceInstance, PRIMARY_REFERENCE_INSTANCE,
    VERIFICATION_PROBLEM_INSTANCE,
};

fn demand_model_description(model: &DemandModel) -> String {
    match model.kind {
        DemandDistributionKind::Deterministic => {
            format!("deterministic({:.3})", model.param1)
        }
        DemandDistributionKind::Poisson => format!("poisson({:.3})", model.param1),
        DemandDistributionKind::Normal => {
            format!("normal({:.3}, {:.3})", model.param1, model.param2)
        }
    }
}

fn edge_description(edge: &NetworkEdge) -> String {
    format!("{}->{}@{}", edge.from, edge.to, edge.lead_time)
}

fn node_mode_description(mode: &NetworkNodeMode) -> &'static str {
    match mode {
        NetworkNodeMode::Single => "single",
        NetworkNodeMode::AssemblyAnd => "assembly_and",
        NetworkNodeMode::AssemblyOr => "assembly_or",
    }
}

fn nested_supply_pipelines(rows: &[&[usize]]) -> String {
    let formatted = rows
        .iter()
        .map(|row| format!("{row:?}"))
        .collect::<Vec<_>>();
    format!("[{}]", formatted.join(", "))
}

pub fn instance_from_reference(reference: &NetworkInventoryReferenceInstance) -> FlowNetInstance {
    FlowNetInstance {
        name: String::from(reference.name),
        flownet_name: String::from(GENERAL_NETWORK_FLOWNET_NAME),
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
                name: String::from("node_modes"),
                value: format!(
                    "[{}]",
                    reference
                        .node_modes
                        .iter()
                        .map(node_mode_description)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            },
            FlowNetParameter {
                name: String::from("external_supplier_lead_times"),
                value: format!("{:?}", reference.external_supplier_lead_times),
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
                name: String::from("pairwise_oul_levels"),
                value: format!("{:?}", reference.pairwise_oul_levels),
            },
            FlowNetParameter {
                name: String::from("initial_finished_inventory"),
                value: format!("{:?}", reference.initial_finished_inventory),
            },
            FlowNetParameter {
                name: String::from("initial_raw_inventory_by_relation"),
                value: format!("{:?}", reference.initial_raw_inventory_by_relation),
            },
            FlowNetParameter {
                name: String::from("initial_internal_backlog_by_edge"),
                value: format!("{:?}", reference.initial_internal_backlog_by_edge),
            },
            FlowNetParameter {
                name: String::from("initial_external_backlog"),
                value: format!("{:?}", reference.initial_external_backlog),
            },
            FlowNetParameter {
                name: String::from("initial_supply_pipelines"),
                value: nested_supply_pipelines(reference.initial_supply_pipelines),
            },
        ],
        horizon_periods: Some(reference.periods),
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
        name: String::from("general_network_exact_verification_reference"),
        flownet_name: String::from(GENERAL_NETWORK_FLOWNET_NAME),
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
                name: String::from("node_modes"),
                value: format!(
                    "[{}]",
                    reference
                        .node_modes
                        .iter()
                        .map(node_mode_description)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            },
            FlowNetParameter {
                name: String::from("external_supplier_lead_times"),
                value: format!("{:?}", reference.external_supplier_lead_times),
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
                name: String::from("max_supply_requests"),
                value: format!("{:?}", reference.max_supply_requests),
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
