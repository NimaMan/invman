#![allow(dead_code)]

use crate::problems::core::flownet::{FlowNetInstance, FlowNetParameter};
use crate::problems::perishable_inventory::env::IssuingPolicy;
use crate::problems::perishable_inventory::flownet::formulation::PERISHABLE_INVENTORY_FLOWNET_NAME;
use crate::problems::perishable_inventory::references::{
    get_primary_reference_instance, PerishableReferenceInstance,
};

pub fn issuing_policy_description(policy: IssuingPolicy) -> &'static str {
    match policy {
        IssuingPolicy::Fifo => "fifo",
        IssuingPolicy::Lifo => "lifo",
    }
}

pub fn instance_from_reference(reference: &PerishableReferenceInstance) -> FlowNetInstance {
    FlowNetInstance {
        name: String::from(reference.name),
        flownet_name: String::from(PERISHABLE_INVENTORY_FLOWNET_NAME),
        parameters: vec![
            FlowNetParameter {
                name: String::from("demand_model"),
                value: format!(
                    "rounded_gamma(mean={:.3}, cov={:.3})",
                    reference.demand_mean, reference.demand_cov
                ),
            },
            FlowNetParameter {
                name: String::from("shelf_life"),
                value: reference.shelf_life.to_string(),
            },
            FlowNetParameter {
                name: String::from("lead_time"),
                value: reference.lead_time.to_string(),
            },
            FlowNetParameter {
                name: String::from("holding_cost"),
                value: reference.holding_cost.to_string(),
            },
            FlowNetParameter {
                name: String::from("shortage_cost"),
                value: reference.shortage_cost.to_string(),
            },
            FlowNetParameter {
                name: String::from("waste_cost"),
                value: reference.waste_cost.to_string(),
            },
            FlowNetParameter {
                name: String::from("procurement_cost"),
                value: reference.procurement_cost.to_string(),
            },
            FlowNetParameter {
                name: String::from("issuing_policy"),
                value: String::from(issuing_policy_description(reference.issuing_policy)),
            },
            FlowNetParameter {
                name: String::from("max_order_size"),
                value: reference.max_order_size.to_string(),
            },
        ],
        horizon_periods: Some(reference.horizon),
        notes: vec![
            String::from(
                "the current FlowNet instance captures the problem physics and literature parameters, not the exact MDP discretization details",
            ),
            String::from(
                "the primary perishable rollouts use rounded Gamma demand and observe the pipeline vector before the on-hand age buckets",
            ),
        ],
    }
}

pub fn primary_reference_instance() -> FlowNetInstance {
    instance_from_reference(&get_primary_reference_instance())
}
