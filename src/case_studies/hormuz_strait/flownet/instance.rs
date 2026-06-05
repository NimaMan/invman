#![allow(dead_code)]

use crate::case_studies::hormuz_strait::flownet::formulation::HORMUZ_STRAIT_FLOWNET_NAME;
use crate::case_studies::hormuz_strait::references::{
    top_destination_2024, top_origin_2024, HormuzScenarioReference, HORMUZ_FULL_CLOSURE_SCENARIO,
};
use crate::problems::core::flownet::{FlowNetInstance, FlowNetParameter};

pub fn instance_from_reference(reference: &HormuzScenarioReference) -> FlowNetInstance {
    FlowNetInstance {
        name: String::from(reference.name),
        flownet_name: String::from(HORMUZ_STRAIT_FLOWNET_NAME),
        parameters: vec![
            FlowNetParameter {
                name: String::from("baseline_year"),
                value: reference.baseline_year.to_string(),
            },
            FlowNetParameter {
                name: String::from("node_count"),
                value: reference.node_count.to_string(),
            },
            FlowNetParameter {
                name: String::from("closure_fraction"),
                value: format!("{:.2}", reference.closure_fraction),
            },
            FlowNetParameter {
                name: String::from("total_oil_flow_million_bpd_2024"),
                value: format!("{:.6}", reference.total_oil_flow_million_bpd_2024),
            },
            FlowNetParameter {
                name: String::from("crude_and_condensate_flow_million_bpd_2024"),
                value: format!("{:.6}", reference.crude_and_condensate_flow_million_bpd_2024),
            },
            FlowNetParameter {
                name: String::from("petroleum_products_flow_million_bpd_2024"),
                value: format!("{:.6}", reference.petroleum_products_flow_million_bpd_2024),
            },
            FlowNetParameter {
                name: String::from("available_bypass_capacity_million_bpd"),
                value: format!("{:.3}", reference.available_bypass_capacity_million_bpd),
            },
            FlowNetParameter {
                name: String::from("asian_destination_share_of_crude_flows"),
                value: format!("{:.2}", reference.asian_destination_share_of_crude_flows),
            },
            FlowNetParameter {
                name: String::from("top_four_asian_destination_share_of_crude_flows"),
                value: format!("{:.2}", reference.top_four_asian_destination_share_of_crude_flows),
            },
            FlowNetParameter {
                name: String::from("largest_origin_2024"),
                value: format!(
                    "{}:{:.6}",
                    top_origin_2024().entity,
                    top_origin_2024().flow_million_bpd_2024
                ),
            },
            FlowNetParameter {
                name: String::from("largest_destination_2024"),
                value: format!(
                    "{}:{:.6}",
                    top_destination_2024().entity,
                    top_destination_2024().flow_million_bpd_2024
                ),
            },
        ],
        horizon_periods: None,
        notes: vec![
            String::from(
                "this source-backed instance defines the physical exposure map used by the executable month-ahead scenario engine",
            ),
            String::from(
                "raw and processed files live under src/case_studies/hormuz_strait/data/ and can be rebuilt with scripts/fetch_and_build.py",
            ),
        ],
    }
}

pub fn baseline_closure_instance() -> FlowNetInstance {
    instance_from_reference(&HORMUZ_FULL_CLOSURE_SCENARIO)
}
