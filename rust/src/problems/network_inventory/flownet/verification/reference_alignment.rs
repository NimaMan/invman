#![allow(dead_code)]

use crate::problems::network_inventory::flownet::instance::{
    exact_verification_instance, primary_reference_instance,
};

pub fn primary_reference_instance_matches_diamond_network() -> bool {
    let instance = primary_reference_instance();
    let has_num_nodes = instance
        .parameters
        .iter()
        .any(|parameter| parameter.name == "num_nodes" && parameter.value == "4");
    let has_source_mask = instance.parameters.iter().any(|parameter| {
        parameter.name == "source_nodes" && parameter.value == "[true, false, false, false]"
    });
    let has_diamond_edges = instance.parameters.iter().any(|parameter| {
        parameter.name == "edges" && parameter.value == "[0->1@1, 0->2@1, 1->3@1, 2->3@1]"
    });

    has_num_nodes && has_source_mask && has_diamond_edges
}

pub fn exact_verification_instance_matches_problem_parameters() -> bool {
    let instance = exact_verification_instance();
    let has_periods = instance
        .parameters
        .iter()
        .any(|parameter| parameter.name == "periods" && parameter.value == "3");
    let has_discount_factor = instance
        .parameters
        .iter()
        .any(|parameter| parameter.name == "discount_factor" && parameter.value == "0.99");
    let has_base_stock_levels = instance
        .parameters
        .iter()
        .any(|parameter| parameter.name == "base_stock_levels" && parameter.value == "[0, 2, 2, 3]");

    has_periods && has_discount_factor && has_base_stock_levels
}

#[cfg(test)]
mod tests {
    use super::{
        exact_verification_instance_matches_problem_parameters,
        primary_reference_instance_matches_diamond_network,
    };

    #[test]
    fn primary_reference_instance_maps_to_expected_graph_parameters() {
        assert!(primary_reference_instance_matches_diamond_network());
    }

    #[test]
    fn exact_verification_instance_maps_to_problem_parameters() {
        assert!(exact_verification_instance_matches_problem_parameters());
    }
}
