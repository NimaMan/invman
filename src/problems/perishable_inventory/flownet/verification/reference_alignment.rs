#![allow(dead_code)]

use crate::problems::perishable_inventory::flownet::instance::primary_reference_instance;

pub fn primary_reference_instance_matches_fifo_semantics() -> bool {
    let instance = primary_reference_instance();
    let has_fifo = instance
        .parameters
        .iter()
        .any(|parameter| parameter.name == "issuing_policy" && parameter.value == "fifo");
    let has_shelf_life = instance
        .parameters
        .iter()
        .any(|parameter| parameter.name == "shelf_life" && parameter.value == "2");
    let has_lead_time = instance
        .parameters
        .iter()
        .any(|parameter| parameter.name == "lead_time" && parameter.value == "1");

    has_fifo && has_shelf_life && has_lead_time
}

#[cfg(test)]
mod tests {
    use super::primary_reference_instance_matches_fifo_semantics;

    #[test]
    fn primary_reference_instance_maps_to_expected_flownet_parameters() {
        assert!(primary_reference_instance_matches_fifo_semantics());
    }
}
