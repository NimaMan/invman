#![allow(dead_code)]

use crate::problems::core::flownet::{validate_flownet, FlowNetValidationIssue};
use crate::problems::network_inventory::flownet::formulation::canonical_network_inventory_flownet;

pub fn validate_network_inventory_flownet_structure() -> Result<(), Vec<FlowNetValidationIssue>> {
    validate_flownet(&canonical_network_inventory_flownet())
}

#[cfg(test)]
mod tests {
    use super::validate_network_inventory_flownet_structure;

    #[test]
    fn canonical_network_inventory_flownet_is_structurally_valid() {
        assert!(validate_network_inventory_flownet_structure().is_ok());
    }
}
