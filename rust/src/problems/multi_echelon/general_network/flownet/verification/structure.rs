#![allow(dead_code)]

use crate::problems::core::flownet::{validate_flownet, FlowNetValidationIssue};
use crate::problems::multi_echelon::general_network::flownet::formulation::canonical_general_network_flownet;

pub fn validate_general_network_flownet_structure() -> Result<(), Vec<FlowNetValidationIssue>> {
    validate_flownet(&canonical_general_network_flownet())
}

#[cfg(test)]
mod tests {
    use super::validate_general_network_flownet_structure;

    #[test]
    fn canonical_general_network_flownet_is_structurally_valid() {
        assert!(validate_general_network_flownet_structure().is_ok());
    }
}
