#![allow(dead_code)]

use crate::problems::core::flownet::{validate_flownet, FlowNetValidationIssue};
use crate::problems::multi_echelon::production_assembly_distribution_network::flownet::formulation::canonical_production_assembly_distribution_network_flownet;

pub fn validate_production_assembly_distribution_network_flownet_structure() -> Result<(), Vec<FlowNetValidationIssue>> {
    validate_flownet(&canonical_production_assembly_distribution_network_flownet())
}

#[cfg(test)]
mod tests {
    use super::validate_production_assembly_distribution_network_flownet_structure;

    #[test]
    fn canonical_production_assembly_distribution_network_flownet_is_structurally_valid() {
        assert!(validate_production_assembly_distribution_network_flownet_structure().is_ok());
    }
}
