#![allow(dead_code)]

use crate::problems::core::flownet::{validate_flownet, FlowNetValidationIssue};
use crate::problems::perishable_inventory::flownet::formulation::canonical_perishable_inventory_flownet;

pub fn validate_perishable_inventory_flownet_structure() -> Result<(), Vec<FlowNetValidationIssue>>
{
    validate_flownet(&canonical_perishable_inventory_flownet())
}

#[cfg(test)]
mod tests {
    use super::validate_perishable_inventory_flownet_structure;

    #[test]
    fn canonical_perishable_inventory_flownet_is_structurally_valid() {
        assert!(validate_perishable_inventory_flownet_structure().is_ok());
    }
}
