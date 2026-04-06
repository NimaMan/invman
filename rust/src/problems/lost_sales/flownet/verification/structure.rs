use crate::problems::core::flownet::{validate_flownet, FlowNetValidationIssue};
use crate::problems::lost_sales::flownet::formulation::canonical_lost_sales_flownet;

pub fn validate_lost_sales_flownet_structure() -> Result<(), Vec<FlowNetValidationIssue>> {
    validate_flownet(&canonical_lost_sales_flownet())
}

#[cfg(test)]
mod tests {
    use super::validate_lost_sales_flownet_structure;

    #[test]
    fn canonical_lost_sales_flownet_is_structurally_valid() {
        assert!(validate_lost_sales_flownet_structure().is_ok());
    }
}
