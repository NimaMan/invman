use crate::problems::core::flownet::formulation::FlowNetFormulation;
use crate::problems::core::physical::StockRole;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlowNetValidationIssue {
    pub message: String,
}

pub fn validate_flownet(
    formulation: &FlowNetFormulation,
) -> Result<(), Vec<FlowNetValidationIssue>> {
    let mut issues = Vec::new();

    if !formulation.physical.has_inventory_states() {
        issues.push(FlowNetValidationIssue {
            message: String::from("physical layer must define at least one inventory state"),
        });
    }
    if !formulation.physical.has_material_movement() {
        issues.push(FlowNetValidationIssue {
            message: String::from("physical layer must define at least one movement path"),
        });
    }
    if !formulation.stochastic.has_random_events() {
        issues.push(FlowNetValidationIssue {
            message: String::from("stochastic layer must define at least one random process"),
        });
    }
    if !formulation.control.has_actions() {
        issues.push(FlowNetValidationIssue {
            message: String::from("control layer must define at least one action"),
        });
    }
    if !formulation.control.has_observations() {
        issues.push(FlowNetValidationIssue {
            message: String::from("control layer must define at least one observation"),
        });
    }
    if formulation
        .physical
        .stock_nodes
        .iter()
        .any(|node| node.role == StockRole::DemandSink)
        && !formulation.control.has_service_policies()
    {
        issues.push(FlowNetValidationIssue {
            message: String::from(
                "control layer must define service semantics when the physical layer includes a demand sink",
            ),
        });
    }
    for service in &formulation.control.service_policies {
        if service.inventory_sources.is_empty() {
            issues.push(FlowNetValidationIssue {
                message: format!(
                    "service policy '{}' must define at least one inventory source",
                    service.name
                ),
            });
        }
        match formulation.physical.stock_node(&service.demand_target) {
            Some(node) if node.role == StockRole::DemandSink => {}
            Some(_) => issues.push(FlowNetValidationIssue {
                message: format!(
                    "service policy '{}' must target a demand sink stock node",
                    service.name
                ),
            }),
            None => issues.push(FlowNetValidationIssue {
                message: format!(
                    "service policy '{}' references unknown demand target '{}'",
                    service.name, service.demand_target
                ),
            }),
        }
        for source in &service.inventory_sources {
            if !formulation.physical.has_stock_node(source) {
                issues.push(FlowNetValidationIssue {
                    message: format!(
                        "service policy '{}' references unknown inventory source '{}'",
                        service.name, source
                    ),
                });
            }
        }
    }
    if !formulation.objective.has_scoring_terms() {
        issues.push(FlowNetValidationIssue {
            message: String::from("objective layer must define at least one scoring term"),
        });
    }
    if !formulation.timing.has_schedule() {
        issues.push(FlowNetValidationIssue {
            message: String::from("timing layer must define stages, events, and a schedule"),
        });
    } else if !formulation.timing.schedule_references_known_events() {
        issues.push(FlowNetValidationIssue {
            message: String::from(
                "timing schedule must reference events present in the event catalog",
            ),
        });
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}
