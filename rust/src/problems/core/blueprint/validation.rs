use crate::problems::core::blueprint::problem_template::InventoryProblemBlueprint;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationIssue {
    pub message: String,
}

pub fn validate_blueprint(
    blueprint: &InventoryProblemBlueprint,
) -> Result<(), Vec<ValidationIssue>> {
    let mut issues = Vec::new();

    if !blueprint.physical.has_inventory_states() {
        issues.push(ValidationIssue {
            message: String::from("physical layer must define at least one inventory state"),
        });
    }
    if !blueprint.physical.has_material_movement() {
        issues.push(ValidationIssue {
            message: String::from("physical layer must define at least one movement path"),
        });
    }
    if !blueprint.stochastic.has_random_events() {
        issues.push(ValidationIssue {
            message: String::from("stochastic layer must define at least one random process"),
        });
    }
    if !blueprint.control.has_actions() {
        issues.push(ValidationIssue {
            message: String::from("control layer must define at least one action"),
        });
    }
    if !blueprint.control.has_observations() {
        issues.push(ValidationIssue {
            message: String::from("control layer must define at least one observation"),
        });
    }
    if !blueprint.objective.has_scoring_terms() {
        issues.push(ValidationIssue {
            message: String::from("objective layer must define at least one scoring term"),
        });
    }
    if !blueprint.timing.has_schedule() {
        issues.push(ValidationIssue {
            message: String::from("timing layer must define stages, events, and a schedule"),
        });
    } else if !blueprint.timing.schedule_references_known_events() {
        issues.push(ValidationIssue {
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
