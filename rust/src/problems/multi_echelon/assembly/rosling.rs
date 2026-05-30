#![allow(dead_code)]

//! Rosling (1989) reduction: an assembly system is equivalent to a serial system.
//!
//! Rosling, K. (1989). "Optimal Inventory Policies for Assembly Systems Under Random
//! Demands." Operations Research 37(4):565-579. For the equal-component-lead-time case,
//! a balanced echelon base-stock policy stocks every component identically, so the
//! components collapse into a single "kit" stage and the assembly system is EXACTLY a
//! 2-stage serial system: kit (upstream) -> finished (downstream).
//!
//! This module maps an `AssemblyConfig` to the equivalent serial instance (installation
//! holding costs and lead times in upstream -> downstream order), so the literature-verified
//! `multi_echelon::serial::exact` solver gives the assembly system's exact optimal echelon
//! base-stock levels and optimal cost.

use crate::problems::multi_echelon::assembly::env::AssemblyConfig;

#[derive(Clone, Debug, PartialEq)]
pub struct SerialEquivalent {
    /// Installation (local) holding costs, upstream -> downstream: [kit, finished].
    pub local_holding_upstream_to_downstream: Vec<f64>,
    /// Lead times, upstream -> downstream: [component_lead_time, finished_lead_time].
    pub lead_times_upstream_to_downstream: Vec<usize>,
    /// Customer backorder penalty.
    pub penalty: f64,
}

/// Reduce an equal-lead-time assembly system to its equivalent 2-stage serial system.
/// Panics if the finished installation holding cost is below the kit holding cost (which
/// would not be a valid echelon system: assembly must add holding value).
pub fn reduce_equal_lead_time(config: &AssemblyConfig) -> SerialEquivalent {
    let kit_holding = config.kit_holding_cost();
    assert!(
        config.finished_holding_cost + 1e-12 >= kit_holding,
        "finished holding {} must be >= kit holding {} for a valid serial equivalent",
        config.finished_holding_cost,
        kit_holding
    );
    SerialEquivalent {
        local_holding_upstream_to_downstream: vec![kit_holding, config.finished_holding_cost],
        lead_times_upstream_to_downstream: vec![
            config.component_lead_time,
            config.finished_lead_time,
        ],
        penalty: config.penalty,
    }
}
