#![allow(dead_code)]

//! Verification for the assembly family: the assembly `env.rs` simulation under the optimal
//! echelon base-stock policy reproduces the EXACT serial optimum that Rosling (1989) says the
//! assembly system equals. The serial optimum comes from the literature-verified serial solver,
//! so this is the assembly literature anchor (Rosling equivalence + the serial Clark-Scarf anchor).
//!
//! VERIFIED SCOPE: the demand-facing (finished) stage has lead time 1. Component lead times may be
//! larger. The shared serial/assembly env currently has a KNOWN discrepancy when the demand-facing
//! stage has lead time >= 2 (the multi-stage simulation under-counts cost vs the exact solver;
//! single-stage is correct at every lead time). That is an open env-correctness item tracked in the
//! env docs and must be resolved before training on finished-lead-time >= 2 instances.

#[cfg(test)]
mod tests {
    use crate::problems::multi_echelon::assembly::echelon_base_stock::simulate;
    use crate::problems::multi_echelon::assembly::env::AssemblyConfig;
    use crate::problems::multi_echelon::assembly::rosling::reduce_equal_lead_time;
    use crate::problems::multi_echelon::serial::exact::{
        solve_from_local_costs, GridParams, SerialDemand,
    };

    fn rel_err(a: f64, b: f64) -> f64 {
        (a - b).abs() / b
    }

    /// The assembly env simulation under the Rosling serial-equivalent echelon base-stock levels
    /// reproduces the exact serial optimum (within Monte-Carlo error). Finished lead time is 1.
    fn check_assembly_matches_serial(config: &AssemblyConfig, demand: SerialDemand, seed: u64) {
        assert_eq!(config.finished_lead_time, 1, "verified scope: finished lead time 1");
        let serial = reduce_equal_lead_time(config);
        let exact = solve_from_local_costs(
            &serial.local_holding_upstream_to_downstream,
            &serial.lead_times_upstream_to_downstream,
            serial.penalty,
            demand,
            GridParams::default(),
        );
        let sim = simulate(config, demand, &exact.echelon_base_stock_levels, 400_000, 5_000, seed);
        assert!(
            rel_err(sim.average_cost, exact.optimal_cost) < 0.02,
            "assembly sim {:.4} should reproduce the Rosling serial optimum {:.4} (holding={:.3} backorder={:.3})",
            sim.average_cost,
            exact.optimal_cost,
            sim.average_holding_cost,
            sim.average_backorder_cost
        );
    }

    #[test]
    fn two_component_poisson_assembly_matches_rosling_serial() {
        // 2 components (holding 1 each -> kit holding 2), L_c=1; finished holding 3, L_a=1; p=10.
        let config = AssemblyConfig {
            component_holding_costs: vec![1.0, 1.0],
            component_lead_time: 1,
            finished_holding_cost: 3.0,
            finished_lead_time: 1,
            penalty: 10.0,
        };
        check_assembly_matches_serial(&config, SerialDemand::Poisson { mean: 5.0 }, 3);
    }

    #[test]
    fn three_component_longer_component_leadtime_matches_rosling_serial() {
        // 3 components (kit holding 3), component L_c=2; finished holding 7, L_a=1; p=37.12.
        // Equivalent serial [kit 3, finished 7], lead [2, 1] -- the two downstream stages of
        // Snyder & Shen Example 6.1. Component (upstream) lead time 2 is fully supported.
        let config = AssemblyConfig {
            component_holding_costs: vec![1.0, 1.0, 1.0],
            component_lead_time: 2,
            finished_holding_cost: 7.0,
            finished_lead_time: 1,
            penalty: 37.12,
        };
        check_assembly_matches_serial(&config, SerialDemand::Poisson { mean: 5.0 }, 7);
    }

    /// Heterogeneous component holding costs (kit holding = their sum) reduce cleanly.
    #[test]
    fn heterogeneous_components_match_rosling_serial() {
        let config = AssemblyConfig {
            component_holding_costs: vec![0.5, 1.5],
            component_lead_time: 2,
            finished_holding_cost: 4.0,
            finished_lead_time: 1,
            penalty: 20.0,
        };
        check_assembly_matches_serial(&config, SerialDemand::Poisson { mean: 4.0 }, 11);
    }
}
