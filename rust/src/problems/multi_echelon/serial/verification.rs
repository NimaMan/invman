#![allow(dead_code)]

//! Verification for the serial Clark-Scarf family: the env simulation under the optimal
//! echelon base-stock policy reproduces the exact analytical optimum, and the analytical
//! optimum matches the published literature value.
//!
//! Two complementary checks:
//! 1. exact: `exact.rs` reproduces the published optimum (Snyder & Shen Example 6.1
//!    optimal cost 47.65; discrete Poisson optima match the stockpyl reference impl);
//! 2. sim: `env.rs` + the optimal echelon base-stock policy reproduce that optimum by
//!    Monte-Carlo simulation, within sampling error. This is the confidence check that
//!    the training env is correct before any policy is trained on it.

#[cfg(test)]
mod tests {
    use crate::problems::multi_echelon::serial::echelon_base_stock::simulate;
    use crate::problems::multi_echelon::serial::env::SerialConfig;
    use crate::problems::multi_echelon::serial::exact::{
        solve_from_local_costs, solve_serial_clark_scarf, GridParams, SerialDemand, SerialStage,
    };

    fn rel_err(a: f64, b: f64) -> f64 {
        (a - b).abs() / b
    }

    /// Snyder & Shen "Fundamentals of Supply Chain Theory" Example 6.1: a 3-stage serial
    /// system, Normal(5,1) demand, lead times [2,1,1], echelon holding [2,2,3], stockout
    /// 37.12, published optimal cost 47.65. The env simulation under the exact-solver
    /// echelon base-stock levels reproduces it.
    #[test]
    fn env_simulation_reproduces_snyder_shen_example_6_1() {
        // Local (installation) holding costs upstream->downstream are [2,4,7]; lead times
        // upstream->downstream [2,1,1]; penalty 37.12; Normal(5,1).
        let demand = SerialDemand::Normal { mean: 5.0, std: 1.0 };
        let exact = solve_from_local_costs(&[2.0, 4.0, 7.0], &[2, 1, 1], 37.12, demand, GridParams::default());
        assert!(
            rel_err(exact.optimal_cost, 47.65) < 0.005,
            "exact solver cost {:.4} should match published 47.65",
            exact.optimal_cost
        );

        // env config is downstream->upstream: local holding [7,4,2], lead times [1,1,2].
        let config = SerialConfig {
            holding_cost: vec![7.0, 4.0, 2.0],
            lead_time: vec![1, 1, 2],
            penalty: 37.12,
        };
        let result = simulate(&config, demand, &exact.echelon_base_stock_levels, 400_000, 5_000, 17);
        assert!(
            rel_err(result.average_cost, 47.65) < 0.02,
            "env simulation cost {:.4} should reproduce the literature optimum 47.65 (holding={:.3} backorder={:.3})",
            result.average_cost,
            result.average_holding_cost,
            result.average_backorder_cost
        );
    }

    /// Discrete Poisson instances: the env simulation reproduces the exact discrete
    /// optima (which `exact.rs` matches to the stockpyl reference implementation).
    #[test]
    fn env_simulation_reproduces_poisson_optima() {
        // N=1: Poisson(5), echelon h=1, L=1, p=9 -> C* = 4.220849, S* = 8.
        let cfg1 = SerialConfig { holding_cost: vec![1.0], lead_time: vec![1], penalty: 9.0 };
        let r1 = simulate(&cfg1, SerialDemand::Poisson { mean: 5.0 }, &[8.0], 400_000, 5_000, 3);
        assert!(rel_err(r1.average_cost, 4.220849) < 0.02, "N1 sim {:.4} vs 4.2208", r1.average_cost);

        // N=2: downstream echelon h=2, upstream h=1, L=[1,1], p=10 -> C* = 16.7978, S*=[7,13].
        let cfg2 = SerialConfig { holding_cost: vec![3.0, 1.0], lead_time: vec![1, 1], penalty: 10.0 };
        // local installation downstream->upstream: H0 = h0+h1 = 2+1 = 3, H1 = 1.
        let r2 = simulate(&cfg2, SerialDemand::Poisson { mean: 5.0 }, &[7.0, 13.0], 400_000, 5_000, 3);
        assert!(rel_err(r2.average_cost, 16.797779) < 0.02, "N2 sim {:.4} vs 16.798", r2.average_cost);

        // N=3 Poisson(5): local holding [2,4,7] up->down, L=[2,1,1], p=37.12 -> C* = 72.0435.
        let exact3 = solve_from_local_costs(
            &[2.0, 4.0, 7.0], &[2, 1, 1], 37.12, SerialDemand::Poisson { mean: 5.0 }, GridParams::default());
        assert_eq!(exact3.echelon_base_stock_levels, vec![9.0, 15.0, 26.0]);
        let cfg3 = SerialConfig { holding_cost: vec![7.0, 4.0, 2.0], lead_time: vec![1, 1, 2], penalty: 37.12 };
        let r3 = simulate(&cfg3, SerialDemand::Poisson { mean: 5.0 }, &exact3.echelon_base_stock_levels, 400_000, 5_000, 3);
        assert!(rel_err(r3.average_cost, 72.043543) < 0.02, "N3 sim {:.4} vs 72.044", r3.average_cost);
    }

    /// The exact solver and the env simulation agree (decomposition vs simulation), an
    /// internal cross-check independent of the published rounding.
    #[test]
    fn exact_and_simulation_agree() {
        let demand = SerialDemand::Poisson { mean: 5.0 };
        let exact = solve_serial_clark_scarf(
            &[
                SerialStage { echelon_holding_cost: 3.0, lead_time: 1 },
                SerialStage { echelon_holding_cost: 2.0, lead_time: 1 },
                SerialStage { echelon_holding_cost: 2.0, lead_time: 2 },
            ],
            37.12,
            demand,
            GridParams::default(),
        );
        let config = SerialConfig { holding_cost: vec![7.0, 4.0, 2.0], lead_time: vec![1, 1, 2], penalty: 37.12 };
        let sim = simulate(&config, demand, &exact.echelon_base_stock_levels, 400_000, 5_000, 21);
        assert!(
            rel_err(sim.average_cost, exact.optimal_cost) < 0.01,
            "sim {:.4} vs exact {:.4}",
            sim.average_cost,
            exact.optimal_cost
        );
    }
}
