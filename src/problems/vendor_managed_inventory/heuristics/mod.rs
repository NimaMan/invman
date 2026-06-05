mod dc_reserve_base_stock;
mod paper_mean_demand;
mod paper_newsvendor;
mod retailer_base_stock;

pub use dc_reserve_base_stock::dc_reserve_base_stock_shipment_quantity;
pub use paper_mean_demand::{paper_mean_demand_dispatch, paper_mean_demand_order_up_to_levels};
pub use paper_newsvendor::{
    paper_allocate_from_order_up_to_levels, paper_allocate_with_trucks, paper_newsvendor_dispatch,
    paper_newsvendor_order_up_to_levels, simulate_paper_policy, PaperPolicySimulationSummary,
};
pub use retailer_base_stock::retailer_base_stock_shipment_quantity;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::problems::vendor_managed_inventory::demand::{sample_demand, DemandDistributionKind};
use crate::problems::vendor_managed_inventory::env::{
    step_state, terminal_salvage_credit, validate_state, VendorManagedInventoryState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicySimulationSummary {
    pub mean_discounted_cost: f64,
    pub std_discounted_cost: f64,
}

fn policy_shipment_quantity(
    policy_name: &str,
    params: &[f64],
    state: &VendorManagedInventoryState,
    max_shipment_quantity: usize,
) -> PyResult<usize> {
    match policy_name {
        "retailer_base_stock" => {
            if params.len() != 1 {
                return Err(PyValueError::new_err(
                    "retailer_base_stock expects params [retailer_base_stock_level]",
                ));
            }
            retailer_base_stock_shipment_quantity(
                state,
                params[0].round().max(0.0) as usize,
                max_shipment_quantity,
            )
        }
        "dc_reserve_base_stock" => {
            if params.len() != 2 {
                return Err(PyValueError::new_err(
                    "dc_reserve_base_stock expects params [retailer_base_stock_level, dc_reserve_quantity]",
                ));
            }
            dc_reserve_base_stock_shipment_quantity(
                state,
                params[0].round().max(0.0) as usize,
                params[1].round().max(0.0) as usize,
                max_shipment_quantity,
            )
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported policy '{policy_name}'"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn policy_rollout_from_demands(
    policy_name: &str,
    params: &[f64],
    initial_state: &VendorManagedInventoryState,
    realized_demands: &[usize],
    dc_replenishment_quantity: usize,
    dc_capacity: usize,
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    max_shipment_quantity: usize,
    discount_factor: f64,
    salvage_value_per_unit: f64,
) -> PyResult<f64> {
    validate_state(initial_state, dc_capacity)?;
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }

    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;
    for demand in realized_demands.iter().copied() {
        let shipment_quantity =
            policy_shipment_quantity(policy_name, params, &state, max_shipment_quantity)?;
        let outcome = step_state(
            &state,
            shipment_quantity,
            demand,
            dc_replenishment_quantity,
            dc_capacity,
            shipment_cost_per_unit,
            dc_holding_cost_per_unit,
            retailer_holding_cost_per_unit,
            stockout_cost_per_unit,
        )?;
        discounted_cost += discount * outcome.period_cost;
        discount *= discount_factor;
        state = outcome.next_state;
    }

    discounted_cost -=
        discount * terminal_salvage_credit(&state, dc_capacity, salvage_value_per_unit)?;
    Ok(discounted_cost)
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_policy(
    policy_name: &str,
    params: &[f64],
    initial_state: &VendorManagedInventoryState,
    periods: usize,
    replications: usize,
    seed: u64,
    demand_mean: f64,
    demand_kind: DemandDistributionKind,
    dc_replenishment_quantity: usize,
    dc_capacity: usize,
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    max_shipment_quantity: usize,
    discount_factor: f64,
    salvage_value_per_unit: f64,
) -> PyResult<PolicySimulationSummary> {
    validate_state(initial_state, dc_capacity)?;
    if periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if replications == 0 {
        return Err(PyValueError::new_err("replications must be at least 1"));
    }
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut discounted_costs = Vec::with_capacity(replications);
    for _ in 0..replications {
        let mut state = initial_state.clone();
        let mut discounted_cost = 0.0;
        let mut discount = 1.0;

        for _ in 0..periods {
            let demand = sample_demand(&mut rng, demand_mean, demand_kind)?;
            let shipment_quantity =
                policy_shipment_quantity(policy_name, params, &state, max_shipment_quantity)?;
            let outcome = step_state(
                &state,
                shipment_quantity,
                demand,
                dc_replenishment_quantity,
                dc_capacity,
                shipment_cost_per_unit,
                dc_holding_cost_per_unit,
                retailer_holding_cost_per_unit,
                stockout_cost_per_unit,
            )?;
            discounted_cost += discount * outcome.period_cost;
            discount *= discount_factor;
            state = outcome.next_state;
        }

        discounted_cost -=
            discount * terminal_salvage_credit(&state, dc_capacity, salvage_value_per_unit)?;
        discounted_costs.push(discounted_cost);
    }

    let mean_discounted_cost = discounted_costs.iter().sum::<f64>() / discounted_costs.len() as f64;
    let variance = discounted_costs
        .iter()
        .map(|value| (value - mean_discounted_cost).powi(2))
        .sum::<f64>()
        / discounted_costs.len() as f64;
    Ok(PolicySimulationSummary {
        mean_discounted_cost,
        std_discounted_cost: variance.sqrt(),
    })
}
