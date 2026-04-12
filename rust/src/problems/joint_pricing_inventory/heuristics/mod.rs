mod inventory_sensitive_base_stock;
mod static_price_base_stock;

pub use inventory_sensitive_base_stock::inventory_sensitive_base_stock_action;
pub use static_price_base_stock::static_price_base_stock_action;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::problems::joint_pricing_inventory::demand::{
    sample_demand, validate_price_ladder, DemandDistributionKind,
};
use crate::problems::joint_pricing_inventory::env::{
    clip_action, step_state, terminal_salvage_credit, JointPricingInventoryState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicySimulationSummary {
    pub mean_discounted_cost: f64,
    pub std_discounted_cost: f64,
}

fn clip_and_validate_action(
    order_quantity: usize,
    price_index: usize,
    max_order_quantity: usize,
    num_prices: usize,
) -> PyResult<(usize, usize)> {
    clip_action(order_quantity, price_index, max_order_quantity, num_prices)
}

fn policy_action(
    policy_name: &str,
    params: &[f64],
    state: &JointPricingInventoryState,
    max_order_quantity: usize,
    num_prices: usize,
) -> PyResult<(usize, usize)> {
    match policy_name {
        "static_price_base_stock" => {
            if params.len() != 2 {
                return Err(PyValueError::new_err(
                    "static_price_base_stock expects params [order_up_to, price_index]",
                ));
            }
            static_price_base_stock_action(
                state.inventory_level,
                params[0].round().max(0.0) as usize,
                params[1].round().max(0.0) as usize,
                max_order_quantity,
                num_prices,
            )
        }
        "inventory_sensitive_base_stock" => {
            if params.len() != 4 {
                return Err(PyValueError::new_err(
                    "inventory_sensitive_base_stock expects params [order_up_to, markdown_threshold, high_price_index, low_price_index]",
                ));
            }
            inventory_sensitive_base_stock_action(
                state.inventory_level,
                params[0].round().max(0.0) as usize,
                params[1].round().max(0.0) as usize,
                params[2].round().max(0.0) as usize,
                params[3].round().max(0.0) as usize,
                max_order_quantity,
                num_prices,
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
    initial_state: &JointPricingInventoryState,
    realized_demands: &[usize],
    price_levels: &[f64],
    procurement_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    max_order_quantity: usize,
    discount_factor: f64,
    salvage_value_per_unit: f64,
) -> PyResult<f64> {
    validate_price_ladder(price_levels, &vec![0.0; price_levels.len()])?;
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }

    let mut state = initial_state.clone();
    let mut discounted_cost = 0.0;
    let mut discount = 1.0;

    for demand in realized_demands.iter().copied() {
        let (order_quantity, price_index) = policy_action(
            policy_name,
            params,
            &state,
            max_order_quantity,
            price_levels.len(),
        )?;
        let outcome = step_state(
            &state,
            order_quantity,
            price_index,
            demand,
            price_levels,
            procurement_cost_per_unit,
            holding_cost_per_unit,
            stockout_cost_per_unit,
        )?;
        discounted_cost += discount * outcome.period_cost;
        discount *= discount_factor;
        state = outcome.next_state;
    }

    discounted_cost -= discount * terminal_salvage_credit(&state, salvage_value_per_unit)?;
    Ok(discounted_cost)
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_policy(
    policy_name: &str,
    params: &[f64],
    initial_state: &JointPricingInventoryState,
    periods: usize,
    replications: usize,
    seed: u64,
    price_levels: &[f64],
    demand_means: &[f64],
    demand_kind: DemandDistributionKind,
    procurement_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    max_order_quantity: usize,
    discount_factor: f64,
    salvage_value_per_unit: f64,
) -> PyResult<PolicySimulationSummary> {
    validate_price_ladder(price_levels, demand_means)?;
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
            let (order_quantity, price_index) = policy_action(
                policy_name,
                params,
                &state,
                max_order_quantity,
                price_levels.len(),
            )?;
            let demand = sample_demand(&mut rng, price_index, demand_means, demand_kind)?;
            let outcome = step_state(
                &state,
                order_quantity,
                price_index,
                demand,
                price_levels,
                procurement_cost_per_unit,
                holding_cost_per_unit,
                stockout_cost_per_unit,
            )?;
            discounted_cost += discount * outcome.period_cost;
            discount *= discount_factor;
            state = outcome.next_state;
        }

        discounted_cost -= discount * terminal_salvage_credit(&state, salvage_value_per_unit)?;
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
