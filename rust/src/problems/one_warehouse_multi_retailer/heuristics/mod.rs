mod echelon_base_stock;

pub use echelon_base_stock::echelon_base_stock_orders;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::problems::one_warehouse_multi_retailer::allocation::{
    min_shortage_shipments, proportional_shipments, random_sequential_shipments, AllocationPolicy,
};
use crate::problems::one_warehouse_multi_retailer::demand::{sample_demand, DemandModel};
use crate::problems::one_warehouse_multi_retailer::env::{
    retailer_inventory_positions, step_state, validate_state, CustomerBehaviorModel,
    OneWarehouseMultiRetailerState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicySimulationSummary {
    pub mean_cost: f64,
    pub cost_std: f64,
}

fn parse_echelon_base_stock_params(
    params: &[f64],
    num_retailers: usize,
) -> PyResult<(usize, Vec<usize>)> {
    if params.len() != num_retailers + 1 {
        return Err(PyValueError::new_err(format!(
            "echelon_base_stock expects {} parameters: warehouse base-stock plus one retailer base-stock per retailer",
            num_retailers + 1
        )));
    }
    Ok((
        params[0].round().max(0.0) as usize,
        params[1..]
            .iter()
            .map(|value| value.round().max(0.0) as usize)
            .collect(),
    ))
}

fn retailer_orders_from_policy(
    policy_name: &str,
    params: &[f64],
    state: &OneWarehouseMultiRetailerState,
) -> PyResult<(usize, Vec<usize>, Vec<usize>)> {
    match policy_name {
        "echelon_base_stock" => {
            let (warehouse_base_stock, retailer_base_stocks) =
                parse_echelon_base_stock_params(params, state.retailer_inventory.len())?;
            let orders =
                echelon_base_stock_orders(state, warehouse_base_stock, &retailer_base_stocks)?;
            Ok((orders[0], orders[1..].to_vec(), retailer_base_stocks))
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported policy '{policy_name}'",
        ))),
    }
}

fn retailer_shipments_for_policy<R: Rng + ?Sized>(
    rng: &mut R,
    state: &OneWarehouseMultiRetailerState,
    retailer_orders: &[usize],
    retailer_base_stocks: &[usize],
    allocation_policy: AllocationPolicy,
) -> PyResult<Vec<usize>> {
    let available_warehouse_inventory =
        (state.warehouse_inventory + state.warehouse_pipeline[0] as i32).max(0) as usize;
    match allocation_policy {
        AllocationPolicy::Proportional => {
            proportional_shipments(available_warehouse_inventory, retailer_orders)
        }
        AllocationPolicy::RandomSequential => {
            random_sequential_shipments(rng, available_warehouse_inventory, retailer_orders)
        }
        AllocationPolicy::MinShortage => min_shortage_shipments(
            available_warehouse_inventory,
            retailer_orders,
            &retailer_inventory_positions(state)?,
            retailer_base_stocks,
        ),
    }
}

pub fn policy_rollout_from_paths(
    policy_name: &str,
    params: &[f64],
    initial_state: &OneWarehouseMultiRetailerState,
    demands: &[Vec<usize>],
    allocation_policy: AllocationPolicy,
    holding_cost_warehouse: f64,
    holding_cost_retailers: &[f64],
    penalty_costs_retailers: &[f64],
    customer_behavior: CustomerBehaviorModel,
    emergency_shipment_probability: f64,
    discount_factor: f64,
    seed: u64,
) -> PyResult<f64> {
    validate_state(initial_state)?;
    if !(0.0..=1.0).contains(&discount_factor) {
        return Err(PyValueError::new_err("discount_factor must lie in [0, 1]"));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut total_discounted_cost = 0.0;
    let mut discount = 1.0;

    for demand in demands.iter() {
        if demand.len() != state.retailer_inventory.len() {
            return Err(PyValueError::new_err(
                "each realized demand vector must match the number of retailers",
            ));
        }
        let (warehouse_order, retailer_orders, retailer_base_stocks) =
            retailer_orders_from_policy(policy_name, params, &state)?;
        let retailer_shipments = retailer_shipments_for_policy(
            &mut rng,
            &state,
            &retailer_orders,
            &retailer_base_stocks,
            allocation_policy,
        )?;
        let emergency_draws = if customer_behavior == CustomerBehaviorModel::PartialBackorder {
            Some(
                (0..state.retailer_inventory.len())
                    .map(|_| rng.gen_bool(emergency_shipment_probability))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let outcome = step_state(
            &state,
            warehouse_order,
            &retailer_shipments,
            demand,
            holding_cost_warehouse,
            holding_cost_retailers,
            penalty_costs_retailers,
            customer_behavior,
            emergency_shipment_probability,
            emergency_draws.as_deref(),
        )?;
        total_discounted_cost += discount * outcome.period_cost;
        discount *= discount_factor;
        state = outcome.next_state;
    }

    Ok(total_discounted_cost)
}

pub fn simulate_policy(
    policy_name: &str,
    params: &[f64],
    initial_state: &OneWarehouseMultiRetailerState,
    periods: usize,
    replications: usize,
    seed: u64,
    demand_models: &[DemandModel],
    allocation_policy: AllocationPolicy,
    holding_cost_warehouse: f64,
    holding_cost_retailers: &[f64],
    penalty_costs_retailers: &[f64],
    customer_behavior: CustomerBehaviorModel,
    emergency_shipment_probability: f64,
    discount_factor: f64,
) -> PyResult<PolicySimulationSummary> {
    if periods == 0 {
        return Err(PyValueError::new_err("periods must be at least 1"));
    }
    if replications == 0 {
        return Err(PyValueError::new_err("replications must be at least 1"));
    }
    if demand_models.len() != initial_state.retailer_inventory.len() {
        return Err(PyValueError::new_err(
            "demand_models length must match the number of retailers",
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut discounted_costs = Vec::with_capacity(replications);

    for _ in 0..replications {
        let mut state = initial_state.clone();
        let mut total_discounted_cost = 0.0;
        let mut discount = 1.0;

        for _ in 0..periods {
            let realized_demands = demand_models
                .iter()
                .map(|model| sample_demand(&mut rng, model))
                .collect::<PyResult<Vec<_>>>()?;
            let (warehouse_order, retailer_orders, retailer_base_stocks) =
                retailer_orders_from_policy(policy_name, params, &state)?;
            let retailer_shipments = retailer_shipments_for_policy(
                &mut rng,
                &state,
                &retailer_orders,
                &retailer_base_stocks,
                allocation_policy,
            )?;
            let emergency_draws = if customer_behavior == CustomerBehaviorModel::PartialBackorder {
                Some(
                    (0..state.retailer_inventory.len())
                        .map(|_| rng.gen_bool(emergency_shipment_probability))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let outcome = step_state(
                &state,
                warehouse_order,
                &retailer_shipments,
                &realized_demands,
                holding_cost_warehouse,
                holding_cost_retailers,
                penalty_costs_retailers,
                customer_behavior,
                emergency_shipment_probability,
                emergency_draws.as_deref(),
            )?;
            total_discounted_cost += discount * outcome.period_cost;
            discount *= discount_factor;
            state = outcome.next_state;
        }

        discounted_costs.push(total_discounted_cost);
    }

    let mean_cost = discounted_costs.iter().sum::<f64>() / discounted_costs.len() as f64;
    let variance = discounted_costs
        .iter()
        .map(|value| (value - mean_cost).powi(2))
        .sum::<f64>()
        / discounted_costs.len() as f64;
    Ok(PolicySimulationSummary {
        mean_cost,
        cost_std: variance.sqrt(),
    })
}
