mod env;
mod policies;
mod rollout;

use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};

use crate::env::lost_sales::{epoch_cost, initialize_state, LostSalesState};
use crate::policies::soft_tree::{
    soft_tree_leaf_probabilities, validate_soft_tree_shapes,
};
use crate::rollout::lost_sales_soft_tree::{
    population_rollout, rollout, rollout_from_demands, LostSalesRolloutConfig,
};

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
#[pyo3(signature = (state, split_weights, split_bias, leaf_logits, depth, max_order_size, temperature=0.25))]
fn soft_tree_action(
    state: Vec<f32>,
    split_weights: Vec<f32>,
    split_bias: Vec<f32>,
    leaf_logits: Vec<f32>,
    depth: usize,
    max_order_size: usize,
    temperature: f32,
) -> PyResult<usize> {
    validate_soft_tree_shapes(
        state.len(),
        split_weights.len(),
        split_bias.len(),
        leaf_logits.len(),
        depth,
    )?;

    let leaf_probs = soft_tree_leaf_probabilities(&state, &split_weights, &split_bias, depth, temperature);
    let mut action_value = 0.0f32;
    for (leaf_prob, leaf_logit) in leaf_probs.iter().zip(leaf_logits.iter()) {
        let quantity = 1.0 / (1.0 + (-leaf_logit).exp()) * max_order_size as f32;
        action_value += leaf_prob * quantity;
    }
    let clipped = action_value.round().clamp(0.0, max_order_size as f32);
    Ok(clipped as usize)
}

#[pyfunction]
#[pyo3(signature = (
    demand_rate,
    lead_time=4,
    max_order_size=20,
    holding_cost=1.0,
    shortage_cost=4.0,
    horizon=2000,
    action=0,
    seed=1234,
    warm_up_periods_ratio=0.2
))]
fn lost_sales_constant_action_rollout(
    demand_rate: f64,
    lead_time: usize,
    max_order_size: usize,
    holding_cost: f64,
    shortage_cost: f64,
    horizon: usize,
    action: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
) -> PyResult<f64> {
    if lead_time < 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "lead_time must be at least 1",
        ));
    }
    if action > max_order_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "action must be <= max_order_size",
        ));
    }
    if !(0.0..=1.0).contains(&warm_up_periods_ratio) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "warm_up_periods_ratio must be in [0, 1]",
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let demand_dist = Poisson::new(demand_rate)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(format!("invalid demand_rate: {err}")))?;
    let mut env_state = initialize_state(demand_rate, lead_time, max_order_size, &mut rng, &demand_dist);
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let mut epoch_costs = Vec::with_capacity(horizon);

    for _ in 0..horizon {
        let arriving_order = env_state.lead_time_orders.remove(0);
        env_state.lead_time_orders.push(action);
        env_state.current_inventory += arriving_order as i64;

        let demand = demand_dist.sample(&mut rng) as i64;
        epoch_costs.push(epoch_cost(
            &mut env_state.current_inventory,
            demand,
            action,
            holding_cost,
            shortage_cost,
            0.0,
            0.0,
        ));
    }

    let active_costs = if warm_up_periods < epoch_costs.len() {
        &epoch_costs[warm_up_periods..]
    } else {
        &epoch_costs[..]
    };
    Ok(active_costs.iter().sum::<f64>() / active_costs.len() as f64)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    max_order_size,
    demand_rate,
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    seed=1234,
    warm_up_periods_ratio=0.2,
    temperature=0.25
))]
fn lost_sales_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    max_order_size: usize,
    demand_rate: f64,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    temperature: f32,
) -> PyResult<f64> {
    let config = LostSalesRolloutConfig {
        input_dim,
        depth,
        max_order_size,
        demand_rate,
        lead_time,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon,
        warm_up_periods_ratio,
        temperature,
    };
    rollout(&flat_params, &config, seed)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    max_order_size,
    current_inventory,
    lead_time_orders,
    demands,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    warm_up_periods_ratio=0.2,
    temperature=0.25
))]
fn lost_sales_soft_tree_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    max_order_size: usize,
    current_inventory: i64,
    lead_time_orders: Vec<usize>,
    demands: Vec<usize>,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    warm_up_periods_ratio: f64,
    temperature: f32,
) -> PyResult<f64> {
    let config = LostSalesRolloutConfig {
        input_dim,
        depth,
        max_order_size,
        demand_rate: 0.0,
        lead_time: lead_time_orders.len(),
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon: demands.len(),
        warm_up_periods_ratio,
        temperature,
    };
    let env_state = LostSalesState {
        current_inventory,
        lead_time_orders,
    };
    rollout_from_demands(&flat_params, &config, env_state, &demands)
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    depth,
    max_order_size,
    demand_rate,
    seeds,
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    warm_up_periods_ratio=0.2,
    temperature=0.25
))]
fn lost_sales_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    max_order_size: usize,
    demand_rate: f64,
    seeds: Vec<u64>,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    warm_up_periods_ratio: f64,
    temperature: f32,
) -> PyResult<Vec<f64>> {
    let config = LostSalesRolloutConfig {
        input_dim,
        depth,
        max_order_size,
        demand_rate,
        lead_time,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon,
        warm_up_periods_ratio,
        temperature,
    };
    population_rollout(&params_batch, &config, &seeds)
}

#[pymodule]
fn invman_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(soft_tree_action, m)?)?;
    m.add_function(wrap_pyfunction!(lost_sales_constant_action_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(lost_sales_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(lost_sales_soft_tree_rollout_from_demands, m)?)?;
    m.add_function(wrap_pyfunction!(lost_sales_soft_tree_population_rollout, m)?)?;
    Ok(())
}
