use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::core::policies::dense::{parse_activation, parse_policy_head};
use crate::core::policies::soft_tree::{parse_leaf_type, parse_split_type};
use crate::problems::lost_sales::env::{
    build_demand_distribution, epoch_cost, initialize_state, parse_demand_kind, sample_demand,
    LostSalesDemandKind, LostSalesState,
};
use crate::problems::lost_sales::rollout::{
    linear_population_rollout as lost_sales_linear_population_rollout_impl,
    linear_rollout as lost_sales_linear_rollout_impl,
    linear_rollout_from_demands as lost_sales_linear_rollout_from_demands_impl,
    neural_population_rollout as lost_sales_neural_population_rollout_impl,
    neural_rollout as lost_sales_neural_rollout_impl,
    neural_rollout_from_demands as lost_sales_neural_rollout_from_demands_impl,
    population_rollout as lost_sales_population_rollout, rollout as lost_sales_rollout,
    rollout_from_demands as lost_sales_rollout_from_demands, LostSalesLinearRolloutConfig,
    LostSalesNeuralRolloutConfig, LostSalesRolloutConfig,
};

fn parse_lost_sales_demand_kind(demand_dist_name: &str) -> PyResult<LostSalesDemandKind> {
    parse_demand_kind(demand_dist_name).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (
    demand_rate,
    demand_dist_name="Poisson",
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
    demand_dist_name: &str,
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
    let demand_dist =
        build_demand_distribution(parse_lost_sales_demand_kind(demand_dist_name)?, demand_rate)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut env_state = initialize_state(
        demand_rate,
        lead_time,
        max_order_size,
        &mut rng,
        &demand_dist,
    );
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let mut epoch_costs = Vec::with_capacity(horizon);

    for _ in 0..horizon {
        let arriving_order = env_state.lead_time_orders.remove(0);
        env_state.lead_time_orders.push(action);
        env_state.current_inventory += arriving_order as i64;

        let demand = sample_demand(&mut rng, &demand_dist);
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
    demand_dist_name="Poisson",
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    seed=1234,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant"
))]
fn lost_sales_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    max_order_size: usize,
    demand_rate: f64,
    demand_dist_name: &str,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
) -> PyResult<f64> {
    let config = LostSalesRolloutConfig {
        input_dim,
        depth,
        max_order_size,
        demand_kind: parse_lost_sales_demand_kind(demand_dist_name)?,
        demand_rate,
        lead_time,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    lost_sales_rollout(&flat_params, &config, seed)
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
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant"
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
    split_type: &str,
    leaf_type: &str,
) -> PyResult<f64> {
    let config = LostSalesRolloutConfig {
        input_dim,
        depth,
        max_order_size,
        demand_kind: LostSalesDemandKind::Poisson,
        demand_rate: 0.0,
        lead_time: lead_time_orders.len(),
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon: demands.len(),
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    let env_state = LostSalesState {
        current_inventory,
        lead_time_orders,
    };
    lost_sales_rollout_from_demands(&flat_params, &config, env_state, &demands)
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    depth,
    max_order_size,
    demand_rate,
    seeds,
    demand_dist_name="Poisson",
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant"
))]
fn lost_sales_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    max_order_size: usize,
    demand_rate: f64,
    seeds: Vec<u64>,
    demand_dist_name: &str,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
) -> PyResult<Vec<f64>> {
    let config = LostSalesRolloutConfig {
        input_dim,
        depth,
        max_order_size,
        demand_kind: parse_lost_sales_demand_kind(demand_dist_name)?,
        demand_rate,
        lead_time,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    lost_sales_population_rollout(&params_batch, &config, &seeds)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    output_dim,
    max_order_size,
    demand_rate,
    demand_dist_name="Poisson",
    policy_head="categorical_quantity",
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    seed=1234,
    warm_up_periods_ratio=0.2
))]
fn lost_sales_linear_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
    max_order_size: usize,
    demand_rate: f64,
    demand_dist_name: &str,
    policy_head: &str,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
) -> PyResult<f64> {
    let config = LostSalesLinearRolloutConfig {
        input_dim,
        output_dim,
        max_order_size,
        policy_head: parse_policy_head(policy_head)?,
        demand_kind: parse_lost_sales_demand_kind(demand_dist_name)?,
        demand_rate,
        lead_time,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon,
        warm_up_periods_ratio,
    };
    lost_sales_linear_rollout_impl(&flat_params, &config, seed)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    output_dim,
    max_order_size,
    current_inventory,
    lead_time_orders,
    demands,
    policy_head="categorical_quantity",
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    warm_up_periods_ratio=0.2
))]
fn lost_sales_linear_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
    max_order_size: usize,
    current_inventory: i64,
    lead_time_orders: Vec<usize>,
    demands: Vec<usize>,
    policy_head: &str,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    warm_up_periods_ratio: f64,
) -> PyResult<f64> {
    let config = LostSalesLinearRolloutConfig {
        input_dim,
        output_dim,
        max_order_size,
        policy_head: parse_policy_head(policy_head)?,
        demand_kind: LostSalesDemandKind::Poisson,
        demand_rate: 0.0,
        lead_time: lead_time_orders.len(),
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon: demands.len(),
        warm_up_periods_ratio,
    };
    let env_state = LostSalesState {
        current_inventory,
        lead_time_orders,
    };
    lost_sales_linear_rollout_from_demands_impl(&flat_params, &config, env_state, &demands)
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    output_dim,
    max_order_size,
    demand_rate,
    seeds,
    demand_dist_name="Poisson",
    policy_head="categorical_quantity",
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    warm_up_periods_ratio=0.2
))]
fn lost_sales_linear_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    output_dim: usize,
    max_order_size: usize,
    demand_rate: f64,
    seeds: Vec<u64>,
    demand_dist_name: &str,
    policy_head: &str,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    warm_up_periods_ratio: f64,
) -> PyResult<Vec<f64>> {
    let config = LostSalesLinearRolloutConfig {
        input_dim,
        output_dim,
        max_order_size,
        policy_head: parse_policy_head(policy_head)?,
        demand_kind: parse_lost_sales_demand_kind(demand_dist_name)?,
        demand_rate,
        lead_time,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon,
        warm_up_periods_ratio,
    };
    lost_sales_linear_population_rollout_impl(&params_batch, &config, &seeds)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    hidden_dims,
    output_dim,
    max_order_size,
    activation,
    demand_rate,
    demand_dist_name="Poisson",
    policy_head="categorical_quantity",
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    seed=1234,
    warm_up_periods_ratio=0.2
))]
fn lost_sales_nn_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    max_order_size: usize,
    activation: &str,
    demand_rate: f64,
    demand_dist_name: &str,
    policy_head: &str,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
) -> PyResult<f64> {
    let config = LostSalesNeuralRolloutConfig {
        input_dim,
        hidden_dims,
        output_dim,
        max_order_size,
        policy_head: parse_policy_head(policy_head)?,
        demand_kind: parse_lost_sales_demand_kind(demand_dist_name)?,
        demand_rate,
        lead_time,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon,
        warm_up_periods_ratio,
        activation: parse_activation(activation)?,
    };
    lost_sales_neural_rollout_impl(&flat_params, &config, seed)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    hidden_dims,
    output_dim,
    max_order_size,
    activation,
    current_inventory,
    lead_time_orders,
    demands,
    policy_head="categorical_quantity",
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    warm_up_periods_ratio=0.2
))]
fn lost_sales_nn_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    max_order_size: usize,
    activation: &str,
    current_inventory: i64,
    lead_time_orders: Vec<usize>,
    demands: Vec<usize>,
    policy_head: &str,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    warm_up_periods_ratio: f64,
) -> PyResult<f64> {
    let config = LostSalesNeuralRolloutConfig {
        input_dim,
        hidden_dims,
        output_dim,
        max_order_size,
        policy_head: parse_policy_head(policy_head)?,
        demand_kind: LostSalesDemandKind::Poisson,
        demand_rate: 0.0,
        lead_time: lead_time_orders.len(),
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon: demands.len(),
        warm_up_periods_ratio,
        activation: parse_activation(activation)?,
    };
    let env_state = LostSalesState {
        current_inventory,
        lead_time_orders,
    };
    lost_sales_neural_rollout_from_demands_impl(&flat_params, &config, env_state, &demands)
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    hidden_dims,
    output_dim,
    max_order_size,
    activation,
    demand_rate,
    seeds,
    demand_dist_name="Poisson",
    policy_head="categorical_quantity",
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    warm_up_periods_ratio=0.2
))]
fn lost_sales_nn_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    max_order_size: usize,
    activation: &str,
    demand_rate: f64,
    seeds: Vec<u64>,
    demand_dist_name: &str,
    policy_head: &str,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    warm_up_periods_ratio: f64,
) -> PyResult<Vec<f64>> {
    let config = LostSalesNeuralRolloutConfig {
        input_dim,
        hidden_dims,
        output_dim,
        max_order_size,
        policy_head: parse_policy_head(policy_head)?,
        demand_kind: parse_lost_sales_demand_kind(demand_dist_name)?,
        demand_rate,
        lead_time,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon,
        warm_up_periods_ratio,
        activation: parse_activation(activation)?,
    };
    lost_sales_neural_population_rollout_impl(&params_batch, &config, &seeds)
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lost_sales_constant_action_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(lost_sales_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_soft_tree_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lost_sales_linear_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(lost_sales_linear_rollout_from_demands, m)?)?;
    m.add_function(wrap_pyfunction!(lost_sales_linear_population_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(lost_sales_nn_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(lost_sales_nn_rollout_from_demands, m)?)?;
    m.add_function(wrap_pyfunction!(lost_sales_nn_population_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_soft_tree_population_rollout,
        m
    )?)?;
    Ok(())
}
