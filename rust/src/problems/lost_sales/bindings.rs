use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::core::policies::dense::{parse_activation, parse_policy_head};
use crate::core::policies::soft_tree::{parse_leaf_type, parse_split_type};
use crate::problems::lost_sales::demand::{
    build_demand_process, parse_demand_kind, sample_demand, LostSalesDemandConfig,
    LostSalesDemandKind, DEFAULT_MMPP2_LAMBDA_HIGH, DEFAULT_MMPP2_LAMBDA_LOW,
    DEFAULT_MMPP2_POSITIVE_P00, DEFAULT_MMPP2_POSITIVE_P11,
};
use crate::problems::lost_sales::env::{
    epoch_cost, initialize_state, LostSalesState, StateNormalizer,
};
use crate::problems::lost_sales::heuristics::{
    evaluate_heuristic_policy, LostSalesHeuristicPolicyKind, LostSalesHeuristicVerificationConfig,
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

fn build_lost_sales_demand_config(
    demand_dist_name: &str,
    demand_rate: f64,
    demand_lambda_low: f64,
    demand_lambda_high: f64,
    demand_p00: f64,
    demand_p11: f64,
) -> PyResult<LostSalesDemandConfig> {
    Ok(LostSalesDemandConfig {
        kind: parse_lost_sales_demand_kind(demand_dist_name)?,
        demand_rate,
        demand_lambda_low,
        demand_lambda_high,
        demand_p00,
        demand_p11,
    })
}

fn empirical_mean_demand(demands: &[usize]) -> f64 {
    if demands.is_empty() {
        return 0.0;
    }
    let total: usize = demands.iter().copied().sum();
    total as f64 / demands.len() as f64
}

fn parse_state_normalizer(state_normalizer: &str) -> PyResult<StateNormalizer> {
    match state_normalizer {
        "identity" | "none" | "raw" => Ok(StateNormalizer::Identity),
        "quantity_scale" | "qscale" | "scale" | "divide_by_scale" | "scalar_divide" => {
            Ok(StateNormalizer::DivideByScale)
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported lost-sales rust state_normalizer '{other}', expected one of: identity, quantity_scale"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (
    demand_rate,
    demand_dist_name="Poisson",
    demand_lambda_low=DEFAULT_MMPP2_LAMBDA_LOW,
    demand_lambda_high=DEFAULT_MMPP2_LAMBDA_HIGH,
    demand_p00=DEFAULT_MMPP2_POSITIVE_P00,
    demand_p11=DEFAULT_MMPP2_POSITIVE_P11,
    lead_time=4,
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
    demand_lambda_low: f64,
    demand_lambda_high: f64,
    demand_p00: f64,
    demand_p11: f64,
    lead_time: usize,
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
    if !(0.0..=1.0).contains(&warm_up_periods_ratio) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "warm_up_periods_ratio must be in [0, 1]",
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let demand_config = build_lost_sales_demand_config(
        demand_dist_name,
        demand_rate,
        demand_lambda_low,
        demand_lambda_high,
        demand_p00,
        demand_p11,
    )?;
    let mut demand_process = build_demand_process(demand_config, &mut rng)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut env_state = initialize_state(demand_rate, lead_time, &mut rng, &mut demand_process);
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let mut epoch_costs = Vec::with_capacity(horizon);

    for _ in 0..horizon {
        let arriving_order = env_state.lead_time_orders.remove(0);
        env_state.lead_time_orders.push(action);
        env_state.current_inventory += arriving_order as i64;

        let demand = sample_demand(&mut rng, &mut demand_process);
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
    demand_rate,
    demand_dist_name="Poisson",
    demand_lambda_low=DEFAULT_MMPP2_LAMBDA_LOW,
    demand_lambda_high=DEFAULT_MMPP2_LAMBDA_HIGH,
    demand_p00=DEFAULT_MMPP2_POSITIVE_P00,
    demand_p11=DEFAULT_MMPP2_POSITIVE_P11,
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
    leaf_type="linear",
    policy_max_quantity=None,
    state_normalizer="identity",
    state_scale=None
))]
fn lost_sales_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    demand_rate: f64,
    demand_dist_name: &str,
    demand_lambda_low: f64,
    demand_lambda_high: f64,
    demand_p00: f64,
    demand_p11: f64,
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
    policy_max_quantity: Option<usize>,
    state_normalizer: &str,
    state_scale: Option<f64>,
) -> PyResult<f64> {
    let config = LostSalesRolloutConfig {
        input_dim,
        depth,
        policy_max_quantity,
        state_scale,
        state_normalizer: parse_state_normalizer(state_normalizer)?,
        demand_config: build_lost_sales_demand_config(
            demand_dist_name,
            demand_rate,
            demand_lambda_low,
            demand_lambda_high,
            demand_p00,
            demand_p11,
        )?,
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
    leaf_type="linear",
    policy_max_quantity=None,
    state_normalizer="identity",
    state_scale=None
))]
fn lost_sales_soft_tree_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
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
    policy_max_quantity: Option<usize>,
    state_normalizer: &str,
    state_scale: Option<f64>,
) -> PyResult<f64> {
    let empirical_mean = empirical_mean_demand(&demands);
    let config = LostSalesRolloutConfig {
        input_dim,
        depth,
        policy_max_quantity,
        state_scale,
        state_normalizer: parse_state_normalizer(state_normalizer)?,
        demand_config: LostSalesDemandConfig {
            kind: LostSalesDemandKind::Poisson,
            demand_rate: empirical_mean,
            demand_lambda_low: DEFAULT_MMPP2_LAMBDA_LOW,
            demand_lambda_high: DEFAULT_MMPP2_LAMBDA_HIGH,
            demand_p00: DEFAULT_MMPP2_POSITIVE_P00,
            demand_p11: DEFAULT_MMPP2_POSITIVE_P11,
        },
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
    demand_rate,
    seeds,
    demand_dist_name="Poisson",
    demand_lambda_low=DEFAULT_MMPP2_LAMBDA_LOW,
    demand_lambda_high=DEFAULT_MMPP2_LAMBDA_HIGH,
    demand_p00=DEFAULT_MMPP2_POSITIVE_P00,
    demand_p11=DEFAULT_MMPP2_POSITIVE_P11,
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear",
    policy_max_quantity=None,
    state_normalizer="identity",
    state_scale=None
))]
fn lost_sales_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    demand_rate: f64,
    seeds: Vec<u64>,
    demand_dist_name: &str,
    demand_lambda_low: f64,
    demand_lambda_high: f64,
    demand_p00: f64,
    demand_p11: f64,
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
    policy_max_quantity: Option<usize>,
    state_normalizer: &str,
    state_scale: Option<f64>,
) -> PyResult<Vec<f64>> {
    let config = LostSalesRolloutConfig {
        input_dim,
        depth,
        policy_max_quantity,
        state_scale,
        state_normalizer: parse_state_normalizer(state_normalizer)?,
        demand_config: build_lost_sales_demand_config(
            demand_dist_name,
            demand_rate,
            demand_lambda_low,
            demand_lambda_high,
            demand_p00,
            demand_p11,
        )?,
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
    policy_max_quantity,
    demand_rate,
    demand_dist_name="Poisson",
    demand_lambda_low=DEFAULT_MMPP2_LAMBDA_LOW,
    demand_lambda_high=DEFAULT_MMPP2_LAMBDA_HIGH,
    demand_p00=DEFAULT_MMPP2_POSITIVE_P00,
    demand_p11=DEFAULT_MMPP2_POSITIVE_P11,
    policy_head="categorical_quantity",
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    seed=1234,
    warm_up_periods_ratio=0.2,
    state_normalizer="identity",
    state_scale=None
))]
fn lost_sales_linear_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
    policy_max_quantity: Option<usize>,
    demand_rate: f64,
    demand_dist_name: &str,
    demand_lambda_low: f64,
    demand_lambda_high: f64,
    demand_p00: f64,
    demand_p11: f64,
    policy_head: &str,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    state_normalizer: &str,
    state_scale: Option<f64>,
) -> PyResult<f64> {
    let config = LostSalesLinearRolloutConfig {
        input_dim,
        output_dim,
        policy_max_quantity,
        state_scale,
        state_normalizer: parse_state_normalizer(state_normalizer)?,
        policy_head: parse_policy_head(policy_head)?,
        demand_config: build_lost_sales_demand_config(
            demand_dist_name,
            demand_rate,
            demand_lambda_low,
            demand_lambda_high,
            demand_p00,
            demand_p11,
        )?,
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
    policy_max_quantity,
    current_inventory,
    lead_time_orders,
    demands,
    policy_head="categorical_quantity",
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    warm_up_periods_ratio=0.2,
    state_normalizer="identity",
    state_scale=None
))]
fn lost_sales_linear_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
    policy_max_quantity: Option<usize>,
    current_inventory: i64,
    lead_time_orders: Vec<usize>,
    demands: Vec<usize>,
    policy_head: &str,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    warm_up_periods_ratio: f64,
    state_normalizer: &str,
    state_scale: Option<f64>,
) -> PyResult<f64> {
    let empirical_mean = empirical_mean_demand(&demands);
    let config = LostSalesLinearRolloutConfig {
        input_dim,
        output_dim,
        policy_max_quantity,
        state_scale,
        state_normalizer: parse_state_normalizer(state_normalizer)?,
        policy_head: parse_policy_head(policy_head)?,
        demand_config: LostSalesDemandConfig {
            kind: LostSalesDemandKind::Poisson,
            demand_rate: empirical_mean,
            demand_lambda_low: DEFAULT_MMPP2_LAMBDA_LOW,
            demand_lambda_high: DEFAULT_MMPP2_LAMBDA_HIGH,
            demand_p00: DEFAULT_MMPP2_POSITIVE_P00,
            demand_p11: DEFAULT_MMPP2_POSITIVE_P11,
        },
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
    policy_max_quantity,
    demand_rate,
    seeds,
    demand_dist_name="Poisson",
    demand_lambda_low=DEFAULT_MMPP2_LAMBDA_LOW,
    demand_lambda_high=DEFAULT_MMPP2_LAMBDA_HIGH,
    demand_p00=DEFAULT_MMPP2_POSITIVE_P00,
    demand_p11=DEFAULT_MMPP2_POSITIVE_P11,
    policy_head="categorical_quantity",
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    warm_up_periods_ratio=0.2,
    state_normalizer="identity",
    state_scale=None
))]
fn lost_sales_linear_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    output_dim: usize,
    policy_max_quantity: Option<usize>,
    demand_rate: f64,
    seeds: Vec<u64>,
    demand_dist_name: &str,
    demand_lambda_low: f64,
    demand_lambda_high: f64,
    demand_p00: f64,
    demand_p11: f64,
    policy_head: &str,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    warm_up_periods_ratio: f64,
    state_normalizer: &str,
    state_scale: Option<f64>,
) -> PyResult<Vec<f64>> {
    let config = LostSalesLinearRolloutConfig {
        input_dim,
        output_dim,
        policy_max_quantity,
        state_scale,
        state_normalizer: parse_state_normalizer(state_normalizer)?,
        policy_head: parse_policy_head(policy_head)?,
        demand_config: build_lost_sales_demand_config(
            demand_dist_name,
            demand_rate,
            demand_lambda_low,
            demand_lambda_high,
            demand_p00,
            demand_p11,
        )?,
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
    policy_max_quantity,
    activation,
    demand_rate,
    demand_dist_name="Poisson",
    demand_lambda_low=DEFAULT_MMPP2_LAMBDA_LOW,
    demand_lambda_high=DEFAULT_MMPP2_LAMBDA_HIGH,
    demand_p00=DEFAULT_MMPP2_POSITIVE_P00,
    demand_p11=DEFAULT_MMPP2_POSITIVE_P11,
    policy_head="categorical_quantity",
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    seed=1234,
    warm_up_periods_ratio=0.2,
    state_normalizer="identity",
    state_scale=None
))]
fn lost_sales_nn_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    policy_max_quantity: Option<usize>,
    activation: &str,
    demand_rate: f64,
    demand_dist_name: &str,
    demand_lambda_low: f64,
    demand_lambda_high: f64,
    demand_p00: f64,
    demand_p11: f64,
    policy_head: &str,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    state_normalizer: &str,
    state_scale: Option<f64>,
) -> PyResult<f64> {
    let config = LostSalesNeuralRolloutConfig {
        input_dim,
        hidden_dims,
        output_dim,
        policy_max_quantity,
        state_scale,
        state_normalizer: parse_state_normalizer(state_normalizer)?,
        policy_head: parse_policy_head(policy_head)?,
        demand_config: build_lost_sales_demand_config(
            demand_dist_name,
            demand_rate,
            demand_lambda_low,
            demand_lambda_high,
            demand_p00,
            demand_p11,
        )?,
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
    policy_max_quantity,
    activation,
    current_inventory,
    lead_time_orders,
    demands,
    policy_head="categorical_quantity",
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    warm_up_periods_ratio=0.2,
    state_normalizer="identity",
    state_scale=None
))]
fn lost_sales_nn_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    policy_max_quantity: Option<usize>,
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
    state_normalizer: &str,
    state_scale: Option<f64>,
) -> PyResult<f64> {
    let empirical_mean = empirical_mean_demand(&demands);
    let config = LostSalesNeuralRolloutConfig {
        input_dim,
        hidden_dims,
        output_dim,
        policy_max_quantity,
        state_scale,
        state_normalizer: parse_state_normalizer(state_normalizer)?,
        policy_head: parse_policy_head(policy_head)?,
        demand_config: LostSalesDemandConfig {
            kind: LostSalesDemandKind::Poisson,
            demand_rate: empirical_mean,
            demand_lambda_low: DEFAULT_MMPP2_LAMBDA_LOW,
            demand_lambda_high: DEFAULT_MMPP2_LAMBDA_HIGH,
            demand_p00: DEFAULT_MMPP2_POSITIVE_P00,
            demand_p11: DEFAULT_MMPP2_POSITIVE_P11,
        },
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
    policy_max_quantity,
    activation,
    demand_rate,
    seeds,
    demand_dist_name="Poisson",
    demand_lambda_low=DEFAULT_MMPP2_LAMBDA_LOW,
    demand_lambda_high=DEFAULT_MMPP2_LAMBDA_HIGH,
    demand_p00=DEFAULT_MMPP2_POSITIVE_P00,
    demand_p11=DEFAULT_MMPP2_POSITIVE_P11,
    policy_head="categorical_quantity",
    lead_time=4,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    horizon=2000,
    warm_up_periods_ratio=0.2,
    state_normalizer="identity",
    state_scale=None
))]
fn lost_sales_nn_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    policy_max_quantity: Option<usize>,
    activation: &str,
    demand_rate: f64,
    seeds: Vec<u64>,
    demand_dist_name: &str,
    demand_lambda_low: f64,
    demand_lambda_high: f64,
    demand_p00: f64,
    demand_p11: f64,
    policy_head: &str,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    warm_up_periods_ratio: f64,
    state_normalizer: &str,
    state_scale: Option<f64>,
) -> PyResult<Vec<f64>> {
    let config = LostSalesNeuralRolloutConfig {
        input_dim,
        hidden_dims,
        output_dim,
        policy_max_quantity,
        state_scale,
        state_normalizer: parse_state_normalizer(state_normalizer)?,
        policy_head: parse_policy_head(policy_head)?,
        demand_config: build_lost_sales_demand_config(
            demand_dist_name,
            demand_rate,
            demand_lambda_low,
            demand_lambda_high,
            demand_p00,
            demand_p11,
        )?,
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

fn parse_lost_sales_heuristic_kind(name: &str) -> PyResult<LostSalesHeuristicPolicyKind> {
    match name {
        "myopic1" => Ok(LostSalesHeuristicPolicyKind::Myopic1),
        "myopic2" => Ok(LostSalesHeuristicPolicyKind::Myopic2),
        "svbs" => Ok(LostSalesHeuristicPolicyKind::StandardVectorBaseStock),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported lost-sales heuristic '{other}', expected one of: myopic1, myopic2, svbs"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_lost_sales_heuristic_config(
    demand_kind: &str,
    demand_rate: f64,
    demand_lambda_low: f64,
    demand_lambda_high: f64,
    demand_p00: f64,
    demand_p11: f64,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    order_search_upper_bound: usize,
    heuristic_discount_factor: f64,
) -> PyResult<LostSalesHeuristicVerificationConfig> {
    Ok(LostSalesHeuristicVerificationConfig {
        reference_name: "python_binding",
        horizon,
        seed,
        warm_up_periods_ratio,
        order_search_upper_bound,
        lead_time,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        heuristic_discount_factor,
        demand_config: build_lost_sales_demand_config(
            demand_kind,
            demand_rate,
            demand_lambda_low,
            demand_lambda_high,
            demand_p00,
            demand_p11,
        )?,
    })
}

/// Evaluate a single lost-sales heuristic (Myopic-1 / Myopic-2 / SVBS) and return
/// its warm-up-adjusted mean per-period cost. The order quantities are computed
/// from a closed-form demand law (the stationary marginal for MMPP2), while the
/// rollout cost is measured on the true demand process.
#[pyfunction]
#[pyo3(signature = (
    heuristic,
    demand_kind,
    demand_rate,
    demand_lambda_low,
    demand_lambda_high,
    demand_p00,
    demand_p11,
    lead_time,
    holding_cost,
    shortage_cost,
    procurement_cost,
    fixed_order_cost,
    horizon,
    seed,
    warm_up_periods_ratio,
    order_search_upper_bound,
    heuristic_discount_factor,
))]
#[allow(clippy::too_many_arguments)]
fn lost_sales_heuristic_mean_cost(
    heuristic: &str,
    demand_kind: &str,
    demand_rate: f64,
    demand_lambda_low: f64,
    demand_lambda_high: f64,
    demand_p00: f64,
    demand_p11: f64,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    order_search_upper_bound: usize,
    heuristic_discount_factor: f64,
) -> PyResult<f64> {
    let kind = parse_lost_sales_heuristic_kind(heuristic)?;
    let config = build_lost_sales_heuristic_config(
        demand_kind,
        demand_rate,
        demand_lambda_low,
        demand_lambda_high,
        demand_p00,
        demand_p11,
        lead_time,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon,
        seed,
        warm_up_periods_ratio,
        order_search_upper_bound,
        heuristic_discount_factor,
    )?;
    let measurement = evaluate_heuristic_policy(config, kind)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(measurement.mean_cost)
}

/// Run all three lost-sales heuristics on one instance in a single call and
/// return `{"myopic1": f, "myopic2": f, "svbs": f}` of warm-up-adjusted mean
/// costs. Convenience wrapper used by the config-fill step.
#[pyfunction]
#[pyo3(signature = (
    demand_kind,
    demand_rate,
    demand_lambda_low,
    demand_lambda_high,
    demand_p00,
    demand_p11,
    lead_time,
    holding_cost,
    shortage_cost,
    procurement_cost,
    fixed_order_cost,
    horizon,
    seed,
    warm_up_periods_ratio,
    order_search_upper_bound,
    heuristic_discount_factor,
))]
#[allow(clippy::too_many_arguments)]
fn lost_sales_heuristics_all(
    py: Python<'_>,
    demand_kind: &str,
    demand_rate: f64,
    demand_lambda_low: f64,
    demand_lambda_high: f64,
    demand_p00: f64,
    demand_p11: f64,
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    order_search_upper_bound: usize,
    heuristic_discount_factor: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    use pyo3::types::PyDict;

    let config = build_lost_sales_heuristic_config(
        demand_kind,
        demand_rate,
        demand_lambda_low,
        demand_lambda_high,
        demand_p00,
        demand_p11,
        lead_time,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        horizon,
        seed,
        warm_up_periods_ratio,
        order_search_upper_bound,
        heuristic_discount_factor,
    )?;

    let result = PyDict::new_bound(py);
    for kind in LostSalesHeuristicPolicyKind::all() {
        let measurement = evaluate_heuristic_policy(config, kind)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        result.set_item(kind.policy_name(), measurement.mean_cost)?;
    }
    Ok(result.into())
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
    m.add_function(wrap_pyfunction!(lost_sales_heuristic_mean_cost, m)?)?;
    m.add_function(wrap_pyfunction!(lost_sales_heuristics_all, m)?)?;
    Ok(())
}
