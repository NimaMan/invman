mod core;
mod problems;

use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};

use crate::core::policies::dense::{parse_activation, parse_policy_head};
use crate::core::policies::soft_tree::{
    build_action_spec, parse_leaf_type, parse_split_type, soft_tree_leaf_probabilities,
    validate_soft_tree_shapes,
};
use crate::problems::dual_sourcing::heuristics::{
    search_capped_dual_index_from_demands, search_dual_index_from_demands,
    search_single_index_from_demands, search_tailored_base_surge_from_demands,
};
use crate::problems::dual_sourcing::policies::parse_action_adapter;
use crate::problems::dual_sourcing::rollout::{
    population_rollout as dual_sourcing_population_rollout, rollout as dual_sourcing_rollout,
    rollout_from_demands as dual_sourcing_rollout_from_demands, DualSourcingRolloutConfig,
};
use crate::problems::lost_sales::env::{epoch_cost, initialize_state, LostSalesState};
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
use crate::problems::lost_sales_fixed_order_cost::heuristics::{
    fixed_policy_rollout_from_demands, search_modified_s_s_q_from_demands,
    search_s_nq_from_demands, search_s_s_from_demands,
};
use crate::problems::multi_echelon::heuristics::search_constant_base_stock_from_demands;
use crate::problems::multi_echelon::rollout::{
    population_rollout as multi_echelon_population_rollout, rollout as multi_echelon_rollout,
    MultiEchelonRolloutConfig,
};

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
#[pyo3(signature = (state, split_weights, split_bias, leaf_logits, depth, max_order_size, temperature=0.25, split_type="oblique"))]
fn soft_tree_action(
    state: Vec<f32>,
    split_weights: Vec<f32>,
    split_bias: Vec<f32>,
    leaf_logits: Vec<f32>,
    depth: usize,
    max_order_size: usize,
    temperature: f32,
    split_type: &str,
) -> PyResult<usize> {
    validate_soft_tree_shapes(
        state.len(),
        split_weights.len(),
        split_bias.len(),
        leaf_logits.len(),
        depth,
    )?;

    let leaf_probs = soft_tree_leaf_probabilities(
        &state,
        &split_weights,
        &split_bias,
        depth,
        temperature,
        parse_split_type(split_type)?,
    );
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
    let demand_dist = Poisson::new(demand_rate).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid demand_rate: {err}"))
    })?;
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
    policy_name,
    params,
    current_inventory,
    lead_time_orders,
    demands,
    max_order_size,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    warm_up_periods_ratio=0.2
))]
fn lost_sales_fixed_policy_rollout_from_demands(
    policy_name: &str,
    params: Vec<usize>,
    current_inventory: i64,
    lead_time_orders: Vec<usize>,
    demands: Vec<usize>,
    max_order_size: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    warm_up_periods_ratio: f64,
) -> PyResult<f64> {
    fixed_policy_rollout_from_demands(
        policy_name,
        &params,
        current_inventory,
        &lead_time_orders,
        &demands,
        max_order_size,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        warm_up_periods_ratio,
    )
}

#[pyfunction]
#[pyo3(signature = (
    current_inventory,
    lead_time_orders,
    demands,
    max_order_size,
    position_upper_bound,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    warm_up_periods_ratio=0.2,
    top_k=12
))]
fn lost_sales_fixed_s_s_search_from_demands(
    current_inventory: i64,
    lead_time_orders: Vec<usize>,
    demands: Vec<usize>,
    max_order_size: usize,
    position_upper_bound: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    warm_up_periods_ratio: f64,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    search_s_s_from_demands(
        current_inventory,
        &lead_time_orders,
        &demands,
        max_order_size,
        position_upper_bound,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        warm_up_periods_ratio,
        top_k,
    )
}

#[pyfunction]
#[pyo3(signature = (
    current_inventory,
    lead_time_orders,
    demands,
    max_order_size,
    position_upper_bound,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    warm_up_periods_ratio=0.2,
    top_k=12
))]
fn lost_sales_fixed_s_nq_search_from_demands(
    current_inventory: i64,
    lead_time_orders: Vec<usize>,
    demands: Vec<usize>,
    max_order_size: usize,
    position_upper_bound: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    warm_up_periods_ratio: f64,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    search_s_nq_from_demands(
        current_inventory,
        &lead_time_orders,
        &demands,
        max_order_size,
        position_upper_bound,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        warm_up_periods_ratio,
        top_k,
    )
}

#[pyfunction]
#[pyo3(signature = (
    current_inventory,
    lead_time_orders,
    demands,
    max_order_size,
    position_upper_bound,
    holding_cost=1.0,
    shortage_cost=4.0,
    procurement_cost=0.0,
    fixed_order_cost=0.0,
    warm_up_periods_ratio=0.2,
    top_k=12
))]
fn lost_sales_fixed_modified_s_s_q_search_from_demands(
    current_inventory: i64,
    lead_time_orders: Vec<usize>,
    demands: Vec<usize>,
    max_order_size: usize,
    position_upper_bound: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    warm_up_periods_ratio: f64,
    top_k: usize,
) -> PyResult<(
    (usize, usize, usize, f64),
    Vec<(usize, usize, usize, f64)>,
    usize,
)> {
    search_modified_s_s_q_from_demands(
        current_inventory,
        &lead_time_orders,
        &demands,
        max_order_size,
        position_upper_bound,
        holding_cost,
        shortage_cost,
        procurement_cost,
        fixed_order_cost,
        warm_up_periods_ratio,
        top_k,
    )
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

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    regular_lead_time,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    regular_max_order_size,
    expedited_max_order_size,
    demand_low,
    demand_high,
    horizon=2000,
    seed=1234,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    action_adapter="identity",
    allowed_values=None
))]
fn dual_sourcing_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    regular_lead_time: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    demand_low: usize,
    demand_high: usize,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    action_adapter: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let config = DualSourcingRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        regular_lead_time,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        regular_max_order_size,
        expedited_max_order_size,
        demand_low,
        demand_high,
        horizon,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        action_adapter: parse_action_adapter(action_adapter)?,
    };
    dual_sourcing_rollout(&flat_params, &config, seed)
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    regular_lead_time,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    regular_max_order_size,
    expedited_max_order_size,
    demand_low,
    demand_high,
    seeds,
    horizon=2000,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    action_adapter="identity",
    allowed_values=None
))]
fn dual_sourcing_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    regular_lead_time: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    demand_low: usize,
    demand_high: usize,
    seeds: Vec<u64>,
    horizon: usize,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    action_adapter: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let config = DualSourcingRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        regular_lead_time,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        regular_max_order_size,
        expedited_max_order_size,
        demand_low,
        demand_high,
        horizon,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        action_adapter: parse_action_adapter(action_adapter)?,
    };
    dual_sourcing_population_rollout(&params_batch, &config, &seeds)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    state,
    demands,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    regular_max_order_size,
    expedited_max_order_size,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    action_adapter="identity",
    allowed_values=None
))]
fn dual_sourcing_soft_tree_rollout_from_demands(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    state: Vec<i64>,
    demands: Vec<usize>,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    action_adapter: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let config = DualSourcingRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        regular_lead_time: state.len(),
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        regular_max_order_size,
        expedited_max_order_size,
        demand_low: 0,
        demand_high: 0,
        horizon: demands.len(),
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        action_adapter: parse_action_adapter(action_adapter)?,
    };
    dual_sourcing_rollout_from_demands(&flat_params, &config, state, &demands)
}

#[pyfunction]
#[pyo3(signature = (
    state,
    demands,
    regular_max_order_size,
    expedited_max_order_size,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    warm_up_periods_ratio=0.2,
    target_upper_bound=20,
    top_k=10
))]
fn dual_sourcing_single_index_search_from_demands(
    state: Vec<i64>,
    demands: Vec<usize>,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    warm_up_periods_ratio: f64,
    target_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    search_single_index_from_demands(
        &state,
        &demands,
        regular_max_order_size,
        expedited_max_order_size,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        warm_up_periods_ratio,
        target_upper_bound,
        top_k,
    )
}

#[pyfunction]
#[pyo3(signature = (
    state,
    demands,
    regular_max_order_size,
    expedited_max_order_size,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    warm_up_periods_ratio=0.2,
    target_upper_bound=20,
    top_k=10
))]
fn dual_sourcing_dual_index_search_from_demands(
    state: Vec<i64>,
    demands: Vec<usize>,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    warm_up_periods_ratio: f64,
    target_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    search_dual_index_from_demands(
        &state,
        &demands,
        regular_max_order_size,
        expedited_max_order_size,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        warm_up_periods_ratio,
        target_upper_bound,
        top_k,
    )
}

#[pyfunction]
#[pyo3(signature = (
    state,
    demands,
    regular_max_order_size,
    expedited_max_order_size,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    warm_up_periods_ratio=0.2,
    target_upper_bound=20,
    top_k=10
))]
fn dual_sourcing_capped_dual_index_search_from_demands(
    state: Vec<i64>,
    demands: Vec<usize>,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    warm_up_periods_ratio: f64,
    target_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, usize, usize, f64), Vec<(usize, usize, usize, f64)>)> {
    search_capped_dual_index_from_demands(
        &state,
        &demands,
        regular_max_order_size,
        expedited_max_order_size,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        warm_up_periods_ratio,
        target_upper_bound,
        top_k,
    )
}

#[pyfunction]
#[pyo3(signature = (
    state,
    demands,
    regular_max_order_size,
    expedited_max_order_size,
    regular_order_cost,
    expedited_order_cost,
    holding_cost,
    shortage_cost,
    warm_up_periods_ratio=0.2,
    target_upper_bound=20,
    top_k=10
))]
fn dual_sourcing_tailored_base_surge_search_from_demands(
    state: Vec<i64>,
    demands: Vec<usize>,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    warm_up_periods_ratio: f64,
    target_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    search_tailored_base_surge_from_demands(
        &state,
        &demands,
        regular_max_order_size,
        expedited_max_order_size,
        regular_order_cost,
        expedited_order_cost,
        holding_cost,
        shortage_cost,
        warm_up_periods_ratio,
        target_upper_bound,
        top_k,
    )
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    warehouse_lead_time,
    retailer_lead_time,
    num_retailers,
    warehouse_holding_cost,
    retailer_holding_cost,
    warehouse_expedited_cost,
    warehouse_lost_sale_cost,
    expedited_service_prob,
    warehouse_capacity,
    warehouse_inventory_cap,
    retailer_inventory_cap,
    demand_mean,
    demand_std,
    horizon=2000,
    seed=1234,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn multi_echelon_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    warehouse_lead_time: usize,
    retailer_lead_time: usize,
    num_retailers: usize,
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    expedited_service_prob: f64,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    demand_mean: f64,
    demand_std: f64,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let warehouse_levels = allowed_values
        .as_ref()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("multi-echelon rollouts require allowed_values")
        })?
        .get(0)
        .cloned()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("missing warehouse allowed_values")
        })?;
    let retailer_levels = allowed_values
        .as_ref()
        .unwrap()
        .get(1)
        .cloned()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("missing retailer allowed_values")
        })?;
    let config = MultiEchelonRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        warehouse_lead_time,
        retailer_lead_time,
        num_retailers,
        warehouse_holding_cost,
        retailer_holding_cost,
        warehouse_expedited_cost,
        warehouse_lost_sale_cost,
        expedited_service_prob,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        demand_mean,
        demand_std,
        horizon,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    multi_echelon_rollout(
        &flat_params,
        &config,
        seed,
        &warehouse_levels,
        &retailer_levels,
    )
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    warehouse_lead_time,
    retailer_lead_time,
    num_retailers,
    warehouse_holding_cost,
    retailer_holding_cost,
    warehouse_expedited_cost,
    warehouse_lost_sale_cost,
    expedited_service_prob,
    warehouse_capacity,
    warehouse_inventory_cap,
    retailer_inventory_cap,
    demand_mean,
    demand_std,
    seeds,
    horizon=2000,
    warm_up_periods_ratio=0.2,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn multi_echelon_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    warehouse_lead_time: usize,
    retailer_lead_time: usize,
    num_retailers: usize,
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    expedited_service_prob: f64,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    demand_mean: f64,
    demand_std: f64,
    seeds: Vec<u64>,
    horizon: usize,
    warm_up_periods_ratio: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let warehouse_levels = allowed_values
        .as_ref()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("multi-echelon rollouts require allowed_values")
        })?
        .get(0)
        .cloned()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("missing warehouse allowed_values")
        })?;
    let retailer_levels = allowed_values
        .as_ref()
        .unwrap()
        .get(1)
        .cloned()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("missing retailer allowed_values")
        })?;
    let config = MultiEchelonRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        warehouse_lead_time,
        retailer_lead_time,
        num_retailers,
        warehouse_holding_cost,
        retailer_holding_cost,
        warehouse_expedited_cost,
        warehouse_lost_sale_cost,
        expedited_service_prob,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        demand_mean,
        demand_std,
        horizon,
        warm_up_periods_ratio,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    multi_echelon_population_rollout(
        &params_batch,
        &config,
        &seeds,
        &warehouse_levels,
        &retailer_levels,
    )
}

#[pyfunction]
#[pyo3(signature = (
    warehouse_inventory,
    warehouse_pipeline,
    retailer_inventory,
    retailer_pipeline,
    demands,
    expedite_uniforms,
    warehouse_levels,
    retailer_levels,
    warehouse_holding_cost,
    retailer_holding_cost,
    warehouse_expedited_cost,
    warehouse_lost_sale_cost,
    expedited_service_prob,
    warehouse_capacity,
    warehouse_inventory_cap,
    retailer_inventory_cap,
    warm_up_periods_ratio=0.2,
    top_k=10
))]
fn multi_echelon_constant_base_stock_search_from_demands(
    warehouse_inventory: i64,
    warehouse_pipeline: Vec<usize>,
    retailer_inventory: Vec<i64>,
    retailer_pipeline: Vec<Vec<usize>>,
    demands: Vec<Vec<usize>>,
    expedite_uniforms: Vec<Vec<Vec<f64>>>,
    warehouse_levels: Vec<usize>,
    retailer_levels: Vec<usize>,
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    expedited_service_prob: f64,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    warm_up_periods_ratio: f64,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    search_constant_base_stock_from_demands(
        warehouse_inventory,
        &warehouse_pipeline,
        &retailer_inventory,
        &retailer_pipeline,
        &demands,
        &expedite_uniforms,
        &warehouse_levels,
        &retailer_levels,
        warehouse_holding_cost,
        retailer_holding_cost,
        warehouse_expedited_cost,
        warehouse_lost_sale_cost,
        expedited_service_prob,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        warm_up_periods_ratio,
        top_k,
    )
}

#[pymodule]
fn invman_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(soft_tree_action, m)?)?;
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
        lost_sales_fixed_policy_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_fixed_s_s_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_fixed_s_nq_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_fixed_modified_s_s_q_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(dual_sourcing_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_soft_tree_rollout_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_single_index_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_dual_index_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_capped_dual_index_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dual_sourcing_tailored_base_surge_search_from_demands,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(multi_echelon_soft_tree_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multi_echelon_constant_base_stock_search_from_demands,
        m
    )?)?;
    Ok(())
}
