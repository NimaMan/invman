use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{build_action_spec, parse_leaf_type, parse_split_type};
use crate::problems::ameliorating_inventory::lp_dataset_loader::{
    load_port_wine, load_spirits_0001, LoadedLpDataset,
};
use crate::problems::ameliorating_inventory::perfect_information_lp::solve_upper_bound;
use crate::problems::ameliorating_inventory::average_profit_blending_env::AverageProfitBlendingConfig;
use crate::problems::ameliorating_inventory::average_profit_rollout::{
    population_rollout as average_profit_population_rollout, rollout as average_profit_rollout,
    AverageProfitRolloutConfig,
};
use crate::problems::ameliorating_inventory::demand::{
    parse_demand_distribution_kind, DemandModel,
};
use crate::problems::ameliorating_inventory::heuristics::{
    newsvendor_purchase_order_quantity, policy_rollout_from_paths, simulate_policy,
    two_dimensional_order_up_to_order_quantity,
};
use crate::problems::ameliorating_inventory::rollout::{
    build_initial_state, population_rollout, rollout, rollout_from_paths,
    AmelioratingInventoryRolloutConfig,
};

fn build_demand_models(
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
) -> PyResult<Vec<DemandModel>> {
    if demand_kinds.len() != demand_means.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "demand_kinds and demand_means must have the same length",
        ));
    }
    demand_kinds
        .iter()
        .zip(demand_means.iter())
        .map(|(kind, mean)| {
            Ok(DemandModel {
                kind: parse_demand_distribution_kind(kind)?,
                param1: *mean,
            })
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    inventory_by_age,
    periods,
    demand_kinds,
    demand_means,
    target_ages,
    product_prices,
    age_retention,
    purchase_cost_per_unit,
    holding_cost_per_unit,
    decay_salvage_values,
    seed=1234,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn ameliorating_inventory_soft_tree_rollout(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_by_age: Vec<usize>,
    periods: usize,
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
    target_ages: Vec<usize>,
    product_prices: Vec<f64>,
    age_retention: Vec<f64>,
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: Vec<f64>,
    seed: u64,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(&inventory_by_age)?;
    let config = AmelioratingInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_models: build_demand_models(demand_kinds, demand_means)?,
        target_ages,
        product_prices,
        age_retention,
        purchase_cost_per_unit,
        holding_cost_per_unit,
        decay_salvage_values,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    rollout(&flat_params, &config, &initial_state, seed)
}

#[pyfunction]
#[pyo3(signature = (
    params_batch,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    inventory_by_age,
    periods,
    demand_kinds,
    demand_means,
    target_ages,
    product_prices,
    age_retention,
    purchase_cost_per_unit,
    holding_cost_per_unit,
    decay_salvage_values,
    seeds,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn ameliorating_inventory_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_by_age: Vec<usize>,
    periods: usize,
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
    target_ages: Vec<usize>,
    product_prices: Vec<f64>,
    age_retention: Vec<f64>,
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: Vec<f64>,
    seeds: Vec<u64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<f64>> {
    let initial_state = build_initial_state(&inventory_by_age)?;
    let config = AmelioratingInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods,
        demand_models: build_demand_models(demand_kinds, demand_means)?,
        target_ages,
        product_prices,
        age_retention,
        purchase_cost_per_unit,
        holding_cost_per_unit,
        decay_salvage_values,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    population_rollout(&params_batch, &config, &initial_state, &seeds)
}

#[pyfunction]
#[pyo3(signature = (
    flat_params,
    input_dim,
    depth,
    min_values,
    max_values,
    action_mode,
    inventory_by_age,
    realized_demands,
    demand_kinds,
    demand_means,
    target_ages,
    product_prices,
    age_retention,
    purchase_cost_per_unit,
    holding_cost_per_unit,
    decay_salvage_values,
    discount_factor=0.99,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant",
    allowed_values=None
))]
fn ameliorating_inventory_soft_tree_rollout_from_paths(
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    action_mode: &str,
    inventory_by_age: Vec<usize>,
    realized_demands: Vec<Vec<usize>>,
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
    target_ages: Vec<usize>,
    product_prices: Vec<f64>,
    age_retention: Vec<f64>,
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: Vec<f64>,
    discount_factor: f64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<f64> {
    let initial_state = build_initial_state(&inventory_by_age)?;
    let config = AmelioratingInventoryRolloutConfig {
        input_dim,
        depth,
        action_spec: build_action_spec(action_mode, min_values, max_values, allowed_values)?,
        periods: realized_demands.len(),
        demand_models: build_demand_models(demand_kinds, demand_means)?,
        target_ages,
        product_prices,
        age_retention,
        purchase_cost_per_unit,
        holding_cost_per_unit,
        decay_salvage_values,
        discount_factor,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
    };
    rollout_from_paths(&flat_params, &config, &initial_state, &realized_demands)
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    inventory_by_age,
    realized_demands,
    target_ages,
    product_prices,
    age_retention,
    purchase_cost_per_unit,
    holding_cost_per_unit,
    decay_salvage_values,
    discount_factor=0.99
))]
fn ameliorating_inventory_policy_rollout_from_paths(
    policy_name: &str,
    params: Vec<f64>,
    inventory_by_age: Vec<usize>,
    realized_demands: Vec<Vec<usize>>,
    target_ages: Vec<usize>,
    product_prices: Vec<f64>,
    age_retention: Vec<f64>,
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: Vec<f64>,
    discount_factor: f64,
) -> PyResult<f64> {
    let initial_state = build_initial_state(&inventory_by_age)?;
    policy_rollout_from_paths(
        policy_name,
        &params,
        &initial_state,
        &realized_demands,
        &target_ages,
        &product_prices,
        &age_retention,
        purchase_cost_per_unit,
        holding_cost_per_unit,
        &decay_salvage_values,
        discount_factor,
    )
}

#[pyfunction]
#[pyo3(signature = (
    policy_name,
    params,
    inventory_by_age,
    periods,
    demand_kinds,
    demand_means,
    target_ages,
    product_prices,
    age_retention,
    purchase_cost_per_unit,
    holding_cost_per_unit,
    decay_salvage_values,
    replications=1000,
    seed=1234,
    discount_factor=0.99
))]
fn ameliorating_inventory_simulate_policy(
    policy_name: &str,
    params: Vec<f64>,
    inventory_by_age: Vec<usize>,
    periods: usize,
    demand_kinds: Vec<String>,
    demand_means: Vec<f64>,
    target_ages: Vec<usize>,
    product_prices: Vec<f64>,
    age_retention: Vec<f64>,
    purchase_cost_per_unit: f64,
    holding_cost_per_unit: f64,
    decay_salvage_values: Vec<f64>,
    replications: usize,
    seed: u64,
    discount_factor: f64,
) -> PyResult<(f64, f64)> {
    let initial_state = build_initial_state(&inventory_by_age)?;
    let summary = simulate_policy(
        policy_name,
        &params,
        &initial_state,
        periods,
        replications,
        seed,
        &build_demand_models(demand_kinds, demand_means)?,
        &target_ages,
        &product_prices,
        &age_retention,
        purchase_cost_per_unit,
        holding_cost_per_unit,
        &decay_salvage_values,
        discount_factor,
    )?;
    Ok((summary.mean_cost, summary.cost_std))
}

#[pyfunction]
#[pyo3(signature = (inventory_by_age, total_target_inventory))]
fn ameliorating_inventory_newsvendor_purchase_order(
    inventory_by_age: Vec<usize>,
    total_target_inventory: usize,
) -> PyResult<usize> {
    let state = build_initial_state(&inventory_by_age)?;
    newsvendor_purchase_order_quantity(&state, total_target_inventory)
}

#[pyfunction]
#[pyo3(signature = (inventory_by_age, total_target_inventory, young_target_inventory, young_age_cutoff))]
fn ameliorating_inventory_two_dimensional_order_up_to_order(
    inventory_by_age: Vec<usize>,
    total_target_inventory: usize,
    young_target_inventory: usize,
    young_age_cutoff: usize,
) -> PyResult<usize> {
    let state = build_initial_state(&inventory_by_age)?;
    two_dimensional_order_up_to_order_quantity(
        &state,
        total_target_inventory,
        young_target_inventory,
        young_age_cutoff,
    )
}

// ============================================================================
// FAITHFUL average-profit env (Pahr & Grunow 2025) soft-tree rollout bindings.
//
// These target `average_profit_blending_env.rs` (long-run AVERAGE PROFIT), NOT the
// reduced discounted-cost `env.rs` above. The controllable action is the scalar
// purchase volume; the per-period reward is the env's expected profit. The Python
// caller passes the full env config (LP-dataset fields plus the demand/sales/price
// processes that the LP bound does not use) so the rollout dynamics exactly match
// `step_state`.
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn build_average_profit_config(
    num_ages: usize,
    num_products: usize,
    target_ages: Vec<usize>,
    max_inventory: f64,
    evaporation: f64,
    decay_mean: Vec<f64>,
    decay_cov: Vec<f64>,
    holding_costs: f64,
    outdating_costs: f64,
    decay_salvage: Vec<f64>,
    allow_blending: bool,
    blending_range: Option<usize>,
    price_mean: f64,
    price_std: f64,
    price_truncation: f64,
    demand_means: Vec<f64>,
    demand_covs: Vec<f64>,
    sales_means: Vec<f64>,
    sales_covs: Vec<f64>,
    correlation_demand_salesprice: Vec<f64>,
    production_step_size: f64,
    sales_bound: Vec<f64>,
    expected_revenue: Vec<Vec<f64>>,
    initial_inventory: Vec<f64>,
    depth: usize,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    periods: usize,
    warm_up: usize,
) -> PyResult<AverageProfitRolloutConfig> {
    let env = AverageProfitBlendingConfig {
        num_ages,
        num_products,
        target_ages,
        max_inventory,
        evaporation,
        decay_mean,
        decay_cov,
        holding_costs,
        outdating_costs,
        decay_salvage,
        allow_blending,
        blending_range,
        price_mean,
        price_std,
        price_truncation,
        demand_means,
        demand_covs,
        sales_means,
        sales_covs,
        correlation_demand_salesprice,
        production_step_size,
        sales_bound,
        expected_revenue,
    };
    Ok(AverageProfitRolloutConfig {
        env,
        initial_inventory,
        depth,
        temperature,
        split_type: parse_split_type(split_type)?,
        leaf_type: parse_leaf_type(leaf_type)?,
        periods,
        warm_up,
    })
}

/// Single faithful average-profit rollout: mean per-period profit after warm-up.
#[pyfunction]
#[pyo3(signature = (
    flat_params,
    num_ages,
    num_products,
    target_ages,
    max_inventory,
    evaporation,
    decay_mean,
    decay_cov,
    holding_costs,
    outdating_costs,
    decay_salvage,
    allow_blending,
    blending_range,
    price_mean,
    price_std,
    price_truncation,
    demand_means,
    demand_covs,
    sales_means,
    sales_covs,
    correlation_demand_salesprice,
    production_step_size,
    sales_bound,
    expected_revenue,
    initial_inventory,
    depth,
    periods,
    warm_up,
    seed=1234,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant"
))]
#[allow(clippy::too_many_arguments)]
fn ameliorating_inventory_average_profit_soft_tree_rollout(
    flat_params: Vec<f32>,
    num_ages: usize,
    num_products: usize,
    target_ages: Vec<usize>,
    max_inventory: f64,
    evaporation: f64,
    decay_mean: Vec<f64>,
    decay_cov: Vec<f64>,
    holding_costs: f64,
    outdating_costs: f64,
    decay_salvage: Vec<f64>,
    allow_blending: bool,
    blending_range: Option<usize>,
    price_mean: f64,
    price_std: f64,
    price_truncation: f64,
    demand_means: Vec<f64>,
    demand_covs: Vec<f64>,
    sales_means: Vec<f64>,
    sales_covs: Vec<f64>,
    correlation_demand_salesprice: Vec<f64>,
    production_step_size: f64,
    sales_bound: Vec<f64>,
    expected_revenue: Vec<Vec<f64>>,
    initial_inventory: Vec<f64>,
    depth: usize,
    periods: usize,
    warm_up: usize,
    seed: u64,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
) -> PyResult<f64> {
    let config = build_average_profit_config(
        num_ages, num_products, target_ages, max_inventory, evaporation, decay_mean, decay_cov,
        holding_costs, outdating_costs, decay_salvage, allow_blending, blending_range, price_mean,
        price_std, price_truncation, demand_means, demand_covs, sales_means, sales_covs,
        correlation_demand_salesprice, production_step_size, sales_bound, expected_revenue,
        initial_inventory, depth, temperature, split_type, leaf_type, periods, warm_up,
    )?;
    average_profit_rollout(&flat_params, &config, seed)
}

/// Paired population rollout for the faithful average-profit env. This is the
/// CMA-ES scoring binding (mean per-period profit per (params, seed) pair).
#[pyfunction]
#[pyo3(signature = (
    params_batch,
    num_ages,
    num_products,
    target_ages,
    max_inventory,
    evaporation,
    decay_mean,
    decay_cov,
    holding_costs,
    outdating_costs,
    decay_salvage,
    allow_blending,
    blending_range,
    price_mean,
    price_std,
    price_truncation,
    demand_means,
    demand_covs,
    sales_means,
    sales_covs,
    correlation_demand_salesprice,
    production_step_size,
    sales_bound,
    expected_revenue,
    initial_inventory,
    depth,
    periods,
    warm_up,
    seeds,
    temperature=0.25,
    split_type="oblique",
    leaf_type="constant"
))]
#[allow(clippy::too_many_arguments)]
fn ameliorating_inventory_average_profit_soft_tree_population_rollout(
    params_batch: Vec<Vec<f32>>,
    num_ages: usize,
    num_products: usize,
    target_ages: Vec<usize>,
    max_inventory: f64,
    evaporation: f64,
    decay_mean: Vec<f64>,
    decay_cov: Vec<f64>,
    holding_costs: f64,
    outdating_costs: f64,
    decay_salvage: Vec<f64>,
    allow_blending: bool,
    blending_range: Option<usize>,
    price_mean: f64,
    price_std: f64,
    price_truncation: f64,
    demand_means: Vec<f64>,
    demand_covs: Vec<f64>,
    sales_means: Vec<f64>,
    sales_covs: Vec<f64>,
    correlation_demand_salesprice: Vec<f64>,
    production_step_size: f64,
    sales_bound: Vec<f64>,
    expected_revenue: Vec<Vec<f64>>,
    initial_inventory: Vec<f64>,
    depth: usize,
    periods: usize,
    warm_up: usize,
    seeds: Vec<u64>,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
) -> PyResult<Vec<f64>> {
    let config = build_average_profit_config(
        num_ages, num_products, target_ages, max_inventory, evaporation, decay_mean, decay_cov,
        holding_costs, outdating_costs, decay_salvage, allow_blending, blending_range, price_mean,
        price_std, price_truncation, demand_means, demand_covs, sales_means, sales_covs,
        correlation_demand_salesprice, production_step_size, sales_bound, expected_revenue,
        initial_inventory, depth, temperature, split_type, leaf_type, periods, warm_up,
    )?;
    average_profit_population_rollout(&params_batch, &config, &seeds)
}

/// Re-run the perfect-information LP upper bound for one reference instance and return its value(s).
///
/// Mirrors the `*_summary` exact-solver bindings (e.g. `random_yield_inventory_exact_dp_summary`):
/// it actually solves the model in-crate so the audit can RE-RUN the bound rather than only carry a
/// snapshot. `reference_name` accepts the catalogued reference-instance names
/// (`pahr_grunow2025_spirits_0001`, `pahr_grunow2025_port_wine`) and the short dataset aliases
/// (`spirits_0001`, `port_wine`). Returns a dict with the re-solved bound plus the dataset-carried
/// published anchor (and the gap) for comparison.
#[pyfunction]
#[pyo3(signature = (reference_name = "pahr_grunow2025_spirits_0001"))]
fn ameliorating_inventory_perfect_info_lp_bound_summary(
    py: Python<'_>,
    reference_name: &str,
) -> PyResult<PyObject> {
    let LoadedLpDataset { inputs, anchor } = match reference_name {
        "pahr_grunow2025_spirits_0001" | "spirits_0001" => load_spirits_0001(),
        "pahr_grunow2025_port_wine" | "port_wine" => load_port_wine(),
        other => {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "unknown ameliorating reference instance '{other}'; expected one of \
                 'pahr_grunow2025_spirits_0001'/'spirits_0001' or \
                 'pahr_grunow2025_port_wine'/'port_wine'"
            )))
        }
    };
    let solution = solve_upper_bound(&inputs);

    let dict = PyDict::new_bound(py);
    dict.set_item("reference_name", reference_name)?;
    dict.set_item("instance", inputs.instance.clone())?;
    // Re-solved (re-run) perfect-information LP upper bound.
    dict.set_item("upper_bound_max_reward", solution.max_reward)?;
    dict.set_item("upper_bound_purchasing", solution.purchasing)?;
    dict.set_item("upper_bound_production", solution.production.clone())?;
    dict.set_item(
        "upper_bound_inventory_position",
        solution.inventory_position.clone(),
    )?;
    // Dataset-carried published companion anchor + reproduction gap.
    dict.set_item("published_max_reward", anchor.max_reward)?;
    dict.set_item("published_purchasing", anchor.purchasing)?;
    dict.set_item(
        "max_reward_gap_to_published",
        (solution.max_reward - anchor.max_reward).abs(),
    )?;
    Ok(dict.into_any().unbind().into())
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_perfect_info_lp_bound_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_soft_tree_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_soft_tree_population_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_soft_tree_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_policy_rollout_from_paths,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ameliorating_inventory_simulate_policy, m)?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_newsvendor_purchase_order,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_two_dimensional_order_up_to_order,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_average_profit_soft_tree_rollout,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ameliorating_inventory_average_profit_soft_tree_population_rollout,
        m
    )?)?;
    Ok(())
}
