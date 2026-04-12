#![allow(dead_code)]

mod base_stock;
mod bsp_low_ew;

pub use base_stock::base_stock_order_quantity;
pub use bsp_low_ew::bsp_low_ew_order_quantity;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Gamma};

use crate::problems::perishable_inventory::env::{step_state, IssuingPolicy, PerishableState};

fn mean_after_warmup(epoch_costs: &[f64], warm_up_periods_ratio: f64) -> PyResult<f64> {
    if epoch_costs.is_empty() {
        return Err(PyValueError::new_err("epoch_costs must be non-empty"));
    }
    if !(0.0..=1.0).contains(&warm_up_periods_ratio) {
        return Err(PyValueError::new_err(
            "warm_up_periods_ratio must be in [0, 1]",
        ));
    }
    let horizon = epoch_costs.len();
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let active_costs = if warm_up_periods < epoch_costs.len() {
        &epoch_costs[warm_up_periods..]
    } else {
        epoch_costs
    };
    Ok(active_costs.iter().sum::<f64>() / active_costs.len() as f64)
}

#[derive(Clone, Debug)]
pub struct DiscountedReturnSummary {
    pub mean_return: f64,
    pub std_return: f64,
    pub min_return: f64,
    pub max_return: f64,
    pub num_seeds: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyTraceSummary {
    pub periods: usize,
    pub total_cost: f64,
    pub mean_period_cost: f64,
    pub total_demand: usize,
    pub total_shortage: usize,
    pub fill_rate: f64,
    pub cycle_service_level: f64,
    pub total_waste: usize,
    pub waste_rate: f64,
    pub mean_holding_inventory: f64,
    pub mean_order_quantity: f64,
    pub positive_order_frequency: f64,
    pub ending_inventory: usize,
    pub ending_pipeline: usize,
}

fn summarize_discounted_returns(returns: &[f64]) -> PyResult<DiscountedReturnSummary> {
    if returns.is_empty() {
        return Err(PyValueError::new_err("returns must be non-empty"));
    }
    let num_seeds = returns.len();
    let mean_return = returns.iter().sum::<f64>() / num_seeds as f64;
    let variance = returns
        .iter()
        .map(|value| {
            let centered = *value - mean_return;
            centered * centered
        })
        .sum::<f64>()
        / num_seeds as f64;
    let min_return = returns.iter().copied().fold(f64::INFINITY, f64::min);
    let max_return = returns.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(DiscountedReturnSummary {
        mean_return,
        std_return: variance.sqrt(),
        min_return,
        max_return,
        num_seeds,
    })
}

fn discounted_return_after_warmup(
    epoch_costs: &[f64],
    warm_up_periods_ratio: f64,
    gamma: f64,
) -> PyResult<f64> {
    if epoch_costs.is_empty() {
        return Err(PyValueError::new_err("epoch_costs must be non-empty"));
    }
    if !(0.0..=1.0).contains(&warm_up_periods_ratio) {
        return Err(PyValueError::new_err(
            "warm_up_periods_ratio must be in [0, 1]",
        ));
    }
    if !(0.0..=1.0).contains(&gamma) {
        return Err(PyValueError::new_err("gamma must be in [0, 1]"));
    }
    let horizon = epoch_costs.len();
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let mut discounted_return = 0.0;
    for (offset, cost) in epoch_costs.iter().skip(warm_up_periods).enumerate() {
        discounted_return += -cost * gamma.powi(offset as i32);
    }
    Ok(discounted_return)
}

fn build_discrete_gamma(demand_mean: f64, demand_cov: f64) -> PyResult<Gamma<f64>> {
    let shape = 1.0 / (demand_cov * demand_cov);
    let scale = if shape > 0.0 {
        demand_mean / shape
    } else {
        0.0
    };
    Gamma::new(shape, scale.max(1e-9))
        .map_err(|err| PyValueError::new_err(format!("invalid gamma demand parameters: {err}")))
}

fn sample_gamma_demand(rng: &mut StdRng, gamma: &Gamma<f64>) -> usize {
    gamma.sample(rng).round().max(0.0) as usize
}

pub fn policy_rollout_from_demands(
    policy_name: &str,
    params: &[usize],
    initial_state: &PerishableState,
    demands: &[usize],
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: IssuingPolicy,
) -> PyResult<f64> {
    let mut state = initial_state.clone();
    let mut epoch_costs = Vec::with_capacity(demands.len());

    for demand in demands.iter().copied() {
        let order_quantity = match policy_name {
            "base_stock" => {
                if params.len() != 1 {
                    return Err(PyValueError::new_err("base_stock expects params [S]"));
                }
                base_stock_order_quantity(&state, params[0], max_order_size)
            }
            "bsp_low_ew" => {
                if params.len() != 3 {
                    return Err(PyValueError::new_err(
                        "bsp_low_ew expects params [S1, S2, b]",
                    ));
                }
                bsp_low_ew_order_quantity(
                    &state,
                    params[0],
                    params[1],
                    params[2],
                    max_order_size,
                    lead_time,
                    demand_mean,
                    issuing_policy,
                )
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown perishable-inventory policy '{policy_name}'"
                )))
            }
        };

        let outcome = step_state(
            &state,
            order_quantity,
            demand,
            holding_cost,
            shortage_cost,
            waste_cost,
            procurement_cost,
            issuing_policy,
        );
        epoch_costs.push(outcome.cost);
        state = outcome.next_state;
    }

    mean_after_warmup(&epoch_costs, warm_up_periods_ratio)
}

#[allow(clippy::too_many_arguments)]
pub fn policy_trace_summary_from_demands(
    policy_name: &str,
    params: &[usize],
    initial_state: &PerishableState,
    demands: &[usize],
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    issuing_policy: IssuingPolicy,
) -> PyResult<PolicyTraceSummary> {
    if demands.is_empty() {
        return Err(PyValueError::new_err("demands must be non-empty"));
    }

    let mut state = initial_state.clone();
    let mut total_cost = 0.0;
    let mut total_demand = 0usize;
    let mut total_shortage = 0usize;
    let mut stockout_periods = 0usize;
    let mut total_waste = 0usize;
    let mut total_holding_inventory = 0usize;
    let mut total_order_quantity = 0usize;
    let mut positive_order_periods = 0usize;

    for demand in demands.iter().copied() {
        let order_quantity = match policy_name {
            "base_stock" => {
                if params.len() != 1 {
                    return Err(PyValueError::new_err("base_stock expects params [S]"));
                }
                base_stock_order_quantity(&state, params[0], max_order_size)
            }
            "bsp_low_ew" => {
                if params.len() != 3 {
                    return Err(PyValueError::new_err(
                        "bsp_low_ew expects params [S1, S2, b]",
                    ));
                }
                bsp_low_ew_order_quantity(
                    &state,
                    params[0],
                    params[1],
                    params[2],
                    max_order_size,
                    lead_time,
                    demand_mean,
                    issuing_policy,
                )
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown perishable-inventory policy '{policy_name}'"
                )))
            }
        };

        let outcome = step_state(
            &state,
            order_quantity,
            demand,
            holding_cost,
            shortage_cost,
            waste_cost,
            procurement_cost,
            issuing_policy,
        );
        total_cost += outcome.cost;
        total_demand += demand;
        total_shortage += outcome.shortage;
        total_waste += outcome.waste;
        total_holding_inventory += outcome.holding_inventory;
        total_order_quantity += order_quantity;
        if order_quantity > 0 {
            positive_order_periods += 1;
        }
        if outcome.shortage > 0 {
            stockout_periods += 1;
        }
        state = outcome.next_state;
    }

    let periods = demands.len();
    let ending_inventory = state.on_hand.iter().copied().sum::<usize>();
    let ending_pipeline = state.pipeline_orders.iter().copied().sum::<usize>();

    Ok(PolicyTraceSummary {
        periods,
        total_cost,
        mean_period_cost: total_cost / periods as f64,
        total_demand,
        total_shortage,
        fill_rate: if total_demand > 0 {
            1.0 - total_shortage as f64 / total_demand as f64
        } else {
            1.0
        },
        cycle_service_level: 1.0 - stockout_periods as f64 / periods as f64,
        total_waste,
        waste_rate: if total_demand > 0 {
            total_waste as f64 / total_demand as f64
        } else {
            0.0
        },
        mean_holding_inventory: total_holding_inventory as f64 / periods as f64,
        mean_order_quantity: total_order_quantity as f64 / periods as f64,
        positive_order_frequency: positive_order_periods as f64 / periods as f64,
        ending_inventory,
        ending_pipeline,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn policy_discounted_return_from_demands(
    policy_name: &str,
    params: &[usize],
    initial_state: &PerishableState,
    demands: &[usize],
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: IssuingPolicy,
) -> PyResult<f64> {
    let mut state = initial_state.clone();
    let mut epoch_costs = Vec::with_capacity(demands.len());

    for demand in demands.iter().copied() {
        let order_quantity = match policy_name {
            "base_stock" => {
                if params.len() != 1 {
                    return Err(PyValueError::new_err("base_stock expects params [S]"));
                }
                base_stock_order_quantity(&state, params[0], max_order_size)
            }
            "bsp_low_ew" => {
                if params.len() != 3 {
                    return Err(PyValueError::new_err(
                        "bsp_low_ew expects params [S1, S2, b]",
                    ));
                }
                bsp_low_ew_order_quantity(
                    &state,
                    params[0],
                    params[1],
                    params[2],
                    max_order_size,
                    lead_time,
                    demand_mean,
                    issuing_policy,
                )
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown perishable-inventory policy '{policy_name}'"
                )))
            }
        };

        let outcome = step_state(
            &state,
            order_quantity,
            demand,
            holding_cost,
            shortage_cost,
            waste_cost,
            procurement_cost,
            issuing_policy,
        );
        epoch_costs.push(outcome.cost);
        state = outcome.next_state;
    }

    discounted_return_after_warmup(&epoch_costs, warm_up_periods_ratio, gamma)
}

#[allow(clippy::too_many_arguments)]
pub fn policy_rollout(
    policy_name: &str,
    params: &[usize],
    initial_state: &PerishableState,
    horizon: usize,
    seed: u64,
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: IssuingPolicy,
) -> PyResult<f64> {
    let gamma = build_discrete_gamma(demand_mean, demand_cov)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut epoch_costs = Vec::with_capacity(horizon);

    for _ in 0..horizon {
        let demand = sample_gamma_demand(&mut rng, &gamma);
        let order_quantity = match policy_name {
            "base_stock" => {
                if params.len() != 1 {
                    return Err(PyValueError::new_err("base_stock expects params [S]"));
                }
                base_stock_order_quantity(&state, params[0], max_order_size)
            }
            "bsp_low_ew" => {
                if params.len() != 3 {
                    return Err(PyValueError::new_err(
                        "bsp_low_ew expects params [S1, S2, b]",
                    ));
                }
                bsp_low_ew_order_quantity(
                    &state,
                    params[0],
                    params[1],
                    params[2],
                    max_order_size,
                    lead_time,
                    demand_mean,
                    issuing_policy,
                )
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown perishable-inventory policy '{policy_name}'"
                )))
            }
        };

        let outcome = step_state(
            &state,
            order_quantity,
            demand,
            holding_cost,
            shortage_cost,
            waste_cost,
            procurement_cost,
            issuing_policy,
        );
        epoch_costs.push(outcome.cost);
        state = outcome.next_state;
    }

    mean_after_warmup(&epoch_costs, warm_up_periods_ratio)
}

#[allow(clippy::too_many_arguments)]
pub fn policy_discounted_return(
    policy_name: &str,
    params: &[usize],
    initial_state: &PerishableState,
    horizon: usize,
    seed: u64,
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: IssuingPolicy,
) -> PyResult<f64> {
    let gamma_dist = build_discrete_gamma(demand_mean, demand_cov)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = initial_state.clone();
    let mut epoch_costs = Vec::with_capacity(horizon);

    for _ in 0..horizon {
        let demand = sample_gamma_demand(&mut rng, &gamma_dist);
        let order_quantity = match policy_name {
            "base_stock" => {
                if params.len() != 1 {
                    return Err(PyValueError::new_err("base_stock expects params [S]"));
                }
                base_stock_order_quantity(&state, params[0], max_order_size)
            }
            "bsp_low_ew" => {
                if params.len() != 3 {
                    return Err(PyValueError::new_err(
                        "bsp_low_ew expects params [S1, S2, b]",
                    ));
                }
                bsp_low_ew_order_quantity(
                    &state,
                    params[0],
                    params[1],
                    params[2],
                    max_order_size,
                    lead_time,
                    demand_mean,
                    issuing_policy,
                )
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown perishable-inventory policy '{policy_name}'"
                )))
            }
        };

        let outcome = step_state(
            &state,
            order_quantity,
            demand,
            holding_cost,
            shortage_cost,
            waste_cost,
            procurement_cost,
            issuing_policy,
        );
        epoch_costs.push(outcome.cost);
        state = outcome.next_state;
    }

    discounted_return_after_warmup(&epoch_costs, warm_up_periods_ratio, gamma)
}

#[allow(clippy::too_many_arguments)]
pub fn policy_discounted_return_summary(
    policy_name: &str,
    params: &[usize],
    initial_state: &PerishableState,
    horizon: usize,
    seeds: &[u64],
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: IssuingPolicy,
) -> PyResult<DiscountedReturnSummary> {
    let mut returns = Vec::with_capacity(seeds.len());
    for seed in seeds.iter().copied() {
        returns.push(policy_discounted_return(
            policy_name,
            params,
            initial_state,
            horizon,
            seed,
            lead_time,
            max_order_size,
            demand_mean,
            demand_cov,
            holding_cost,
            shortage_cost,
            waste_cost,
            procurement_cost,
            warm_up_periods_ratio,
            gamma,
            issuing_policy,
        )?);
    }
    summarize_discounted_returns(&returns)
}

pub fn search_base_stock_from_demands(
    initial_state: &PerishableState,
    demands: &[usize],
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: IssuingPolicy,
    position_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, f64), Vec<(usize, f64)>)> {
    if top_k == 0 {
        return Err(PyValueError::new_err("top_k must be positive"));
    }
    let mut results = Vec::new();
    for level in 0..=position_upper_bound {
        let mean_cost = policy_rollout_from_demands(
            "base_stock",
            &[level],
            initial_state,
            demands,
            lead_time,
            max_order_size,
            demand_mean,
            holding_cost,
            shortage_cost,
            waste_cost,
            procurement_cost,
            warm_up_periods_ratio,
            issuing_policy,
        )?;
        results.push((level, mean_cost));
    }
    results.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    Ok((results[0], results.into_iter().take(top_k).collect()))
}

#[allow(clippy::too_many_arguments)]
pub fn search_base_stock(
    initial_state: &PerishableState,
    horizon: usize,
    seed: u64,
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: IssuingPolicy,
    position_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, f64), Vec<(usize, f64)>)> {
    if top_k == 0 {
        return Err(PyValueError::new_err("top_k must be positive"));
    }
    let mut results = Vec::new();
    for level in 0..=position_upper_bound {
        let mean_cost = policy_rollout(
            "base_stock",
            &[level],
            initial_state,
            horizon,
            seed,
            lead_time,
            max_order_size,
            demand_mean,
            demand_cov,
            holding_cost,
            shortage_cost,
            waste_cost,
            procurement_cost,
            warm_up_periods_ratio,
            issuing_policy,
        )?;
        results.push((level, mean_cost));
    }
    results.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    Ok((results[0], results.into_iter().take(top_k).collect()))
}

#[allow(clippy::too_many_arguments)]
pub fn search_base_stock_discounted_return(
    initial_state: &PerishableState,
    horizon: usize,
    seed: u64,
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: IssuingPolicy,
    position_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, f64), Vec<(usize, f64)>)> {
    if top_k == 0 {
        return Err(PyValueError::new_err("top_k must be positive"));
    }
    let mut results = Vec::new();
    for level in 0..=position_upper_bound {
        let discounted_return = policy_discounted_return(
            "base_stock",
            &[level],
            initial_state,
            horizon,
            seed,
            lead_time,
            max_order_size,
            demand_mean,
            demand_cov,
            holding_cost,
            shortage_cost,
            waste_cost,
            procurement_cost,
            warm_up_periods_ratio,
            gamma,
            issuing_policy,
        )?;
        results.push((level, discounted_return));
    }
    results.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    Ok((results[0], results.into_iter().take(top_k).collect()))
}

#[allow(clippy::too_many_arguments)]
pub fn search_base_stock_discounted_return_summary(
    initial_state: &PerishableState,
    horizon: usize,
    seeds: &[u64],
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: IssuingPolicy,
    position_upper_bound: usize,
    top_k: usize,
) -> PyResult<(
    (usize, DiscountedReturnSummary),
    Vec<(usize, DiscountedReturnSummary)>,
)> {
    if top_k == 0 {
        return Err(PyValueError::new_err("top_k must be positive"));
    }
    let mut results = Vec::new();
    for level in 0..=position_upper_bound {
        let summary = policy_discounted_return_summary(
            "base_stock",
            &[level],
            initial_state,
            horizon,
            seeds,
            lead_time,
            max_order_size,
            demand_mean,
            demand_cov,
            holding_cost,
            shortage_cost,
            waste_cost,
            procurement_cost,
            warm_up_periods_ratio,
            gamma,
            issuing_policy,
        )?;
        results.push((level, summary));
    }
    results.sort_by(|left, right| {
        right
            .1
            .mean_return
            .total_cmp(&left.1.mean_return)
            .then_with(|| left.0.cmp(&right.0))
    });
    Ok((
        results[0].clone(),
        results.into_iter().take(top_k).collect(),
    ))
}

pub fn search_bsp_low_ew_from_demands(
    initial_state: &PerishableState,
    demands: &[usize],
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: IssuingPolicy,
    position_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, usize, usize, f64), Vec<(usize, usize, usize, f64)>)> {
    if top_k == 0 {
        return Err(PyValueError::new_err("top_k must be positive"));
    }
    let mut results = Vec::new();
    for low_inventory_level in 0..=position_upper_bound {
        for high_inventory_level in 0..=low_inventory_level {
            for threshold in 0..=position_upper_bound {
                let mean_cost = policy_rollout_from_demands(
                    "bsp_low_ew",
                    &[low_inventory_level, high_inventory_level, threshold],
                    initial_state,
                    demands,
                    lead_time,
                    max_order_size,
                    demand_mean,
                    holding_cost,
                    shortage_cost,
                    waste_cost,
                    procurement_cost,
                    warm_up_periods_ratio,
                    issuing_policy,
                )?;
                results.push((
                    low_inventory_level,
                    high_inventory_level,
                    threshold,
                    mean_cost,
                ));
            }
        }
    }
    results.sort_by(|left, right| {
        left.3
            .total_cmp(&right.3)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    Ok((results[0], results.into_iter().take(top_k).collect()))
}

#[allow(clippy::too_many_arguments)]
pub fn search_bsp_low_ew(
    initial_state: &PerishableState,
    horizon: usize,
    seed: u64,
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    issuing_policy: IssuingPolicy,
    position_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, usize, usize, f64), Vec<(usize, usize, usize, f64)>)> {
    if top_k == 0 {
        return Err(PyValueError::new_err("top_k must be positive"));
    }
    let mut results = Vec::new();
    for low_inventory_level in 0..=position_upper_bound {
        for high_inventory_level in 0..=low_inventory_level {
            for threshold in 0..=position_upper_bound {
                let mean_cost = policy_rollout(
                    "bsp_low_ew",
                    &[low_inventory_level, high_inventory_level, threshold],
                    initial_state,
                    horizon,
                    seed,
                    lead_time,
                    max_order_size,
                    demand_mean,
                    demand_cov,
                    holding_cost,
                    shortage_cost,
                    waste_cost,
                    procurement_cost,
                    warm_up_periods_ratio,
                    issuing_policy,
                )?;
                results.push((
                    low_inventory_level,
                    high_inventory_level,
                    threshold,
                    mean_cost,
                ));
            }
        }
    }
    results.sort_by(|left, right| {
        left.3
            .total_cmp(&right.3)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    Ok((results[0], results.into_iter().take(top_k).collect()))
}

#[allow(clippy::too_many_arguments)]
pub fn search_bsp_low_ew_discounted_return(
    initial_state: &PerishableState,
    horizon: usize,
    seed: u64,
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: IssuingPolicy,
    position_upper_bound: usize,
    top_k: usize,
) -> PyResult<((usize, usize, usize, f64), Vec<(usize, usize, usize, f64)>)> {
    if top_k == 0 {
        return Err(PyValueError::new_err("top_k must be positive"));
    }
    let mut results = Vec::new();
    for low_inventory_level in 0..=position_upper_bound {
        for high_inventory_level in 0..=low_inventory_level {
            for threshold in 0..=position_upper_bound {
                let discounted_return = policy_discounted_return(
                    "bsp_low_ew",
                    &[low_inventory_level, high_inventory_level, threshold],
                    initial_state,
                    horizon,
                    seed,
                    lead_time,
                    max_order_size,
                    demand_mean,
                    demand_cov,
                    holding_cost,
                    shortage_cost,
                    waste_cost,
                    procurement_cost,
                    warm_up_periods_ratio,
                    gamma,
                    issuing_policy,
                )?;
                results.push((
                    low_inventory_level,
                    high_inventory_level,
                    threshold,
                    discounted_return,
                ));
            }
        }
    }
    results.sort_by(|left, right| {
        right
            .3
            .total_cmp(&left.3)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    Ok((results[0], results.into_iter().take(top_k).collect()))
}

#[allow(clippy::too_many_arguments)]
pub fn search_bsp_low_ew_discounted_return_summary(
    initial_state: &PerishableState,
    horizon: usize,
    seeds: &[u64],
    lead_time: usize,
    max_order_size: usize,
    demand_mean: f64,
    demand_cov: f64,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    warm_up_periods_ratio: f64,
    gamma: f64,
    issuing_policy: IssuingPolicy,
    position_upper_bound: usize,
    top_k: usize,
) -> PyResult<(
    (usize, usize, usize, DiscountedReturnSummary),
    Vec<(usize, usize, usize, DiscountedReturnSummary)>,
)> {
    if top_k == 0 {
        return Err(PyValueError::new_err("top_k must be positive"));
    }
    let mut results = Vec::new();
    for low_inventory_level in 0..=position_upper_bound {
        for high_inventory_level in 0..=low_inventory_level {
            for threshold in 0..=position_upper_bound {
                let summary = policy_discounted_return_summary(
                    "bsp_low_ew",
                    &[low_inventory_level, high_inventory_level, threshold],
                    initial_state,
                    horizon,
                    seeds,
                    lead_time,
                    max_order_size,
                    demand_mean,
                    demand_cov,
                    holding_cost,
                    shortage_cost,
                    waste_cost,
                    procurement_cost,
                    warm_up_periods_ratio,
                    gamma,
                    issuing_policy,
                )?;
                results.push((
                    low_inventory_level,
                    high_inventory_level,
                    threshold,
                    summary,
                ));
            }
        }
    }
    results.sort_by(|left, right| {
        right
            .3
            .mean_return
            .total_cmp(&left.3.mean_return)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    Ok((
        results[0].clone(),
        results.into_iter().take(top_k).collect(),
    ))
}
