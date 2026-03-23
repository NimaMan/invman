use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::env::dual_sourcing::{epoch_cost, step_state};

fn mean_after_warmup(epoch_costs: &[f64], warm_up_periods_ratio: f64) -> f64 {
    let horizon = epoch_costs.len();
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let active_costs = if warm_up_periods < epoch_costs.len() {
        &epoch_costs[warm_up_periods..]
    } else {
        epoch_costs
    };
    active_costs.iter().sum::<f64>() / active_costs.len() as f64
}

fn single_index_action(regular_inventory_position: i64, s_e: usize, s_r: usize, max_regular: usize, max_expedited: usize) -> (usize, usize) {
    let expedited = s_e.saturating_sub(regular_inventory_position.max(0) as usize).min(max_expedited);
    let regular = s_r
        .saturating_sub(regular_inventory_position.max(0) as usize + expedited)
        .min(max_regular);
    (regular, expedited)
}

fn dual_index_action(expedited_inventory_position: i64, regular_inventory_position: i64, s_e: usize, s_r: usize, max_regular: usize, max_expedited: usize) -> (usize, usize) {
    let expedited = s_e.saturating_sub(expedited_inventory_position.max(0) as usize).min(max_expedited);
    let regular = s_r
        .saturating_sub(regular_inventory_position.max(0) as usize + expedited)
        .min(max_regular);
    (regular, expedited)
}

fn capped_dual_index_action(expedited_inventory_position: i64, regular_inventory_position: i64, s_e: usize, s_r: usize, cap_r: usize, max_regular: usize, max_expedited: usize) -> (usize, usize) {
    let expedited = s_e.saturating_sub(expedited_inventory_position.max(0) as usize).min(max_expedited);
    let desired_regular = s_r.saturating_sub(regular_inventory_position.max(0) as usize + expedited);
    (desired_regular.min(cap_r).min(max_regular), expedited)
}

fn tailored_base_surge_action(expedited_inventory_position: i64, surge_level: usize, regular_qty: usize, max_regular: usize, max_expedited: usize) -> (usize, usize) {
    let expedited = surge_level.saturating_sub(expedited_inventory_position.max(0) as usize).min(max_expedited);
    (regular_qty.min(max_regular), expedited)
}

fn rollout_policy_from_demands(
    policy_name: &str,
    params: &[usize],
    state: &[i64],
    demands: &[usize],
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
    warm_up_periods_ratio: f64,
) -> PyResult<f64> {
    let mut reduced_state = state.to_vec();
    let mut epoch_costs = Vec::with_capacity(demands.len());
    for demand in demands.iter().copied() {
        let expedited_inventory_position = reduced_state[0];
        let regular_inventory_position = reduced_state.iter().sum::<i64>();
        let (regular_order, expedited_order) = match policy_name {
            "single_index" => single_index_action(
                regular_inventory_position,
                params[0],
                params[1],
                regular_max_order_size,
                expedited_max_order_size,
            ),
            "dual_index" => dual_index_action(
                expedited_inventory_position,
                regular_inventory_position,
                params[0],
                params[1],
                regular_max_order_size,
                expedited_max_order_size,
            ),
            "capped_dual_index" => capped_dual_index_action(
                expedited_inventory_position,
                regular_inventory_position,
                params[0],
                params[1],
                params[2],
                regular_max_order_size,
                expedited_max_order_size,
            ),
            "tailored_base_surge" => tailored_base_surge_action(
                expedited_inventory_position,
                params[0],
                params[1],
                regular_max_order_size,
                expedited_max_order_size,
            ),
            _ => return Err(PyValueError::new_err(format!("unknown dual-sourcing policy '{policy_name}'"))),
        };
        epoch_costs.push(epoch_cost(
            &reduced_state,
            regular_order,
            expedited_order,
            demand,
            regular_order_cost,
            expedited_order_cost,
            holding_cost,
            shortage_cost,
        ));
        reduced_state = step_state(&reduced_state, regular_order, expedited_order, demand);
    }
    Ok(mean_after_warmup(&epoch_costs, warm_up_periods_ratio))
}

fn search_two_param_policy(
    policy_name: &str,
    state: &[i64],
    demands: &[usize],
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
    let mut results = Vec::new();
    for s_e in 0..=target_upper_bound {
        for s_r in s_e..=target_upper_bound {
            let cost = rollout_policy_from_demands(
                policy_name,
                &[s_e, s_r],
                state,
                demands,
                regular_max_order_size,
                expedited_max_order_size,
                regular_order_cost,
                expedited_order_cost,
                holding_cost,
                shortage_cost,
                warm_up_periods_ratio,
            )?;
            results.push((s_e, s_r, cost));
        }
    }
    results.sort_by(|left, right| left.2.partial_cmp(&right.2).unwrap());
    Ok((results[0], results.into_iter().take(top_k).collect()))
}

pub fn search_single_index_from_demands(
    state: &[i64],
    demands: &[usize],
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
    search_two_param_policy(
        "single_index",
        state,
        demands,
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

pub fn search_dual_index_from_demands(
    state: &[i64],
    demands: &[usize],
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
    search_two_param_policy(
        "dual_index",
        state,
        demands,
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

pub fn search_capped_dual_index_from_demands(
    state: &[i64],
    demands: &[usize],
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
    let mut results = Vec::new();
    for s_e in 0..=target_upper_bound {
        for s_r in s_e..=target_upper_bound {
            for cap_r in 0..=regular_max_order_size {
                let cost = rollout_policy_from_demands(
                    "capped_dual_index",
                    &[s_e, s_r, cap_r],
                    state,
                    demands,
                    regular_max_order_size,
                    expedited_max_order_size,
                    regular_order_cost,
                    expedited_order_cost,
                    holding_cost,
                    shortage_cost,
                    warm_up_periods_ratio,
                )?;
                results.push((s_e, s_r, cap_r, cost));
            }
        }
    }
    results.sort_by(|left, right| left.3.partial_cmp(&right.3).unwrap());
    Ok((results[0], results.into_iter().take(top_k).collect()))
}

pub fn search_tailored_base_surge_from_demands(
    state: &[i64],
    demands: &[usize],
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
    let mut results = Vec::new();
    for surge_level in 0..=target_upper_bound {
        for regular_qty in 0..=regular_max_order_size {
            let cost = rollout_policy_from_demands(
                "tailored_base_surge",
                &[surge_level, regular_qty],
                state,
                demands,
                regular_max_order_size,
                expedited_max_order_size,
                regular_order_cost,
                expedited_order_cost,
                holding_cost,
                shortage_cost,
                warm_up_periods_ratio,
            )?;
            results.push((surge_level, regular_qty, cost));
        }
    }
    results.sort_by(|left, right| left.2.partial_cmp(&right.2).unwrap());
    Ok((results[0], results.into_iter().take(top_k).collect()))
}
