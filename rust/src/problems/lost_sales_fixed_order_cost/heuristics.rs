use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::lost_sales::env::epoch_cost;

fn mean_after_warmup(epoch_costs: &[f64], warm_up_periods_ratio: f64) -> PyResult<f64> {
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

fn inventory_position(current_inventory: i64, lead_time_orders: &[usize]) -> i64 {
    current_inventory
        + lead_time_orders
            .iter()
            .map(|order| *order as i64)
            .sum::<i64>()
}

pub fn s_s_order_quantity(
    inventory_position: i64,
    s: usize,
    s_up_to: usize,
    max_order_size: usize,
) -> usize {
    if inventory_position > s as i64 {
        return 0;
    }
    s_up_to
        .saturating_sub(inventory_position.max(0) as usize)
        .min(max_order_size)
}

pub fn s_nq_order_quantity(
    inventory_position: i64,
    s: usize,
    q: usize,
    max_order_size: usize,
) -> PyResult<usize> {
    if q == 0 {
        return Err(PyValueError::new_err("q must be positive"));
    }
    if inventory_position > s as i64 {
        return Ok(0);
    }
    let deficit = (s + 1).saturating_sub(inventory_position.max(0) as usize);
    let batches = (deficit + q - 1) / q;
    Ok((batches * q).min(max_order_size))
}

pub fn modified_s_s_q_order_quantity(
    inventory_position: i64,
    s: usize,
    s_up_to: usize,
    q: usize,
    max_order_size: usize,
) -> PyResult<usize> {
    if q == 0 {
        return Err(PyValueError::new_err("q must be positive"));
    }
    if inventory_position > s as i64 {
        return Ok(0);
    }
    Ok(
        q.min(s_up_to.saturating_sub(inventory_position.max(0) as usize))
            .min(max_order_size),
    )
}

pub fn fixed_policy_rollout_from_demands(
    policy_name: &str,
    params: &[usize],
    current_inventory: i64,
    lead_time_orders: &[usize],
    demands: &[usize],
    max_order_size: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    warm_up_periods_ratio: f64,
) -> PyResult<f64> {
    if lead_time_orders.is_empty() {
        return Err(PyValueError::new_err("lead_time_orders must be non-empty"));
    }

    let mut current_inventory = current_inventory;
    let mut lead_time_orders = lead_time_orders.to_vec();
    let mut epoch_costs = Vec::with_capacity(demands.len());

    for demand in demands.iter() {
        let inventory_position = inventory_position(current_inventory, &lead_time_orders);
        let order_quantity = match policy_name {
            "s_s" => {
                if params.len() != 2 {
                    return Err(PyValueError::new_err("s_s expects params [s, S]"));
                }
                s_s_order_quantity(inventory_position, params[0], params[1], max_order_size)
            }
            "s_nq" => {
                if params.len() != 2 {
                    return Err(PyValueError::new_err("s_nq expects params [s, q]"));
                }
                s_nq_order_quantity(inventory_position, params[0], params[1], max_order_size)?
            }
            "modified_s_s_q" => {
                if params.len() != 3 {
                    return Err(PyValueError::new_err(
                        "modified_s_s_q expects params [s, S, q]",
                    ));
                }
                modified_s_s_q_order_quantity(
                    inventory_position,
                    params[0],
                    params[1],
                    params[2],
                    max_order_size,
                )?
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported policy '{}'",
                    policy_name
                )))
            }
        };

        let arriving_order = lead_time_orders.remove(0);
        lead_time_orders.push(order_quantity);
        current_inventory += arriving_order as i64;

        epoch_costs.push(epoch_cost(
            &mut current_inventory,
            *demand as i64,
            order_quantity,
            holding_cost,
            shortage_cost,
            procurement_cost,
            fixed_order_cost,
        ));
    }

    mean_after_warmup(&epoch_costs, warm_up_periods_ratio)
}

pub fn search_s_s_from_demands(
    current_inventory: i64,
    lead_time_orders: &[usize],
    demands: &[usize],
    max_order_size: usize,
    position_upper_bound: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    warm_up_periods_ratio: f64,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    let mut results = Vec::new();
    for s in 0..position_upper_bound {
        for s_up_to in (s + 1)..=position_upper_bound {
            let mean_cost = fixed_policy_rollout_from_demands(
                "s_s",
                &[s, s_up_to],
                current_inventory,
                lead_time_orders,
                demands,
                max_order_size,
                holding_cost,
                shortage_cost,
                procurement_cost,
                fixed_order_cost,
                warm_up_periods_ratio,
            )?;
            results.push((s, s_up_to, mean_cost));
        }
    }
    results.sort_by(|a, b| a.2.total_cmp(&b.2));
    Ok((results[0], results.into_iter().take(top_k).collect()))
}

pub fn search_s_nq_from_demands(
    current_inventory: i64,
    lead_time_orders: &[usize],
    demands: &[usize],
    max_order_size: usize,
    position_upper_bound: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
    warm_up_periods_ratio: f64,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    let mut results = Vec::new();
    for s in 0..position_upper_bound {
        for q in 1..=position_upper_bound {
            let mean_cost = fixed_policy_rollout_from_demands(
                "s_nq",
                &[s, q],
                current_inventory,
                lead_time_orders,
                demands,
                max_order_size,
                holding_cost,
                shortage_cost,
                procurement_cost,
                fixed_order_cost,
                warm_up_periods_ratio,
            )?;
            results.push((s, q, mean_cost));
        }
    }
    results.sort_by(|a, b| a.2.total_cmp(&b.2));
    Ok((results[0], results.into_iter().take(top_k).collect()))
}

pub fn search_modified_s_s_q_from_demands(
    current_inventory: i64,
    lead_time_orders: &[usize],
    demands: &[usize],
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
    let mut results = Vec::new();
    for s in 0..position_upper_bound {
        for s_up_to in (s + 1)..=position_upper_bound {
            for q in 1..=max_order_size {
                let mean_cost = fixed_policy_rollout_from_demands(
                    "modified_s_s_q",
                    &[s, s_up_to, q],
                    current_inventory,
                    lead_time_orders,
                    demands,
                    max_order_size,
                    holding_cost,
                    shortage_cost,
                    procurement_cost,
                    fixed_order_cost,
                    warm_up_periods_ratio,
                )?;
                results.push((s, s_up_to, q, mean_cost));
            }
        }
    }
    results.sort_by(|a, b| a.3.total_cmp(&b.3));
    let evaluated_candidates = results.len();
    Ok((
        results[0],
        results.into_iter().take(top_k).collect(),
        evaluated_candidates,
    ))
}
