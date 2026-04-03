use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::problems::lost_sales_fixed_order_cost::heuristics::{
    fixed_policy_rollout_from_demands, search_modified_s_s_q_from_demands,
    search_s_nq_from_demands, search_s_s_from_demands,
};

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

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    Ok(())
}
