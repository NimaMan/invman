use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::problems::lost_sales_fixed_order_cost::exact_value_iteration::{
    evaluate_policy, solve_optimal_policy, ExactPolicyKind,
};
use crate::problems::lost_sales_fixed_order_cost::heuristics::{
    fixed_policy_rollout_from_demands, search_modified_s_s_q_from_demands,
    search_s_nq_from_demands, search_s_s_from_demands,
};
use crate::problems::lost_sales_fixed_order_cost::references::{
    get_reference_instance, list_reference_instances, FixedCostLostSalesReferenceInstance,
    PublishedHeuristicRow, BIJVANK_2015_REFERENCE, BIJVANK_2015_TABLE1_REFERENCE,
};

fn heuristic_row_to_py(py: Python<'_>, row: &PublishedHeuristicRow) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("policy_name", row.policy_name)?;
    dict.set_item("params", row.params.to_vec())?;
    dict.set_item("mean_cost", row.mean_cost)?;
    Ok(dict.into_any().unbind().into())
}

fn reference_to_py(
    py: Python<'_>,
    reference: &FixedCostLostSalesReferenceInstance,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", reference.name)?;
    dict.set_item("source", reference.source)?;
    dict.set_item("url", reference.url)?;
    dict.set_item("literature_verified", reference.literature_verified)?;
    dict.set_item("review_periods", reference.review_periods)?;
    dict.set_item("lead_time", reference.lead_time)?;
    dict.set_item("demand_distribution", reference.demand_distribution)?;
    dict.set_item(
        "demand_mean_per_review_period",
        reference.demand_mean_per_review_period,
    )?;
    dict.set_item("holding_cost", reference.holding_cost)?;
    dict.set_item("shortage_cost", reference.shortage_cost)?;
    dict.set_item("fixed_order_cost", reference.fixed_order_cost)?;
    dict.set_item("published_optimal_cost", reference.published_optimal_cost)?;
    dict.set_item("benchmark_policies", reference.benchmark_policies.to_vec())?;
    dict.set_item(
        "published_heuristic_rows",
        reference
            .published_heuristic_rows
            .iter()
            .map(|row| heuristic_row_to_py(py, row))
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    dict.set_item("notes", reference.notes)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn lost_sales_fixed_order_cost_reference_catalog(py: Python<'_>) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("source", BIJVANK_2015_REFERENCE.source)?;
    dict.set_item("url", BIJVANK_2015_REFERENCE.url)?;
    dict.set_item(
        "benchmark_policies",
        BIJVANK_2015_REFERENCE.benchmark_policies.to_vec(),
    )?;
    dict.set_item(
        "reported_numbers_available",
        BIJVANK_2015_REFERENCE.reported_numbers_available,
    )?;
    dict.set_item("notes", BIJVANK_2015_REFERENCE.notes)?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
fn lost_sales_fixed_order_cost_list_reference_instances() -> Vec<&'static str> {
    list_reference_instances()
}

#[pyfunction]
fn lost_sales_fixed_order_cost_get_reference_instance(
    py: Python<'_>,
    name: &str,
) -> PyResult<PyObject> {
    let reference = get_reference_instance(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown fixed-cost reference '{name}'")))?;
    reference_to_py(py, reference)
}

#[pyfunction]
fn lost_sales_fixed_order_cost_primary_reference_instance(py: Python<'_>) -> PyResult<PyObject> {
    reference_to_py(py, &BIJVANK_2015_TABLE1_REFERENCE)
}

#[pyfunction]
#[pyo3(signature = (reference_name="bijvank2015_table1_l2_p14_k5", inventory_position_cap=24))]
fn lost_sales_fixed_order_cost_exact_literature_summary(
    py: Python<'_>,
    reference_name: &str,
    inventory_position_cap: usize,
) -> PyResult<PyObject> {
    let reference = get_reference_instance(reference_name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown fixed-cost reference '{reference_name}'")))?;
    let published_s_s = reference
        .published_heuristic_rows
        .iter()
        .find(|row| row.policy_name == "s_s")
        .expect("published (s,S) row exists");
    let published_s_nq = reference
        .published_heuristic_rows
        .iter()
        .find(|row| row.policy_name == "s_nq")
        .expect("published (s,nQ) row exists");
    let published_modified = reference
        .published_heuristic_rows
        .iter()
        .find(|row| row.policy_name == "modified_s_s_q")
        .expect("published modified (s,S,q) row exists");
    let optimal = solve_optimal_policy(reference, inventory_position_cap)?;
    let s_s = evaluate_policy(
        reference,
        inventory_position_cap,
        ExactPolicyKind::Ss {
            s: published_s_s.params[0],
            s_up_to: published_s_s.params[1],
        },
    )?;
    let s_nq = evaluate_policy(
        reference,
        inventory_position_cap,
        ExactPolicyKind::Snq {
            s: published_s_nq.params[0],
            q: published_s_nq.params[1],
        },
    )?;
    let modified = evaluate_policy(
        reference,
        inventory_position_cap,
        ExactPolicyKind::ModifiedSsQ {
            s: published_modified.params[0],
            s_up_to: published_modified.params[1],
            q: published_modified.params[2],
        },
    )?;

    let dict = PyDict::new_bound(py);
    dict.set_item("reference", reference_to_py(py, reference)?)?;
    dict.set_item("inventory_position_cap", inventory_position_cap)?;
    dict.set_item("optimal_average_cost", optimal.average_cost)?;
    dict.set_item("optimal_first_action", optimal.first_action)?;
    dict.set_item("optimal_iterations", optimal.iterations)?;
    dict.set_item("optimal_final_span", optimal.final_span)?;
    dict.set_item("s_s_average_cost", s_s.average_cost)?;
    dict.set_item("s_s_first_action", s_s.first_action)?;
    dict.set_item("s_nq_average_cost", s_nq.average_cost)?;
    dict.set_item("s_nq_first_action", s_nq.first_action)?;
    dict.set_item("modified_s_s_q_average_cost", modified.average_cost)?;
    dict.set_item("modified_s_s_q_first_action", modified.first_action)?;
    dict.set_item(
        "published_optimal_cost",
        reference.published_optimal_cost.expect("published optimal cost exists"),
    )?;
    dict.set_item("published_s_s_cost", published_s_s.mean_cost)?;
    dict.set_item("published_s_nq_cost", published_s_nq.mean_cost)?;
    dict.set_item("published_modified_s_s_q_cost", published_modified.mean_cost)?;
    dict.set_item(
        "optimal_gap_to_published",
        optimal.average_cost - reference.published_optimal_cost.expect("published optimal cost exists"),
    )?;
    dict.set_item("s_s_gap_to_published", s_s.average_cost - published_s_s.mean_cost)?;
    dict.set_item("s_nq_gap_to_published", s_nq.average_cost - published_s_nq.mean_cost)?;
    dict.set_item(
        "modified_s_s_q_gap_to_published",
        modified.average_cost - published_modified.mean_cost,
    )?;
    Ok(dict.into_any().unbind().into())
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

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        lost_sales_fixed_order_cost_reference_catalog,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_fixed_order_cost_list_reference_instances,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_fixed_order_cost_get_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_fixed_order_cost_primary_reference_instance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_fixed_order_cost_exact_literature_summary,
        m
    )?)?;
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
