use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::problems::lost_sales_fixed_order_cost::heuristics::{
    fixed_policy_rollout_from_demands, search_modified_s_s_q_from_demands,
    search_s_nq_from_demands, search_s_s_from_demands,
};
use crate::problems::lost_sales_fixed_order_cost::literature::{
    get_reference_instance, list_reference_instances, FixedCostLostSalesReferenceInstance,
    PublishedHeuristicRow, BIJVANK_2015_REFERENCE, BIJVANK_2015_TABLE1_REFERENCE,
};
use crate::problems::lost_sales_fixed_order_cost::exact_value_iteration::{
    evaluate_policy, solve_optimal_policy, ExactPolicyKind,
};
use crate::problems::lost_sales_fixed_order_cost::experiments::{
    expand_experiment_grid, get_experiment_grid, list_experiment_grids,
    FixedCostExperimentDemandCase, FixedCostExperimentGrid, FixedCostExperimentInstance,
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

fn demand_case_to_py(py: Python<'_>, case: &FixedCostExperimentDemandCase) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("key", case.key)?;
    dict.set_item("display_name", case.display_name)?;
    dict.set_item("name_token", case.name_token)?;
    dict.set_item("demand_distribution", case.demand_distribution)?;
    dict.set_item("demand_rate", case.demand_rate)?;
    dict.set_item("demand_lambda_low", case.demand_lambda_low)?;
    dict.set_item("demand_lambda_high", case.demand_lambda_high)?;
    dict.set_item("demand_p00", case.demand_p00)?;
    dict.set_item("demand_p11", case.demand_p11)?;
    dict.set_item("notes", case.notes)?;
    Ok(dict.into_any().unbind().into())
}

fn experiment_grid_to_py(py: Python<'_>, grid: &FixedCostExperimentGrid) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", grid.name)?;
    dict.set_item("description", grid.description)?;
    dict.set_item(
        "demand_cases",
        grid.demand_cases
            .iter()
            .map(|case| demand_case_to_py(py, case))
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    dict.set_item("shortage_costs", grid.shortage_costs.to_vec())?;
    dict.set_item("fixed_order_costs", grid.fixed_order_costs.to_vec())?;
    dict.set_item("lead_times", grid.lead_times.to_vec())?;
    dict.set_item("holding_cost", grid.holding_cost)?;
    dict.set_item("mean_demand", grid.mean_demand)?;
    Ok(dict.into_any().unbind().into())
}

fn experiment_instance_to_py(
    py: Python<'_>,
    instance: &FixedCostExperimentInstance,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", instance.name.as_str())?;
    dict.set_item("description", instance.description.as_str())?;
    dict.set_item("demand_case_key", instance.demand_case_key)?;
    dict.set_item("demand_case_display_name", instance.demand_case_display_name)?;

    let params = PyDict::new_bound(py);
    params.set_item("problem", "lost_sales_fixed_order_cost")?;
    params.set_item("demand_dist_name", instance.demand_distribution)?;
    params.set_item("demand_rate", instance.demand_rate)?;
    if let Some(value) = instance.demand_lambda_low {
        params.set_item("demand_lambda_low", value)?;
    }
    if let Some(value) = instance.demand_lambda_high {
        params.set_item("demand_lambda_high", value)?;
    }
    if let Some(value) = instance.demand_p00 {
        params.set_item("demand_p00", value)?;
    }
    if let Some(value) = instance.demand_p11 {
        params.set_item("demand_p11", value)?;
    }
    params.set_item("lead_time", instance.lead_time)?;
    params.set_item("shortage_cost", instance.shortage_cost)?;
    params.set_item("fixed_order_cost", instance.fixed_order_cost)?;
    params.set_item("holding_cost", instance.holding_cost)?;
    params.set_item("procurement_cost", instance.procurement_cost)?;
    params.set_item("max_order_size", instance.max_order_size)?;
    params.set_item("horizon", instance.horizon)?;
    params.set_item("eval_horizon", instance.eval_horizon)?;
    params.set_item("warm_up_periods_ratio", instance.warm_up_periods_ratio)?;
    params.set_item("seed", instance.seed)?;
    params.set_item("state_normalizer", instance.state_normalizer)?;
    params.set_item("state_scale", instance.state_scale)?;
    dict.set_item("params", params)?;

    let search = PyDict::new_bound(py);
    search.set_item("position_upper_bound", instance.position_upper_bound)?;
    search.set_item("search_horizon", instance.search_horizon)?;
    search.set_item("search_seed", instance.search_seed)?;
    search.set_item("top_k_s_s_pairs", instance.top_k_s_s_pairs)?;
    search.set_item("q_window", instance.q_window)?;
    dict.set_item("search", search)?;

    let evaluation = PyDict::new_bound(py);
    evaluation.set_item("eval_horizon", instance.evaluation_eval_horizon)?;
    evaluation.set_item("eval_seeds", instance.evaluation_eval_seeds)?;
    dict.set_item("evaluation", evaluation)?;

    let literature_metadata = PyDict::new_bound(py);
    literature_metadata.set_item("benchmark_family", instance.benchmark_family)?;
    literature_metadata.set_item("parent_problem_family", instance.parent_problem_family)?;
    literature_metadata.set_item("demand_case", instance.demand_case_key)?;
    literature_metadata.set_item("demand_case_display_name", instance.demand_case_display_name)?;
    literature_metadata.set_item("notes", instance.notes.as_str())?;
    dict.set_item("literature_metadata", literature_metadata)?;

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
fn lost_sales_fixed_order_cost_list_experiment_grids(py: Python<'_>) -> PyResult<Vec<PyObject>> {
    list_experiment_grids()
        .iter()
        .map(|grid| experiment_grid_to_py(py, grid))
        .collect()
}

#[pyfunction]
fn lost_sales_fixed_order_cost_get_experiment_grid(
    py: Python<'_>,
    name: &str,
) -> PyResult<PyObject> {
    let grid = get_experiment_grid(name).ok_or_else(|| {
        PyValueError::new_err(format!("unknown fixed-cost experiment grid '{name}'"))
    })?;
    experiment_grid_to_py(py, grid)
}

#[pyfunction]
fn lost_sales_fixed_order_cost_expand_experiment_grid(
    py: Python<'_>,
    name: &str,
) -> PyResult<Vec<PyObject>> {
    expand_experiment_grid(name)
        .map_err(PyValueError::new_err)?
        .iter()
        .map(|instance| experiment_instance_to_py(py, instance))
        .collect()
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
        lost_sales_fixed_order_cost_list_experiment_grids,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_fixed_order_cost_get_experiment_grid,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_fixed_order_cost_expand_experiment_grid,
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
