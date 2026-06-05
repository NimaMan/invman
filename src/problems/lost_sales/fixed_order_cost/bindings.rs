use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::problems::lost_sales::fixed_order_cost::exact_value_iteration::{
    evaluate_policy, solve_optimal_policy, ExactPolicyKind,
};
use crate::problems::lost_sales::fixed_order_cost::experiments::{
    expand_experiment_grid, get_experiment_grid, list_experiment_grids,
    FixedCostExperimentDemandCase, FixedCostExperimentGrid, FixedCostExperimentInstance,
};
use crate::problems::lost_sales::demand::{
    build_demand_process, parse_demand_kind, sample_demand, LostSalesDemandConfig,
};
use crate::problems::lost_sales::fixed_order_cost::heuristics::{
    fixed_policy_rollout_from_demands, fixed_policy_trace_from_demands,
    search_modified_s_s_q_from_demands, search_s_nq_from_demands, search_s_s_from_demands,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::problems::lost_sales::fixed_order_cost::literature::{
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
    dict.set_item(
        "demand_case_display_name",
        instance.demand_case_display_name,
    )?;

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
    literature_metadata.set_item(
        "demand_case_display_name",
        instance.demand_case_display_name,
    )?;
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
    let reference = get_reference_instance(reference_name).ok_or_else(|| {
        PyValueError::new_err(format!("unknown fixed-cost reference '{reference_name}'"))
    })?;
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
        reference
            .published_optimal_cost
            .expect("published optimal cost exists"),
    )?;
    dict.set_item("published_s_s_cost", published_s_s.mean_cost)?;
    dict.set_item("published_s_nq_cost", published_s_nq.mean_cost)?;
    dict.set_item(
        "published_modified_s_s_q_cost",
        published_modified.mean_cost,
    )?;
    dict.set_item(
        "optimal_gap_to_published",
        optimal.average_cost
            - reference
                .published_optimal_cost
                .expect("published optimal cost exists"),
    )?;
    dict.set_item(
        "s_s_gap_to_published",
        s_s.average_cost - published_s_s.mean_cost,
    )?;
    dict.set_item(
        "s_nq_gap_to_published",
        s_nq.average_cost - published_s_nq.mean_cost,
    )?;
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
fn lost_sales_fixed_policy_trace_from_demands(
    py: Python<'_>,
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
) -> PyResult<Py<PyDict>> {
    let warm_up_periods =
        ((warm_up_periods_ratio * demands.len() as f64).floor() as usize).min(demands.len());
    let (mean_cost, trace) = fixed_policy_trace_from_demands(
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
    )?;

    let rows = trace
        .iter()
        .map(|row| {
            let dict = PyDict::new_bound(py);
            dict.set_item("period", row.period)?;
            dict.set_item("demand", row.demand)?;
            dict.set_item("pipeline_before_order", row.pipeline_before_order.clone())?;
            dict.set_item(
                "inventory_position_before_order",
                row.inventory_position_before_order,
            )?;
            dict.set_item("order_quantity", row.order_quantity)?;
            dict.set_item("arriving_order", row.arriving_order)?;
            dict.set_item("inventory_before_demand", row.inventory_before_demand)?;
            dict.set_item("ending_inventory", row.ending_inventory)?;
            dict.set_item("period_cost", row.period_cost)?;
            dict.set_item("active_after_warmup", row.active_after_warmup)?;
            Ok(dict.into_any().unbind().into())
        })
        .collect::<PyResult<Vec<PyObject>>>()?;

    let result = PyDict::new_bound(py);
    result.set_item("policy_name", policy_name)?;
    result.set_item("params", params)?;
    result.set_item("mean_cost", mean_cost)?;
    result.set_item("warm_up_periods", warm_up_periods)?;
    result.set_item("trace", rows)?;
    Ok(result.into())
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

/// Best (s,S), (s,nQ) and modified (s,S,q) mean costs for a fixed-cost instance,
/// generating the demand sample internally (mirrors the vanilla
/// `lost_sales_heuristics_all`). Each heuristic searches its best parameters on
/// the sampled demand and returns the warm-up-trimmed mean cost.
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
    max_order_size,
    position_upper_bound,
    horizon,
    seed,
    warm_up_periods_ratio=0.2,
    top_k=1
))]
fn lost_sales_fixed_heuristics_all(
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
    max_order_size: usize,
    position_upper_bound: usize,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    top_k: usize,
) -> PyResult<Py<PyDict>> {
    let kind = parse_demand_kind(demand_kind).map_err(PyValueError::new_err)?;
    let demand_config = LostSalesDemandConfig {
        kind,
        demand_rate,
        demand_lambda_low,
        demand_lambda_high,
        demand_p00,
        demand_p11,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut process = build_demand_process(demand_config, &mut rng).map_err(PyValueError::new_err)?;
    let demands: Vec<usize> = (0..horizon)
        .map(|_| sample_demand(&mut rng, &mut process).max(0) as usize)
        .collect();
    let lead_time_orders = vec![0usize; lead_time];
    let current_inventory: i64 = 0;

    let (s_s_best, _) = search_s_s_from_demands(
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
    )?;
    let (s_nq_best, _) = search_s_nq_from_demands(
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
    )?;
    let (ssq_best, _, _) = search_modified_s_s_q_from_demands(
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
    )?;

    let result = PyDict::new_bound(py);
    result.set_item("s_s", s_s_best.2)?;
    result.set_item("s_nq", s_nq_best.2)?;
    result.set_item("modified_s_s_q", ssq_best.3)?;
    Ok(result.into())
}

/// Detailed best (s,S), (s,nQ), and modified (s,S,q) search results for a
/// fixed-cost instance. This keeps `lost_sales_fixed_heuristics_all` as the
/// legacy cost-only API while exposing the winning parameters needed by paper
/// tables and benchmark summaries.
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
    max_order_size,
    position_upper_bound,
    horizon,
    seed,
    warm_up_periods_ratio=0.2,
    top_k=1
))]
fn lost_sales_fixed_heuristics_all_detailed(
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
    max_order_size: usize,
    position_upper_bound: usize,
    horizon: usize,
    seed: u64,
    warm_up_periods_ratio: f64,
    top_k: usize,
) -> PyResult<Py<PyDict>> {
    let kind = parse_demand_kind(demand_kind).map_err(PyValueError::new_err)?;
    let demand_config = LostSalesDemandConfig {
        kind,
        demand_rate,
        demand_lambda_low,
        demand_lambda_high,
        demand_p00,
        demand_p11,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut process = build_demand_process(demand_config, &mut rng).map_err(PyValueError::new_err)?;
    let demands: Vec<usize> = (0..horizon)
        .map(|_| sample_demand(&mut rng, &mut process).max(0) as usize)
        .collect();
    let lead_time_orders = vec![0usize; lead_time];
    let current_inventory: i64 = 0;

    let (s_s_best, s_s_top) = search_s_s_from_demands(
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
    )?;
    let (s_nq_best, s_nq_top) = search_s_nq_from_demands(
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
    )?;
    let (ssq_best, ssq_top, ssq_evaluated_candidates) =
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
        )?;

    let result = PyDict::new_bound(py);

    let s_s = PyDict::new_bound(py);
    s_s.set_item("policy_name", "s_s")?;
    s_s.set_item("params", vec![s_s_best.0, s_s_best.1])?;
    s_s.set_item("mean_cost", s_s_best.2)?;
    let s_s_top_rows = s_s_top
        .iter()
        .map(|row| {
            let dict = PyDict::new_bound(py);
            dict.set_item("params", vec![row.0, row.1])?;
            dict.set_item("mean_cost", row.2)?;
            Ok(dict.into_any().unbind().into())
        })
        .collect::<PyResult<Vec<PyObject>>>()?;
    s_s.set_item("top", s_s_top_rows)?;
    s_s.set_item("evaluated_candidates", position_upper_bound * (position_upper_bound + 1) / 2)?;
    result.set_item("s_s", s_s)?;

    let s_nq = PyDict::new_bound(py);
    s_nq.set_item("policy_name", "s_nq")?;
    s_nq.set_item("params", vec![s_nq_best.0, s_nq_best.1])?;
    s_nq.set_item("mean_cost", s_nq_best.2)?;
    let s_nq_top_rows = s_nq_top
        .iter()
        .map(|row| {
            let dict = PyDict::new_bound(py);
            dict.set_item("params", vec![row.0, row.1])?;
            dict.set_item("mean_cost", row.2)?;
            Ok(dict.into_any().unbind().into())
        })
        .collect::<PyResult<Vec<PyObject>>>()?;
    s_nq.set_item("top", s_nq_top_rows)?;
    s_nq.set_item("evaluated_candidates", position_upper_bound * position_upper_bound)?;
    result.set_item("s_nq", s_nq)?;

    let modified = PyDict::new_bound(py);
    modified.set_item("policy_name", "modified_s_s_q")?;
    modified.set_item("params", vec![ssq_best.0, ssq_best.1, ssq_best.2])?;
    modified.set_item("mean_cost", ssq_best.3)?;
    let ssq_top_rows = ssq_top
        .iter()
        .map(|row| {
            let dict = PyDict::new_bound(py);
            dict.set_item("params", vec![row.0, row.1, row.2])?;
            dict.set_item("mean_cost", row.3)?;
            Ok(dict.into_any().unbind().into())
        })
        .collect::<PyResult<Vec<PyObject>>>()?;
    modified.set_item("top", ssq_top_rows)?;
    modified.set_item("evaluated_candidates", ssq_evaluated_candidates)?;
    result.set_item("modified_s_s_q", modified)?;

    result.set_item("search_horizon", horizon)?;
    result.set_item("search_seed", seed)?;
    result.set_item("position_upper_bound", position_upper_bound)?;
    result.set_item("top_k", top_k)?;
    Ok(result.into())
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
        lost_sales_fixed_policy_trace_from_demands,
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
    m.add_function(wrap_pyfunction!(lost_sales_fixed_heuristics_all, m)?)?;
    m.add_function(wrap_pyfunction!(
        lost_sales_fixed_heuristics_all_detailed,
        m
    )?)?;
    Ok(())
}
