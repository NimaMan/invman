use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use crate::case_studies::hormuz_strait::scenarios::{
    simulate_month_ahead_price_scenarios, HormuzDailyPriceSummary,
    HormuzMonthAheadSimulationReport, HormuzScenarioAssumption, HormuzScenarioSimulationSummary,
};

fn assumption_to_py(py: Python<'_>, assumption: &HormuzScenarioAssumption) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", &assumption.name)?;
    dict.set_item("value", assumption.value)?;
    dict.set_item("units", &assumption.units)?;
    Ok(dict.into_any().unbind().into())
}

fn daily_summary_to_py(py: Python<'_>, daily: &HormuzDailyPriceSummary) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("day_index", daily.day_index)?;
    dict.set_item(
        "mean_brent_price_usd_per_bbl",
        daily.mean_brent_price_usd_per_bbl,
    )?;
    dict.set_item(
        "p10_brent_price_usd_per_bbl",
        daily.p10_brent_price_usd_per_bbl,
    )?;
    dict.set_item(
        "p50_brent_price_usd_per_bbl",
        daily.p50_brent_price_usd_per_bbl,
    )?;
    dict.set_item(
        "p90_brent_price_usd_per_bbl",
        daily.p90_brent_price_usd_per_bbl,
    )?;
    dict.set_item("closure_fraction", daily.closure_fraction)?;
    dict.set_item("blocked_flow_million_bpd", daily.blocked_flow_million_bpd)?;
    dict.set_item("rerouted_flow_million_bpd", daily.rerouted_flow_million_bpd)?;
    dict.set_item(
        "reserve_release_million_bpd",
        daily.reserve_release_million_bpd,
    )?;
    dict.set_item(
        "floating_storage_release_million_bpd",
        daily.floating_storage_release_million_bpd,
    )?;
    dict.set_item(
        "non_hormuz_supply_response_million_bpd",
        daily.non_hormuz_supply_response_million_bpd,
    )?;
    dict.set_item(
        "inventory_buffer_draw_million_bpd",
        daily.inventory_buffer_draw_million_bpd,
    )?;
    dict.set_item(
        "effective_tightness_million_bpd",
        daily.effective_tightness_million_bpd,
    )?;
    dict.set_item("target_price_usd_per_bbl", daily.target_price_usd_per_bbl)?;
    Ok(dict.into_any().unbind().into())
}

fn scenario_summary_to_py(
    py: Python<'_>,
    scenario: &HormuzScenarioSimulationSummary,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("scenario_id", &scenario.scenario_id)?;
    dict.set_item("label", &scenario.label)?;
    dict.set_item("description", &scenario.description)?;
    dict.set_item(
        "day_30_mean_brent_price_usd_per_bbl",
        scenario.day_30_mean_brent_price_usd_per_bbl,
    )?;
    dict.set_item(
        "day_30_p10_brent_price_usd_per_bbl",
        scenario.day_30_p10_brent_price_usd_per_bbl,
    )?;
    dict.set_item(
        "day_30_p50_brent_price_usd_per_bbl",
        scenario.day_30_p50_brent_price_usd_per_bbl,
    )?;
    dict.set_item(
        "day_30_p90_brent_price_usd_per_bbl",
        scenario.day_30_p90_brent_price_usd_per_bbl,
    )?;
    dict.set_item(
        "monthly_average_mean_brent_price_usd_per_bbl",
        scenario.monthly_average_mean_brent_price_usd_per_bbl,
    )?;
    dict.set_item(
        "peak_mean_brent_price_usd_per_bbl",
        scenario.peak_mean_brent_price_usd_per_bbl,
    )?;
    dict.set_item("peak_mean_price_day", scenario.peak_mean_price_day)?;
    dict.set_item(
        "mean_effective_tightness_million_bpd",
        scenario.mean_effective_tightness_million_bpd,
    )?;
    dict.set_item(
        "max_effective_tightness_million_bpd",
        scenario.max_effective_tightness_million_bpd,
    )?;
    dict.set_item(
        "assumptions",
        scenario
            .assumptions
            .iter()
            .map(|assumption| assumption_to_py(py, assumption))
            .collect::<PyResult<Vec<PyObject>>>()?,
    )?;
    dict.set_item(
        "daily",
        scenario
            .daily
            .iter()
            .map(|daily| daily_summary_to_py(py, daily))
            .collect::<PyResult<Vec<PyObject>>>()?,
    )?;
    Ok(dict.into_any().unbind().into())
}

fn report_to_py(py: Python<'_>, report: &HormuzMonthAheadSimulationReport) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("analysis_date", &report.analysis_date)?;
    dict.set_item(
        "latest_observed_close_date",
        &report.latest_observed_close_date,
    )?;
    dict.set_item(
        "latest_observed_brent_price_usd_per_bbl",
        report.latest_observed_brent_price_usd_per_bbl,
    )?;
    dict.set_item(
        "latest_observed_wti_price_usd_per_bbl",
        report.latest_observed_wti_price_usd_per_bbl,
    )?;
    dict.set_item(
        "eia_next_two_month_floor_brent_usd_per_bbl",
        report.eia_next_two_month_floor_brent_usd_per_bbl,
    )?;
    dict.set_item(
        "eia_q2_2026_average_brent_usd_per_bbl",
        report.eia_q2_2026_average_brent_usd_per_bbl,
    )?;
    dict.set_item("days", report.days)?;
    dict.set_item("paths", report.paths)?;
    dict.set_item("notes", &report.notes)?;
    dict.set_item(
        "scenarios",
        report
            .scenarios
            .iter()
            .map(|scenario| scenario_summary_to_py(py, scenario))
            .collect::<PyResult<Vec<PyObject>>>()?,
    )?;
    Ok(dict.into_any().unbind().into())
}

#[pyfunction]
#[pyo3(signature = (days=30, paths=4000, seed=20260406))]
fn hormuz_strait_month_ahead_price_scenarios(
    days: usize,
    paths: usize,
    seed: u64,
) -> PyResult<PyObject> {
    if days == 0 || days > 365 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "days must be between 1 and 365",
        ));
    }
    if paths == 0 || paths > 100_000 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "paths must be between 1 and 100000",
        ));
    }
    Python::with_gil(|py| {
        report_to_py(py, &simulate_month_ahead_price_scenarios(days, paths, seed))
    })
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        hormuz_strait_month_ahead_price_scenarios,
        m
    )?)?;
    Ok(())
}
