use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::multi_echelon::env::{parse_allocation_mode, parse_inventory_dynamics_mode};
use crate::problems::multi_echelon::heuristics::{
    evaluate_stationary_policy, HeuristicSimulationConfig, StationaryPolicyKind,
};
use crate::problems::multi_echelon::references::{
    MultiEchelonReferenceInstance, GIJSBRECHTS_2022_REFERENCE, LITERATURE_REFERENCE_INSTANCES,
};
use crate::problems::multi_echelon::rollout::{parse_demand_distribution, parse_rollout_objective};

pub const DEFAULT_GIJS_RELATIVE_VERIFICATION_REPLICATIONS: usize = 20;
pub const DEFAULT_GIJS_RELATIVE_VERIFICATION_SEED: u64 = 123;
pub const PUBLISHED_CONSTANT_BASE_STOCK_RELATIVE_TOLERANCE_PCT: f64 = 1.0;

#[derive(Clone, Debug, PartialEq)]
pub struct GijsRelativeVerificationRow {
    pub instance_name: &'static str,
    pub published_constant_base_stock_levels: Vec<usize>,
    pub published_constant_base_stock_mean_cost: f64,
    pub published_a3c_savings_pct: f64,
    pub published_a3c_confidence_half_width_pct: f64,
    pub published_a3c_implied_mean_cost: f64,
    pub published_van_roy_savings_pct_approx: f64,
    pub published_van_roy_implied_mean_cost: f64,
    pub repo_published_constant_base_stock_mean_cost: f64,
    pub repo_published_constant_base_stock_cost_std: f64,
    pub repo_gap_vs_published_constant_cost: f64,
    pub repo_gap_vs_published_constant_cost_pct: f64,
    pub published_constant_base_stock_reproduced_within_tolerance: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GijsRelativeVerificationSummary {
    pub source: &'static str,
    pub url: &'static str,
    pub repo_audit_replications: usize,
    pub seed: u64,
    pub rows: Vec<GijsRelativeVerificationRow>,
    pub mean_published_a3c_savings_pct: f64,
    pub mean_repo_gap_vs_published_constant_cost: f64,
    pub all_published_constant_base_stock_rows_reproduced_within_tolerance: bool,
    pub repo_generates_published_relative_rows: bool,
    pub can_mark_literature_verified: bool,
    pub verification_note: &'static str,
}

fn implied_target_cost_from_savings_pct(base_cost: f64, savings_pct: f64) -> f64 {
    base_cost * (1.0 - savings_pct / 100.0)
}

fn heuristic_config_from_reference(
    reference: &MultiEchelonReferenceInstance,
) -> PyResult<HeuristicSimulationConfig> {
    Ok(HeuristicSimulationConfig {
        warehouse_lead_time: reference.warehouse_lead_time,
        retailer_lead_time: reference.retailer_lead_time,
        num_retailers: reference.num_retailers,
        warehouse_holding_cost: reference.warehouse_holding_cost,
        retailer_holding_cost: reference.retailer_holding_cost,
        warehouse_expedited_cost: reference.warehouse_expedited_cost,
        warehouse_lost_sale_cost: reference.warehouse_lost_sale_cost,
        expedited_service_prob: reference.expedited_service_prob,
        warehouse_capacity: reference.warehouse_capacity,
        warehouse_inventory_cap: reference.warehouse_inventory_cap,
        retailer_inventory_cap: reference.retailer_inventory_cap,
        inventory_dynamics_mode: parse_inventory_dynamics_mode(reference.inventory_dynamics_mode)?,
        demand_distribution: parse_demand_distribution(reference.demand_distribution)?,
        demand_mean: reference.demand_mean,
        demand_std: reference.demand_std,
        horizon: reference.benchmark_periods,
        warm_up_periods_ratio: reference.warm_up_periods_ratio,
        discount_factor: 1.0,
        objective: parse_rollout_objective(reference.rollout_objective)?,
    })
}

fn published_constant_base_stock_levels(
    reference: &MultiEchelonReferenceInstance,
) -> PyResult<(usize, usize)> {
    if reference.published_constant_base_stock_levels.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "expected exactly two published constant base-stock levels for '{}'",
            reference.name
        )));
    }
    Ok((
        reference.published_constant_base_stock_levels[0],
        reference.published_constant_base_stock_levels[1],
    ))
}

pub fn gijs_relative_reference_instances() -> Vec<&'static MultiEchelonReferenceInstance> {
    LITERATURE_REFERENCE_INSTANCES
        .iter()
        .filter(|reference| reference.published_a3c_savings_pct.is_some())
        .collect()
}

pub fn evaluate_published_constant_base_stock_row(
    reference: &MultiEchelonReferenceInstance,
    replications: usize,
    seed: u64,
) -> PyResult<(f64, f64)> {
    let (warehouse_level, retailer_level) = published_constant_base_stock_levels(reference)?;
    let config = heuristic_config_from_reference(reference)?;
    let allocation_mode = parse_allocation_mode(reference.policy_allocation_mode)?;
    evaluate_stationary_policy(
        &config,
        &[warehouse_level],
        &[retailer_level],
        warehouse_level,
        retailer_level,
        StationaryPolicyKind::RegularBaseStock,
        allocation_mode,
        replications,
        seed,
    )
}

pub fn gijs_relative_verification_summary(
    repo_audit_replications: usize,
    seed: u64,
) -> PyResult<GijsRelativeVerificationSummary> {
    let rows = gijs_relative_reference_instances()
        .into_iter()
        .map(|reference| {
            let (warehouse_level, retailer_level) =
                published_constant_base_stock_levels(reference)?;
            let published_constant_base_stock_mean_cost = reference
                .published_constant_base_stock_mean_cost
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "missing published constant base-stock mean cost for '{}'",
                        reference.name
                    ))
                })?;
            let published_a3c_savings_pct =
                reference.published_a3c_savings_pct.ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "missing published A3C savings row for '{}'",
                        reference.name
                    ))
                })?;
            let published_a3c_confidence_half_width_pct = reference
                .published_a3c_confidence_half_width_pct
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "missing published A3C half-width row for '{}'",
                        reference.name
                    ))
                })?;
            let published_van_roy_savings_pct_approx = reference
                .published_van_roy_savings_pct_approx
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "missing published Van Roy savings approximation for '{}'",
                        reference.name
                    ))
                })?;
            let (repo_mean_cost, repo_cost_std) = evaluate_published_constant_base_stock_row(
                reference,
                repo_audit_replications,
                seed,
            )?;
            let repo_gap_vs_published_constant_cost =
                repo_mean_cost - published_constant_base_stock_mean_cost;
            let repo_gap_vs_published_constant_cost_pct = 100.0
                * repo_gap_vs_published_constant_cost
                / published_constant_base_stock_mean_cost;
            Ok(GijsRelativeVerificationRow {
                instance_name: reference.name,
                published_constant_base_stock_levels: vec![warehouse_level, retailer_level],
                published_constant_base_stock_mean_cost,
                published_a3c_savings_pct,
                published_a3c_confidence_half_width_pct,
                published_a3c_implied_mean_cost: implied_target_cost_from_savings_pct(
                    published_constant_base_stock_mean_cost,
                    published_a3c_savings_pct,
                ),
                published_van_roy_savings_pct_approx,
                published_van_roy_implied_mean_cost: implied_target_cost_from_savings_pct(
                    published_constant_base_stock_mean_cost,
                    published_van_roy_savings_pct_approx,
                ),
                repo_published_constant_base_stock_mean_cost: repo_mean_cost,
                repo_published_constant_base_stock_cost_std: repo_cost_std,
                repo_gap_vs_published_constant_cost,
                repo_gap_vs_published_constant_cost_pct,
                published_constant_base_stock_reproduced_within_tolerance:
                    repo_gap_vs_published_constant_cost_pct.abs()
                        <= PUBLISHED_CONSTANT_BASE_STOCK_RELATIVE_TOLERANCE_PCT,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mean_published_a3c_savings_pct = rows
        .iter()
        .map(|row| row.published_a3c_savings_pct)
        .sum::<f64>()
        / rows.len().max(1) as f64;
    let mean_repo_gap_vs_published_constant_cost = rows
        .iter()
        .map(|row| row.repo_gap_vs_published_constant_cost)
        .sum::<f64>()
        / rows.len().max(1) as f64;
    let all_published_constant_base_stock_rows_reproduced_within_tolerance = rows
        .iter()
        .all(|row| row.published_constant_base_stock_reproduced_within_tolerance);
    let repo_generates_published_relative_rows = false;

    Ok(GijsRelativeVerificationSummary {
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        repo_audit_replications,
        seed,
        rows,
        mean_published_a3c_savings_pct,
        mean_repo_gap_vs_published_constant_cost,
        all_published_constant_base_stock_rows_reproduced_within_tolerance,
        repo_generates_published_relative_rows,
        can_mark_literature_verified: all_published_constant_base_stock_rows_reproduced_within_tolerance
            && repo_generates_published_relative_rows,
        verification_note:
            "This summary freezes the carried Gijs relative rows and audits the repo heuristic at the published Van Roy levels. It does not by itself make the instance literature-verified because the repo does not yet generate the published A3C row, and the published constant base-stock rows are not both matched within tolerance.",
    })
}

#[cfg(test)]
mod tests;
