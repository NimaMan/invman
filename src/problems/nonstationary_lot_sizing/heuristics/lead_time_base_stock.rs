use statrs::distribution::{ContinuousCDF, Normal};

use crate::problems::nonstationary_lot_sizing::demand::{demand_std, DemandDistributionKind};
use crate::problems::nonstationary_lot_sizing::env::{
    inventory_position, NonstationaryLotSizingState,
};

pub fn lead_time_demand_moments(
    forecast_window: &[f64],
    lead_time: usize,
    demand_cv: f64,
    demand_kind: DemandDistributionKind,
) -> (f64, f64) {
    let coverage = (lead_time + 1).min(forecast_window.len());
    let relevant = &forecast_window[..coverage];
    let mean = relevant.iter().sum::<f64>();
    let variance = relevant
        .iter()
        .map(|period_mean| demand_std(*period_mean, demand_cv, demand_kind).powi(2))
        .sum::<f64>();
    (mean, variance.sqrt())
}

pub fn lead_time_base_stock_level(
    forecast_window: &[f64],
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    demand_cv: f64,
    demand_kind: DemandDistributionKind,
) -> f64 {
    let (mean, std) = lead_time_demand_moments(forecast_window, lead_time, demand_cv, demand_kind);
    if shortage_cost <= 0.0 {
        return mean.max(0.0);
    }
    if std <= 0.0 {
        return mean.max(0.0);
    }
    let critical_ratio =
        (shortage_cost / (shortage_cost + holding_cost.max(1e-9))).clamp(1e-9, 1.0 - 1e-9);
    let dist = Normal::new(mean, std).expect("positive std must define a Normal distribution");
    dist.inverse_cdf(critical_ratio).max(0.0)
}

pub fn lead_time_base_stock_order_quantity(
    state: &NonstationaryLotSizingState,
    holding_cost: f64,
    shortage_cost: f64,
    demand_cv: f64,
    demand_kind: DemandDistributionKind,
) -> f64 {
    let level = lead_time_base_stock_level(
        &state.forecast_window,
        state.pipeline_orders.len(),
        holding_cost,
        shortage_cost,
        demand_cv,
        demand_kind,
    );
    (level - inventory_position(state)).max(0.0)
}
