use crate::problems::nonstationary_lot_sizing::demand::DemandDistributionKind;
use crate::problems::nonstationary_lot_sizing::env::{
    inventory_position, NonstationaryLotSizingState,
};
use crate::problems::nonstationary_lot_sizing::heuristics::lead_time_base_stock_level;

pub fn simple_s_s_levels(
    forecast_window: &[f64],
    lead_time: usize,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    demand_cv: f64,
    demand_kind: DemandDistributionKind,
) -> (f64, f64) {
    let s = lead_time_base_stock_level(
        forecast_window,
        lead_time,
        holding_cost,
        shortage_cost,
        demand_cv,
        demand_kind,
    );
    let mean_forecast = if forecast_window.is_empty() {
        0.0
    } else {
        forecast_window.iter().sum::<f64>() / forecast_window.len() as f64
    };
    let eoq = if holding_cost <= 0.0 {
        0.0
    } else {
        (2.0 * mean_forecast.max(0.0) * fixed_order_cost.max(0.0) / holding_cost).sqrt()
    };
    (s, s + eoq)
}

pub fn s_s_order_quantity(inventory_position: f64, s: f64, s_up_to: f64) -> f64 {
    if inventory_position > s {
        0.0
    } else {
        (s_up_to - inventory_position).max(0.0)
    }
}

pub fn simple_s_s_order_quantity(
    state: &NonstationaryLotSizingState,
    holding_cost: f64,
    shortage_cost: f64,
    fixed_order_cost: f64,
    demand_cv: f64,
    demand_kind: DemandDistributionKind,
) -> f64 {
    let (s, s_up_to) = simple_s_s_levels(
        &state.forecast_window,
        state.pipeline_orders.len(),
        holding_cost,
        shortage_cost,
        fixed_order_cost,
        demand_cv,
        demand_kind,
    );
    s_s_order_quantity(inventory_position(state), s, s_up_to)
}
