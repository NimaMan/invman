use crate::problems::perishable_inventory::env::{
    estimate_waste_during_lead_time, inventory_position, IssuingPolicy, PerishableState,
};

pub fn bsp_low_ew_order_quantity(
    state: &PerishableState,
    low_inventory_level: usize,
    high_inventory_level: usize,
    threshold: usize,
    max_order_size: usize,
    lead_time: usize,
    demand_mean: f64,
    issuing_policy: IssuingPolicy,
) -> usize {
    let current_position = inventory_position(state);
    let estimated_waste =
        estimate_waste_during_lead_time(state, lead_time, demand_mean, issuing_policy);
    let raw_order = if threshold > 0 && current_position < threshold {
        let alpha =
            1.0 - (high_inventory_level as f64 - low_inventory_level as f64) / threshold as f64;
        (low_inventory_level as f64 - alpha * current_position as f64 + estimated_waste).max(0.0)
    } else {
        (high_inventory_level as f64 - current_position as f64 + estimated_waste).max(0.0)
    };
    raw_order.round().clamp(0.0, max_order_size as f64) as usize
}
