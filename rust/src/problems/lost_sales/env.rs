use rand::rngs::StdRng;
use crate::problems::lost_sales::demand::{sample_demand, LostSalesDemandProcess};

#[derive(Clone)]
pub struct LostSalesState {
    pub current_inventory: i64,
    pub lead_time_orders: Vec<usize>,
}

pub fn build_pipeline_state(
    current_inventory: i64,
    lead_time_orders: &[usize],
    demand_mean: f64,
) -> Vec<f32> {
    let scale = demand_mean.max(1.0) as f32;
    let mut state: Vec<f32> = lead_time_orders.iter().map(|&x| x as f32 / scale).collect();
    state[0] += (current_inventory.max(0) as f32) / scale;
    state
}

pub fn epoch_cost(
    current_inventory: &mut i64,
    demand: i64,
    order_quantity: usize,
    holding_cost: f64,
    shortage_cost: f64,
    procurement_cost: f64,
    fixed_order_cost: f64,
) -> f64 {
    let mut cost = procurement_cost * order_quantity as f64;
    if order_quantity > 0 {
        cost += fixed_order_cost;
    }

    if demand < *current_inventory {
        *current_inventory -= demand;
        cost += *current_inventory as f64 * holding_cost;
    } else {
        let lost_sales = demand - *current_inventory;
        *current_inventory = 0;
        cost += shortage_cost * lost_sales as f64;
    }

    cost
}

pub fn initialize_state(
    demand_mean: f64,
    lead_time: usize,
    rng: &mut StdRng,
    demand_process: &mut LostSalesDemandProcess,
) -> LostSalesState {
    let mut current_inventory = (2.0 * demand_mean).round() as i64;
    let mut lead_time_orders = vec![0usize; lead_time];
    let initial_order_quantity = demand_mean.max(0.0).round() as usize;

    for slot in lead_time_orders.iter_mut() {
        *slot = initial_order_quantity;
        let demand = sample_demand(rng, demand_process);
        current_inventory = (current_inventory - demand).max(0);
    }

    LostSalesState {
        current_inventory,
        lead_time_orders,
    }
}

#[cfg(test)]
mod tests {
    use super::build_pipeline_state;

    #[test]
    fn pipeline_state_is_scaled_by_mean_demand() {
        let state = build_pipeline_state(2, &[3, 4, 5], 5.0);
        assert_eq!(state, vec![1.0, 0.8, 1.0]);
    }

    #[test]
    fn pipeline_state_uses_minimum_unit_scale() {
        let state = build_pipeline_state(2, &[3, 4], 0.0);
        assert_eq!(state, vec![5.0, 4.0]);
    }

    #[test]
    fn pipeline_state_handles_very_large_orders_without_integer_overflow() {
        let state = build_pipeline_state(2, &[usize::MAX], 5.0);
        assert!(state[0].is_finite());
        assert!(state[0] > 0.0);
    }
}
