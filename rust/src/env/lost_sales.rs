use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution, Poisson};

#[derive(Clone)]
pub struct LostSalesState {
    pub current_inventory: i64,
    pub lead_time_orders: Vec<usize>,
}

pub fn build_pipeline_state(
    current_inventory: i64,
    lead_time_orders: &[usize],
    max_order_size: usize,
) -> Vec<f32> {
    let mut state = lead_time_orders.to_vec();
    state[0] += current_inventory as usize;
    let scale = max_order_size.max(1) as f32;
    state.into_iter().map(|x| x as f32 / scale).collect()
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
    demand_rate: f64,
    lead_time: usize,
    max_order_size: usize,
    rng: &mut StdRng,
    demand_dist: &Poisson<f64>,
) -> LostSalesState {
    let mut current_inventory = (2.0 * demand_rate).round() as i64;
    let mut lead_time_orders = vec![0usize; lead_time];

    for slot in lead_time_orders.iter_mut() {
        *slot = if max_order_size == 0 {
            0
        } else {
            rng.gen_range(1..=max_order_size)
        };
        let demand = demand_dist.sample(rng) as i64;
        current_inventory = (current_inventory - demand).max(0);
    }

    LostSalesState {
        current_inventory,
        lead_time_orders,
    }
}
