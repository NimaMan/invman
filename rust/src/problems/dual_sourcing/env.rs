use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

#[derive(Clone, Debug)]
pub struct DualSourcingState {
    pub reduced_state: Vec<i64>,
}

pub fn initialize_state(
    regular_lead_time: usize,
    regular_max_order_size: usize,
    demand_low: usize,
    demand_high: usize,
    seed: u64,
) -> DualSourcingState {
    let mut rng = StdRng::seed_from_u64(seed);
    let mean_demand = 0.5 * (demand_low + demand_high) as f64;
    let mut reduced_state = vec![((regular_lead_time + 1) as f64 * mean_demand).round() as i64];
    for _ in 0..regular_lead_time.saturating_sub(1) {
        reduced_state.push(rng.gen_range(0..=regular_max_order_size) as i64);
    }
    DualSourcingState { reduced_state }
}

pub fn validate_action(
    regular_order: usize,
    expedited_order: usize,
    max_regular: usize,
    max_expedited: usize,
) -> PyResult<()> {
    if regular_order > max_regular {
        return Err(PyValueError::new_err(format!(
            "regular order {regular_order} exceeds max_regular {max_regular}"
        )));
    }
    if expedited_order > max_expedited {
        return Err(PyValueError::new_err(format!(
            "expedited order {expedited_order} exceeds max_expedited {max_expedited}"
        )));
    }
    Ok(())
}

pub fn step_state(
    reduced_state: &[i64],
    regular_order: usize,
    expedited_order: usize,
    demand: usize,
) -> Vec<i64> {
    if reduced_state.len() == 1 {
        return vec![
            reduced_state[0] + expedited_order as i64 - demand as i64 + regular_order as i64,
        ];
    }
    let end_inventory = reduced_state[0] + expedited_order as i64 - demand as i64;
    let mut next_state = Vec::with_capacity(reduced_state.len());
    next_state.push(end_inventory + reduced_state[1]);
    for value in reduced_state.iter().copied().skip(2) {
        next_state.push(value);
    }
    next_state.push(regular_order as i64);
    next_state
}

pub fn epoch_cost(
    reduced_state: &[i64],
    regular_order: usize,
    expedited_order: usize,
    demand: usize,
    regular_order_cost: f64,
    expedited_order_cost: f64,
    holding_cost: f64,
    shortage_cost: f64,
) -> f64 {
    let end_inventory = reduced_state[0] + expedited_order as i64 - demand as i64;
    regular_order_cost * regular_order as f64
        + expedited_order_cost * expedited_order as f64
        + holding_cost * (end_inventory.max(0) as f64)
        + shortage_cost * ((-end_inventory).max(0) as f64)
}
