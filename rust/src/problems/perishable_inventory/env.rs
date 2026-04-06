use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IssuingPolicy {
    Fifo,
    Lifo,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PerishableState {
    pub on_hand: Vec<usize>,
    pub pipeline_orders: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PerishableStepOutcome {
    pub next_state: PerishableState,
    pub shortage: usize,
    pub waste: usize,
    pub holding_inventory: usize,
    pub cost: f64,
}

pub fn parse_issuing_policy(issuing_policy: &str) -> PyResult<IssuingPolicy> {
    match issuing_policy {
        "fifo" | "FIFO" => Ok(IssuingPolicy::Fifo),
        "lifo" | "LIFO" => Ok(IssuingPolicy::Lifo),
        _ => Err(PyValueError::new_err(format!(
            "unknown issuing policy '{issuing_policy}'; expected 'fifo' or 'lifo'"
        ))),
    }
}

pub fn validate_state(
    state: &PerishableState,
    shelf_life: usize,
    lead_time: usize,
) -> PyResult<()> {
    if shelf_life < 1 {
        return Err(PyValueError::new_err("shelf_life must be at least 1"));
    }
    if lead_time < 1 {
        return Err(PyValueError::new_err("lead_time must be at least 1"));
    }
    if state.on_hand.len() != shelf_life {
        return Err(PyValueError::new_err(format!(
            "on_hand length {} does not match shelf_life {}",
            state.on_hand.len(),
            shelf_life
        )));
    }
    if state.pipeline_orders.len() != lead_time.saturating_sub(1) {
        return Err(PyValueError::new_err(format!(
            "pipeline_orders length {} does not match lead_time - 1 {}",
            state.pipeline_orders.len(),
            lead_time.saturating_sub(1)
        )));
    }
    Ok(())
}

pub fn initialize_state(_demand_mean: f64, shelf_life: usize, lead_time: usize) -> PerishableState {
    PerishableState {
        on_hand: vec![0usize; shelf_life],
        pipeline_orders: vec![0usize; lead_time.saturating_sub(1)],
    }
}

pub fn build_raw_state(state: &PerishableState) -> Vec<f32> {
    let mut raw_state = state
        .pipeline_orders
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    raw_state.extend(state.on_hand.iter().map(|value| *value as f32));
    raw_state
}

pub fn inventory_position(state: &PerishableState) -> usize {
    state.pipeline_orders.iter().copied().sum::<usize>()
        + state.on_hand.iter().copied().sum::<usize>()
}

fn apply_demand_to_inventory(
    remaining_inventory: &mut [usize],
    demand: usize,
    issuing_policy: IssuingPolicy,
) -> usize {
    let mut unmet = demand;
    match issuing_policy {
        IssuingPolicy::Fifo => {
            for idx in (0..remaining_inventory.len()).rev() {
                let served = remaining_inventory[idx].min(unmet);
                remaining_inventory[idx] -= served;
                unmet -= served;
                if unmet == 0 {
                    break;
                }
            }
        }
        IssuingPolicy::Lifo => {
            for value in remaining_inventory.iter_mut() {
                let served = (*value).min(unmet);
                *value -= served;
                unmet -= served;
                if unmet == 0 {
                    break;
                }
            }
        }
    }
    unmet
}

fn next_pipeline_and_arrival<T: Copy + Default>(
    pipeline_orders: &[T],
    new_order: T,
) -> (Vec<T>, T) {
    if pipeline_orders.is_empty() {
        return (Vec::new(), new_order);
    }

    let arrival = *pipeline_orders
        .last()
        .expect("non-empty pipeline must have a final element");
    let mut next_pipeline = Vec::with_capacity(pipeline_orders.len());
    next_pipeline.push(new_order);
    next_pipeline.extend(
        pipeline_orders
            .iter()
            .copied()
            .take(pipeline_orders.len().saturating_sub(1)),
    );
    (next_pipeline, arrival)
}

pub fn step_state(
    state: &PerishableState,
    order_quantity: usize,
    demand: usize,
    holding_cost: f64,
    shortage_cost: f64,
    waste_cost: f64,
    procurement_cost: f64,
    issuing_policy: IssuingPolicy,
) -> PerishableStepOutcome {
    let opening_inventory = state.on_hand.iter().copied().sum::<usize>();
    let mut remaining_inventory = state.on_hand.clone();
    let shortage = apply_demand_to_inventory(&mut remaining_inventory, demand, issuing_policy);
    debug_assert_eq!(shortage, demand.saturating_sub(opening_inventory));

    let waste = *remaining_inventory.last().unwrap_or(&0usize);
    let holding_inventory = remaining_inventory
        .iter()
        .take(remaining_inventory.len().saturating_sub(1))
        .copied()
        .sum::<usize>();

    let (next_pipeline, arrival) =
        next_pipeline_and_arrival(&state.pipeline_orders, order_quantity);
    let mut next_on_hand = Vec::with_capacity(state.on_hand.len());
    if !state.on_hand.is_empty() {
        next_on_hand.push(arrival);
        next_on_hand.extend(
            remaining_inventory
                .iter()
                .copied()
                .take(state.on_hand.len().saturating_sub(1)),
        );
    }

    let cost = procurement_cost * order_quantity as f64
        + holding_cost * holding_inventory as f64
        + shortage_cost * shortage as f64
        + waste_cost * waste as f64;

    PerishableStepOutcome {
        next_state: PerishableState {
            on_hand: next_on_hand,
            pipeline_orders: next_pipeline,
        },
        shortage,
        waste,
        holding_inventory,
        cost,
    }
}

fn apply_fractional_demand_to_inventory(
    remaining_inventory: &mut [f64],
    demand: f64,
    issuing_policy: IssuingPolicy,
) {
    let mut unmet = demand.max(0.0);
    match issuing_policy {
        IssuingPolicy::Fifo => {
            for idx in (0..remaining_inventory.len()).rev() {
                let served = remaining_inventory[idx].min(unmet);
                remaining_inventory[idx] -= served;
                unmet -= served;
                if unmet <= 1e-9 {
                    break;
                }
            }
        }
        IssuingPolicy::Lifo => {
            for value in remaining_inventory.iter_mut() {
                let served = (*value).min(unmet);
                *value -= served;
                unmet -= served;
                if unmet <= 1e-9 {
                    break;
                }
            }
        }
    }
}

pub fn estimate_waste_during_lead_time(
    state: &PerishableState,
    lead_time: usize,
    demand_mean: f64,
    issuing_policy: IssuingPolicy,
) -> f64 {
    let mut on_hand = state
        .on_hand
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let mut pipeline = state
        .pipeline_orders
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let mut estimated_waste = 0.0;

    for _ in 0..lead_time {
        let mut remaining_inventory = on_hand.clone();
        apply_fractional_demand_to_inventory(&mut remaining_inventory, demand_mean, issuing_policy);
        estimated_waste += *remaining_inventory.last().unwrap_or(&0.0);

        let (next_pipeline, arrival) = next_pipeline_and_arrival(&pipeline, 0.0);
        let mut next_on_hand = Vec::with_capacity(on_hand.len());
        if !on_hand.is_empty() {
            next_on_hand.push(arrival);
            next_on_hand.extend(
                remaining_inventory
                    .iter()
                    .copied()
                    .take(on_hand.len().saturating_sub(1)),
            );
        }

        on_hand = next_on_hand;
        pipeline = next_pipeline;
    }

    estimated_waste
}
