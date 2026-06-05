use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::SeedableRng;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::problems::vendor_managed_inventory::env::{
    initialize_paper_state, paper_model_param, paper_signal_multiplier, step_paper_state,
    PaperVendorManagedInventoryModel, PaperVendorManagedInventoryState, UniformTimeDistribution,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaperPolicySimulationSummary {
    pub mean_profit_per_unit_time: f64,
    pub std_profit_per_unit_time: f64,
}

fn uniform_mean(distribution: UniformTimeDistribution) -> f64 {
    0.5 * (distribution.low + distribution.high)
}

fn uniform_variance(distribution: UniformTimeDistribution) -> f64 {
    (distribution.high - distribution.low).powi(2) / 12.0
}

fn route_cycle_time_moments(model: &PaperVendorManagedInventoryModel) -> (f64, f64) {
    let mut mean = uniform_mean(model.dc_service_time)
        + uniform_mean(model.dc_to_first_retailer_time)
        + uniform_mean(model.last_retailer_to_dc_time);
    let mut variance = uniform_variance(model.dc_service_time)
        + uniform_variance(model.dc_to_first_retailer_time)
        + uniform_variance(model.last_retailer_to_dc_time);
    for _ in 0..model.num_retailers {
        mean += uniform_mean(model.retailer_service_time);
        variance += uniform_variance(model.retailer_service_time);
    }
    for _ in 1..model.num_retailers {
        mean += uniform_mean(model.retailer_to_retailer_time);
        variance += uniform_variance(model.retailer_to_retailer_time);
    }
    (mean, variance)
}

fn retailer_lead_time_moments(
    model: &PaperVendorManagedInventoryModel,
    retailer_index: usize,
) -> (f64, f64) {
    let mut mean =
        uniform_mean(model.dc_service_time) + uniform_mean(model.dc_to_first_retailer_time);
    let mut variance =
        uniform_variance(model.dc_service_time) + uniform_variance(model.dc_to_first_retailer_time);
    for _ in 0..retailer_index {
        mean += uniform_mean(model.retailer_service_time);
        variance += uniform_variance(model.retailer_service_time);
        mean += uniform_mean(model.retailer_to_retailer_time);
        variance += uniform_variance(model.retailer_to_retailer_time);
    }
    (mean, variance)
}

pub fn paper_newsvendor_order_up_to_levels(
    model: &PaperVendorManagedInventoryModel,
    state: &PaperVendorManagedInventoryState,
) -> PyResult<Vec<Vec<f64>>> {
    let (cycle_mean, cycle_variance) = route_cycle_time_moments(model);
    let mut order_up_to = vec![vec![0.0; model.num_products]; model.num_retailers];
    let standard_normal = Normal::new(0.0, 1.0).expect("standard normal must build");

    for (retailer, row) in order_up_to.iter_mut().enumerate() {
        let (lead_mean, lead_variance) = retailer_lead_time_moments(model, retailer);
        for (product, level) in row.iter_mut().enumerate() {
            let param = paper_model_param(model, retailer, product);
            let signal_multiplier =
                paper_signal_multiplier(state.demand_signal_high[retailer][product], model);
            let demand_mean_unit = signal_multiplier
                * param.arrival_rate
                * 0.5
                * (param.demand_low + param.demand_high);
            let demand_variance_unit = signal_multiplier.powi(2)
                * (param.arrival_rate * (param.demand_high - param.demand_low).powi(2) / 12.0
                    + param.arrival_rate * (0.5 * (param.demand_low + param.demand_high)).powi(2));
            let cycle_demand_mean = demand_mean_unit * cycle_mean;
            let cycle_demand_variance =
                cycle_mean * demand_variance_unit + demand_mean_unit.powi(2) * cycle_variance;

            let next_mean_unit = model.expected_signal_multiplier
                * param.arrival_rate
                * 0.5
                * (param.demand_low + param.demand_high);
            let next_variance_unit = model.expected_signal_multiplier.powi(2)
                * (param.arrival_rate * (param.demand_high - param.demand_low).powi(2) / 12.0
                    + param.arrival_rate * (0.5 * (param.demand_low + param.demand_high)).powi(2));
            let lead_demand_mean = next_mean_unit * lead_mean;
            let lead_demand_variance =
                lead_mean * next_variance_unit + next_mean_unit.powi(2) * lead_variance;

            let total_mean = cycle_demand_mean + lead_demand_mean;
            let total_stddev = (cycle_demand_variance + lead_demand_variance)
                .max(0.0)
                .sqrt();
            let critical_ratio = param.retailer_stockout_cost_per_unit
                / (param.retailer_stockout_cost_per_unit
                    + param.retailer_holding_cost_per_unit_time);
            let z = standard_normal.inverse_cdf(critical_ratio.clamp(1e-9, 1.0 - 1e-9));
            *level = (total_mean + z * total_stddev).max(0.0);
        }
    }

    Ok(order_up_to)
}

pub fn paper_mean_demand_order_up_to_levels(
    model: &PaperVendorManagedInventoryModel,
    state: &PaperVendorManagedInventoryState,
) -> PyResult<Vec<Vec<f64>>> {
    let (cycle_mean, cycle_variance) = route_cycle_time_moments(model);
    let mut order_up_to = vec![vec![0.0; model.num_products]; model.num_retailers];

    for (retailer, row) in order_up_to.iter_mut().enumerate() {
        let (lead_mean, _lead_variance) = retailer_lead_time_moments(model, retailer);
        let _ = cycle_variance;
        for (product, level) in row.iter_mut().enumerate() {
            let param = paper_model_param(model, retailer, product);
            let signal_multiplier =
                paper_signal_multiplier(state.demand_signal_high[retailer][product], model);
            let demand_mean_unit = signal_multiplier
                * param.arrival_rate
                * 0.5
                * (param.demand_low + param.demand_high);
            let next_mean_unit = model.expected_signal_multiplier
                * param.arrival_rate
                * 0.5
                * (param.demand_low + param.demand_high);
            *level = (demand_mean_unit * cycle_mean + next_mean_unit * lead_mean).max(0.0);
        }
    }

    Ok(order_up_to)
}

fn total_inventory(state: &PaperVendorManagedInventoryState) -> f64 {
    state
        .retailer_inventory
        .iter()
        .flat_map(|row| row.iter())
        .sum::<f64>()
}

pub fn paper_allocate_with_trucks(
    model: &PaperVendorManagedInventoryModel,
    state: &PaperVendorManagedInventoryState,
    order_up_to_levels: &[Vec<f64>],
    trucks_dispatched: usize,
) -> PyResult<Vec<Vec<f64>>> {
    if order_up_to_levels.len() != model.num_retailers
        || order_up_to_levels
            .iter()
            .any(|row| row.len() != model.num_products)
    {
        return Err(PyValueError::new_err(
            "order_up_to_levels shape must match [num_retailers][num_products]",
        ));
    }
    if trucks_dispatched > model.max_trucks {
        return Err(PyValueError::new_err(format!(
            "trucks_dispatched {} cannot exceed max_trucks {}",
            trucks_dispatched, model.max_trucks
        )));
    }
    if trucks_dispatched == 0 {
        return Ok(vec![vec![0.0; model.num_products]; model.num_retailers]);
    }

    let total_retailer_inventory = total_inventory(state);
    let total_target = order_up_to_levels
        .iter()
        .flat_map(|row| row.iter())
        .sum::<f64>()
        .max(1e-9);

    let mut desired = vec![vec![0.0; model.num_products]; model.num_retailers];
    for retailer in 0..model.num_retailers {
        for product in 0..model.num_products {
            let y = (((total_retailer_inventory
                + trucks_dispatched as f64 * model.truck_capacity)
                * order_up_to_levels[retailer][product]
                / total_target)
                .round())
                - state.retailer_inventory[retailer][product];
            desired[retailer][product] = y.max(0.0);
        }
    }

    let desired_total = desired.iter().flat_map(|row| row.iter()).sum::<f64>();
    let capacity_scale = if desired_total > 0.0 {
        (trucks_dispatched as f64 * model.truck_capacity / desired_total).min(1.0)
    } else {
        1.0
    };

    let mut product_scales = vec![1.0; model.num_products];
    for product in 0..model.num_products {
        let product_desired = desired.iter().map(|row| row[product]).sum::<f64>();
        if product_desired > 0.0 {
            product_scales[product] = (state.dc_inventory[product] / product_desired).min(1.0);
        }
    }

    let mut dispatched = vec![vec![0.0; model.num_products]; model.num_retailers];
    for retailer in 0..model.num_retailers {
        for product in 0..model.num_products {
            let quantity = desired[retailer][product] * capacity_scale * product_scales[product];
            dispatched[retailer][product] = quantity;
        }
    }

    Ok(dispatched)
}

pub fn paper_allocate_from_order_up_to_levels(
    model: &PaperVendorManagedInventoryModel,
    state: &PaperVendorManagedInventoryState,
    order_up_to_levels: &[Vec<f64>],
) -> PyResult<(usize, Vec<Vec<f64>>)> {
    if order_up_to_levels.len() != model.num_retailers
        || order_up_to_levels
            .iter()
            .any(|row| row.len() != model.num_products)
    {
        return Err(PyValueError::new_err(
            "order_up_to_levels shape must match [num_retailers][num_products]",
        ));
    }

    let mut total_gap = 0.0;
    for retailer in 0..model.num_retailers {
        for product in 0..model.num_products {
            total_gap += (order_up_to_levels[retailer][product]
                - state.retailer_inventory[retailer][product])
                .max(0.0);
        }
    }
    let trucks_dispatched = ((total_gap / model.truck_capacity).round() as isize)
        .clamp(0, model.max_trucks as isize) as usize;
    let dispatched =
        paper_allocate_with_trucks(model, state, order_up_to_levels, trucks_dispatched)?;
    Ok((trucks_dispatched, dispatched))
}

pub fn paper_newsvendor_dispatch(
    model: &PaperVendorManagedInventoryModel,
    state: &PaperVendorManagedInventoryState,
) -> PyResult<(usize, Vec<Vec<f64>>)> {
    let order_up_to_levels = paper_newsvendor_order_up_to_levels(model, state)?;
    paper_allocate_from_order_up_to_levels(model, state, &order_up_to_levels)
}

pub fn paper_mean_demand_dispatch(
    model: &PaperVendorManagedInventoryModel,
    state: &PaperVendorManagedInventoryState,
) -> PyResult<(usize, Vec<Vec<f64>>)> {
    let order_up_to_levels = paper_mean_demand_order_up_to_levels(model, state)?;
    paper_allocate_from_order_up_to_levels(model, state, &order_up_to_levels)
}

pub fn simulate_paper_policy(
    model: &PaperVendorManagedInventoryModel,
    policy_name: &str,
    replications: usize,
    warmup_time: f64,
    evaluation_time: f64,
    seed: u64,
) -> PyResult<PaperPolicySimulationSummary> {
    if replications == 0 {
        return Err(PyValueError::new_err("replications must be at least 1"));
    }
    if !warmup_time.is_finite() || warmup_time < 0.0 {
        return Err(PyValueError::new_err(
            "warmup_time must be finite and non-negative",
        ));
    }
    if !evaluation_time.is_finite() || evaluation_time <= 0.0 {
        return Err(PyValueError::new_err(
            "evaluation_time must be finite and strictly positive",
        ));
    }

    let mut replication_values = Vec::with_capacity(replications);
    for replication in 0..replications {
        let mut rng = StdRng::seed_from_u64(seed + replication as u64);
        let mut state = initialize_paper_state(model, &mut rng)?;
        let mut elapsed = 0.0;
        let mut measured_time = 0.0;
        let mut measured_profit = 0.0;

        while measured_time < evaluation_time {
            let (trucks_dispatched, dispatch_quantities) = match policy_name {
                "paper_newsvendor" => paper_newsvendor_dispatch(model, &state)?,
                "paper_mean_demand" => paper_mean_demand_dispatch(model, &state)?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported paper policy '{}'",
                        policy_name
                    )))
                }
            };
            let outcome = step_paper_state(
                model,
                &state,
                trucks_dispatched,
                &dispatch_quantities,
                &mut rng,
            )?;
            if elapsed >= warmup_time {
                measured_profit += outcome.cycle_profit;
                measured_time += outcome.route_cycle_time;
            }
            elapsed += outcome.route_cycle_time;
            state = outcome.next_state;
        }

        replication_values.push(measured_profit / measured_time.max(1e-9));
    }

    let mean = replication_values.iter().sum::<f64>() / replication_values.len() as f64;
    let variance = replication_values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / replication_values.len() as f64;
    Ok(PaperPolicySimulationSummary {
        mean_profit_per_unit_time: mean,
        std_profit_per_unit_time: variance.sqrt(),
    })
}
