use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::env::multi_echelon::{flattened_policy_state, initialize_state};
use crate::policies::soft_tree::{action_vector_from_flat_params, SoftTreeActionSpec, SoftTreeLeafType, SoftTreeSplitType};

#[derive(Clone)]
pub struct MultiEchelonRolloutConfig {
    pub input_dim: usize,
    pub depth: usize,
    pub action_spec: SoftTreeActionSpec,
    pub warehouse_lead_time: usize,
    pub retailer_lead_time: usize,
    pub num_retailers: usize,
    pub warehouse_holding_cost: f64,
    pub retailer_holding_cost: f64,
    pub warehouse_expedited_cost: f64,
    pub warehouse_lost_sale_cost: f64,
    pub expedited_service_prob: f64,
    pub warehouse_capacity: usize,
    pub warehouse_inventory_cap: usize,
    pub retailer_inventory_cap: usize,
    pub demand_mean: f64,
    pub demand_std: f64,
    pub horizon: usize,
    pub warm_up_periods_ratio: f64,
    pub temperature: f32,
    pub split_type: SoftTreeSplitType,
    pub leaf_type: SoftTreeLeafType,
}

fn mean_after_warmup(epoch_costs: &[f64], warm_up_periods_ratio: f64) -> f64 {
    let horizon = epoch_costs.len();
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let active_costs = if warm_up_periods < epoch_costs.len() {
        &epoch_costs[warm_up_periods..]
    } else {
        epoch_costs
    };
    active_costs.iter().sum::<f64>() / active_costs.len() as f64
}

pub fn rollout(
    flat_params: &[f32],
    config: &MultiEchelonRolloutConfig,
    seed: u64,
    warehouse_levels: &[usize],
    retailer_levels: &[usize],
) -> PyResult<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let demand_dist = Normal::new(config.demand_mean, config.demand_std.max(1e-6))
        .map_err(|err| PyValueError::new_err(format!("invalid demand parameters: {err}")))?;
    let mut state = initialize_state(
        config.warehouse_lead_time,
        config.retailer_lead_time,
        config.num_retailers,
        warehouse_levels,
        retailer_levels,
        config.demand_mean,
        seed,
    );
    let mut epoch_costs = Vec::with_capacity(config.horizon);

    for _ in 0..config.horizon {
        let policy_state = flattened_policy_state(&state, config.warehouse_inventory_cap, config.retailer_inventory_cap);
        if policy_state.len() != config.input_dim {
            return Err(PyValueError::new_err("policy state length does not match input_dim"));
        }
        let action = action_vector_from_flat_params(
            &policy_state,
            flat_params,
            config.input_dim,
            config.depth,
            config.temperature,
            config.split_type,
            config.leaf_type,
            &config.action_spec,
        )?;
        let warehouse_target = action[0].min(config.warehouse_inventory_cap);
        let retailer_target = action[1].min(config.retailer_inventory_cap);

        let warehouse_available = state.warehouse_inventory + state.warehouse_pipeline[0] as i64;
        let mut retailer_available = state.retailer_inventory.clone();
        for retailer_idx in 0..config.num_retailers {
            retailer_available[retailer_idx] += state.retailer_pipeline[retailer_idx][0] as i64;
        }
        let warehouse_future = state.warehouse_pipeline.iter().copied().skip(1).collect::<Vec<_>>();
        let retailer_future = state
            .retailer_pipeline
            .iter()
            .map(|row| row.iter().copied().skip(1).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let warehouse_ip = warehouse_available + warehouse_future.iter().copied().sum::<usize>() as i64;
        let warehouse_order = warehouse_target
            .saturating_sub(warehouse_ip.max(0) as usize)
            .min(config.warehouse_capacity);

        let retailer_ip = retailer_available
            .iter()
            .enumerate()
            .map(|(idx, inventory)| *inventory + retailer_future[idx].iter().copied().sum::<usize>() as i64)
            .collect::<Vec<_>>();

        let mut desired_retail_orders = vec![0usize; config.num_retailers];
        for retailer_idx in 0..config.num_retailers {
            desired_retail_orders[retailer_idx] = retailer_target.saturating_sub(retailer_ip[retailer_idx].max(0) as usize);
        }

        let mut remaining_warehouse_inventory = warehouse_available.max(0) as usize;
        let mut shipped_retail_orders = vec![0usize; config.num_retailers];
        for retailer_idx in 0..config.num_retailers {
            let shipped = desired_retail_orders[retailer_idx].min(remaining_warehouse_inventory);
            shipped_retail_orders[retailer_idx] = shipped;
            remaining_warehouse_inventory -= shipped;
        }

        state.warehouse_pipeline = warehouse_future;
        state.warehouse_pipeline.push(warehouse_order);
        state.retailer_pipeline = retailer_future;
        for retailer_idx in 0..config.num_retailers {
            state.retailer_pipeline[retailer_idx].push(shipped_retail_orders[retailer_idx]);
        }

        let mut retailer_end_inventory = vec![0i64; config.num_retailers];
        let mut total_accepted = 0usize;
        let mut lost_at_retailer = 0usize;
        for retailer_idx in 0..config.num_retailers {
            let demand = demand_dist.sample(&mut rng).round().max(0.0) as usize;
            let served = (retailer_available[retailer_idx].max(0) as usize).min(demand);
            let unmet = demand - served;
            retailer_end_inventory[retailer_idx] = retailer_available[retailer_idx] - served as i64;
            let mut accepted = 0usize;
            for _ in 0..unmet {
                if rng.gen::<f64>() < config.expedited_service_prob {
                    accepted += 1;
                }
            }
            total_accepted += accepted;
            lost_at_retailer += unmet - accepted;
        }

        let expedited_shipped = total_accepted.min(remaining_warehouse_inventory);
        remaining_warehouse_inventory -= expedited_shipped;
        let lost_at_warehouse = total_accepted - expedited_shipped;

        state.warehouse_inventory = remaining_warehouse_inventory as i64;
        state.retailer_inventory = retailer_end_inventory;

        epoch_costs.push(
            config.warehouse_holding_cost * state.warehouse_inventory.max(0) as f64
                + config.retailer_holding_cost
                    * state.retailer_inventory.iter().copied().map(|value| value.max(0) as f64).sum::<f64>()
                + config.warehouse_expedited_cost * expedited_shipped as f64
                + config.warehouse_lost_sale_cost * (lost_at_retailer + lost_at_warehouse) as f64,
        );
    }

    Ok(mean_after_warmup(&epoch_costs, config.warm_up_periods_ratio))
}

pub fn population_rollout(
    params_batch: &[Vec<f32>],
    config: &MultiEchelonRolloutConfig,
    seeds: &[u64],
    warehouse_levels: &[usize],
    retailer_levels: &[usize],
) -> PyResult<Vec<f64>> {
    if params_batch.len() != seeds.len() {
        return Err(PyValueError::new_err("params batch size must match seeds size"));
    }
    let mut costs = Vec::with_capacity(params_batch.len());
    for (params, seed) in params_batch.iter().zip(seeds.iter().copied()) {
        costs.push(rollout(params, config, seed, warehouse_levels, retailer_levels)?);
    }
    Ok(costs)
}
