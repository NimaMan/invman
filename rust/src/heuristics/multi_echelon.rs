use pyo3::PyResult;

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

fn rollout_constant_base_stock_from_demands(
    warehouse_level: usize,
    retailer_level: usize,
    warehouse_inventory: i64,
    warehouse_pipeline: &[usize],
    retailer_inventory: &[i64],
    retailer_pipeline: &[Vec<usize>],
    demands: &[Vec<usize>],
    expedite_uniforms: &[Vec<Vec<f64>>],
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    expedited_service_prob: f64,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    warm_up_periods_ratio: f64,
) -> f64 {
    let mut warehouse_inventory = warehouse_inventory;
    let mut warehouse_pipeline = warehouse_pipeline.to_vec();
    let mut retailer_inventory = retailer_inventory.to_vec();
    let mut retailer_pipeline = retailer_pipeline.to_vec();
    let num_retailers = retailer_inventory.len();
    let mut epoch_costs = Vec::with_capacity(demands.len());

    for (period_idx, demand_vector) in demands.iter().enumerate() {
        let warehouse_available = warehouse_inventory + warehouse_pipeline[0] as i64;
        let mut retailer_available = retailer_inventory.clone();
        for retailer_idx in 0..num_retailers {
            retailer_available[retailer_idx] += retailer_pipeline[retailer_idx][0] as i64;
        }
        let warehouse_future = warehouse_pipeline.iter().copied().skip(1).collect::<Vec<_>>();
        let retailer_future = retailer_pipeline
            .iter()
            .map(|row| row.iter().copied().skip(1).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let warehouse_ip = warehouse_available + warehouse_future.iter().copied().sum::<usize>() as i64;
        let warehouse_order = warehouse_level
            .min(warehouse_inventory_cap)
            .saturating_sub(warehouse_ip.max(0) as usize)
            .min(warehouse_capacity);

        let retailer_ip = retailer_available
            .iter()
            .enumerate()
            .map(|(idx, inventory)| *inventory + retailer_future[idx].iter().copied().sum::<usize>() as i64)
            .collect::<Vec<_>>();
        let mut desired_orders = vec![0usize; num_retailers];
        for retailer_idx in 0..num_retailers {
            desired_orders[retailer_idx] = retailer_level
                .min(retailer_inventory_cap)
                .saturating_sub(retailer_ip[retailer_idx].max(0) as usize);
        }

        let mut remaining_warehouse_inventory = warehouse_available.max(0) as usize;
        let mut shipped_orders = vec![0usize; num_retailers];
        for retailer_idx in 0..num_retailers {
            let shipped = desired_orders[retailer_idx].min(remaining_warehouse_inventory);
            shipped_orders[retailer_idx] = shipped;
            remaining_warehouse_inventory -= shipped;
        }

        warehouse_pipeline = warehouse_future;
        warehouse_pipeline.push(warehouse_order);
        retailer_pipeline = retailer_future;
        for retailer_idx in 0..num_retailers {
            retailer_pipeline[retailer_idx].push(shipped_orders[retailer_idx]);
        }

        let mut retailer_end_inventory = vec![0i64; num_retailers];
        let mut total_accepted = 0usize;
        let mut lost_at_retailer = 0usize;
        for retailer_idx in 0..num_retailers {
            let demand = demand_vector[retailer_idx];
            let served = (retailer_available[retailer_idx].max(0) as usize).min(demand);
            let unmet = demand - served;
            retailer_end_inventory[retailer_idx] = retailer_available[retailer_idx] - served as i64;
            let accepted = expedite_uniforms[period_idx][retailer_idx]
                .iter()
                .take(unmet)
                .filter(|value| **value < expedited_service_prob)
                .count();
            total_accepted += accepted;
            lost_at_retailer += unmet - accepted;
        }

        let expedited_shipped = total_accepted.min(remaining_warehouse_inventory);
        remaining_warehouse_inventory -= expedited_shipped;
        let lost_at_warehouse = total_accepted - expedited_shipped;

        warehouse_inventory = remaining_warehouse_inventory as i64;
        retailer_inventory = retailer_end_inventory;

        epoch_costs.push(
            warehouse_holding_cost * warehouse_inventory.max(0) as f64
                + retailer_holding_cost * retailer_inventory.iter().copied().map(|value| value.max(0) as f64).sum::<f64>()
                + warehouse_expedited_cost * expedited_shipped as f64
                + warehouse_lost_sale_cost * (lost_at_retailer + lost_at_warehouse) as f64,
        );
    }

    mean_after_warmup(&epoch_costs, warm_up_periods_ratio)
}

pub fn search_constant_base_stock_from_demands(
    warehouse_inventory: i64,
    warehouse_pipeline: &[usize],
    retailer_inventory: &[i64],
    retailer_pipeline: &[Vec<usize>],
    demands: &[Vec<usize>],
    expedite_uniforms: &[Vec<Vec<f64>>],
    warehouse_levels: &[usize],
    retailer_levels: &[usize],
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    expedited_service_prob: f64,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    warm_up_periods_ratio: f64,
    top_k: usize,
) -> PyResult<((usize, usize, f64), Vec<(usize, usize, f64)>)> {
    let mut results = Vec::new();
    for warehouse_level in warehouse_levels.iter().copied() {
        for retailer_level in retailer_levels.iter().copied() {
            let cost = rollout_constant_base_stock_from_demands(
                warehouse_level,
                retailer_level,
                warehouse_inventory,
                warehouse_pipeline,
                retailer_inventory,
                retailer_pipeline,
                demands,
                expedite_uniforms,
                warehouse_holding_cost,
                retailer_holding_cost,
                warehouse_expedited_cost,
                warehouse_lost_sale_cost,
                expedited_service_prob,
                warehouse_capacity,
                warehouse_inventory_cap,
                retailer_inventory_cap,
                warm_up_periods_ratio,
            );
            results.push((warehouse_level, retailer_level, cost));
        }
    }
    results.sort_by(|left, right| left.2.partial_cmp(&right.2).unwrap());
    Ok((results[0], results.into_iter().take(top_k).collect()))
}
