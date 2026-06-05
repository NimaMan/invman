use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::Rng;

use crate::problems::vendor_managed_inventory::demand::simulate_compound_poisson_interval;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VendorManagedInventoryState {
    pub period: usize,
    pub dc_on_hand: usize,
    pub retailer_on_hand: usize,
    pub retailer_pipeline: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VendorManagedInventoryStepOutcome {
    pub next_state: VendorManagedInventoryState,
    pub shipment_quantity: usize,
    pub realized_demand: usize,
    pub arrivals_to_retailer: usize,
    pub sales: usize,
    pub lost_sales: usize,
    pub dc_replenishment: usize,
    pub shipment_cost: f64,
    pub dc_holding_cost: f64,
    pub retailer_holding_cost: f64,
    pub stockout_cost: f64,
    pub period_cost: f64,
    pub reward: f64,
}

pub fn validate_costs(
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
    salvage_value_per_unit: f64,
) -> PyResult<()> {
    let values = [
        shipment_cost_per_unit,
        dc_holding_cost_per_unit,
        retailer_holding_cost_per_unit,
        stockout_cost_per_unit,
        salvage_value_per_unit,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "all costs and salvage values must be finite and non-negative",
        ));
    }
    Ok(())
}

pub fn validate_state(state: &VendorManagedInventoryState, dc_capacity: usize) -> PyResult<()> {
    if state.dc_on_hand > dc_capacity {
        return Err(PyValueError::new_err(format!(
            "dc_on_hand {} cannot exceed dc_capacity {}",
            state.dc_on_hand, dc_capacity
        )));
    }
    Ok(())
}

pub fn initialize_state(
    dc_on_hand: usize,
    retailer_on_hand: usize,
    retailer_pipeline: usize,
    dc_capacity: usize,
) -> PyResult<VendorManagedInventoryState> {
    let state = VendorManagedInventoryState {
        period: 0,
        dc_on_hand,
        retailer_on_hand,
        retailer_pipeline,
    };
    validate_state(&state, dc_capacity)?;
    Ok(state)
}

pub fn retailer_inventory_position(state: &VendorManagedInventoryState) -> usize {
    state.retailer_on_hand + state.retailer_pipeline
}

pub fn build_policy_state(
    state: &VendorManagedInventoryState,
    expected_demand: f64,
    periods: usize,
    dc_capacity: usize,
    dc_replenishment_quantity: usize,
) -> PyResult<Vec<f32>> {
    validate_state(state, dc_capacity)?;
    if !expected_demand.is_finite() || expected_demand < 0.0 {
        return Err(PyValueError::new_err(
            "expected_demand must be finite and non-negative",
        ));
    }
    let scale = dc_capacity
        .max(dc_replenishment_quantity)
        .max(expected_demand.ceil() as usize)
        .max(1) as f32;
    let remaining_fraction = if periods == 0 {
        0.0
    } else {
        (periods.saturating_sub(state.period) as f32) / periods as f32
    };
    Ok(vec![
        state.dc_on_hand as f32 / scale,
        state.retailer_on_hand as f32 / scale,
        state.retailer_pipeline as f32 / scale,
        retailer_inventory_position(state) as f32 / scale,
        expected_demand as f32 / scale,
        dc_replenishment_quantity as f32 / scale,
        remaining_fraction,
    ])
}

pub fn clip_action(
    state: &VendorManagedInventoryState,
    shipment_quantity: usize,
    dc_capacity: usize,
    max_shipment_quantity: usize,
) -> PyResult<usize> {
    validate_state(state, dc_capacity)?;
    Ok(shipment_quantity
        .min(max_shipment_quantity)
        .min(state.dc_on_hand))
}

#[allow(clippy::too_many_arguments)]
pub fn step_state(
    state: &VendorManagedInventoryState,
    shipment_quantity: usize,
    realized_demand: usize,
    dc_replenishment_quantity: usize,
    dc_capacity: usize,
    shipment_cost_per_unit: f64,
    dc_holding_cost_per_unit: f64,
    retailer_holding_cost_per_unit: f64,
    stockout_cost_per_unit: f64,
) -> PyResult<VendorManagedInventoryStepOutcome> {
    validate_state(state, dc_capacity)?;
    validate_costs(
        shipment_cost_per_unit,
        dc_holding_cost_per_unit,
        retailer_holding_cost_per_unit,
        stockout_cost_per_unit,
        0.0,
    )?;

    if shipment_quantity > state.dc_on_hand {
        return Err(PyValueError::new_err(format!(
            "shipment_quantity {} cannot exceed dc_on_hand {}",
            shipment_quantity, state.dc_on_hand
        )));
    }

    let arrivals_to_retailer = state.retailer_pipeline;
    let retailer_available = state.retailer_on_hand + arrivals_to_retailer;
    let sales = retailer_available.min(realized_demand);
    let lost_sales = realized_demand - sales;
    let next_retailer_on_hand = retailer_available - sales;
    let next_retailer_pipeline = shipment_quantity;

    let dc_after_shipment = state.dc_on_hand - shipment_quantity;
    let dc_replenishment = dc_replenishment_quantity.min(dc_capacity - dc_after_shipment);
    let next_dc_on_hand = dc_after_shipment + dc_replenishment;

    let next_state = VendorManagedInventoryState {
        period: state.period + 1,
        dc_on_hand: next_dc_on_hand,
        retailer_on_hand: next_retailer_on_hand,
        retailer_pipeline: next_retailer_pipeline,
    };

    let shipment_cost = shipment_cost_per_unit * shipment_quantity as f64;
    let dc_holding_cost = dc_holding_cost_per_unit * next_dc_on_hand as f64;
    let retailer_holding_cost = retailer_holding_cost_per_unit * next_retailer_on_hand as f64;
    let stockout_cost = stockout_cost_per_unit * lost_sales as f64;
    let period_cost = shipment_cost + dc_holding_cost + retailer_holding_cost + stockout_cost;

    Ok(VendorManagedInventoryStepOutcome {
        next_state,
        shipment_quantity,
        realized_demand,
        arrivals_to_retailer,
        sales,
        lost_sales,
        dc_replenishment,
        shipment_cost,
        dc_holding_cost,
        retailer_holding_cost,
        stockout_cost,
        period_cost,
        reward: -period_cost,
    })
}

pub fn terminal_salvage_credit(
    state: &VendorManagedInventoryState,
    dc_capacity: usize,
    salvage_value_per_unit: f64,
) -> PyResult<f64> {
    validate_state(state, dc_capacity)?;
    validate_costs(0.0, 0.0, 0.0, 0.0, salvage_value_per_unit)?;
    Ok(salvage_value_per_unit
        * (state.dc_on_hand + state.retailer_on_hand + state.retailer_pipeline) as f64)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaperRetailerProductParams {
    pub retailer_index: usize,
    pub product_index: usize,
    pub arrival_rate: f64,
    pub demand_low: f64,
    pub demand_high: f64,
    pub retailer_holding_cost_per_unit_time: f64,
    pub retailer_stockout_cost_per_unit: f64,
    pub revenue_per_unit_sold: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaperDcProductParams {
    pub product_index: usize,
    pub dc_holding_cost_per_unit_time: f64,
    pub dc_shortage_penalty_per_unit: f64,
    pub reorder_quantity: f64,
    pub reorder_point: f64,
    pub fixed_reorder_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformTimeDistribution {
    pub low: f64,
    pub high: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutstandingManufacturerOrder {
    pub quantity: f64,
    pub remaining_time: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PaperVendorManagedInventoryModel {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub num_retailers: usize,
    pub num_products: usize,
    pub retailer_product_params: Vec<PaperRetailerProductParams>,
    pub dc_product_params: Vec<PaperDcProductParams>,
    pub truck_capacity: f64,
    pub transport_cost_per_truck_per_unit_time: f64,
    pub dc_service_time: UniformTimeDistribution,
    pub dc_to_first_retailer_time: UniformTimeDistribution,
    pub retailer_to_retailer_time: UniformTimeDistribution,
    pub retailer_service_time: UniformTimeDistribution,
    pub last_retailer_to_dc_time: UniformTimeDistribution,
    pub manufacturer_lead_time: UniformTimeDistribution,
    pub max_trucks: usize,
    pub low_signal_multiplier: f64,
    pub high_signal_multiplier: f64,
    pub expected_signal_multiplier: f64,
    pub high_signal_probability: f64,
    pub initial_dc_inventory: Vec<f64>,
    pub initial_retailer_inventory: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PaperVendorManagedInventoryState {
    pub cycle_index: usize,
    pub retailer_inventory: Vec<Vec<f64>>,
    pub dc_inventory: Vec<f64>,
    pub demand_signal_high: Vec<Vec<bool>>,
    pub outstanding_dc_orders: Vec<Vec<OutstandingManufacturerOrder>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PaperVendorManagedInventoryStepOutcome {
    pub next_state: PaperVendorManagedInventoryState,
    pub trucks_dispatched: usize,
    pub dispatched_quantities: Vec<Vec<f64>>,
    pub route_cycle_time: f64,
    pub retailer_arrival_times: Vec<f64>,
    pub total_revenue: f64,
    pub transport_cost: f64,
    pub retailer_holding_cost: f64,
    pub retailer_stockout_cost: f64,
    pub dc_holding_cost: f64,
    pub dc_shortage_cost: f64,
    pub dc_reorder_cost: f64,
    pub cycle_profit: f64,
    pub average_profit_rate: f64,
}

fn sample_uniform_time(rng: &mut StdRng, distribution: UniformTimeDistribution) -> PyResult<f64> {
    if !distribution.low.is_finite()
        || !distribution.high.is_finite()
        || distribution.low < 0.0
        || distribution.high < distribution.low
    {
        return Err(PyValueError::new_err(
            "uniform time bounds must be finite, non-negative, and satisfy high >= low",
        ));
    }
    if (distribution.high - distribution.low).abs() < 1e-12 {
        return Ok(distribution.low);
    }
    Ok(rng.gen_range(distribution.low..distribution.high))
}

pub fn paper_model_param(
    model: &PaperVendorManagedInventoryModel,
    retailer_index: usize,
    product_index: usize,
) -> &PaperRetailerProductParams {
    model
        .retailer_product_params
        .iter()
        .find(|param| {
            param.retailer_index == retailer_index && param.product_index == product_index
        })
        .expect("retailer/product parameter must exist")
}

pub fn paper_signal_multiplier(is_high: bool, model: &PaperVendorManagedInventoryModel) -> f64 {
    if is_high {
        model.high_signal_multiplier
    } else {
        model.low_signal_multiplier
    }
}

pub fn validate_paper_model(model: &PaperVendorManagedInventoryModel) -> PyResult<()> {
    if model.num_retailers == 0 || model.num_products == 0 {
        return Err(PyValueError::new_err(
            "paper VMI model must have at least one retailer and one product",
        ));
    }
    if model.retailer_product_params.len() != model.num_retailers * model.num_products {
        return Err(PyValueError::new_err(
            "retailer_product_params length must match num_retailers * num_products",
        ));
    }
    if model.dc_product_params.len() != model.num_products {
        return Err(PyValueError::new_err(
            "dc_product_params length must match num_products",
        ));
    }
    if model.initial_dc_inventory.len() != model.num_products {
        return Err(PyValueError::new_err(
            "initial_dc_inventory length must match num_products",
        ));
    }
    if model.initial_retailer_inventory.len() != model.num_retailers
        || model
            .initial_retailer_inventory
            .iter()
            .any(|inventory| inventory.len() != model.num_products)
    {
        return Err(PyValueError::new_err(
            "initial_retailer_inventory shape must match [num_retailers][num_products]",
        ));
    }
    if !(0.0..=1.0).contains(&model.high_signal_probability) {
        return Err(PyValueError::new_err(
            "high_signal_probability must lie in [0, 1]",
        ));
    }
    Ok(())
}

pub fn initialize_paper_state(
    model: &PaperVendorManagedInventoryModel,
    rng: &mut StdRng,
) -> PyResult<PaperVendorManagedInventoryState> {
    validate_paper_model(model)?;
    let mut demand_signal_high = vec![vec![false; model.num_products]; model.num_retailers];
    for retailer in 0..model.num_retailers {
        for product in 0..model.num_products {
            demand_signal_high[retailer][product] =
                rng.gen_bool(model.high_signal_probability.clamp(0.0, 1.0));
        }
    }
    Ok(PaperVendorManagedInventoryState {
        cycle_index: 0,
        retailer_inventory: model.initial_retailer_inventory.clone(),
        dc_inventory: model.initial_dc_inventory.clone(),
        demand_signal_high,
        outstanding_dc_orders: vec![Vec::new(); model.num_products],
    })
}

pub fn build_paper_policy_state(
    model: &PaperVendorManagedInventoryModel,
    state: &PaperVendorManagedInventoryState,
) -> PyResult<Vec<f32>> {
    validate_paper_model(model)?;
    let mut features =
        Vec::with_capacity(model.num_retailers * model.num_products * 2 + model.num_products * 3);
    for retailer in 0..model.num_retailers {
        for product in 0..model.num_products {
            features.push(state.retailer_inventory[retailer][product] as f32);
        }
    }
    for product in 0..model.num_products {
        features.push(state.dc_inventory[product] as f32);
    }
    for retailer in 0..model.num_retailers {
        for product in 0..model.num_products {
            features.push(if state.demand_signal_high[retailer][product] {
                1.0
            } else {
                0.0
            });
        }
    }
    for product in 0..model.num_products {
        let quantity_in_transit = state.outstanding_dc_orders[product]
            .iter()
            .map(|order| order.quantity)
            .sum::<f64>();
        let nearest_arrival = state.outstanding_dc_orders[product]
            .iter()
            .map(|order| order.remaining_time)
            .fold(f64::INFINITY, f64::min);
        features.push(quantity_in_transit as f32);
        features.push(if nearest_arrival.is_finite() {
            nearest_arrival as f32
        } else {
            0.0
        });
    }
    Ok(features)
}

pub fn sample_route_cycle(
    model: &PaperVendorManagedInventoryModel,
    rng: &mut StdRng,
) -> PyResult<(Vec<f64>, f64)> {
    validate_paper_model(model)?;
    let mut retailer_arrival_times = vec![0.0; model.num_retailers];
    let mut elapsed = sample_uniform_time(rng, model.dc_service_time)?;
    elapsed += sample_uniform_time(rng, model.dc_to_first_retailer_time)?;
    retailer_arrival_times[0] = elapsed;
    for retailer in 0..model.num_retailers {
        elapsed += sample_uniform_time(rng, model.retailer_service_time)?;
        if retailer + 1 < model.num_retailers {
            elapsed += sample_uniform_time(rng, model.retailer_to_retailer_time)?;
            retailer_arrival_times[retailer + 1] = elapsed;
        } else {
            elapsed += sample_uniform_time(rng, model.last_retailer_to_dc_time)?;
        }
    }
    Ok((retailer_arrival_times, elapsed))
}

fn simulate_dc_inventory_path(
    model: &PaperVendorManagedInventoryModel,
    state: &PaperVendorManagedInventoryState,
    cycle_time: f64,
    rng: &mut StdRng,
) -> PyResult<(
    Vec<f64>,
    Vec<Vec<OutstandingManufacturerOrder>>,
    f64,
    f64,
    f64,
)> {
    let mut dc_inventory = state.dc_inventory.clone();

    let mut outstanding = state.outstanding_dc_orders.clone();
    let mut dc_holding_cost = 0.0;
    let mut dc_reorder_cost = 0.0;

    for product in 0..model.num_products {
        let mut arrivals = outstanding[product].clone();
        arrivals.sort_by(|a, b| {
            a.remaining_time
                .partial_cmp(&b.remaining_time)
                .expect("lead times must be finite")
        });
        let mut next_orders = Vec::new();
        let mut inventory = dc_inventory[product];
        let dc_param = &model.dc_product_params[product];
        let mut local_time = 0.0;
        for order in arrivals {
            if order.remaining_time <= cycle_time {
                dc_holding_cost += dc_param.dc_holding_cost_per_unit_time
                    * inventory
                    * (order.remaining_time - local_time);
                local_time = order.remaining_time;
                inventory += order.quantity;
            } else {
                next_orders.push(OutstandingManufacturerOrder {
                    quantity: order.quantity,
                    remaining_time: order.remaining_time - cycle_time,
                });
            }
        }
        dc_holding_cost +=
            dc_param.dc_holding_cost_per_unit_time * inventory * (cycle_time - local_time);
        dc_inventory[product] = inventory;
        outstanding[product] = next_orders;
    }

    for product in 0..model.num_products {
        let dc_param = &model.dc_product_params[product];
        if dc_inventory[product] < dc_param.reorder_point {
            let lead_time = sample_uniform_time(rng, model.manufacturer_lead_time)?;
            outstanding[product].push(OutstandingManufacturerOrder {
                quantity: dc_param.reorder_quantity,
                remaining_time: lead_time,
            });
            dc_reorder_cost += dc_param.fixed_reorder_cost;
        }
    }

    Ok((
        dc_inventory,
        outstanding,
        dc_holding_cost,
        dc_reorder_cost,
        cycle_time,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn step_paper_state(
    model: &PaperVendorManagedInventoryModel,
    state: &PaperVendorManagedInventoryState,
    trucks_dispatched: usize,
    dispatch_quantities: &[Vec<f64>],
    rng: &mut StdRng,
) -> PyResult<PaperVendorManagedInventoryStepOutcome> {
    validate_paper_model(model)?;
    if trucks_dispatched > model.max_trucks {
        return Err(PyValueError::new_err(format!(
            "trucks_dispatched {} cannot exceed max_trucks {}",
            trucks_dispatched, model.max_trucks
        )));
    }
    if dispatch_quantities.len() != model.num_retailers
        || dispatch_quantities
            .iter()
            .any(|quantities| quantities.len() != model.num_products)
    {
        return Err(PyValueError::new_err(
            "dispatch_quantities shape must match [num_retailers][num_products]",
        ));
    }

    let dispatch_total = dispatch_quantities
        .iter()
        .flat_map(|quantities| quantities.iter())
        .sum::<f64>();
    if dispatch_total > model.truck_capacity * trucks_dispatched as f64 + 1e-9 {
        return Err(PyValueError::new_err(
            "dispatch quantities exceed available truck capacity",
        ));
    }

    let mut dc_shortage_cost = 0.0;
    let mut dc_inventory_after_loading = state.dc_inventory.clone();
    for product in 0..model.num_products {
        let desired = dispatch_quantities
            .iter()
            .map(|quantities| quantities[product])
            .sum::<f64>();
        if desired > dc_inventory_after_loading[product] + 1e-9 {
            dc_shortage_cost += model.dc_product_params[product].dc_shortage_penalty_per_unit
                * (desired - dc_inventory_after_loading[product]);
        }
        dc_inventory_after_loading[product] =
            (dc_inventory_after_loading[product] - desired).max(0.0);
    }

    let (retailer_arrival_times, route_cycle_time) = sample_route_cycle(model, rng)?;
    let mut retailer_inventory = state.retailer_inventory.clone();
    let mut total_revenue = 0.0;
    let mut retailer_holding_cost = 0.0;
    let mut retailer_stockout_cost = 0.0;

    for retailer in 0..model.num_retailers {
        for product in 0..model.num_products {
            let param = paper_model_param(model, retailer, product);
            let signal_multiplier =
                paper_signal_multiplier(state.demand_signal_high[retailer][product], model);
            retailer_inventory[retailer][product] = retailer_inventory[retailer][product].max(0.0);
            let pre_arrival = simulate_compound_poisson_interval(
                rng,
                retailer_inventory[retailer][product],
                param.arrival_rate * signal_multiplier,
                param.demand_low,
                param.demand_high,
                retailer_arrival_times[retailer],
            )?;
            retailer_inventory[retailer][product] = pre_arrival.ending_inventory;
            total_revenue += pre_arrival.sales * param.revenue_per_unit_sold;
            retailer_stockout_cost +=
                pre_arrival.lost_sales * param.retailer_stockout_cost_per_unit;
            retailer_holding_cost += pre_arrival.positive_inventory_time_area
                * param.retailer_holding_cost_per_unit_time;

            retailer_inventory[retailer][product] += dispatch_quantities[retailer][product];
            retailer_inventory[retailer][product] = retailer_inventory[retailer][product].max(0.0);
            let post_arrival = simulate_compound_poisson_interval(
                rng,
                retailer_inventory[retailer][product],
                param.arrival_rate * signal_multiplier,
                param.demand_low,
                param.demand_high,
                route_cycle_time - retailer_arrival_times[retailer],
            )?;
            retailer_inventory[retailer][product] = post_arrival.ending_inventory;
            total_revenue += post_arrival.sales * param.revenue_per_unit_sold;
            retailer_stockout_cost +=
                post_arrival.lost_sales * param.retailer_stockout_cost_per_unit;
            retailer_holding_cost += post_arrival.positive_inventory_time_area
                * param.retailer_holding_cost_per_unit_time;
            retailer_inventory[retailer][product] = retailer_inventory[retailer][product].max(0.0);
        }
    }

    let mut simulated_state = PaperVendorManagedInventoryState {
        cycle_index: state.cycle_index + 1,
        retailer_inventory,
        dc_inventory: dc_inventory_after_loading,
        demand_signal_high: vec![vec![false; model.num_products]; model.num_retailers],
        outstanding_dc_orders: state.outstanding_dc_orders.clone(),
    };

    let (dc_inventory, outstanding_dc_orders, dc_holding_cost, dc_reorder_cost, _) =
        simulate_dc_inventory_path(model, &simulated_state, route_cycle_time, rng)?;
    simulated_state.dc_inventory = dc_inventory;
    simulated_state.outstanding_dc_orders = outstanding_dc_orders;
    for retailer in 0..model.num_retailers {
        for product in 0..model.num_products {
            simulated_state.demand_signal_high[retailer][product] =
                rng.gen_bool(model.high_signal_probability.clamp(0.0, 1.0));
        }
    }

    let transport_cost =
        model.transport_cost_per_truck_per_unit_time * trucks_dispatched as f64 * route_cycle_time;
    let cycle_profit =
        total_revenue - transport_cost - retailer_holding_cost - retailer_stockout_cost;

    Ok(PaperVendorManagedInventoryStepOutcome {
        next_state: simulated_state,
        trucks_dispatched,
        dispatched_quantities: dispatch_quantities.to_vec(),
        route_cycle_time,
        retailer_arrival_times,
        total_revenue,
        transport_cost,
        retailer_holding_cost,
        retailer_stockout_cost,
        dc_holding_cost,
        dc_shortage_cost,
        dc_reorder_cost,
        cycle_profit,
        average_profit_rate: if route_cycle_time > 0.0 {
            cycle_profit / route_cycle_time
        } else {
            0.0
        },
    })
}
