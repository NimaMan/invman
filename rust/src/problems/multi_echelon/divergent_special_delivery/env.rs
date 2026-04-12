#![allow(dead_code)]

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MultiEchelonState {
    pub period: usize,
    pub warehouse_inventory: i32,
    pub warehouse_pipeline: Vec<u32>,
    pub retailer_inventory: Vec<i32>,
    pub retailer_pipeline: Vec<Vec<u32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocationMode {
    SequentialIndex,
    Proportional,
    MinShortage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WarehouseBaseStockMode {
    Regular,
    Echelon,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InventoryDynamicsMode {
    Gijs2022,
    VanRoy1997,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionState {
    pub warehouse_available: i32,
    pub warehouse_future: Vec<u32>,
    pub warehouse_regular_inventory_position: i32,
    pub warehouse_echelon_inventory_position: i32,
    pub retailer_available: Vec<i32>,
    pub retailer_future: Vec<Vec<u32>>,
    pub retailer_inventory_positions: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrderPlan {
    pub warehouse_target: usize,
    pub retailer_target: usize,
    pub warehouse_order: usize,
    pub desired_retail_orders: Vec<usize>,
    pub shipped_retail_orders: Vec<usize>,
    pub remaining_warehouse_inventory_after_regular: usize,
    pub decision_state: DecisionState,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiEchelonStepOutcome {
    pub next_state: MultiEchelonState,
    pub order_plan: OrderPlan,
    pub realized_demands: Vec<u32>,
    pub total_unmet_demand: usize,
    pub accepted_emergency_shipments: usize,
    pub expedited_shipments: usize,
    pub lost_sales: usize,
    pub warehouse_holding_cost: f64,
    pub retailer_holding_cost: f64,
    pub expedited_cost: f64,
    pub lost_sale_cost: f64,
    pub period_cost: f64,
    pub reward: f64,
}

pub fn parse_allocation_mode(value: &str) -> PyResult<AllocationMode> {
    match value {
        "sequential_index" => Ok(AllocationMode::SequentialIndex),
        "proportional" => Ok(AllocationMode::Proportional),
        "min_shortage" => Ok(AllocationMode::MinShortage),
        other => Err(PyValueError::new_err(format!(
            "unsupported allocation_mode '{other}'"
        ))),
    }
}

pub fn parse_warehouse_base_stock_mode(value: &str) -> PyResult<WarehouseBaseStockMode> {
    match value {
        "regular" => Ok(WarehouseBaseStockMode::Regular),
        "echelon" => Ok(WarehouseBaseStockMode::Echelon),
        other => Err(PyValueError::new_err(format!(
            "unsupported warehouse_base_stock_mode '{other}'"
        ))),
    }
}

pub fn parse_inventory_dynamics_mode(value: &str) -> PyResult<InventoryDynamicsMode> {
    match value {
        "gijs_2022" | "gijs" => Ok(InventoryDynamicsMode::Gijs2022),
        "van_roy_1997" | "van_roy" => Ok(InventoryDynamicsMode::VanRoy1997),
        other => Err(PyValueError::new_err(format!(
            "unsupported inventory_dynamics_mode '{other}'"
        ))),
    }
}

pub fn validate_state(state: &MultiEchelonState) -> PyResult<()> {
    if state.retailer_inventory.is_empty() {
        return Err(PyValueError::new_err(
            "multi_echelon state must contain at least one retailer",
        ));
    }
    if state.retailer_pipeline.len() != state.retailer_inventory.len() {
        return Err(PyValueError::new_err(format!(
            "retailer_pipeline has {} rows but retailer_inventory has {} entries",
            state.retailer_pipeline.len(),
            state.retailer_inventory.len()
        )));
    }
    if state
        .retailer_pipeline
        .iter()
        .any(|row| row.len() != state.retailer_pipeline[0].len())
    {
        return Err(PyValueError::new_err(
            "all retailer pipeline rows must have the same length",
        ));
    }
    Ok(())
}

pub fn initialize_state(
    warehouse_inventory: i32,
    warehouse_pipeline: &[u32],
    retailer_inventory: &[i32],
    retailer_pipeline: &[Vec<u32>],
) -> PyResult<MultiEchelonState> {
    let state = MultiEchelonState {
        period: 0,
        warehouse_inventory,
        warehouse_pipeline: warehouse_pipeline.to_vec(),
        retailer_inventory: retailer_inventory.to_vec(),
        retailer_pipeline: retailer_pipeline.to_vec(),
    };
    validate_state(&state)?;
    Ok(state)
}

pub fn initialize_random_state(
    warehouse_lead_time: usize,
    retailer_lead_time: usize,
    num_retailers: usize,
    warehouse_levels: &[usize],
    retailer_levels: &[usize],
    demand_mean: f64,
    seed: u64,
) -> PyResult<MultiEchelonState> {
    if num_retailers == 0 {
        return Err(PyValueError::new_err(
            "num_retailers must be strictly positive",
        ));
    }
    if warehouse_levels.is_empty() {
        return Err(PyValueError::new_err(
            "warehouse_levels must contain at least one candidate",
        ));
    }
    if retailer_levels.is_empty() {
        return Err(PyValueError::new_err(
            "retailer_levels must contain at least one candidate",
        ));
    }

    let _ = (demand_mean, seed);
    initialize_state(
        0,
        &vec![0u32; warehouse_lead_time],
        &vec![0i32; num_retailers],
        &vec![vec![0u32; retailer_lead_time]; num_retailers],
    )
}

pub fn build_raw_state(state: &MultiEchelonState) -> PyResult<Vec<f32>> {
    validate_state(state)?;
    let mut raw_state = vec![state.warehouse_inventory as f32];
    raw_state.extend(state.warehouse_pipeline.iter().map(|value| *value as f32));
    raw_state.extend(state.retailer_inventory.iter().map(|value| *value as f32));
    for pipeline_row in &state.retailer_pipeline {
        raw_state.extend(pipeline_row.iter().map(|value| *value as f32));
    }
    raw_state.push(state.period as f32);
    Ok(raw_state)
}

pub fn build_decision_state_with_mode(
    state: &MultiEchelonState,
    inventory_dynamics_mode: InventoryDynamicsMode,
) -> PyResult<DecisionState> {
    validate_state(state)?;
    let warehouse_available = match inventory_dynamics_mode {
        InventoryDynamicsMode::Gijs2022 => {
            state.warehouse_inventory
                + state.warehouse_pipeline.first().copied().unwrap_or(0) as i32
        }
        InventoryDynamicsMode::VanRoy1997 => state.warehouse_inventory,
    };
    let warehouse_future = match inventory_dynamics_mode {
        InventoryDynamicsMode::Gijs2022 => state
            .warehouse_pipeline
            .iter()
            .copied()
            .skip(1)
            .collect::<Vec<_>>(),
        InventoryDynamicsMode::VanRoy1997 => state.warehouse_pipeline.clone(),
    };
    let warehouse_regular_inventory_position =
        warehouse_available + warehouse_future.iter().map(|value| *value as i32).sum::<i32>();

    let mut retailer_available = Vec::with_capacity(state.retailer_inventory.len());
    let mut retailer_future = Vec::with_capacity(state.retailer_inventory.len());
    let mut retailer_inventory_positions = Vec::with_capacity(state.retailer_inventory.len());
    for (retailer_idx, inventory_level) in state.retailer_inventory.iter().copied().enumerate() {
        let available = match inventory_dynamics_mode {
            InventoryDynamicsMode::Gijs2022 => {
                inventory_level
                    + state.retailer_pipeline[retailer_idx]
                        .first()
                        .copied()
                        .unwrap_or(0) as i32
            }
            InventoryDynamicsMode::VanRoy1997 => inventory_level,
        };
        let future = match inventory_dynamics_mode {
            InventoryDynamicsMode::Gijs2022 => state.retailer_pipeline[retailer_idx]
                .iter()
                .copied()
                .skip(1)
                .collect::<Vec<_>>(),
            InventoryDynamicsMode::VanRoy1997 => state.retailer_pipeline[retailer_idx].clone(),
        };
        let inventory_position = available + future.iter().map(|value| *value as i32).sum::<i32>();
        retailer_available.push(available);
        retailer_future.push(future);
        retailer_inventory_positions.push(inventory_position);
    }

    let warehouse_echelon_inventory_position = warehouse_regular_inventory_position
        + retailer_inventory_positions.iter().copied().sum::<i32>();

    Ok(DecisionState {
        warehouse_available,
        warehouse_future,
        warehouse_regular_inventory_position,
        warehouse_echelon_inventory_position,
        retailer_available,
        retailer_future,
        retailer_inventory_positions,
    })
}

pub fn build_decision_state(state: &MultiEchelonState) -> PyResult<DecisionState> {
    build_decision_state_with_mode(state, InventoryDynamicsMode::Gijs2022)
}

fn proportional_allocation(desired_retail_orders: &[usize], available_inventory: usize) -> Vec<usize> {
    let total_desired = desired_retail_orders.iter().sum::<usize>();
    if total_desired <= available_inventory {
        return desired_retail_orders.to_vec();
    }
    let mut shipped = desired_retail_orders
        .iter()
        .map(|desired| {
            ((available_inventory as f64 * *desired as f64) / total_desired as f64).floor() as usize
        })
        .collect::<Vec<_>>();
    let mut remaining = available_inventory.saturating_sub(shipped.iter().sum::<usize>());
    if remaining == 0 {
        return shipped;
    }

    let mut priorities = desired_retail_orders
        .iter()
        .enumerate()
        .map(|(idx, desired)| {
            let exact_share = available_inventory as f64 * *desired as f64 / total_desired as f64;
            let fractional = exact_share - shipped[idx] as f64;
            (idx, fractional, *desired)
        })
        .collect::<Vec<_>>();
    priorities.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| left.0.cmp(&right.0))
    });

    for (idx, _, desired) in priorities {
        if remaining == 0 {
            break;
        }
        if shipped[idx] < desired {
            shipped[idx] += 1;
            remaining -= 1;
        }
    }
    shipped
}

fn min_shortage_allocation(desired_retail_orders: &[usize], available_inventory: usize) -> Vec<usize> {
    let total_desired = desired_retail_orders.iter().sum::<usize>();
    if total_desired <= available_inventory {
        return desired_retail_orders.to_vec();
    }
    if available_inventory == 0 {
        return vec![0usize; desired_retail_orders.len()];
    }

    let mut shortfalls = desired_retail_orders.to_vec();
    let mut allocations = vec![0usize; desired_retail_orders.len()];
    let mut remaining = available_inventory;

    while remaining > 0 {
        let max_shortfall = shortfalls.iter().copied().max().unwrap_or(0);
        if max_shortfall == 0 {
            break;
        }
        let active = shortfalls
            .iter()
            .enumerate()
            .filter_map(|(idx, shortfall)| (*shortfall == max_shortfall).then_some(idx))
            .collect::<Vec<_>>();
        let next_shortfall = shortfalls
            .iter()
            .copied()
            .filter(|shortfall| *shortfall < max_shortfall)
            .max()
            .unwrap_or(0);
        let decrement = max_shortfall - next_shortfall;
        let required = decrement * active.len();

        if remaining >= required {
            for idx in &active {
                allocations[*idx] += decrement;
                shortfalls[*idx] -= decrement;
            }
            remaining -= required;
            continue;
        }

        let shared = remaining / active.len();
        let residue = remaining % active.len();
        for (rank, idx) in active.iter().enumerate() {
            let increment = shared + usize::from(rank < residue);
            allocations[*idx] += increment;
            shortfalls[*idx] -= increment;
        }
        remaining = 0;
    }

    allocations
}

fn allocate_regular_shipments(
    desired_retail_orders: &[usize],
    available_inventory: usize,
    allocation_mode: AllocationMode,
) -> Vec<usize> {
    match allocation_mode {
        AllocationMode::SequentialIndex => {
            let mut remaining = available_inventory;
            let mut shipped = vec![0usize; desired_retail_orders.len()];
            for retailer_idx in 0..desired_retail_orders.len() {
                let quantity = desired_retail_orders[retailer_idx].min(remaining);
                shipped[retailer_idx] = quantity;
                remaining -= quantity;
            }
            shipped
        }
        AllocationMode::Proportional => proportional_allocation(desired_retail_orders, available_inventory),
        AllocationMode::MinShortage => min_shortage_allocation(desired_retail_orders, available_inventory),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_order_plan_with_mode(
    state: &MultiEchelonState,
    warehouse_target: usize,
    retailer_target: usize,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    warehouse_base_stock_mode: WarehouseBaseStockMode,
    allocation_mode: AllocationMode,
    inventory_dynamics_mode: InventoryDynamicsMode,
) -> PyResult<OrderPlan> {
    let decision_state = build_decision_state_with_mode(state, inventory_dynamics_mode)?;
    let warehouse_target = warehouse_target.min(warehouse_inventory_cap);
    let retailer_target = retailer_target.min(retailer_inventory_cap);
    let warehouse_future_total = decision_state
        .warehouse_future
        .iter()
        .map(|value| *value as i32)
        .sum::<i32>();
    let desired_retail_orders = decision_state
        .retailer_inventory_positions
        .iter()
        .map(|inventory_position| {
            retailer_target.saturating_sub((*inventory_position).max(0) as usize)
        })
        .collect::<Vec<_>>();
    let shipped_retail_orders = allocate_regular_shipments(
        &desired_retail_orders,
        decision_state.warehouse_available.max(0) as usize,
        allocation_mode,
    );
    let shipped_total = shipped_retail_orders.iter().sum::<usize>();
    let remaining_warehouse_inventory_after_regular =
        (decision_state.warehouse_available.max(0) as usize).saturating_sub(shipped_total);
    let warehouse_regular_inventory_position_after_regular =
        remaining_warehouse_inventory_after_regular as i32 + warehouse_future_total;
    let warehouse_echelon_inventory_position_after_regular =
        warehouse_regular_inventory_position_after_regular
            + decision_state
                .retailer_inventory_positions
                .iter()
                .zip(shipped_retail_orders.iter())
                .map(|(inventory_position, shipped)| *inventory_position + *shipped as i32)
                .sum::<i32>();
    let warehouse_reference_inventory_position = match warehouse_base_stock_mode {
        WarehouseBaseStockMode::Regular => warehouse_regular_inventory_position_after_regular,
        WarehouseBaseStockMode::Echelon => warehouse_echelon_inventory_position_after_regular,
    };
    let warehouse_order = warehouse_target
        .saturating_sub(warehouse_reference_inventory_position.max(0) as usize)
        .min(warehouse_capacity);

    Ok(OrderPlan {
        warehouse_target,
        retailer_target,
        warehouse_order,
        desired_retail_orders,
        shipped_retail_orders,
        remaining_warehouse_inventory_after_regular,
        decision_state,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build_order_plan_with_explicit_warehouse_order_and_mode(
    state: &MultiEchelonState,
    warehouse_order: usize,
    retailer_target: usize,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    allocation_mode: AllocationMode,
    inventory_dynamics_mode: InventoryDynamicsMode,
) -> PyResult<OrderPlan> {
    let decision_state = build_decision_state_with_mode(state, inventory_dynamics_mode)?;
    let retailer_target = retailer_target.min(retailer_inventory_cap);
    let warehouse_future_total = decision_state
        .warehouse_future
        .iter()
        .map(|value| *value as i32)
        .sum::<i32>();
    let desired_retail_orders = decision_state
        .retailer_inventory_positions
        .iter()
        .map(|inventory_position| {
            retailer_target.saturating_sub((*inventory_position).max(0) as usize)
        })
        .collect::<Vec<_>>();
    let shipped_retail_orders = allocate_regular_shipments(
        &desired_retail_orders,
        decision_state.warehouse_available.max(0) as usize,
        allocation_mode,
    );
    let shipped_total = shipped_retail_orders.iter().sum::<usize>();
    let remaining_warehouse_inventory_after_regular =
        (decision_state.warehouse_available.max(0) as usize).saturating_sub(shipped_total);
    let warehouse_regular_inventory_position_after_regular =
        remaining_warehouse_inventory_after_regular as i32 + warehouse_future_total;
    let warehouse_order = warehouse_order
        .min(warehouse_capacity)
        .min(
            warehouse_inventory_cap.saturating_sub(
                warehouse_regular_inventory_position_after_regular.max(0) as usize,
            ),
        );
    Ok(OrderPlan {
        warehouse_target: warehouse_regular_inventory_position_after_regular.max(0) as usize
            + warehouse_order,
        retailer_target,
        warehouse_order,
        desired_retail_orders,
        shipped_retail_orders,
        remaining_warehouse_inventory_after_regular,
        decision_state,
    })
}

pub fn build_order_plan(
    state: &MultiEchelonState,
    warehouse_target: usize,
    retailer_target: usize,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    warehouse_base_stock_mode: WarehouseBaseStockMode,
    allocation_mode: AllocationMode,
) -> PyResult<OrderPlan> {
    build_order_plan_with_mode(
        state,
        warehouse_target,
        retailer_target,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        warehouse_base_stock_mode,
        allocation_mode,
        InventoryDynamicsMode::Gijs2022,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn step_state_with_mode(
    state: &MultiEchelonState,
    warehouse_target: usize,
    retailer_target: usize,
    realized_demands: &[u32],
    accepted_emergency_shipments: usize,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    warehouse_base_stock_mode: WarehouseBaseStockMode,
    allocation_mode: AllocationMode,
    inventory_dynamics_mode: InventoryDynamicsMode,
) -> PyResult<MultiEchelonStepOutcome> {
    if realized_demands.len() != state.retailer_inventory.len() {
        return Err(PyValueError::new_err(format!(
            "realized_demands length {} does not match num_retailers {}",
            realized_demands.len(),
            state.retailer_inventory.len()
        )));
    }

    let order_plan = build_order_plan_with_mode(
        state,
        warehouse_target,
        retailer_target,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        warehouse_base_stock_mode,
        allocation_mode,
        inventory_dynamics_mode,
    )?;

    let mut retailer_inventory = Vec::with_capacity(realized_demands.len());
    let mut total_unmet_demand = 0usize;
    for retailer_idx in 0..realized_demands.len() {
        let on_hand = order_plan.decision_state.retailer_available[retailer_idx].max(0) as u32;
        let served = on_hand.min(realized_demands[retailer_idx]);
        let unmet = realized_demands[retailer_idx] - served;
        total_unmet_demand += unmet as usize;
        retailer_inventory.push(
            order_plan.decision_state.retailer_available[retailer_idx] - served as i32,
        );
    }

    let warehouse_inventory_before_emergency = match inventory_dynamics_mode {
        InventoryDynamicsMode::Gijs2022 => order_plan.remaining_warehouse_inventory_after_regular,
        InventoryDynamicsMode::VanRoy1997 => {
            if order_plan.decision_state.warehouse_future.is_empty() {
                order_plan.remaining_warehouse_inventory_after_regular + order_plan.warehouse_order
            } else {
                order_plan.remaining_warehouse_inventory_after_regular
            }
        }
    };

    let expedited_shipments = accepted_emergency_shipments
        .min(warehouse_inventory_before_emergency);
    let lost_sales = total_unmet_demand.saturating_sub(expedited_shipments);
    let mut warehouse_inventory = warehouse_inventory_before_emergency as i32
        - expedited_shipments as i32;
    let mut warehouse_pipeline;
    let mut retailer_pipeline;

    match inventory_dynamics_mode {
        InventoryDynamicsMode::Gijs2022 => {
            warehouse_pipeline = order_plan.decision_state.warehouse_future.clone();
            warehouse_pipeline.push(order_plan.warehouse_order as u32);

            retailer_pipeline = order_plan.decision_state.retailer_future.clone();
            for retailer_idx in 0..retailer_pipeline.len() {
                retailer_pipeline[retailer_idx]
                    .push(order_plan.shipped_retail_orders[retailer_idx] as u32);
            }
        }
        InventoryDynamicsMode::VanRoy1997 => {
            warehouse_pipeline = order_plan.decision_state.warehouse_future.clone();
            if let Some(last) = warehouse_pipeline.last_mut() {
                *last += order_plan.warehouse_order as u32;
                let arriving = warehouse_pipeline.remove(0);
                warehouse_inventory += arriving as i32;
                warehouse_pipeline.push(0);
            }

            retailer_pipeline = order_plan.decision_state.retailer_future.clone();
            for retailer_idx in 0..retailer_pipeline.len() {
                if let Some(last) = retailer_pipeline[retailer_idx].last_mut() {
                    *last += order_plan.shipped_retail_orders[retailer_idx] as u32;
                    let arriving = retailer_pipeline[retailer_idx].remove(0);
                    retailer_inventory[retailer_idx] += arriving as i32;
                    retailer_pipeline[retailer_idx].push(0);
                } else {
                    retailer_inventory[retailer_idx] += order_plan.shipped_retail_orders[retailer_idx] as i32;
                }
            }
        }
    }

    let holding_warehouse_inventory = match inventory_dynamics_mode {
        InventoryDynamicsMode::Gijs2022 => warehouse_inventory,
        // Van Roy defines costs as g(y_t, w_t), so storage is charged on the
        // post-decision on-hand inventory before demand is realized.
        InventoryDynamicsMode::VanRoy1997 => warehouse_inventory_before_emergency as i32,
    };
    let holding_retailer_inventory = match inventory_dynamics_mode {
        InventoryDynamicsMode::Gijs2022 => retailer_inventory.as_slice(),
        InventoryDynamicsMode::VanRoy1997 => order_plan.decision_state.retailer_available.as_slice(),
    };
    let warehouse_holding_cost_component =
        warehouse_holding_cost * holding_warehouse_inventory.max(0) as f64;
    let retailer_holding_cost_component = retailer_holding_cost
        * holding_retailer_inventory
            .iter()
            .map(|value| (*value).max(0) as f64)
            .sum::<f64>();
    let expedited_cost_component = warehouse_expedited_cost * expedited_shipments as f64;
    let lost_sale_cost_component = warehouse_lost_sale_cost * lost_sales as f64;
    let period_cost = warehouse_holding_cost_component
        + retailer_holding_cost_component
        + expedited_cost_component
        + lost_sale_cost_component;

    Ok(MultiEchelonStepOutcome {
        next_state: MultiEchelonState {
            period: state.period + 1,
            warehouse_inventory,
            warehouse_pipeline,
            retailer_inventory,
            retailer_pipeline,
        },
        order_plan,
        realized_demands: realized_demands.to_vec(),
        total_unmet_demand,
        accepted_emergency_shipments,
        expedited_shipments,
        lost_sales,
        warehouse_holding_cost: warehouse_holding_cost_component,
        retailer_holding_cost: retailer_holding_cost_component,
        expedited_cost: expedited_cost_component,
        lost_sale_cost: lost_sale_cost_component,
        period_cost,
        reward: -period_cost,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn step_state_with_explicit_warehouse_order_and_mode(
    state: &MultiEchelonState,
    warehouse_order: usize,
    retailer_target: usize,
    realized_demands: &[u32],
    accepted_emergency_shipments: usize,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    allocation_mode: AllocationMode,
    inventory_dynamics_mode: InventoryDynamicsMode,
) -> PyResult<MultiEchelonStepOutcome> {
    if realized_demands.len() != state.retailer_inventory.len() {
        return Err(PyValueError::new_err(format!(
            "realized_demands length {} does not match num_retailers {}",
            realized_demands.len(),
            state.retailer_inventory.len()
        )));
    }

    let order_plan = build_order_plan_with_explicit_warehouse_order_and_mode(
        state,
        warehouse_order,
        retailer_target,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        allocation_mode,
        inventory_dynamics_mode,
    )?;

    let mut retailer_inventory = Vec::with_capacity(realized_demands.len());
    let mut total_unmet_demand = 0usize;
    for retailer_idx in 0..realized_demands.len() {
        let on_hand = order_plan.decision_state.retailer_available[retailer_idx].max(0) as u32;
        let served = on_hand.min(realized_demands[retailer_idx]);
        let unmet = realized_demands[retailer_idx] - served;
        total_unmet_demand += unmet as usize;
        retailer_inventory.push(
            order_plan.decision_state.retailer_available[retailer_idx] - served as i32,
        );
    }

    let warehouse_inventory_before_emergency = match inventory_dynamics_mode {
        InventoryDynamicsMode::Gijs2022 => order_plan.remaining_warehouse_inventory_after_regular,
        InventoryDynamicsMode::VanRoy1997 => {
            if order_plan.decision_state.warehouse_future.is_empty() {
                order_plan.remaining_warehouse_inventory_after_regular + order_plan.warehouse_order
            } else {
                order_plan.remaining_warehouse_inventory_after_regular
            }
        }
    };

    let expedited_shipments = accepted_emergency_shipments
        .min(warehouse_inventory_before_emergency);
    let lost_sales = total_unmet_demand.saturating_sub(expedited_shipments);
    let mut warehouse_inventory = warehouse_inventory_before_emergency as i32
        - expedited_shipments as i32;
    let mut warehouse_pipeline;
    let mut retailer_pipeline;

    match inventory_dynamics_mode {
        InventoryDynamicsMode::Gijs2022 => {
            warehouse_pipeline = order_plan.decision_state.warehouse_future.clone();
            warehouse_pipeline.push(order_plan.warehouse_order as u32);

            retailer_pipeline = order_plan.decision_state.retailer_future.clone();
            for retailer_idx in 0..retailer_pipeline.len() {
                retailer_pipeline[retailer_idx]
                    .push(order_plan.shipped_retail_orders[retailer_idx] as u32);
            }
        }
        InventoryDynamicsMode::VanRoy1997 => {
            warehouse_pipeline = order_plan.decision_state.warehouse_future.clone();
            if let Some(last) = warehouse_pipeline.last_mut() {
                *last += order_plan.warehouse_order as u32;
                let arriving = warehouse_pipeline.remove(0);
                warehouse_inventory += arriving as i32;
                warehouse_pipeline.push(0);
            }

            retailer_pipeline = order_plan.decision_state.retailer_future.clone();
            for retailer_idx in 0..retailer_pipeline.len() {
                if let Some(last) = retailer_pipeline[retailer_idx].last_mut() {
                    *last += order_plan.shipped_retail_orders[retailer_idx] as u32;
                    let arriving = retailer_pipeline[retailer_idx].remove(0);
                    retailer_inventory[retailer_idx] += arriving as i32;
                    retailer_pipeline[retailer_idx].push(0);
                } else {
                    retailer_inventory[retailer_idx] +=
                        order_plan.shipped_retail_orders[retailer_idx] as i32;
                }
            }
        }
    }

    let holding_warehouse_inventory = match inventory_dynamics_mode {
        InventoryDynamicsMode::Gijs2022 => warehouse_inventory,
        InventoryDynamicsMode::VanRoy1997 => warehouse_inventory_before_emergency as i32,
    };
    let holding_retailer_inventory = match inventory_dynamics_mode {
        InventoryDynamicsMode::Gijs2022 => retailer_inventory.as_slice(),
        InventoryDynamicsMode::VanRoy1997 => order_plan.decision_state.retailer_available.as_slice(),
    };
    let warehouse_holding_cost_component =
        warehouse_holding_cost * holding_warehouse_inventory.max(0) as f64;
    let retailer_holding_cost_component = retailer_holding_cost
        * holding_retailer_inventory
            .iter()
            .map(|value| (*value).max(0) as f64)
            .sum::<f64>();
    let expedited_cost_component = warehouse_expedited_cost * expedited_shipments as f64;
    let lost_sale_cost_component = warehouse_lost_sale_cost * lost_sales as f64;
    let period_cost = warehouse_holding_cost_component
        + retailer_holding_cost_component
        + expedited_cost_component
        + lost_sale_cost_component;

    Ok(MultiEchelonStepOutcome {
        next_state: MultiEchelonState {
            period: state.period + 1,
            warehouse_inventory,
            warehouse_pipeline,
            retailer_inventory,
            retailer_pipeline,
        },
        order_plan,
        realized_demands: realized_demands.to_vec(),
        total_unmet_demand,
        accepted_emergency_shipments,
        expedited_shipments,
        lost_sales,
        warehouse_holding_cost: warehouse_holding_cost_component,
        retailer_holding_cost: retailer_holding_cost_component,
        expedited_cost: expedited_cost_component,
        lost_sale_cost: lost_sale_cost_component,
        period_cost,
        reward: -period_cost,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn step_state(
    state: &MultiEchelonState,
    warehouse_target: usize,
    retailer_target: usize,
    realized_demands: &[u32],
    accepted_emergency_shipments: usize,
    warehouse_capacity: usize,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
    warehouse_holding_cost: f64,
    retailer_holding_cost: f64,
    warehouse_expedited_cost: f64,
    warehouse_lost_sale_cost: f64,
    warehouse_base_stock_mode: WarehouseBaseStockMode,
    allocation_mode: AllocationMode,
) -> PyResult<MultiEchelonStepOutcome> {
    step_state_with_mode(
        state,
        warehouse_target,
        retailer_target,
        realized_demands,
        accepted_emergency_shipments,
        warehouse_capacity,
        warehouse_inventory_cap,
        retailer_inventory_cap,
        warehouse_holding_cost,
        retailer_holding_cost,
        warehouse_expedited_cost,
        warehouse_lost_sale_cost,
        warehouse_base_stock_mode,
        allocation_mode,
        InventoryDynamicsMode::Gijs2022,
    )
}
