use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocationPolicy {
    Proportional,
    RandomSequential,
    MinShortage,
}

pub fn parse_allocation_policy(policy: &str) -> PyResult<AllocationPolicy> {
    match policy {
        "proportional" => Ok(AllocationPolicy::Proportional),
        "random_sequential" | "sequential" => Ok(AllocationPolicy::RandomSequential),
        "min_shortage" => Ok(AllocationPolicy::MinShortage),
        _ => Err(PyValueError::new_err(format!(
            "unknown allocation policy '{policy}'; expected 'proportional', 'random_sequential', or 'min_shortage'"
        ))),
    }
}

pub fn proportional_shipments(
    available_inventory: usize,
    retailer_orders: &[usize],
) -> PyResult<Vec<usize>> {
    if retailer_orders.is_empty() {
        return Err(PyValueError::new_err(
            "retailer_orders must contain at least one retailer order",
        ));
    }
    let total_orders = retailer_orders.iter().sum::<usize>();
    if total_orders <= available_inventory {
        return Ok(retailer_orders.to_vec());
    }
    if total_orders == 0 {
        return Ok(vec![0; retailer_orders.len()]);
    }
    Ok(retailer_orders
        .iter()
        .map(|order| order.saturating_mul(available_inventory) / total_orders)
        .collect())
}

pub fn random_sequential_shipments<R: Rng + ?Sized>(
    rng: &mut R,
    available_inventory: usize,
    retailer_orders: &[usize],
) -> PyResult<Vec<usize>> {
    if retailer_orders.is_empty() {
        return Err(PyValueError::new_err(
            "retailer_orders must contain at least one retailer order",
        ));
    }
    let mut shipments = vec![0usize; retailer_orders.len()];
    let mut retailer_indices = (0..retailer_orders.len()).collect::<Vec<_>>();
    retailer_indices.shuffle(rng);

    let mut remaining_inventory = available_inventory;
    for retailer_idx in retailer_indices {
        let shipment = retailer_orders[retailer_idx].min(remaining_inventory);
        shipments[retailer_idx] = shipment;
        remaining_inventory -= shipment;
        if remaining_inventory == 0 {
            break;
        }
    }
    Ok(shipments)
}

pub fn min_shortage_shipments(
    available_inventory: usize,
    retailer_orders: &[usize],
    retailer_inventory_positions: &[i32],
    retailer_base_stock_levels: &[usize],
) -> PyResult<Vec<usize>> {
    if retailer_orders.is_empty()
        || retailer_inventory_positions.len() != retailer_orders.len()
        || retailer_base_stock_levels.len() != retailer_orders.len()
    {
        return Err(PyValueError::new_err(
            "all retailer-wise arrays must have the same positive length",
        ));
    }

    let total_orders = retailer_orders.iter().sum::<usize>();
    if total_orders <= available_inventory {
        return Ok(retailer_orders.to_vec());
    }

    let mut shipments = vec![0usize; retailer_orders.len()];
    let mut remaining_inventory = available_inventory;

    while remaining_inventory > 0 {
        let mut best_index = None;
        let mut largest_remaining_shortfall = i32::MIN;

        for retailer_idx in 0..retailer_orders.len() {
            if shipments[retailer_idx] >= retailer_orders[retailer_idx] {
                continue;
            }
            let current_shortfall = retailer_base_stock_levels[retailer_idx] as i32
                - retailer_inventory_positions[retailer_idx]
                - shipments[retailer_idx] as i32;
            if current_shortfall > largest_remaining_shortfall {
                largest_remaining_shortfall = current_shortfall;
                best_index = Some(retailer_idx);
            }
        }

        match best_index {
            Some(retailer_idx) => {
                shipments[retailer_idx] += 1;
                remaining_inventory -= 1;
            }
            None => break,
        }
    }

    Ok(shipments)
}
