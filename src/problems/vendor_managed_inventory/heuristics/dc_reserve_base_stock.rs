use pyo3::PyResult;

use crate::problems::vendor_managed_inventory::env::{
    retailer_inventory_position, VendorManagedInventoryState,
};

pub fn dc_reserve_base_stock_shipment_quantity(
    state: &VendorManagedInventoryState,
    retailer_base_stock_level: usize,
    dc_reserve_quantity: usize,
    max_shipment_quantity: usize,
) -> PyResult<usize> {
    let desired = retailer_base_stock_level.saturating_sub(retailer_inventory_position(state));
    let available_for_shipment = state.dc_on_hand.saturating_sub(dc_reserve_quantity);
    Ok(desired
        .min(max_shipment_quantity)
        .min(available_for_shipment))
}
