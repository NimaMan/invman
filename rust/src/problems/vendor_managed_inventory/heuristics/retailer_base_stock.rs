use pyo3::PyResult;

use crate::problems::vendor_managed_inventory::env::{
    retailer_inventory_position, VendorManagedInventoryState,
};

pub fn retailer_base_stock_shipment_quantity(
    state: &VendorManagedInventoryState,
    retailer_base_stock_level: usize,
    max_shipment_quantity: usize,
) -> PyResult<usize> {
    let desired = retailer_base_stock_level.saturating_sub(retailer_inventory_position(state));
    Ok(desired.min(max_shipment_quantity).min(state.dc_on_hand))
}
