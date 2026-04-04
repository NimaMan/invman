use crate::problems::perishable_inventory::env::{inventory_position, PerishableState};

pub fn base_stock_order_quantity(
    state: &PerishableState,
    base_stock_level: usize,
    max_order_size: usize,
) -> usize {
    base_stock_level
        .saturating_sub(inventory_position(state))
        .min(max_order_size)
}
