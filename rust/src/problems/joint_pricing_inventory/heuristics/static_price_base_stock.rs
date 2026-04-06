use pyo3::PyResult;

pub fn static_price_base_stock_action(
    inventory_level: usize,
    order_up_to: usize,
    price_index: usize,
    max_order_quantity: usize,
    num_prices: usize,
) -> PyResult<(usize, usize)> {
    let order_quantity = order_up_to.saturating_sub(inventory_level);
    super::clip_and_validate_action(order_quantity, price_index, max_order_quantity, num_prices)
}
