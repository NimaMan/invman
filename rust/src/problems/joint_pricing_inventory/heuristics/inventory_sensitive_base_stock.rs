use pyo3::PyResult;

pub fn inventory_sensitive_base_stock_action(
    inventory_level: usize,
    order_up_to: usize,
    markdown_threshold: usize,
    high_price_index: usize,
    low_price_index: usize,
    max_order_quantity: usize,
    num_prices: usize,
) -> PyResult<(usize, usize)> {
    let order_quantity = order_up_to.saturating_sub(inventory_level);
    let price_index = if inventory_level >= markdown_threshold {
        low_price_index
    } else {
        high_price_index
    };
    super::clip_and_validate_action(order_quantity, price_index, max_order_quantity, num_prices)
}
