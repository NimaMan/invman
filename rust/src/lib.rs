mod core;
mod problems;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn invman_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    core::policies::bindings::register_py(m)?;
    problems::ameliorating_inventory::bindings::register_py(m)?;
    problems::decentralized_inventory_control::bindings::register_py(m)?;
    problems::lost_sales::bindings::register_py(m)?;
    problems::lost_sales_fixed_order_cost::bindings::register_py(m)?;
    problems::dual_sourcing::bindings::register_py(m)?;
    problems::joint_replenishment::bindings::register_py(m)?;
    problems::multi_echelon::bindings::register_py(m)?;
    problems::nonstationary_lot_sizing::bindings::register_py(m)?;
    problems::network_inventory::bindings::register_py(m)?;
    problems::one_warehouse_multi_retailer::bindings::register_py(m)?;
    problems::perishable_inventory::bindings::register_py(m)?;
    problems::procurement_removal_inventory::bindings::register_py(m)?;
    problems::random_yield_inventory::bindings::register_py(m)?;
    problems::spare_parts_inventory::bindings::register_py(m)?;
    Ok(())
}
