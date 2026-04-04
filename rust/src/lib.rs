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
    problems::lost_sales::bindings::register_py(m)?;
    problems::lost_sales_fixed_order_cost::bindings::register_py(m)?;
    problems::dual_sourcing::bindings::register_py(m)?;
    problems::multi_echelon::bindings::register_py(m)?;
    problems::perishable_inventory::bindings::register_py(m)?;
    Ok(())
}
