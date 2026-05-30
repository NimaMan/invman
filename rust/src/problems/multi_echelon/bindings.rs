use pyo3::prelude::*;

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::problems::multi_echelon::divergent_special_delivery::bindings::register_py(m)?;
    crate::problems::multi_echelon::general_backorder_fixed_cost::bindings::register_py(m)?;
    crate::problems::multi_echelon::general_network::bindings::register_py(m)?;
    Ok(())
}
