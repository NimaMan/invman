use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualSourcingActionAdapter {
    Identity,
    SingleIndexTargets,
    DualIndexTargets,
    CappedDualIndexTargets,
    BaseSurgeTargets,
}

pub fn parse_action_adapter(action_adapter: &str) -> PyResult<DualSourcingActionAdapter> {
    match action_adapter {
        "identity" | "direct" | "direct_orders" => Ok(DualSourcingActionAdapter::Identity),
        "dual_sourcing_single_index_targets" | "single_index_targets" => {
            Ok(DualSourcingActionAdapter::SingleIndexTargets)
        }
        "dual_sourcing_dual_index_targets" | "dual_index_targets" => {
            Ok(DualSourcingActionAdapter::DualIndexTargets)
        }
        "dual_sourcing_capped_dual_index_targets" | "capped_dual_index_targets" => {
            Ok(DualSourcingActionAdapter::CappedDualIndexTargets)
        }
        "dual_sourcing_base_surge_targets" | "base_surge_targets" => {
            Ok(DualSourcingActionAdapter::BaseSurgeTargets)
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown dual-sourcing action adapter '{action_adapter}'"
        ))),
    }
}

pub fn action_from_controls(
    reduced_state: &[i64],
    controls: &[usize],
    action_adapter: DualSourcingActionAdapter,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
) -> PyResult<Vec<usize>> {
    let expedited_inventory_position = reduced_state[0];
    let regular_inventory_position = reduced_state.iter().sum::<i64>();
    match action_adapter {
        DualSourcingActionAdapter::Identity => {
            if controls.len() != 2 {
                return Err(PyValueError::new_err(
                    "identity dual-sourcing control vector must have length 2",
                ));
            }
            Ok(vec![
                controls[0].min(regular_max_order_size),
                controls[1].min(expedited_max_order_size),
            ])
        }
        DualSourcingActionAdapter::SingleIndexTargets => {
            if controls.len() != 2 {
                return Err(PyValueError::new_err(
                    "single-index target control vector must have length 2",
                ));
            }
            let s_e = controls[0] as i64;
            let s_r = controls[1].max(controls[0]) as i64;
            let expedited = (s_e - regular_inventory_position).max(0) as usize;
            let expedited = expedited.min(expedited_max_order_size);
            let regular = (s_r - regular_inventory_position - expedited as i64).max(0) as usize;
            Ok(vec![regular.min(regular_max_order_size), expedited])
        }
        DualSourcingActionAdapter::DualIndexTargets => {
            if controls.len() != 2 {
                return Err(PyValueError::new_err(
                    "dual-index target control vector must have length 2",
                ));
            }
            let s_e = controls[0] as i64;
            let s_r = controls[1].max(controls[0]) as i64;
            let expedited = (s_e - expedited_inventory_position).max(0) as usize;
            let expedited = expedited.min(expedited_max_order_size);
            let regular = (s_r - regular_inventory_position - expedited as i64).max(0) as usize;
            Ok(vec![regular.min(regular_max_order_size), expedited])
        }
        DualSourcingActionAdapter::CappedDualIndexTargets => {
            if controls.len() != 3 {
                return Err(PyValueError::new_err(
                    "capped dual-index target control vector must have length 3",
                ));
            }
            let s_e = controls[0] as i64;
            let s_r = controls[1].max(controls[0]) as i64;
            let cap_r = controls[2];
            let expedited = (s_e - expedited_inventory_position).max(0) as usize;
            let expedited = expedited.min(expedited_max_order_size);
            let desired_regular =
                (s_r - regular_inventory_position - expedited as i64).max(0) as usize;
            Ok(vec![
                desired_regular.min(cap_r).min(regular_max_order_size),
                expedited,
            ])
        }
        DualSourcingActionAdapter::BaseSurgeTargets => {
            if controls.len() != 2 {
                return Err(PyValueError::new_err(
                    "base-surge target control vector must have length 2",
                ));
            }
            let surge_level = controls[0] as i64;
            let regular_qty = controls[1];
            let expedited = (surge_level - expedited_inventory_position).max(0) as usize;
            Ok(vec![
                regular_qty.min(regular_max_order_size),
                expedited.min(expedited_max_order_size),
            ])
        }
    }
}
