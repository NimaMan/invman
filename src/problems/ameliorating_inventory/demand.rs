use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::Rng;
use rand_distr::{Distribution, Poisson};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DemandDistributionKind {
    Deterministic,
    Poisson,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemandModel {
    pub kind: DemandDistributionKind,
    pub param1: f64,
}

pub fn parse_demand_distribution_kind(value: &str) -> PyResult<DemandDistributionKind> {
    match value {
        "deterministic" => Ok(DemandDistributionKind::Deterministic),
        "poisson" => Ok(DemandDistributionKind::Poisson),
        _ => Err(PyValueError::new_err(format!(
            "unsupported demand distribution '{value}'"
        ))),
    }
}

pub fn validate_demand_model(model: &DemandModel) -> PyResult<()> {
    if !model.param1.is_finite() || model.param1 < 0.0 {
        return Err(PyValueError::new_err(
            "demand model parameter must be finite and non-negative",
        ));
    }
    Ok(())
}

pub fn mean_demand(model: &DemandModel) -> PyResult<f64> {
    validate_demand_model(model)?;
    Ok(model.param1)
}

pub fn sample_demand<R: Rng + ?Sized>(rng: &mut R, model: &DemandModel) -> PyResult<usize> {
    validate_demand_model(model)?;
    match model.kind {
        DemandDistributionKind::Deterministic => Ok(model.param1.round().max(0.0) as usize),
        DemandDistributionKind::Poisson => {
            if model.param1 == 0.0 {
                return Ok(0);
            }
            let distribution = Poisson::new(model.param1)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(distribution.sample(rng).round().max(0.0) as usize)
        }
    }
}
