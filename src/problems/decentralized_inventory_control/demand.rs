use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemandDistributionKind {
    Poisson,
    Deterministic,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemandModel {
    pub kind: DemandDistributionKind,
    pub param1: f64,
}

pub fn parse_demand_distribution_kind(kind: &str) -> PyResult<DemandDistributionKind> {
    match kind {
        "poisson" | "Poisson" => Ok(DemandDistributionKind::Poisson),
        "deterministic" | "Deterministic" => Ok(DemandDistributionKind::Deterministic),
        _ => Err(PyValueError::new_err(format!(
            "unknown demand distribution '{kind}'; expected 'poisson' or 'deterministic'"
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

pub fn sample_demand(rng: &mut StdRng, model: &DemandModel) -> PyResult<usize> {
    validate_demand_model(model)?;
    match model.kind {
        DemandDistributionKind::Deterministic => Ok(model.param1.round().max(0.0) as usize),
        DemandDistributionKind::Poisson => {
            if model.param1 == 0.0 {
                return Ok(0);
            }
            let distribution = Poisson::new(model.param1).map_err(|err| {
                PyValueError::new_err(format!("invalid Poisson mean {}: {err}", model.param1))
            })?;
            Ok(distribution.sample(rng) as usize)
        }
    }
}
