use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemandDistributionKind {
    Poisson,
    Deterministic,
    Normal,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemandModel {
    pub kind: DemandDistributionKind,
    pub param1: f64,
    pub param2: f64,
}

pub fn parse_demand_distribution_kind(kind: &str) -> PyResult<DemandDistributionKind> {
    match kind {
        "poisson" | "Poisson" => Ok(DemandDistributionKind::Poisson),
        "deterministic" | "Deterministic" => Ok(DemandDistributionKind::Deterministic),
        "normal" | "Normal" => Ok(DemandDistributionKind::Normal),
        _ => Err(PyValueError::new_err(format!(
            "unknown demand distribution '{kind}'; expected 'poisson', 'deterministic', or 'normal'"
        ))),
    }
}

pub fn validate_demand_model(model: &DemandModel) -> PyResult<()> {
    if !model.param1.is_finite() || model.param1 < 0.0 {
        return Err(PyValueError::new_err(
            "demand model param1 must be finite and non-negative",
        ));
    }
    if !model.param2.is_finite() || model.param2 < 0.0 {
        return Err(PyValueError::new_err(
            "demand model param2 must be finite and non-negative",
        ));
    }
    if matches!(model.kind, DemandDistributionKind::Normal) && model.param2 == 0.0 {
        return Err(PyValueError::new_err(
            "normal demand requires a strictly positive standard deviation",
        ));
    }
    Ok(())
}

pub fn demand_mean(model: &DemandModel) -> f64 {
    model.param1
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
        DemandDistributionKind::Normal => {
            let distribution = Normal::new(model.param1, model.param2).map_err(|err| {
                PyValueError::new_err(format!(
                    "invalid normal demand parameters mean={} std={}: {err}",
                    model.param1, model.param2
                ))
            })?;
            Ok(distribution.sample(rng).round().max(0.0) as usize)
        }
    }
}
