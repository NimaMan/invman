use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::Rng;
use rand_distr::{Distribution, Normal, Poisson};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemandDistributionKind {
    Poisson,
    RoundedNormal,
    DiscreteUniform,
    Deterministic,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemandModel {
    pub kind: DemandDistributionKind,
    pub param1: f64,
    pub param2: f64,
}

pub fn parse_demand_distribution_kind(kind: &str) -> PyResult<DemandDistributionKind> {
    match kind {
        "poisson" => Ok(DemandDistributionKind::Poisson),
        "normal" | "rounded_normal" | "gaussian" => Ok(DemandDistributionKind::RoundedNormal),
        "uniform" | "discrete_uniform" => Ok(DemandDistributionKind::DiscreteUniform),
        "deterministic" => Ok(DemandDistributionKind::Deterministic),
        _ => Err(PyValueError::new_err(format!(
            "unknown demand distribution '{kind}'; expected 'poisson', 'rounded_normal', 'discrete_uniform', or 'deterministic'"
        ))),
    }
}

pub fn validate_demand_model(model: &DemandModel) -> PyResult<()> {
    match model.kind {
        DemandDistributionKind::Poisson => {
            if !model.param1.is_finite() || model.param1 < 0.0 {
                return Err(PyValueError::new_err(
                    "Poisson mean must be finite and non-negative",
                ));
            }
        }
        DemandDistributionKind::RoundedNormal => {
            if !model.param1.is_finite() || model.param1 < 0.0 {
                return Err(PyValueError::new_err(
                    "normal demand mean must be finite and non-negative",
                ));
            }
            if !model.param2.is_finite() || model.param2 < 0.0 {
                return Err(PyValueError::new_err(
                    "normal demand std must be finite and non-negative",
                ));
            }
        }
        DemandDistributionKind::DiscreteUniform => {
            if !model.param1.is_finite() || !model.param2.is_finite() {
                return Err(PyValueError::new_err(
                    "uniform demand bounds must be finite",
                ));
            }
            if model.param1 < 0.0 || model.param2 < 0.0 {
                return Err(PyValueError::new_err(
                    "uniform demand bounds must be non-negative",
                ));
            }
            if model.param1.round() > model.param2.round() {
                return Err(PyValueError::new_err(
                    "uniform low bound cannot exceed the high bound",
                ));
            }
        }
        DemandDistributionKind::Deterministic => {
            if !model.param1.is_finite() || model.param1 < 0.0 {
                return Err(PyValueError::new_err(
                    "deterministic demand must be finite and non-negative",
                ));
            }
        }
    }
    Ok(())
}

pub fn validate_demand_models(models: &[DemandModel]) -> PyResult<()> {
    if models.is_empty() {
        return Err(PyValueError::new_err(
            "one_warehouse_multi_retailer requires at least one retailer demand model",
        ));
    }
    for model in models {
        validate_demand_model(model)?;
    }
    Ok(())
}

pub fn mean_demand(model: &DemandModel) -> PyResult<f64> {
    validate_demand_model(model)?;
    Ok(match model.kind {
        DemandDistributionKind::Poisson => model.param1,
        DemandDistributionKind::RoundedNormal => model.param1,
        DemandDistributionKind::DiscreteUniform => {
            (model.param1.round() + model.param2.round()) / 2.0
        }
        DemandDistributionKind::Deterministic => model.param1,
    })
}

pub fn sample_demand<R: Rng + ?Sized>(rng: &mut R, model: &DemandModel) -> PyResult<usize> {
    validate_demand_model(model)?;
    Ok(match model.kind {
        DemandDistributionKind::Poisson => {
            if model.param1 == 0.0 {
                0
            } else {
                let distribution = Poisson::new(model.param1).map_err(|err| {
                    PyValueError::new_err(format!(
                        "invalid Poisson mean {}: {err}",
                        model.param1
                    ))
                })?;
                distribution.sample(rng).round().max(0.0) as usize
            }
        }
        DemandDistributionKind::RoundedNormal => {
            if model.param2 == 0.0 {
                model.param1.round().max(0.0) as usize
            } else {
                let distribution = Normal::new(model.param1, model.param2).map_err(|err| {
                    PyValueError::new_err(format!(
                        "invalid normal demand parameters ({}, {}): {err}",
                        model.param1, model.param2
                    ))
                })?;
                distribution.sample(rng).round().max(0.0) as usize
            }
        }
        DemandDistributionKind::DiscreteUniform => {
            let low = model.param1.round().max(0.0) as usize;
            let high = model.param2.round().max(low as f64) as usize;
            rng.gen_range(low..=high)
        }
        DemandDistributionKind::Deterministic => model.param1.round().max(0.0) as usize,
    })
}
