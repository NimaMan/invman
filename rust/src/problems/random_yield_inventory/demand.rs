use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemandDistributionKind {
    Poisson,
    Deterministic,
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

pub fn sample_demand(
    rng: &mut StdRng,
    demand_mean: f64,
    kind: DemandDistributionKind,
) -> PyResult<f64> {
    if !demand_mean.is_finite() || demand_mean < 0.0 {
        return Err(PyValueError::new_err(
            "demand_mean must be finite and non-negative",
        ));
    }

    match kind {
        DemandDistributionKind::Deterministic => Ok(demand_mean),
        DemandDistributionKind::Poisson => {
            if demand_mean == 0.0 {
                return Ok(0.0);
            }
            let distribution = Poisson::new(demand_mean).map_err(|err| {
                PyValueError::new_err(format!(
                    "invalid Poisson mean {demand_mean}: {err}"
                ))
            })?;
            Ok(distribution.sample(rng) as f64)
        }
    }
}
