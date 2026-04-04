use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemandDistributionKind {
    Poisson,
    CvNormal,
}

pub fn parse_demand_distribution_kind(kind: &str) -> PyResult<DemandDistributionKind> {
    match kind {
        "poisson" | "Poisson" => Ok(DemandDistributionKind::Poisson),
        "cv_normal" | "cvnormal" | "normal" | "Normal" => Ok(DemandDistributionKind::CvNormal),
        _ => Err(PyValueError::new_err(format!(
            "unknown demand distribution '{kind}'; expected 'poisson' or 'cv_normal'"
        ))),
    }
}

pub fn validate_distribution_parameters(
    demand_mean: f64,
    demand_cv: f64,
    kind: DemandDistributionKind,
) -> PyResult<()> {
    if !demand_mean.is_finite() || demand_mean < 0.0 {
        return Err(PyValueError::new_err(
            "demand mean must be finite and non-negative",
        ));
    }
    if !demand_cv.is_finite() || demand_cv < 0.0 {
        return Err(PyValueError::new_err(
            "demand_cv must be finite and non-negative",
        ));
    }
    if kind == DemandDistributionKind::Poisson && demand_mean == 0.0 {
        return Ok(());
    }
    Ok(())
}

pub fn demand_std(demand_mean: f64, demand_cv: f64, kind: DemandDistributionKind) -> f64 {
    match kind {
        DemandDistributionKind::Poisson => demand_mean.sqrt(),
        DemandDistributionKind::CvNormal => demand_mean * demand_cv,
    }
}

pub fn sample_demand(
    rng: &mut StdRng,
    demand_mean: f64,
    demand_cv: f64,
    kind: DemandDistributionKind,
) -> PyResult<f64> {
    validate_distribution_parameters(demand_mean, demand_cv, kind)?;
    if demand_mean <= 0.0 {
        return Ok(0.0);
    }

    match kind {
        DemandDistributionKind::Poisson => {
            let dist = Poisson::new(demand_mean).map_err(|err| {
                PyValueError::new_err(format!("invalid Poisson mean {demand_mean}: {err}"))
            })?;
            Ok(dist.sample(rng) as f64)
        }
        DemandDistributionKind::CvNormal => {
            let std = demand_std(demand_mean, demand_cv, kind);
            if std <= 0.0 {
                return Ok(demand_mean);
            }
            let dist = Normal::new(demand_mean, std).map_err(|err| {
                PyValueError::new_err(format!(
                    "invalid Normal parameters mean={demand_mean}, std={std}: {err}"
                ))
            })?;
            Ok(dist.sample(rng).max(0.0))
        }
    }
}
