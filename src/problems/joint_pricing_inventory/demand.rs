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

pub fn validate_price_ladder(price_levels: &[f64], demand_means: &[f64]) -> PyResult<()> {
    if price_levels.is_empty() {
        return Err(PyValueError::new_err("price_levels must be non-empty"));
    }
    if price_levels.len() != demand_means.len() {
        return Err(PyValueError::new_err(
            "price_levels and demand_means must have the same length",
        ));
    }
    if price_levels
        .iter()
        .any(|price| !price.is_finite() || *price <= 0.0)
    {
        return Err(PyValueError::new_err(
            "price_levels must be finite and strictly positive",
        ));
    }
    if demand_means
        .iter()
        .any(|mean| !mean.is_finite() || *mean < 0.0)
    {
        return Err(PyValueError::new_err(
            "demand_means must be finite and non-negative",
        ));
    }
    Ok(())
}

pub fn sample_demand(
    rng: &mut StdRng,
    price_index: usize,
    demand_means: &[f64],
    kind: DemandDistributionKind,
) -> PyResult<usize> {
    if price_index >= demand_means.len() {
        return Err(PyValueError::new_err(format!(
            "price_index {price_index} out of range for {} demand means",
            demand_means.len()
        )));
    }
    let demand_mean = demand_means[price_index];
    match kind {
        DemandDistributionKind::Deterministic => Ok(demand_mean.round().max(0.0) as usize),
        DemandDistributionKind::Poisson => {
            if demand_mean == 0.0 {
                return Ok(0);
            }
            let distribution = Poisson::new(demand_mean).map_err(|err| {
                PyValueError::new_err(format!("invalid Poisson mean {demand_mean}: {err}"))
            })?;
            Ok(distribution.sample(rng) as usize)
        }
    }
}
