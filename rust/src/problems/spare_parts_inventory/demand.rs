use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::Rng;
use rand_distr::{Binomial, Distribution};

pub fn validate_failure_probability(failure_probability: f64) -> PyResult<()> {
    if !failure_probability.is_finite() || !(0.0..=1.0).contains(&failure_probability) {
        return Err(PyValueError::new_err(
            "failure_probability must be finite and lie in [0, 1]",
        ));
    }
    Ok(())
}

pub fn sample_failures<R: Rng + ?Sized>(
    rng: &mut R,
    operational_units: usize,
    failure_probability: f64,
) -> PyResult<usize> {
    validate_failure_probability(failure_probability)?;
    if operational_units == 0 || failure_probability == 0.0 {
        return Ok(0);
    }
    if failure_probability == 1.0 {
        return Ok(operational_units);
    }
    let distribution = Binomial::new(operational_units as u64, failure_probability)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(distribution.sample(rng) as usize)
}

pub fn failure_probabilities(
    operational_units: usize,
    failure_probability: f64,
) -> PyResult<Vec<f64>> {
    validate_failure_probability(failure_probability)?;
    if operational_units == 0 {
        return Ok(vec![1.0]);
    }
    if failure_probability == 0.0 {
        let mut probabilities = vec![0.0; operational_units + 1];
        probabilities[0] = 1.0;
        return Ok(probabilities);
    }
    if failure_probability == 1.0 {
        let mut probabilities = vec![0.0; operational_units + 1];
        probabilities[operational_units] = 1.0;
        return Ok(probabilities);
    }

    let n = operational_units as f64;
    let p = failure_probability;
    let q = 1.0 - p;
    let mut probabilities = vec![0.0; operational_units + 1];
    probabilities[0] = q.powf(n);
    for failures in 0..operational_units {
        let failures_f64 = failures as f64;
        probabilities[failures + 1] =
            probabilities[failures] * ((n - failures_f64) / (failures_f64 + 1.0)) * (p / q);
    }
    let total_probability = probabilities.iter().sum::<f64>();
    if total_probability <= 0.0 || !total_probability.is_finite() {
        return Err(PyValueError::new_err(
            "failed to build a valid binomial failure distribution",
        ));
    }
    for probability in probabilities.iter_mut() {
        *probability /= total_probability;
    }
    Ok(probabilities)
}
