use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DemandRange {
    pub low: usize,
    pub high: usize,
}

pub fn validate_demand_ranges(demand_ranges: &[DemandRange]) -> PyResult<()> {
    if demand_ranges.is_empty() {
        return Err(PyValueError::new_err(
            "joint_replenishment requires at least one item demand range",
        ));
    }
    for (index, range) in demand_ranges.iter().enumerate() {
        if range.low > range.high {
            return Err(PyValueError::new_err(format!(
                "demand range {index} has low {} > high {}",
                range.low, range.high
            )));
        }
    }
    Ok(())
}

pub fn support(range: DemandRange) -> PyResult<Vec<(usize, f64)>> {
    if range.low > range.high {
        return Err(PyValueError::new_err(format!(
            "invalid demand range [{}, {}]",
            range.low, range.high
        )));
    }
    let cardinality = range.high - range.low + 1;
    let probability = 1.0 / cardinality as f64;
    Ok((range.low..=range.high)
        .map(|value| (value, probability))
        .collect())
}

pub fn sample_demands<R: Rng + ?Sized>(
    rng: &mut R,
    demand_ranges: &[DemandRange],
) -> PyResult<Vec<usize>> {
    validate_demand_ranges(demand_ranges)?;
    Ok(demand_ranges
        .iter()
        .map(|range| rng.gen_range(range.low..=range.high))
        .collect())
}
