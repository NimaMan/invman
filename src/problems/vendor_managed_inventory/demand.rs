use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::Rng;
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
) -> PyResult<usize> {
    if !demand_mean.is_finite() || demand_mean < 0.0 {
        return Err(PyValueError::new_err(
            "demand_mean must be finite and non-negative",
        ));
    }

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemandPathSummary {
    pub sales: f64,
    pub lost_sales: f64,
    pub positive_inventory_time_area: f64,
    pub ending_inventory: f64,
}

pub fn sample_discrete_uniform_half_step(rng: &mut StdRng, low: f64, high: f64) -> PyResult<f64> {
    if !low.is_finite() || !high.is_finite() || low < 0.0 || high < low {
        return Err(PyValueError::new_err(
            "demand-size bounds must be finite, non-negative, and satisfy high >= low",
        ));
    }
    let steps = ((high - low) * 2.0).round();
    if (low + steps * 0.5 - high).abs() > 1e-9 {
        return Err(PyValueError::new_err(
            "demand-size bounds must lie on a 0.5-spaced discrete grid",
        ));
    }
    let draw = rng.gen_range(0..=(steps as usize));
    Ok(low + 0.5 * draw as f64)
}

pub fn simulate_compound_poisson_interval(
    rng: &mut StdRng,
    starting_inventory: f64,
    arrival_rate: f64,
    demand_size_low: f64,
    demand_size_high: f64,
    duration: f64,
) -> PyResult<DemandPathSummary> {
    if !starting_inventory.is_finite() || starting_inventory < -1e-9 {
        return Err(PyValueError::new_err(format!(
            "starting_inventory must be finite and non-negative, found {}",
            starting_inventory
        )));
    }
    if !arrival_rate.is_finite() || arrival_rate < 0.0 {
        return Err(PyValueError::new_err(
            "arrival_rate must be finite and non-negative",
        ));
    }
    if !duration.is_finite() || duration < 0.0 {
        return Err(PyValueError::new_err(
            "duration must be finite and non-negative",
        ));
    }

    let mut elapsed = 0.0;
    let mut inventory = starting_inventory.max(0.0);
    let mut sales = 0.0;
    let mut lost_sales = 0.0;
    let mut positive_inventory_time_area = 0.0;

    if arrival_rate == 0.0 || duration == 0.0 {
        return Ok(DemandPathSummary {
            sales,
            lost_sales,
            positive_inventory_time_area: inventory * duration,
            ending_inventory: inventory,
        });
    }

    let mean_arrivals = arrival_rate * duration;
    let arrivals = if mean_arrivals == 0.0 {
        0usize
    } else {
        let distribution = Poisson::new(mean_arrivals).map_err(|err| {
            PyValueError::new_err(format!(
                "invalid compound Poisson mean {}: {}",
                mean_arrivals, err
            ))
        })?;
        distribution.sample(rng) as usize
    };
    if arrivals == 0 {
        return Ok(DemandPathSummary {
            sales,
            lost_sales,
            positive_inventory_time_area: inventory * duration,
            ending_inventory: inventory,
        });
    }

    let mut event_times = Vec::with_capacity(arrivals);
    for _ in 0..arrivals {
        event_times.push(rng.gen_range(0.0..duration));
    }
    event_times.sort_by(|a, b| a.partial_cmp(b).expect("event times must be finite"));

    for event_time in event_times {
        positive_inventory_time_area += inventory * (event_time - elapsed);
        elapsed = event_time;
        let demand_size =
            sample_discrete_uniform_half_step(rng, demand_size_low, demand_size_high)?;
        let realized_sales = inventory.min(demand_size);
        sales += realized_sales;
        inventory -= realized_sales;
        lost_sales += demand_size - realized_sales;
    }
    positive_inventory_time_area += inventory * (duration - elapsed);

    Ok(DemandPathSummary {
        sales,
        lost_sales,
        positive_inventory_time_area,
        ending_inventory: inventory,
    })
}
