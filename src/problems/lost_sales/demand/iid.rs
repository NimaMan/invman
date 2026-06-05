use rand::rngs::StdRng;
use rand_distr::{Distribution, Geometric, Poisson};

pub struct PoissonDemand {
    dist: Poisson<f64>,
}

impl PoissonDemand {
    pub fn new(demand_rate: f64) -> Result<Self, String> {
        Poisson::new(demand_rate)
            .map(|dist| Self { dist })
            .map_err(|err| format!("invalid Poisson demand_rate: {err}"))
    }

    pub fn sample(&self, rng: &mut StdRng) -> i64 {
        self.dist.sample(rng) as i64
    }
}

pub struct GeometricDemand {
    dist: Geometric,
}

impl GeometricDemand {
    pub fn new(demand_rate: f64) -> Result<Self, String> {
        let success_prob = 1.0 / (1.0 + demand_rate);
        Geometric::new(success_prob)
            .map(|dist| Self { dist })
            .map_err(|err| format!("invalid Geometric demand_rate: {err}"))
    }

    pub fn sample(&self, rng: &mut StdRng) -> i64 {
        self.dist.sample(rng) as i64
    }
}
