use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution, Poisson};

pub struct MarkovModulatedPoisson2Demand {
    low_dist: Poisson<f64>,
    high_dist: Poisson<f64>,
    p00: f64,
    p11: f64,
    state: usize,
}

impl MarkovModulatedPoisson2Demand {
    pub fn new(
        demand_lambda_low: f64,
        demand_lambda_high: f64,
        demand_p00: f64,
        demand_p11: f64,
        rng: &mut StdRng,
    ) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&demand_p00) {
            return Err("demand_p00 must be in [0, 1]".to_string());
        }
        if !(0.0..=1.0).contains(&demand_p11) {
            return Err("demand_p11 must be in [0, 1]".to_string());
        }
        let denominator = 2.0 - demand_p00 - demand_p11;
        if denominator <= 0.0 {
            return Err("demand_p00 + demand_p11 must be strictly less than 2".to_string());
        }

        let low_dist = Poisson::new(demand_lambda_low)
            .map_err(|err| format!("invalid MarkovModulatedPoisson2 demand_lambda_low: {err}"))?;
        let high_dist = Poisson::new(demand_lambda_high)
            .map_err(|err| format!("invalid MarkovModulatedPoisson2 demand_lambda_high: {err}"))?;
        let stationary_prob_high = (1.0 - demand_p00) / denominator;
        let state = if rng.gen::<f64>() < stationary_prob_high {
            1
        } else {
            0
        };

        Ok(Self {
            low_dist,
            high_dist,
            p00: demand_p00,
            p11: demand_p11,
            state,
        })
    }

    pub fn sample(&mut self, rng: &mut StdRng) -> i64 {
        let demand = if self.state == 0 {
            self.low_dist.sample(rng) as i64
        } else {
            self.high_dist.sample(rng) as i64
        };
        let transition_draw = rng.gen::<f64>();
        self.state = if self.state == 0 {
            if transition_draw < self.p00 {
                0
            } else {
                1
            }
        } else if transition_draw < self.p11 {
            1
        } else {
            0
        };
        demand
    }
}
