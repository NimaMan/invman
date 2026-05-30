// Core order-quantity computation for the three vanilla lost-sales heuristics.
//
// The lost-sales inventory state is the order pipeline `state = [x_0, x_1, ...,
// x_{L-1}]`, where `x_0` is on-hand inventory plus the order arriving this
// period and `x_j` (j >= 1) is the order placed `L - j` periods ago that will
// arrive in `j` periods. An order quantity `z >= 0` appends a new slot to the
// pipeline. After demand `d` is realised the pipeline advances: on-hand becomes
// `(x_0 - d)^+ + x_1` and every later slot shifts down one position.
//
// ----------------------------------------------------------------------------
// One-period cost
// ----------------------------------------------------------------------------
// `get_one_period_cost(y)` is the expected single-period newsvendor cost of
// reaching inventory position `y` before demand:
//   c_p * y + c_h * E[(y - D)^+] + c_s * E[(D - y)^+]
// computed as an expectation over the truncated demand support.
//
// ----------------------------------------------------------------------------
// Lookahead cost recursion `q_l`
// ----------------------------------------------------------------------------
// `q_l(state, l)` is the expected discounted cost-to-go over an `l`-period
// lookahead where NO further ordering is allowed (a myopic continuation). At
// depth 0 it is the one-period cost of the first slot; at depth l it takes the
// demand expectation of `q_{l-1}` over the advanced pipeline, discounted by
// `beta`. Results are memoised by `(l, state)`.
//
// `q_l_from_state_action(state, z)` appends order `z` and evaluates the
// lead-time-deep `q_l` recursion: the expected discounted cost incurred when
// `z` is the only future order. This is the Myopic-1 action value.
//
// ----------------------------------------------------------------------------
// best_quantity (line search)
// ----------------------------------------------------------------------------
// Each heuristic chooses the order quantity by scanning `z = 0, 1, 2, ...` and
// stopping at the first local minimum of the (convex-in-practice) action value,
// capped at `order_search_upper_bound`. `best_quantity` returns the minimiser
// and its value.
//
// ----------------------------------------------------------------------------
// Myopic-1
// ----------------------------------------------------------------------------
// `myopic_1_order_quantity` minimises `q_l_from_state_action` over `z`: a single
// lead-time-deep newsvendor lookahead assuming no future ordering.
//
// ----------------------------------------------------------------------------
// Myopic-2
// ----------------------------------------------------------------------------
// `myopic_2_q_value(state, z)` is the Myopic-1 action value `q_l_from_state_action`
// PLUS a discounted one-step continuation: the expected Myopic-1 quantity chosen
// in the next state (over next-period demand). `myopic_2_order_quantity`
// minimises this over `z`. This two-period view typically beats Myopic-1.
//
// ----------------------------------------------------------------------------
// Standard Vector Base Stock (SVBS)
// ----------------------------------------------------------------------------
// `standard_vector_base_stock_levels` computes, for each pipeline position
// `l = 0..=L`, a base-stock level `S_l`: the smallest `s` such that the survival
// function of the `(L - l + 1)`-period demand at `s` is below the critical
// fractile `(c_p + c_h) / (c_h + c_s)`. The order quantity is
//   min over l of  (S_l - sum of pipeline from position l onward)
// clamped to `[0, order_search_upper_bound]`. Levels are cached.

use std::collections::HashMap;

use crate::problems::lost_sales::demand::{LostSalesDemandConfig, LostSalesDemandKind};
use crate::problems::lost_sales::vanilla::heuristics::demand_support::{
    cumulative_demand_cdf, iid_demand_support,
};

/// Per-instance configuration for the lost-sales heuristics and their rollout.
///
/// Named `LostSalesHeuristicVerificationConfig` for backwards compatibility with
/// the flownet verification module, which historically owned this type.
#[derive(Clone, Copy)]
pub struct LostSalesHeuristicVerificationConfig {
    pub reference_name: &'static str,
    pub horizon: usize,
    pub seed: u64,
    pub warm_up_periods_ratio: f64,
    pub order_search_upper_bound: usize,
    pub lead_time: usize,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub procurement_cost: f64,
    pub fixed_order_cost: f64,
    pub heuristic_discount_factor: f64,
    pub demand_config: LostSalesDemandConfig,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RecursiveCostKey {
    lookahead_depth: usize,
    state: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ActionStateKey {
    order_quantity: usize,
    state: Vec<usize>,
}

/// Stateful evaluator that computes heuristic order quantities, caching the
/// one-period costs, lookahead values, Myopic-2 values, and SVBS levels.
pub struct LostSalesHeuristicEvaluator {
    config: LostSalesHeuristicVerificationConfig,
    demand_support: Vec<(usize, f64)>,
    one_period_cache: HashMap<usize, f64>,
    q_l_cache: HashMap<RecursiveCostKey, f64>,
    q_l_from_state_action_cache: HashMap<ActionStateKey, f64>,
    myopic2_cache: HashMap<ActionStateKey, f64>,
    svbs_levels: Option<Vec<usize>>,
}

impl LostSalesHeuristicEvaluator {
    pub fn new(config: LostSalesHeuristicVerificationConfig) -> Result<Self, String> {
        validate_heuristic_config(&config)?;
        Ok(Self {
            demand_support: iid_demand_support(&config.demand_config)?,
            config,
            one_period_cache: HashMap::new(),
            q_l_cache: HashMap::new(),
            q_l_from_state_action_cache: HashMap::new(),
            myopic2_cache: HashMap::new(),
            svbs_levels: None,
        })
    }

    fn get_one_period_cost(&mut self, y: usize) -> f64 {
        if let Some(cost) = self.one_period_cache.get(&y) {
            return *cost;
        }

        let expected_overage = self
            .demand_support
            .iter()
            .map(|(demand, probability)| y.saturating_sub(*demand) as f64 * probability)
            .sum::<f64>();
        let expected_underage = self
            .demand_support
            .iter()
            .map(|(demand, probability)| demand.saturating_sub(y) as f64 * probability)
            .sum::<f64>();
        let total_cost = self.config.procurement_cost * y as f64
            + self.config.holding_cost * expected_overage
            + self.config.shortage_cost * expected_underage;
        self.one_period_cache.insert(y, total_cost);
        total_cost
    }

    fn q_l(&mut self, state: &[usize], lookahead_depth: usize) -> Result<f64, String> {
        let key = RecursiveCostKey {
            lookahead_depth,
            state: state.to_vec(),
        };
        if let Some(value) = self.q_l_cache.get(&key) {
            return Ok(*value);
        }

        let value = if lookahead_depth == 0 {
            self.get_one_period_cost(state[0])
        } else {
            let demand_support = self.demand_support.clone();
            let expected_cost = demand_support
                .iter()
                .map(|(demand, probability)| {
                    let next_state =
                        state_after_demand_and_pipeline_advance(state, *demand, lookahead_depth);
                    self.q_l(&next_state, lookahead_depth - 1)
                        .map(|cost| probability * cost)
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .sum::<f64>();
            self.config.heuristic_discount_factor * expected_cost
        };

        self.q_l_cache.insert(key, value);
        Ok(value)
    }

    fn q_l_from_state_action(
        &mut self,
        state: &[usize],
        order_quantity: usize,
    ) -> Result<f64, String> {
        let key = ActionStateKey {
            order_quantity,
            state: state.to_vec(),
        };
        if let Some(value) = self.q_l_from_state_action_cache.get(&key) {
            return Ok(*value);
        }

        let extended_state = state_with_appended_order(state, order_quantity);
        let demand_support = self.demand_support.clone();
        let expected_cost = demand_support
            .iter()
            .map(|(demand, probability)| {
                let next_state = state_after_demand_and_pipeline_advance(
                    &extended_state,
                    *demand,
                    self.config.lead_time,
                );
                self.q_l(&next_state, self.config.lead_time - 1)
                    .map(|cost| probability * cost)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<f64>();
        let value = self.config.heuristic_discount_factor * expected_cost;

        self.q_l_from_state_action_cache.insert(key, value);
        Ok(value)
    }

    fn best_quantity(
        &mut self,
        state: &[usize],
        evaluator: fn(&mut Self, &[usize], usize) -> Result<f64, String>,
    ) -> Result<(usize, f64), String> {
        let mut best_quantity = 0usize;
        let mut current_value = evaluator(self, state, 0)?;
        let mut previous_value = f64::INFINITY;

        while best_quantity < self.config.order_search_upper_bound && previous_value > current_value
        {
            best_quantity += 1;
            previous_value = current_value;
            current_value = evaluator(self, state, best_quantity)?;
        }

        if previous_value > current_value {
            Ok((best_quantity, current_value))
        } else {
            Ok((best_quantity.saturating_sub(1), previous_value))
        }
    }

    pub fn myopic_1_order_quantity(&mut self, state: &[usize]) -> Result<(usize, f64), String> {
        self.best_quantity(state, Self::q_l_from_state_action)
    }

    fn myopic_2_q_value(&mut self, state: &[usize], order_quantity: usize) -> Result<f64, String> {
        let key = ActionStateKey {
            order_quantity,
            state: state.to_vec(),
        };
        if let Some(value) = self.myopic2_cache.get(&key) {
            return Ok(*value);
        }

        let extended_state = state_with_appended_order(state, order_quantity);
        let q_hat_z = self.q_l_from_state_action(state, order_quantity)?;
        let demand_support = self.demand_support.clone();
        let future_value = demand_support
            .iter()
            .map(|(demand, probability)| {
                let next_state = state_after_demand_and_pipeline_advance(
                    &extended_state,
                    *demand,
                    self.config.lead_time,
                );
                self.myopic_1_order_quantity(&next_state)
                    .map(|(_, q_hat)| probability * q_hat)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<f64>();
        let value = q_hat_z + self.config.heuristic_discount_factor * future_value;

        self.myopic2_cache.insert(key, value);
        Ok(value)
    }

    pub fn myopic_2_order_quantity(&mut self, state: &[usize]) -> Result<(usize, f64), String> {
        self.best_quantity(state, Self::myopic_2_q_value)
    }

    pub fn standard_vector_base_stock_levels(&mut self) -> Result<&[usize], String> {
        if self.svbs_levels.is_none() {
            let levels = (0..=self.config.lead_time)
                .map(|l| {
                    let mut s = 0usize;
                    while (1.0
                        - cumulative_demand_cdf(
                            &self.config.demand_config,
                            s,
                            self.config.lead_time - l + 1,
                        )?)
                        >= critical_fractile(&self.config)
                    {
                        s += 1;
                    }
                    Ok(s)
                })
                .collect::<Result<Vec<_>, String>>()?;
            self.svbs_levels = Some(levels);
        }

        Ok(self.svbs_levels.as_deref().unwrap_or(&[]))
    }

    pub fn standard_vector_base_stock_order_quantity(
        &mut self,
        state: &[usize],
    ) -> Result<usize, String> {
        let levels = self.standard_vector_base_stock_levels()?.to_vec();
        let order_quantity = (0..=self.config.lead_time)
            .map(|l| {
                let pipeline_partial_sum =
                    order_pipeline_partial_sum(self.config.lead_time, l, state);
                levels[l].saturating_sub(pipeline_partial_sum)
            })
            .min()
            .unwrap_or(0);
        Ok(order_quantity.min(self.config.order_search_upper_bound))
    }
}

/// Validate that the configuration is supported by the heuristics.
pub fn validate_heuristic_config(
    config: &LostSalesHeuristicVerificationConfig,
) -> Result<(), String> {
    if config.lead_time < 1 {
        return Err(String::from("lead_time must be at least 1"));
    }
    if config.horizon < 1 {
        return Err(String::from("horizon must be at least 1"));
    }
    if !(0.0..=1.0).contains(&config.warm_up_periods_ratio) {
        return Err(String::from("warm_up_periods_ratio must be in [0, 1]"));
    }
    if !config.holding_cost.is_finite()
        || !config.shortage_cost.is_finite()
        || !config.procurement_cost.is_finite()
        || !config.fixed_order_cost.is_finite()
    {
        return Err(String::from("cost coefficients must be finite"));
    }
    if !matches!(
        config.demand_config.kind,
        LostSalesDemandKind::Poisson
            | LostSalesDemandKind::Geometric
            | LostSalesDemandKind::MarkovModulatedPoisson2
    ) {
        return Err(String::from(
            "current lost-sales heuristic verification supports IID Poisson, IID Geometric, and MarkovModulatedPoisson2 demand",
        ));
    }
    Ok(())
}

fn critical_fractile(config: &LostSalesHeuristicVerificationConfig) -> f64 {
    (config.procurement_cost + config.holding_cost) / (config.holding_cost + config.shortage_cost)
}

fn order_pipeline_partial_sum(lead_time: usize, l: usize, state: &[usize]) -> usize {
    if l == lead_time {
        0
    } else {
        state[l..].iter().sum()
    }
}

fn state_with_appended_order(state: &[usize], order_quantity: usize) -> Vec<usize> {
    let mut extended_state = state.to_vec();
    extended_state.push(order_quantity);
    extended_state
}

fn state_after_demand_and_pipeline_advance(
    state: &[usize],
    demand: usize,
    lookahead_depth: usize,
) -> Vec<usize> {
    let mut next_state = state[..state.len() - 1].to_vec();
    next_state[0] = state[0].saturating_sub(demand).saturating_add(state[1]);
    next_state[1..lookahead_depth].copy_from_slice(&state[2..(lookahead_depth + 1)]);
    next_state
}
