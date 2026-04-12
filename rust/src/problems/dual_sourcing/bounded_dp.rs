use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::problems::dual_sourcing::env::{epoch_cost, step_state};
use crate::problems::dual_sourcing::heuristics::{
    named_policy_action, search_capped_dual_index_from_demands, search_dual_index_from_demands,
    search_single_index_from_demands, search_tailored_base_surge_from_demands, target_upper_bound,
};
use crate::problems::dual_sourcing::literature::{
    get_figure_9_gap_reference, get_reference_instance, DualSourcingReferenceInstance,
    PublishedOptimalityGapReference,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundedDpConfig {
    pub inventory_lower: i64,
    pub inventory_upper: i64,
    pub tolerance: f64,
    pub max_iterations: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AverageCostPolicyEvaluation {
    pub average_cost: f64,
    pub first_action: [usize; 2],
    pub iterations: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeuristicBenchmarkResult {
    pub policy_name: &'static str,
    pub params: Vec<usize>,
    pub search_cost: f64,
    pub average_cost: f64,
    pub first_action: [usize; 2],
    pub optimality_gap_pct: f64,
    pub published_optimality_gap_pct: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkReport {
    pub reference_name: String,
    pub initial_state: Vec<i64>,
    pub optimal: AverageCostPolicyEvaluation,
    pub heuristics: Vec<HeuristicBenchmarkResult>,
}

fn validate_config(config: &BoundedDpConfig) -> PyResult<()> {
    if config.inventory_lower > config.inventory_upper {
        return Err(PyValueError::new_err(
            "inventory_lower must not exceed inventory_upper",
        ));
    }
    if config.tolerance <= 0.0 {
        return Err(PyValueError::new_err("tolerance must be positive"));
    }
    if config.max_iterations == 0 {
        return Err(PyValueError::new_err("max_iterations must be positive"));
    }
    Ok(())
}

fn deterministic_initial_state(reference: &DualSourcingReferenceInstance) -> Vec<i64> {
    let mean_demand = 0.5 * (reference.demand_low + reference.demand_high) as f64;
    let mut state = vec![((reference.regular_lead_time + 1) as f64 * mean_demand).round() as i64];
    state.extend(vec![0; reference.regular_lead_time.saturating_sub(1)]);
    state
}

fn enumerate_state_space(
    reference: &DualSourcingReferenceInstance,
    config: &BoundedDpConfig,
) -> Vec<Vec<i64>> {
    fn recurse(
        dim: usize,
        total_dims: usize,
        max_regular: usize,
        inventory_lower: i64,
        inventory_upper: i64,
        partial: &mut Vec<i64>,
        output: &mut Vec<Vec<i64>>,
    ) {
        if dim == total_dims {
            output.push(partial.clone());
            return;
        }
        if dim == 0 {
            for inventory in inventory_lower..=inventory_upper {
                partial.push(inventory);
                recurse(
                    dim + 1,
                    total_dims,
                    max_regular,
                    inventory_lower,
                    inventory_upper,
                    partial,
                    output,
                );
                partial.pop();
            }
            return;
        }
        for pipeline in 0..=max_regular {
            partial.push(pipeline as i64);
            recurse(
                dim + 1,
                total_dims,
                max_regular,
                inventory_lower,
                inventory_upper,
                partial,
                output,
            );
            partial.pop();
        }
    }

    let mut states = Vec::new();
    recurse(
        0,
        reference.regular_lead_time,
        reference.regular_max_order_size,
        config.inventory_lower,
        config.inventory_upper,
        &mut Vec::new(),
        &mut states,
    );
    states
}

fn clamp_state(
    next_state: Vec<i64>,
    reference: &DualSourcingReferenceInstance,
    config: &BoundedDpConfig,
) -> Vec<i64> {
    next_state
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            if idx == 0 {
                value.clamp(config.inventory_lower, config.inventory_upper)
            } else {
                value.clamp(0, reference.regular_max_order_size as i64)
            }
        })
        .collect()
}

fn demand_values(reference: &DualSourcingReferenceInstance) -> Vec<usize> {
    (reference.demand_low..=reference.demand_high).collect()
}

fn lexicographically_smaller(lhs: [usize; 2], rhs: [usize; 2]) -> bool {
    lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1])
}

pub fn solve_bounded_average_cost_optimal_policy(
    reference: &DualSourcingReferenceInstance,
    config: &BoundedDpConfig,
) -> PyResult<AverageCostPolicyEvaluation> {
    validate_config(config)?;
    let states = enumerate_state_space(reference, config);
    let state_to_idx: HashMap<Vec<i64>, usize> = states
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, state)| (state, idx))
        .collect();
    let demand_values = demand_values(reference);
    let demand_probability = 1.0 / demand_values.len() as f64;
    let initial_state = deterministic_initial_state(reference);
    let initial_idx = *state_to_idx
        .get(&initial_state)
        .ok_or_else(|| PyValueError::new_err("initial state missing from bounded state space"))?;
    let reference_idx = 0usize;

    let mut values = vec![0.0; states.len()];
    let mut policy = vec![[0usize, 0usize]; states.len()];
    let mut completed_iterations = 0usize;
    let mut average_cost = 0.0f64;

    for iteration in 1..=config.max_iterations {
        let mut new_values = vec![0.0; states.len()];
        for (state_idx, state) in states.iter().enumerate() {
            let mut best_cost = f64::INFINITY;
            let mut best_action = [0usize, 0usize];
            for regular_order in 0..=reference.regular_max_order_size {
                for expedited_order in 0..=reference.expedited_max_order_size {
                    let mut expected_cost = 0.0;
                    for demand in demand_values.iter().copied() {
                        let period_cost = epoch_cost(
                            state,
                            regular_order,
                            expedited_order,
                            demand,
                            reference.regular_order_cost,
                            reference.expedited_order_cost,
                            reference.holding_cost,
                            reference.shortage_cost,
                        );
                        let next_state = clamp_state(
                            step_state(state, regular_order, expedited_order, demand),
                            reference,
                            config,
                        );
                        let next_idx = *state_to_idx.get(&next_state).ok_or_else(|| {
                            PyValueError::new_err("next state missing from bounded state space")
                        })?;
                        expected_cost += demand_probability * (period_cost + values[next_idx]);
                    }
                    let action = [regular_order, expedited_order];
                    if expected_cost < best_cost - 1e-12
                        || ((expected_cost - best_cost).abs() < 1e-12
                            && lexicographically_smaller(action, best_action))
                    {
                        best_cost = expected_cost;
                        best_action = action;
                    }
                }
            }
            new_values[state_idx] = best_cost;
            policy[state_idx] = best_action;
        }
        let baseline = new_values[reference_idx];
        average_cost = baseline;
        for value in new_values.iter_mut() {
            *value -= baseline;
        }
        let max_delta = new_values
            .iter()
            .zip(values.iter())
            .map(|(new_value, old_value)| (new_value - old_value).abs())
            .fold(0.0f64, f64::max);
        values = new_values;
        completed_iterations = iteration;
        if max_delta < config.tolerance {
            break;
        }
    }

    Ok(AverageCostPolicyEvaluation {
        average_cost,
        first_action: policy[initial_idx],
        iterations: completed_iterations,
    })
}

pub fn evaluate_bounded_average_cost_named_policy(
    reference: &DualSourcingReferenceInstance,
    config: &BoundedDpConfig,
    policy_name: &str,
    params: &[usize],
) -> PyResult<AverageCostPolicyEvaluation> {
    validate_config(config)?;
    let states = enumerate_state_space(reference, config);
    let state_to_idx: HashMap<Vec<i64>, usize> = states
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, state)| (state, idx))
        .collect();
    let demand_values = demand_values(reference);
    let demand_probability = 1.0 / demand_values.len() as f64;
    let initial_state = deterministic_initial_state(reference);
    let initial_idx = *state_to_idx
        .get(&initial_state)
        .ok_or_else(|| PyValueError::new_err("initial state missing from bounded state space"))?;

    let mut expected_costs = vec![0.0; states.len()];
    let mut transitions = vec![Vec::<(usize, f64)>::new(); states.len()];
    let mut completed_iterations = 0usize;
    let mut initial_first_action = [0usize, 0usize];

    for (state_idx, state) in states.iter().enumerate() {
        let (regular_order, expedited_order) = named_policy_action(
            policy_name,
            params,
            state,
            reference.regular_max_order_size,
            reference.expedited_max_order_size,
        )?;
        let action = [regular_order, expedited_order];
        if state_idx == initial_idx {
            initial_first_action = action;
        }

        let mut next_state_mass: HashMap<usize, f64> = HashMap::new();
        let mut expected_cost = 0.0;
        for demand in demand_values.iter().copied() {
            let period_cost = epoch_cost(
                state,
                regular_order,
                expedited_order,
                demand,
                reference.regular_order_cost,
                reference.expedited_order_cost,
                reference.holding_cost,
                reference.shortage_cost,
            );
            let next_state = clamp_state(
                step_state(state, regular_order, expedited_order, demand),
                reference,
                config,
            );
            let next_idx = *state_to_idx.get(&next_state).ok_or_else(|| {
                PyValueError::new_err("next state missing from bounded state space")
            })?;
            expected_cost += demand_probability * period_cost;
            *next_state_mass.entry(next_idx).or_insert(0.0) += demand_probability;
        }
        expected_costs[state_idx] = expected_cost;
        transitions[state_idx] = next_state_mass.into_iter().collect();
    }

    let mut distribution = vec![0.0; states.len()];
    distribution[initial_idx] = 1.0;
    let policy_eval_iterations = config.max_iterations.saturating_mul(50).max(5_000);

    for iteration in 1..=policy_eval_iterations {
        let mut next_distribution = vec![0.0; states.len()];
        for (state_idx, mass) in distribution.iter().copied().enumerate() {
            if mass <= 0.0 {
                continue;
            }
            for (next_idx, probability) in transitions[state_idx].iter().copied() {
                next_distribution[next_idx] += mass * probability;
            }
        }
        let max_delta = next_distribution
            .iter()
            .zip(distribution.iter())
            .map(|(next_mass, current_mass)| (next_mass - current_mass).abs())
            .fold(0.0f64, f64::max);
        distribution = next_distribution;
        completed_iterations = iteration;
        if max_delta < config.tolerance {
            break;
        }
    }

    let average_cost = distribution
        .iter()
        .zip(expected_costs.iter())
        .map(|(mass, expected_cost)| mass * expected_cost)
        .sum::<f64>();

    Ok(AverageCostPolicyEvaluation {
        average_cost,
        first_action: initial_first_action,
        iterations: completed_iterations,
    })
}

fn fixed_demand_path(
    reference: &DualSourcingReferenceInstance,
    seed: u64,
    horizon: usize,
) -> (Vec<i64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let demands = (0..horizon)
        .map(|_| rng.gen_range(reference.demand_low..=reference.demand_high))
        .collect();
    (deterministic_initial_state(reference), demands)
}

fn published_gap_for_policy(
    reference: Option<&PublishedOptimalityGapReference>,
    policy_name: &str,
) -> Option<f64> {
    reference.map(|reference| match policy_name {
        "single_index" => reference.single_index_gap_pct,
        "dual_index" => reference.dual_index_gap_pct,
        "capped_dual_index" => reference.capped_dual_index_gap_pct,
        "tailored_base_surge" => reference.tailored_base_surge_gap_pct,
        _ => unreachable!("unsupported heuristic name"),
    })
}

pub fn benchmark_reference_instance(
    reference_name: &str,
    config: &BoundedDpConfig,
    search_seed: u64,
    search_horizon: usize,
    warm_up_periods_ratio: f64,
) -> PyResult<BenchmarkReport> {
    let reference = get_reference_instance(reference_name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown dual-sourcing reference instance '{reference_name}'"
        ))
    })?;
    let figure_9 = get_figure_9_gap_reference(reference_name);
    let (initial_state, demands) = fixed_demand_path(reference, search_seed, search_horizon);
    let upper = target_upper_bound(
        reference.regular_lead_time,
        reference.demand_low,
        reference.demand_high,
        reference.expedited_max_order_size,
    );

    let optimal = solve_bounded_average_cost_optimal_policy(reference, config)?;

    let mut heuristics = Vec::new();

    let (single_best, _) = search_single_index_from_demands(
        &initial_state,
        &demands,
        reference.regular_max_order_size,
        reference.expedited_max_order_size,
        reference.regular_order_cost,
        reference.expedited_order_cost,
        reference.holding_cost,
        reference.shortage_cost,
        warm_up_periods_ratio,
        upper,
        1,
    )?;
    let single_eval = evaluate_bounded_average_cost_named_policy(
        reference,
        config,
        "single_index",
        &[single_best.0, single_best.1],
    )?;
    heuristics.push(HeuristicBenchmarkResult {
        policy_name: "single_index",
        params: vec![single_best.0, single_best.1],
        search_cost: single_best.2,
        average_cost: single_eval.average_cost,
        first_action: single_eval.first_action,
        optimality_gap_pct: 100.0 * (single_eval.average_cost / optimal.average_cost - 1.0),
        published_optimality_gap_pct: published_gap_for_policy(figure_9, "single_index"),
    });

    let (dual_best, _) = search_dual_index_from_demands(
        &initial_state,
        &demands,
        reference.regular_max_order_size,
        reference.expedited_max_order_size,
        reference.regular_order_cost,
        reference.expedited_order_cost,
        reference.holding_cost,
        reference.shortage_cost,
        warm_up_periods_ratio,
        upper,
        1,
    )?;
    let dual_eval = evaluate_bounded_average_cost_named_policy(
        reference,
        config,
        "dual_index",
        &[dual_best.0, dual_best.1],
    )?;
    heuristics.push(HeuristicBenchmarkResult {
        policy_name: "dual_index",
        params: vec![dual_best.0, dual_best.1],
        search_cost: dual_best.2,
        average_cost: dual_eval.average_cost,
        first_action: dual_eval.first_action,
        optimality_gap_pct: 100.0 * (dual_eval.average_cost / optimal.average_cost - 1.0),
        published_optimality_gap_pct: published_gap_for_policy(figure_9, "dual_index"),
    });

    let (capped_best, _) = search_capped_dual_index_from_demands(
        &initial_state,
        &demands,
        reference.regular_max_order_size,
        reference.expedited_max_order_size,
        reference.regular_order_cost,
        reference.expedited_order_cost,
        reference.holding_cost,
        reference.shortage_cost,
        warm_up_periods_ratio,
        upper,
        1,
    )?;
    let capped_eval = evaluate_bounded_average_cost_named_policy(
        reference,
        config,
        "capped_dual_index",
        &[capped_best.0, capped_best.1, capped_best.2],
    )?;
    heuristics.push(HeuristicBenchmarkResult {
        policy_name: "capped_dual_index",
        params: vec![capped_best.0, capped_best.1, capped_best.2],
        search_cost: capped_best.3,
        average_cost: capped_eval.average_cost,
        first_action: capped_eval.first_action,
        optimality_gap_pct: 100.0 * (capped_eval.average_cost / optimal.average_cost - 1.0),
        published_optimality_gap_pct: published_gap_for_policy(figure_9, "capped_dual_index"),
    });

    let (tbs_best, _) = search_tailored_base_surge_from_demands(
        &initial_state,
        &demands,
        reference.regular_max_order_size,
        reference.expedited_max_order_size,
        reference.regular_order_cost,
        reference.expedited_order_cost,
        reference.holding_cost,
        reference.shortage_cost,
        warm_up_periods_ratio,
        upper,
        1,
    )?;
    let tbs_eval = evaluate_bounded_average_cost_named_policy(
        reference,
        config,
        "tailored_base_surge",
        &[tbs_best.0, tbs_best.1],
    )?;
    heuristics.push(HeuristicBenchmarkResult {
        policy_name: "tailored_base_surge",
        params: vec![tbs_best.0, tbs_best.1],
        search_cost: tbs_best.2,
        average_cost: tbs_eval.average_cost,
        first_action: tbs_eval.first_action,
        optimality_gap_pct: 100.0 * (tbs_eval.average_cost / optimal.average_cost - 1.0),
        published_optimality_gap_pct: published_gap_for_policy(figure_9, "tailored_base_surge"),
    });

    heuristics.sort_by(|lhs, rhs| {
        lhs.optimality_gap_pct
            .partial_cmp(&rhs.optimality_gap_pct)
            .unwrap()
    });

    Ok(BenchmarkReport {
        reference_name: reference_name.to_string(),
        initial_state,
        optimal,
        heuristics,
    })
}
