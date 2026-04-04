use statrs::distribution::{ContinuousCDF, Gamma};

use crate::problems::perishable_inventory::env::{step_state, PerishableState};
use crate::problems::perishable_inventory::heuristics::base_stock_order_quantity;
use crate::problems::perishable_inventory::references::get_reference_instance;

const MAX_DEMAND: usize = 100;
const BURN_IN_PERIODS: usize = 100;
const EVAL_PERIODS: usize = 365;
const VALUE_ITERATION_TOLERANCE: f64 = 1e-12;
const VALUE_ITERATION_MAX_ITERS: usize = 4_000;

pub(crate) struct ExactMdp {
    pub(crate) states: Vec<PerishableState>,
    pub(crate) expected_rewards: Vec<Vec<f64>>,
    pub(crate) transitions: Vec<Vec<Vec<f64>>>,
}

fn state_space_size(instance_name: &str) -> usize {
    let instance = get_reference_instance(instance_name).expect("reference instance must exist");
    let components = instance.shelf_life + instance.lead_time - 1;
    (instance.max_order_size + 1).pow(components as u32)
}

fn decode_state(
    index: usize,
    shelf_life: usize,
    lead_time: usize,
    max_order_size: usize,
) -> PerishableState {
    let base = max_order_size + 1;
    let components = shelf_life + lead_time - 1;
    let mut digits = vec![0usize; components];
    let mut remaining = index;
    for digit in digits.iter_mut().rev() {
        *digit = remaining % base;
        remaining /= base;
    }
    let pipeline_len = lead_time.saturating_sub(1);
    PerishableState {
        pipeline_orders: digits[..pipeline_len].to_vec(),
        on_hand: digits[pipeline_len..].to_vec(),
    }
}

fn encode_state(state: &PerishableState, max_order_size: usize) -> usize {
    let base = max_order_size + 1;
    state
        .pipeline_orders
        .iter()
        .chain(state.on_hand.iter())
        .fold(0usize, |acc, value| acc * base + value)
}

fn build_demand_probabilities(demand_mean: f64, demand_cov: f64, max_demand: usize) -> Vec<f64> {
    let alpha = 1.0 / (demand_cov * demand_cov);
    let rate = 1.0 / (demand_mean * demand_cov * demand_cov);
    let gamma = Gamma::new(alpha, rate).expect("gamma parameters must be valid");

    (0..=max_demand)
        .map(|demand| {
            let lower = if demand == 0 {
                0.0
            } else {
                demand as f64 - 0.5
            };
            let upper = if demand == max_demand {
                f64::INFINITY
            } else {
                demand as f64 + 0.5
            };
            gamma.cdf(upper) - gamma.cdf(lower)
        })
        .collect()
}

pub(crate) fn build_exact_mdp(instance_name: &str) -> ExactMdp {
    let instance = get_reference_instance(instance_name).expect("reference instance must exist");
    let num_states = state_space_size(instance_name);
    let states = (0..num_states)
        .map(|index| {
            decode_state(
                index,
                instance.shelf_life,
                instance.lead_time,
                instance.max_order_size,
            )
        })
        .collect::<Vec<_>>();
    let demand_probabilities =
        build_demand_probabilities(instance.demand_mean, instance.demand_cov, MAX_DEMAND);

    let mut expected_rewards = vec![vec![0.0; instance.max_order_size + 1]; num_states];
    let mut transitions =
        vec![vec![vec![0.0; num_states]; instance.max_order_size + 1]; num_states];

    for (state_idx, state) in states.iter().enumerate() {
        for action in 0..=instance.max_order_size {
            for (demand, probability) in demand_probabilities.iter().copied().enumerate() {
                let outcome = step_state(
                    state,
                    action,
                    demand,
                    instance.holding_cost,
                    instance.shortage_cost,
                    instance.waste_cost,
                    instance.procurement_cost,
                    instance.issuing_policy,
                );
                let next_idx = encode_state(&outcome.next_state, instance.max_order_size);
                expected_rewards[state_idx][action] += probability * -outcome.cost;
                transitions[state_idx][action][next_idx] += probability;
            }
        }
    }

    ExactMdp {
        states,
        expected_rewards,
        transitions,
    }
}

pub(crate) fn value_iteration_best_action_values(
    mdp: &ExactMdp,
    gamma: f64,
) -> (Vec<usize>, Vec<f64>) {
    let mut values = vec![0.0; mdp.states.len()];

    for _ in 0..VALUE_ITERATION_MAX_ITERS {
        let mut new_values = vec![0.0; values.len()];
        let mut delta = 0.0f64;

        for state_idx in 0..mdp.states.len() {
            let mut best_value = f64::NEG_INFINITY;
            for action in 0..mdp.expected_rewards[state_idx].len() {
                let continuation_value = mdp.transitions[state_idx][action]
                    .iter()
                    .enumerate()
                    .map(|(next_idx, probability)| probability * values[next_idx])
                    .sum::<f64>();
                let q_value = mdp.expected_rewards[state_idx][action] + gamma * continuation_value;
                if q_value > best_value {
                    best_value = q_value;
                }
            }
            delta = delta.max((best_value - values[state_idx]).abs());
            new_values[state_idx] = best_value;
        }

        values = new_values;
        if delta < VALUE_ITERATION_TOLERANCE {
            break;
        }
    }

    let mut policy = vec![0usize; mdp.states.len()];
    for (state_idx, action_slot) in policy.iter_mut().enumerate() {
        let mut best_action = 0usize;
        let mut best_value = f64::NEG_INFINITY;
        for action in 0..mdp.expected_rewards[state_idx].len() {
            let continuation_value = mdp.transitions[state_idx][action]
                .iter()
                .enumerate()
                .map(|(next_idx, probability)| probability * values[next_idx])
                .sum::<f64>();
            let q_value = mdp.expected_rewards[state_idx][action] + gamma * continuation_value;
            if q_value > best_value {
                best_value = q_value;
                best_action = action;
            }
        }
        *action_slot = best_action;
    }

    (policy, values)
}

pub(crate) fn expected_discounted_return_from_zero_state(
    instance_name: &str,
    mdp: &ExactMdp,
    policy: &[usize],
) -> f64 {
    let instance = get_reference_instance(instance_name).expect("reference instance must exist");
    let zero_state = PerishableState {
        on_hand: vec![0usize; instance.shelf_life],
        pipeline_orders: vec![0usize; instance.lead_time.saturating_sub(1)],
    };
    let zero_state_index = encode_state(&zero_state, instance.max_order_size);
    let mut state_distribution = vec![0.0; mdp.states.len()];
    state_distribution[zero_state_index] = 1.0;

    let mut expected_return = 0.0f64;
    let gamma = 0.99f64;
    for period in 0..(BURN_IN_PERIODS + EVAL_PERIODS) {
        if period >= BURN_IN_PERIODS {
            let discounted_reward = state_distribution
                .iter()
                .enumerate()
                .map(|(state_idx, probability)| {
                    probability * mdp.expected_rewards[state_idx][policy[state_idx]]
                })
                .sum::<f64>();
            expected_return += discounted_reward * gamma.powi((period - BURN_IN_PERIODS) as i32);
        }

        let mut next_distribution = vec![0.0; mdp.states.len()];
        for (state_idx, probability) in state_distribution.iter().copied().enumerate() {
            if probability == 0.0 {
                continue;
            }
            for (next_idx, transition_probability) in mdp.transitions[state_idx][policy[state_idx]]
                .iter()
                .copied()
                .enumerate()
            {
                if transition_probability > 0.0 {
                    next_distribution[next_idx] += probability * transition_probability;
                }
            }
        }
        state_distribution = next_distribution;
    }
    expected_return
}

pub(crate) fn best_base_stock_level_by_expected_return(
    instance_name: &str,
    mdp: &ExactMdp,
) -> usize {
    let instance = get_reference_instance(instance_name).expect("reference instance must exist");
    let mut best_level = 0usize;
    let mut best_return = f64::NEG_INFINITY;

    for level in 0..=instance.max_order_size {
        let policy = mdp
            .states
            .iter()
            .map(|state| base_stock_order_quantity(state, level, instance.max_order_size))
            .collect::<Vec<_>>();
        let expected_return =
            expected_discounted_return_from_zero_state(instance_name, mdp, &policy);
        if expected_return > best_return {
            best_return = expected_return;
            best_level = level;
        }
    }

    best_level
}

pub(crate) fn build_policy_table_9x9(policy: &[usize], mdp: &ExactMdp) -> [[usize; 9]; 9] {
    let mut table = [[0usize; 9]; 9];
    for (state_idx, state) in mdp.states.iter().enumerate() {
        if state.pipeline_orders.is_empty()
            && state.on_hand.len() == 2
            && state.on_hand[0] <= 8
            && state.on_hand[1] <= 8
        {
            table[8 - state.on_hand[0]][state.on_hand[1]] = policy[state_idx];
        }
    }
    table
}
