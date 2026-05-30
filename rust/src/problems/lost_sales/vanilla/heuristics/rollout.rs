// Rollout / average-cost evaluation of a lost-sales heuristic policy.
//
// `evaluate_heuristic_policy(config, policy)` simulates the lost-sales
// environment for `config.horizon` periods under the chosen heuristic and
// returns the mean per-period cost after a warm-up burn-in. Each period:
//
//   1. The environment pipeline state is folded into the heuristic's pipeline
//      representation by adding on-hand inventory into the first slot
//      (`pipeline_state_with_inventory_folded_into_first_slot`).
//   2. The heuristic chooses an order quantity (Myopic-1 / Myopic-2 / SVBS).
//   3. The oldest outstanding order arrives into on-hand inventory and the new
//      order is appended to the pipeline.
//   4. Demand is sampled, the period cost is computed by `epoch_cost`, and on-
//      hand inventory is updated.
//
// The reported mean discards the first `floor(warm_up_periods_ratio * horizon)`
// periods so the estimate reflects steady-state cost — matching the warm-up
// convention used by the learned-policy rollouts in `rollout.rs`.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::problems::lost_sales::demand::{build_demand_process, sample_demand};
use crate::problems::lost_sales::vanilla::env::{epoch_cost, initialize_state};
use crate::problems::lost_sales::vanilla::heuristics::evaluator::{
    validate_heuristic_config, LostSalesHeuristicEvaluator, LostSalesHeuristicVerificationConfig,
};
use crate::problems::lost_sales::vanilla::heuristics::policy_kind::LostSalesHeuristicPolicyKind;

/// An observed mean-cost measurement for a named policy.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyPerformanceMeasurement {
    pub policy_name: &'static str,
    pub mean_cost: f64,
}

/// Build a measurement from a policy name and observed mean cost.
pub fn measurement_from_observed_mean_cost(
    policy_name: &'static str,
    mean_cost: f64,
) -> PolicyPerformanceMeasurement {
    PolicyPerformanceMeasurement {
        policy_name,
        mean_cost,
    }
}

/// Roll out a single heuristic policy and report its warm-up-adjusted mean cost.
pub fn evaluate_heuristic_policy(
    config: LostSalesHeuristicVerificationConfig,
    policy: LostSalesHeuristicPolicyKind,
) -> Result<PolicyPerformanceMeasurement, String> {
    validate_heuristic_config(&config)?;
    let demand_mean = config.demand_config.implied_mean()?;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut demand_process = build_demand_process(config.demand_config, &mut rng)?;
    let mut env_state =
        initialize_state(demand_mean, config.lead_time, &mut rng, &mut demand_process);
    let mut heuristic = LostSalesHeuristicEvaluator::new(config)?;
    let mut epoch_costs = Vec::with_capacity(config.horizon);

    for _period in 0..config.horizon {
        let state = pipeline_state_with_inventory_folded_into_first_slot(
            env_state.current_inventory,
            &env_state.lead_time_orders,
        );
        let action = match policy {
            LostSalesHeuristicPolicyKind::Myopic1 => heuristic.myopic_1_order_quantity(&state)?.0,
            LostSalesHeuristicPolicyKind::Myopic2 => heuristic.myopic_2_order_quantity(&state)?.0,
            LostSalesHeuristicPolicyKind::StandardVectorBaseStock => {
                heuristic.standard_vector_base_stock_order_quantity(&state)?
            }
        };

        let arriving_order = env_state.lead_time_orders.remove(0);
        env_state.lead_time_orders.push(action);
        env_state.current_inventory = env_state
            .current_inventory
            .saturating_add(arriving_order.min(i64::MAX as usize) as i64);

        let demand = sample_demand(&mut rng, &mut demand_process);
        let cost = epoch_cost(
            &mut env_state.current_inventory,
            demand,
            action,
            config.holding_cost,
            config.shortage_cost,
            config.procurement_cost,
            config.fixed_order_cost,
        );
        epoch_costs.push(cost);
    }

    Ok(PolicyPerformanceMeasurement {
        policy_name: policy.policy_name(),
        mean_cost: mean_after_warmup_like_rollout(&epoch_costs, config.warm_up_periods_ratio),
    })
}

fn mean_after_warmup_like_rollout(epoch_costs: &[f64], warm_up_periods_ratio: f64) -> f64 {
    let horizon = epoch_costs.len();
    let warm_up_periods = ((warm_up_periods_ratio * horizon as f64).floor() as usize).min(horizon);
    let active_costs = if warm_up_periods < epoch_costs.len() {
        &epoch_costs[warm_up_periods..]
    } else {
        epoch_costs
    };
    active_costs.iter().sum::<f64>() / active_costs.len() as f64
}

fn pipeline_state_with_inventory_folded_into_first_slot(
    current_inventory: i64,
    lead_time_orders: &[usize],
) -> Vec<usize> {
    let mut state = lead_time_orders.to_vec();
    if let Some(first_slot) = state.first_mut() {
        *first_slot = first_slot.saturating_add(current_inventory.max(0) as usize);
    }
    state
}
