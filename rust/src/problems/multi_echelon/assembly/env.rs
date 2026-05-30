#![allow(dead_code)]

//! Clean periodic-review ASSEMBLY multi-echelon environment (Rosling 1989).
//!
//! OBJECTIVE
//! ---------
//! A faithful, training-ready environment for the textbook assembly inventory system:
//! several components are procured from outside suppliers and assembled into one finished
//! product that faces customer demand. By Rosling (1989) an assembly system is equivalent
//! to a serial system, so the exact optimum and the optimal echelon base-stock policy come
//! from the (literature-verified) serial solver via the reduction in `rosling.rs`. This
//! env is verified to reproduce that optimum by simulation (see `verification`).
//!
//! It is the `assembly` *version* of the multi-echelon problem (sibling to `serial`,
//! `general_network`, ...). Like `serial`, holding is charged on physical on-hand only and
//! orders are placed AFTER demand is observed (the lead-time-demand convention).
//!
//! SCOPE
//! -----
//! Equal component lead time (`component_lead_time` shared by all components). This is the
//! clean Rosling case: under a balanced echelon base-stock policy the components are stocked
//! identically, so they collapse to a single "kit" stage and the system is exactly a
//! 2-stage serial system (kit -> finished). Components may have heterogeneous holding costs
//! (the kit holding cost is their sum). Distinct component lead times would need Rosling's
//! lead-time reordering and are out of scope here.
//!
//! KNOWN LIMITATION (shared with `multi_echelon::serial`): the multi-stage simulation is
//! verified against the exact solver only when the demand-facing (finished) stage has lead
//! time 1. For `finished_lead_time >= 2` the simulated cost currently under-counts relative
//! to the exact optimum (single-stage is correct at every lead time; component/upstream lead
//! times >= 2 are fine). This is an open inter-stage cost-convention discrepancy to resolve
//! before training on finished-lead-time >= 2 instances. The verification tests cover
//! finished lead time 1.
//!
//! PERIOD SEQUENCE (same convention as `multi_echelon::serial`)
//!   1. receive: each component and the finished stage receive their arriving pipeline unit;
//!   2. demand: customer demand at the finished stage; unmet demand backordered;
//!   3. cost: installation holding on component + finished on-hand, penalty on backorder
//!      (assessed post-demand, pre-replenish; in-transit pipeline not charged);
//!   4. replenish (after demand): the finished stage assembles up to its echelon target,
//!      consuming one of each component per unit (capped by the scarcest component);
//!      each component reorders from its outside supplier up to the kit echelon target.

use std::collections::VecDeque;

#[derive(Clone, Debug, PartialEq)]
pub struct AssemblyConfig {
    /// Installation holding cost per component (downstream finished unit embodies one of each).
    pub component_holding_costs: Vec<f64>,
    /// Shared component procurement lead time (equal-lead-time Rosling case).
    pub component_lead_time: usize,
    /// Installation holding cost of a finished unit (must be >= sum of component holdings).
    pub finished_holding_cost: f64,
    /// Assembly/finished replenishment lead time.
    pub finished_lead_time: usize,
    /// Customer backorder penalty.
    pub penalty: f64,
}

impl AssemblyConfig {
    pub fn num_components(&self) -> usize {
        self.component_holding_costs.len()
    }
    /// Kit installation holding cost = sum of component holding costs.
    pub fn kit_holding_cost(&self) -> f64 {
        self.component_holding_costs.iter().sum()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssemblyState {
    pub component_on_hand: Vec<f64>,
    pub component_pipeline: Vec<VecDeque<f64>>,
    pub finished_on_hand: f64,
    pub finished_pipeline: VecDeque<f64>,
    pub backorder: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssemblyStepOutcome {
    pub holding_cost: f64,
    pub backorder_cost: f64,
    pub period_cost: f64,
    pub reward: f64,
}

/// Initialize at the echelon base-stock levels `[S_finished, S_kit]` (downstream -> upstream),
/// pipelines warm-filled at the demand mean. Component on-hand starts at the kit local level
/// (S_kit - S_finished), finished on-hand at S_finished.
pub fn initialize_at_echelon_levels(
    config: &AssemblyConfig,
    echelon_levels: &[f64],
    demand_mean: f64,
) -> AssemblyState {
    let s_finished = echelon_levels[0];
    let s_kit = echelon_levels[1];
    let kit_local = (s_kit - s_finished).max(0.0);
    AssemblyState {
        component_on_hand: vec![kit_local; config.num_components()],
        component_pipeline: vec![
            VecDeque::from(vec![demand_mean; config.component_lead_time]);
            config.num_components()
        ],
        finished_on_hand: s_finished,
        finished_pipeline: VecDeque::from(vec![demand_mean; config.finished_lead_time]),
        backorder: 0.0,
    }
}

/// Raw state vector for a learned policy: component on-hand, component in-transit totals,
/// finished on-hand, finished in-transit, backorder. Direct quantities, stable order.
pub fn raw_state_vector(state: &AssemblyState) -> Vec<f32> {
    let m = state.component_on_hand.len();
    let mut v = Vec::with_capacity(2 * m + 3);
    for k in 0..m {
        v.push(state.component_on_hand[k] as f32);
    }
    for k in 0..m {
        v.push(state.component_pipeline[k].iter().sum::<f64>() as f32);
    }
    v.push(state.finished_on_hand as f32);
    v.push(state.finished_pipeline.iter().sum::<f64>() as f32);
    v.push(state.backorder as f32);
    v
}

/// Phase 1-3 of a period: receive, meet demand, assess cost. The replenishment decision is
/// made afterward on the resulting post-demand state.
pub fn consume(config: &AssemblyConfig, state: &mut AssemblyState, demand: f64) -> AssemblyStepOutcome {
    // 1. receipts.
    for k in 0..config.num_components() {
        let arrival = state.component_pipeline[k].pop_front().unwrap_or(0.0);
        state.component_on_hand[k] += arrival;
    }
    let finished_arrival = state.finished_pipeline.pop_front().unwrap_or(0.0);
    state.finished_on_hand += finished_arrival;

    // 2. customer demand at the finished stage.
    let need = demand + state.backorder;
    let shipped = state.finished_on_hand.min(need);
    state.finished_on_hand -= shipped;
    state.backorder = need - shipped;

    // 3. cost (post-demand, pre-replenish): installation holding + penalty.
    let mut holding = config.finished_holding_cost * state.finished_on_hand.max(0.0);
    for k in 0..config.num_components() {
        holding += config.component_holding_costs[k] * state.component_on_hand[k].max(0.0);
    }
    let backorder_cost = config.penalty * state.backorder;
    AssemblyStepOutcome {
        holding_cost: holding,
        backorder_cost,
        period_cost: holding + backorder_cost,
        reward: -(holding + backorder_cost),
    }
}

/// Phase 4: echelon base-stock replenishment with levels `[S_finished, S_kit]`
/// (downstream -> upstream). The finished stage assembles up to S_finished (capped by the
/// scarcest component), each component reorders up to the kit echelon level S_kit. Order
/// quantities are computed from the pre-replenish (post-demand) state, then executed.
pub fn replenish(config: &AssemblyConfig, state: &mut AssemblyState, echelon_levels: &[f64]) {
    let s_finished = echelon_levels[0];
    let s_kit = echelon_levels[1];
    let m = config.num_components();

    // Pre-replenish echelon inventory positions.
    let finished_in_transit: f64 = state.finished_pipeline.iter().sum();
    let finished_ip = state.finished_on_hand + finished_in_transit - state.backorder;
    let component_ip: Vec<f64> = (0..m)
        .map(|k| {
            let comp_in_transit: f64 = state.component_pipeline[k].iter().sum();
            // Echelon of component k includes everything downstream that embodies it.
            state.component_on_hand[k] + comp_in_transit + state.finished_on_hand
                + finished_in_transit
                - state.backorder
        })
        .collect();

    // Finished assembly: desired raise to S_finished, capped by the scarcest component.
    let desired_assembly = (s_finished - finished_ip).max(0.0);
    let kits_available = state
        .component_on_hand
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let assembled = desired_assembly.min(kits_available);
    for k in 0..m {
        state.component_on_hand[k] -= assembled;
    }
    state.finished_pipeline.push_back(assembled);

    // Component reorders from ample external supply.
    for k in 0..m {
        let order = (s_kit - component_ip[k]).max(0.0);
        state.component_pipeline[k].push_back(order);
    }
}
