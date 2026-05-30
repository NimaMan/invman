#![allow(dead_code)]

//! Clean periodic-review serial multi-echelon environment (textbook Clark-Scarf model).
//!
//! OBJECTIVE
//! ---------
//! A faithful, training-ready environment for the classical serial multi-echelon
//! inventory system (Clark and Scarf 1960; Snyder and Shen, "Fundamentals of Supply
//! Chain Theory", Ch. 6). Unlike the richer Pirhooshyaran network model in
//! `network_inventory` (which adds per-node production steps and pipeline holding and
//! therefore does NOT reproduce the textbook serial optimum), this env implements
//! exactly the model that `exact.rs` solves, so simulating the optimal echelon
//! base-stock policy reproduces the published optimal cost. This is the env we train
//! policies on for this problem family.
//!
//! MODEL
//! -----
//! - N stages in series, indexed downstream -> upstream as k = 0..N-1. Stage 0 faces
//!   i.i.d. customer demand; stage N-1 replenishes from an outside source with ample
//!   stock. Stage k+1 ships to stage k.
//! - Deterministic integer lead time `lead_time[k]` on the link feeding stage k.
//! - Installation (local) holding cost `holding_cost[k]` charged on physical on-hand at
//!   stage k; backorder penalty `penalty` charged on the customer (stage-0) backorder.
//! - Action: a per-stage order vector (echelon-base-stock or learned). Internal
//!   shipments are constrained by the immediate upstream stage's on-hand; the most
//!   upstream stage draws from ample external supply.
//!
//! PERIOD SEQUENCE (matches `exact.rs` / the literature lead-time-demand convention;
//! verified empirically against the single-stage newsvendor and the stockpyl optima):
//!   1. receive: each stage receives the shipment that has finished its lead time;
//!   2. demand: customer demand realized at stage 0; unmet demand is backordered;
//!   3. cost: installation holding on physical on-hand + penalty on stage-0 backorder
//!      (assessed BEFORE this period's replenishment; in-transit pipeline is NOT
//!      charged, matching the optimized Clark-Scarf cost);
//!   4. replenish: orders are placed AFTER demand and shipped upstream -> downstream,
//!      each internal shipment capped by the upstream stage's on-hand.
//! Ordering AFTER demand (not before) is what yields the L-period (not L+1) lead-time
//! demand window used by the literature; reversing it is the classic off-by-one error.

#[derive(Clone, Debug, PartialEq)]
pub struct SerialState {
    /// Physical on-hand inventory per stage, downstream -> upstream (index 0 = customer-facing).
    pub on_hand: Vec<f64>,
    /// In-transit pipelines per stage; `pipeline[k][0]` arrives next period at stage k.
    pub pipeline: Vec<std::collections::VecDeque<f64>>,
    /// Customer (stage-0) backorder.
    pub backorder: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SerialConfig {
    /// Installation (local) holding cost per stage, downstream -> upstream.
    pub holding_cost: Vec<f64>,
    /// Lead time on the link feeding each stage, downstream -> upstream.
    pub lead_time: Vec<usize>,
    /// Customer backorder penalty (stage 0).
    pub penalty: f64,
}

impl SerialConfig {
    pub fn num_stages(&self) -> usize {
        self.holding_cost.len()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SerialStepOutcome {
    pub holding_cost: f64,
    pub backorder_cost: f64,
    pub period_cost: f64,
    pub reward: f64,
}

/// Initialize a state at the echelon base-stock levels with pipelines warm-filled at
/// the demand mean (a neutral steady-ish start for long-run-average estimation).
pub fn initialize_at_echelon_levels(
    config: &SerialConfig,
    echelon_levels: &[f64],
    demand_mean: f64,
) -> SerialState {
    let n = config.num_stages();
    let on_hand = (0..n)
        .map(|k| {
            if k == 0 {
                echelon_levels[0]
            } else {
                echelon_levels[k] - echelon_levels[k - 1]
            }
        })
        .collect();
    let pipeline = config
        .lead_time
        .iter()
        .map(|l| std::collections::VecDeque::from(vec![demand_mean; *l]))
        .collect();
    SerialState {
        on_hand,
        pipeline,
        backorder: 0.0,
    }
}

/// Raw state vector for a learned policy: on-hand per stage, then per-stage in-transit
/// totals, then the customer backorder. Direct state quantities in a stable order (no
/// hidden normalization), per the repo state-interface rule.
pub fn raw_state_vector(state: &SerialState) -> Vec<f32> {
    let n = state.on_hand.len();
    let mut v = Vec::with_capacity(2 * n + 1);
    for k in 0..n {
        v.push(state.on_hand[k] as f32);
    }
    for k in 0..n {
        v.push(state.pipeline[k].iter().sum::<f64>() as f32);
    }
    v.push(state.backorder as f32);
    v
}

/// Echelon inventory positions (downstream -> upstream), computed after receipts and
/// demand: IP_e[k] = sum_{i<=k}(on_hand[i] + in-transit[i]) - backorder.
pub fn echelon_inventory_positions(state: &SerialState) -> Vec<f64> {
    let n = state.on_hand.len();
    let mut prefix_oh = 0.0;
    let mut prefix_pipe = 0.0;
    let mut ip = vec![0.0; n];
    for k in 0..n {
        prefix_oh += state.on_hand[k];
        prefix_pipe += state.pipeline[k].iter().sum::<f64>();
        ip[k] = prefix_oh + prefix_pipe - state.backorder;
    }
    ip
}

/// Phase 1-3 of a period: receive arriving shipments, meet customer demand, and assess
/// the period cost. Mutates `state` (on-hand, backorder) and returns the cost. The
/// replenishment decision is made AFTER this, on the resulting post-demand state (which
/// is what `echelon_inventory_positions` should be read from). This ordering is the
/// crux of matching the literature lead-time-demand convention.
pub fn consume(config: &SerialConfig, state: &mut SerialState, demand: f64) -> SerialStepOutcome {
    let n = config.num_stages();

    // 1. receipts.
    for k in 0..n {
        let arrival = state.pipeline[k].pop_front().unwrap_or(0.0);
        state.on_hand[k] += arrival;
    }

    // 2. customer demand at stage 0.
    let need = demand + state.backorder;
    let shipped = state.on_hand[0].min(need);
    state.on_hand[0] -= shipped;
    state.backorder = need - shipped;

    // 3. cost (post-demand, pre-replenish): installation holding + penalty.
    let mut holding = 0.0;
    for k in 0..n {
        holding += config.holding_cost[k] * state.on_hand[k].max(0.0);
    }
    let backorder_cost = config.penalty * state.backorder;
    let period_cost = holding + backorder_cost;

    SerialStepOutcome {
        holding_cost: holding,
        backorder_cost,
        period_cost,
        reward: -period_cost,
    }
}

/// Phase 4 of a period: place the per-stage replenishment orders (decided AFTER demand)
/// and ship them upstream -> downstream, each internal shipment capped by the upstream
/// stage's on-hand. `orders[k]` is the desired order for the link feeding stage k.
pub fn replenish(config: &SerialConfig, state: &mut SerialState, orders: &[f64]) {
    let n = config.num_stages();
    for k in (0..n).rev() {
        let shipment = if k == n - 1 {
            orders[k].max(0.0) // ample external supply
        } else {
            orders[k].max(0.0).min(state.on_hand[k + 1])
        };
        if k < n - 1 {
            state.on_hand[k + 1] -= shipment;
        }
        state.pipeline[k].push_back(shipment);
    }
}

/// Convenience full-period step for callers that compute orders from the POST-demand
/// state: callers should `consume` first, read the state for their policy, then call
/// `replenish`. This helper assumes `orders` were already computed from the post-demand
/// state of the PREVIOUS `consume`; prefer the explicit two-phase form for training.
pub fn step(
    config: &SerialConfig,
    state: &mut SerialState,
    demand: f64,
    orders: &[f64],
) -> SerialStepOutcome {
    let outcome = consume(config, state, demand);
    replenish(config, state, orders);
    outcome
}
