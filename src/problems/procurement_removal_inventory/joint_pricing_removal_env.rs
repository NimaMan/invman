// joint_pricing_removal_env.rs
//
// PURPOSE
// -------
// Faithful single-period transition and reward accounting for the Maggiar &
// Sadighian (2017) "Joint Inventory and Revenue Management with Removal
// Decisions" model. Unlike the legacy control-only slice in `env.rs` (ordering +
// removals, no pricing), this module adds the PRICING / MARKDOWN decision so the
// model matches the paper's actual reward structure (Eq. 4), under backlogging
// with the paper's backorder-cost convention h^- = c + k.
//
// STATE  (paper Section 3.1, "z_t = (x_t, y_t)")
// ---------------------------------------------
//   period t
//   inventory_level  x : on-hand inventory at the start of the period
//   returnable_level y : units that may still be RETURNED to the vendor (s)
//                        rather than only liquidated (l). Always 0 <= y <= x.
//
// DECISIONS (made at the start of the period)
// -------------------------------------------
//   target_demand    d : the expected demand the retailer steers to via price.
//                        With pricing constraint p <= p0 (markdowns only), the
//                        admissible region is d >= mu_t. The implied price is
//                        p(d) = p0 - (1/beta) ln(d / mu_t) and the expected
//                        revenue is r_t(d) = d * p(d) (see
//                        price_dependent_gamma_demand.rs).
//   net_flow         q : a single signed quantity.
//                          q > 0  => REMOVE q units (return first, then liquidate)
//                          q < 0  => PURCHASE |q| units (returnable up to cap)
//                        This matches the paper's q = q_r + q_nr with the
//                        Corollary-1 simplification (never liquidate a unit that
//                        could be returned; never buy non-returnable when a
//                        returnable purchase is available).
//
// EVENT TIMING (paper Section 3.1, "sequence of events")
// ------------------------------------------------------
//   1. Observe (x, y); choose d (price), and q (removal/purchase).
//   2. Random demand D = d + eps realizes.
//   3. Stockouts are satisfied in the period they occur at unit cost h^- = c + k
//      (Section 3.1: "h^-_t = c_t + k_t"); leftover inventory is carried over at
//      unit holding cost h^+.
//
// REWARD  (paper Eq. 4, profit form)
// ----------------------------------
//   Let q be the signed net flow. Define the purchase/return/liquidation term:
//       b(q, y) = s * q^+ + (l - s) * (q - y)^+ - c * q^-
//     - s * q^+               : refund value on removed units (returns earn s)
//     - (l - s) * (q - y)^+   : the part of removals beyond the returnable level
//                               earns only l instead of s (s - l penalty)
//     - c * q^-               : purchase cost on purchased units (q^- = max(-q,0))
//   Net inventory position after decisions and demand:
//       w = x - D - q
//       w^+ = max(w, 0)   (carried inventory)   -> holding cost h^+ * w^+
//       w^- = max(-w, 0)  (backlogged demand)   -> stockout cost h^- * w^-
//   Period profit:
//       profit = r(d) + b(q, y) - h^+ * w^+ - h^- * w^-
//   The repo's objective layer minimizes cost, so we also expose
//       period_cost = -profit
//   and reward = profit, so a maximizing controller maximizes discounted profit.
//
// STATE TRANSITION
// ----------------
//   Next inventory:   x' = w^+ = (x - D - q)^+   (backlog satisfied same period,
//                     not carried as negative inventory; Eq. 4 uses x^+).
//   Next returnable:  removals consume the returnable level first
//                     (Corollary 1: return before liquidate). Purchases of
//                     returnable units add to y (capped by `returnable_purchase
//                     _cap`, the fixed-returnability quota = a quantile of the
//                     base-price forecast in the paper). Sales/backlog/holding do
//                     not by themselves change returnability beyond the on-hand
//                     bound y' <= x'.
//       y_after_flow = if q >= 0 { y - min(q, y) }            // removals
//                      else      { y + min(|q|, cap) }        // purchases
//       y' = min(y_after_flow, x')                            // bounded by on-hand
//
// TERMINAL VALUE  (paper Assumption 4 example, Section 7.1.2 "fully returned and
// liquidated")
//   V_T(x, y) = s * min(x, y) + l * max(x - y, 0)
//
// All quantities are kept on an integer grid so the finite-horizon DP in
// joint_pricing_removal_dp.rs can solve the value function exactly on a small
// instance for verification.

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::problems::procurement_removal_inventory::price_dependent_gamma_demand::expected_revenue;

/// Faithful state z_t = (x_t, y_t) plus the period index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JointPricingRemovalState {
    pub period: usize,
    pub inventory_level: i64,
    pub returnable_level: i64,
}

/// Economic parameters of the joint pricing / inventory / removal model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointPricingRemovalParameters {
    pub base_price: f64,             // p0
    pub purchase_cost: f64,          // c
    pub refund_value: f64,           // s  (return value)
    pub liquidation_value: f64,      // l
    pub holding_cost: f64,           // h+
    pub backorder_supplement: f64,   // k  (so h- = c + k)
    pub elasticity_at_base_price: f64, // E  (negative)
    pub beta: f64,                   // price sensitivity, beta = -E / p0
    pub coefficient_of_variation: f64, // demand CV (1 in the paper)
}

/// Outcome of a single faithful period.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointPricingRemovalOutcome {
    pub next_state: JointPricingRemovalState,
    pub implied_price: f64,
    pub expected_revenue: f64,
    pub realized_demand: f64,
    pub returned_units: i64,
    pub liquidated_units: i64,
    pub purchased_units: i64,
    pub carried_inventory: f64,
    pub backlogged_units: f64,
    pub flow_value: f64, // b(q, y)
    pub holding_cost: f64,
    pub backorder_cost: f64,
    pub period_profit: f64,
    pub period_cost: f64,
    pub reward: f64,
}

pub fn validate_parameters(parameters: &JointPricingRemovalParameters) -> PyResult<()> {
    let values = [
        parameters.base_price,
        parameters.purchase_cost,
        parameters.refund_value,
        parameters.liquidation_value,
        parameters.holding_cost,
        parameters.backorder_supplement,
        parameters.beta,
        parameters.coefficient_of_variation,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("all parameters must be finite"));
    }
    // Paper Assumption 2: c > s > l  (purchase cost exceeds refund exceeds
    // liquidation). This is what makes removal a genuine (lossy) option.
    if !(parameters.purchase_cost > parameters.refund_value) {
        return Err(PyValueError::new_err(
            "Assumption 2(ii): purchase_cost c must exceed refund_value s",
        ));
    }
    if !(parameters.refund_value > parameters.liquidation_value) {
        return Err(PyValueError::new_err(
            "Assumption 2(iii): refund_value s must exceed liquidation_value l",
        ));
    }
    if parameters.holding_cost < 0.0 || parameters.backorder_supplement < 0.0 {
        return Err(PyValueError::new_err(
            "holding_cost h+ and backorder_supplement k must be non-negative",
        ));
    }
    if parameters.beta <= 0.0 || parameters.coefficient_of_variation <= 0.0 {
        return Err(PyValueError::new_err(
            "beta and coefficient_of_variation must be positive",
        ));
    }
    Ok(())
}

pub fn validate_state(state: &JointPricingRemovalState) -> PyResult<()> {
    if state.returnable_level < 0 {
        return Err(PyValueError::new_err("returnable_level must be >= 0"));
    }
    if state.returnable_level > state.inventory_level.max(0) {
        return Err(PyValueError::new_err(format!(
            "returnable_level {} cannot exceed inventory_level {}",
            state.returnable_level, state.inventory_level
        )));
    }
    Ok(())
}

/// Per-unit backorder cost h^- = c + k (paper Section 3.1).
pub fn backorder_cost_per_unit(parameters: &JointPricingRemovalParameters) -> f64 {
    parameters.purchase_cost + parameters.backorder_supplement
}

/// Single faithful period transition + reward.
///
/// `net_flow` q > 0 removes, q < 0 purchases. `realized_demand` is one demand
/// realization D = d + eps (the DP averages this over the demand quantiles).
#[allow(clippy::too_many_arguments)]
pub fn step_period(
    state: &JointPricingRemovalState,
    target_demand: f64,
    net_flow: i64,
    realized_demand: f64,
    returnable_purchase_cap: i64,
    forecast_mean_demand: f64,
    parameters: &JointPricingRemovalParameters,
) -> PyResult<JointPricingRemovalOutcome> {
    validate_state(state)?;
    validate_parameters(parameters)?;
    if returnable_purchase_cap < 0 {
        return Err(PyValueError::new_err(
            "returnable_purchase_cap must be >= 0",
        ));
    }
    if realized_demand < 0.0 || !realized_demand.is_finite() {
        return Err(PyValueError::new_err(
            "realized_demand must be finite and non-negative",
        ));
    }

    let implied_price = price_for_decision(state, target_demand, forecast_mean_demand, parameters)?;
    let revenue = expected_revenue(
        target_demand,
        forecast_mean_demand,
        parameters.base_price,
        parameters.beta,
    )?;

    // Decompose the signed net flow into removals (q^+) and purchases (q^-).
    let removed = net_flow.max(0);
    let purchased = (-net_flow).max(0);

    // Removals: return up to the returnable level, liquidate the rest.
    let returned_units = removed.min(state.returnable_level);
    let liquidated_units = removed - returned_units;
    // Purchases: returnable up to the fixed-returnability cap.
    let returnable_purchased = purchased.min(returnable_purchase_cap);

    // b(q, y) = s*q^+ + (l - s)(q - y)^+ - c*q^-
    let flow_value = parameters.refund_value * returned_units as f64
        + parameters.liquidation_value * liquidated_units as f64
        - parameters.purchase_cost * purchased as f64;

    // Net inventory after decisions and demand: w = x - D - q.
    // q > 0 removes inventory, q < 0 (purchase) adds inventory; subtracting q
    // handles both since x - D - q = x + purchased - removed - D.
    let net_position = state.inventory_level as f64 - realized_demand - net_flow as f64;
    let carried_inventory = net_position.max(0.0);
    let backlogged_units = (-net_position).max(0.0);

    let holding_cost = parameters.holding_cost * carried_inventory;
    let backorder_cost = backorder_cost_per_unit(parameters) * backlogged_units;

    let period_profit = revenue + flow_value - holding_cost - backorder_cost;

    // Next inventory: x' = w^+ (backlog satisfied in-period, not carried).
    let next_inventory_level = carried_inventory.round() as i64;
    // Next returnable level: removals consume returnability first; purchases of
    // returnable units add to it; bounded by on-hand inventory.
    let returnable_after_flow = if net_flow >= 0 {
        state.returnable_level - returned_units
    } else {
        state.returnable_level + returnable_purchased
    };
    let next_returnable_level = returnable_after_flow.max(0).min(next_inventory_level);

    let next_state = JointPricingRemovalState {
        period: state.period + 1,
        inventory_level: next_inventory_level,
        returnable_level: next_returnable_level,
    };

    Ok(JointPricingRemovalOutcome {
        next_state,
        implied_price,
        expected_revenue: revenue,
        realized_demand,
        returned_units,
        liquidated_units,
        purchased_units: purchased,
        carried_inventory,
        backlogged_units,
        flow_value,
        holding_cost,
        backorder_cost,
        period_profit,
        period_cost: -period_profit,
        reward: period_profit,
    })
}

/// Implied price p(d) for a target-demand decision, validating the markdown
/// constraint p <= p0 (i.e. d >= mu_t).
pub fn price_for_decision(
    _state: &JointPricingRemovalState,
    target_demand: f64,
    forecast_mean_demand: f64,
    parameters: &JointPricingRemovalParameters,
) -> PyResult<f64> {
    use crate::problems::procurement_removal_inventory::price_dependent_gamma_demand::price_at_demand;
    let price = price_at_demand(
        target_demand,
        forecast_mean_demand,
        parameters.base_price,
        parameters.beta,
    )?;
    Ok(price)
}

/// Terminal value V_T(x, y) = s * min(x, y) + l * max(x - y, 0).
pub fn terminal_value(
    state: &JointPricingRemovalState,
    parameters: &JointPricingRemovalParameters,
) -> PyResult<f64> {
    validate_state(state)?;
    let inventory = state.inventory_level.max(0);
    let returnable = state.returnable_level.max(0).min(inventory);
    Ok(parameters.refund_value * returnable as f64
        + parameters.liquidation_value * (inventory - returnable) as f64)
}
