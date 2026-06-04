// ameliorating_inventory
//
// Canonical, literature-verified executable home for the Pahr & Grunow (2025)
// ameliorating-inventory problem family.
//
// Faithful model (canonical):
//   - `average_profit_blending_env.rs` : long-run average-profit dynamics with a
//     price-augmented state, 3-part action (purchase / production / issuance),
//     stochastic Beta decay + evaporation, truncated-Normal purchase price,
//     correlated demand/sales price, per-age capacity, and blending issuance.
//   - `issuance_blending_lp.rs`        : the per-period blending issuance LP.
//   - `perfect_information_lp.rs`      : the perfect-information (steady-state,
//     expected-value) LP that produces the published average-profit UPPER BOUND
//     (`max_reward`). This is the literature-verification anchor.
//   - `lp_dataset_loader.rs`           : parser for the checked-in companion
//     datasets (config + expected-revenue tables + published bounds).
//   - `references.rs`                  : literature instances and the published
//     anchors (PRIMARY_REFERENCE_INSTANCE, VERIFICATION_PROBLEM_INSTANCE).
//
// Reduced model (retained tractable companion, NOT the verification target):
//   - `env.rs`, `issuance.rs`, `rollout.rs`, `heuristics/`, `finite_horizon_dp.rs`,
//     `bindings.rs` implement an earlier discrete, discounted-cost approximation
//     used by the soft-tree rollout path. It is kept for the existing Python
//     bindings but is no longer the canonical formulation.

pub mod average_profit_blending_env;
pub mod bindings;
pub mod demand;
pub mod env;
pub mod experiments;
pub mod heuristics;
pub mod issuance;
pub mod issuance_blending_lp;
pub mod literature;
pub mod lp_dataset_loader;
pub mod perfect_information_lp;
pub mod practical;
pub mod references;
pub mod rollout;

#[cfg(test)]
pub(crate) mod finite_horizon_dp;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod verification;
