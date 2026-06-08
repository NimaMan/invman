pub mod allocation;
pub mod bindings;
pub mod demand;
pub mod env;
pub mod finite_horizon_dp;
pub mod heuristics;
/// OWMR as a batched PPO environment (reusable Rust PPO trainer). Feature-gated.
#[cfg(feature = "ppo")]
pub mod ppo_environment;
/// Python entry point for the OWMR PPO trainer. Feature-gated.
#[cfg(feature = "ppo")]
pub mod ppo_bindings;
pub mod references;
pub mod rollout;

#[cfg(test)]
mod tests;
