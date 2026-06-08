pub mod policies;

/// Reusable in-crate PPO trainer (candle autodiff backend). Feature-gated so the
/// default build stays candle-free; enable with `--features ppo`.
#[cfg(feature = "ppo")]
pub mod ppo;
