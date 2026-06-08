//! # `gae` — Generalized Advantage Estimation (finite horizon)
//!
//! ## Objective
//! Turn a collected batch of per-step rewards and value estimates into advantage
//! and return targets for the PPO update. This is a direct, byte-faithful port
//! of `compute_gae` in the reference `ppo_owmr.py`, so the Rust trainer's
//! advantage signal matches the validated PyTorch baseline.
//!
//! ## Algorithm (GAE-lambda, finite horizon)
//! Inputs are laid out as `[T][B]` (time-major): `rewards[t][b]`, `values[t][b]`.
//! The horizon is fixed and finite, so the value bootstrapped past the last step
//! is 0 (`nonterminal = 0` at `t = T-1`). Iterating backwards:
//! ```text
//!   delta_t   = r_t + gamma * V_{t+1} * nonterminal - V_t
//!   A_t       = delta_t + gamma * lambda * nonterminal * A_{t+1}
//!   return_t  = A_t + V_t
//! ```
//! `gamma = 1.0` (undiscounted, matching the 100-period OWMR protocol) and
//! `lambda = 0.95` are the reference defaults. Returns `(advantages, returns)`
//! both `[T][B]`. The PPO update normalizes advantages across the whole flattened
//! batch (done in the trainer, not here).

/// Compute GAE advantages and value-targets (returns) for a time-major batch.
///
/// `rewards` and `values` are `[T][B]`. Returns `(advantages, returns)`, each
/// `[T][B]`. Terminal value past `T-1` is 0 (finite horizon).
pub fn compute_gae(
    rewards: &[Vec<f32>],
    values: &[Vec<f32>],
    gamma: f32,
    lam: f32,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let t_len = rewards.len();
    if t_len == 0 {
        return (Vec::new(), Vec::new());
    }
    let b_len = rewards[0].len();
    let mut advantages = vec![vec![0.0f32; b_len]; t_len];
    let mut returns = vec![vec![0.0f32; b_len]; t_len];

    let mut last_gae = vec![0.0f32; b_len];
    let mut next_val = vec![0.0f32; b_len]; // terminal value = 0 at the horizon

    for t in (0..t_len).rev() {
        let nonterminal = if t < t_len - 1 { 1.0 } else { 0.0 };
        for b in 0..b_len {
            let delta = rewards[t][b] + gamma * next_val[b] * nonterminal - values[t][b];
            last_gae[b] = delta + gamma * lam * nonterminal * last_gae[b];
            advantages[t][b] = last_gae[b];
            returns[t][b] = last_gae[b] + values[t][b];
        }
        // Carry V_t backward as the "next" value for step t-1.
        for b in 0..b_len {
            next_val[b] = values[t][b];
        }
    }
    (advantages, returns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gae_undiscounted_zero_value_equals_return_to_go() {
        // gamma=1, lam=1, V=0 => advantage_t = sum of future rewards (return-to-go).
        let rewards = vec![vec![1.0f32], vec![2.0], vec![3.0]];
        let values = vec![vec![0.0f32], vec![0.0], vec![0.0]];
        let (adv, ret) = compute_gae(&rewards, &values, 1.0, 1.0);
        // return-to-go: t0=6, t1=5, t2=3
        assert!((adv[0][0] - 6.0).abs() < 1e-5, "adv0={}", adv[0][0]);
        assert!((adv[1][0] - 5.0).abs() < 1e-5, "adv1={}", adv[1][0]);
        assert!((adv[2][0] - 3.0).abs() < 1e-5, "adv2={}", adv[2][0]);
        // returns = adv + V = adv (V=0)
        assert!((ret[0][0] - 6.0).abs() < 1e-5);
        assert!((ret[2][0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn gae_perfect_value_gives_zero_advantage_at_last_step() {
        // At the final step, nonterminal=0 so delta = r - V; if V == r, advantage = 0.
        let rewards = vec![vec![5.0f32]];
        let values = vec![vec![5.0f32]];
        let (adv, ret) = compute_gae(&rewards, &values, 1.0, 0.95);
        assert!(adv[0][0].abs() < 1e-6, "adv={}", adv[0][0]);
        assert!((ret[0][0] - 5.0).abs() < 1e-6, "ret={}", ret[0][0]);
    }
}
