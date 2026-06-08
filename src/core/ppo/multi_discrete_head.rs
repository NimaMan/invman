//! # `multi_discrete_head` — factored multi-discrete categorical action math
//!
//! ## Objective
//! Provide the categorical action distribution used by the OWMR-style policy: a
//! FACTORED multi-discrete action where dimension `j` is an independent
//! Categorical over `{0..size_j-1}` and the joint log-prob / entropy are the SUM
//! over dimensions. This mirrors the reference PyTorch `ActorCritic.act` /
//! `.evaluate` (per-head `torch.distributions.Categorical`, summed log-prob and
//! entropy).
//!
//! ## Two code paths
//! 1. **Rollout (no gradient, CPU):** `sample_head` turns one head's logits into
//!    a sampled (or greedy) category and its log-prob, computed in plain Rust
//!    from logits pulled off the candle tensor. Used while collecting
//!    trajectories — no autodiff graph needed, so it is cheap and uses the
//!    crate's `rand` for sampling.
//! 2. **Update (differentiable, candle):** `joint_logp_entropy` recomputes, with
//!    gradients, the joint log-prob and entropy of STORED actions under the
//!    current network, so PPO's importance ratio `exp(logp - logp_old)` and the
//!    entropy bonus backprop into the trunk + heads. Built from
//!    `log_softmax` + `gather` + `softmax`.

use candle_core::{Result, Tensor, D};
use candle_nn::ops::log_softmax;

/// Numerically stable softmax + log-softmax of one logit row (CPU).
fn softmax_logsoftmax(logits: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let logsumexp = max + sum.ln();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    let logprobs: Vec<f32> = logits.iter().map(|&l| l - logsumexp).collect();
    (probs, logprobs)
}

/// Sample (or take greedy argmax of) one head's category from its logits, and
/// return `(action, log_prob)`. CPU, no autodiff. `u` is a uniform(0,1) draw
/// (ignored when `greedy`).
pub fn sample_head(logits: &[f32], greedy: bool, u: f32) -> (usize, f32) {
    let (probs, logprobs) = softmax_logsoftmax(logits);
    let action = if greedy {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if l > best_v {
                best_v = l;
                best = i;
            }
        }
        best
    } else {
        // Inverse-CDF sampling.
        let mut cum = 0.0f32;
        let mut chosen = probs.len() - 1;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if u <= cum {
                chosen = i;
                break;
            }
        }
        chosen
    };
    (action, logprobs[action])
}

/// Differentiable joint log-prob and entropy of stored multi-discrete actions
/// under the current network. `per_head_logits[j]` has shape `(B, size_j)`;
/// `actions` is `(B, n_heads)` with `u32` category indices. Returns
/// `(logp (B,), entropy (B,))`, summed over heads. Backprops into the logits.
pub fn joint_logp_entropy(
    per_head_logits: &[Tensor],
    actions: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let mut logp: Option<Tensor> = None;
    let mut entropy: Option<Tensor> = None;
    for (j, logits) in per_head_logits.iter().enumerate() {
        // log-softmax and softmax over the category dimension.
        let logp_all = log_softmax(logits, D::Minus1)?; // (B, size_j)
        let p_all = logp_all.exp()?; // softmax (B, size_j)
        // Gather the chosen category's log-prob: actions[:, j] -> (B, 1).
        // `narrow` returns a strided view; candle's `gather` requires a
        // contiguous index, so materialize it (only the single-head case is
        // already contiguous).
        let idx = actions.narrow(1, j, 1)?.contiguous()?; // (B, 1) u32
        let head_logp = logp_all.gather(&idx, 1)?.squeeze(1)?; // (B,)
        // Entropy H = -sum_k p_k * logp_k over the category dim.
        let head_entropy = (p_all * &logp_all)?.sum(D::Minus1)?.neg()?; // (B,)
        logp = Some(match logp {
            None => head_logp,
            Some(acc) => (acc + head_logp)?,
        });
        entropy = Some(match entropy {
            None => head_entropy,
            Some(acc) => (acc + head_entropy)?,
        });
    }
    Ok((
        logp.expect("at least one head"),
        entropy.expect("at least one head"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax_and_logprob_is_consistent() {
        let logits = vec![0.1f32, 2.0, -1.0, 0.5];
        let (a, lp) = sample_head(&logits, true, 0.0);
        assert_eq!(a, 1);
        // log-prob of argmax must be the largest log-prob and < 0.
        let (_p, logp_all) = softmax_logsoftmax(&logits);
        assert!((lp - logp_all[1]).abs() < 1e-6);
        assert!(lp < 0.0);
    }

    #[test]
    fn sampling_respects_cdf_extremes() {
        let logits = vec![0.0f32, 0.0]; // uniform 50/50
        // u just above 0 -> first category; u = 1.0 -> last category.
        assert_eq!(sample_head(&logits, false, 0.0001).0, 0);
        assert_eq!(sample_head(&logits, false, 1.0).0, 1);
    }
}
