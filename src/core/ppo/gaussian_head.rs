//! # `gaussian_head` — diagonal-Gaussian continuous action math
//!
//! ## Objective
//! Provide the continuous action distribution for scalar / vector continuous
//! problems (lost_sales scalar order, serial echelon levels, ...), so the SAME
//! PPO trainer that handles OWMR's factored multi-discrete action also handles
//! continuous actions. The policy is a diagonal Gaussian: per action dimension a
//! mean (state-dependent, from a Linear head) and a learnable log-standard-
//! deviation (state-INDEPENDENT, the standard PPO parameterization). The joint
//! log-prob / entropy are sums over dimensions.
//!
//! ## Integer-action problems
//! Inventory orders are integers, but the policy is over reals: the trainer
//! samples a real action, stores it for the (continuous) log-prob, and passes
//! `round(clip(·))` to the environment. The log-prob/ratio therefore use the raw
//! Gaussian sample (a proper density); the environment clamps to its feasible
//! integer order range. This is the standard "continuous policy over a
//! discretized environment" recipe.
//!
//! ## Two code paths (mirroring the multi-discrete head)
//! 1. **Rollout (no gradient, CPU):** `sample_gaussian` draws `a = mean + std*z`
//!    (`z ~ N(0,1)`, or `a = mean` when greedy) and returns the per-sample joint
//!    log-prob.
//! 2. **Update (differentiable, candle):** `gaussian_logp_entropy` recomputes the
//!    joint log-prob and entropy of STORED continuous actions under the current
//!    `mean`/`log_std`, so PPO's ratio and entropy bonus backprop into the trunk,
//!    the mean head, and `log_std`.

use candle_core::{Result, Tensor, D};
use rand::Rng;
use rand_distr::StandardNormal;

const LN_2PI: f64 = 1.837_877_066_409_345_6; // ln(2*pi)

/// Sample one continuous action vector from per-dim `(mean, log_std)` and return
/// `(action, joint_log_prob)`. CPU, no autodiff. `greedy` returns the mean.
pub fn sample_gaussian<R: Rng + ?Sized>(
    mean: &[f32],
    log_std: &[f32],
    greedy: bool,
    rng: &mut R,
) -> (Vec<f32>, f32) {
    let mut action = Vec::with_capacity(mean.len());
    let mut logp = 0f32;
    for (&m, &ls) in mean.iter().zip(log_std.iter()) {
        let std = ls.exp();
        let a = if greedy {
            m
        } else {
            let z: f32 = rng.sample(StandardNormal);
            m + std * z
        };
        // log N(a; m, std) = -0.5*((a-m)/std)^2 - ls - 0.5*ln(2pi)
        let z = (a - m) / std;
        logp += -0.5 * z * z - ls - 0.5 * LN_2PI as f32;
        action.push(a);
    }
    (action, logp)
}

/// Differentiable joint log-prob and entropy of stored continuous actions under
/// the current Gaussian. `mean` is `(B, dim)`, `log_std` is `(dim,)` (broadcast
/// over the batch), `actions` is `(B, dim)`. Returns `(logp (B,), entropy (B,))`,
/// summed over dimensions. Backprops into `mean` and `log_std`.
pub fn gaussian_logp_entropy(
    mean: &Tensor,
    log_std: &Tensor,
    actions: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (b, dim) = mean.dims2()?;
    let std = log_std.exp()?; // (dim,)
                              // z = (a - mean) / std   -> (B, dim)
    let centered = (actions - mean)?;
    let z = centered.broadcast_div(&std)?;
    // per-element log-prob: -0.5 z^2 - log_std - 0.5 ln(2pi)
    let neg_half_z2 = z.sqr()?.affine(-0.5, -0.5 * LN_2PI)?; // -0.5 z^2 - 0.5 ln2pi
    let logp_elems = neg_half_z2.broadcast_sub(log_std)?; // ... - log_std
    let logp = logp_elems.sum(D::Minus1)?; // (B,)

    // Differentiable entropy (depends on log_std only): per dim = log_std + 0.5*(ln2pi+1).
    // Joint entropy = sum_d log_std_d + dim*0.5*(ln2pi+1); broadcast to (B,).
    let entropy_scalar = log_std
        .sum_all()?
        .affine(1.0, dim as f64 * 0.5 * (LN_2PI + 1.0))?; // 0-dim
    let entropy = entropy_scalar.broadcast_as((b,))?; // (B,)
    Ok((logp, entropy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use rand::SeedableRng;

    #[test]
    fn greedy_sample_is_mean_with_finite_logprob() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (a, lp) = sample_gaussian(&[2.0, -1.0], &[0.0, 0.5], true, &mut rng);
        assert_eq!(a, vec![2.0, -1.0]);
        assert!(lp.is_finite() && lp < 0.0);
    }

    #[test]
    fn differentiable_logp_matches_closed_form_at_mean() {
        // At a == mean, logp_d = -log_std_d - 0.5 ln(2pi). Sum over dims.
        let dev = Device::Cpu;
        let mean = Tensor::from_vec(vec![2.0f32, -1.0], (1, 2), &dev).unwrap();
        let log_std = Tensor::from_vec(vec![0.0f32, 0.5], 2, &dev).unwrap();
        let actions = mean.clone();
        let (logp, entropy) = gaussian_logp_entropy(&mean, &log_std, &actions).unwrap();
        let logp_v = logp.to_vec1::<f32>().unwrap()[0];
        let expected = (-0.0 - 0.5 * LN_2PI as f32) + (-0.5 - 0.5 * LN_2PI as f32);
        assert!((logp_v - expected).abs() < 1e-4, "logp {logp_v} vs {expected}");
        // entropy per dim = log_std + 0.5(ln2pi+1); sum over 2 dims.
        let ent_v = entropy.to_vec1::<f32>().unwrap()[0];
        let exp_ent = (0.0 + 0.5 * (LN_2PI as f32 + 1.0)) + (0.5 + 0.5 * (LN_2PI as f32 + 1.0));
        assert!((ent_v - exp_ent).abs() < 1e-4, "entropy {ent_v} vs {exp_ent}");
    }
}
