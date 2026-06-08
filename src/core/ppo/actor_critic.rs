//! # `actor_critic` — candle shared-trunk actor-critic network
//!
//! ## Objective
//! The differentiable policy + value network PPO optimizes, ported from the
//! reference PyTorch `ActorCritic` (`ppo_owmr.py`). A SHARED MLP trunk feeds both
//! the policy heads and a scalar value head:
//! ```text
//!   z      = tanh(L2(tanh(L1(obs))))         # trunk, two hidden layers
//!   logits = [head_j(z) for j in 0..n_heads] # factored multi-discrete policy
//!   value  = value_head(z)                   # scalar baseline V(s)
//! ```
//! For OWMR `n_heads = K+1 = 11`, `head_j` is `Linear(hidden, max_order_j + 1)`,
//! and `hidden = 128`. The heads are LINEAR in the number of retailers — the
//! action distribution grows linearly in K, matching Kaynov's published design.
//!
//! ## Notes
//! - Built over a `candle_nn::VarMap`; the trainer owns the `VarMap` so it can
//!   collect `all_vars()` for the optimizer and clone parameters for
//!   best-checkpoint selection.
//! - `forward` returns the per-head logit tensors and the value tensor with the
//!   autodiff graph attached (used by the differentiable PPO/BC updates). The
//!   trainer pulls logits to CPU (via `to_vec2`) for no-grad action sampling
//!   during rollouts.
//! - Layer init uses candle's default (`linear` => Kaiming-style), which is close
//!   to PyTorch's default Linear init. The exact small-gain head init of the
//!   reference is not reproduced because the policy is behavior-cloned to the
//!   gate before PPO, which dominates the initial parameters.

use candle_core::{Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Shared-trunk actor-critic with factored multi-discrete policy heads.
pub struct ActorCritic {
    l1: Linear,
    l2: Linear,
    heads: Vec<Linear>,
    value_head: Linear,
    head_sizes: Vec<usize>,
    device: Device,
}

impl ActorCritic {
    /// Build the network. `head_sizes[j]` is the number of categories of head
    /// `j` (for OWMR DirectOrders, `max_order_j + 1`). `hidden` is the trunk
    /// width.
    pub fn new(
        vb: VarBuilder,
        obs_dim: usize,
        head_sizes: &[usize],
        hidden: usize,
        device: Device,
    ) -> Result<Self> {
        let l1 = linear(obs_dim, hidden, vb.pp("l1"))?;
        let l2 = linear(hidden, hidden, vb.pp("l2"))?;
        let mut heads = Vec::with_capacity(head_sizes.len());
        for (j, &size) in head_sizes.iter().enumerate() {
            heads.push(linear(hidden, size, vb.pp(format!("head_{j}")))?);
        }
        let value_head = linear(hidden, 1, vb.pp("value"))?;
        Ok(Self {
            l1,
            l2,
            heads,
            value_head,
            head_sizes: head_sizes.to_vec(),
            device,
        })
    }

    /// Number of policy heads (action dimensions).
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }

    /// Category counts per head.
    pub fn head_sizes(&self) -> &[usize] {
        &self.head_sizes
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Shared trunk: `tanh(L2(tanh(L1(obs))))`.
    fn trunk(&self, obs: &Tensor) -> Result<Tensor> {
        let h1 = self.l1.forward(obs)?.tanh()?;
        self.l2.forward(&h1)?.tanh()
    }

    /// Forward pass. Returns `(per_head_logits, value)` where
    /// `per_head_logits[j]` is `(B, head_sizes[j])` and `value` is `(B,)`. The
    /// autodiff graph is attached.
    pub fn forward(&self, obs: &Tensor) -> Result<(Vec<Tensor>, Tensor)> {
        let z = self.trunk(obs)?;
        let mut logits = Vec::with_capacity(self.heads.len());
        for head in &self.heads {
            logits.push(head.forward(&z)?);
        }
        let value = self.value_head.forward(&z)?.squeeze(1)?; // (B,1) -> (B,)
        Ok((logits, value))
    }

    /// Convenience: build a `(B, obs_dim)` f32 tensor from a batch of raw/normed
    /// observation rows.
    pub fn obs_to_tensor(&self, batch: &[Vec<f32>]) -> Result<Tensor> {
        let b = batch.len();
        let dim = if b > 0 { batch[0].len() } else { 0 };
        let mut flat = Vec::with_capacity(b * dim);
        for row in batch {
            flat.extend_from_slice(row);
        }
        Tensor::from_vec(flat, (b, dim), &self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    #[test]
    fn forward_shapes_are_correct() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let head_sizes = vec![4usize, 3, 7];
        let ac = ActorCritic::new(vb, 5, &head_sizes, 16, device).unwrap();
        let obs = vec![vec![0.1f32, 0.2, 0.3, 0.4, 0.5], vec![1.0, 0.0, -1.0, 0.5, 0.2]];
        let obs_t = ac.obs_to_tensor(&obs).unwrap();
        let (logits, value) = ac.forward(&obs_t).unwrap();
        assert_eq!(logits.len(), 3);
        assert_eq!(logits[0].dims(), &[2, 4]);
        assert_eq!(logits[1].dims(), &[2, 3]);
        assert_eq!(logits[2].dims(), &[2, 7]);
        assert_eq!(value.dims(), &[2]);
    }
}
