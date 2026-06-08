//! # `actor_critic` — candle shared-trunk actor-critic network
//!
//! ## Objective
//! The differentiable policy + value network PPO optimizes. A SHARED MLP trunk
//! feeds both the policy head and a scalar value head:
//! ```text
//!   z      = tanh(L2(tanh(L1(obs))))         # trunk, two hidden layers
//!   value  = value_head(z)                   # scalar baseline V(s)
//!   policy = <head>(z)                       # multi-discrete OR continuous
//! ```
//! The head is chosen from the env's `ActionSpec`, so ONE network class serves
//! every action geometry:
//! - **Multi-discrete** (OWMR order quantities): `n` independent `Linear`
//!   heads producing per-dimension Categorical logits (ported from the reference
//!   PyTorch `ActorCritic`; heads linear in the number of retailers).
//! - **Continuous** (lost_sales scalar order, serial echelon levels): a `mean`
//!   `Linear` head plus a learnable state-independent `log_std` vector — a
//!   diagonal Gaussian.
//!
//! `forward` returns a `PolicyOutput` (logits or mean/log_std) plus the value,
//! with the autodiff graph attached. The trainer dispatches sampling and the
//! differentiable log-prob/entropy on this enum (see `multi_discrete_head` /
//! `gaussian_head`).

use candle_core::{Device, Result, Tensor};
use candle_nn::{linear, Init, Linear, Module, VarBuilder};

use super::environment::ActionSpec;

/// The policy half of a forward pass (graph attached).
pub enum PolicyOutput {
    /// Per-head Categorical logits; `logits[j]` is `(B, sizes[j])`.
    MultiDiscrete(Vec<Tensor>),
    /// Diagonal Gaussian: `mean` is `(B, dim)`, `log_std` is `(dim,)`.
    Continuous { mean: Tensor, log_std: Tensor },
}

enum Head {
    MultiDiscrete { heads: Vec<Linear> },
    Continuous { mean: Linear, log_std: Tensor },
}

/// Shared-trunk actor-critic with a discrete or continuous policy head.
pub struct ActorCritic {
    l1: Linear,
    l2: Linear,
    value_head: Linear,
    head: Head,
    action_dim: usize,
    device: Device,
}

impl ActorCritic {
    /// Build the network for the given `ActionSpec` and trunk width `hidden`.
    pub fn new(
        vb: VarBuilder,
        obs_dim: usize,
        spec: &ActionSpec,
        hidden: usize,
        device: Device,
    ) -> Result<Self> {
        let l1 = linear(obs_dim, hidden, vb.pp("l1"))?;
        let l2 = linear(hidden, hidden, vb.pp("l2"))?;
        let value_head = linear(hidden, 1, vb.pp("value"))?;
        let (head, action_dim) = match spec {
            ActionSpec::MultiDiscrete { sizes } => {
                let mut heads = Vec::with_capacity(sizes.len());
                for (j, &size) in sizes.iter().enumerate() {
                    heads.push(linear(hidden, size, vb.pp(format!("head_{j}")))?);
                }
                (Head::MultiDiscrete { heads }, sizes.len())
            }
            ActionSpec::Continuous { dim } => {
                let mean = linear(hidden, *dim, vb.pp("mean"))?;
                // State-independent learnable log-std, initialized to 0 (std = 1).
                let log_std = vb.get_with_hints(*dim, "log_std", Init::Const(0.0))?;
                (Head::Continuous { mean, log_std }, *dim)
            }
        };
        Ok(Self {
            l1,
            l2,
            value_head,
            head,
            action_dim,
            device,
        })
    }

    /// Number of action dimensions (heads for multi-discrete, action width for
    /// continuous).
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Shared trunk: `tanh(L2(tanh(L1(obs))))`.
    fn trunk(&self, obs: &Tensor) -> Result<Tensor> {
        let h1 = self.l1.forward(obs)?.tanh()?;
        self.l2.forward(&h1)?.tanh()
    }

    /// Forward pass. Returns `(policy_output, value)` where `value` is `(B,)`.
    pub fn forward(&self, obs: &Tensor) -> Result<(PolicyOutput, Tensor)> {
        let z = self.trunk(obs)?;
        let value = self.value_head.forward(&z)?.squeeze(1)?; // (B,1) -> (B,)
        let output = match &self.head {
            Head::MultiDiscrete { heads } => {
                let mut logits = Vec::with_capacity(heads.len());
                for head in heads {
                    logits.push(head.forward(&z)?);
                }
                PolicyOutput::MultiDiscrete(logits)
            }
            Head::Continuous { mean, log_std } => PolicyOutput::Continuous {
                mean: mean.forward(&z)?,
                log_std: log_std.clone(),
            },
        };
        Ok((output, value))
    }

    /// Convenience: build a `(B, obs_dim)` f32 tensor from a batch of observation
    /// rows.
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
    fn multi_discrete_forward_shapes() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let spec = ActionSpec::MultiDiscrete { sizes: vec![4, 3, 7] };
        let ac = ActorCritic::new(vb, 5, &spec, 16, device).unwrap();
        let obs = vec![vec![0.1f32, 0.2, 0.3, 0.4, 0.5], vec![1.0, 0.0, -1.0, 0.5, 0.2]];
        let obs_t = ac.obs_to_tensor(&obs).unwrap();
        let (out, value) = ac.forward(&obs_t).unwrap();
        assert_eq!(value.dims(), &[2]);
        match out {
            PolicyOutput::MultiDiscrete(logits) => {
                assert_eq!(logits.len(), 3);
                assert_eq!(logits[0].dims(), &[2, 4]);
                assert_eq!(logits[2].dims(), &[2, 7]);
            }
            _ => panic!("expected multi-discrete"),
        }
    }

    #[test]
    fn continuous_forward_shapes() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let spec = ActionSpec::Continuous { dim: 2 };
        let ac = ActorCritic::new(vb, 5, &spec, 16, device).unwrap();
        let obs = vec![vec![0.1f32, 0.2, 0.3, 0.4, 0.5]];
        let obs_t = ac.obs_to_tensor(&obs).unwrap();
        let (out, value) = ac.forward(&obs_t).unwrap();
        assert_eq!(value.dims(), &[1]);
        match out {
            PolicyOutput::Continuous { mean, log_std } => {
                assert_eq!(mean.dims(), &[1, 2]);
                assert_eq!(log_std.dims(), &[2]);
            }
            _ => panic!("expected continuous"),
        }
    }
}
