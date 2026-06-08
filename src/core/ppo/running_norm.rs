//! # `running_norm` — Welford running observation normalizer
//!
//! ## Objective
//! Standardize raw observations to ~zero-mean/unit-variance using running
//! statistics, exactly as the reference PyTorch PPO baseline (`RunningNorm` in
//! `ppo_owmr.py`). PPO is sensitive to observation scale; a running normalizer
//! that is UPDATED during training rollouts and FROZEN during evaluation gives a
//! stable, problem-agnostic input scale. This is part of the trainer (not the
//! env) so every problem gets identical normalization behavior.
//!
//! ## Algorithm (Welford / Chan parallel variance)
//! Maintain `mean`, `var`, `count`. On a batch `X` (N x dim):
//! 1. batch mean `bmean`, batch var `bvar`, batch size `bn`.
//! 2. `delta = bmean - mean`; `tot = count + bn`.
//! 3. `mean += delta * bn / tot`.
//! 4. Combine second moments: `M2 = var*count + bvar*bn + delta^2 * count*bn/tot`;
//!    `var = M2 / tot`; `count = tot`.
//! This is the chunked parallel variance update; it is numerically equivalent to
//! streaming Welford and matches the reference implementation byte-for-byte
//! (initial `count = 1e-4`, `var = 1`). Normalization is
//! `(x - mean) / sqrt(var + 1e-8)`.

/// Running mean/variance normalizer over a fixed-dim observation vector.
#[derive(Clone, Debug)]
pub struct RunningNorm {
    mean: Vec<f64>,
    var: Vec<f64>,
    count: f64,
}

impl RunningNorm {
    /// New normalizer for `dim`-dimensional observations. Matches the reference:
    /// `mean = 0`, `var = 1`, `count = 1e-4`.
    pub fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.0; dim],
            var: vec![1.0; dim],
            count: 1e-4,
        }
    }

    /// Update the running statistics with a batch of raw observations
    /// (`rows` x `dim`). No-op on an empty batch.
    pub fn update(&mut self, batch: &[Vec<f32>]) {
        let n = batch.len();
        if n == 0 {
            return;
        }
        let dim = self.mean.len();
        let bn = n as f64;
        // Batch mean.
        let mut bmean = vec![0.0f64; dim];
        for row in batch {
            for (m, &v) in bmean.iter_mut().zip(row.iter()) {
                *m += v as f64;
            }
        }
        for m in bmean.iter_mut() {
            *m /= bn;
        }
        // Batch (population) variance, matching numpy's `x.var(axis=0)`.
        let mut bvar = vec![0.0f64; dim];
        for row in batch {
            for ((bv, &v), &bm) in bvar.iter_mut().zip(row.iter()).zip(bmean.iter()) {
                let d = v as f64 - bm;
                *bv += d * d;
            }
        }
        for bv in bvar.iter_mut() {
            *bv /= bn;
        }
        // Chan parallel merge into the running stats.
        let tot = self.count + bn;
        for i in 0..dim {
            let delta = bmean[i] - self.mean[i];
            self.mean[i] += delta * bn / tot;
            let m_a = self.var[i] * self.count;
            let m_b = bvar[i] * bn;
            let m2 = m_a + m_b + delta * delta * self.count * bn / tot;
            self.var[i] = m2 / tot;
        }
        self.count = tot;
    }

    /// Normalize one raw observation: `(x - mean) / sqrt(var + 1e-8)`.
    pub fn normalize(&self, raw: &[f32]) -> Vec<f32> {
        raw.iter()
            .zip(self.mean.iter())
            .zip(self.var.iter())
            .map(|((&x, &m), &v)| ((x as f64 - m) / (v + 1e-8).sqrt()) as f32)
            .collect()
    }

    /// Normalize a batch of raw observations.
    pub fn normalize_batch(&self, batch: &[Vec<f32>]) -> Vec<Vec<f32>> {
        batch.iter().map(|row| self.normalize(row)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_norm_matches_batch_statistics() {
        // After a single update with a batch, normalizing the batch mean should
        // give ~0, and the running mean/var should be close to the batch's.
        let mut norm = RunningNorm::new(2);
        let batch: Vec<Vec<f32>> = vec![
            vec![1.0, 10.0],
            vec![3.0, 20.0],
            vec![5.0, 30.0],
            vec![7.0, 40.0],
        ];
        norm.update(&batch);
        // Batch means are 4.0 and 25.0; count starts at 1e-4 so running mean ~ batch mean.
        assert!((norm.mean[0] - 4.0).abs() < 1e-2, "mean0={}", norm.mean[0]);
        assert!((norm.mean[1] - 25.0).abs() < 1e-2, "mean1={}", norm.mean[1]);
        let normed = norm.normalize(&[4.0, 25.0]);
        assert!(normed[0].abs() < 1e-2, "normed0={}", normed[0]);
        assert!(normed[1].abs() < 1e-2, "normed1={}", normed[1]);
        // A value one batch-std above the mean should normalize to ~+1.
        let std1 = (norm.var[1] + 1e-8).sqrt() as f32;
        let normed_hi = norm.normalize(&[4.0, 25.0 + std1]);
        assert!((normed_hi[1] - 1.0).abs() < 1e-2, "normed_hi1={}", normed_hi[1]);
    }
}
