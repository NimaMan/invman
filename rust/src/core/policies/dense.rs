use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Copy)]
pub enum ActivationKind {
    Relu,
    Selu,
    Gelu,
}

pub fn parse_activation(activation: &str) -> PyResult<ActivationKind> {
    match activation {
        "relu" => Ok(ActivationKind::Relu),
        "selu" => Ok(ActivationKind::Selu),
        "gelu" => Ok(ActivationKind::Gelu),
        other => Err(PyValueError::new_err(format!(
            "unsupported activation '{other}', expected one of: relu, selu, gelu"
        ))),
    }
}

#[derive(Clone, Copy)]
pub enum DensePolicyHead {
    CategoricalQuantity,
    DirectQuantity,
    CappedDirectQuantity,
    SigmoidDirectQuantity,
    SoftGatedDirectQuantity,
    GatedSigmoidDirectQuantity,
    HardGatedDirectQuantity,
    SoftGatedOrdinalQuantity,
    HardGatedOrdinalQuantity,
}

pub fn parse_policy_head(policy_head: &str) -> PyResult<DensePolicyHead> {
    match policy_head {
        "categorical_quantity" => Ok(DensePolicyHead::CategoricalQuantity),
        "direct_quantity"
        | "positive_quantity"
        | "softplus_quantity"
        | "nonnegative_quantity" => Ok(DensePolicyHead::DirectQuantity),
        "capped_direct_quantity" | "capped_softplus_quantity" | "capped_positive_quantity" => {
            Ok(DensePolicyHead::CappedDirectQuantity)
        }
        "sigmoid_direct_quantity" | "scaled_direct_quantity" => {
            Ok(DensePolicyHead::SigmoidDirectQuantity)
        }
        "soft_gated_direct_quantity" | "gated_direct_quantity" => {
            Ok(DensePolicyHead::SoftGatedDirectQuantity)
        }
        "gated_positive_quantity" => Ok(DensePolicyHead::SoftGatedDirectQuantity),
        "gated_sigmoid_direct_quantity" | "scaled_gated_direct_quantity" => {
            Ok(DensePolicyHead::GatedSigmoidDirectQuantity)
        }
        "hard_gated_direct_quantity" | "two_stage_direct_quantity" => {
            Ok(DensePolicyHead::HardGatedDirectQuantity)
        }
        "two_stage_positive_quantity" => Ok(DensePolicyHead::HardGatedDirectQuantity),
        "soft_gated_ordinal_quantity" | "gated_ordinal_quantity" => {
            Ok(DensePolicyHead::SoftGatedOrdinalQuantity)
        }
        "hard_gated_ordinal_quantity" | "two_stage_ordinal_quantity" => {
            Ok(DensePolicyHead::HardGatedOrdinalQuantity)
        }
        other => Err(PyValueError::new_err(format!(
            "unsupported dense policy head '{other}', expected one of: categorical_quantity, direct_quantity, capped_direct_quantity, sigmoid_direct_quantity, soft_gated_direct_quantity, gated_sigmoid_direct_quantity, hard_gated_direct_quantity, soft_gated_ordinal_quantity, hard_gated_ordinal_quantity"
        ))),
    }
}

fn apply_activation(x: f32, activation: ActivationKind) -> f32 {
    match activation {
        ActivationKind::Relu => x.max(0.0),
        ActivationKind::Selu => {
            let alpha = 1.673_263_2_f32;
            let scale = 1.050_701_f32;
            if x > 0.0 {
                scale * x
            } else {
                scale * (alpha * x.exp() - alpha)
            }
        }
        ActivationKind::Gelu => {
            // Match PyTorch's common tanh-based approximation closely enough for rollout parity.
            let c = (2.0_f32 / std::f32::consts::PI).sqrt();
            0.5 * x * (1.0 + (c * (x + 0.044_715 * x.powi(3))).tanh())
        }
    }
}

fn argmax_first(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = values[0];
    for (idx, value) in values.iter().enumerate().skip(1) {
        if *value > best_val {
            best_val = *value;
            best_idx = idx;
        }
    }
    best_idx
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn require_policy_max_quantity(
    policy_max_quantity: Option<usize>,
    policy_name: &str,
) -> PyResult<usize> {
    policy_max_quantity.ok_or_else(|| {
        PyValueError::new_err(format!(
            "{policy_name} requires a policy-side quantity cap, but none was provided"
        ))
    })
}

fn dense_forward(input: &[f32], weights: &[f32], bias: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    let mut output = vec![0.0_f32; out_dim];
    for out_idx in 0..out_dim {
        let row_offset = out_idx * in_dim;
        let mut acc = bias[out_idx];
        for in_idx in 0..in_dim {
            acc += weights[row_offset + in_idx] * input[in_idx];
        }
        output[out_idx] = acc;
    }
    output
}

fn dense_action_from_logits(
    logits: &[f32],
    policy_head: DensePolicyHead,
    policy_max_quantity: Option<usize>,
) -> PyResult<usize> {
    match policy_head {
        DensePolicyHead::CategoricalQuantity => Ok(argmax_first(logits)),
        DensePolicyHead::DirectQuantity => {
            if logits.len() != 1 {
                return Err(PyValueError::new_err(format!(
                    "direct quantity logits length {} does not match expected 1",
                    logits.len()
                )));
            }
            let quantity_value = (1.0 + logits[0].exp()).ln();
            let action = quantity_value.round().max(0.0) as usize;
            Ok(action)
        }
        DensePolicyHead::CappedDirectQuantity => {
            if logits.len() != 1 {
                return Err(PyValueError::new_err(format!(
                    "capped direct quantity logits length {} does not match expected 1",
                    logits.len()
                )));
            }
            let cap = require_policy_max_quantity(policy_max_quantity, "capped direct quantity")?;
            let quantity_value = (1.0 + logits[0].exp()).ln();
            let action = quantity_value.round().clamp(0.0, cap as f32) as usize;
            Ok(action)
        }
        DensePolicyHead::SigmoidDirectQuantity => {
            if logits.len() != 1 {
                return Err(PyValueError::new_err(format!(
                    "scaled direct quantity logits length {} does not match expected 1",
                    logits.len()
                )));
            }
            let cap = require_policy_max_quantity(policy_max_quantity, "sigmoid direct quantity")?;
            let scaled_quantity = sigmoid(logits[0]) * cap as f32;
            let action = scaled_quantity.round().clamp(0.0, cap as f32) as usize;
            Ok(action)
        }
        DensePolicyHead::SoftGatedDirectQuantity => {
            if logits.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "gated direct quantity logits length {} does not match expected 2",
                    logits.len()
                )));
            }
            let gate_prob = sigmoid(logits[0]);
            let quantity_value = (1.0 + logits[1].exp()).ln();
            let action = (gate_prob * quantity_value).round().max(0.0) as usize;
            Ok(action)
        }
        DensePolicyHead::GatedSigmoidDirectQuantity => {
            if logits.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "scaled gated direct quantity logits length {} does not match expected 2",
                    logits.len()
                )));
            }
            let cap = require_policy_max_quantity(
                policy_max_quantity,
                "gated sigmoid direct quantity",
            )?;
            let gate_prob = sigmoid(logits[0]);
            let quantity_value = sigmoid(logits[1]) * cap as f32;
            let action = (gate_prob * quantity_value).round().clamp(0.0, cap as f32) as usize;
            Ok(action)
        }
        DensePolicyHead::HardGatedDirectQuantity => {
            if logits.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "two-stage direct quantity logits length {} does not match expected 2",
                    logits.len()
                )));
            }
            let gate_prob = sigmoid(logits[0]);
            if gate_prob < 0.5 {
                return Ok(0);
            }
            let cap =
                require_policy_max_quantity(policy_max_quantity, "hard-gated direct quantity")?;
            let quantity_value = (1.0 + logits[1].exp()).ln();
            let action = quantity_value.round().clamp(1.0, cap as f32) as usize;
            Ok(action)
        }
        DensePolicyHead::SoftGatedOrdinalQuantity => {
            if logits.len() < 2 {
                return Err(PyValueError::new_err(format!(
                    "gated ordinal logits length {} does not match minimum expected 2",
                    logits.len(),
                )));
            }
            let gate_prob = sigmoid(logits[0]);
            let quantity_score: f32 = logits[1..].iter().map(|value| sigmoid(*value)).sum();
            let action = (gate_prob * quantity_score).round().max(0.0) as usize;
            Ok(action)
        }
        DensePolicyHead::HardGatedOrdinalQuantity => {
            if logits.len() < 2 {
                return Err(PyValueError::new_err(format!(
                    "two-stage ordinal logits length {} does not match minimum expected 2",
                    logits.len(),
                )));
            }
            let gate_prob = sigmoid(logits[0]);
            if gate_prob < 0.5 {
                return Ok(0);
            }
            let quantity_score: f32 = logits[1..].iter().map(|value| sigmoid(*value)).sum();
            let action = quantity_score.round().max(1.0) as usize;
            Ok(action)
        }
    }
}

pub fn linear_action_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    output_dim: usize,
    policy_head: DensePolicyHead,
    policy_max_quantity: Option<usize>,
) -> PyResult<usize> {
    if state.len() != input_dim {
        return Err(PyValueError::new_err(format!(
            "state length {} does not match input_dim {}",
            state.len(),
            input_dim
        )));
    }
    let expected = output_dim * input_dim + output_dim;
    if flat_params.len() != expected {
        return Err(PyValueError::new_err(format!(
            "linear flat params length {} does not match expected {}",
            flat_params.len(),
            expected
        )));
    }
    let split = output_dim * input_dim;
    let logits = dense_forward(
        state,
        &flat_params[..split],
        &flat_params[split..],
        output_dim,
    );
    dense_action_from_logits(&logits, policy_head, policy_max_quantity)
}

#[cfg(test)]
mod tests {
    use super::{dense_action_from_logits, DensePolicyHead};

    #[test]
    fn soft_gated_direct_quantity_does_not_require_policy_cap() {
        let action = dense_action_from_logits(
            &[10.0, 30.0],
            DensePolicyHead::SoftGatedDirectQuantity,
            None,
        )
        .expect("soft-gated direct quantity should be uncapped on Rust");
        assert!(action > 20);
    }
}

pub fn mlp_action_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    hidden_dims: &[usize],
    output_dim: usize,
    activation: ActivationKind,
    policy_head: DensePolicyHead,
    policy_max_quantity: Option<usize>,
) -> PyResult<usize> {
    if state.len() != input_dim {
        return Err(PyValueError::new_err(format!(
            "state length {} does not match input_dim {}",
            state.len(),
            input_dim
        )));
    }
    if hidden_dims.is_empty() {
        return Err(PyValueError::new_err(
            "hidden_dims must be non-empty for mlp policy",
        ));
    }

    let mut expected = 0usize;
    let mut prev_dim = input_dim;
    for hidden_dim in hidden_dims.iter().copied() {
        expected += hidden_dim * prev_dim + hidden_dim;
        prev_dim = hidden_dim;
    }
    expected += output_dim * prev_dim + output_dim;
    if flat_params.len() != expected {
        return Err(PyValueError::new_err(format!(
            "mlp flat params length {} does not match expected {}",
            flat_params.len(),
            expected
        )));
    }

    let mut cursor = 0usize;
    let mut current = state.to_vec();
    let mut prev_width = input_dim;
    for hidden_dim in hidden_dims.iter().copied() {
        let weight_len = hidden_dim * prev_width;
        let bias_len = hidden_dim;
        let weights = &flat_params[cursor..cursor + weight_len];
        cursor += weight_len;
        let bias = &flat_params[cursor..cursor + bias_len];
        cursor += bias_len;
        let mut next = dense_forward(&current, weights, bias, hidden_dim);
        for value in next.iter_mut() {
            *value = apply_activation(*value, activation);
        }
        current = next;
        prev_width = hidden_dim;
    }

    let weight_len = output_dim * prev_width;
    let logits = dense_forward(
        &current,
        &flat_params[cursor..cursor + weight_len],
        &flat_params[cursor + weight_len..cursor + weight_len + output_dim],
        output_dim,
    );
    dense_action_from_logits(&logits, policy_head, policy_max_quantity)
}
