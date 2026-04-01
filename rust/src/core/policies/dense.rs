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
    GatedOrdinalQuantity,
    TwoStageOrdinalQuantity,
}

pub fn parse_policy_head(policy_head: &str) -> PyResult<DensePolicyHead> {
    match policy_head {
        "categorical_quantity" => Ok(DensePolicyHead::CategoricalQuantity),
        "gated_ordinal_quantity" => Ok(DensePolicyHead::GatedOrdinalQuantity),
        "two_stage_ordinal_quantity" => Ok(DensePolicyHead::TwoStageOrdinalQuantity),
        other => Err(PyValueError::new_err(format!(
            "unsupported dense policy head '{other}', expected one of: categorical_quantity, gated_ordinal_quantity, two_stage_ordinal_quantity"
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
    max_order_size: usize,
) -> PyResult<usize> {
    match policy_head {
        DensePolicyHead::CategoricalQuantity => Ok(argmax_first(logits)),
        DensePolicyHead::GatedOrdinalQuantity => {
            if logits.len() != max_order_size + 1 {
                return Err(PyValueError::new_err(format!(
                    "gated ordinal logits length {} does not match expected {}",
                    logits.len(),
                    max_order_size + 1
                )));
            }
            let gate_prob = sigmoid(logits[0]);
            let quantity_score: f32 = logits[1..].iter().map(|value| sigmoid(*value)).sum();
            let action = (gate_prob * quantity_score)
                .round()
                .clamp(0.0, max_order_size as f32) as usize;
            Ok(action)
        }
        DensePolicyHead::TwoStageOrdinalQuantity => {
            if logits.len() != max_order_size + 1 {
                return Err(PyValueError::new_err(format!(
                    "two-stage ordinal logits length {} does not match expected {}",
                    logits.len(),
                    max_order_size + 1
                )));
            }
            let gate_prob = sigmoid(logits[0]);
            if gate_prob < 0.5 {
                return Ok(0);
            }
            let quantity_score: f32 = logits[1..].iter().map(|value| sigmoid(*value)).sum();
            let action = quantity_score.round().clamp(1.0, max_order_size as f32) as usize;
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
    max_order_size: usize,
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
    dense_action_from_logits(&logits, policy_head, max_order_size)
}

pub fn mlp_action_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    hidden_dims: &[usize],
    output_dim: usize,
    activation: ActivationKind,
    policy_head: DensePolicyHead,
    max_order_size: usize,
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
    dense_action_from_logits(&logits, policy_head, max_order_size)
}
