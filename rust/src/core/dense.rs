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

pub fn linear_categorical_action_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    output_dim: usize,
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
    let logits = dense_forward(state, &flat_params[..split], &flat_params[split..], output_dim);
    Ok(argmax_first(&logits))
}

pub fn mlp_categorical_action_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    hidden_dims: &[usize],
    output_dim: usize,
    activation: ActivationKind,
) -> PyResult<usize> {
    if state.len() != input_dim {
        return Err(PyValueError::new_err(format!(
            "state length {} does not match input_dim {}",
            state.len(),
            input_dim
        )));
    }
    if hidden_dims.is_empty() {
        return Err(PyValueError::new_err("hidden_dims must be non-empty for mlp policy"));
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
    Ok(argmax_first(&logits))
}
