use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

pub fn validate_soft_tree_shapes(
    state_len: usize,
    split_weights_len: usize,
    split_bias_len: usize,
    leaf_logits_len: usize,
    depth: usize,
) -> PyResult<()> {
    if depth < 1 {
        return Err(PyValueError::new_err("depth must be at least 1"));
    }
    let num_internal_nodes = (1usize << depth) - 1;
    let num_leaves = 1usize << depth;
    let expected_weights = num_internal_nodes * state_len;
    if split_weights_len != expected_weights {
        return Err(PyValueError::new_err(format!(
            "split_weights length {} does not match expected {}",
            split_weights_len, expected_weights
        )));
    }
    if split_bias_len != num_internal_nodes {
        return Err(PyValueError::new_err(format!(
            "split_bias length {} does not match expected {}",
            split_bias_len, num_internal_nodes
        )));
    }
    if leaf_logits_len != num_leaves {
        return Err(PyValueError::new_err(format!(
            "leaf_logits length {} does not match expected {}",
            leaf_logits_len, num_leaves
        )));
    }
    Ok(())
}

pub fn validate_soft_tree_flat_params(
    flat_params_len: usize,
    input_dim: usize,
    depth: usize,
) -> PyResult<(usize, usize, usize)> {
    if depth < 1 {
        return Err(PyValueError::new_err("depth must be at least 1"));
    }
    if input_dim < 1 {
        return Err(PyValueError::new_err("input_dim must be at least 1"));
    }
    let num_internal_nodes = (1usize << depth) - 1;
    let num_leaves = 1usize << depth;
    let weights_end = num_internal_nodes * input_dim;
    let bias_end = weights_end + num_internal_nodes;
    let expected_len = bias_end + num_leaves;
    if flat_params_len != expected_len {
        return Err(PyValueError::new_err(format!(
            "flat_params length {} does not match expected {}",
            flat_params_len, expected_len
        )));
    }
    Ok((weights_end, bias_end, num_leaves))
}

pub fn soft_tree_leaf_probabilities(
    state: &[f32],
    split_weights: &[f32],
    split_bias: &[f32],
    depth: usize,
    temperature: f32,
) -> Vec<f32> {
    let state_len = state.len();
    let num_internal_nodes = (1usize << depth) - 1;
    let mut gates = vec![0.0f32; num_internal_nodes];
    for node_idx in 0..num_internal_nodes {
        let start = node_idx * state_len;
        let mut logit = split_bias[node_idx];
        for feat_idx in 0..state_len {
            logit += split_weights[start + feat_idx] * state[feat_idx];
        }
        gates[node_idx] = 1.0 / (1.0 + (-(logit / temperature)).exp());
    }

    let mut level_probs = vec![1.0f32];
    for level in 0..depth {
        let start_idx = (1usize << level) - 1;
        let mut next_level_probs = Vec::with_capacity(level_probs.len() * 2);
        for (offset, parent_prob) in level_probs.iter().enumerate() {
            let gate = gates[start_idx + offset];
            next_level_probs.push(parent_prob * (1.0 - gate));
            next_level_probs.push(parent_prob * gate);
        }
        level_probs = next_level_probs;
    }
    level_probs
}

pub fn action_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    depth: usize,
    max_order_size: usize,
    temperature: f32,
) -> PyResult<usize> {
    if temperature <= 0.0 {
        return Err(PyValueError::new_err("temperature must be positive"));
    }
    let (weights_end, bias_end, _) =
        validate_soft_tree_flat_params(flat_params.len(), input_dim, depth)?;
    if state.len() != input_dim {
        return Err(PyValueError::new_err(format!(
            "state length {} does not match input_dim {}",
            state.len(),
            input_dim
        )));
    }

    let split_weights = &flat_params[..weights_end];
    let split_bias = &flat_params[weights_end..bias_end];
    let leaf_logits = &flat_params[bias_end..];
    let leaf_probs = soft_tree_leaf_probabilities(state, split_weights, split_bias, depth, temperature);

    let mut action_value = 0.0f32;
    for (leaf_prob, leaf_logit) in leaf_probs.iter().zip(leaf_logits.iter()) {
        let quantity = 1.0 / (1.0 + (-leaf_logit).exp()) * max_order_size as f32;
        action_value += leaf_prob * quantity;
    }
    let clipped = action_value.round().clamp(0.0, max_order_size as f32);
    Ok(clipped as usize)
}
