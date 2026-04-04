use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::core::policies::soft_tree::{
    action_from_flat_params, parse_leaf_type, parse_split_type, soft_tree_leaf_probabilities,
    validate_soft_tree_shapes,
};

#[pyfunction]
#[pyo3(signature = (state, split_weights, split_bias, leaf_logits, depth, policy_max_quantity, temperature=0.25, split_type="oblique"))]
fn soft_tree_action(
    state: Vec<f32>,
    split_weights: Vec<f32>,
    split_bias: Vec<f32>,
    leaf_logits: Vec<f32>,
    depth: usize,
    policy_max_quantity: usize,
    temperature: f32,
    split_type: &str,
) -> PyResult<usize> {
    validate_soft_tree_shapes(
        state.len(),
        split_weights.len(),
        split_bias.len(),
        leaf_logits.len(),
        depth,
    )?;

    let leaf_probs = soft_tree_leaf_probabilities(
        &state,
        &split_weights,
        &split_bias,
        depth,
        temperature,
        parse_split_type(split_type)?,
    );
    let mut action_value = 0.0f32;
    for (leaf_prob, leaf_logit) in leaf_probs.iter().zip(leaf_logits.iter()) {
        let quantity = 1.0 / (1.0 + (-leaf_logit).exp()) * policy_max_quantity as f32;
        action_value += leaf_prob * quantity;
    }
    let clipped = action_value.round().clamp(0.0, policy_max_quantity as f32);
    Ok(clipped as usize)
}

#[pyfunction]
#[pyo3(signature = (
    state,
    flat_params,
    input_dim,
    depth,
    temperature=0.25,
    split_type="oblique",
    leaf_type="linear"
))]
fn soft_tree_action_from_flat_params(
    state: Vec<f32>,
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
) -> PyResult<usize> {
    action_from_flat_params(
        &state,
        &flat_params,
        input_dim,
        depth,
        temperature,
        parse_split_type(split_type)?,
        parse_leaf_type(leaf_type)?,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soft_tree_action, m)?)?;
    m.add_function(wrap_pyfunction!(soft_tree_action_from_flat_params, m)?)?;
    Ok(())
}
