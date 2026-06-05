use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::core::policies::dense::{
    linear_action_from_flat_params, mlp_action_from_flat_params, parse_activation,
    parse_policy_head,
};
use crate::core::policies::soft_tree::{
    action_from_flat_params, action_vector_from_flat_params, build_action_spec, parse_leaf_type,
    parse_split_type, soft_tree_leaf_probabilities, validate_soft_tree_shapes,
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

#[pyfunction]
fn soft_tree_action_vector_from_flat_params(
    state: Vec<f32>,
    flat_params: Vec<f32>,
    input_dim: usize,
    depth: usize,
    temperature: f32,
    split_type: &str,
    leaf_type: &str,
    control_mode: &str,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<usize>> {
    let action_spec = build_action_spec(control_mode, min_values, max_values, allowed_values)?;
    action_vector_from_flat_params(
        &state,
        &flat_params,
        input_dim,
        depth,
        temperature,
        parse_split_type(split_type)?,
        parse_leaf_type(leaf_type)?,
        &action_spec,
    )
}

#[pyfunction]
fn linear_policy_action_from_flat_params(
    state: Vec<f32>,
    flat_params: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
    policy_head: &str,
    policy_max_quantity: Option<usize>,
) -> PyResult<usize> {
    linear_action_from_flat_params(
        &state,
        &flat_params,
        input_dim,
        output_dim,
        parse_policy_head(policy_head)?,
        policy_max_quantity,
    )
}

#[pyfunction]
fn nn_policy_action_from_flat_params(
    state: Vec<f32>,
    flat_params: Vec<f32>,
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    activation: &str,
    policy_head: &str,
    policy_max_quantity: Option<usize>,
) -> PyResult<usize> {
    mlp_action_from_flat_params(
        &state,
        &flat_params,
        input_dim,
        &hidden_dims,
        output_dim,
        parse_activation(activation)?,
        parse_policy_head(policy_head)?,
        policy_max_quantity,
    )
}

pub fn register_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soft_tree_action, m)?)?;
    m.add_function(wrap_pyfunction!(soft_tree_action_from_flat_params, m)?)?;
    m.add_function(wrap_pyfunction!(
        soft_tree_action_vector_from_flat_params,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(linear_policy_action_from_flat_params, m)?)?;
    m.add_function(wrap_pyfunction!(nn_policy_action_from_flat_params, m)?)?;
    Ok(())
}
