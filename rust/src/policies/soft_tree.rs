use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftTreeSplitType {
    Oblique,
    AxisAligned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftTreeLeafType {
    Constant,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftTreeActionMode {
    ScalarQuantity,
    VectorQuantity,
    DiscreteGrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftTreeActionAdapter {
    Identity,
    DualSourcingSingleIndexTargets,
    DualSourcingDualIndexTargets,
    DualSourcingCappedDualIndexTargets,
    DualSourcingBaseSurgeTargets,
}

#[derive(Clone, Debug)]
pub struct SoftTreeActionSpec {
    pub action_dim: usize,
    pub action_mode: SoftTreeActionMode,
    pub min_values: Vec<usize>,
    pub max_values: Vec<usize>,
    pub allowed_values: Option<Vec<Vec<usize>>>,
}

pub fn parse_split_type(split_type: &str) -> PyResult<SoftTreeSplitType> {
    match split_type {
        "oblique" => Ok(SoftTreeSplitType::Oblique),
        "axis_aligned" | "axis" => Ok(SoftTreeSplitType::AxisAligned),
        _ => Err(PyValueError::new_err(format!(
            "unknown soft tree split type '{split_type}'; expected 'oblique' or 'axis_aligned'"
        ))),
    }
}

pub fn parse_leaf_type(leaf_type: &str) -> PyResult<SoftTreeLeafType> {
    match leaf_type {
        "constant" => Ok(SoftTreeLeafType::Constant),
        "linear" => Ok(SoftTreeLeafType::Linear),
        _ => Err(PyValueError::new_err(format!(
            "unknown soft tree leaf type '{leaf_type}'; expected 'constant' or 'linear'"
        ))),
    }
}

pub fn parse_action_mode(action_mode: &str) -> PyResult<SoftTreeActionMode> {
    match action_mode {
        "scalar_quantity" | "scalar" => Ok(SoftTreeActionMode::ScalarQuantity),
        "vector_quantity" | "vector" => Ok(SoftTreeActionMode::VectorQuantity),
        "discrete_grid" | "grid" => Ok(SoftTreeActionMode::DiscreteGrid),
        _ => Err(PyValueError::new_err(format!(
            "unknown soft tree action mode '{action_mode}'; expected 'scalar_quantity', 'vector_quantity', or 'discrete_grid'"
        ))),
    }
}

pub fn parse_action_adapter(action_adapter: &str) -> PyResult<SoftTreeActionAdapter> {
    match action_adapter {
        "identity" | "direct" | "direct_orders" => Ok(SoftTreeActionAdapter::Identity),
        "dual_sourcing_single_index_targets" | "single_index_targets" => {
            Ok(SoftTreeActionAdapter::DualSourcingSingleIndexTargets)
        }
        "dual_sourcing_dual_index_targets" | "dual_index_targets" => {
            Ok(SoftTreeActionAdapter::DualSourcingDualIndexTargets)
        }
        "dual_sourcing_capped_dual_index_targets" | "capped_dual_index_targets" => {
            Ok(SoftTreeActionAdapter::DualSourcingCappedDualIndexTargets)
        }
        "dual_sourcing_base_surge_targets" | "base_surge_targets" => {
            Ok(SoftTreeActionAdapter::DualSourcingBaseSurgeTargets)
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown soft tree action adapter '{action_adapter}'"
        ))),
    }
}

pub fn build_action_spec(
    action_mode: &str,
    min_values: Vec<usize>,
    max_values: Vec<usize>,
    allowed_values: Option<Vec<Vec<usize>>>,
) -> PyResult<SoftTreeActionSpec> {
    let parsed_action_mode = parse_action_mode(action_mode)?;
    if min_values.len() != max_values.len() {
        return Err(PyValueError::new_err("min_values and max_values must have the same length"));
    }
    if min_values.is_empty() {
        return Err(PyValueError::new_err("action specs must contain at least one dimension"));
    }
    if let Some(ref values) = allowed_values {
        if values.len() != min_values.len() {
            return Err(PyValueError::new_err("allowed_values must match action dimensionality"));
        }
        if parsed_action_mode != SoftTreeActionMode::DiscreteGrid {
            return Err(PyValueError::new_err(
                "allowed_values may only be provided for discrete_grid action specs",
            ));
        }
        for allowed in values.iter() {
            if allowed.is_empty() {
                return Err(PyValueError::new_err("each allowed_values entry must be non-empty"));
            }
        }
    }
    Ok(SoftTreeActionSpec {
        action_dim: min_values.len(),
        action_mode: parsed_action_mode,
        min_values,
        max_values,
        allowed_values,
    })
}

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
    leaf_type: SoftTreeLeafType,
    action_dim: usize,
) -> PyResult<(usize, usize, usize)> {
    if depth < 1 {
        return Err(PyValueError::new_err("depth must be at least 1"));
    }
    if input_dim < 1 {
        return Err(PyValueError::new_err("input_dim must be at least 1"));
    }
    if action_dim < 1 {
        return Err(PyValueError::new_err("action_dim must be at least 1"));
    }
    let num_internal_nodes = (1usize << depth) - 1;
    let num_leaves = 1usize << depth;
    let weights_end = num_internal_nodes * input_dim;
    let bias_end = weights_end + num_internal_nodes;
    let leaf_param_count = match leaf_type {
        SoftTreeLeafType::Constant => num_leaves * action_dim,
        SoftTreeLeafType::Linear => num_leaves * action_dim * input_dim + num_leaves * action_dim,
    };
    let expected_len = bias_end + leaf_param_count;
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
    split_type: SoftTreeSplitType,
) -> Vec<f32> {
    let state_len = state.len();
    let num_internal_nodes = (1usize << depth) - 1;
    let mut gates = vec![0.0f32; num_internal_nodes];
    for node_idx in 0..num_internal_nodes {
        let start = node_idx * state_len;
        let logit = match split_type {
            SoftTreeSplitType::Oblique => {
                let mut value = split_bias[node_idx];
                for feat_idx in 0..state_len {
                    value += split_weights[start + feat_idx] * state[feat_idx];
                }
                value
            }
            SoftTreeSplitType::AxisAligned => {
                let mut best_feat_idx = 0usize;
                let mut best_abs_weight = f32::NEG_INFINITY;
                for feat_idx in 0..state_len {
                    let abs_weight = split_weights[start + feat_idx].abs();
                    if abs_weight > best_abs_weight {
                        best_abs_weight = abs_weight;
                        best_feat_idx = feat_idx;
                    }
                }
                let selected_weight = split_weights[start + best_feat_idx];
                split_bias[node_idx] + selected_weight * state[best_feat_idx]
            }
        };
        gates[node_idx] = if logit.is_nan() {
            0.5
        } else {
            1.0 / (1.0 + (-(logit / temperature)).exp())
        };
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

fn leaf_output_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    bias_end: usize,
    num_leaves: usize,
    leaf_idx: usize,
    leaf_type: SoftTreeLeafType,
    action_dim: usize,
) -> Vec<f32> {
    match leaf_type {
        SoftTreeLeafType::Constant => {
            let start = bias_end + leaf_idx * action_dim;
            flat_params[start..start + action_dim].to_vec()
        }
        SoftTreeLeafType::Linear => {
            let weights_len = num_leaves * action_dim * input_dim;
            let weights_start = bias_end;
            let bias_start = bias_end + weights_len;
            let mut outputs = vec![0.0f32; action_dim];
            for action_idx in 0..action_dim {
                let row_start = weights_start + leaf_idx * action_dim * input_dim + action_idx * input_dim;
                let mut raw = flat_params[bias_start + leaf_idx * action_dim + action_idx];
                for feat_idx in 0..input_dim {
                    raw += flat_params[row_start + feat_idx] * state[feat_idx];
                }
                outputs[action_idx] = raw;
            }
            outputs
        }
    }
}

fn project_action_value(action_value: &[f32], action_spec: &SoftTreeActionSpec) -> Vec<usize> {
    let mut projected = Vec::with_capacity(action_spec.action_dim);
    match action_spec.action_mode {
        SoftTreeActionMode::ScalarQuantity | SoftTreeActionMode::VectorQuantity => {
            for (dim_idx, value) in action_value.iter().enumerate() {
                let min_value = action_spec.min_values[dim_idx] as f32;
                let max_value = action_spec.max_values[dim_idx] as f32;
                let clipped = value.round().clamp(min_value, max_value);
                projected.push(clipped as usize);
            }
        }
        SoftTreeActionMode::DiscreteGrid => {
            let allowed_values = action_spec.allowed_values.as_ref().expect("discrete grid requires allowed values");
            for (dim_idx, value) in action_value.iter().enumerate() {
                let mut best = allowed_values[dim_idx][0];
                let mut best_distance = (best as f32 - *value).abs();
                for candidate in allowed_values[dim_idx].iter().copied().skip(1) {
                    let distance = (candidate as f32 - *value).abs();
                    if distance < best_distance {
                        best = candidate;
                        best_distance = distance;
                    }
                }
                projected.push(best);
            }
        }
    }
    projected
}

pub fn action_vector_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    depth: usize,
    temperature: f32,
    split_type: SoftTreeSplitType,
    leaf_type: SoftTreeLeafType,
    action_spec: &SoftTreeActionSpec,
) -> PyResult<Vec<usize>> {
    if temperature <= 0.0 {
        return Err(PyValueError::new_err("temperature must be positive"));
    }
    let action_dim = action_spec.action_dim;
    let (weights_end, bias_end, num_leaves) =
        validate_soft_tree_flat_params(flat_params.len(), input_dim, depth, leaf_type, action_dim)?;
    if state.len() != input_dim {
        return Err(PyValueError::new_err(format!(
            "state length {} does not match input_dim {}",
            state.len(),
            input_dim
        )));
    }

    let split_weights = &flat_params[..weights_end];
    let split_bias = &flat_params[weights_end..bias_end];
    let leaf_probs = soft_tree_leaf_probabilities(
        state,
        split_weights,
        split_bias,
        depth,
        temperature,
        split_type,
    );

    let mut action_value = vec![0.0f32; action_dim];
    for (leaf_idx, leaf_prob) in leaf_probs.iter().enumerate() {
        let leaf_output = leaf_output_from_flat_params(
            state,
            flat_params,
            input_dim,
            bias_end,
            num_leaves,
            leaf_idx,
            leaf_type,
            action_dim,
        );
        for action_idx in 0..action_dim {
            let min_value = action_spec.min_values[action_idx] as f32;
            let max_value = action_spec.max_values[action_idx] as f32;
            let span = max_value - min_value;
            let scaled = min_value + (1.0 / (1.0 + (-leaf_output[action_idx]).exp())) * span;
            action_value[action_idx] += leaf_prob * scaled;
        }
    }
    Ok(project_action_value(&action_value, action_spec))
}

pub fn dual_sourcing_action_from_controls(
    reduced_state: &[i64],
    controls: &[usize],
    action_adapter: SoftTreeActionAdapter,
    regular_max_order_size: usize,
    expedited_max_order_size: usize,
) -> PyResult<Vec<usize>> {
    let expedited_inventory_position = reduced_state[0];
    let regular_inventory_position = reduced_state.iter().sum::<i64>();
    match action_adapter {
        SoftTreeActionAdapter::Identity => {
            if controls.len() != 2 {
                return Err(PyValueError::new_err("identity dual-sourcing control vector must have length 2"));
            }
            Ok(vec![
                controls[0].min(regular_max_order_size),
                controls[1].min(expedited_max_order_size),
            ])
        }
        SoftTreeActionAdapter::DualSourcingSingleIndexTargets => {
            if controls.len() != 2 {
                return Err(PyValueError::new_err("single-index target control vector must have length 2"));
            }
            let s_e = controls[0] as i64;
            let s_r = (controls[1].max(controls[0])) as i64;
            let expedited = (s_e - regular_inventory_position).max(0) as usize;
            let expedited = expedited.min(expedited_max_order_size);
            let regular = (s_r - regular_inventory_position - expedited as i64).max(0) as usize;
            Ok(vec![regular.min(regular_max_order_size), expedited])
        }
        SoftTreeActionAdapter::DualSourcingDualIndexTargets => {
            if controls.len() != 2 {
                return Err(PyValueError::new_err("dual-index target control vector must have length 2"));
            }
            let s_e = controls[0] as i64;
            let s_r = (controls[1].max(controls[0])) as i64;
            let expedited = (s_e - expedited_inventory_position).max(0) as usize;
            let expedited = expedited.min(expedited_max_order_size);
            let regular = (s_r - regular_inventory_position - expedited as i64).max(0) as usize;
            Ok(vec![regular.min(regular_max_order_size), expedited])
        }
        SoftTreeActionAdapter::DualSourcingCappedDualIndexTargets => {
            if controls.len() != 3 {
                return Err(PyValueError::new_err("capped dual-index target control vector must have length 3"));
            }
            let s_e = controls[0] as i64;
            let s_r = (controls[1].max(controls[0])) as i64;
            let cap_r = controls[2];
            let expedited = (s_e - expedited_inventory_position).max(0) as usize;
            let expedited = expedited.min(expedited_max_order_size);
            let desired_regular = (s_r - regular_inventory_position - expedited as i64).max(0) as usize;
            Ok(vec![desired_regular.min(cap_r).min(regular_max_order_size), expedited])
        }
        SoftTreeActionAdapter::DualSourcingBaseSurgeTargets => {
            if controls.len() != 2 {
                return Err(PyValueError::new_err("base-surge target control vector must have length 2"));
            }
            let surge_level = controls[0] as i64;
            let regular_qty = controls[1];
            let expedited = (surge_level - expedited_inventory_position).max(0) as usize;
            Ok(vec![
                regular_qty.min(regular_max_order_size),
                expedited.min(expedited_max_order_size),
            ])
        }
    }
}

pub fn action_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    depth: usize,
    max_order_size: usize,
    temperature: f32,
    split_type: SoftTreeSplitType,
    leaf_type: SoftTreeLeafType,
) -> PyResult<usize> {
    let action_spec = SoftTreeActionSpec {
        action_dim: 1,
        action_mode: SoftTreeActionMode::ScalarQuantity,
        min_values: vec![0],
        max_values: vec![max_order_size],
        allowed_values: None,
    };
    Ok(action_vector_from_flat_params(
        state,
        flat_params,
        input_dim,
        depth,
        temperature,
        split_type,
        leaf_type,
        &action_spec,
    )?[0])
}
