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
    SigmoidLinear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftTreeActionMode {
    ScalarQuantity,
    VectorQuantity,
    DiscreteGrid,
    /// Gate-backbone residual head: the decoded order is `gate_order + round(Delta)`,
    /// where `gate_order` is supplied by a problem-specific gate (e.g. pairwise
    /// base-stock) and `Delta` is the SIGNED soft-tree residual from
    /// `action_residual_signed_from_flat_params` (identity leaf transform, so
    /// `Delta == 0` at the all-zero warm-start => the head reproduces the gate
    /// byte-exact). Requires `backbone_levels`; the problem rollout fuses the two.
    ResidualBaseStock,
}

#[derive(Clone, Debug)]
pub struct SoftTreeActionSpec {
    pub action_dim: usize,
    pub action_mode: SoftTreeActionMode,
    pub min_values: Vec<usize>,
    pub max_values: Vec<usize>,
    pub allowed_values: Option<Vec<Vec<usize>>>,
    /// For `ResidualBaseStock`: the fixed gate order-up-to level per action dimension
    /// (the structural backbone the residual is added to). `None` for all other modes.
    pub backbone_levels: Option<Vec<usize>>,
    /// For `ResidualBaseStock`: optional per-dimension group index (length == action_dim)
    /// that ties the residual within groups by averaging (e.g. per-echelon). `None` =>
    /// an independent residual per dimension (per-relation). Averaging zeros is zero, so
    /// this never breaks the `Delta == 0` gate-invertibility at the warm-start.
    pub residual_group_of: Option<Vec<usize>>,
}

impl SoftTreeActionSpec {
    /// Attach the gate backbone (and optional residual grouping) for `ResidualBaseStock`.
    /// Additive builder so existing `build_action_spec` callers are unaffected.
    pub fn with_backbone(
        mut self,
        backbone_levels: Option<Vec<usize>>,
        residual_group_of: Option<Vec<usize>>,
    ) -> Self {
        self.backbone_levels = backbone_levels;
        self.residual_group_of = residual_group_of;
        self
    }
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
        "positive_linear" | "softplus_linear" | "nonnegative_linear" => {
            Ok(SoftTreeLeafType::Linear)
        }
        "sigmoid_linear" | "scaled_linear" => Ok(SoftTreeLeafType::SigmoidLinear),
        _ => Err(PyValueError::new_err(format!(
            "unknown soft tree leaf type '{leaf_type}'; expected 'constant', 'linear', or 'sigmoid_linear'"
        ))),
    }
}

pub fn parse_action_mode(action_mode: &str) -> PyResult<SoftTreeActionMode> {
    match action_mode {
        "scalar_quantity" | "scalar" => Ok(SoftTreeActionMode::ScalarQuantity),
        "vector_quantity" | "vector" => Ok(SoftTreeActionMode::VectorQuantity),
        "discrete_grid" | "grid" => Ok(SoftTreeActionMode::DiscreteGrid),
        "residual_base_stock" | "residual" => Ok(SoftTreeActionMode::ResidualBaseStock),
        _ => Err(PyValueError::new_err(format!(
            "unknown soft tree action mode '{action_mode}'; expected 'scalar_quantity', 'vector_quantity', 'discrete_grid', or 'residual_base_stock'"
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
        return Err(PyValueError::new_err(
            "min_values and max_values must have the same length",
        ));
    }
    if min_values.is_empty() {
        return Err(PyValueError::new_err(
            "action specs must contain at least one dimension",
        ));
    }
    if let Some(ref values) = allowed_values {
        if values.len() != min_values.len() {
            return Err(PyValueError::new_err(
                "allowed_values must match action dimensionality",
            ));
        }
        if parsed_action_mode != SoftTreeActionMode::DiscreteGrid {
            return Err(PyValueError::new_err(
                "allowed_values may only be provided for discrete_grid action specs",
            ));
        }
        for allowed in values.iter() {
            if allowed.is_empty() {
                return Err(PyValueError::new_err(
                    "each allowed_values entry must be non-empty",
                ));
            }
        }
    }
    Ok(SoftTreeActionSpec {
        action_dim: min_values.len(),
        action_mode: parsed_action_mode,
        min_values,
        max_values,
        allowed_values,
        backbone_levels: None,
        residual_group_of: None,
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
        SoftTreeLeafType::Linear | SoftTreeLeafType::SigmoidLinear => {
            num_leaves * action_dim * input_dim + num_leaves * action_dim
        }
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
        SoftTreeLeafType::Linear | SoftTreeLeafType::SigmoidLinear => {
            let weights_len = num_leaves * action_dim * input_dim;
            let weights_start = bias_end;
            let bias_start = bias_end + weights_len;
            let mut outputs = vec![0.0f32; action_dim];
            for action_idx in 0..action_dim {
                let row_start =
                    weights_start + leaf_idx * action_dim * input_dim + action_idx * input_dim;
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
        // ResidualBaseStock never reaches this projection (its order is decoded in the
        // problem rollout: gate_order + round(signed residual), then clamped there). It is
        // listed here only to keep the match exhaustive; round+clip is a sane no-harm default.
        SoftTreeActionMode::ScalarQuantity
        | SoftTreeActionMode::VectorQuantity
        | SoftTreeActionMode::ResidualBaseStock => {
            for (dim_idx, value) in action_value.iter().enumerate() {
                let min_value = action_spec.min_values[dim_idx] as f32;
                let max_value = action_spec.max_values[dim_idx] as f32;
                let rounded = if *value >= 0.0 {
                    (*value + 0.5).floor()
                } else {
                    (*value - 0.5).ceil()
                };
                let clipped = rounded.clamp(min_value, max_value);
                projected.push(clipped as usize);
            }
        }
        SoftTreeActionMode::DiscreteGrid => {
            let allowed_values = action_spec
                .allowed_values
                .as_ref()
                .expect("discrete grid requires allowed values");
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
            let scaled = match leaf_type {
                SoftTreeLeafType::Constant | SoftTreeLeafType::SigmoidLinear => {
                    let max_value = action_spec.max_values[action_idx] as f32;
                    let span = max_value - min_value;
                    min_value + (1.0 / (1.0 + (-leaf_output[action_idx]).exp())) * span
                }
                SoftTreeLeafType::Linear => {
                    let raw = leaf_output[action_idx];
                    let softplus = raw.max(0.0) + (-(raw.abs())).exp().ln_1p();
                    min_value + softplus
                }
            };
            action_value[action_idx] += leaf_prob * scaled;
        }
    }
    if action_spec.action_dim == 1
        && action_spec.action_mode == SoftTreeActionMode::ScalarQuantity
        && matches!(
            leaf_type,
            SoftTreeLeafType::Linear | SoftTreeLeafType::SigmoidLinear
        )
    {
        let rounded = action_value[0].round().max(0.0) as usize;
        return Ok(vec![rounded]);
    }
    Ok(project_action_value(&action_value, action_spec))
}

/// Continuous-valued soft-tree action head.
///
/// Identical to `action_vector_from_flat_params` up to the final projection: it
/// returns the soft mixture of per-leaf outputs as CONTINUOUS `f32` values (after
/// the same per-dimension min + span/softplus leaf transform), WITHOUT the integer
/// rounding/clipping that `project_action_value` applies. This is the head used by
/// envs whose decisions are genuinely continuous (serial echelon base-stock levels;
/// the ameliorating purchase volume), so a learned policy can express a fractional
/// order-up-to level rather than being quantised to the nearest integer.
///
/// `min_values`/`max_values` are interpreted as f32 bounds: for `Constant` and
/// `SigmoidLinear` leaves the output is `min + sigmoid(leaf) * (max - min)`
/// (bounded to `[min, max]`); for `Linear` leaves the output is
/// `min + softplus(leaf)` (lower-bounded at `min`, unbounded above), matching the
/// integer head's leaf transform so the same warm-start encoding applies.
pub fn action_vector_continuous_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    depth: usize,
    temperature: f32,
    split_type: SoftTreeSplitType,
    leaf_type: SoftTreeLeafType,
    min_values: &[f32],
    max_values: &[f32],
) -> PyResult<Vec<f32>> {
    if temperature <= 0.0 {
        return Err(PyValueError::new_err("temperature must be positive"));
    }
    let action_dim = min_values.len();
    if action_dim == 0 {
        return Err(PyValueError::new_err(
            "continuous action head requires at least one dimension",
        ));
    }
    if max_values.len() != action_dim {
        return Err(PyValueError::new_err(
            "min_values and max_values must have the same length",
        ));
    }
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
            let min_value = min_values[action_idx];
            let scaled = match leaf_type {
                SoftTreeLeafType::Constant | SoftTreeLeafType::SigmoidLinear => {
                    let span = max_values[action_idx] - min_value;
                    min_value + (1.0 / (1.0 + (-leaf_output[action_idx]).exp())) * span
                }
                SoftTreeLeafType::Linear => {
                    let raw = leaf_output[action_idx];
                    let softplus = raw.max(0.0) + (-(raw.abs())).exp().ln_1p();
                    min_value + softplus
                }
            };
            action_value[action_idx] += leaf_prob * scaled;
        }
    }
    Ok(action_value)
}

/// Signed residual soft-tree head (IDENTITY leaf transform; exact neutral element 0).
///
/// Returns the soft mixture of per-leaf RAW outputs as SIGNED `f32` deltas, with NO
/// min/softplus/sigmoid transform applied. This is the load-bearing detail of the
/// `ResidualBaseStock` head: at the all-zero warm-start every leaf output is exactly 0
/// (constant leaf = 0 slice; linear leaf = bias(0) + weights(0)*state = 0), and the
/// convex leaf-probability mixture of identical-zero leaf outputs is 0 for ANY split
/// weights, so the returned delta is EXACTLY 0 (split-independent). The caller computes
/// `order = clamp(gate_order + round(delta), min, max)`, so delta=0 reproduces the gate
/// byte-exact (gate-invertible warm-start). NOTE: `min`/`max` bounds are intentionally
/// NOT applied here — the caller clamps `gate + delta` to the per-dimension order box.
pub fn action_residual_signed_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    depth: usize,
    temperature: f32,
    split_type: SoftTreeSplitType,
    leaf_type: SoftTreeLeafType,
    action_dim: usize,
) -> PyResult<Vec<f32>> {
    if temperature <= 0.0 {
        return Err(PyValueError::new_err("temperature must be positive"));
    }
    if action_dim == 0 {
        return Err(PyValueError::new_err(
            "residual action head requires at least one dimension",
        ));
    }
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

    let mut delta = vec![0.0f32; action_dim];
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
            // Identity transform: the neutral element is exactly 0 at zero params.
            delta[action_idx] += leaf_prob * leaf_output[action_idx];
        }
    }
    Ok(delta)
}

pub fn uncapped_scalar_action_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    depth: usize,
    temperature: f32,
    split_type: SoftTreeSplitType,
    leaf_type: SoftTreeLeafType,
) -> PyResult<usize> {
    if temperature <= 0.0 {
        return Err(PyValueError::new_err("temperature must be positive"));
    }
    if leaf_type != SoftTreeLeafType::Linear {
        return Err(PyValueError::new_err(
            "uncapped scalar soft-tree quantities require linear leaves",
        ));
    }
    let action_dim = 1usize;
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

    let mut action_value = 0.0f32;
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
        let raw = leaf_output[0];
        let softplus = raw.max(0.0) + (-(raw.abs())).exp().ln_1p();
        action_value += leaf_prob * softplus;
    }
    Ok(action_value.round().max(0.0) as usize)
}

pub fn action_from_flat_params(
    state: &[f32],
    flat_params: &[f32],
    input_dim: usize,
    depth: usize,
    temperature: f32,
    split_type: SoftTreeSplitType,
    leaf_type: SoftTreeLeafType,
) -> PyResult<usize> {
    uncapped_scalar_action_from_flat_params(
        state,
        flat_params,
        input_dim,
        depth,
        temperature,
        split_type,
        leaf_type,
    )
}
