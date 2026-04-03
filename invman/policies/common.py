def get_activation_function(activation="gelu"):
    import torch.nn.functional as F

    if activation == "selu":
        return F.selu
    if activation == "gelu":
        return F.gelu
    if activation == "relu":
        return F.relu
    raise NotImplementedError(f"Unsupported activation: {activation}")


def normalize_policy_head(policy_head: str) -> str:
    aliases = {
        "categorical_quantity": "categorical_quantity",
        "direct_quantity": "direct_quantity",
        "positive_quantity": "direct_quantity",
        "softplus_quantity": "direct_quantity",
        "nonnegative_quantity": "direct_quantity",
        "sigmoid_direct_quantity": "sigmoid_direct_quantity",
        "scaled_direct_quantity": "sigmoid_direct_quantity",
        "q_scaled_direct_quantity": "sigmoid_direct_quantity",
        "soft_gated_direct_quantity": "soft_gated_direct_quantity",
        "gated_direct_quantity": "soft_gated_direct_quantity",
        "gated_direct": "soft_gated_direct_quantity",
        "gated_positive_quantity": "soft_gated_direct_quantity",
        "soft_gated_positive_quantity": "soft_gated_direct_quantity",
        "gated_sigmoid_direct_quantity": "gated_sigmoid_direct_quantity",
        "scaled_gated_direct_quantity": "gated_sigmoid_direct_quantity",
        "q_scaled_gated_direct_quantity": "gated_sigmoid_direct_quantity",
        "hard_gated_direct_quantity": "hard_gated_direct_quantity",
        "two_stage_direct_quantity": "hard_gated_direct_quantity",
        "two_stage_positive_quantity": "hard_gated_direct_quantity",
        "hard_gated_positive_quantity": "hard_gated_direct_quantity",
        "bounded_quantity": "bounded_quantity",
        "vector_quantity": "bounded_quantity",
        "bounded_vector_quantity": "bounded_quantity",
        "soft_gated_ordinal_quantity": "soft_gated_ordinal_quantity",
        "gated_ordinal_quantity": "soft_gated_ordinal_quantity",
        "hard_gated_ordinal_quantity": "hard_gated_ordinal_quantity",
        "two_stage_ordinal_quantity": "hard_gated_ordinal_quantity",
        "conditional_ordinal_quantity": "hard_gated_ordinal_quantity",
        "discrete_logits": "categorical_quantity",
        "scalar_quantity": "direct_quantity",
    }
    normalized = aliases.get(policy_head)
    if normalized is None:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown policy head '{policy_head}'. Expected one of: {valid}")
    return normalized


def normalize_tree_split_type(tree_split_type: str) -> str:
    aliases = {
        "oblique": "oblique",
        "axis_aligned": "axis_aligned",
        "axis": "axis_aligned",
    }
    normalized = aliases.get(tree_split_type)
    if normalized is None:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown tree split type '{tree_split_type}'. Expected one of: {valid}")
    return normalized


def normalize_tree_leaf_type(tree_leaf_type: str) -> str:
    aliases = {
        "constant": "constant",
        "linear": "linear",
        "positive_linear": "linear",
        "softplus_linear": "linear",
        "nonnegative_linear": "linear",
        "sigmoid_linear": "sigmoid_linear",
        "scaled_linear": "sigmoid_linear",
    }
    normalized = aliases.get(tree_leaf_type)
    if normalized is None:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown tree leaf type '{tree_leaf_type}'. Expected one of: {valid}")
    return normalized


def normalize_tree_action_adapter(tree_action_adapter: str) -> str:
    from invman.problems.dual_sourcing.policies import normalize_action_adapter as _normalize_tree_action_adapter

    return _normalize_tree_action_adapter(tree_action_adapter)


def normalize_action_adapter(action_adapter: str) -> str:
    return normalize_tree_action_adapter(action_adapter)


def normalize_action_mode(action_mode: str) -> str:
    aliases = {
        "scalar_quantity": "scalar_quantity",
        "scalar": "scalar_quantity",
        "vector_quantity": "vector_quantity",
        "vector": "vector_quantity",
        "discrete_grid": "discrete_grid",
        "grid": "discrete_grid",
    }
    normalized = aliases.get(action_mode)
    if normalized is None:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown action mode '{action_mode}'. Expected one of: {valid}")
    return normalized


def build_scalar_action_spec(max_order_size: int) -> dict:
    max_value = int(max_order_size)
    if max_value < 0:
        raise ValueError("max_order_size must be non-negative")
    return {
        "action_dim": 1,
        "action_mode": "scalar_quantity",
        "min_values": [0],
        "max_values": [max_value],
        "allowed_values": None,
    }


def normalize_action_spec(action_spec=None, *, default_max_order_size=None) -> dict:
    if action_spec is None:
        if default_max_order_size is None:
            raise ValueError("action_spec is required when default_max_order_size is not provided")
        return build_scalar_action_spec(default_max_order_size)

    if not isinstance(action_spec, dict):
        raise ValueError("action_spec must be a dictionary")

    action_mode = normalize_action_mode(str(action_spec.get("action_mode", "scalar_quantity")))
    action_dim = int(action_spec.get("action_dim", 1))
    if action_dim < 1:
        raise ValueError("action_dim must be at least 1")

    allowed_values = action_spec.get("allowed_values")
    normalized_allowed = None
    if action_mode == "discrete_grid":
        if allowed_values is None:
            raise ValueError("discrete_grid action specs require allowed_values")
        normalized_allowed = []
        if len(allowed_values) != action_dim:
            raise ValueError("len(allowed_values) must equal action_dim")
        for idx, values in enumerate(allowed_values):
            if not values:
                raise ValueError(f"allowed_values[{idx}] must be non-empty")
            sorted_values = sorted({int(value) for value in values})
            normalized_allowed.append(sorted_values)
    raw_max_values = action_spec.get("max_values")
    if raw_max_values is None:
        if normalized_allowed is not None:
            max_values = [int(values[-1]) for values in normalized_allowed]
        elif default_max_order_size is not None:
            max_values = [int(default_max_order_size)] * action_dim
        else:
            raise ValueError("action_spec must define max_values")
    else:
        max_values = [int(value) for value in raw_max_values]
        if len(max_values) != action_dim:
            raise ValueError("len(max_values) must equal action_dim")

    raw_min_values = action_spec.get("min_values")
    if raw_min_values is None:
        if normalized_allowed is not None:
            min_values = [int(values[0]) for values in normalized_allowed]
        else:
            min_values = [0] * action_dim
    else:
        min_values = [int(value) for value in raw_min_values]
        if len(min_values) != action_dim:
            raise ValueError("len(min_values) must equal action_dim")

    for min_value, max_value in zip(min_values, max_values):
        if min_value > max_value:
            raise ValueError("each min_value must be <= max_value")

    return {
        "action_dim": action_dim,
        "action_mode": action_mode,
        "min_values": min_values,
        "max_values": max_values,
        "allowed_values": normalized_allowed,
    }
