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
        "gated_ordinal_quantity": "gated_ordinal_quantity",
        "two_stage_ordinal_quantity": "two_stage_ordinal_quantity",
        "conditional_ordinal_quantity": "two_stage_ordinal_quantity",
        "discrete_logits": "categorical_quantity",
        "scalar_quantity": "direct_quantity",
    }
    normalized = aliases.get(policy_head)
    if normalized is None:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown policy head '{policy_head}'. Expected one of: {valid}")
    return normalized
