"""Policy support for the fixed-order-cost lost-sales problem."""

from invman.policies.common import normalize_action_spec, normalize_state_normalizer

SUPPORTED_POLICY_BACKBONES = ("linear", "nn", "soft_tree")


def _state_normalization_config(args):
    state_normalizer = normalize_state_normalizer(
        getattr(args, "state_normalizer", "quantity_scale")
    )
    state_scale = getattr(args, "state_scale", None)
    if state_normalizer != "identity" and state_scale is None:
        state_scale = float(getattr(args, "max_order_size", 1))
    return state_normalizer, state_scale


def build_policy_context(args, env):
    del env
    action_spec = normalize_action_spec(
        getattr(args, "action_spec", None),
        default_max_order_size=getattr(args, "max_order_size", None),
    )
    state_normalizer, state_scale = _state_normalization_config(args)
    return {
        "supported_policy_backbones": SUPPORTED_POLICY_BACKBONES,
        "action_spec": action_spec,
        "control_spec": None,
        "action_adapter": "identity",
        "action_adapter_config": None,
        "state_normalizer": state_normalizer,
        "state_scale": state_scale,
    }
