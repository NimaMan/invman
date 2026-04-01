"""Policy support for the fixed-order-cost lost-sales problem."""

from invman.policies.common import normalize_action_spec

SUPPORTED_POLICY_BACKBONES = ("linear", "nn", "soft_tree")


def build_policy_context(args, env):
    del args
    action_spec = normalize_action_spec(
        getattr(env, "action_spec", None),
        default_max_order_size=getattr(env, "max_order_size", None),
    )
    return {
        "supported_policy_backbones": SUPPORTED_POLICY_BACKBONES,
        "action_spec": action_spec,
        "control_spec": None,
        "action_adapter": "identity",
        "action_adapter_config": None,
    }
