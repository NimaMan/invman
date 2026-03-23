"""Policy support for the multi-echelon benchmark."""

from invman.policies.common import normalize_action_spec

SUPPORTED_POLICY_TYPES = ("soft_tree",)


def build_policy_context(args, env):
    del args
    action_spec = normalize_action_spec(getattr(env, "action_spec", None))
    return {
        "supported_policy_types": SUPPORTED_POLICY_TYPES,
        "action_spec": action_spec,
        "control_spec": None,
        "action_adapter": "identity",
        "action_adapter_config": None,
    }
