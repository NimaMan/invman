from invman.policies.linear import LinearPolicyNet
from invman.policies.neural import PolicyNet
from invman.policies.registry import get_policy_spec
from invman.policies.soft_tree import SoftTreePolicy
from invman.problems import get_problem_module


def _policy_max_order_size(action_spec):
    max_values = action_spec.get("max_values")
    if not max_values:
        return None
    return int(max_values[0])


def _dense_output_dim(policy_decoder: str, action_spec: dict) -> int:
    if policy_decoder in {
        "categorical_quantity",
        "soft_gated_ordinal_quantity",
        "hard_gated_ordinal_quantity",
    }:
        max_order_size = _policy_max_order_size(action_spec)
        if max_order_size is None:
            raise ValueError(f"{policy_decoder} requires a finite scalar action spec")
        return max_order_size + 1
    return int(action_spec["action_dim"])


def build_policy(args, env):
    policy_spec = get_policy_spec(args)
    problem_module = get_problem_module(getattr(args, "problem", "lost_sales"))
    context = problem_module.build_policy_context(args, env)
    supported_policy_backbones = tuple(context.get("supported_policy_backbones", ()))
    if supported_policy_backbones and policy_spec.policy_backbone not in supported_policy_backbones:
        valid = ", ".join(supported_policy_backbones)
        raise ValueError(f"Problem '{args.problem}' supports policy backbones: {valid}")

    action_spec = context["action_spec"]
    control_spec = context.get("control_spec")
    action_adapter = context.get("action_adapter", "identity")
    action_adapter_config = context.get("action_adapter_config")
    state_normalizer = context.get("state_normalizer", "identity")
    state_scale = context.get("state_scale")
    policy_max_order_size = _policy_max_order_size(action_spec)

    if policy_spec.policy_backbone == "linear":
        return LinearPolicyNet(
            input_dim=env.state_space_dim,
            output_dim=_dense_output_dim(policy_spec.policy_decoder, action_spec),
            action_output_mode=policy_spec.policy_decoder,
            max_order_size=policy_max_order_size,
            action_spec=action_spec,
            control_spec=control_spec,
            action_adapter=action_adapter,
            action_adapter_config=action_adapter_config,
            state_normalizer=state_normalizer,
            state_scale=state_scale,
        )
    if policy_spec.policy_backbone == "nn":
        return PolicyNet(
            input_dim=env.state_space_dim,
            hidden_dim=policy_spec.hidden_dim,
            output_dim=_dense_output_dim(policy_spec.policy_decoder, action_spec),
            activation=policy_spec.activation,
            action_output_mode=policy_spec.policy_decoder,
            max_order_size=policy_max_order_size,
            action_spec=action_spec,
            control_spec=control_spec,
            action_adapter=action_adapter,
            action_adapter_config=action_adapter_config,
            state_normalizer=state_normalizer,
            state_scale=state_scale,
        )
    if policy_spec.policy_backbone == "soft_tree":
        return SoftTreePolicy(
            input_dim=env.state_space_dim,
            max_order_size=policy_max_order_size,
            action_spec=action_spec,
            control_spec=control_spec,
            depth=policy_spec.tree_depth,
            temperature=policy_spec.tree_temperature,
            split_type=policy_spec.tree_split_type,
            leaf_type=policy_spec.tree_leaf_type,
            action_adapter=action_adapter,
            action_adapter_config=action_adapter_config,
            state_normalizer=state_normalizer,
            state_scale=state_scale,
        )
    raise NotImplementedError(f"Unknown policy backbone: {policy_spec.policy_backbone}")
