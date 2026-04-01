from invman.policies.linear import LinearPolicyNet
from invman.policies.neural import PolicyNet
from invman.policies.registry import get_policy_spec
from invman.policies.soft_tree import SoftTreePolicy
from invman.problems import get_problem_module


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

    if policy_spec.policy_backbone == "linear":
        return LinearPolicyNet(
            input_dim=env.state_space_dim,
            output_dim=env.action_space_dim,
            action_output_mode=policy_spec.policy_decoder,
            max_order_size=getattr(env, "max_order_size", None),
            action_spec=action_spec,
            control_spec=control_spec,
            action_adapter=action_adapter,
            action_adapter_config=action_adapter_config,
        )
    if policy_spec.policy_backbone == "nn":
        return PolicyNet(
            input_dim=env.state_space_dim,
            hidden_dim=policy_spec.hidden_dim,
            output_dim=env.action_space_dim,
            activation=policy_spec.activation,
            action_output_mode=policy_spec.policy_decoder,
            max_order_size=getattr(env, "max_order_size", None),
            action_spec=action_spec,
            control_spec=control_spec,
            action_adapter=action_adapter,
            action_adapter_config=action_adapter_config,
        )
    if policy_spec.policy_backbone == "soft_tree":
        return SoftTreePolicy(
            input_dim=env.state_space_dim,
            max_order_size=getattr(env, "max_order_size", None),
            action_spec=action_spec,
            control_spec=control_spec,
            depth=policy_spec.tree_depth,
            temperature=policy_spec.tree_temperature,
            split_type=policy_spec.tree_split_type,
            leaf_type=policy_spec.tree_leaf_type,
            action_adapter=action_adapter,
            action_adapter_config=action_adapter_config,
        )
    raise NotImplementedError(f"Unknown policy backbone: {policy_spec.policy_backbone}")
