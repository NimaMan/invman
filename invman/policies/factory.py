from invman.policies.linear import LinearPolicyNet
from invman.policies.neural import PolicyNet
from invman.policies.soft_tree import SoftTreePolicy
from invman.problems import get_problem_module


def build_policy(args, env):
    problem_module = get_problem_module(getattr(args, "problem", "lost_sales"))
    context = problem_module.build_policy_context(args, env)
    supported_policy_types = tuple(context.get("supported_policy_types", ()))
    if supported_policy_types and args.policy_type not in supported_policy_types:
        valid = ", ".join(supported_policy_types)
        raise ValueError(f"Problem '{args.problem}' supports policy types: {valid}")

    action_spec = context["action_spec"]
    control_spec = context.get("control_spec")
    action_adapter = context.get("action_adapter", "identity")
    action_adapter_config = context.get("action_adapter_config")

    if args.policy_type == "linear":
        return LinearPolicyNet(
            input_dim=env.state_space_dim,
            output_dim=env.action_space_dim,
            action_output_mode=args.policy_head,
            max_order_size=getattr(env, "max_order_size", None),
            action_spec=action_spec,
            control_spec=control_spec,
            action_adapter=action_adapter,
            action_adapter_config=action_adapter_config,
        )
    if args.policy_type == "nn":
        return PolicyNet(
            input_dim=env.state_space_dim,
            hidden_dim=args.hidden_dim,
            output_dim=env.action_space_dim,
            activation=args.activation,
            action_output_mode=args.policy_head,
            max_order_size=getattr(env, "max_order_size", None),
            action_spec=action_spec,
            control_spec=control_spec,
            action_adapter=action_adapter,
            action_adapter_config=action_adapter_config,
        )
    if args.policy_type == "soft_tree":
        return SoftTreePolicy(
            input_dim=env.state_space_dim,
            max_order_size=getattr(env, "max_order_size", None),
            action_spec=action_spec,
            control_spec=control_spec,
            depth=args.tree_depth,
            temperature=args.tree_temperature,
            split_type=args.tree_split_type,
            leaf_type=args.tree_leaf_type,
            action_adapter=action_adapter,
            action_adapter_config=action_adapter_config,
        )
    raise NotImplementedError(f"Unknown policy type: {args.policy_type}")
