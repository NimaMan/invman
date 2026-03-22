from invman.policies.linear import LinearPolicyNet
from invman.policies.neural import PolicyNet
from invman.policies.soft_tree import SoftTreePolicy


def build_policy(args, env):
    if args.policy_type == "linear":
        return LinearPolicyNet(
            input_dim=env.state_space_dim,
            output_dim=env.action_space_dim,
            action_output_mode=args.policy_head,
            max_order_size=env.max_order_size,
        )
    if args.policy_type == "nn":
        return PolicyNet(
            input_dim=env.state_space_dim,
            hidden_dim=args.hidden_dim,
            output_dim=env.action_space_dim,
            activation=args.activation,
            action_output_mode=args.policy_head,
            max_order_size=env.max_order_size,
        )
    if args.policy_type == "soft_tree":
        return SoftTreePolicy(
            input_dim=env.state_space_dim,
            max_order_size=env.max_order_size,
            depth=args.tree_depth,
            temperature=args.tree_temperature,
        )
    raise NotImplementedError(f"Unknown policy type: {args.policy_type}")
