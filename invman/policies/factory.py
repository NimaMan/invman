from invman.policies.linear import LinearPolicyNet
from invman.policies.neural import PolicyNet
from invman.policies.soft_tree import SoftTreePolicy


def build_policy(args, env):
    action_spec = getattr(env, "action_spec", None)
    if action_spec is None:
        action_spec = {
            "action_dim": 1,
            "action_mode": "scalar_quantity",
            "min_values": [0],
            "max_values": [int(env.max_order_size)],
            "allowed_values": None,
        }

    if args.policy_type == "linear":
        if int(action_spec["action_dim"]) != 1 or action_spec["action_mode"] != "scalar_quantity":
            raise ValueError("Linear policies currently support only scalar_quantity action specs.")
        return LinearPolicyNet(
            input_dim=env.state_space_dim,
            output_dim=env.action_space_dim,
            action_output_mode=args.policy_head,
            max_order_size=env.max_order_size,
        )
    if args.policy_type == "nn":
        if int(action_spec["action_dim"]) != 1 or action_spec["action_mode"] != "scalar_quantity":
            raise ValueError("Neural policies currently support only scalar_quantity action specs.")
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
            max_order_size=getattr(env, "max_order_size", None),
            action_spec=action_spec,
            depth=args.tree_depth,
            temperature=args.tree_temperature,
            split_type=args.tree_split_type,
            leaf_type=args.tree_leaf_type,
        )
    raise NotImplementedError(f"Unknown policy type: {args.policy_type}")
