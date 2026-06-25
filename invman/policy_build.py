"""Build a self-contained `Policy` from CLI args, deriving action bounds per problem.

OBJECTIVE
---------
Replaces the old Python-forward policy factories and per-problem context builders.
Given the resolved CLI `args` (after `apply_policy_name`), produce a single
`invman.policy.Policy` that carries the architecture AND the action bounds the Rust
rollout needs -- so "the bounds are part of the policy itself".

Per-problem bound derivation (the only problem-specific logic that must survive the
removal of the native Python envs):
  - lost_sales / lost_sales_fixed_order_cost: scalar action, input_dim = lead_time,
    cap = max_order_size; supports soft_tree / linear / nn (all have Rust rollouts).
  - dual_sourcing: input_dim = regular_lead_time; control bounds from the action
    adapter (`build_control_spec`); soft_tree only (only soft_tree has a Rust rollout).
  - multi_echelon: input_dim = warehouse_lead_time + num_retailers*retailer_lead_time;
    discrete-grid action over base-stock levels; soft_tree only.
"""

from __future__ import annotations

from invman.policy import Policy
from invman.policy_common import normalize_action_spec, normalize_state_normalizer
from invman.policy_registry import get_policy_spec
from invman.dual_sourcing_policy_spec import build_control_spec

_DEFAULT_WAREHOUSE_BSL = [50, 60, 70, 80, 90, 100]
_DEFAULT_RETAILER_BSL = [0, 5, 10, 15, 20, 25, 30, 35, 40]

_CATEGORICAL_OR_ORDINAL = {
    "categorical_quantity",
    "soft_gated_ordinal_quantity",
    "hard_gated_ordinal_quantity",
}
_TWO_LOGIT_HEADS = {
    "gated_sigmoid_direct_quantity",
    "soft_gated_direct_quantity",
    "hard_gated_direct_quantity",
}


def _policy_output_dim(decoder: str, action_spec: dict, control_dim: int) -> int:
    """Width of the dense net's output layer (mirrors the old Linear/PolicyNet logic)."""
    if decoder in _CATEGORICAL_OR_ORDINAL:
        return int(action_spec["max_values"][0]) + 1
    if decoder in _TWO_LOGIT_HEADS:
        return 2
    return int(control_dim)


def _lost_sales_state_norm(args):
    state_normalizer = normalize_state_normalizer(getattr(args, "state_normalizer", "quantity_scale"))
    state_scale = getattr(args, "state_scale", None)
    if state_normalizer != "identity" and state_scale is None:
        state_scale = float(getattr(args, "max_order_size", 1))
    return state_normalizer, (None if state_scale is None else float(state_scale))


def _build_lost_sales(args, spec) -> Policy:
    max_order_size = int(args.max_order_size)
    action_spec = normalize_action_spec(None, default_max_order_size=max_order_size)
    control_dim = int(action_spec["action_dim"])
    state_normalizer, state_scale = _lost_sales_state_norm(args)
    common = dict(
        backbone=spec.policy_backbone,
        input_dim=int(args.lead_time),
        control_dim=control_dim,
        control_mode=str(action_spec["action_mode"]),
        min_values=tuple(action_spec["min_values"]),
        max_values=tuple(action_spec["max_values"]),
        allowed_values=action_spec.get("allowed_values"),
        max_order_size=int(action_spec["max_values"][0]),
        action_adapter="identity",
        state_normalizer=state_normalizer,
        state_scale=state_scale,
    )
    if spec.policy_backbone == "soft_tree":
        return Policy(
            depth=int(spec.tree_depth),
            temperature=float(spec.tree_temperature),
            split_type=spec.tree_split_type,
            leaf_type=spec.tree_leaf_type,
            **common,
        )
    decoder = spec.policy_decoder
    out_dim = _policy_output_dim(decoder, action_spec, control_dim)
    if spec.policy_backbone == "linear":
        return Policy(output_dim=out_dim, action_output_mode=decoder, **common)
    if spec.policy_backbone == "nn":
        return Policy(
            output_dim=out_dim,
            action_output_mode=decoder,
            hidden_dim=tuple(spec.hidden_dim),
            activation_name=spec.activation,
            **common,
        )
    raise ValueError(f"Unknown policy backbone: {spec.policy_backbone}")


def _build_dual_sourcing(args, spec) -> Policy:
    if spec.policy_backbone != "soft_tree":
        raise NotImplementedError(
            "dual_sourcing supports only the soft_tree backbone via the Rust rollout"
        )
    reg_max = int(args.regular_max_order_size)
    exp_max = int(args.expedited_max_order_size)
    action_adapter = spec.action_adapter
    if action_adapter == "identity":
        spec_dict = {
            "action_dim": 2,
            "action_mode": "vector_quantity",
            "min_values": [0, 0],
            "max_values": [reg_max, exp_max],
            "allowed_values": None,
        }
    else:
        spec_dict = build_control_spec(
            action_adapter,
            regular_lead_time=int(args.regular_lead_time),
            demand_low=int(args.dual_demand_low),
            demand_high=int(args.dual_demand_high),
            regular_max_order_size=reg_max,
            expedited_max_order_size=exp_max,
        )
    control_spec = normalize_action_spec(spec_dict)
    return Policy(
        backbone="soft_tree",
        input_dim=int(args.regular_lead_time),
        control_dim=int(control_spec["action_dim"]),
        control_mode=str(control_spec["action_mode"]),
        min_values=tuple(control_spec["min_values"]),
        max_values=tuple(control_spec["max_values"]),
        allowed_values=control_spec.get("allowed_values"),
        max_order_size=max(reg_max, exp_max),
        action_adapter=action_adapter,
        depth=int(spec.tree_depth),
        temperature=float(spec.tree_temperature),
        split_type=spec.tree_split_type,
        leaf_type=spec.tree_leaf_type,
    )


def _build_multi_echelon(args, spec) -> Policy:
    if spec.policy_backbone != "soft_tree":
        raise NotImplementedError("multi_echelon supports only the soft_tree backbone")
    wbsl = list(getattr(args, "warehouse_base_stock_levels", _DEFAULT_WAREHOUSE_BSL))
    rbsl = list(getattr(args, "retailer_base_stock_levels", _DEFAULT_RETAILER_BSL))
    warehouse_cap = int(getattr(args, "warehouse_inventory_cap", max(wbsl)))
    retailer_cap = int(getattr(args, "retailer_inventory_cap", max(rbsl)))
    # Action parameterization is a policy-DESIGN choice (an autoresearch search dimension),
    # not a fixed grid -- the action space is whatever the chosen policy can express:
    #   "grid"         pick warehouse/retailer order-up-to levels from the discrete Gijs
    #                  reduced grid (the published reduced action set).
    #   "direct_level" directly estimate (continuous -> non-negative int) the warehouse and
    #                  retailer order-up-to LEVELS, bounded only by the physical inventory-
    #                  position caps (Cw, Cr); mirrors lost_sales' direct quantity estimation
    #                  and lets the policy reach the operating region (~330) without a
    #                  hand-set grid.
    design = str(getattr(args, "multi_action_design", "direct_level"))
    if design == "grid":
        action_spec = normalize_action_spec(
            {
                "action_dim": 2,
                "action_mode": "discrete_grid",
                "allowed_values": [[int(v) for v in wbsl], [int(v) for v in rbsl]],
            }
        )
        state_scale_default = max(max(wbsl), max(rbsl))
        max_order_size = int(max(wbsl))
    elif design == "direct_level":
        action_spec = normalize_action_spec(
            {
                "action_dim": 2,
                "action_mode": "vector_quantity",
                "min_values": [0, 0],
                "max_values": [warehouse_cap, retailer_cap],
            }
        )
        state_scale_default = retailer_cap
        max_order_size = warehouse_cap
    else:
        raise ValueError(
            f"unknown multi_action_design '{design}' (expected 'grid' or 'direct_level')"
        )
    # The learned-policy input dimension is owned by the problem: ask the Rust module to
    # report it (it runs the same feature builder the rollout uses), rather than re-deriving
    # it here with a formula that drifts from the env's decision-state layout (the
    # `lw + K*lr` formula was only correct for gijs_2022 with lw>=1, lr>=1 and broke on the
    # lw=0 van_roy_1997 simple problem).
    import invman_rust

    input_dim = int(
        invman_rust.multi_echelon_policy_feature_dim(
            num_retailers=int(args.num_retailers),
            warehouse_lead_time=int(args.warehouse_lead_time),
            retailer_lead_time=int(args.retailer_lead_time),
            inventory_dynamics_mode=str(getattr(args, "inventory_dynamics_mode", "gijs_2022")),
            policy_feature_mode="raw_decision_state",
            include_period_feature=bool(getattr(args, "include_period_feature", False)),
        )
    )
    return Policy(
        backbone="soft_tree",
        input_dim=input_dim,
        control_dim=int(action_spec["action_dim"]),
        control_mode=str(action_spec["action_mode"]),
        min_values=tuple(action_spec["min_values"]),
        max_values=tuple(action_spec["max_values"]),
        allowed_values=action_spec.get("allowed_values"),
        max_order_size=max_order_size,
        action_adapter="identity",
        # lost-sales-style policy interface: the env emits the pure decision state and the
        # policy normalizes it before acting. The multi-echelon policy owns its own
        # divide-by-scale normalization (rather than the lost-sales --state_normalizer arg),
        # scaling by the largest base-stock / order-up-to level across both echelons -- the
        # action magnitude that bounds the inventory positions the policy steers to, the
        # multi-echelon analogue of lost_sales' state_scale = max_order_size.
        state_normalizer="divide_by_scale",
        state_scale=float(getattr(args, "state_scale", None) or state_scale_default),
        depth=int(spec.tree_depth),
        temperature=float(spec.tree_temperature),
        split_type=spec.tree_split_type,
        leaf_type=spec.tree_leaf_type,
    )


def build_policy(args) -> Policy:
    spec = get_policy_spec(args)
    problem = getattr(args, "problem", "lost_sales")
    if problem in ("lost_sales", "lost_sales_fixed_order_cost"):
        return _build_lost_sales(args, spec)
    if problem == "dual_sourcing":
        return _build_dual_sourcing(args, spec)
    if problem == "multi_echelon":
        return _build_multi_echelon(args, spec)
    raise ValueError(f"Unknown problem '{problem}'")
