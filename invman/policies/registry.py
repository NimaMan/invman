from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from invman.policies.common import (
    normalize_policy_head,
    normalize_tree_leaf_type,
    normalize_tree_split_type,
)


_DENSE_DECODERS = (
    "categorical_quantity",
    "direct_quantity",
    "sigmoid_direct_quantity",
    "soft_gated_direct_quantity",
    "gated_sigmoid_direct_quantity",
    "hard_gated_direct_quantity",
    "bounded_quantity",
    "soft_gated_ordinal_quantity",
    "hard_gated_ordinal_quantity",
)

_NN_ACTIVATIONS = ("selu", "gelu", "relu")
_ACTION_ADAPTER_ALIASES = {
    "identity": "identity",
    "direct": "identity",
    "direct_orders": "identity",
    "dual_sourcing_single_index_targets": "dual_sourcing_single_index_targets",
    "single_index_targets": "dual_sourcing_single_index_targets",
    "dual_sourcing_dual_index_targets": "dual_sourcing_dual_index_targets",
    "dual_index_targets": "dual_sourcing_dual_index_targets",
    "dual_sourcing_capped_dual_index_targets": "dual_sourcing_capped_dual_index_targets",
    "capped_dual_index_targets": "dual_sourcing_capped_dual_index_targets",
    "dual_sourcing_base_surge_targets": "dual_sourcing_base_surge_targets",
    "base_surge_targets": "dual_sourcing_base_surge_targets",
}


def _format_float_token(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace(".", "p")


def _parse_float_token(token: str) -> float:
    return float(token.replace("p", "."))


def _short_action_adapter(action_adapter: str) -> str:
    normalized = _normalized_action_adapter(action_adapter)
    prefix = "dual_sourcing_"
    if normalized.startswith(prefix):
        return normalized[len(prefix):]
    return normalized


def _normalized_action_adapter(action_adapter: str) -> str:
    normalized = _ACTION_ADAPTER_ALIASES.get(str(action_adapter))
    if normalized is None:
        valid = ", ".join(sorted(_ACTION_ADAPTER_ALIASES))
        raise ValueError(f"Unknown action adapter '{action_adapter}'. Expected one of: {valid}")
    return normalized


@dataclass(frozen=True)
class PolicySpec:
    policy_name: str
    policy_backbone: str
    policy_decoder: str
    action_adapter: str = "identity"
    hidden_dim: tuple[int, ...] = ()
    activation: str | None = None
    tree_depth: int | None = None
    tree_temperature: float | None = None
    tree_split_type: str | None = None
    tree_leaf_type: str | None = None
    max_order_size: int | None = None

    @property
    def action_output_mode(self) -> str:
        if self.policy_backbone == "soft_tree":
            return f"tree_{self.tree_leaf_type}_leaf_quantity"
        return self.policy_decoder

    @property
    def architecture_suffix(self) -> str:
        if self.action_adapter == "identity":
            return ""
        return f"_{self.action_adapter}"

    def architecture_label(self, state_features: str) -> str:
        if self.policy_backbone == "soft_tree":
            return (
                f"{self.policy_backbone}_{self.tree_split_type}_{self.action_output_mode}"
                f"{self.architecture_suffix}_{state_features}"
            )
        return f"{self.policy_backbone}_{self.action_output_mode}{self.architecture_suffix}_{state_features}"


def make_dense_policy_name(
    policy_backbone: str,
    policy_decoder: str,
    *,
    hidden_dim: Sequence[int] | None = None,
    activation: str | None = None,
    action_adapter: str = "identity",
    max_order_size: int | None = None,
) -> str:
    backbone = str(policy_backbone)
    if backbone not in {"linear", "nn"}:
        raise ValueError(f"dense policies require backbone 'linear' or 'nn', got '{policy_backbone}'")
    decoder = normalize_policy_head(policy_decoder)
    parts = [backbone, decoder]
    if backbone == "nn":
        if not hidden_dim:
            raise ValueError("nn policy names require hidden_dim")
        activation_name = str(activation or "selu")
        if activation_name not in _NN_ACTIVATIONS:
            valid = ", ".join(_NN_ACTIVATIONS)
            raise ValueError(f"Unknown activation '{activation_name}'. Expected one of: {valid}")
        parts.extend([f"h{'x'.join(str(int(width)) for width in hidden_dim)}", activation_name])
    adapter = _short_action_adapter(action_adapter)
    if adapter != "identity":
        parts.append(f"adapter-{adapter}")
    if max_order_size is not None:
        parts.append(f"q{int(max_order_size)}")
    return "_".join(parts)


def make_soft_tree_policy_name(
    *,
    depth: int,
    temperature: float = 0.25,
    split_type: str = "oblique",
    leaf_type: str = "constant",
    action_adapter: str = "identity",
    max_order_size: int | None = None,
) -> str:
    parts = [
        "soft_tree",
        f"d{int(depth)}",
        f"t{_format_float_token(temperature)}",
        normalize_tree_split_type(split_type),
        f"{normalize_tree_leaf_type(leaf_type)}_leaf",
    ]
    adapter = _short_action_adapter(action_adapter)
    if adapter != "identity":
        parts.append(f"adapter-{adapter}")
    if max_order_size is not None:
        parts.append(f"q{int(max_order_size)}")
    return "_".join(parts)


def _dense_spec(
    policy_name: str,
    *,
    policy_backbone: str,
    policy_decoder: str,
    hidden_dim: Sequence[int] | None = None,
    activation: str | None = None,
    action_adapter: str = "identity",
    max_order_size: int | None = None,
) -> PolicySpec:
    return PolicySpec(
        policy_name=policy_name,
        policy_backbone=policy_backbone,
        policy_decoder=normalize_policy_head(policy_decoder),
        hidden_dim=tuple(int(width) for width in (hidden_dim or ())),
        activation=activation,
        action_adapter=_normalized_action_adapter(action_adapter),
        max_order_size=None if max_order_size is None else int(max_order_size),
    )


def _soft_tree_spec(
    policy_name: str,
    *,
    depth: int,
    temperature: float = 0.25,
    split_type: str = "oblique",
    leaf_type: str = "constant",
    action_adapter: str = "identity",
    max_order_size: int | None = None,
) -> PolicySpec:
    return PolicySpec(
        policy_name=policy_name,
        policy_backbone="soft_tree",
        policy_decoder=f"tree_{normalize_tree_leaf_type(leaf_type)}_leaf_quantity",
        tree_depth=int(depth),
        tree_temperature=float(temperature),
        tree_split_type=normalize_tree_split_type(split_type),
        tree_leaf_type=normalize_tree_leaf_type(leaf_type),
        action_adapter=_normalized_action_adapter(action_adapter),
        max_order_size=None if max_order_size is None else int(max_order_size),
    )


_POLICY_ALIASES = {
    "linear_categorical_quantity": _dense_spec(
        "linear_categorical_quantity",
        policy_backbone="linear",
        policy_decoder="categorical_quantity",
    ),
    "linear_soft_gated_ordinal_quantity": _dense_spec(
        "linear_soft_gated_ordinal_quantity",
        policy_backbone="linear",
        policy_decoder="soft_gated_ordinal_quantity",
    ),
    "linear_gated_ordinal_quantity": _dense_spec(
        "linear_soft_gated_ordinal_quantity",
        policy_backbone="linear",
        policy_decoder="soft_gated_ordinal_quantity",
    ),
    "linear_hard_gated_ordinal_quantity": _dense_spec(
        "linear_hard_gated_ordinal_quantity",
        policy_backbone="linear",
        policy_decoder="hard_gated_ordinal_quantity",
    ),
    "linear_positive_quantity": _dense_spec(
        "linear_direct_quantity",
        policy_backbone="linear",
        policy_decoder="direct_quantity",
    ),
    "linear_direct_quantity": _dense_spec(
        "linear_direct_quantity",
        policy_backbone="linear",
        policy_decoder="direct_quantity",
    ),
    "linear_soft_gated_direct_quantity": _dense_spec(
        "linear_soft_gated_direct_quantity",
        policy_backbone="linear",
        policy_decoder="soft_gated_direct_quantity",
    ),
    "linear_gated_direct_quantity": _dense_spec(
        "linear_soft_gated_direct_quantity",
        policy_backbone="linear",
        policy_decoder="soft_gated_direct_quantity",
    ),
    "linear_gated_positive_quantity": _dense_spec(
        "linear_soft_gated_direct_quantity",
        policy_backbone="linear",
        policy_decoder="soft_gated_direct_quantity",
    ),
    "linear_sigmoid_direct_quantity": _dense_spec(
        "linear_sigmoid_direct_quantity",
        policy_backbone="linear",
        policy_decoder="sigmoid_direct_quantity",
    ),
    "linear_scaled_direct_quantity": _dense_spec(
        "linear_sigmoid_direct_quantity",
        policy_backbone="linear",
        policy_decoder="sigmoid_direct_quantity",
    ),
    "linear_gated_sigmoid_direct_quantity": _dense_spec(
        "linear_gated_sigmoid_direct_quantity",
        policy_backbone="linear",
        policy_decoder="gated_sigmoid_direct_quantity",
    ),
    "linear_scaled_gated_direct_quantity": _dense_spec(
        "linear_gated_sigmoid_direct_quantity",
        policy_backbone="linear",
        policy_decoder="gated_sigmoid_direct_quantity",
    ),
    "linear_hard_gated_direct_quantity": _dense_spec(
        "linear_hard_gated_direct_quantity",
        policy_backbone="linear",
        policy_decoder="hard_gated_direct_quantity",
    ),
    "linear_two_stage_direct_quantity": _dense_spec(
        "linear_hard_gated_direct_quantity",
        policy_backbone="linear",
        policy_decoder="hard_gated_direct_quantity",
    ),
    "linear_two_stage_positive_quantity": _dense_spec(
        "linear_hard_gated_direct_quantity",
        policy_backbone="linear",
        policy_decoder="hard_gated_direct_quantity",
    ),
    "nn_categorical_quantity": _dense_spec(
        "nn_categorical_quantity",
        policy_backbone="nn",
        policy_decoder="categorical_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_soft_gated_ordinal_quantity": _dense_spec(
        "nn_soft_gated_ordinal_quantity",
        policy_backbone="nn",
        policy_decoder="soft_gated_ordinal_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_gated_ordinal_quantity": _dense_spec(
        "nn_soft_gated_ordinal_quantity",
        policy_backbone="nn",
        policy_decoder="soft_gated_ordinal_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_hard_gated_ordinal_quantity": _dense_spec(
        "nn_hard_gated_ordinal_quantity",
        policy_backbone="nn",
        policy_decoder="hard_gated_ordinal_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_positive_quantity": _dense_spec(
        "nn_direct_quantity",
        policy_backbone="nn",
        policy_decoder="direct_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_direct_quantity": _dense_spec(
        "nn_direct_quantity",
        policy_backbone="nn",
        policy_decoder="direct_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_soft_gated_direct_quantity": _dense_spec(
        "nn_soft_gated_direct_quantity",
        policy_backbone="nn",
        policy_decoder="soft_gated_direct_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_gated_direct_quantity": _dense_spec(
        "nn_soft_gated_direct_quantity",
        policy_backbone="nn",
        policy_decoder="soft_gated_direct_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_gated_positive_quantity": _dense_spec(
        "nn_soft_gated_direct_quantity",
        policy_backbone="nn",
        policy_decoder="soft_gated_direct_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_sigmoid_direct_quantity": _dense_spec(
        "nn_sigmoid_direct_quantity",
        policy_backbone="nn",
        policy_decoder="sigmoid_direct_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_scaled_direct_quantity": _dense_spec(
        "nn_sigmoid_direct_quantity",
        policy_backbone="nn",
        policy_decoder="sigmoid_direct_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_gated_sigmoid_direct_quantity": _dense_spec(
        "nn_gated_sigmoid_direct_quantity",
        policy_backbone="nn",
        policy_decoder="gated_sigmoid_direct_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_scaled_gated_direct_quantity": _dense_spec(
        "nn_gated_sigmoid_direct_quantity",
        policy_backbone="nn",
        policy_decoder="gated_sigmoid_direct_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_hard_gated_direct_quantity": _dense_spec(
        "nn_hard_gated_direct_quantity",
        policy_backbone="nn",
        policy_decoder="hard_gated_direct_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_two_stage_direct_quantity": _dense_spec(
        "nn_hard_gated_direct_quantity",
        policy_backbone="nn",
        policy_decoder="hard_gated_direct_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "nn_two_stage_positive_quantity": _dense_spec(
        "nn_hard_gated_direct_quantity",
        policy_backbone="nn",
        policy_decoder="hard_gated_direct_quantity",
        hidden_dim=(50,),
        activation="selu",
    ),
    "soft_tree_depth2_linear_leaf": _soft_tree_spec(
        "soft_tree_depth2_linear_leaf",
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
    ),
    "soft_tree_depth2_sigmoid_linear_leaf": _soft_tree_spec(
        "soft_tree_depth2_sigmoid_linear_leaf",
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="sigmoid_linear",
    ),
    "soft_tree_depth2_scaled_linear_leaf": _soft_tree_spec(
        "soft_tree_depth2_sigmoid_linear_leaf",
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="sigmoid_linear",
    ),
    "soft_tree_depth2_positive_linear_leaf": _soft_tree_spec(
        "soft_tree_depth2_linear_leaf",
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
    ),
    "soft_tree_depth1_linear_leaf": _soft_tree_spec(
        "soft_tree_depth1_linear_leaf",
        depth=1,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
    ),
    "soft_tree_depth1_sigmoid_linear_leaf": _soft_tree_spec(
        "soft_tree_depth1_sigmoid_linear_leaf",
        depth=1,
        temperature=0.25,
        split_type="oblique",
        leaf_type="sigmoid_linear",
    ),
    "soft_tree_depth1_scaled_linear_leaf": _soft_tree_spec(
        "soft_tree_depth1_sigmoid_linear_leaf",
        depth=1,
        temperature=0.25,
        split_type="oblique",
        leaf_type="sigmoid_linear",
    ),
    "soft_tree_depth1_positive_linear_leaf": _soft_tree_spec(
        "soft_tree_depth1_linear_leaf",
        depth=1,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
    ),
    "linear_categorical_quantity_q8": _dense_spec(
        "linear_categorical_quantity_q8",
        policy_backbone="linear",
        policy_decoder="categorical_quantity",
        max_order_size=8,
    ),
    "linear_categorical_quantity_q20": _dense_spec(
        "linear_categorical_quantity_q20",
        policy_backbone="linear",
        policy_decoder="categorical_quantity",
        max_order_size=20,
    ),
    "nn_categorical_quantity_q8": _dense_spec(
        "nn_categorical_quantity_q8",
        policy_backbone="nn",
        policy_decoder="categorical_quantity",
        hidden_dim=(50,),
        activation="selu",
        max_order_size=8,
    ),
    "nn_categorical_quantity_q20": _dense_spec(
        "nn_categorical_quantity_q20",
        policy_backbone="nn",
        policy_decoder="categorical_quantity",
        hidden_dim=(50,),
        activation="selu",
        max_order_size=20,
    ),
    "soft_tree_depth2_linear_leaf_q8": _soft_tree_spec(
        "soft_tree_depth2_linear_leaf_q8",
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
        max_order_size=8,
    ),
    "linear_bounded_quantity_identity": _dense_spec(
        "linear_bounded_quantity_identity",
        policy_backbone="linear",
        policy_decoder="bounded_quantity",
        action_adapter="identity",
    ),
    "nn_bounded_quantity_identity": _dense_spec(
        "nn_bounded_quantity_identity",
        policy_backbone="nn",
        policy_decoder="bounded_quantity",
        hidden_dim=(16, 16),
        activation="selu",
        action_adapter="identity",
    ),
    "linear_base_surge_targets": _dense_spec(
        "linear_base_surge_targets",
        policy_backbone="linear",
        policy_decoder="bounded_quantity",
        action_adapter="base_surge_targets",
    ),
    "nn_base_surge_targets": _dense_spec(
        "nn_base_surge_targets",
        policy_backbone="nn",
        policy_decoder="bounded_quantity",
        hidden_dim=(16, 16),
        activation="selu",
        action_adapter="base_surge_targets",
    ),
    "soft_tree_identity": _soft_tree_spec(
        "soft_tree_identity",
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
        action_adapter="identity",
    ),
    "soft_tree_base_surge_targets": _soft_tree_spec(
        "soft_tree_base_surge_targets",
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
        action_adapter="base_surge_targets",
    ),
    "soft_tree_constant_leaf": _soft_tree_spec(
        "soft_tree_constant_leaf",
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="constant",
    ),
    "soft_tree_linear_leaf": _soft_tree_spec(
        "soft_tree_linear_leaf",
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="linear",
    ),
}


_LINEAR_RE = re.compile(
    r"^(?P<decoder>categorical_quantity|direct_quantity|positive_quantity|sigmoid_direct_quantity|scaled_direct_quantity|soft_gated_direct_quantity|gated_direct_quantity|gated_positive_quantity|gated_sigmoid_direct_quantity|scaled_gated_direct_quantity|hard_gated_direct_quantity|two_stage_direct_quantity|two_stage_positive_quantity|bounded_quantity|soft_gated_ordinal_quantity|gated_ordinal_quantity|hard_gated_ordinal_quantity|two_stage_ordinal_quantity)"
    r"(?:_adapter-(?P<adapter>.+?))?(?:_q(?P<q>\d+))?$"
)
_NN_RE = re.compile(
    r"^(?P<decoder>categorical_quantity|direct_quantity|positive_quantity|sigmoid_direct_quantity|scaled_direct_quantity|soft_gated_direct_quantity|gated_direct_quantity|gated_positive_quantity|gated_sigmoid_direct_quantity|scaled_gated_direct_quantity|hard_gated_direct_quantity|two_stage_direct_quantity|two_stage_positive_quantity|bounded_quantity|soft_gated_ordinal_quantity|gated_ordinal_quantity|hard_gated_ordinal_quantity|two_stage_ordinal_quantity)"
    r"_h(?P<hidden>\d+(?:x\d+)*)_(?P<activation>selu|gelu|relu)"
    r"(?:_adapter-(?P<adapter>.+?))?(?:_q(?P<q>\d+))?$"
)
_SOFT_TREE_RE = re.compile(
    r"^d(?P<depth>\d+)_t(?P<temperature>[0-9p]+)_(?P<split>oblique|axis_aligned)_(?P<leaf>constant|linear|positive_linear|sigmoid_linear|scaled_linear)_leaf"
    r"(?:_adapter-(?P<adapter>.+?))?(?:_q(?P<q>\d+))?$"
)


def resolve_policy_name(policy_name: str) -> PolicySpec:
    name = str(policy_name)
    if name in _POLICY_ALIASES:
        return _POLICY_ALIASES[name]

    if name.startswith("linear_"):
        match = _LINEAR_RE.fullmatch(name[len("linear_"):])
        if match is None:
            raise ValueError(f"Unknown linear policy name '{name}'")
        return _dense_spec(
            name,
            policy_backbone="linear",
            policy_decoder=match.group("decoder"),
            action_adapter=match.group("adapter") or "identity",
            max_order_size=int(match.group("q")) if match.group("q") else None,
        )

    if name.startswith("nn_"):
        match = _NN_RE.fullmatch(name[len("nn_"):])
        if match is None:
            raise ValueError(f"Unknown nn policy name '{name}'")
        return _dense_spec(
            name,
            policy_backbone="nn",
            policy_decoder=match.group("decoder"),
            hidden_dim=tuple(int(width) for width in match.group("hidden").split("x")),
            activation=match.group("activation"),
            action_adapter=match.group("adapter") or "identity",
            max_order_size=int(match.group("q")) if match.group("q") else None,
        )

    if name.startswith("soft_tree_"):
        match = _SOFT_TREE_RE.fullmatch(name[len("soft_tree_"):])
        if match is None:
            raise ValueError(f"Unknown soft-tree policy name '{name}'")
        return _soft_tree_spec(
            name,
            depth=int(match.group("depth")),
            temperature=_parse_float_token(match.group("temperature")),
            split_type=match.group("split"),
            leaf_type=match.group("leaf"),
            action_adapter=match.group("adapter") or "identity",
            max_order_size=int(match.group("q")) if match.group("q") else None,
        )

    known = ", ".join(sorted(_POLICY_ALIASES))
    raise ValueError(f"Unknown policy name '{name}'. Known named policies: {known}")


def apply_policy_name(args, policy_name: str | None = None) -> PolicySpec:
    target_name = policy_name or getattr(args, "policy_name", None)
    if target_name is None:
        raise ValueError("policy_name must be set before resolving a learned policy")
    resolved = resolve_policy_name(target_name)
    args.policy_name = resolved.policy_name
    args.policy_backbone = resolved.policy_backbone
    args.policy_decoder = resolved.policy_decoder
    args.action_adapter = resolved.action_adapter
    if resolved.policy_backbone == "nn":
        args.hidden_dim = list(resolved.hidden_dim)
        args.activation = resolved.activation
    else:
        args.hidden_dim = None
        args.activation = None
    if resolved.policy_backbone == "soft_tree":
        args.tree_depth = resolved.tree_depth
        args.tree_temperature = resolved.tree_temperature
        args.tree_split_type = resolved.tree_split_type
        args.tree_leaf_type = resolved.tree_leaf_type
    else:
        args.tree_depth = None
        args.tree_temperature = None
        args.tree_split_type = None
        args.tree_leaf_type = None
    if resolved.max_order_size is not None:
        args.max_order_size = resolved.max_order_size
    args._resolved_policy_spec = resolved
    return resolved


def get_policy_spec(args) -> PolicySpec:
    cached = getattr(args, "_resolved_policy_spec", None)
    if cached is not None and cached.policy_name == getattr(args, "policy_name", None):
        return cached
    return apply_policy_name(args)
