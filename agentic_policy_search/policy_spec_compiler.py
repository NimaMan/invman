# =============================================================================
# policy_spec_compiler.py -- DSL(JSON) -> invman OWMR Policy + gate-invertible warm start
# =============================================================================
# OBJECTIVE
#   Turn the README policy-spec DSL (the JSON Codex emits, one tool call per
#   proposal) into a concrete, *evaluable* invman `Policy` for the
#   one_warehouse_multi_retailer (OWMR, Kaynov 2024) problem, PLUS the
#   gate-invertible warm-start parameter vector so the inner CMA-ES generation-0
#   reproduces the in-repo echelon-base-stock gate EXACTLY. This is the single
#   "compile" step of the oracle pipeline described in the crate README:
#       spec(JSON, our DSL)  ->  invman.Policy  (+ warm-start anchor)
#
# WHY THIS EXISTS (relation to the existing invman machinery)
#   invman.policy_build.build_policy only wires lost_sales / dual_sourcing /
#   multi_echelon -- it has NO OWMR branch. The OWMR Policy is instead built by
#   scripts/one_warehouse_multi_retailer/common.build_soft_tree_model, and the
#   gate-invertible warm start is the per-dimension leaf inversion implemented in
#   run_asymmetric_learned_vs_gate._warm_start_flat_params. This compiler is a thin,
#   validating adapter: it maps the DSL enums onto those EXISTING, verified builders
#   rather than re-deriving the OWMR action geometry / leaf inversion (which would
#   drift from the Rust rollout's action decoder). We reuse, we do not reinvent.
#
# FULL ALGORITHMIC DESCRIPTION
#   compile_policy_spec(spec, reference) :
#     1. VALIDATE the DSL (no silent coercion; raise PolicySpecError on any bad
#        field -- unknown enum, wrong problem, wrong instance, missing key):
#          problem               must be "one_warehouse_multi_retailer"
#          backbone              {soft_tree, linear}   (linear == depth-0 tree here)
#          split_type            {oblique, axis_aligned}
#          leaf_type             {constant, linear}
#          action_head           {echelon_targets, symmetric_echelon_targets,
#                                 echelon_targets_with_alloc_targets, direct_orders}
#          warm_start            {gate_invertible, none}
#          depth                 int >= 1 (soft_tree); ignored for linear (-> depth 1)
#          temperature           float > 0
#          per_retailer_targets  bool   (consistency-checked vs action_head)
#          features              subset of the supported OWMR feature bases
#     2. RESOLVE the action geometry. The DSL action_head maps directly to the
#        binding's policy_action_mode. per_retailer_targets must agree with the head:
#          - symmetric_echelon_targets => one shared retailer target (control_dim=2);
#            requires per_retailer_targets == False AND a symmetric reference.
#          - echelon_targets / *_with_alloc_targets / direct_orders are per-retailer
#            (control_dim = K+1 or 1+2K) => require per_retailer_targets == True.
#        The README's `features` field selects the OWMR policy_state_mode:
#          "normalized" decision state              <=> features == default basis
#          adding raw absolute positions ("absolute")<=> policy_state_mode
#                                                       "absolute_augmented".
#     3. BUILD the Policy via common.build_soft_tree_model (the verified OWMR builder),
#        passing depth / temperature / split_type / leaf_type / action_head / state mode.
#        backbone=="linear" is realized as a depth-1 axis-aligned soft tree with a
#        linear leaf (a single state-dependent affine map -- the linear backbone of
#        the DSL), so it goes through the same Rust rollout.
#     4. WARM START. If warm_start == "gate_invertible" and the head is a target head
#        (not direct_orders), invert the per-dimension leaf transform so EVERY leaf
#        emits the gate's target vector state-independently (constant leaf -> logit;
#        linear leaf -> zero weights + softplus_inv bias). This is the exact inversion
#        in run_asymmetric_learned_vs_gate._warm_start_flat_params, reused verbatim by
#        import so the anchor is bit-for-bit the gate the held-out eval scores. The
#        gate target vector is [W, r_1..r_K] for echelon_targets, [W, R_shared] for
#        symmetric, and [W, r_1..r_K, r_1..r_K] for the decoupled-alloc head.
#        direct_orders emits raw orders (not a target position) so it has no
#        gate-reproducing anchor -- warm_started is reported False and CMA-ES starts
#        from the model's default (the honest floor then deploys the gate if the raw
#        policy loses, so the spec is still downside-safe).
#
#   The compiler returns a CompiledPolicy carrying: the Policy, the resolved
#   policy_action_mode / policy_state_mode / train+eval allocation set, the warm-start
#   flat params (or None), and the warm_started flag. evaluate_policy_spec.py consumes
#   exactly these.
#
# NO SILENT FALLBACKS
#   Every validation failure raises PolicySpecError with the raw offending value and
#   the allowed set. The caller (the CLI) catches it and returns compiled_ok=false +
#   the error string (README contract), never a coerced "closest" spec.
# =============================================================================

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# The OWMR builders + the gate-invertible warm start live in the problem's scripts
# dir, not in the importable invman package. Make that dir importable so we reuse the
# verified machinery instead of duplicating the action geometry / leaf inversion.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_OWMR_SCRIPT_DIR = _PACKAGE_ROOT / "scripts" / "one_warehouse_multi_retailer"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))
if str(_OWMR_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OWMR_SCRIPT_DIR))


# --------------------------------------------------------------------------- #
# Errors + allowed enum sets (the contract Codex must satisfy)                 #
# --------------------------------------------------------------------------- #
class PolicySpecError(ValueError):
    """Raised on any invalid DSL field. Carries a human-readable, raw-value message
    so the oracle CLI can return it verbatim under compiled_ok=false."""


PROBLEM = "one_warehouse_multi_retailer"
BACKBONES = ("soft_tree", "linear")
SPLIT_TYPES = ("oblique", "axis_aligned")
LEAF_TYPES = ("constant", "linear")
ACTION_HEADS = (
    "echelon_targets",
    "symmetric_echelon_targets",
    "echelon_targets_with_alloc_targets",
    "direct_orders",
)
WARM_STARTS = ("gate_invertible", "none")

# OWMR decision-state feature bases the Rust rollout understands. The "normalized"
# basis is the default pipeline-aware decision state (warehouse inventory + pipeline,
# per-retailer inventory + pipeline, scale, total echelon position). Naming any of the
# raw-absolute features selects the "absolute_augmented" rollout state mode.
_NORMALIZED_FEATURES = {"on_hand", "backlog", "pipeline", "scale", "total_position"}
_ABSOLUTE_FEATURES = {"absolute", "absolute_position", "raw_position", "absolute_augmented"}
SUPPORTED_FEATURES = _NORMALIZED_FEATURES | _ABSOLUTE_FEATURES

# Per-retailer (asymmetric-capable) heads grow the control dim to K+1 or 1+2K.
_PER_RETAILER_HEADS = (
    "echelon_targets",
    "echelon_targets_with_alloc_targets",
    "direct_orders",
)
# Target heads support both proportional and min_shortage allocation (they emit a
# target position min_shortage can ration against). direct_orders emits raw orders,
# so only proportional is supported.
_TARGET_HEADS = (
    "echelon_targets",
    "symmetric_echelon_targets",
    "echelon_targets_with_alloc_targets",
)


@dataclass
class CompiledPolicy:
    """Everything evaluate_policy_spec.py needs to train + score one spec."""

    model: Any  # invman.policy.Policy
    policy_action_mode: str
    policy_state_mode: str
    eval_allocations: tuple[str, ...]
    warm_flat: list[float] | None  # gate-invertible anchor (None if not applicable)
    warm_started: bool
    depth: int
    temperature: float
    split_type: str
    leaf_type: str
    backbone: str
    per_retailer_targets: bool
    features: tuple[str, ...]
    instance_name: str


# --------------------------------------------------------------------------- #
# Validation helpers (explicit failure, no coercion)                          #
# --------------------------------------------------------------------------- #
def _require(spec: dict, key: str) -> Any:
    if key not in spec:
        raise PolicySpecError(f"missing required spec field '{key}'")
    return spec[key]


def _check_enum(value: Any, allowed: tuple[str, ...], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise PolicySpecError(
            f"invalid '{field}' = {value!r}; expected one of: {', '.join(allowed)}"
        )
    return value


def _check_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise PolicySpecError(f"invalid '{field}' = {value!r}; expected a boolean")
    return value


def _resolve_state_mode(features: Any) -> tuple[str, tuple[str, ...]]:
    """Map the DSL feature basis onto the OWMR rollout policy_state_mode.

    Any recognized raw-absolute feature switches the rollout to absolute_augmented
    (which appends scale + raw total echelon position + raw retailer positions to the
    normalized decision state); otherwise the default normalized decision state is
    used. Unknown features fail explicitly."""
    if features is None:
        return "normalized", tuple()
    if not isinstance(features, (list, tuple)):
        raise PolicySpecError(f"invalid 'features' = {features!r}; expected a list of strings")
    resolved: list[str] = []
    for feat in features:
        if not isinstance(feat, str) or feat not in SUPPORTED_FEATURES:
            raise PolicySpecError(
                f"invalid feature {feat!r}; expected a subset of: "
                f"{', '.join(sorted(SUPPORTED_FEATURES))}"
            )
        resolved.append(feat)
    state_mode = (
        "absolute_augmented"
        if any(feat in _ABSOLUTE_FEATURES for feat in resolved)
        else "normalized"
    )
    return state_mode, tuple(resolved)


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def compile_policy_spec(spec: dict, reference: dict) -> CompiledPolicy:
    """Compile one DSL spec into a CompiledPolicy for `reference` (the resolved OWMR
    Kaynov instance dict from common.get_reference). Raises PolicySpecError on any
    invalid field. The gate target vector for the warm start is provided later by the
    caller via attach_gate_warm_start (the gate must be searched first)."""
    import common  # OWMR builders (scripts/one_warehouse_multi_retailer/common.py)

    if not isinstance(spec, dict):
        raise PolicySpecError(f"spec must be a JSON object, got {type(spec).__name__}")

    problem = _require(spec, "problem")
    if problem != PROBLEM:
        raise PolicySpecError(
            f"this compiler only targets '{PROBLEM}', got problem={problem!r}"
        )

    backbone = _check_enum(_require(spec, "backbone"), BACKBONES, "backbone")
    split_type = _check_enum(_require(spec, "split_type"), SPLIT_TYPES, "split_type")
    leaf_type = _check_enum(_require(spec, "leaf_type"), LEAF_TYPES, "leaf_type")
    action_head = _check_enum(_require(spec, "action_head"), ACTION_HEADS, "action_head")
    warm_start = _check_enum(_require(spec, "warm_start"), WARM_STARTS, "warm_start")
    per_retailer = _check_bool(_require(spec, "per_retailer_targets"), "per_retailer_targets")

    # temperature: soft trees only, but always validate if present.
    raw_temp = spec.get("temperature", 0.25)
    try:
        temperature = float(raw_temp)
    except (TypeError, ValueError):
        raise PolicySpecError(f"invalid 'temperature' = {raw_temp!r}; expected a number")
    if not (temperature > 0.0):
        raise PolicySpecError(f"invalid 'temperature' = {temperature}; must be > 0")

    # depth: soft_tree honors it; linear backbone is realized as a depth-1 tree.
    if backbone == "linear":
        depth = 1
        # axis-aligned single split is the canonical "linear" surface in this rollout.
        split_type = "axis_aligned"
    else:
        raw_depth = _require(spec, "depth")
        if not isinstance(raw_depth, int) or isinstance(raw_depth, bool) or raw_depth < 1:
            raise PolicySpecError(
                f"invalid 'depth' = {raw_depth!r}; expected an integer >= 1 for soft_tree"
            )
        depth = int(raw_depth)

    state_mode, features = _resolve_state_mode(spec.get("features"))

    # --- action-head <-> per_retailer_targets consistency -------------------- #
    symmetric_reference = bool(common.is_symmetric_retailer_case(reference))
    if action_head == "symmetric_echelon_targets":
        if per_retailer:
            raise PolicySpecError(
                "symmetric_echelon_targets uses a single shared retailer target; "
                "per_retailer_targets must be false"
            )
        if not symmetric_reference:
            raise PolicySpecError(
                "symmetric_echelon_targets requires a symmetric retailer instance; "
                f"reference {reference.get('name')!r} is asymmetric -- use echelon_targets"
            )
    else:
        if not per_retailer:
            raise PolicySpecError(
                f"action_head '{action_head}' is per-retailer; per_retailer_targets must be true"
            )

    eval_allocations = (
        ("proportional",) if action_head == "direct_orders" else ("proportional", "min_shortage")
    )

    # --- build the verified OWMR soft tree ----------------------------------- #
    try:
        model = common.build_soft_tree_model(
            reference,
            depth=depth,
            temperature=temperature,
            split_type=split_type,
            leaf_type=leaf_type,
            policy_action_mode=action_head,
            policy_state_mode=state_mode,
        )
    except Exception as exc:  # surface the builder's raw error, do not mask it
        raise PolicySpecError(f"OWMR model build failed: {exc.__class__.__name__}: {exc}")

    return CompiledPolicy(
        model=model,
        policy_action_mode=action_head,
        policy_state_mode=state_mode,
        eval_allocations=eval_allocations,
        warm_flat=None,
        warm_started=False,
        depth=depth,
        temperature=temperature,
        split_type=split_type,
        leaf_type=leaf_type,
        backbone=backbone,
        per_retailer_targets=per_retailer,
        features=features,
        instance_name=str(reference.get("name")),
    )


def attach_gate_warm_start(
    compiled: CompiledPolicy,
    reference: dict,
    warm_start_mode: str,
    gate_warehouse_level: int,
    gate_retailer_levels: list[int],
) -> CompiledPolicy:
    """Build the gate-invertible warm-start anchor for `compiled` from the searched
    gate (W, [r_1..r_K]) and attach it. The warm start is meaningful only for target
    heads; direct_orders cannot reproduce a base-stock target and stays unwarmed.

    Reuses run_asymmetric_learned_vs_gate._warm_start_flat_params VERBATIM (the same
    per-dimension leaf inversion the production runner uses), so generation-0 of the
    inner CMA-ES reproduces the gate bit-for-bit -> the honest warm-start floor.
    Returns the same CompiledPolicy with warm_flat / warm_started populated."""
    if warm_start_mode == "none":
        compiled.warm_flat = None
        compiled.warm_started = False
        return compiled
    if compiled.policy_action_mode not in _TARGET_HEADS:
        # direct_orders has no gate-reproducing anchor; not an error -- just no warm start.
        compiled.warm_flat = None
        compiled.warm_started = False
        return compiled

    from run_asymmetric_learned_vs_gate import _warm_start_flat_params

    w_level = int(gate_warehouse_level)
    r_levels = [int(v) for v in gate_retailer_levels]
    head = compiled.policy_action_mode
    if head == "symmetric_echelon_targets":
        target_vector = [w_level, int(round(float(np.mean(r_levels))))]
    elif head == "echelon_targets_with_alloc_targets":
        target_vector = [w_level] + r_levels + r_levels
    else:  # echelon_targets
        target_vector = [w_level] + r_levels

    warm_flat, warm_started = _warm_start_flat_params(compiled.model, target_vector)
    compiled.warm_flat = warm_flat if warm_started else None
    compiled.warm_started = bool(warm_started)
    return compiled


def is_target_head(action_head: str) -> bool:
    return action_head in _TARGET_HEADS


def softplus_inv(delta: float) -> float:
    """Inverse softplus used by the linear-leaf inversion (exposed for tests)."""
    return math.log(math.expm1(max(float(delta), 1e-6)))
