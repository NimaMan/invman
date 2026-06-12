# =============================================================================
# policy_spec_compiler_padn.py -- PADN DSL(JSON) -> soft-tree config + residual-zero warm start
# =============================================================================
# OBJECTIVE
#   Turn the policy_search/agentic policy-spec DSL into a concrete, *evaluable*
#   soft-tree configuration for the PADN problem
#   ("production_assembly_distribution_network", the mixed distribution+assembly SCN of
#   Pirhooshyaran & Snyder 2021, Fig. 1 / Table 5), PLUS the gate-residual warm start
#   so the inner CMA-ES generation-0 reproduces the in-repo pairwise base-stock GATE
#   byte-exact. This is the PADN twin of policy_spec_compiler.py (OWMR):
#       spec(JSON, the DSL)  ->  PADN soft-tree config (+ warm-start anchor)
#
# WHY THIS EXISTS (relation to the existing PADN machinery)
#   The verified PADN action geometry, topology, gate search and rollout bindings live
#   in scripts/production_assembly_distribution_network/
#   autoresearch_mixed_distribution_assembly_network.py (imported here as `base`). This
#   compiler is a thin, validating adapter onto that machinery: it maps the DSL enums
#   onto the base module's flat-param layout (base._flat_param_count) and the Rust
#   residual head (action_mode="residual_base_stock", backbone_levels=gate OUL,
#   residual_group_of=per-echelon tying). We reuse the verified env / decoder; we do not
#   re-derive the action geometry.
#
#   The KEY difference from OWMR. The OWMR warm start inverts the per-leaf transform so
#   every leaf emits the gate TARGET POSITION (an explicit echelon-target head). PADN's
#   linear leaf consumes a scale-NORMALIZED policy state, so an affine
#   order = clip(level - inventory_position) gate is NOT expressible by a leaf and is NOT
#   invertible (see the seed_robust runner header). The fix -- already built+verified in
#   Rust -- is a RESIDUAL gate-backbone head: order = clamp(gate_order + round(Delta)),
#   Delta produced by the soft tree, Delta == 0 at the all-zero flat vector. So the
#   gate-reproducing warm start for residual_base_stock is simply the ZERO flat vector of
#   length base._flat_param_count(depth, leaf_type) -- gen-0 == gate by construction, no
#   leaf inversion. This compiler emits exactly that.
#
# FULL ALGORITHMIC DESCRIPTION
#   compile_padn_spec(spec) :
#     1. VALIDATE the DSL (no silent coercion; raise PolicySpecError on any bad field):
#          problem        must be "production_assembly_distribution_network"
#          backbone       must be "soft_tree" (the only PADN backbone; learning the
#                         pairwise base-stock GATE is rejected -- it is the comparator)
#          split_type     {oblique, axis_aligned}
#          leaf_type      {constant, linear}
#          temperature    float > 0
#          depth          int >= 1
#          action_head    {residual_base_stock (gate-anchored, default),
#                          vector_quantity (direct per-relation order, NO gate anchor)}
#          per_echelon    granularity of the residual tying:
#                            "per_relation" -> residual_group_of = None (free per relation)
#                            "per_echelon"  -> residual_group_of = the 3-echelon map
#                                              [e1,e2,e2,e3,e3,e3,e3,e1] (relation order =
#                                              [(0,1),(1,2),(1,3),(2,4),(2,5),(3,4),(3,5),
#                                               ext->0]); echelon ids 0/1/2)
#          warm_start     {gate_residual_zero (default for residual_base_stock), none}
#          features       optional subset of the recognized PADN feature bases (the env
#                         policy state is fixed/normalized; features are accepted for DSL
#                         expressiveness and validated, but do NOT change the rollout state
#                         -- naming any unknown feature fails explicitly).
#     2. RESOLVE the residual tying (residual_group_of) from `per_echelon`.
#     3. WARM START. For action_head == "residual_base_stock" and
#        warm_start == "gate_residual_zero", the compiled warm-start flat vector is the
#        ZERO vector of length base._flat_param_count(depth, leaf_type) -> gen-0 == gate
#        byte-exact (the Rust residual head adds round(Delta)=round(0)=0). For
#        warm_start == "none", warm_flat is None (CMA-ES starts from the model default;
#        the honest deploy floor still pins deployment at the gate). vector_quantity has
#        NO gate-reproducing anchor (it emits raw orders, not a gate residual) -> any
#        warm_start other than "none" is REJECTED (no silent fallback).
#     4. REJECT learning the backbone. The backbone gate OUL is searched by the oracle
#        and supplied to the Rust head via backbone_levels; the DSL may NOT carry / learn
#        backbone levels. A `learn_backbone: true` (or a backbone_levels field) is an
#        explicit error.
#
#   The compiler returns a CompiledPadnPolicy carrying: backbone, depth, leaf_type,
#   split_type, temperature, policy_action_mode (= action_head), residual_group_of,
#   warm_start mode, warm_flat (or None), warm_started, and features.
#   evaluate_policy_spec_padn.py consumes exactly these and supplies the searched gate
#   OUL as backbone_levels at rollout time.
#
# NO SILENT FALLBACKS
#   Every validation failure raises PolicySpecError with the raw offending value and the
#   allowed set. The oracle CLI catches it and returns compiled_ok=false + the error
#   string, never a coerced "closest" spec.
# =============================================================================

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The verified PADN topology / flat-param layout / bindings live in the problem's scripts
# dir, not in the importable invman package. Make that dir importable so we reuse the
# verified machinery (base._flat_param_count, ACTION_DIM, the Rust action modes) instead
# of duplicating the action geometry.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_PADN_SCRIPT_DIR = _PACKAGE_ROOT / "scripts" / "production_assembly_distribution_network"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))
if str(_PADN_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_PADN_SCRIPT_DIR))


class PolicySpecError(ValueError):
    """Raised on any invalid PADN DSL field. Carries a human-readable, raw-value message
    so the oracle CLI can return it verbatim under compiled_ok=false."""


PROBLEM = "production_assembly_distribution_network"
BACKBONES = ("soft_tree",)
SPLIT_TYPES = ("oblique", "axis_aligned")
LEAF_TYPES = ("constant", "linear")
# Gate-anchored residual head (default) + the legacy direct per-relation order head.
ACTION_HEADS = ("residual_base_stock", "vector_quantity")
# residual tying granularity -> residual_group_of map.
PER_ECHELON_MODES = ("per_relation", "per_echelon")
WARM_STARTS = ("gate_residual_zero", "none")

# Only the gate-anchored head has a gate-reproducing warm start.
_GATE_ANCHORED_HEADS = ("residual_base_stock",)

# PADN supply-relation order (length ACTION_DIM = 8):
#   [(0,1),(1,2),(1,3),(2,4),(2,5),(3,4),(3,5), ext->0]
# Per-echelon tying map (echelon ids 0/1/2): e1 = relations (0,1) & ext->0; e2 = (1,2),(1,3);
# e3 = the four customer relations (2,4),(2,5),(3,4),(3,5). This MIRRORS the gate's own
# per-echelon grid (a/b/c) in base.search_best_pairwise_base_stock, so a per-echelon
# residual lives in the same 3-D subspace the gate is searched over.
PER_ECHELON_GROUP_OF = (0, 1, 1, 2, 2, 2, 2, 0)

# Recognized PADN feature names (accepted for DSL expressiveness; the env policy state is
# fixed/normalized so these are validated but do not change the rollout state mode). These
# mirror the env's actual NetworkInventoryState components (env.rs): finished_inventory,
# raw_inventory_by_relation, internal_backlog_by_edge, external_backlog, supply_pipelines,
# plus the remaining-horizon and scale features. Naming any other feature fails explicitly
# (no silent acceptance of unknown features).
SUPPORTED_FEATURES = {
    "finished_inventory",
    "raw_inventory",
    "pipeline",
    "internal_backlog",
    "external_backlog",
    "backlog",
    "remaining_horizon",
    "scale",
}


@dataclass
class CompiledPadnPolicy:
    """Everything evaluate_policy_spec_padn.py needs to train + score one PADN spec."""

    backbone: str
    depth: int
    leaf_type: str
    split_type: str
    temperature: float
    policy_action_mode: str          # the Rust action_mode (residual_base_stock | vector_quantity)
    per_echelon: str
    residual_group_of: list[int] | None
    warm_start_mode: str
    warm_flat: list[float] | None    # gate-residual-zero anchor (None if warm_start == none)
    warm_started: bool
    n_params: int
    features: tuple[str, ...]
    problem: str = PROBLEM


# --------------------------------------------------------------------------- #
# Validation helpers (explicit failure, no coercion)                          #
# --------------------------------------------------------------------------- #
def _require(spec: dict, key: str) -> Any:
    if key not in spec:
        raise PolicySpecError(f"missing required spec field '{key}'")
    return spec[key]


def _check_enum(value: Any, allowed: tuple[str, ...], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise PolicySpecError(
            f"invalid '{field_name}' = {value!r}; expected one of: {', '.join(allowed)}"
        )
    return value


def _resolve_per_echelon(value: Any) -> str:
    """Normalize the `per_echelon` granularity field, accepting BOTH a boolean (the
    natural DSL encoding: True == per-echelon tying, False == free per-relation residual)
    and the explicit string enum {per_relation, per_echelon}. Anything else fails."""
    if isinstance(value, bool):  # bool BEFORE str/int (bool is an int subclass)
        return "per_echelon" if value else "per_relation"
    if isinstance(value, str) and value in PER_ECHELON_MODES:
        return value
    raise PolicySpecError(
        f"invalid 'per_echelon' = {value!r}; expected a boolean or one of: "
        f"{', '.join(PER_ECHELON_MODES)}"
    )


def _resolve_features(features: Any) -> tuple[str, ...]:
    if features is None:
        return tuple()
    if not isinstance(features, (list, tuple)):
        raise PolicySpecError(
            f"invalid 'features' = {features!r}; expected a list of strings"
        )
    resolved: list[str] = []
    for feat in features:
        if not isinstance(feat, str) or feat not in SUPPORTED_FEATURES:
            raise PolicySpecError(
                f"invalid feature {feat!r}; expected a subset of: "
                f"{', '.join(sorted(SUPPORTED_FEATURES))}"
            )
        resolved.append(feat)
    return tuple(resolved)


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def compile_padn_spec(spec: dict) -> CompiledPadnPolicy:
    """Compile one PADN DSL spec into a CompiledPadnPolicy. Raises PolicySpecError on any
    invalid field. The searched gate OUL (backbone_levels) is supplied later by the oracle
    at rollout time -- the warm start for residual_base_stock is the all-zero flat vector
    and is gate-reproducing regardless of the specific gate OUL (Delta == 0)."""
    import autoresearch_mixed_distribution_assembly_network as base  # verified PADN machinery

    if not isinstance(spec, dict):
        raise PolicySpecError(f"spec must be a JSON object, got {type(spec).__name__}")

    problem = _require(spec, "problem")
    if problem != PROBLEM:
        raise PolicySpecError(
            f"this compiler only targets '{PROBLEM}', got problem={problem!r}"
        )

    # REJECT learning the backbone: it is the comparator, searched by the oracle, never
    # carried/learned through the DSL.
    if spec.get("learn_backbone"):
        raise PolicySpecError(
            "learn_backbone is not allowed: the pairwise base-stock backbone is the GATE "
            "comparator (searched by the oracle), not a learnable head"
        )
    if "backbone_levels" in spec:
        raise PolicySpecError(
            "the DSL may not carry 'backbone_levels': the gate OUL is searched by the "
            "oracle and supplied to the residual head at rollout time"
        )

    backbone = _check_enum(_require(spec, "backbone"), BACKBONES, "backbone")
    split_type = _check_enum(_require(spec, "split_type"), SPLIT_TYPES, "split_type")
    leaf_type = _check_enum(_require(spec, "leaf_type"), LEAF_TYPES, "leaf_type")
    action_head = _check_enum(_require(spec, "action_head"), ACTION_HEADS, "action_head")
    warm_start = _check_enum(_require(spec, "warm_start"), WARM_STARTS, "warm_start")
    per_echelon = _resolve_per_echelon(_require(spec, "per_echelon"))

    raw_temp = spec.get("temperature", base.TEMPERATURE_DEFAULT)
    try:
        temperature = float(raw_temp)
    except (TypeError, ValueError):
        raise PolicySpecError(f"invalid 'temperature' = {raw_temp!r}; expected a number")
    if not (temperature > 0.0):
        raise PolicySpecError(f"invalid 'temperature' = {temperature}; must be > 0")

    raw_depth = _require(spec, "depth")
    if not isinstance(raw_depth, int) or isinstance(raw_depth, bool) or raw_depth < 1:
        raise PolicySpecError(
            f"invalid 'depth' = {raw_depth!r}; expected an integer >= 1 for soft_tree"
        )
    depth = int(raw_depth)

    features = _resolve_features(spec.get("features"))

    # --- residual tying -------------------------------------------------------
    if per_echelon == "per_echelon":
        residual_group_of: list[int] | None = list(PER_ECHELON_GROUP_OF)
        if len(residual_group_of) != base.ACTION_DIM:
            raise PolicySpecError(
                f"internal per-echelon group map length {len(residual_group_of)} != "
                f"ACTION_DIM {base.ACTION_DIM}"
            )
    else:
        residual_group_of = None  # free per-relation residual

    # --- warm start <-> action-head consistency -------------------------------
    n_params = int(base._flat_param_count(depth, leaf_type))
    if warm_start == "none":
        warm_flat: list[float] | None = None
        warm_started = False
    else:  # gate_residual_zero
        if action_head not in _GATE_ANCHORED_HEADS:
            raise PolicySpecError(
                f"warm_start 'gate_residual_zero' requires a gate-anchored action_head "
                f"(one of {', '.join(_GATE_ANCHORED_HEADS)}); got action_head={action_head!r}. "
                f"vector_quantity emits raw orders and has no gate-reproducing anchor -- "
                f"use warm_start 'none'"
            )
        # The gate-reproducing warm start IS the all-zero flat vector: the Rust residual
        # head computes order = clamp(gate_order + round(Delta)) and Delta(zeros) == 0, so
        # gen-0 == gate byte-exact regardless of the specific searched gate OUL.
        warm_flat = [0.0] * n_params
        warm_started = True

    return CompiledPadnPolicy(
        backbone=backbone,
        depth=depth,
        leaf_type=leaf_type,
        split_type=split_type,
        temperature=temperature,
        policy_action_mode=action_head,
        per_echelon=per_echelon,
        residual_group_of=residual_group_of,
        warm_start_mode=warm_start,
        warm_flat=warm_flat,
        warm_started=warm_started,
        n_params=n_params,
        features=features,
    )
