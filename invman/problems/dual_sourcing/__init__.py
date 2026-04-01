"""Dual-sourcing inventory control problem package."""

from invman.problems.dual_sourcing.benchmark import evaluate_default_heuristics
from invman.problems.dual_sourcing.dp import solve_bounded_dp
from invman.problems.dual_sourcing.env import (
    DualSourcingEnv,
    build_env_from_args,
    get_model_fitness,
    get_population_fitness,
)
from invman.problems.dual_sourcing.heuristics import (
    build_fixed_demand_path,
    evaluate_policy_across_seeds,
    evaluate_policy_cost,
    get_capped_dual_index_action,
    get_dual_index_action,
    get_single_index_action,
    get_tailored_base_surge_action,
    search_best_capped_dual_index_policy,
    search_best_dual_index_policy,
    search_best_single_index_policy,
    search_best_tailored_base_surge_policy,
)
from invman.problems.dual_sourcing.policies import (
    SUPPORTED_POLICY_BACKBONES,
    apply_action_adapter,
    build_action_adapter_config,
    build_control_spec,
    build_policy_context,
    normalize_action_adapter,
)
from invman.problems.dual_sourcing.reference_instances import (
    DUAL_SOURCING_BENCHMARK_REFERENCE,
    DUAL_SOURCING_REFERENCE_INSTANCES,
    build_reference_args,
    get_benchmark_reference,
    get_primary_reference_instance,
    get_reference_instance,
    list_reference_instances,
)

__all__ = [
    "DualSourcingEnv",
    "DUAL_SOURCING_BENCHMARK_REFERENCE",
    "DUAL_SOURCING_REFERENCE_INSTANCES",
    "SUPPORTED_POLICY_BACKBONES",
    "apply_action_adapter",
    "build_action_adapter_config",
    "build_control_spec",
    "build_env_from_args",
    "build_fixed_demand_path",
    "build_policy_context",
    "build_reference_args",
    "evaluate_default_heuristics",
    "evaluate_policy_across_seeds",
    "evaluate_policy_cost",
    "get_capped_dual_index_action",
    "get_dual_index_action",
    "get_benchmark_reference",
    "get_model_fitness",
    "get_population_fitness",
    "get_primary_reference_instance",
    "get_reference_instance",
    "get_single_index_action",
    "get_tailored_base_surge_action",
    "list_reference_instances",
    "normalize_action_adapter",
    "search_best_capped_dual_index_policy",
    "search_best_dual_index_policy",
    "search_best_single_index_policy",
    "search_best_tailored_base_surge_policy",
    "solve_bounded_dp",
]
