from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSuite:
    suite_id: str
    problem: str
    status: str
    purpose: str
    heuristics: tuple[str, ...]
    base_policies: tuple[str, ...]
    improved_policies: tuple[str, ...]
    script_path: str
    script_args: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def command(self, project_root: Path, python_bin: str) -> list[str]:
        return [
            python_bin,
            str(project_root / self.script_path),
            *self.script_args,
        ]


EXPERIMENT_SUITES: tuple[ExperimentSuite, ...] = (
    ExperimentSuite(
        suite_id="lost_sales_reference_validation",
        problem="lost_sales",
        status="ready",
        purpose="Validate the canonical vanilla benchmark against trusted heuristic anchors.",
        heuristics=("myopic1", "myopic2", "svbs"),
        base_policies=("linear_categorical_quantity", "nn_categorical_quantity"),
        improved_policies=("soft_tree_oblique_linear_leaf",),
        script_path="scripts/validate_reference_instance.py",
        notes=(
            "Fast sanity check for the canonical lost-sales instance.",
            "Does not train policies; it validates the benchmark layer.",
        ),
    ),
    ExperimentSuite(
        suite_id="fixed_cost_canonical_paperlike",
        problem="lost_sales_fixed_order_cost",
        status="ready",
        purpose="Run the canonical fixed-order-cost benchmark table on the L=4, p=4, K=5 instance.",
        heuristics=("s_s", "s_nq", "modified_s_s_q"),
        base_policies=("linear_categorical_quantity", "nn_categorical_quantity"),
        improved_policies=(
            "linear_gated_ordinal_quantity",
            "nn_gated_ordinal_quantity",
            "soft_tree_depth2_linear_leaf",
            "soft_tree_depth1_linear_leaf",
        ),
        script_path="scripts/benchmark_fixed_cost_canonical_suite.py",
        notes=(
            "Paper-like long-horizon evaluation with the current fixed-cost six-policy matrix.",
            "This is the current main fixed-cost suite.",
        ),
    ),
    ExperimentSuite(
        suite_id="fixed_cost_grid_benchmark",
        problem="lost_sales_fixed_order_cost",
        status="ready",
        purpose="Benchmark heuristic policies on the 16-instance literature subset grid.",
        heuristics=("s_s", "s_nq", "modified_s_s_q"),
        base_policies=(),
        improved_policies=(),
        script_path="scripts/benchmark_fixed_order_cost_grid.py",
        notes=(
            "Grid-level heuristic benchmark only.",
            "Use this to refresh the literature-subset heuristic baseline.",
        ),
    ),
    ExperimentSuite(
        suite_id="fixed_cost_full_policy_grid",
        problem="lost_sales_fixed_order_cost",
        status="ready",
        purpose="Run the full fixed-cost paper-style grid with heuristics and learned policy families on the literature-aligned 16-instance subset.",
        heuristics=("s_s", "s_nq", "modified_s_s_q"),
        base_policies=("linear_categorical_quantity", "nn_categorical_quantity"),
        improved_policies=(
            "linear_gated_ordinal_quantity",
            "nn_gated_ordinal_quantity",
            "soft_tree_depth2_linear_leaf",
            "soft_tree_depth1_linear_leaf",
        ),
        script_path="scripts/benchmark_fixed_cost_full_suite.py",
        notes=(
            "This suite is the full data-generation path for the fixed-cost paper section.",
            "It emits per-instance JSONs with heuristic parameters, learned-policy results, and benchmark metadata.",
        ),
    ),
    ExperimentSuite(
        suite_id="dual_sourcing_reference_grid",
        problem="dual_sourcing",
        status="ready",
        purpose="Validate the six literature dual-sourcing instances and their heuristic/DP baselines.",
        heuristics=("single_index", "dual_index", "capped_dual_index", "tailored_base_surge", "optimal_dp"),
        base_policies=(),
        improved_policies=(),
        script_path="scripts/validate_dual_sourcing_reference_grid.py",
        notes=(
            "Fast benchmark-layer validation for the dual-sourcing package.",
        ),
    ),
    ExperimentSuite(
        suite_id="dual_sourcing_backbone_screen",
        problem="dual_sourcing",
        status="exploratory",
        purpose="Screen linear and NN backbones under identity vs structured action adapters.",
        heuristics=("single_index", "dual_index", "capped_dual_index", "tailored_base_surge"),
        base_policies=("linear_bounded_quantity_identity", "nn_bounded_quantity_identity"),
        improved_policies=("linear_base_surge_targets", "nn_base_surge_targets"),
        script_path="scripts/autoresearch_dual_sourcing_backbones.py",
        script_args=("--budget", "full"),
        notes=(
            "This suite is for policy-design work, not yet a final paper table.",
        ),
    ),
    ExperimentSuite(
        suite_id="dual_sourcing_tree_autoresearch",
        problem="dual_sourcing",
        status="exploratory",
        purpose="Run structured soft-tree policy search on the primary dual-sourcing instance.",
        heuristics=("single_index", "dual_index", "capped_dual_index", "tailored_base_surge"),
        base_policies=("soft_tree_identity",),
        improved_policies=("soft_tree_base_surge_targets",),
        script_path="scripts/autoresearch_dual_sourcing.py",
        script_args=("--budget", "full", "--description", "numerical_experiments_run"),
        notes=(
            "Current dual-sourcing policy-design suite.",
        ),
    ),
    ExperimentSuite(
        suite_id="multi_echelon_autoresearch",
        problem="multi_echelon",
        status="exploratory",
        purpose="Run the current multi-echelon soft-tree benchmark loop on the larger reference setting.",
        heuristics=("constant_base_stock",),
        base_policies=("soft_tree_constant_leaf",),
        improved_policies=("soft_tree_linear_leaf",),
        script_path="scripts/autoresearch_multi_echelon.py",
        script_args=("--budget", "full", "--description", "numerical_experiments_run"),
        notes=(
            "Current multi-echelon suite is still exploratory; the final policy family is not frozen.",
        ),
    ),
)


def list_suites(*, status: str | None = None) -> list[ExperimentSuite]:
    suites = list(EXPERIMENT_SUITES)
    if status is not None:
        suites = [suite for suite in suites if suite.status == status]
    return suites


def get_suite(suite_id: str) -> ExperimentSuite:
    for suite in EXPERIMENT_SUITES:
        if suite.suite_id == suite_id:
            return suite
    known = ", ".join(suite.suite_id for suite in EXPERIMENT_SUITES)
    raise KeyError(f"Unknown numerical experiment suite '{suite_id}'. Available: {known}")
