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
        suite_id="lost_sales_single_instance_check",
        problem="lost_sales",
        status="ready",
        purpose="Run one canonical vanilla lost-sales instance end-to-end as a preflight check before the full literature grid run.",
        heuristics=("myopic1", "myopic2", "svbs", "capped_base_stock_literature"),
        base_policies=(
            "linear_categorical_quantity_q20",
            "nn_categorical_quantity_q20",
        ),
        improved_policies=(
            "linear_sigmoid_direct_quantity",
            "linear_soft_gated_direct_quantity",
            "nn_soft_gated_direct_quantity_h8_selu",
            "linear_hard_gated_direct_quantity",
            "linear_soft_gated_ordinal_quantity",
            "nn_soft_gated_ordinal_quantity_h8_selu",
            "nn_soft_gated_ordinal_quantity",
            "soft_tree_depth1_linear_leaf",
            "soft_tree_depth2_linear_leaf",
            "soft_tree_depth1_sigmoid_linear_leaf_q20",
            "soft_tree_depth2_sigmoid_linear_leaf_q20",
        ),
        script_path="scripts/lost_sales/benchmark_canonical_suite.py",
        notes=(
            "Uses the canonical L=4, p=4, Poisson(5) instance.",
            "This is the quick correctness/performance check before launching the full paper suite.",
        ),
    ),
    ExperimentSuite(
        suite_id="lost_sales_full_policy_grid",
        problem="lost_sales",
        status="ready",
        purpose="Run the full 32-instance vanilla lost-sales paper-style grid with heuristics and learned policy families.",
        heuristics=("myopic1", "myopic2", "svbs", "capped_base_stock_literature", "optimal_literature_when_available"),
        base_policies=(
            "linear_categorical_quantity_q20",
            "nn_categorical_quantity_q20",
        ),
        improved_policies=(
            "linear_sigmoid_direct_quantity",
            "linear_soft_gated_direct_quantity",
            "nn_soft_gated_direct_quantity_h8_selu",
            "linear_hard_gated_direct_quantity",
            "linear_soft_gated_ordinal_quantity",
            "nn_soft_gated_ordinal_quantity_h8_selu",
            "nn_soft_gated_ordinal_quantity",
            "soft_tree_depth1_linear_leaf",
            "soft_tree_depth2_linear_leaf",
            "soft_tree_depth1_sigmoid_linear_leaf_q20",
            "soft_tree_depth2_sigmoid_linear_leaf_q20",
        ),
        script_path="scripts/lost_sales/benchmark_full_suite.py",
        notes=(
            "This suite is the main data-generation path for the vanilla lost-sales paper section.",
            "It emits per-instance JSONs with heuristic evaluations, literature anchors, and learned-policy results.",
        ),
    ),
    ExperimentSuite(
        suite_id="fixed_cost_known_optimum_validation",
        problem="lost_sales_fixed_order_cost",
        status="ready",
        purpose="Run the published Bijvank-Bhulai-Huh Table 1 validation instance end-to-end and compare heuristics and learned policies to the known exact optimum.",
        heuristics=("s_s", "s_nq", "modified_s_s_q", "optimal_literature_when_available"),
        base_policies=("linear_categorical_quantity", "nn_categorical_quantity"),
        improved_policies=(
            "linear_sigmoid_direct_quantity",
            "linear_soft_gated_direct_quantity",
            "nn_soft_gated_direct_quantity_h8_selu",
            "linear_hard_gated_direct_quantity",
            "linear_soft_gated_ordinal_quantity",
            "nn_soft_gated_ordinal_quantity_h8_selu",
            "nn_soft_gated_ordinal_quantity",
            "soft_tree_depth2_linear_leaf",
            "soft_tree_depth1_linear_leaf",
            "soft_tree_depth2_sigmoid_linear_leaf_q20",
            "soft_tree_depth1_sigmoid_linear_leaf_q20",
        ),
        script_path="scripts/lost_sales_fixed_order_cost/benchmark_canonical_suite.py",
        script_args=(
            "--reference",
            "bijvank2015_table1_l2_p14_k5",
            "--run_tag",
            "fixed_cost_known_optimum_validation_5k",
        ),
        notes=(
            "Uses the single published validation instance with exact optimal cost 11.46.",
            "This suite is the fixed-cost analogue of the vanilla lost-sales known-optimum comparisons.",
        ),
    ),
    ExperimentSuite(
        suite_id="fixed_cost_single_instance_check",
        problem="lost_sales_fixed_order_cost",
        status="ready",
        purpose="Run one fixed-cost literature-aligned instance end-to-end as a preflight check before the full grid run.",
        heuristics=("s_s", "s_nq", "modified_s_s_q"),
        base_policies=("linear_categorical_quantity", "nn_categorical_quantity"),
        improved_policies=(
            "linear_sigmoid_direct_quantity",
            "linear_soft_gated_direct_quantity",
            "nn_soft_gated_direct_quantity_h8_selu",
            "linear_hard_gated_direct_quantity",
            "linear_soft_gated_ordinal_quantity",
            "nn_soft_gated_ordinal_quantity_h8_selu",
            "nn_soft_gated_ordinal_quantity",
            "soft_tree_depth2_linear_leaf",
            "soft_tree_depth1_linear_leaf",
            "soft_tree_depth2_sigmoid_linear_leaf_q20",
            "soft_tree_depth1_sigmoid_linear_leaf_q20",
        ),
        script_path="scripts/lost_sales_fixed_order_cost/benchmark_canonical_suite.py",
        notes=(
            "Uses the canonical L=4, p=4, K=5 literature-aligned instance.",
            "This is the quick correctness/performance check before launching the full grid suite.",
        ),
    ),
    ExperimentSuite(
        suite_id="fixed_cost_full_policy_grid",
        problem="lost_sales_fixed_order_cost",
        status="ready",
        purpose="Run the full fixed-cost paper-style grid with heuristics and learned policy families on the literature-aligned instance set.",
        heuristics=("s_s", "s_nq", "modified_s_s_q"),
        base_policies=("linear_categorical_quantity", "nn_categorical_quantity"),
        improved_policies=(
            "linear_sigmoid_direct_quantity",
            "linear_soft_gated_direct_quantity",
            "nn_soft_gated_direct_quantity_h8_selu",
            "linear_hard_gated_direct_quantity",
            "linear_soft_gated_ordinal_quantity",
            "nn_soft_gated_ordinal_quantity_h8_selu",
            "nn_soft_gated_ordinal_quantity",
            "soft_tree_depth2_linear_leaf",
            "soft_tree_depth1_linear_leaf",
            "soft_tree_depth2_sigmoid_linear_leaf_q20",
            "soft_tree_depth1_sigmoid_linear_leaf_q20",
        ),
        script_path="scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py",
        notes=(
            "This suite is the main data-generation path for the fixed-cost paper section.",
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
        script_path="scripts/dual_sourcing/validate_reference_grid.py",
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
        script_path="scripts/dual_sourcing/autoresearch_dual_sourcing_backbones.py",
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
        script_path="scripts/dual_sourcing/autoresearch_dual_sourcing.py",
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
        script_path="scripts/multi_echelon/autoresearch_multi_echelon.py",
        script_args=("--budget", "full", "--description", "numerical_experiments_run"),
        notes=(
            "Current multi-echelon suite is still exploratory; the final policy family is not frozen.",
        ),
    ),
    ExperimentSuite(
        suite_id="owmr_kaynov_full_paper_benchmark",
        problem="one_warehouse_multi_retailer",
        status="ready",
        purpose="Run the full 14-instance Kaynov OWMR paper benchmark with literature-aligned heuristic evaluation and soft-tree learned policies.",
        heuristics=("echelon_base_stock_proportional", "echelon_base_stock_min_shortage", "published_ppo"),
        base_policies=(),
        improved_policies=("soft_tree_axis_aligned_linear_targets",),
        script_path="scripts/one_warehouse_multi_retailer/run_paper_benchmark.py",
        notes=(
            "Uses Kaynov Section 4.1 / Table A.3 evaluation protocol: 1000 trajectories of length 100.",
            "Heuristic search uses common random numbers; learned policies train with random-sequential allocation and evaluate with proportional allocation.",
        ),
    ),
    ExperimentSuite(
        suite_id="multi_echelon_gijs_full_paper_benchmark",
        problem="multi_echelon",
        status="exploratory",
        purpose="Run the two Gijs / Van Roy literature settings with the current best Rust-backed soft-tree policy family and compare to the published relative improvements.",
        heuristics=("constant_base_stock_grid_search", "published_a3c_relative_savings", "published_van_roy_relative_savings"),
        base_policies=(),
        improved_policies=("soft_tree_summary_state_discrete_targets",),
        script_path="scripts/multi_echelon/run_paper_benchmark.py",
        script_args=(
            "--depth",
            "2",
            "--split_type",
            "oblique",
            "--leaf_type",
            "linear",
            "--temperature",
            "0.1",
            "--training_horizon",
            "5000",
            "--training_episodes",
            "60",
            "--es_population",
            "16",
            "--sigma_init",
            "0.75",
            "--train_seed_batch",
            "4",
        ),
        notes=(
            "Uses the two Gijs Table 3 settings and the paper-style evaluation budget of 100 sample paths of length 100,000.",
            "Current report is provisional because the paper text on the constant-base-stock comparator domain still needs final clarification.",
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
