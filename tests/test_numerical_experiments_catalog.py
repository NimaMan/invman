from pathlib import Path

from numerical_experiments.catalog import get_suite, list_suites


def test_ready_suites_include_fixed_cost_canonical_suite():
    ready_ids = {suite.suite_id for suite in list_suites(status="ready")}
    assert "fixed_cost_canonical_paperlike" in ready_ids
    assert "lost_sales_reference_validation" in ready_ids


def test_can_build_command_for_suite():
    suite = get_suite("fixed_cost_canonical_paperlike")
    command = suite.command(Path("/tmp/project"), "python3")
    assert command[0] == "python3"
    assert command[1].endswith("scripts/benchmark_fixed_cost_canonical_suite.py")


def test_dual_sourcing_suite_is_marked_exploratory():
    suite = get_suite("dual_sourcing_backbone_screen")
    assert suite.status == "exploratory"
    assert "capped_dual_index" in suite.heuristics
