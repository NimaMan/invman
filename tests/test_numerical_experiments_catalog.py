from pathlib import Path
import sys
import subprocess

from invman.cpu_limits import CPU_THREAD_ENV_VARS, cpu_limited_environ
from numerical_experiments.catalog import get_suite, list_suites


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ready_suites_include_lost_sales_and_fixed_cost_preflight_and_full_grid():
    ready_ids = {suite.suite_id for suite in list_suites(status="ready")}
    assert "fixed_cost_known_optimum_validation" in ready_ids
    assert "fixed_cost_single_instance_check" in ready_ids
    assert "fixed_cost_full_policy_grid" in ready_ids
    assert "lost_sales_single_instance_check" in ready_ids
    assert "lost_sales_full_policy_grid" in ready_ids


def test_can_build_command_for_suite():
    suite = get_suite("lost_sales_single_instance_check")
    command = suite.command(Path("/tmp/project"), "python3")
    assert command[0] == "python3"
    assert command[1].endswith("scripts/lost_sales/benchmark_canonical_suite.py")


def test_can_build_command_for_lost_sales_and_fixed_cost_full_grid_suites():
    suite = get_suite("lost_sales_full_policy_grid")
    command = suite.command(Path("/tmp/project"), "python3")
    assert command[1].endswith("scripts/lost_sales/benchmark_full_suite.py")

    suite = get_suite("fixed_cost_known_optimum_validation")
    command = suite.command(Path("/tmp/project"), "python3")
    assert command[1].endswith("scripts/lost_sales_fixed_order_cost/benchmark_canonical_suite.py")
    assert "--reference" in command
    assert "bijvank2015_table1_l2_p14_k5" in command

    suite = get_suite("fixed_cost_full_policy_grid")
    command = suite.command(Path("/tmp/project"), "python3")
    assert command[1].endswith("scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py")


def test_ready_suite_scripts_exist():
    for suite in list_suites(status="ready"):
        assert (PROJECT_ROOT / suite.script_path).exists(), suite.suite_id


def test_ready_suite_commands_are_repo_local_and_do_not_reference_nested_rust():
    for suite in list_suites(status="ready"):
        command = suite.command(PROJECT_ROOT, sys.executable)
        assert Path(command[1]).is_absolute(), suite.suite_id
        assert Path(command[1]).is_relative_to(PROJECT_ROOT), suite.suite_id
        rendered = " ".join(command)
        assert "rust/src" not in rendered
        assert "rust/Cargo" not in rendered
        assert "--manifest-path rust" not in rendered


def test_ready_suite_scripts_expose_help_under_cpu_cap():
    for suite in list_suites(status="ready"):
        command = [sys.executable, str(PROJECT_ROOT / suite.script_path), "--help"]
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=cpu_limited_environ(1),
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, f"{suite.suite_id}: {result.stderr}"
        assert "usage:" in result.stdout.lower()


def test_runner_launches_suite_with_cpu_limited_environment(monkeypatch):
    from numerical_experiments import run

    captured = {}

    def fake_run(command, *, cwd, env):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        return subprocess.CompletedProcess(command, 0)

    for name in CPU_THREAD_ENV_VARS:
        monkeypatch.setenv(name, "32")
    monkeypatch.setattr(run.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run.sys,
        "argv",
        [
            "run.py",
            "--suite",
            "lost_sales_single_instance_check",
            "--mp_num_processors",
            "2",
        ],
    )

    run.main()

    assert captured["cwd"] == run.PROJECT_ROOT
    assert captured["command"][1].endswith("scripts/lost_sales/benchmark_canonical_suite.py")
    for name in CPU_THREAD_ENV_VARS:
        assert 1 <= int(captured["env"][name]) <= 2


def test_owmr_ready_suite_outputs_to_root_rust_source_tree():
    from scripts.one_warehouse_multi_retailer import run_paper_benchmark

    expected_dir = (
        PROJECT_ROOT
        / "src"
        / "problems"
        / "one_warehouse_multi_retailer"
        / "experiments"
        / "reports"
    )
    assert run_paper_benchmark.DEFAULT_OUTPUT_JSON == expected_dir / "latest_report.json"
    assert run_paper_benchmark.DEFAULT_OUTPUT_MARKDOWN == expected_dir / "README.md"
    assert "rust/src" not in str(run_paper_benchmark.DEFAULT_OUTPUT_JSON)


def test_dual_sourcing_suite_is_marked_exploratory():
    suite = get_suite("dual_sourcing_backbone_screen")
    assert suite.status == "exploratory"
    assert "capped_dual_index" in suite.heuristics
