import json
import re
from pathlib import Path

import pytest

from invman.benchmarks import verify


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBLEMS_DIR = REPO_ROOT / "src" / "problems"


def problem_dirs() -> list[Path]:
    return sorted(
        p
        for p in PROBLEMS_DIR.iterdir()
        if p.is_dir() and p.name != "core" and not p.name.startswith("__")
    )


CONTRACT_RE = re.compile(
    r"```json verification-target\n(?P<body>.*?)\n```",
    flags=re.DOTALL,
)


def readme_contract(problem: Path) -> dict:
    text = (problem / "README.md").read_text(encoding="utf-8")
    match = CONTRACT_RE.search(text)
    assert match is not None, f"{problem.name} missing json verification-target block"
    return json.loads(match.group("body"))


def test_every_problem_has_readme_verification_contract():
    missing = [p.name for p in problem_dirs() if not (p / "README.md").is_file()]
    assert missing == []


def test_no_problem_uses_separate_verification_file():
    leftovers = [p.name for p in problem_dirs() if (p / "VERIFICATION.md").exists()]
    assert leftovers == []


def test_readme_verification_contracts_are_machine_readable():
    required_top_level = {
        "schema_version",
        "problem",
        "instance_id",
        "instance_parameters",
        "policy",
        "metric",
        "expected_value",
        "reference",
        "code_value",
        "tolerance",
        "command",
    }
    required_reference = {
        "citation",
        "locator",
        "doi_or_url",
        "literature_verified",
    }
    for problem in problem_dirs():
        contract = readme_contract(problem)
        assert required_top_level <= set(contract), problem.name
        assert contract["schema_version"] == 1
        assert contract["problem"] == problem.name
        assert contract["instance_id"]
        assert isinstance(contract["instance_parameters"], dict)
        assert contract["policy"]
        assert contract["metric"]
        assert required_reference <= set(contract["reference"]), problem.name
        assert contract["reference"]["citation"]
        assert isinstance(contract["reference"]["literature_verified"], bool)
        assert isinstance(contract["tolerance"], dict) and contract["tolerance"]
        assert contract["command"]
        verify.validate_contract(contract, expected_problem=problem.name)
        assert verify.compare_contract_values(contract).passed


def test_no_problem_readme_contract_keeps_legacy_nested_contract_fields():
    legacy_fields = {"status", "instance", "comparator", "literature", "reproduction"}
    runtime_budget_fields = {"runtime_budget", "runtime_budget_seconds", "timeout_seconds"}
    for problem in problem_dirs():
        contract = readme_contract(problem)
        assert contract.keys().isdisjoint(legacy_fields), problem.name
        assert contract.keys().isdisjoint(runtime_budget_fields), problem.name


def test_no_literature_targets_are_explicitly_not_literature_verified():
    for problem in problem_dirs():
        contract = readme_contract(problem)
        if contract["expected_value"] is None:
            assert contract["reference"]["literature_verified"] is False, problem.name
            result = verify.compare_contract_values(contract)
            assert result.passed
            assert result.status == "not_literature_verified"
            assert "repo-native" in result.detail


def test_readme_verification_contracts_have_human_audit_sections():
    required = [
        "## Verification target",
        "### Primary target",
        "### Source",
        "### Validation command",
        "Last validated",
    ]
    for problem in problem_dirs():
        text = (problem / "README.md").read_text(encoding="utf-8")
        for needle in required:
            assert needle in text, f"{problem.name} missing {needle!r}"


def test_lost_sales_verifier_runs_contract_command():
    pytest.importorskip("invman_rust")
    result = verify.verify_problem("lost_sales", run_command=True)
    assert result.passed
    assert result.problem == "lost_sales"
    assert result.instance_id == "vanilla_l4_p4_poisson5"
    assert result.comparison.status == "literature_verified"
    assert result.command_ran
    assert result.command_returncode == 0
