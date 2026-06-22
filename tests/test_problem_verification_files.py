import json
import re
from pathlib import Path


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
        "status",
        "instance",
        "comparator",
        "literature",
        "reproduction",
    }
    for problem in problem_dirs():
        contract = readme_contract(problem)
        assert required_top_level <= set(contract), problem.name
        assert contract["schema_version"] == 1
        assert contract["problem"] == problem.name
        assert contract["status"]
        assert contract["instance"]["id"]
        assert contract["comparator"]["metric"]
        assert "value" in contract["literature"]
        assert contract["literature"]["source"]
        assert "current_value" in contract["reproduction"]
        assert contract["reproduction"]["command"]
        assert contract["reproduction"]["tolerance"]
        assert contract["reproduction"]["last_validated"]


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
