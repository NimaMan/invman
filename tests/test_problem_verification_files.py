from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBLEMS_DIR = REPO_ROOT / "src" / "problems"


def problem_dirs() -> list[Path]:
    return sorted(
        p
        for p in PROBLEMS_DIR.iterdir()
        if p.is_dir() and p.name != "core" and not p.name.startswith("__")
    )


def test_every_problem_has_verification_file():
    missing = [p.name for p in problem_dirs() if not (p / "VERIFICATION.md").is_file()]
    assert missing == []


def test_verification_files_have_minimum_contract():
    required = [
        "# Verification Target - ",
        "## Primary Target",
        "## Source",
        "## Validation Command",
        "Last validated",
    ]
    for problem in problem_dirs():
        text = (problem / "VERIFICATION.md").read_text(encoding="utf-8")
        for needle in required:
            assert needle in text, f"{problem.name} missing {needle!r}"
