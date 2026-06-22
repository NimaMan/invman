"""README-based verification contracts for problem benchmark targets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEMS_DIR = REPO_ROOT / "src" / "problems"
CONTRACT_RE = re.compile(
    r"```json verification-target\n(?P<body>.*?)\n```",
    flags=re.DOTALL,
)
REQUIRED_FIELDS = {
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
REQUIRED_REFERENCE_FIELDS = {"citation", "locator", "doi_or_url", "literature_verified"}


class ContractError(ValueError):
    """Raised when a README verification contract is missing or malformed."""


@dataclass(frozen=True)
class Comparison:
    """Contract comparison result."""

    passed: bool
    status: str
    detail: str


@dataclass(frozen=True)
class VerificationResult:
    """Full verifier result for one problem README contract."""

    problem: str
    instance_id: str
    metric: str
    literature_verified: bool
    comparison: Comparison
    command_ran: bool = False
    command_returncode: int | None = None
    command_stdout: str = ""
    command_stderr: str = ""

    @property
    def passed(self) -> bool:
        return self.comparison.passed and (
            not self.command_ran or self.command_returncode == 0
        )


def problem_readme_path(problem: str, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "src" / "problems" / str(problem) / "README.md"


def read_contract(problem: str, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Read and parse one problem README's fenced verification-target JSON."""

    path = problem_readme_path(problem, repo_root)
    if not path.is_file():
        raise ContractError(f"missing README for problem {problem!r}: {path}")
    text = path.read_text(encoding="utf-8")
    match = CONTRACT_RE.search(text)
    if match is None:
        raise ContractError(f"{problem!r} README missing json verification-target block")
    try:
        contract = json.loads(match.group("body"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{problem!r} verification-target is not valid JSON: {exc}") from exc
    if not isinstance(contract, dict):
        raise ContractError(f"{problem!r} verification-target must be a JSON object")
    validate_contract(contract, expected_problem=str(problem))
    return contract


def validate_contract(contract: dict[str, Any], *, expected_problem: str | None = None) -> None:
    """Validate the minimal README contract shape."""

    missing = sorted(REQUIRED_FIELDS - set(contract))
    if missing:
        raise ContractError(f"missing required field(s): {', '.join(missing)}")
    if contract.get("schema_version", 1) != 1:
        raise ContractError("schema_version must be 1 when present")
    if expected_problem is not None and contract["problem"] != expected_problem:
        raise ContractError(
            f"problem field {contract['problem']!r} does not match {expected_problem!r}"
        )
    if not isinstance(contract["problem"], str) or not contract["problem"]:
        raise ContractError("problem must be a non-empty string")
    if not isinstance(contract["instance_id"], str) or not contract["instance_id"]:
        raise ContractError("instance_id must be a non-empty string")
    if not isinstance(contract["instance_parameters"], dict):
        raise ContractError("instance_parameters must be an object")
    if not isinstance(contract["policy"], str) or not contract["policy"]:
        raise ContractError("policy must be a non-empty string")
    if not isinstance(contract["metric"], str) or not contract["metric"]:
        raise ContractError("metric must be a non-empty string")
    if not isinstance(contract["tolerance"], dict) or not contract["tolerance"]:
        raise ContractError("tolerance must be a non-empty object")
    if not isinstance(contract["command"], str) or not contract["command"].strip():
        raise ContractError("command must be a non-empty string")

    reference = contract["reference"]
    if not isinstance(reference, dict):
        raise ContractError("reference must be an object")
    missing_reference = sorted(REQUIRED_REFERENCE_FIELDS - set(reference))
    if missing_reference:
        raise ContractError(
            f"reference missing required field(s): {', '.join(missing_reference)}"
        )
    if not isinstance(reference["citation"], str) or not reference["citation"]:
        raise ContractError("reference.citation must be a non-empty string")
    if reference["locator"] is not None and not isinstance(reference["locator"], str):
        raise ContractError("reference.locator must be a string or null")
    if reference["doi_or_url"] is not None and not isinstance(reference["doi_or_url"], str):
        raise ContractError("reference.doi_or_url must be a string or null")
    if not isinstance(reference["literature_verified"], bool):
        raise ContractError("reference.literature_verified must be a boolean")


def compare_contract_values(contract: dict[str, Any]) -> Comparison:
    """Compare declared code_value to expected_value using declared tolerance."""

    expected = contract["expected_value"]
    code = contract["code_value"]
    tolerance = contract["tolerance"]
    literature_verified = bool(contract["reference"]["literature_verified"])

    if expected is None:
        return Comparison(
            passed=True,
            status="not_literature_verified",
            detail=(
                "No literature/reference expected_value is declared; this is a "
                f"repo-native anchor with code_value={code!r}."
            ),
        )

    if code is None:
        return Comparison(
            passed=False,
            status="missing_code_value",
            detail="expected_value is declared but code_value is null",
        )

    if tolerance.get("exact") is True:
        return Comparison(
            passed=code == expected,
            status=_target_status(literature_verified),
            detail=f"exact comparison: code_value={code!r}, expected_value={expected!r}",
        )

    if tolerance.get("rounded_exact") is True:
        if not isinstance(code, (int, float)) or not isinstance(expected, (int, float)):
            raise ContractError("rounded_exact tolerance requires numeric code/expected values")
        rounded = round(float(code))
        return Comparison(
            passed=rounded == expected,
            status=_target_status(literature_verified),
            detail=(
                f"rounded comparison: round(code_value)={rounded!r}, "
                f"expected_value={expected!r}"
            ),
        )

    abs_key = next(
        (
            key
            for key in ("absolute", "absolute_percentage_points", "display_rounding_absolute")
            if key in tolerance
        ),
        None,
    )
    if abs_key is not None:
        diff = _abs_numeric_delta(code, expected)
        allowed = float(tolerance[abs_key])
        return Comparison(
            passed=diff <= allowed,
            status=_target_status(literature_verified),
            detail=(
                f"{abs_key}: |code_value - expected_value|={diff:.12g}, "
                f"allowed={allowed:.12g}"
            ),
        )

    if "relative_percent" in tolerance:
        if not isinstance(code, (int, float)) or not isinstance(expected, (int, float)):
            raise ContractError("relative_percent tolerance requires numeric code/expected values")
        if float(expected) == 0.0:
            raise ContractError("relative_percent tolerance cannot use expected_value=0")
        diff = abs(float(code) - float(expected)) / abs(float(expected)) * 100.0
        allowed = float(tolerance["relative_percent"])
        return Comparison(
            passed=diff <= allowed,
            status=_target_status(literature_verified),
            detail=f"relative_percent: gap={diff:.12g}%, allowed={allowed:.12g}%",
        )

    raise ContractError(
        "unsupported tolerance; use exact, rounded_exact, absolute, "
        "absolute_percentage_points, display_rounding_absolute, or relative_percent"
    )


def verify_problem(
    problem: str,
    *,
    run_command: bool = False,
    repo_root: Path = REPO_ROOT,
) -> VerificationResult:
    """Validate one problem README contract and optionally execute its command."""

    contract = read_contract(problem, repo_root=repo_root)
    comparison = compare_contract_values(contract)
    command_ran = False
    command_returncode = None
    command_stdout = ""
    command_stderr = ""

    if run_command:
        command_ran = True
        completed = subprocess.run(
            contract["command"],
            cwd=repo_root,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        command_returncode = completed.returncode
        command_stdout = completed.stdout
        command_stderr = completed.stderr

    return VerificationResult(
        problem=contract["problem"],
        instance_id=contract["instance_id"],
        metric=contract["metric"],
        literature_verified=bool(contract["reference"]["literature_verified"]),
        comparison=comparison,
        command_ran=command_ran,
        command_returncode=command_returncode,
        command_stdout=command_stdout,
        command_stderr=command_stderr,
    )


def _abs_numeric_delta(code: Any, expected: Any) -> float:
    if isinstance(code, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(code) - float(expected))
    if isinstance(code, list) and isinstance(expected, list) and len(code) == len(expected):
        if all(isinstance(v, (int, float)) for v in code + expected):
            return max(abs(float(a) - float(b)) for a, b in zip(code, expected))
    raise ContractError("absolute tolerance requires numeric code/expected values")


def _target_status(literature_verified: bool) -> str:
    return "literature_verified" if literature_verified else "reference_not_literature_verified"


def _format_result(result: VerificationResult) -> str:
    lines = [
        f"{result.problem}: {'PASS' if result.passed else 'FAIL'}",
        f"instance: {result.instance_id}",
        f"metric: {result.metric}",
        f"status: {result.comparison.status}",
        result.comparison.detail,
    ]
    if not result.literature_verified:
        lines.append("literature: not strict literature-verified")
    if result.command_ran:
        lines.append(f"command return code: {result.command_returncode}")
        if result.command_stdout.strip():
            lines.append("command stdout:")
            lines.append(result.command_stdout.rstrip())
        if result.command_stderr.strip():
            lines.append("command stderr:")
            lines.append(result.command_stderr.rstrip())
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a src/problems/<problem>/README.md verification-target contract.",
    )
    parser.add_argument("problem", help="Problem directory name under src/problems")
    parser.add_argument(
        "--run-command",
        action="store_true",
        help="Execute the contract command after validating and comparing declared values.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable verifier result.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify_problem(args.problem, run_command=args.run_command)
    except ContractError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "problem": result.problem,
                    "instance_id": result.instance_id,
                    "metric": result.metric,
                    "literature_verified": result.literature_verified,
                    "passed": result.passed,
                    "comparison": {
                        "passed": result.comparison.passed,
                        "status": result.comparison.status,
                        "detail": result.comparison.detail,
                    },
                    "command_ran": result.command_ran,
                    "command_returncode": result.command_returncode,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_format_result(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
