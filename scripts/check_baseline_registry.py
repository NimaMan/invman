#!/usr/bin/env python3
"""Validate baseline/problem-instance registry YAML files.

The registry is intentionally human-authored, so this checker enforces the
parts that are easy to drift: required fields, row coverage, local paths, and
BibTeX citation keys.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only on misconfigured envs.
    raise SystemExit("PyYAML is required; install requirements.txt first") from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOB = "src/problems/**/literature/baselines.yaml"
BIB_PATH = ROOT / "paper" / "references.bib"

VERIFICATION_STATUSES = {
    "strict_literature_verified",
    "partial",
    "table_only",
    "repo_native",
    "not_verified",
}
COMPARATOR_TYPES = {
    "published_exact_optimum",
    "published_heuristics",
    "mixed_published_optimum_and_heuristics",
    "repo_native_heuristics",
    "repo_native_learned_policy",
    "repo_native_practical_trace",
    "none",
}
PAPER_STATUSES = {
    "ready",
    "usable_with_caveat",
    "context_only",
    "not_for_claim",
}
VALUE_KINDS = {
    "scalar",
    "integer",
    "table",
    "artifact_ref",
    "derived",
}
TOP_LEVEL_REQUIRED = {
    "schema_version",
    "problem",
    "registry_owner",
    "source_of_truth",
    "entries",
}
ENTRY_REQUIRED = {
    "id",
    "problem",
    "instance_name",
    "roles",
    "status",
    "source",
    "instance",
    "published_numbers",
    "repo_verification",
    "repo_baseline_gate",
    "paper_link",
}
ROW_REQUIRED = {
    "row_id",
    "label",
    "policy_id",
    "metric",
    "value",
    "value_kind",
    "source_ref",
    "verification_status",
}
GATE_ROW_REQUIRED = {
    "row_id",
    "policy_id",
    "metric",
    "value",
    "value_kind",
}
SNAKE_CASE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
BIB_ENTRY = re.compile(r"@\w+\s*\{\s*([^,\s]+)")
LOCAL_PATH_TOKEN = re.compile(r"(?:src|scripts|docs|paper)/[^\s,;\"')]+")


class RegistryError:
    def __init__(self, path: Path, location: str, message: str) -> None:
        self.path = path
        self.location = location
        self.message = message

    def __str__(self) -> str:
        rel = self.path.relative_to(ROOT)
        return f"{rel}:{self.location}: {self.message}"


def _load_yaml(path: Path) -> tuple[Any, list[RegistryError]]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")), []
    except Exception as exc:  # noqa: BLE001 - include parser details in the error.
        return None, [RegistryError(path, "$", f"cannot parse YAML: {exc}")]


def _bib_keys() -> set[str]:
    if not BIB_PATH.exists():
        return set()
    return {
        match.group(1)
        for match in BIB_ENTRY.finditer(BIB_PATH.read_text(encoding="utf-8"))
    }


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _check_mapping(
    path: Path,
    location: str,
    value: Any,
    required: set[str],
    errors: list[RegistryError],
) -> bool:
    if not isinstance(value, dict):
        errors.append(RegistryError(path, location, "expected a mapping"))
        return False
    missing = sorted(required - set(value))
    for key in missing:
        errors.append(RegistryError(path, location, f"missing required field '{key}'"))
    return not missing


def _path_part(value: str) -> str:
    return value.split("::", 1)[0]


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith(("src/", "scripts/", "docs/", "paper/"))
        or "/" in value
        or value.endswith((".rs", ".py", ".md", ".json", ".yaml", ".yml", ".tex"))
    )


def _path_tokens(value: str) -> list[str]:
    tokens = LOCAL_PATH_TOKEN.findall(value)
    if tokens:
        return tokens
    return [value]


def _check_local_path(
    path: Path,
    location: str,
    value: Any,
    errors: list[RegistryError],
    *,
    required: bool = True,
) -> None:
    if value is None:
        if required:
            errors.append(RegistryError(path, location, "path is required"))
        return
    if not isinstance(value, str):
        errors.append(RegistryError(path, location, "expected path string"))
        return
    for token in _path_tokens(value):
        candidate = _path_part(token)
        if not candidate:
            errors.append(RegistryError(path, location, "empty path"))
            continue
        if candidate.startswith(("http://", "https://")):
            continue
        if not _looks_like_path(candidate):
            continue
        if candidate.startswith("/"):
            errors.append(RegistryError(path, location, f"absolute path is not portable: {candidate}"))
            continue
        if not (ROOT / candidate).exists():
            errors.append(RegistryError(path, location, f"referenced path does not exist: {candidate}"))


def _check_citation_key(
    path: Path,
    location: str,
    key: Any,
    known_keys: set[str],
    errors: list[RegistryError],
) -> None:
    if key is None:
        return
    if not isinstance(key, str) or not key:
        errors.append(RegistryError(path, location, "citation key must be a non-empty string"))
        return
    if key not in known_keys:
        errors.append(RegistryError(path, location, f"citation key not found in {BIB_PATH.relative_to(ROOT)}: {key}"))


def _check_citation_keys(
    path: Path,
    location: str,
    keys: Any,
    known_keys: set[str],
    errors: list[RegistryError],
) -> None:
    if keys is None:
        return
    if not isinstance(keys, list) or not keys:
        errors.append(RegistryError(path, location, "citation_keys must be a non-empty list when present"))
        return
    for idx, key in enumerate(keys):
        _check_citation_key(path, f"{location}[{idx}]", key, known_keys, errors)


def _check_unique(
    path: Path,
    location: str,
    values: list[str],
    errors: list[RegistryError],
) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            errors.append(RegistryError(path, location, f"duplicate id '{value}'"))
        seen.add(value)


def _check_rows(
    path: Path,
    location: str,
    rows: Any,
    known_keys: set[str],
    errors: list[RegistryError],
) -> list[str]:
    if not isinstance(rows, list):
        errors.append(RegistryError(path, location, "rows must be a list"))
        return []

    row_ids: list[str] = []
    for idx, row in enumerate(rows):
        row_loc = f"{location}[{idx}]"
        if not _check_mapping(path, row_loc, row, ROW_REQUIRED, errors):
            if not isinstance(row, dict):
                continue

        row_id = row.get("row_id")
        if isinstance(row_id, str):
            row_ids.append(row_id)
            if not SNAKE_CASE.match(row_id):
                errors.append(RegistryError(path, f"{row_loc}.row_id", "row_id must be snake_case"))
        else:
            errors.append(RegistryError(path, f"{row_loc}.row_id", "row_id must be a string"))

        value_kind = row.get("value_kind")
        if value_kind not in VALUE_KINDS:
            errors.append(RegistryError(path, f"{row_loc}.value_kind", f"unsupported value_kind '{value_kind}'"))

        verification_status = row.get("verification_status")
        if verification_status not in VERIFICATION_STATUSES:
            errors.append(
                RegistryError(path, f"{row_loc}.verification_status", f"unsupported verification_status '{verification_status}'")
            )

        source_ref = row.get("source_ref")
        if isinstance(source_ref, dict):
            _check_citation_key(
                path,
                f"{row_loc}.source_ref.citation_key",
                source_ref.get("citation_key"),
                known_keys,
                errors,
            )
            _check_local_path(
                path,
                f"{row_loc}.source_ref.artifact",
                source_ref.get("artifact"),
                errors,
                required=False,
            )
        elif source_ref is not None:
            errors.append(RegistryError(path, f"{row_loc}.source_ref", "source_ref must be a mapping"))

        _check_local_path(
            path,
            f"{row_loc}.artifact_ref",
            row.get("artifact_ref"),
            errors,
            required=False,
        )

        derived_from = row.get("derived_from")
        if derived_from is not None:
            if not isinstance(derived_from, list) or not derived_from:
                errors.append(RegistryError(path, f"{row_loc}.derived_from", "derived_from must be a non-empty list"))

    _check_unique(path, f"{location}.row_id", row_ids, errors)
    return row_ids


def _check_gate_rows(
    path: Path,
    location: str,
    rows: Any,
    known_keys: set[str],
    errors: list[RegistryError],
) -> None:
    if not isinstance(rows, list):
        errors.append(RegistryError(path, location, "rows must be a list"))
        return

    row_ids: list[str] = []
    for idx, row in enumerate(rows):
        row_loc = f"{location}[{idx}]"
        if not _check_mapping(path, row_loc, row, GATE_ROW_REQUIRED, errors):
            if not isinstance(row, dict):
                continue

        row_id = row.get("row_id")
        if isinstance(row_id, str):
            row_ids.append(row_id)
            if not SNAKE_CASE.match(row_id):
                errors.append(RegistryError(path, f"{row_loc}.row_id", "row_id must be snake_case"))
        else:
            errors.append(RegistryError(path, f"{row_loc}.row_id", "row_id must be a string"))

        value_kind = row.get("value_kind")
        if value_kind not in VALUE_KINDS:
            errors.append(RegistryError(path, f"{row_loc}.value_kind", f"unsupported value_kind '{value_kind}'"))

        source_ref = row.get("source_ref")
        if isinstance(source_ref, dict):
            _check_citation_key(
                path,
                f"{row_loc}.source_ref.citation_key",
                source_ref.get("citation_key"),
                known_keys,
                errors,
            )
            _check_local_path(
                path,
                f"{row_loc}.source_ref.artifact",
                source_ref.get("artifact"),
                errors,
                required=False,
            )
        elif source_ref is not None:
            errors.append(RegistryError(path, f"{row_loc}.source_ref", "source_ref must be a mapping"))

        _check_local_path(
            path,
            f"{row_loc}.artifact_ref",
            row.get("artifact_ref"),
            errors,
            required=False,
        )

    _check_unique(path, f"{location}.row_id", row_ids, errors)


def _check_repo_verification(
    path: Path,
    location: str,
    repo_verification: dict[str, Any],
    row_ids: set[str],
    errors: list[RegistryError],
) -> tuple[set[str], set[str]]:
    reproduced = repo_verification.get("reproduced_rows")
    table_only = repo_verification.get("table_only_rows")
    if not isinstance(reproduced, list):
        errors.append(RegistryError(path, f"{location}.reproduced_rows", "must be a list"))
        reproduced = []
    if not isinstance(table_only, list):
        errors.append(RegistryError(path, f"{location}.table_only_rows", "must be a list"))
        table_only = []

    reproduced_set = set(reproduced)
    table_only_set = set(table_only)
    unknown = sorted((reproduced_set | table_only_set) - row_ids)
    for row_id in unknown:
        errors.append(RegistryError(path, location, f"coverage references unknown row_id '{row_id}'"))
    overlap = sorted(reproduced_set & table_only_set)
    for row_id in overlap:
        errors.append(RegistryError(path, location, f"row_id '{row_id}' cannot be both reproduced and table_only"))
    missing = sorted(row_ids - (reproduced_set | table_only_set))
    for row_id in missing:
        errors.append(RegistryError(path, location, f"published row_id '{row_id}' is not covered by repo_verification"))

    _check_local_path(path, f"{location}.test", repo_verification.get("test"), errors, required=False)
    _check_local_path(path, f"{location}.artifact", repo_verification.get("artifact"), errors, required=False)
    return reproduced_set, table_only_set


def _check_entry(
    path: Path,
    entry: Any,
    idx: int,
    top_problem: str,
    known_keys: set[str],
    errors: list[RegistryError],
) -> None:
    location = f"entries[{idx}]"
    if not _check_mapping(path, location, entry, ENTRY_REQUIRED, errors):
        if not isinstance(entry, dict):
            return

    entry_id = entry.get("id")
    if not isinstance(entry_id, str) or not SNAKE_CASE.match(entry_id):
        errors.append(RegistryError(path, f"{location}.id", "id must be snake_case"))
    if entry.get("problem") != top_problem:
        errors.append(RegistryError(path, f"{location}.problem", "entry problem must match top-level problem"))
    if not isinstance(entry.get("roles"), list) or not entry.get("roles"):
        errors.append(RegistryError(path, f"{location}.roles", "roles must be a non-empty list"))

    status = entry.get("status")
    if isinstance(status, dict):
        verification = status.get("verification")
        comparator_type = status.get("comparator_type")
        paper_status = status.get("paper_status")
        if verification not in VERIFICATION_STATUSES:
            errors.append(RegistryError(path, f"{location}.status.verification", f"unsupported status '{verification}'"))
        if comparator_type not in COMPARATOR_TYPES:
            errors.append(RegistryError(path, f"{location}.status.comparator_type", f"unsupported comparator_type '{comparator_type}'"))
        if paper_status not in PAPER_STATUSES:
            errors.append(RegistryError(path, f"{location}.status.paper_status", f"unsupported paper_status '{paper_status}'"))
    else:
        errors.append(RegistryError(path, f"{location}.status", "status must be a mapping"))
        verification = None

    source = entry.get("source")
    if isinstance(source, dict):
        _check_citation_keys(path, f"{location}.source.citation_keys", source.get("citation_keys"), known_keys, errors)
    else:
        errors.append(RegistryError(path, f"{location}.source", "source must be a mapping"))

    instance = entry.get("instance")
    if isinstance(instance, dict):
        _check_local_path(path, f"{location}.instance.reference_path", instance.get("reference_path"), errors)
    else:
        errors.append(RegistryError(path, f"{location}.instance", "instance must be a mapping"))

    published = entry.get("published_numbers")
    if isinstance(published, dict):
        if published.get("sign") not in {"lower_is_better", "higher_is_better"}:
            errors.append(RegistryError(path, f"{location}.published_numbers.sign", "unsupported sign"))
        row_ids = set(_check_rows(path, f"{location}.published_numbers.rows", published.get("rows"), known_keys, errors))
    else:
        errors.append(RegistryError(path, f"{location}.published_numbers", "published_numbers must be a mapping"))
        row_ids = set()

    repo_verification = entry.get("repo_verification")
    if isinstance(repo_verification, dict):
        reproduced, table_only = _check_repo_verification(
            path,
            f"{location}.repo_verification",
            repo_verification,
            row_ids,
            errors,
        )
    else:
        errors.append(RegistryError(path, f"{location}.repo_verification", "repo_verification must be a mapping"))
        reproduced, table_only = set(), set()

    if verification == "strict_literature_verified" and table_only:
        errors.append(RegistryError(path, f"{location}.status.verification", "strict entries cannot have table_only rows"))
    if verification == "table_only" and reproduced:
        errors.append(RegistryError(path, f"{location}.status.verification", "table_only entries cannot have reproduced rows"))
    if verification == "partial" and row_ids and (not reproduced or not table_only):
        errors.append(RegistryError(path, f"{location}.status.verification", "partial entries should have both reproduced and table_only rows"))
    if verification == "repo_native" and row_ids:
        errors.append(RegistryError(path, f"{location}.status.verification", "repo_native entries should keep published_numbers.rows empty"))

    gate = entry.get("repo_baseline_gate")
    if isinstance(gate, dict):
        _check_local_path(path, f"{location}.repo_baseline_gate.script", gate.get("script"), errors, required=False)
        _check_local_path(path, f"{location}.repo_baseline_gate.latest_report", gate.get("latest_report"), errors, required=False)
        gate_rows = gate.get("rows")
        if gate_rows is not None:
            _check_gate_rows(path, f"{location}.repo_baseline_gate.rows", gate_rows, known_keys, errors)
    else:
        errors.append(RegistryError(path, f"{location}.repo_baseline_gate", "repo_baseline_gate must be a mapping"))

    paper_link = entry.get("paper_link")
    if isinstance(paper_link, dict):
        _check_local_path(path, f"{location}.paper_link.section", paper_link.get("section"), errors, required=False)
    else:
        errors.append(RegistryError(path, f"{location}.paper_link", "paper_link must be a mapping"))


def check_registry(path: Path, known_keys: set[str]) -> list[RegistryError]:
    data, errors = _load_yaml(path)
    if errors:
        return errors
    if not _check_mapping(path, "$", data, TOP_LEVEL_REQUIRED, errors):
        if not isinstance(data, dict):
            return errors

    if data.get("schema_version") != 1:
        errors.append(RegistryError(path, "$.schema_version", "schema_version must be 1"))
    problem = data.get("problem")
    if not isinstance(problem, str) or not SNAKE_CASE.match(problem):
        errors.append(RegistryError(path, "$.problem", "problem must be snake_case"))
        problem = ""

    _check_local_path(path, "$.registry_owner", data.get("registry_owner"), errors)
    source_of_truth = data.get("source_of_truth")
    if not isinstance(source_of_truth, list) or not source_of_truth:
        errors.append(RegistryError(path, "$.source_of_truth", "source_of_truth must be a non-empty list"))
    else:
        for idx, source_path in enumerate(source_of_truth):
            _check_local_path(path, f"$.source_of_truth[{idx}]", source_path, errors)

    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append(RegistryError(path, "$.entries", "entries must be a non-empty list"))
        return errors

    entry_ids: list[str] = []
    for idx, entry in enumerate(entries):
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            entry_ids.append(entry["id"])
        _check_entry(path, entry, idx, problem, known_keys, errors)
    _check_unique(path, "$.entries.id", entry_ids, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=f"Registry YAML files to check. Defaults to {DEFAULT_GLOB}.",
    )
    args = parser.parse_args(argv)

    paths = args.paths or sorted(ROOT.glob(DEFAULT_GLOB))
    if not paths:
        print(f"No registry files found for {DEFAULT_GLOB}", file=sys.stderr)
        return 1

    known_keys = _bib_keys()
    errors: list[RegistryError] = []
    for path in paths:
        full_path = path if path.is_absolute() else ROOT / path
        errors.extend(check_registry(full_path, known_keys))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"FAILED: {len(errors)} registry issue(s) across {len(paths)} file(s)", file=sys.stderr)
        return 1

    print(f"OK: checked {len(paths)} baseline registry file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
