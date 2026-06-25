#!/usr/bin/env python3
"""Validate per-problem machine-readable instance catalogs.

The catalog convention is intentionally light-weight: every problem family can
own an `instances/` directory with one `README.md` and one JSON file per problem
instance. This validator checks the cross-family contract without forcing a
single domain-specific schema onto all inventory models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CLASSIFICATIONS = {
    "strict_literature",
    "companion_code",
    "table_only",
    "faithful_unverified",
    "generated",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"{path}: top-level JSON value must be an object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_instance_file(path: Path) -> None:
    payload = _load_json(path)
    expected_id = path.stem

    _require(payload.get("schema_version") == 1, f"{path}: schema_version must be 1")
    _require(payload.get("instance_id") == expected_id, f"{path}: instance_id must match filename")
    _require(isinstance(payload.get("problem_family"), str), f"{path}: missing problem_family")

    classification = payload.get("classification")
    _require(
        classification in CLASSIFICATIONS,
        f"{path}: classification must be one of {sorted(CLASSIFICATIONS)}",
    )

    source = payload.get("source") or payload.get("provenance")
    _require(isinstance(source, (dict, list)), f"{path}: missing source/provenance object")

    has_model = any(key in payload for key in ("parameters", "model", "network"))
    _require(has_model, f"{path}: missing parameters/model/network block")
    _require(isinstance(payload.get("verification"), dict), f"{path}: missing verification object")


def iter_instance_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("src/problems/**/instances") if path.is_dir())


def validate_catalog(root: Path) -> tuple[int, int]:
    instance_dirs = iter_instance_dirs(root)
    for directory in instance_dirs:
        _require((directory / "README.md").is_file(), f"{directory}: missing README.md")
        json_files = sorted(directory.glob("*.json"))
        _require(json_files, f"{directory}: no JSON instance files found")
        seen: set[str] = set()
        for path in json_files:
            validate_instance_file(path)
            instance_id = path.stem
            _require(instance_id not in seen, f"{directory}: duplicate instance_id {instance_id}")
            seen.add(instance_id)
    return len(instance_dirs), sum(1 for directory in instance_dirs for _ in directory.glob("*.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    directories, files = validate_catalog(args.root)
    print(f"validated {files} instance files in {directories} instance directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
