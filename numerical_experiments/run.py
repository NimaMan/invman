from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from numerical_experiments.catalog import get_suite, list_suites


def parse_args():
    parser = argparse.ArgumentParser(
        description="List and run the curated numerical-experiment suites for the inventory-management project."
    )
    parser.add_argument("--list", action="store_true", help="List available suites and exit.")
    parser.add_argument("--suite", action="append", default=[], help="Suite id to run. May be passed multiple times.")
    parser.add_argument(
        "--all-ready",
        action="store_true",
        help="Run all suites marked as ready in the catalog.",
    )
    parser.add_argument(
        "--status",
        choices=["ready", "exploratory"],
        default=None,
        help="Optional status filter when using --list.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to use when launching suite scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next suite if one suite fails.",
    )
    return parser.parse_args()


def _format_suite(suite) -> str:
    lines = [
        f"{suite.suite_id} [{suite.status}]",
        f"  problem: {suite.problem}",
        f"  purpose: {suite.purpose}",
        f"  heuristics: {', '.join(suite.heuristics) if suite.heuristics else '-'}",
        f"  base_policies: {', '.join(suite.base_policies) if suite.base_policies else '-'}",
        f"  improved_policies: {', '.join(suite.improved_policies) if suite.improved_policies else '-'}",
        f"  script: {suite.script_path} {' '.join(suite.script_args)}".rstrip(),
    ]
    for note in suite.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def _resolve_run_list(parsed) -> list:
    suites = []
    if parsed.all_ready:
        suites.extend(list_suites(status="ready"))
    for suite_id in parsed.suite:
        suites.append(get_suite(suite_id))
    # preserve order but remove duplicates
    deduped = []
    seen = set()
    for suite in suites:
        if suite.suite_id in seen:
            continue
        seen.add(suite.suite_id)
        deduped.append(suite)
    return deduped


def main():
    parsed = parse_args()

    if parsed.list or (not parsed.suite and not parsed.all_ready):
        for suite in list_suites(status=parsed.status):
            print(_format_suite(suite))
            print()
        return

    suites = _resolve_run_list(parsed)
    if not suites:
        raise SystemExit("No suites selected. Use --suite <id> or --all-ready.")

    failures = []
    for suite in suites:
        command = suite.command(PROJECT_ROOT, parsed.python_bin)
        rendered = " ".join(shlex.quote(part) for part in command)
        print(f"[suite] {suite.suite_id}")
        print(rendered)
        if parsed.dry_run:
            continue
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            failures.append((suite.suite_id, result.returncode))
            if not parsed.continue_on_error:
                raise SystemExit(result.returncode)

    if failures:
        details = ", ".join(f"{suite_id}:{code}" for suite_id, code in failures)
        raise SystemExit(f"One or more suites failed: {details}")


if __name__ == "__main__":
    main()
