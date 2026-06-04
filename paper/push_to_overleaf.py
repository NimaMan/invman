#!/usr/bin/env python3
"""Upload the manuscript (learning_inventory_control_policies_es.tex) to the Overleaf revision project.

Authentication is delegated to the local Overleaf session helper in
/home/nima/code/tools/security/access/login_sessions/overleaf. That helper reads
the stored overleaf_session2 cookie or the OVERLEAF_SESSION_COOKIE environment
variable; this script never prints the cookie value.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT = "invman_paper (revision)"
DEFAULT_LOCAL_TEX = PAPER_DIR / "learning_inventory_control_policies_es.tex"
DEFAULT_REMOTE_TEX = "learning_inventory_control_policies_es.tex"
DEFAULT_HELPER_DIR = Path(
    os.environ.get(
        "OVERLEAF_HELPER_DIR",
        "/home/nima/code/tools/security/access/login_sessions/overleaf",
    )
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_overleaf_helpers(helper_dir: Path):
    if not helper_dir.exists():
        raise SystemExit(
            f"Overleaf helper directory not found: {helper_dir}\n"
            "Set OVERLEAF_HELPER_DIR to the directory containing "
            "overleaf_file_manager.py."
        )

    sys.path.insert(0, str(helper_dir))
    try:
        import overleaf_file_manager as overleaf_helpers
        from pyoverleaf._io import ProjectIO
    except ImportError as exc:
        raise SystemExit(
            "Could not import pyoverleaf helpers. Install pyoverleaf and "
            "browser-cookie3, or use the existing tools environment."
        ) from exc
    return overleaf_helpers, ProjectIO


def _find_project(api, project_name: str):
    projects = api.get_projects()
    matches = [project for project in projects if project.name == project_name]
    if matches:
        return matches[0]

    casefold_matches = [
        project for project in projects if project.name.casefold() == project_name.casefold()
    ]
    if casefold_matches:
        return casefold_matches[0]

    available = "\n".join(f"  - {project.name}" for project in sorted(projects, key=lambda p: p.name.lower()))
    raise SystemExit(f"No exact Overleaf project named {project_name!r}.\nAvailable projects:\n{available}")


def _read_remote(io, remote_path: str) -> str | None:
    if not io.exists(remote_path):
        return None
    with io.open(remote_path, "r", encoding="utf-8") as fh:
        return fh.read()


def _write_remote(io, remote_path: str, content: str) -> None:
    with io.open(remote_path, "w", encoding="utf-8") as fh:
        fh.write(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push paper/learning_inventory_control_policies_es.tex to Overleaf."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"Overleaf project name. Default: {DEFAULT_PROJECT!r}")
    parser.add_argument(
        "--local-tex",
        type=Path,
        default=DEFAULT_LOCAL_TEX,
        help=f"Local TeX file. Default: {DEFAULT_LOCAL_TEX}",
    )
    parser.add_argument(
        "--remote-tex",
        default=DEFAULT_REMOTE_TEX,
        help=f"Remote Overleaf path. Default: {DEFAULT_REMOTE_TEX!r}",
    )
    parser.add_argument(
        "--helper-dir",
        type=Path,
        default=DEFAULT_HELPER_DIR,
        help=f"Directory containing overleaf_file_manager.py. Default: {DEFAULT_HELPER_DIR}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compare hashes without writing to Overleaf.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    local_path = args.local_tex.resolve()
    if not local_path.exists():
        raise SystemExit(f"Local TeX file not found: {local_path}")

    local_content = local_path.read_text(encoding="utf-8")
    overleaf_helpers, ProjectIO = _load_overleaf_helpers(args.helper_dir.resolve())
    api = overleaf_helpers.get_api()
    project = _find_project(api, args.project)
    io = ProjectIO(api, project.id)

    remote_before = _read_remote(io, args.remote_tex)
    local_hash = _sha256(local_content)
    remote_before_hash = _sha256(remote_before) if remote_before is not None else "<missing>"

    print(f"Project: {project.name}")
    print(f"Local:   {local_path}")
    print(f"Remote:  {args.remote_tex}")
    print(f"Local sha256:        {local_hash}")
    print(f"Remote before sha256:{remote_before_hash}")

    if remote_before == local_content:
        print("[OK] Overleaf already has the latest local file.")
        return 0

    if args.dry_run:
        print("[DRY-RUN] Remote differs; no write performed.")
        return 0

    _write_remote(io, args.remote_tex, local_content)
    remote_after = _read_remote(io, args.remote_tex)
    remote_after_hash = _sha256(remote_after or "")
    print(f"Remote after sha256: {remote_after_hash}")
    if remote_after != local_content:
        raise SystemExit("[ERROR] Remote verification failed after write.")

    print("[OK] Uploaded and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
