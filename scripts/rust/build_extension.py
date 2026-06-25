import os
import shutil
import subprocess
import sys
from pathlib import Path


def _candidate_paths(project_root: Path, *parts: str) -> list[Path]:
    return [
        project_root.joinpath(*parts),
        project_root.parent.joinpath(*parts),
    ]


def _resolve_python(project_root: Path) -> Path:
    candidates = []
    if os.environ.get("PYO3_PYTHON"):
        candidates.append(Path(os.environ["PYO3_PYTHON"]))
    candidates.append(Path(sys.executable))
    candidates.extend(_candidate_paths(project_root, ".venv", "bin", "python"))
    candidates.extend(_candidate_paths(project_root, ".venv", "Scripts", "python.exe"))

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    raise SystemExit(
        "could not find a Python interpreter for PyO3; activate a virtualenv first or set PYO3_PYTHON"
    )


def _resolve_maturin(project_root: Path, python_executable: Path) -> list[str]:
    maturin_on_path = shutil.which("maturin")
    if maturin_on_path:
        return [maturin_on_path]

    for candidate in _candidate_paths(project_root, ".venv", "bin", "maturin"):
        if candidate.exists():
            return [str(candidate)]
    for candidate in _candidate_paths(project_root, ".venv", "Scripts", "maturin.exe"):
        if candidate.exists():
            return [str(candidate)]

    probe = subprocess.run(
        [str(python_executable), "-m", "maturin", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode == 0:
        return [str(python_executable), "-m", "maturin"]

    raise SystemExit(
        "could not find maturin; install it in the active environment with `python -m pip install maturin`"
    )


def main():
    project_root = Path(__file__).resolve().parents[2]
    manifest_path = project_root / "Cargo.toml"
    if not manifest_path.exists():
        raise SystemExit(f"missing Cargo manifest: {manifest_path}")

    python_executable = _resolve_python(project_root)
    maturin_cmd = _resolve_maturin(project_root, python_executable)

    cmd = [
        *maturin_cmd,
        "develop",
        "--locked",
        "--manifest-path",
        str(manifest_path),
        "--features",
        "python-extension",
    ]
    env = os.environ.copy()
    env["PYO3_PYTHON"] = str(python_executable)
    subprocess.run(cmd, check=True, cwd=project_root, env=env)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
