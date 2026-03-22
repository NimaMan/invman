import os
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    venv_python = project_root.parent / ".venv" / "bin" / "python"
    maturin_bin = project_root.parent / ".venv" / "bin" / "maturin"
    manifest_path = project_root / "rust" / "Cargo.toml"

    if not venv_python.exists():
        raise SystemExit(f"missing Python interpreter: {venv_python}")
    if not maturin_bin.exists():
        raise SystemExit(f"missing maturin binary: {maturin_bin}")
    if not manifest_path.exists():
        raise SystemExit(f"missing Cargo manifest: {manifest_path}")

    cmd = [
        str(maturin_bin),
        "develop",
        "--locked",
        "--manifest-path",
        str(manifest_path),
    ]
    env = os.environ.copy()
    env["PYO3_PYTHON"] = str(venv_python)
    subprocess.run(cmd, check=True, cwd=project_root.parent, env=env)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
