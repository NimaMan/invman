import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".py", ".rs", ".tex", ".toml"}
ACTIVE_SURFACES = (
    "README.md",
    "AGENTS.md",
    "setup.py",
    "Cargo.toml",
    "docs",
    "invman",
    "policy_search",
    "scripts",
    "numerical_experiments",
    "src",
    "paper",
)


def _active_text_files():
    for surface in ACTIVE_SURFACES:
        root = PROJECT_ROOT / surface
        if root.is_file():
            yield root
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                yield path


def _regex_matches(pattern):
    regex = re.compile(pattern, re.MULTILINE)
    matches = []
    for path in _active_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if regex.search(text):
            matches.append(str(path.relative_to(PROJECT_ROOT)))
    return matches


def test_root_rust_crate_layout_is_current_source_of_truth():
    assert (PROJECT_ROOT / "Cargo.toml").is_file()
    assert (PROJECT_ROOT / "Cargo.lock").is_file()
    assert (PROJECT_ROOT / "src" / "lib.rs").is_file()
    assert (PROJECT_ROOT / "src" / "problems" / "lost_sales").is_dir()
    assert (PROJECT_ROOT / "setup.py").is_file()
    assert not (PROJECT_ROOT / "bindings").exists()

    root_pyproject = PROJECT_ROOT / "pyproject.toml"
    if root_pyproject.exists():
        root_pyproject_text = root_pyproject.read_text(encoding="utf-8")
        assert 'build-backend = "maturin"' not in root_pyproject_text
        assert 'name = "invman_rust"' not in root_pyproject_text

    assert not (PROJECT_ROOT / "rust" / "Cargo.toml").exists()
    assert not (PROJECT_ROOT / "rust" / "Cargo.lock").exists()
    assert not (PROJECT_ROOT / "rust" / "src").exists()
    assert not (PROJECT_ROOT / "rust" / "pyproject.toml").exists()


def test_cargo_features_keep_rust_tests_separate_from_python_extension_builds():
    manifest = (PROJECT_ROOT / "Cargo.toml").read_text(encoding="utf-8")

    assert 'crate-type = ["cdylib", "rlib"]' in manifest
    assert "default = []" in manifest
    assert 'python-extension = ["pyo3/extension-module"]' in manifest


def test_build_script_uses_root_manifest_and_explicit_python_extension_feature():
    source = (
        PROJECT_ROOT / "scripts" / "rust" / "build_extension.py"
    ).read_text(encoding="utf-8")

    assert 'manifest_path = project_root / "Cargo.toml"' in source
    assert 'manifest_path = project_root / "rust" / "Cargo.toml"' not in source
    assert '"--manifest-path"' in source
    assert '"python-extension"' in source


def test_active_surfaces_do_not_reference_deleted_nested_rust_crate_paths():
    matches = _regex_matches(
        r"rust/src|rust/Cargo|--manifest-path rust|"
        r"PACKAGE_ROOT\s*/\s*[\"']rust[\"']|Path\([^\n]*[\"']rust[\"']"
    )

    assert matches == []


def test_active_surfaces_do_not_import_deleted_python_problem_packages():
    matches = _regex_matches(r"^\s*(?:from|import)\s+invman\.(?:problems|policies)\b")

    assert matches == []
