from pathlib import Path

from setuptools import find_packages, setup


HERE = Path(__file__).resolve().parent


def read_requirements() -> list[str]:
    requirements_path = HERE / "requirements.txt"
    return [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name="invman",
    version="0.2.0",
    description="Inventory-management experiments with evolution strategies",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.9",
    install_requires=read_requirements(),
)
