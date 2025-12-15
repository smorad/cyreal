from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="cereal",
    version="0.1.0",
    description="Jittable data loading utilities for JAX.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "tests.*", "*.tests")),
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
