"""Dynamic module discovery for the flat research harness collection."""

from pathlib import Path

from setuptools import setup

setup(
    py_modules=sorted(
        path.stem for path in Path("src").glob("*.py") if path.stem.isidentifier()
    )
)
