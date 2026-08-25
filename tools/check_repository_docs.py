"""Validate maintained repository documentation without rewriting evidence.

Research author: Hayden Austin.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = Path("SSM-Models/hybrid_memory_v1_4")
SKIP_PARTS = {
    ".git",
    ".private",
    ".pytest_cache",
    ".ruff_cache",
    ".playwright-cli",
    "node_modules",
}

MAINTAINED_AUTHOR_DOCS = (
    Path("README.md"),
    Path("REPOSITORY_MAP.md"),
    Path("PUBLICATION_SCOPE.md"),
    Path("AUTHORSHIP.md"),
    Path("CITATION.md"),
    Path("CONTRIBUTING.md"),
    Path("research-programs/README.md"),
    Path("research-programs/SUPPORTING_TRACKS.md"),
    Path("SSM-Models/README.md"),
    Path("SSM-Models/MODEL_STATUS.md"),
    Path("Spin-Space-Research/README.md"),
    Path("Spin-Space-Research/docs/README.md"),
    Path("Spin-Space-Research/docs/RESEARCH_MAP.md"),
    Path("Spin-Space-Research/docs/EXPERIMENT_INDEX.md"),
    Path("Spin8-SSM-Benchmark/README.md"),
    Path("SpinorModel/README.md"),
)

LICENSE_METADATA = {
    Path("CITATION.cff"): "license: Apache-2.0",
    Path("Spin-Space-Research/CITATION.cff"): "license: Apache-2.0",
    Path("Spin-Space-Research/pyproject.toml"): 'license = "Apache-2.0"',
    Path("Spin-Space-Research/NOTICE"): "Apache-2.0",
}
APACHE_2_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"

INLINE_LINK = re.compile(
    r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)"
)
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(?P<target><[^>]+>|\S+)")
IGNORED_SCHEMES = {
    "http",
    "https",
    "mailto",
    "data",
    "javascript",
    "chatgpt-conversation",
}


def markdown_files(include_protected: bool) -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.md"):
        relative = path.relative_to(ROOT)
        if any(
            part in SKIP_PARTS or part.startswith(".venv") or part in {"venv", "env"}
            for part in relative.parts
        ):
            continue
        if not include_protected and relative.is_relative_to(PROTECTED):
            continue
        files.append(path)
    return sorted(files)


def local_target(raw: str) -> str | None:
    target = raw.strip().strip("<>")
    if target.startswith("#"):
        return None
    parsed = urlparse(target)
    if parsed.scheme.lower() in IGNORED_SCHEMES:
        return None
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    return target or None


def check_links(files: list[Path]) -> list[str]:
    failures: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            matches = list(INLINE_LINK.finditer(line))
            reference = REFERENCE_LINK.match(line)
            if reference:
                matches.append(reference)
            for match in matches:
                target = local_target(match.group("target"))
                if target is None:
                    continue
                candidate = Path(target)
                if candidate.is_absolute():
                    failures.append(
                        f"{relative}:{line_number}: absolute local link: {target}"
                    )
                    continue
                resolved = (path.parent / candidate).resolve()
                try:
                    resolved.relative_to(ROOT)
                except ValueError:
                    failures.append(
                        f"{relative}:{line_number}: link escapes repository: {target}"
                    )
                    continue
                if not resolved.exists():
                    failures.append(
                        f"{relative}:{line_number}: missing local target: {target}"
                    )
    return failures


def check_authorship(files: list[Path]) -> list[str]:
    failures: list[str] = []
    required = set(MAINTAINED_AUTHOR_DOCS)
    required.update(
        path.relative_to(ROOT)
        for path in files
        if path.name.casefold() == "readme.md"
        and not path.relative_to(ROOT).is_relative_to(PROTECTED)
    )
    for relative in sorted(required):
        path = ROOT / relative
        if not path.exists():
            failures.append(f"missing maintained document: {relative}")
            continue
        if "Hayden Austin" not in path.read_text(encoding="utf-8"):
            failures.append(f"maintained document lacks research author: {relative}")
    cff = ROOT / "CITATION.cff"
    if not cff.exists() or not {"Hayden", "Austin"}.issubset(
        set(re.findall(r"[A-Za-z]+", cff.read_text(encoding="utf-8")))
    ):
        failures.append("CITATION.cff lacks Hayden Austin")
    return failures


def check_licensing() -> list[str]:
    failures: list[str] = []
    root_license = ROOT / "LICENSE"
    component_license = ROOT / "Spin-Space-Research/LICENSE"
    for path in (root_license, component_license):
        if not path.exists():
            failures.append(f"missing licence file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        if "Apache License" not in text or "Version 2.0, January 2004" not in text:
            failures.append(f"non-Apache-2.0 licence text: {path.relative_to(ROOT)}")
        normalized = text.replace("\r\n", "\n").encode("utf-8")
        if hashlib.sha256(normalized).hexdigest() != APACHE_2_SHA256:
            failures.append(f"modified Apache-2.0 text: {path.relative_to(ROOT)}")
    if (
        root_license.exists()
        and component_license.exists()
        and root_license.read_bytes() != component_license.read_bytes()
    ):
        failures.append("root and Spin-Space Apache licence texts differ")
    for relative, marker in LICENSE_METADATA.items():
        path = ROOT / relative
        if not path.exists() or marker not in path.read_text(encoding="utf-8"):
            failures.append(f"Apache-2.0 metadata missing: {relative}")
    notice = ROOT / "NOTICE"
    if not notice.exists() or "Apache-2.0" not in notice.read_text(encoding="utf-8"):
        failures.append("NOTICE lacks Apache-2.0 attribution boundary")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-protected",
        action="store_true",
        help="also scan the independently active hybrid_memory_v1_4 workspace",
    )
    args = parser.parse_args()

    files = markdown_files(args.include_protected)
    failures = check_links(files) + check_authorship(files) + check_licensing()
    if failures:
        print(f"documentation check failed ({len(failures)} issue(s))")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(
        f"documentation check passed: {len(files)} Markdown files, "
        "all maintained author and Apache-2.0 surfaces"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
