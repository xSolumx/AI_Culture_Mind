"""Audit repository Markdown for mechanical mathematical-writing defects."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


DEPRECATED_PROMOTIONAL_TERM = "break" + "through"
STALE_NOTE_NAME = "BREAK" + "THROUGH_NOTE_2026-08-06.md"
MALFORMED_PATTERNS = {
    "double-dollar display delimiter": re.compile(r"\$\$"),
    "malformed generated subscript": re.compile(r"(?:_|[A-Za-z])\*\{"),
    "malformed norm command": re.compile(r"\\Vert\{"),
    "malformed exponent spacing": re.compile(r"\^\{,"),
    "missing backslash on spacing command": re.compile(
        r"(?<!\\)\b(?:quad|qquad)\b"
    ),
}
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
CODE_FENCE = re.compile(r"^\s*```", re.MULTILINE)


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str


def markdown_files(root: Path) -> list[Path]:
    excluded = {".git", ".venv", "node_modules", "__pycache__"}
    return sorted(
        path
        for path in root.rglob("*.md")
        if not any(part in excluded or part.startswith(".venv") for part in path.parts)
    )


def audit_file(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    findings: list[Finding] = []

    if DEPRECATED_PROMOTIONAL_TERM in path.name.casefold():
        findings.append(Finding(path, 0, "deprecated promotional term in filename"))

    for line_number, line in enumerate(text.splitlines(), start=1):
        folded = line.casefold()
        if DEPRECATED_PROMOTIONAL_TERM in folded:
            findings.append(
                Finding(path, line_number, "deprecated promotional terminology")
            )
        if STALE_NOTE_NAME.casefold() in folded:
            findings.append(Finding(path, line_number, "stale renamed-note path"))
        for label, pattern in MALFORMED_PATTERNS.items():
            if pattern.search(line):
                findings.append(Finding(path, line_number, label))

    if text.count(r"\[") != text.count(r"\]"):
        findings.append(Finding(path, 0, "unbalanced display-math delimiters"))
    if text.count(r"\(") != text.count(r"\)"):
        findings.append(Finding(path, 0, "unbalanced inline-math delimiters"))
    if len(CODE_FENCE.findall(text)) % 2:
        findings.append(Finding(path, 0, "unbalanced fenced-code delimiters"))

    for line_number, line in enumerate(text.splitlines(), start=1):
        for raw_target in MARKDOWN_LINK.findall(line):
            target = unquote(raw_target.strip().split()[0].strip("<>"))
            if target.startswith(("http://", "https://", "mailto:", "#", "/")):
                continue
            if re.match(r"^[A-Za-z]:", target):
                continue
            target = target.split("#", 1)[0]
            if target and not (path.parent / target).resolve().exists():
                findings.append(Finding(path, line_number, f"broken local link: {target}"))

    return findings


def audit_roots(roots: list[Path]) -> tuple[int, list[Finding]]:
    files: list[Path] = []
    for root in roots:
        files.extend(markdown_files(root.resolve()))
    findings = [finding for path in files for finding in audit_file(path)]
    return len(files), findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path(__file__).resolve().parents[1]],
        help="repository roots to audit (default: this repository)",
    )
    args = parser.parse_args()
    file_count, findings = audit_roots(args.roots)

    for finding in findings:
        location = str(finding.path)
        if finding.line:
            location += f":{finding.line}"
        print(f"{location}: {finding.message}")

    if findings:
        print(f"FAILED: {len(findings)} finding(s) in {file_count} Markdown files")
        return 1

    print(f"PASS: {file_count} Markdown files satisfy the mechanical writing contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
