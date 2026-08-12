"""Verify every current artifact against ``ARTIFACTS.sha256``."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path, manifest: Path) -> tuple[int, list[str]]:
    root = root.resolve()
    failures: list[str] = []
    seen: set[str] = set()
    entries = 0

    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        match = MANIFEST_LINE.fullmatch(line)
        if match is None:
            failures.append(f"line {line_number}: malformed manifest entry")
            continue

        expected, relative = match.groups()
        entries += 1
        if relative in seen:
            failures.append(f"line {line_number}: duplicate path: {relative}")
            continue
        seen.add(relative)

        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            failures.append(f"line {line_number}: path escapes repository: {relative}")
        elif not target.is_file():
            failures.append(f"line {line_number}: missing artifact: {relative}")
        else:
            actual = file_sha256(target)
            if actual != expected:
                failures.append(
                    f"line {line_number}: hash mismatch: {relative} "
                    f"(expected {expected}, got {actual})"
                )

    artifact_directory = root / "artifacts"
    if artifact_directory.is_dir():
        published = {
            path.relative_to(root).as_posix()
            for path in artifact_directory.glob("*.json")
            if path.is_file()
        }
        unmanifested = sorted(published - seen)
        for relative in unmanifested:
            failures.append(f"unmanifested artifact: {relative}")

    return entries, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = args.manifest or root / "ARTIFACTS.sha256"
    entries, failures = verify_manifest(root, manifest)

    for failure in failures:
        print(failure)
    if failures:
        print(f"FAILED: {len(failures)} failure(s) across {entries} entries")
        return 1
    print(f"PASS: all {entries} artifact hashes match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
