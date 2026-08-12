from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from verify_artifact_manifest import file_sha256, verify_manifest  # noqa: E402


class ArtifactManifestTests(unittest.TestCase):
    def test_all_published_artifacts_match_manifest(self) -> None:
        entries, failures = verify_manifest(ROOT, ROOT / "ARTIFACTS.sha256")
        self.assertGreater(entries, 250)
        self.assertEqual([], failures)

    def test_unmanifested_json_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            listed = artifacts / "listed.json"
            listed.write_text("{}\n", encoding="utf-8")
            orphan = artifacts / "orphan.json"
            orphan.write_text("{}\n", encoding="utf-8")
            manifest = root / "ARTIFACTS.sha256"
            manifest.write_text(
                f"{file_sha256(listed)}  artifacts/listed.json\n",
                encoding="utf-8",
            )

            entries, failures = verify_manifest(root, manifest)
            self.assertEqual(entries, 1)
            self.assertEqual(
                failures,
                ["unmanifested artifact: artifacts/orphan.json"],
            )


if __name__ == "__main__":
    unittest.main()
