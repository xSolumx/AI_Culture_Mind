"""Regression test for the compact Cayley-spectrum referee package."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CayleyRefereePackageTests(unittest.TestCase):
    def test_independent_exact_verifier(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        verifier = repository / "referee" / "cayley-information-spectrum" / "verify.py"
        completed = subprocess.run(
            [sys.executable, str(verifier)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PASS: independent exact", completed.stdout)

    def test_stored_pass_flag_is_not_trusted(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        package = repository / "referee" / "cayley-information-spectrum"
        stored = (package / "artifacts" / "certificate.json").read_text(
            encoding="utf-8"
        )
        tampered = stored.replace('"passed": true', '"passed": false')
        self.assertNotEqual(tampered, stored)

        with tempfile.TemporaryDirectory() as directory:
            certificate = Path(directory) / "tampered.json"
            certificate.write_text(tampered, encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(package / "verify.py"),
                    "--certificate",
                    str(certificate),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("stored certificate differs", completed.stderr)


if __name__ == "__main__":
    unittest.main()
