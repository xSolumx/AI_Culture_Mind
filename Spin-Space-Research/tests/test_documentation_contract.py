from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from audit_math_docs import audit_roots  # noqa: E402


class DocumentationContractTests(unittest.TestCase):
    def test_markdown_mechanical_contract(self) -> None:
        file_count, findings = audit_roots([ROOT])
        self.assertGreater(file_count, 100)
        self.assertEqual([], findings)


if __name__ == "__main__":
    unittest.main()
