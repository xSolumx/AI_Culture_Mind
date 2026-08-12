from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aggregate_fla_delta_rule_benchmarks import aggregate as aggregate_fla
from aggregate_matched_memory_core_benchmarks import aggregate as aggregate_local
from select_fla_delta_rule_implementations import select as select_fla
from select_matched_memory_core_implementations import select as select_local


class BenchmarkReplicationContractTests(unittest.TestCase):
    def test_duplicate_contents_cannot_masquerade_as_independent_processes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text("{}\n", encoding="utf-8")
            for operation in (aggregate_local, aggregate_fla, select_local, select_fla):
                with self.subTest(
                    operation=operation.__module__
                ), self.assertRaisesRegex(ValueError, "distinct contents"):
                    operation([path, path])


if __name__ == "__main__":
    unittest.main()
