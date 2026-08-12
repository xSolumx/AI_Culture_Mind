import json
import tempfile
import unittest
from pathlib import Path

from aggregate_hierarchical_matched_retrieval import aggregate


class AggregateHierarchicalRetrievalTests(unittest.TestCase):
    def test_rejects_duplicate_seed_processes_before_summary(self) -> None:
        report = {
            "experiment": "test",
            "protocol": "HIERARCHICAL_MATCHED_RETRIEVAL_PREREGISTRATION.md",
            "seeds": [0],
            "grid": {},
            "results": [{"seed": 0}],
            "summary": {"implementation_gate_passed": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"row{index}.json" for index in range(2)]
            for path in paths:
                path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                aggregate(paths, [0, 1])


if __name__ == "__main__":
    unittest.main()
