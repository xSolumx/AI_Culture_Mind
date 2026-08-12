from __future__ import annotations

import hashlib
import json
import unittest

from spin9_v1_v5_global import (
    DEFAULT_COEFFICIENT_ARTIFACT,
    ROOT,
    _compact_handoff,
    _sparse_digest,
    _validate_atlas_partition,
    chart_gap_coefficients,
)

ARTIFACT = ROOT / "artifacts" / "spin9_v1_v5_global_20260812.json"


class Spin9V1V5GlobalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_global_cover_is_exhaustive(self) -> None:
        required: set[str] = set()
        for scalar_sign in (1, -1):
            compact = self.report["compact_reports"][str(scalar_sign)]
            _validate_atlas_partition(compact["atlas"], allow_unresolved=True)
            recomputed = [
                _compact_handoff(scalar_sign, cell["box"])
                for cell in compact["atlas"]["unresolved"]
            ]
            self.assertEqual(recomputed, compact["local_handoffs"])
            self.assertTrue(all(row["passed"] for row in recomputed))
            for row in recomputed:
                for d_sign in row["d_signs"]:
                    required.add(f"{row['family']}:core:{d_sign}")
                    required.add(
                        f"{row['family']}:r_infinity_q_low:{d_sign}"
                    )
        self.assertEqual(required, set(self.report["required_local_charts"]))
        for key in required:
            local = self.report["local_reports"][key]
            _validate_atlas_partition(local["atlas"], allow_unresolved=False)
            self.assertTrue(local["passed"])

    def test_sparse_gap_digests_replay_from_coefficients(self) -> None:
        for family in ("A", "B"):
            _, _, gap = chart_gap_coefficients(
                family,
                DEFAULT_COEFFICIENT_ARTIFACT,
            )
            expected = _sparse_digest(gap)
            observed = {
                report["gap_rows_sha256"]
                for key, report in self.report["local_reports"].items()
                if key.startswith(f"{family}:")
            }
            self.assertEqual(observed, {expected})

    def test_promoted_scope_and_raw_identity_boundary(self) -> None:
        coefficient_hash = hashlib.sha256(
            DEFAULT_COEFFICIENT_ARTIFACT.read_bytes()
        ).hexdigest()
        self.assertEqual(
            self.report["coefficient_artifact_sha256"],
            coefficient_hash,
        )
        self.assertTrue(self.report["passed"])
        self.assertTrue(
            self.report["reconstructed_rational_function_bound_certified"]
        )
        self.assertFalse(
            self.report["raw_characteristic_zero_determinant_identity_certified"]
        )
        self.assertFalse(self.report["global_determinant_theorem_claimed"])
        self.assertEqual(
            self.report["theorem"],
            "N(x,p,y)/delta(x,p,y)^14 <= 21/20",
        )


if __name__ == "__main__":
    unittest.main()
