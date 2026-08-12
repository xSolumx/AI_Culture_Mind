"""Focused regression tests for the global endpoint-octet cubic theorem."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from flint import fmpz_mpoly_ctx

from spin8_dirac_endpoint_octet_cubic_00010_atlas import (
    verify_report as verify_cubic_00010,
)
from spin8_dirac_endpoint_octet_cubic_atlas import _check_stored_bernstein_audit
from spin8_dirac_endpoint_octet_cubic_blowup import _batched_bernstein_audit
from spin8_dirac_endpoint_octet_cubic_certificate import (
    verify_report as verify_cubic_certificate,
)
from spin8_dirac_endpoint_octet_cubic_coarse_atlas import (
    verify_nested_00001_report,
)
from spin8_dirac_endpoint_octet_cubic_coarse_atlas import (
    verify_report as verify_cubic_coarse,
)
from spin8_dirac_endpoint_octet_cubic_corner_certificate import (
    verify as verify_cubic_corner,
)
from spin8_dirac_endpoint_octet_cubic_tangent import (
    verify_report as verify_cubic_tangent,
)
from spin8_dirac_endpoint_octet_cubic_yzero import verify_report as verify_cubic_yzero
from spin8_dirac_endpoint_octet_quadratic import _native_bernstein_audit

ROOT = Path(__file__).resolve().parents[1]


class EndpointOctetCubicTests(unittest.TestCase):
    def test_compact_atlas_verifier_rejects_inconsistent_summaries(self) -> None:
        audit = {
            "multidegree": [1, 2],
            "tensor_shape": [2, 3],
            "coefficient_count": 6,
            "axis_positive_scales": [1, 2],
            "minimum_scaled_coefficient": "7",
            "negative_scaled_coefficient_count": 0,
            "zero_scaled_coefficient_count": 0,
            "negative_rows_sample": [],
        }
        failures: list[str] = []
        _check_stored_bernstein_audit(audit, label="valid", failures=failures)
        self.assertEqual(failures, [])

        audit["coefficient_count"] = 5
        failures = []
        _check_stored_bernstein_audit(audit, label="tampered", failures=failures)
        self.assertIn(
            "tampered coefficient count disagrees with its tensor shape", failures
        )

    def test_batched_bernstein_audit_matches_full_tensor_engine(self) -> None:
        context = fmpz_mpoly_ctx.get(["x", "y", "z"])
        x, y, z = context.gens()
        polynomial = (
            11 - 7 * x + 5 * y**2 + 13 * x * y * z - 3 * x**3 * z**2 + 17 * y * z**3
        )
        native = _native_bernstein_audit(polynomial, sample_limit=32)
        batched = _batched_bernstein_audit(
            polynomial, sample_limit=32, batch_entry_limit=7
        )
        for key in (
            "multidegree",
            "tensor_shape",
            "coefficient_count",
            "axis_positive_scales",
            "minimum_scaled_coefficient",
            "minimum_bernstein_index",
            "negative_scaled_coefficient_count",
            "zero_scaled_coefficient_count",
            "negative_boundary_histogram",
            "negative_rows_sample",
        ):
            self.assertEqual(batched[key], native[key])

    def test_complete_cubic_certificate_recomputes_final_identity(self) -> None:
        coefficient_dir = (
            ROOT / "artifacts" / "spin8_dirac_unrestricted_coefficients_20260807"
        )
        artifact = (
            ROOT
            / "artifacts"
            / "spin8_dirac_endpoint_octet_cubic_certificate_20260811.json"
        )
        verification = verify_cubic_certificate(
            artifact, coefficient_dir=coefficient_dir
        )
        self.assertIs(verification["verified"], True)
        self.assertIs(verification["recomputed_characteristic_zero_identity"], True)

    def test_yzero_artifact_replays_load_bearing_algebra(self) -> None:
        artifact = (
            ROOT / "artifacts" / "spin8_dirac_endpoint_octet_cubic_yzero_20260808.json"
        )
        report = json.loads(artifact.read_text(encoding="utf-8"))
        verification = verify_cubic_yzero(report)
        self.assertIs(verification["verified"], True)

    def test_tangent_artifact_replays_radical_factor_proof(self) -> None:
        artifact = (
            ROOT
            / "artifacts"
            / "spin8_dirac_endpoint_octet_cubic_tangent_20260810.json"
        )
        report = json.loads(artifact.read_text(encoding="utf-8"))
        verification = verify_cubic_tangent(report)
        self.assertIs(verification["verified"], True)

    def test_finite_radius_corner_certificate_replays(self) -> None:
        artifact = (
            ROOT / "artifacts" / "spin8_dirac_endpoint_octet_cubic_corner_20260810.json"
        )
        verification = verify_cubic_corner(artifact)
        self.assertIs(verification["verified"], True)
        self.assertEqual(verification["source_artifact_count"], 19)
        self.assertEqual(verification["pivot_count"], 5)

    def test_00010_selected_face_atlas_replays(self) -> None:
        artifact = (
            ROOT
            / "artifacts"
            / "spin8_dirac_endpoint_octet_cubic_boundary_00010_atlas_20260810.json"
        )
        verification = verify_cubic_00010(artifact)
        self.assertIs(verification["verified"], True)
        self.assertEqual(verification["box_count"], 16)

    def test_00001_complete_child_atlas_replays(self) -> None:
        artifact = (
            ROOT
            / "artifacts"
            / "spin8_dirac_endpoint_octet_cubic_atlas_nested_00001_complete_20260810.json"
        )
        verification = verify_nested_00001_report(artifact)
        self.assertIs(verification["verified"], True)
        self.assertEqual(verification["box_count"], 32)
        self.assertEqual(verification["delegated_box_count"], 1)

    def test_complete_coarse_quotient_atlas_replays(self) -> None:
        artifact = (
            ROOT
            / "artifacts"
            / "spin8_dirac_endpoint_octet_cubic_coarse_atlas_20260811.json"
        )
        verification = verify_cubic_coarse(artifact)
        self.assertIs(verification["verified"], True)
        self.assertEqual(verification["box_count"], 32)


if __name__ == "__main__":
    unittest.main()
