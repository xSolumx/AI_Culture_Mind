"""Regression tests for the narrow endpoint replay compatibility rules."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
replay = importlib.import_module("replay_endpoint_octet_runtime_artifacts")


class EndpointOctetRuntimeReplayTests(unittest.TestCase):
    def test_audit_engine_metadata_is_schema_only(self) -> None:
        old = {"audit": {"coefficient_count": 17}}
        new = {
            "audit": {
                "audit_engine": "exact_batched_in_place",
                "batch_entry_limit": 500_000,
                "coefficient_count": 17,
            }
        }
        self.assertEqual(
            replay._strip_schema_metadata(old),
            replay._strip_schema_metadata(new),
        )

    def test_double_face_redundant_null_route_is_schema_only(self) -> None:
        old = {
            "selected_face": "double",
            "post_zero_path": [],
            "selected_zero_face": None,
            "passed": True,
        }
        new = {"selected_face": "double", "passed": True}
        self.assertEqual(
            replay._strip_schema_metadata(old),
            replay._strip_schema_metadata(new),
        )

    def test_substantive_route_metadata_is_never_masked(self) -> None:
        reference = {"selected_face": "double", "passed": True}
        for altered in (
            {
                "selected_face": "double",
                "post_zero_path": [0],
                "passed": True,
            },
            {
                "selected_face": "double",
                "selected_zero_face": 2,
                "passed": True,
            },
            {
                "selected_face": "ui",
                "post_zero_path": [],
                "selected_zero_face": None,
                "passed": True,
            },
        ):
            with self.subTest(altered=altered):
                self.assertNotEqual(
                    replay._strip_schema_metadata(reference),
                    replay._strip_schema_metadata(altered),
                )

    def test_empty_post_zero_path_is_a_global_default(self) -> None:
        old = {"selected_face": "ui", "selected_zero_face": {"axis": 0}}
        new = {**old, "post_zero_path": []}
        self.assertEqual(
            replay._strip_schema_metadata(old),
            replay._strip_schema_metadata(new),
        )

    def test_empty_parent_path_is_a_global_default(self) -> None:
        old = {"selected_face": "ui", "passed": False}
        new = {**old, "parent_path": []}
        self.assertEqual(
            replay._strip_schema_metadata(old),
            replay._strip_schema_metadata(new),
        )

        substantive = {**old, "parent_path": [{"active_bits": "0010"}]}
        self.assertNotEqual(
            replay._strip_schema_metadata(old),
            replay._strip_schema_metadata(substantive),
        )

    def test_null_selected_zero_face_is_a_global_default(self) -> None:
        old = {"selected_face": "ui", "passed": False}
        new = {**old, "selected_zero_face": None}
        self.assertEqual(
            replay._strip_schema_metadata(old),
            replay._strip_schema_metadata(new),
        )

    def test_exact_legacy_ui_label_and_scope_aliases(self) -> None:
        old = {
            "selected_face": replay.LEGACY_UI_FACE_LABEL,
            "scope_boundary": replay.LEGACY_UI_SCOPE,
        }
        new = {
            "selected_face": replay.CURRENT_UI_FACE_LABEL,
            "scope_boundary": replay.CURRENT_SELECTED_FACE_SCOPE,
        }
        self.assertEqual(
            replay._strip_schema_metadata(old),
            replay._strip_schema_metadata(new),
        )

    def test_unregistered_scope_change_is_never_masked(self) -> None:
        reference = {"scope_boundary": replay.CURRENT_SELECTED_FACE_SCOPE}
        altered = {"scope_boundary": "This would change the claim boundary."}
        self.assertNotEqual(
            replay._strip_schema_metadata(reference),
            replay._strip_schema_metadata(altered),
        )

    def test_difference_paths_preserve_full_audit_trail(self) -> None:
        old = {"audit": {"count": 4}}
        new = {"audit": {"count": 5, "audit_engine": "batched"}}
        self.assertEqual(
            replay._difference_paths(old, new),
            ["$.audit.audit_engine", "$.audit.count"],
        )


if __name__ == "__main__":
    unittest.main()
