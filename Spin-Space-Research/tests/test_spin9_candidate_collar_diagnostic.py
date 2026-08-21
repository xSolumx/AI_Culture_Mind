"""Claim-boundary gates for the near-candidate Spin(9) collar diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from spin9_candidate_collar_diagnostic import verify_report

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "spin9_candidate_collar_20260821.json"


def test_stored_candidate_collar_is_diagnostic_not_theorem() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert verify_report(report)
    assert report["unresolved_count"] > 0
    assert not report["mixed_candidate_optimality_certified"]
