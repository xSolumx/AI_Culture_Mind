"""Exact replay for the Spin(9) candidate invariant normal form."""

from __future__ import annotations

import json
from pathlib import Path

from spin9_candidate_normal_form import certificate, verify_report

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "spin9_candidate_normal_form_20260821.json"


def test_candidate_normal_form_replays_exactly() -> None:
    report = certificate()
    assert report["passed"]
    assert report["pure_gap_divisible_by_candidate_fiber_squared"]
    assert report["mixed_radial_coefficient_positive_at_all_preimages"]
    assert not report["explicit_finite_radius_certified"]


def test_stored_candidate_normal_form_claim_boundary() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert verify_report(report)
