"""Exact replay for the four finite Spin(9) candidate collars."""

from __future__ import annotations

import json
from pathlib import Path

from spin9_candidate_explicit_collar import certificate, verify_report

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "spin9_candidate_explicit_collar_20260821.json"


def test_explicit_candidate_collars_replay_exactly() -> None:
    report = certificate()
    assert report["passed"]
    assert report["all_four_explicit_collars_certified"]
    assert not report["compact_complement_certified_at_candidate_ratio"]


def test_stored_collar_claim_boundary() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert verify_report(report)
