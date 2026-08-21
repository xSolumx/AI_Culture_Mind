"""Classify exact-candidate atlas cells by incidence with equality edges."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from spin9_candidate_normal_form import ISOLATING_CELLS
from spin9_v1_v5_reconstruction import ROOT

ATLAS_ARTIFACT = ROOT / "artifacts" / "spin9_candidate_quadratic_atlas_20260821.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spin9_candidate_handoff_map_20260821.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_root_intervals() -> list[dict[str, object]]:
    rows = []
    for root_index, (numerator, decimal_power) in enumerate(ISOLATING_CELLS):
        x_lower = Fraction(numerator, 10**decimal_power)
        x_upper = x_lower + Fraction(1, 10**decimal_power)
        scalar_sign = 1 if x_lower > 0 else -1
        if scalar_sign == 1:
            magnitude_lower, magnitude_upper = x_lower, x_upper
        else:
            magnitude_lower, magnitude_upper = -x_upper, -x_lower
        t_lower = magnitude_lower / (1 + magnitude_lower)
        t_upper = magnitude_upper / (1 + magnitude_upper)
        rows.append(
            {
                "root_index": root_index,
                "scalar_sign": scalar_sign,
                "t_lower": str(t_lower),
                "t_upper": str(t_upper),
                "t_midpoint_decimal": str(float((t_lower + t_upper) / 2)),
            }
        )
    return rows


def certificate() -> dict[str, object]:
    """Separate equality-incident handoffs from generically refinable cells."""

    atlas = json.loads(ATLAS_ARTIFACT.read_text(encoding="utf-8"))
    roots = _compact_root_intervals()
    chart_rows = []
    incidence_per_root = [0] * len(roots)
    for chart in atlas["charts"]:
        incident = []
        generic = []
        for cell in chart["atlas"]["unresolved"]:
            box = cell["box"]
            t_lower = Fraction(box["t_lower"])
            t_upper = Fraction(box["t_upper"])
            root_hits = [
                int(root["root_index"])
                for root in roots
                if root["scalar_sign"] == chart["scalar_sign"]
                and Fraction(box["a_upper"]) == 1
                and not (
                    t_upper < Fraction(root["t_lower"])
                    or Fraction(root["t_upper"]) < t_lower
                )
            ]
            row = {"path": cell["path"], "root_indices": root_hits}
            if root_hits:
                incident.append(row)
                for root_index in root_hits:
                    incidence_per_root[root_index] += 1
            else:
                generic.append(row)
        chart_rows.append(
            {
                "scalar_sign": chart["scalar_sign"],
                "unresolved_count": len(incident) + len(generic),
                "equality_incident_count": len(incident),
                "generic_refinement_count": len(generic),
                "equality_incident": incident,
                "generic_refinement": generic,
            }
        )

    passed = bool(
        atlas["passed"]
        and sum(row["unresolved_count"] for row in chart_rows) == 29
        and sum(row["equality_incident_count"] for row in chart_rows) == 16
        and sum(row["generic_refinement_count"] for row in chart_rows) == 13
        and incidence_per_root == [4, 4, 4, 4]
    )
    return {
        "schema_version": 1,
        "claim_scope": (
            "exact incidence map for the depth-six Q(sqrt(241)) atlas; "
            "this selects local charts and is not a positivity theorem"
        ),
        "atlas_artifact": ATLAS_ARTIFACT.name,
        "atlas_artifact_sha256": _sha256(ATLAS_ARTIFACT),
        "compact_candidate_root_intervals": roots,
        "charts": chart_rows,
        "incidence_count_per_root": incidence_per_root,
        "equality_incident_count": 16,
        "generic_refinement_count": 13,
        "minimal_shape_independent_cusp_charts": 4,
        "reason": (
            "each equality edge is independent of z, so its four depth-six "
            "z strips belong to one cusp chart"
        ),
        "compact_complement_certified_at_candidate_ratio": False,
        "passed": passed,
    }


def verify_report(report: dict[str, object]) -> bool:
    return bool(
        report.get("passed") is True
        and report.get("atlas_artifact_sha256") == _sha256(ATLAS_ARTIFACT)
        and report.get("incidence_count_per_root") == [4, 4, 4, 4]
        and report.get("equality_incident_count") == 16
        and report.get("generic_refinement_count") == 13
        and report.get("minimal_shape_independent_cusp_charts") == 4
        and report.get("compact_complement_certified_at_candidate_ratio") is False
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    report = certificate()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
