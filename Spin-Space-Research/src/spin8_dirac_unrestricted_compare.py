"""Compare disjoint unrestricted reconstructions and replay exact holdouts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import sympy as sp

from spin8_dirac_final_residual import (
    exact_full_chart_sign_certificate,
    exact_normalized_determinant_from_half_angles,
)
from spin8_dirac_star import rational_circle
from spin8_dirac_unrestricted_grid import _sector_metadata


def _read_gzip(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evaluate_coefficients(
    rows: list[dict[str, object]], point: tuple[sp.Rational, ...]
) -> sp.Rational:
    return sp.factor(
        sum(
            sp.Rational(row["coefficient"])
            * sp.prod(
                coordinate**power
                for coordinate, power in zip(point, row["powers"], strict=True)
            )
            for row in rows
        )
    )


def _direct_residuals(
    half_angles: tuple[sp.Rational, ...],
) -> tuple[list[sp.Rational], tuple[sp.Rational, ...]]:
    masks, complements, representatives, hadamard = _sector_metadata()
    pairs = tuple(rational_circle(value) for value in half_angles)
    sine_sixth = pairs[-1][1] ** 6
    determinants = [
        sp.factor(
            1024
            * exact_normalized_determinant_from_half_angles(half_angles, signs)
            / sine_sixth
        )
        for signs in representatives
    ]
    sectors = [
        sp.factor(
            sum(
                sign * determinant
                for sign, determinant in zip(row, determinants, strict=True)
            )
            / 16
        )
        for row in hadamard
    ]
    target = (9 - pairs[-1][0] ** 2) ** 2
    residuals = []
    for sector_index, (mask, sector) in enumerate(zip(masks, sectors, strict=True)):
        margin = target - sector if sector_index == 0 else -sector
        forced = sp.prod(
            pair[0] ** lower_bit * pair[1] ** complement_bit
            for pair, lower_bit, complement_bit in zip(
                pairs, mask, complements[mask], strict=True
            )
        )
        residuals.append(sp.factor(margin / forced))
    return residuals, tuple(pair[0] ** 2 for pair in pairs)


def compare(*, coefficient_dir: Path, holdout_count: int) -> dict[str, object]:
    chart = exact_full_chart_sign_certificate()
    masks = tuple(tuple(row["lower_mask"]) for row in chart["chart_characters"])
    map_rows = {}
    comparison_rows = []
    for mask in masks:
        text = "".join(map(str, mask))
        alpha_path = coefficient_dir / f"alpha_sector_{text}.json.gz"
        beta_path = coefficient_dir / f"beta_sector_{text}.json.gz"
        alpha = _read_gzip(alpha_path)
        beta = _read_gzip(beta_path)
        maps_match = alpha["coefficient_rows"] == beta["coefficient_rows"]
        comparison_rows.append(
            {
                "mask": list(mask),
                "alpha_sha256": alpha["coefficient_rows_sha256"],
                "beta_sha256": beta["coefficient_rows_sha256"],
                "complete_maps_match": maps_match,
                "nonzero_coefficient_count": alpha["nonzero_coefficient_count"],
                "observed_multidegree": alpha["observed_multidegree"],
            }
        )
        if not maps_match:
            raise AssertionError(f"coefficient maps disagree in sector {text}")
        map_rows[mask] = alpha["coefficient_rows"]

    holdout_rows = []
    for index in range(holdout_count):
        half_angles = tuple(
            sp.Rational(2 + ((13 * index + 7 * axis) % 23), 41 + 2 * axis)
            for axis in range(7)
        )
        observed, squared = _direct_residuals(half_angles)
        sector_rows = []
        for mask, value in zip(masks, observed, strict=True):
            predicted = _evaluate_coefficients(map_rows[mask], squared)
            sector_rows.append(
                {
                    "mask": list(mask),
                    "exact_match": value == predicted,
                    "observed_sha256": hashlib.sha256(str(value).encode()).hexdigest(),
                    "predicted_sha256": hashlib.sha256(
                        str(predicted).encode()
                    ).hexdigest(),
                }
            )
        holdout_rows.append(
            {
                "holdout_index": index,
                "half_angle_coordinates": [str(value) for value in half_angles],
                "sector_rows": sector_rows,
                "passed": all(row["exact_match"] for row in sector_rows),
            }
        )

    passed = bool(
        len(comparison_rows) == 16
        and all(row["complete_maps_match"] for row in comparison_rows)
        and len(holdout_rows) == holdout_count
        and all(row["passed"] for row in holdout_rows)
    )
    return {
        "experiment": "unrestricted disjoint-grid identity and holdout certificate",
        "sector_comparison_rows": comparison_rows,
        "sector_comparison_rows_sha256": _digest(comparison_rows),
        "holdout_count": holdout_count,
        "direct_determinant_count": 16 * holdout_count,
        "holdout_rows": holdout_rows,
        "holdout_rows_sha256": _digest(holdout_rows),
        "all_complete_maps_match": all(
            row["complete_maps_match"] for row in comparison_rows
        ),
        "all_holdouts_match": all(row["passed"] for row in holdout_rows),
        "scope_boundary": (
            "This proves the reconstructed polynomial identities under the exact "
            "structural degree certificate. It does not prove positivity."
        ),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficient-dir",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_coefficients_20260807"),
    )
    parser.add_argument("--holdouts", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_comparison_20260807.json"),
    )
    arguments = parser.parse_args()
    report = compare(
        coefficient_dir=arguments.coefficient_dir,
        holdout_count=arguments.holdouts,
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("unrestricted comparison failed")


if __name__ == "__main__":
    main()
