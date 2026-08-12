"""Exact symmetry and anchor reconnaissance for the two-edge Dirac bridge."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import sympy as sp

from spin8_cayley_spectrum import (
    symbolic_query_projector,
    symbolic_triality_generators,
)
from spin8_dirac_edge import _character, exact_walsh_symmetry_certificate
from spin8_dirac_one_edge import _symbolic_vector
from spin8_dirac_star import rational_circle

PARAMETER_NAMES = ("a", "d", "e", "g", "i", "c")
SIGNS = tuple(itertools.product((1, -1), repeat=len(PARAMETER_NAMES)))
ANCHORS = (
    (
        sp.Rational(1, 7),
        sp.Rational(2, 9),
        sp.Rational(3, 11),
        sp.Rational(2, 13),
        sp.Rational(3, 14),
        sp.Rational(4, 15),
    ),
    (
        sp.Rational(2, 7),
        sp.Rational(1, 9),
        sp.Rational(4, 11),
        sp.Rational(3, 13),
        sp.Rational(5, 14),
        sp.Rational(2, 15),
    ),
)


@lru_cache(maxsize=1)
def _context():
    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    fixed = symbolic_query_projector(
        0, basis[0], generators
    ) + symbolic_query_projector(1, basis[0], generators)
    return generators, basis, fixed


def exact_normalized_determinant(
    parameters: tuple[sp.Rational, ...], signs: tuple[int, ...]
) -> sp.Expr:
    """Evaluate one exact normalized determinant in the frozen ``h=0`` family."""

    generators, basis, fixed = _context()
    (a, A), (d, D), (e, E), (g, G), (i, I), (cayley, sine) = tuple(
        rational_circle(value) for value in parameters
    )
    sign_a, sign_d, sign_e, sign_g, sign_i, sign_c = signs
    x2 = _symbolic_vector((sign_a * a, A), (0, 1), basis)
    x3 = _symbolic_vector((sign_d * d, D * sign_e * e, D * E), (0, 1, 2), basis)
    q4 = _symbolic_vector((sign_c * cayley, sine), (3, 4), basis)
    x4 = [
        sign_g * g * basis[0][column]
        + G * (sign_i * i * basis[2][column] + I * q4[column])
        for column in range(8)
    ]
    information = fixed + symbolic_query_projector(1, x2, generators)
    information += symbolic_query_projector(2, x3, generators)
    information += symbolic_query_projector(2, x4, generators)
    delta = A**2 * D**2 * E**2 * G**2 * I**2
    return sp.factor(1024 * information.det(method="domain-ge") / delta**3)


def exact_sign_symmetry_certificate() -> dict[str, object]:
    """Derive the six-parameter sign quotient from common triality actions."""

    base = exact_walsh_symmetry_certificate()
    induced = set()
    for action in base["triality_representation_actions"]:
        coordinate_signs = action["vector_signs"]
        t1, t2, t3, t4 = (
            coordinate_signs[1],
            coordinate_signs[2],
            coordinate_signs[3],
            coordinate_signs[4],
        )
        induced.add((t1, t2, t1 * t2, t4, t2 * t4, t3 * t4))
    annihilator = {
        mask
        for mask in itertools.product((0, 1), repeat=len(PARAMETER_NAMES))
        if all(_character(signs, mask) == 1 for signs in induced)
    }
    return {
        "parameter_order": list(PARAMETER_NAMES),
        "induced_sign_group": [list(row) for row in sorted(induced)],
        "walsh_annihilator": [list(row) for row in sorted(annihilator)],
        "common_triality_conjugacy_verified": base["common_adjoint_conjugacy_verified"],
        "passed": (
            base["passed"]
            and base["common_adjoint_conjugacy_verified"]
            and len(induced) == 8
            and len(annihilator) == 8
        ),
    }


def _worker(job):
    anchor_index, parameters, signs = job
    return anchor_index, signs, str(exact_normalized_determinant(parameters, signs))


def _rows_digest(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def exact_anchor_certificate(*, workers: int = 1) -> dict[str, object]:
    symmetry = exact_sign_symmetry_certificate()
    annihilator = {tuple(row) for row in symmetry["walsh_annihilator"]}
    jobs = [
        (anchor_index, parameters, signs)
        for anchor_index, parameters in enumerate(ANCHORS)
        for signs in SIGNS
    ]
    if workers == 1:
        evaluated = list(map(_worker, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            evaluated = list(pool.map(_worker, jobs, chunksize=4))

    direct = {
        (anchor_index, signs): sp.Rational(value)
        for anchor_index, signs, value in evaluated
    }
    rows = [
        {
            "anchor_index": anchor_index,
            "signs": list(signs),
            "normalized_determinant": str(value),
        }
        for (anchor_index, signs), value in sorted(direct.items())
    ]
    anchor_rows = []
    all_supports = []
    for anchor_index, parameters in enumerate(ANCHORS):
        coefficients = {}
        for mask in itertools.product((0, 1), repeat=len(PARAMETER_NAMES)):
            coefficient = sp.factor(
                sum(
                    _character(signs, mask) * direct[anchor_index, signs]
                    for signs in SIGNS
                )
                / len(SIGNS)
            )
            if coefficient != 0:
                coefficients[mask] = coefficient
        support = set(coefficients)
        all_supports.append(support)
        cayley = rational_circle(parameters[-1])[0]
        target = (1 - cayley**2) ** 3 * (9 - cayley**2) ** 2
        margins = [target - direct[anchor_index, signs] for signs in SIGNS]
        anchor_rows.append(
            {
                "anchor_index": anchor_index,
                "rational_circle_parameters": [str(value) for value in parameters],
                "nonzero_walsh_support": [list(mask) for mask in sorted(support)],
                "nonzero_sector_count": len(support),
                "support_within_symmetry_annihilator": support <= annihilator,
                "support_equals_symmetry_annihilator": support == annihilator,
                "minimum_exact_target_margin": str(min(margins)),
            }
        )

    passed = (
        symmetry["passed"]
        and len(rows) == len(ANCHORS) * len(SIGNS)
        and all(row["support_within_symmetry_annihilator"] for row in anchor_rows)
    )
    return {
        "experiment": "variable-Cayley two-edge exact symmetry and anchor audit",
        "family": "h=0 with residuals e and i active",
        "symmetry": symmetry,
        "anchors": anchor_rows,
        "supports_match_between_anchors": all_supports[0] == all_supports[1],
        "exact_determinant_count": len(rows),
        "determinant_rows_sha256": _rows_digest(rows),
        "determinant_rows": rows,
        "interpretation": (
            "Exact anchor support is reconnaissance. It does not prove global "
            "sector vanishing or the target inequality."
        ),
        "passed": passed,
    }


def verify_anchor_report(report: dict[str, object]) -> bool:
    symmetry = exact_sign_symmetry_certificate()
    rows = report.get("determinant_rows")
    if not isinstance(rows, list) or report.get("determinant_rows_sha256") != (
        _rows_digest(rows)
    ):
        return False
    if report.get("symmetry") != symmetry:
        return False
    expected_keys = set(itertools.product(range(len(ANCHORS)), SIGNS))
    observed_keys = {(int(row["anchor_index"]), tuple(row["signs"])) for row in rows}
    if observed_keys != expected_keys:
        return False
    direct = {
        (int(row["anchor_index"]), tuple(row["signs"])): sp.Rational(
            row["normalized_determinant"]
        )
        for row in rows
    }
    annihilator = {tuple(mask) for mask in symmetry["walsh_annihilator"]}
    reconstructed_supports = []
    for anchor_index, parameters in enumerate(ANCHORS):
        support = set()
        for mask in itertools.product((0, 1), repeat=len(PARAMETER_NAMES)):
            coefficient = sp.factor(
                sum(
                    _character(signs, mask) * direct[anchor_index, signs]
                    for signs in SIGNS
                )
                / len(SIGNS)
            )
            if coefficient != 0:
                support.add(mask)
        reconstructed_supports.append(support)
        stored = report["anchors"][anchor_index]
        cayley = rational_circle(parameters[-1])[0]
        target = (1 - cayley**2) ** 3 * (9 - cayley**2) ** 2
        minimum_margin = min(target - direct[anchor_index, signs] for signs in SIGNS)
        if stored["nonzero_walsh_support"] != [list(mask) for mask in sorted(support)]:
            return False
        if stored["minimum_exact_target_margin"] != str(minimum_margin):
            return False
        if stored["support_within_symmetry_annihilator"] is not (
            support <= annihilator
        ):
            return False
        if stored["support_equals_symmetry_annihilator"] is not (
            support == annihilator
        ):
            return False
    return bool(
        report.get("exact_determinant_count") == len(expected_keys)
        and report.get("supports_match_between_anchors")
        is (reconstructed_supports[0] == reconstructed_supports[1])
        and report.get("passed")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    report = exact_anchor_certificate(workers=arguments.workers)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "exact_determinant_count": report["exact_determinant_count"],
                "supports_match_between_anchors": report[
                    "supports_match_between_anchors"
                ],
                "anchors": report["anchors"],
                "passed": report["passed"],
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit("two-edge exact anchor gate failed")


if __name__ == "__main__":
    main()
