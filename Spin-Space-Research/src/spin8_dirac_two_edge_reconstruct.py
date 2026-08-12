"""Crash-safe exact reconstruction of one two-edge Walsh sector."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp

from spin8_dirac_star import rational_circle
from spin8_dirac_two_edge import exact_normalized_determinant
from spin8_dirac_two_edge_amplitude import exact_extended_chart_sign_certificate
from spin8_dirac_two_edge_degree import _sector_setup

TARGET_MASK = (1, 1, 0, 1, 0, 1)
VARIABLE_ORDER = ("a2", "d2", "e2", "g2", "i2", "c2")
NODE_SETS = {
    "alpha": (
        sp.Rational(1, 7),
        sp.Rational(2, 9),
        sp.Rational(3, 11),
        sp.Rational(4, 13),
    ),
    "beta": (
        sp.Rational(1, 8),
        sp.Rational(2, 11),
        sp.Rational(3, 13),
        sp.Rational(4, 15),
    ),
}


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=1)
def _target_setup():
    masks, signs, inverse = _sector_setup()
    index = masks.index(TARGET_MASK)
    weights = tuple(inverse[index, column] for column in range(len(signs)))
    chart_signs = exact_extended_chart_sign_certificate()
    character = next(
        row
        for row in chart_signs["chart_characters"]
        if tuple(row["lower_mask"]) == TARGET_MASK
    )
    degree = (3, 3, 3, 3, 3, 3)
    return signs, weights, tuple(character["complement_mask"]), tuple(degree)


def _point_worker(job):
    multi_index, parameters, signs, weights, complement_mask = job
    determinants = [
        exact_normalized_determinant(parameters, signs_row) for signs_row in signs
    ]
    sector = sp.factor(
        sum(weight * value for weight, value in zip(weights, determinants, strict=True))
    )
    pairs = tuple(rational_circle(value) for value in parameters)
    forced = sp.prod(
        pair[0] ** lower_bit * pair[1] ** complement_bit
        for pair, lower_bit, complement_bit in zip(
            pairs, TARGET_MASK, complement_mask, strict=True
        )
    )
    forced *= pairs[-1][1] ** 6
    residual = sp.factor(sector / forced)
    return multi_index, str(residual)


def evaluate_slab(*, node_set: str, slab: int, workers: int) -> dict[str, object]:
    signs, weights, complement_mask, degree = _target_setup()
    nodes = NODE_SETS[node_set]
    if degree != (3, 3, 3, 3, 3, 3):
        raise AssertionError("frozen target degree changed")
    if slab not in range(len(nodes)):
        raise ValueError("slab must be between 0 and 3")
    indices = [(slab, *tail) for tail in itertools.product(range(len(nodes)), repeat=5)]
    jobs = [
        (
            index,
            tuple(nodes[position] for position in index),
            signs,
            weights,
            complement_mask,
        )
        for index in indices
    ]
    if workers == 1:
        evaluated = list(map(_point_worker, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            evaluated = list(pool.map(_point_worker, jobs, chunksize=2))
    rows = [
        {"multi_index": list(index), "residual": value}
        for index, value in sorted(evaluated)
    ]
    return {
        "experiment": "two-edge sector 110101 exact reconstruction slab",
        "node_set": node_set,
        "nodes": [str(value) for value in nodes],
        "squared_nodes": [str(rational_circle(value)[0] ** 2) for value in nodes],
        "slab": slab,
        "sector_mask": list(TARGET_MASK),
        "complement_mask": list(complement_mask),
        "degree_bound": list(degree),
        "point_count": len(rows),
        "rows_sha256": _digest(rows),
        "rows": rows,
        "passed": len(rows) == 1024,
    }


def _write_gzip_json(path: Path, value: object) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def _read_gzip_json(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def reconstruct(slabs: list[Path]) -> dict[str, object]:
    if len(slabs) != 4:
        raise ValueError("exactly four slab files are required")
    reports = [_read_gzip_json(path) for path in slabs]
    node_set = reports[0]["node_set"]
    if any(report["node_set"] != node_set for report in reports):
        raise ValueError("slabs use different node sets")
    if {int(report["slab"]) for report in reports} != set(range(4)):
        raise ValueError("slabs do not cover 0..3")
    for report in reports:
        if report["rows_sha256"] != _digest(report["rows"]) or not report["passed"]:
            raise ValueError("invalid slab integrity or acceptance flag")

    tensor = np.empty((4,) * 6, dtype=object)
    for report in reports:
        for row in report["rows"]:
            tensor[tuple(row["multi_index"])] = sp.Rational(row["residual"])
    squared_nodes = tuple(sp.Rational(value) for value in reports[0]["squared_nodes"])
    vandermonde = sp.Matrix(
        [[node**power for power in range(4)] for node in squared_nodes]
    )
    inverse = vandermonde.inv()
    coefficients = tensor
    for axis in range(6):
        moved = np.moveaxis(coefficients, axis, 0)
        flat = moved.reshape(4, -1)
        transformed = np.empty_like(flat)
        for column in range(flat.shape[1]):
            result = inverse * sp.Matrix(list(flat[:, column]))
            transformed[:, column] = list(result)
        coefficients = np.moveaxis(transformed.reshape(moved.shape), 0, axis)

    coefficient_rows = []
    for powers in itertools.product(range(4), repeat=6):
        value = sp.factor(coefficients[powers])
        if value != 0:
            coefficient_rows.append({"powers": list(powers), "coefficient": str(value)})
    observed_degree = [
        max(row["powers"][axis] for row in coefficient_rows) for axis in range(6)
    ]
    return {
        "experiment": "two-edge sector 110101 exact coefficient reconstruction",
        "node_set": node_set,
        "sector_mask": list(TARGET_MASK),
        "variable_order": list(VARIABLE_ORDER),
        "degree_bound": [3] * 6,
        "point_count": int(tensor.size),
        "nonzero_coefficient_count": len(coefficient_rows),
        "observed_multidegree": observed_degree,
        "coefficient_rows_sha256": _digest(coefficient_rows),
        "coefficient_rows": coefficient_rows,
        "source_slab_hashes": [report["rows_sha256"] for report in reports],
        "passed": int(tensor.size) == 4096,
    }


def verify_coefficient_report(report: dict[str, object]) -> bool:
    """Replay every lightweight acceptance predicate for a coefficient map."""

    rows = report.get("coefficient_rows")
    if not isinstance(rows, list) or not rows:
        return False
    if report.get("coefficient_rows_sha256") != _digest(rows):
        return False
    if report.get("sector_mask") != list(TARGET_MASK):
        return False
    if report.get("variable_order") != list(VARIABLE_ORDER):
        return False
    if report.get("degree_bound") != [3] * 6 or report.get("point_count") != 4096:
        return False
    powers = [tuple(int(value) for value in row["powers"]) for row in rows]
    if len(set(powers)) != len(powers):
        return False
    if any(
        len(power) != 6 or any(value not in range(4) for value in power)
        for power in powers
    ):
        return False
    try:
        coefficients = [sp.Rational(row["coefficient"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    if any(value == 0 for value in coefficients):
        return False
    observed_degree = [max(power[axis] for power in powers) for axis in range(6)]
    return bool(
        report.get("nonzero_coefficient_count") == len(rows)
        and report.get("observed_multidegree") == observed_degree
        and report.get("passed") is True
    )


def verify_comparison_report(
    report: dict[str, object], coefficient_report: dict[str, object]
) -> bool:
    """Verify that a stored two-grid comparison names the accepted coefficient map."""

    return bool(
        verify_coefficient_report(coefficient_report)
        and report.get("left_node_set") != report.get("right_node_set")
        and report.get("coefficient_maps_match") is True
        and report.get("coefficient_rows_sha256")
        == coefficient_report.get("coefficient_rows_sha256")
        and report.get("observed_multidegree")
        == coefficient_report.get("observed_multidegree")
        and report.get("nonzero_coefficient_count")
        == coefficient_report.get("nonzero_coefficient_count")
        and report.get("passed") is True
    )


def verify_holdout_report(
    report: dict[str, object], coefficient_report: dict[str, object]
) -> bool:
    """Replay holdout integrity and every stored exact equality."""

    rows = report.get("rows")
    if not isinstance(rows, list):
        return False
    exact_rows = all(
        row.get("exact_match") is True
        and sp.Rational(row["observed"]) == sp.Rational(row["predicted"])
        for row in rows
    )
    return bool(
        verify_coefficient_report(coefficient_report)
        and report.get("coefficient_rows_sha256")
        == coefficient_report.get("coefficient_rows_sha256")
        and report.get("holdout_count") == len(rows) == 32
        and report.get("direct_determinant_count") == 256
        and report.get("rows_sha256") == _digest(rows)
        and report.get("all_exact_matches") is True
        and exact_rows
        and report.get("passed") is True
    )


def _factor_from_coefficient_report(
    coefficient_report: dict[str, object],
) -> dict[str, object]:
    if not verify_coefficient_report(coefficient_report):
        raise ValueError("coefficient report failed replay verification")
    variables = sp.symbols("a2 d2 e2 g2 i2 c2")
    polynomial = sp.Poly(
        sum(
            sp.Rational(row["coefficient"])
            * sp.prod(variables[axis] ** int(row["powers"][axis]) for axis in range(6))
            for row in coefficient_report["coefficient_rows"]
        ),
        *variables,
        domain=sp.QQ,
    )
    divisor = sp.Poly(1 - variables[0], *variables, domain=sp.QQ)
    quotient, remainder = sp.div(polynomial, divisor)
    quotient_rows = [
        {"powers": list(powers), "coefficient": str(coefficient)}
        for powers, coefficient in quotient.terms()
    ]
    quotient_degree = [quotient.degree(variable) for variable in variables]
    recomposes = polynomial == divisor * quotient
    a2, d2, e2, g2, i2, c2 = variables
    quotient_expression = quotient.as_expr()
    base = sp.Poly(quotient_expression.subs({i2: 0, c2: 0}), *variables, domain=sp.QQ)
    late_boundary = sp.Poly((1 - d2) * (1 - e2) * (1 - g2), *variables, domain=sp.QQ)
    correction, correction_remainder = sp.div(quotient - base, late_boundary)
    late_linear_form = d2 * e2 + d2 - e2 - g2
    late_core = (
        6 * a2 * d2 * e2 * g2
        + 25 * a2 * d2 * e2
        + 9 * a2 * d2
        - 6 * a2 * e2 * g2
        - 25 * a2 * e2
        - 9 * a2 * g2
        + 60 * d2 * e2 * g2
        - 47 * d2 * e2
        - 13 * d2
        - 60 * e2 * g2
        + 47 * e2
        + 13 * g2
    )
    compact_correction = sp.Poly(
        -(c2 * (a2 - 1) * (1 - i2) * late_linear_form + i2 * late_core) / 2,
        *variables,
        domain=sp.QQ,
    )
    correction_rows = [
        {"powers": list(powers), "coefficient": str(coefficient)}
        for powers, coefficient in correction.terms()
    ]
    nested_recomposes = quotient == base + late_boundary * correction
    compact_matches = correction == compact_correction
    return {
        "experiment": "two-edge sector 110101 exact factor certificate",
        "source_coefficient_rows_sha256": coefficient_report["coefficient_rows_sha256"],
        "variable_order": list(VARIABLE_ORDER),
        "exact_factor": "1-a2",
        "geometric_factor": "A^2 where A^2=1-a^2",
        "source_nonzero_coefficient_count": len(polynomial.terms()),
        "source_multidegree": [polynomial.degree(variable) for variable in variables],
        "quotient_nonzero_coefficient_count": len(quotient_rows),
        "quotient_multidegree": quotient_degree,
        "quotient_coefficient_rows_sha256": _digest(quotient_rows),
        "quotient_coefficient_rows": quotient_rows,
        "nested_boundary_decomposition": {
            "identity": ("Q=Q(a2,d2,e2,g2,0,0)" "+(1-d2)(1-e2)(1-g2)R"),
            "late_boundary_factor": "(1-d2)(1-e2)(1-g2)",
            "geometric_late_boundary_factor": "D^2 E^2 G^2",
            "base_nonzero_coefficient_count": len(base.terms()),
            "correction_nonzero_coefficient_count": len(correction_rows),
            "correction_multidegree": [
                correction.degree(variable) for variable in variables
            ],
            "correction_coefficient_rows_sha256": _digest(correction_rows),
            "correction_coefficient_rows": correction_rows,
            "compact_correction": ("R=-(c2(a2-1)(1-i2)(d2*e2+d2-e2-g2)" "+i2*N)/2"),
            "compact_core_N": (
                "6*a2*d2*e2*g2+25*a2*d2*e2+9*a2*d2"
                "-6*a2*e2*g2-25*a2*e2-9*a2*g2"
                "+60*d2*e2*g2-47*d2*e2-13*d2"
                "-60*e2*g2+47*e2+13*g2"
            ),
            "exact_zero_remainder": correction_remainder.is_zero,
            "exact_recomposition": nested_recomposes,
            "compact_formula_matches": compact_matches,
        },
        "exact_zero_remainder": remainder.is_zero,
        "exact_recomposition": recomposes,
        "passed": bool(
            remainder.is_zero
            and recomposes
            and correction_remainder.is_zero
            and nested_recomposes
            and compact_matches
        ),
    }


def factor_report(coefficients_path: Path) -> dict[str, object]:
    """Extract and certify the exact extra ``1-a2`` factor in the target sector."""

    coefficient_report = json.loads(coefficients_path.read_text(encoding="utf-8"))
    return _factor_from_coefficient_report(coefficient_report)


def verify_factor_report(
    report: dict[str, object], coefficient_report: dict[str, object]
) -> bool:
    """Rebuild the exact factor certificate and compare all mathematical fields."""

    if not verify_coefficient_report(coefficient_report):
        return False
    return report == _factor_from_coefficient_report(coefficient_report)


def _face_from_coefficient_report(
    coefficient_report: dict[str, object],
) -> dict[str, object]:
    """Certify the two closed-form transverse faces of the quotient sector."""

    factor = _factor_from_coefficient_report(coefficient_report)
    variables = sp.symbols("a2 d2 e2 g2 i2 c2")
    a2, d2, e2, g2, _i2, _c2 = variables
    quotient = sp.Poly(
        sum(
            sp.Rational(row["coefficient"])
            * sp.prod(variables[axis] ** int(row["powers"][axis]) for axis in range(6))
            for row in factor["quotient_coefficient_rows"]
        ),
        *variables,
        domain=sp.QQ,
    ).as_expr()
    expected_d = 3 * (a2 - 1) * (g2 - 1) ** 2
    expected_g = 3 * (a2 - 1) * (d2 - 1) ** 2 * (e2 - 1) * (3 * e2 + 1)
    d_remainder = sp.factor(quotient.subs(d2, 1) - expected_d)
    g_remainder = sp.factor(quotient.subs(g2, 1) - expected_g)
    return {
        "experiment": "two-edge sector 110101 exact transverse face certificate",
        "source_coefficient_rows_sha256": coefficient_report["coefficient_rows_sha256"],
        "source_quotient_coefficient_rows_sha256": factor[
            "quotient_coefficient_rows_sha256"
        ],
        "variable_order": list(VARIABLE_ORDER),
        "d2_equals_one_identity": "Q=3(a2-1)(g2-1)^2",
        "g2_equals_one_identity": "Q=3(a2-1)(d2-1)^2(e2-1)(3e2+1)",
        "d2_face_independent_of": ["e2", "i2", "c2"],
        "g2_face_independent_of": ["i2", "c2"],
        "d2_face_exact_zero_remainder": d_remainder == 0,
        "g2_face_exact_zero_remainder": g_remainder == 0,
        "unit_cube_signs": {
            "d2_equals_one": "nonpositive",
            "g2_equals_one": "nonnegative",
        },
        "passed": bool(d_remainder == 0 and g_remainder == 0),
    }


def face_report(coefficients_path: Path) -> dict[str, object]:
    coefficient_report = json.loads(coefficients_path.read_text(encoding="utf-8"))
    return _face_from_coefficient_report(coefficient_report)


def verify_face_report(
    report: dict[str, object], coefficient_report: dict[str, object]
) -> bool:
    return report == _face_from_coefficient_report(coefficient_report)


def compare(left: Path, right: Path) -> dict[str, object]:
    left_report = json.loads(left.read_text(encoding="utf-8"))
    right_report = json.loads(right.read_text(encoding="utf-8"))
    maps_match = left_report["coefficient_rows"] == right_report["coefficient_rows"]
    return {
        "experiment": "two-edge sector 110101 disjoint-grid comparison",
        "left_node_set": left_report["node_set"],
        "right_node_set": right_report["node_set"],
        "coefficient_maps_match": maps_match,
        "coefficient_rows_sha256": (
            left_report["coefficient_rows_sha256"] if maps_match else None
        ),
        "observed_multidegree": (
            left_report["observed_multidegree"] if maps_match else None
        ),
        "nonzero_coefficient_count": (
            left_report["nonzero_coefficient_count"] if maps_match else None
        ),
        "passed": bool(left_report["passed"] and right_report["passed"] and maps_match),
    }


def holdouts(coefficients_path: Path, *, workers: int) -> dict[str, object]:
    coefficient_report = json.loads(coefficients_path.read_text(encoding="utf-8"))
    coefficient_rows = coefficient_report["coefficient_rows"]
    signs, weights, complement_mask, _degree = _target_setup()
    parameter_rows = []
    for holdout_index in range(32):
        parameters = tuple(
            sp.Rational(1 + ((7 * holdout_index + 3 * axis) % 10), 29 + 2 * axis)
            for axis in range(6)
        )
        parameter_rows.append(parameters)
    jobs = [
        (
            (holdout_index,) * 6,
            parameters,
            signs,
            weights,
            complement_mask,
        )
        for holdout_index, parameters in enumerate(parameter_rows)
    ]
    if workers == 1:
        evaluated = list(map(_point_worker, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            evaluated = list(pool.map(_point_worker, jobs, chunksize=2))
    observed = {index[0]: sp.Rational(value) for index, value in evaluated}
    rows = []
    for holdout_index, parameters in enumerate(parameter_rows):
        squared = tuple(rational_circle(value)[0] ** 2 for value in parameters)
        predicted = sp.factor(
            sum(
                sp.Rational(row["coefficient"])
                * sp.prod(
                    squared[axis] ** int(row["powers"][axis]) for axis in range(6)
                )
                for row in coefficient_rows
            )
        )
        rows.append(
            {
                "holdout_index": holdout_index,
                "parameters": [str(value) for value in parameters],
                "observed": str(observed[holdout_index]),
                "predicted": str(predicted),
                "exact_match": observed[holdout_index] == predicted,
            }
        )
    return {
        "experiment": "two-edge sector 110101 fresh exact holdouts",
        "coefficient_rows_sha256": coefficient_report["coefficient_rows_sha256"],
        "holdout_count": len(rows),
        "direct_determinant_count": len(rows) * len(signs),
        "rows_sha256": _digest(rows),
        "rows": rows,
        "all_exact_matches": all(row["exact_match"] for row in rows),
        "passed": len(rows) == 32 and all(row["exact_match"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    evaluate_parser = subparsers.add_parser("evaluate-slab")
    evaluate_parser.add_argument("--node-set", choices=tuple(NODE_SETS), required=True)
    evaluate_parser.add_argument("--slab", type=int, required=True)
    evaluate_parser.add_argument("--workers", type=int, default=1)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    reconstruct_parser = subparsers.add_parser("reconstruct")
    reconstruct_parser.add_argument("--slabs", nargs=4, type=Path, required=True)
    reconstruct_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--left", type=Path, required=True)
    compare_parser.add_argument("--right", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    holdout_parser = subparsers.add_parser("holdouts")
    holdout_parser.add_argument("--coefficients", type=Path, required=True)
    holdout_parser.add_argument("--workers", type=int, default=1)
    holdout_parser.add_argument("--output", type=Path, required=True)
    factor_parser = subparsers.add_parser("factor")
    factor_parser.add_argument("--coefficients", type=Path, required=True)
    factor_parser.add_argument("--output", type=Path, required=True)
    face_parser = subparsers.add_parser("faces")
    face_parser.add_argument("--coefficients", type=Path, required=True)
    face_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.stage == "evaluate-slab":
        report = evaluate_slab(
            node_set=arguments.node_set, slab=arguments.slab, workers=arguments.workers
        )
        _write_gzip_json(arguments.output, report)
    elif arguments.stage == "reconstruct":
        report = reconstruct(arguments.slabs)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif arguments.stage == "compare":
        report = compare(arguments.left, arguments.right)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif arguments.stage == "holdouts":
        report = holdouts(arguments.coefficients, workers=arguments.workers)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif arguments.stage == "factor":
        report = factor_report(arguments.coefficients)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        report = face_report(arguments.coefficients)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "rows" and key != "coefficient_rows"
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit("two-edge reconstruction stage failed")


if __name__ == "__main__":
    main()
