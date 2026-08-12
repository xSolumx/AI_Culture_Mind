"""Crash-safe shared-grid reconstruction of all eight two-edge sectors."""

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

from spin8_dirac_edge import _character
from spin8_dirac_star import rational_circle
from spin8_dirac_two_edge import exact_normalized_determinant
from spin8_dirac_two_edge_amplitude import exact_extended_chart_sign_certificate
from spin8_dirac_two_edge_degree import _sector_setup

VARIABLE_ORDER = ("a2", "d2", "e2", "g2", "i2", "c2")
NODE_SETS = {
    "alpha": (
        sp.Rational(1, 7),
        sp.Rational(2, 9),
        sp.Rational(3, 11),
        sp.Rational(4, 13),
        sp.Rational(5, 17),
    ),
    "beta": (
        sp.Rational(1, 8),
        sp.Rational(2, 11),
        sp.Rational(3, 13),
        sp.Rational(4, 15),
        sp.Rational(5, 19),
    ),
}
DEGREE_BOUNDS = {
    (0, 0, 0, 0, 0, 0): (4, 4, 4, 4, 4, 4),
    (0, 0, 1, 1, 0, 1): (4, 4, 3, 3, 3, 3),
    (0, 1, 0, 1, 1, 0): (4, 3, 3, 3, 3, 4),
    (0, 1, 1, 0, 1, 1): (4, 3, 3, 4, 3, 3),
    (1, 0, 0, 0, 1, 1): (3, 4, 4, 4, 3, 3),
    (1, 0, 1, 1, 1, 0): (3, 4, 3, 3, 3, 4),
    (1, 1, 0, 1, 0, 1): (3, 3, 3, 3, 3, 3),
    (1, 1, 1, 0, 0, 0): (3, 3, 3, 4, 4, 4),
}
SLICE_ENDPOINT_PREDICTIONS = {
    (0, 0, 0, 0, 0, 0): (),
    (0, 0, 1, 1, 0, 1): ("a2", "d2"),
    (0, 1, 0, 1, 1, 0): (),
    (0, 1, 1, 0, 1, 1): ("a2", "d2", "e2", "g2"),
    (1, 0, 0, 0, 1, 1): ("a2", "d2", "e2", "g2"),
    (1, 0, 1, 1, 1, 0): ("d2",),
    (1, 1, 0, 1, 0, 1): ("a2",),
    (1, 1, 1, 0, 0, 0): (),
}


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _mask_text(mask: tuple[int, ...]) -> str:
    return "".join(map(str, mask))


@lru_cache(maxsize=1)
def _shared_setup():
    masks, signs, inverse = _sector_setup()
    chart = exact_extended_chart_sign_certificate()
    complements = {
        tuple(row["lower_mask"]): tuple(row["complement_mask"])
        for row in chart["chart_characters"]
    }
    if set(masks) != set(DEGREE_BOUNDS) or set(masks) != set(complements):
        raise AssertionError("shared-grid sector metadata is incomplete")
    return masks, signs, inverse, complements


def _shared_point_worker(job):
    multi_index, parameters, masks, signs, inverse, complements = job
    determinants = sp.Matrix(
        [exact_normalized_determinant(parameters, sign_row) for sign_row in signs]
    )
    sectors = inverse * determinants
    pairs = tuple(rational_circle(value) for value in parameters)
    residuals = []
    for mask, sector in zip(masks, sectors, strict=True):
        complement_mask = complements[mask]
        forced = pairs[-1][1] ** 6
        forced *= sp.prod(
            pair[0] ** lower_bit * pair[1] ** complement_bit
            for pair, lower_bit, complement_bit in zip(
                pairs, mask, complement_mask, strict=True
            )
        )
        residuals.append(str(sp.factor(sector / forced)))
    return multi_index, residuals


def evaluate_tile(
    *, node_set: str, tile_a: int, tile_d: int, workers: int
) -> dict[str, object]:
    masks, signs, inverse, complements = _shared_setup()
    nodes = NODE_SETS[node_set]
    if tile_a not in range(5) or tile_d not in range(5):
        raise ValueError("tile indices must be between 0 and 4")
    indices = [
        (tile_a, tile_d, *tail) for tail in itertools.product(range(5), repeat=4)
    ]
    jobs = [
        (
            index,
            tuple(nodes[position] for position in index),
            masks,
            signs,
            inverse,
            complements,
        )
        for index in indices
    ]
    if workers == 1:
        evaluated = list(map(_shared_point_worker, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            evaluated = list(pool.map(_shared_point_worker, jobs, chunksize=2))
    rows = [
        {"multi_index": list(index), "sector_residuals": residuals}
        for index, residuals in sorted(evaluated)
    ]
    return {
        "experiment": "two-edge all-sector exact shared-grid tile",
        "node_set": node_set,
        "nodes": [str(value) for value in nodes],
        "squared_nodes": [str(rational_circle(value)[0] ** 2) for value in nodes],
        "tile": [tile_a, tile_d],
        "sector_masks": [list(mask) for mask in masks],
        "degree_bounds": {
            _mask_text(mask): list(DEGREE_BOUNDS[mask]) for mask in masks
        },
        "point_count": len(rows),
        "direct_determinant_count": len(rows) * len(signs),
        "rows_sha256": _digest(rows),
        "rows": rows,
        "passed": len(rows) == 625,
    }


def _write_gzip_json(path: Path, value: object) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def _read_gzip_json(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def verify_tile_report(report: dict[str, object]) -> bool:
    """Replay tile integrity and exact coverage without trusting ``passed``."""

    if report.get("node_set") not in NODE_SETS:
        return False
    tile = report.get("tile")
    if (
        not isinstance(tile, list)
        or len(tile) != 2
        or any(int(value) not in range(5) for value in tile)
    ):
        return False
    masks, _signs, _inverse, _complements = _shared_setup()
    expected_masks = [list(mask) for mask in masks]
    expected_bounds = {_mask_text(mask): list(DEGREE_BOUNDS[mask]) for mask in masks}
    nodes = NODE_SETS[report["node_set"]]
    if report.get("nodes") != [str(value) for value in nodes]:
        return False
    if report.get("squared_nodes") != [
        str(rational_circle(value)[0] ** 2) for value in nodes
    ]:
        return False
    rows = report.get("rows")
    if not isinstance(rows, list) or report.get("rows_sha256") != _digest(rows):
        return False
    expected_indices = {
        (int(tile[0]), int(tile[1]), *tail)
        for tail in itertools.product(range(5), repeat=4)
    }
    try:
        observed_indices = {
            tuple(int(value) for value in row["multi_index"]) for row in rows
        }
        residuals_are_exact = all(
            len(row["sector_residuals"]) == 8
            and all(sp.Rational(value).is_Rational for value in row["sector_residuals"])
            for row in rows
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        len(rows) == len(observed_indices) == len(expected_indices) == 625
        and observed_indices == expected_indices
        and residuals_are_exact
        and report.get("sector_masks") == expected_masks
        and report.get("degree_bounds") == expected_bounds
        and report.get("point_count") == 625
        and report.get("direct_determinant_count") == 5000
        and report.get("passed") is True
    )


def reconstruct(tiles: list[Path]) -> dict[str, object]:
    if len(tiles) != 25:
        raise ValueError("exactly 25 tile files are required")
    reports = [_read_gzip_json(path) for path in tiles]
    node_set = reports[0]["node_set"]
    if any(report["node_set"] != node_set for report in reports):
        raise ValueError("tiles use different node sets")
    if {tuple(report["tile"]) for report in reports} != set(
        itertools.product(range(5), repeat=2)
    ):
        raise ValueError("tiles do not cover the 5 x 5 leading-axis grid")
    if not all(verify_tile_report(report) for report in reports):
        raise ValueError("tile failed replay verification")

    masks = tuple(tuple(mask) for mask in reports[0]["sector_masks"])
    tensors = {mask: np.empty((5,) * 6, dtype=object) for mask in masks}
    for report in reports:
        for row in report["rows"]:
            index = tuple(row["multi_index"])
            for mask, value in zip(masks, row["sector_residuals"], strict=True):
                tensors[mask][index] = sp.Rational(value)

    squared_nodes = tuple(sp.Rational(value) for value in reports[0]["squared_nodes"])
    vandermonde = sp.Matrix(
        [[node**power for power in range(5)] for node in squared_nodes]
    )
    inverse = vandermonde.inv()
    sector_rows = []
    for mask in masks:
        coefficients = tensors[mask]
        for axis in range(6):
            moved = np.moveaxis(coefficients, axis, 0)
            flat = moved.reshape(5, -1)
            transformed = np.empty_like(flat)
            for column in range(flat.shape[1]):
                transformed[:, column] = list(
                    inverse * sp.Matrix(list(flat[:, column]))
                )
            coefficients = np.moveaxis(transformed.reshape(moved.shape), 0, axis)

        coefficient_rows = []
        out_of_bound_rows = []
        bound = DEGREE_BOUNDS[mask]
        for powers in itertools.product(range(5), repeat=6):
            value = sp.factor(coefficients[powers])
            if value == 0:
                continue
            row = {"powers": list(powers), "coefficient": str(value)}
            coefficient_rows.append(row)
            if any(
                power > ceiling for power, ceiling in zip(powers, bound, strict=True)
            ):
                out_of_bound_rows.append(row)
        observed_degree = [
            max(row["powers"][axis] for row in coefficient_rows) for axis in range(6)
        ]
        sector_rows.append(
            {
                "mask": list(mask),
                "degree_bound": list(bound),
                "observed_multidegree": observed_degree,
                "nonzero_coefficient_count": len(coefficient_rows),
                "out_of_bound_nonzero_count": len(out_of_bound_rows),
                "coefficient_rows_sha256": _digest(coefficient_rows),
                "coefficient_rows": coefficient_rows,
            }
        )
    all_bounds_pass = all(row["out_of_bound_nonzero_count"] == 0 for row in sector_rows)
    return {
        "experiment": "two-edge all-sector exact shared-grid reconstruction",
        "node_set": node_set,
        "variable_order": list(VARIABLE_ORDER),
        "point_count": 5**6,
        "direct_determinant_count": 8 * 5**6,
        "sector_count": len(sector_rows),
        "all_individual_degree_bounds_pass": all_bounds_pass,
        "sector_rows_sha256": _digest(sector_rows),
        "sector_rows": sector_rows,
        "source_tile_hashes": [report["rows_sha256"] for report in reports],
        "passed": bool(len(sector_rows) == 8 and all_bounds_pass),
    }


def verify_coefficient_report(report: dict[str, object]) -> bool:
    """Replay every lightweight acceptance predicate for all eight maps."""

    rows = report.get("sector_rows")
    if not isinstance(rows, list) or report.get("sector_rows_sha256") != _digest(rows):
        return False
    if report.get("variable_order") != list(VARIABLE_ORDER):
        return False
    if (
        report.get("point_count") != 15625
        or report.get("direct_determinant_count") != 125000
    ):
        return False
    expected_masks = set(DEGREE_BOUNDS)
    try:
        observed_masks = {tuple(row["mask"]) for row in rows}
        for row in rows:
            mask = tuple(row["mask"])
            coefficient_rows = row["coefficient_rows"]
            if row["coefficient_rows_sha256"] != _digest(coefficient_rows):
                return False
            powers = [
                tuple(int(value) for value in item["powers"])
                for item in coefficient_rows
            ]
            if len(set(powers)) != len(powers):
                return False
            if any(sp.Rational(item["coefficient"]) == 0 for item in coefficient_rows):
                return False
            observed_degree = [
                max(power[axis] for power in powers) for axis in range(6)
            ]
            bound = DEGREE_BOUNDS[mask]
            out_of_bound = sum(
                any(power > ceiling for power, ceiling in zip(item, bound, strict=True))
                for item in powers
            )
            if (
                row["degree_bound"] != list(bound)
                or row["observed_multidegree"] != observed_degree
                or row["nonzero_coefficient_count"] != len(coefficient_rows)
                or row["out_of_bound_nonzero_count"] != out_of_bound
                or out_of_bound != 0
            ):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        observed_masks == expected_masks
        and len(rows) == report.get("sector_count") == 8
        and report.get("all_individual_degree_bounds_pass") is True
        and report.get("passed") is True
    )


def compare(left: Path, right: Path) -> dict[str, object]:
    left_report = json.loads(left.read_text(encoding="utf-8"))
    right_report = json.loads(right.read_text(encoding="utf-8"))
    maps_match = left_report.get("sector_rows") == right_report.get("sector_rows")
    return {
        "experiment": "two-edge all-sector disjoint shared-grid comparison",
        "left_node_set": left_report.get("node_set"),
        "right_node_set": right_report.get("node_set"),
        "both_coefficient_reports_verify": bool(
            verify_coefficient_report(left_report)
            and verify_coefficient_report(right_report)
        ),
        "complete_sector_maps_match": maps_match,
        "sector_rows_sha256": (
            left_report.get("sector_rows_sha256") if maps_match else None
        ),
        "sector_count": left_report.get("sector_count") if maps_match else None,
        "total_grid_points": (
            left_report.get("point_count", 0) + right_report.get("point_count", 0)
        ),
        "total_direct_determinants": (
            left_report.get("direct_determinant_count", 0)
            + right_report.get("direct_determinant_count", 0)
        ),
        "passed": bool(
            left_report.get("node_set") != right_report.get("node_set")
            and verify_coefficient_report(left_report)
            and verify_coefficient_report(right_report)
            and maps_match
        ),
    }


def verify_comparison_report(
    report: dict[str, object],
    left_report: dict[str, object],
    right_report: dict[str, object],
) -> bool:
    """Recompute the complete two-grid comparison instead of trusting flags."""

    if not verify_coefficient_report(left_report) or not verify_coefficient_report(
        right_report
    ):
        return False
    maps_match = left_report["sector_rows"] == right_report["sector_rows"]
    expected = {
        "experiment": "two-edge all-sector disjoint shared-grid comparison",
        "left_node_set": left_report.get("node_set"),
        "right_node_set": right_report.get("node_set"),
        "both_coefficient_reports_verify": True,
        "complete_sector_maps_match": maps_match,
        "sector_rows_sha256": (
            left_report.get("sector_rows_sha256") if maps_match else None
        ),
        "sector_count": left_report.get("sector_count") if maps_match else None,
        "total_grid_points": (
            left_report.get("point_count", 0) + right_report.get("point_count", 0)
        ),
        "total_direct_determinants": (
            left_report.get("direct_determinant_count", 0)
            + right_report.get("direct_determinant_count", 0)
        ),
        "passed": bool(
            left_report.get("node_set") != right_report.get("node_set") and maps_match
        ),
    }
    return report == expected


def holdouts(coefficients_path: Path, *, workers: int) -> dict[str, object]:
    coefficient_report = json.loads(coefficients_path.read_text(encoding="utf-8"))
    if not verify_coefficient_report(coefficient_report):
        raise ValueError("coefficient report failed replay verification")
    masks, signs, inverse, complements = _shared_setup()
    parameters_rows = [
        tuple(
            sp.Rational(1 + ((11 * index + 5 * axis) % 13), 31 + 2 * axis)
            for axis in range(6)
        )
        for index in range(32)
    ]
    jobs = [
        ((index,) * 6, parameters, masks, signs, inverse, complements)
        for index, parameters in enumerate(parameters_rows)
    ]
    if workers == 1:
        evaluated = list(map(_shared_point_worker, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            evaluated = list(pool.map(_shared_point_worker, jobs, chunksize=2))
    observed = {index[0]: values for index, values in evaluated}
    coefficient_maps = {
        tuple(row["mask"]): row["coefficient_rows"]
        for row in coefficient_report["sector_rows"]
    }


def verify_holdout_report(
    report: dict[str, object], coefficient_report: dict[str, object]
) -> bool:
    """Replay every stored exact holdout equality and its integrity metadata."""

    if not verify_coefficient_report(coefficient_report):
        return False
    rows = report.get("rows")
    if not isinstance(rows, list) or report.get("rows_sha256") != _digest(rows):
        return False
    if report.get("source_sector_rows_sha256") != coefficient_report.get(
        "sector_rows_sha256"
    ):
        return False
    expected_masks = set(DEGREE_BOUNDS)
    try:
        for index, row in enumerate(rows):
            if row["holdout_index"] != index or len(row["parameters"]) != 6:
                return False
            sector_rows = row["sector_rows"]
            if len(sector_rows) != 8:
                return False
            if {tuple(sector["mask"]) for sector in sector_rows} != expected_masks:
                return False
            for sector in sector_rows:
                equality = sp.Rational(sector["observed"]) == sp.Rational(
                    sector["predicted"]
                )
                if sector["exact_match"] is not equality or not equality:
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        len(rows) == report.get("holdout_count") == 32
        and report.get("sector_equality_count") == 256
        and report.get("direct_determinant_count") == 256
        and report.get("all_exact_matches") is True
        and report.get("passed") is True
    )
    rows = []
    for holdout_index, parameters in enumerate(parameters_rows):
        squared = tuple(rational_circle(value)[0] ** 2 for value in parameters)
        sector_rows = []
        for sector_index, mask in enumerate(masks):
            predicted = sp.factor(
                sum(
                    sp.Rational(row["coefficient"])
                    * sp.prod(
                        squared[axis] ** int(row["powers"][axis]) for axis in range(6)
                    )
                    for row in coefficient_maps[mask]
                )
            )
            observed_value = sp.Rational(observed[holdout_index][sector_index])
            sector_rows.append(
                {
                    "mask": list(mask),
                    "observed": str(observed_value),
                    "predicted": str(predicted),
                    "exact_match": observed_value == predicted,
                }
            )
        rows.append(
            {
                "holdout_index": holdout_index,
                "parameters": [str(value) for value in parameters],
                "sector_rows": sector_rows,
            }
        )
    all_exact = all(
        sector["exact_match"] for row in rows for sector in row["sector_rows"]
    )
    return {
        "experiment": "two-edge all-sector fresh exact holdouts",
        "source_sector_rows_sha256": coefficient_report["sector_rows_sha256"],
        "holdout_count": len(rows),
        "sector_equality_count": len(rows) * len(masks),
        "direct_determinant_count": len(rows) * len(signs),
        "rows_sha256": _digest(rows),
        "rows": rows,
        "all_exact_matches": all_exact,
        "passed": bool(len(rows) == 32 and all_exact),
    }


def _polynomial_rows(polynomial: sp.Poly) -> list[dict[str, object]]:
    return [
        {"powers": list(powers), "coefficient": str(coefficient)}
        for powers, coefficient in polynomial.terms()
    ]


def exact_i_parity_block_certificate() -> dict[str, object]:
    """Prove the eight orientation margins split into two four-sector blocks."""

    masks, signs, _inverse, _complements = _shared_setup()
    even_indices = [index for index, mask in enumerate(masks) if mask[4] == 0]
    odd_indices = [index for index, mask in enumerate(masks) if mask[4] == 1]
    pairs: dict[tuple[int, ...], list[int]] = {}
    for index, sign_row in enumerate(signs):
        pairs.setdefault(sign_row[:4] + sign_row[5:], []).append(index)

    pair_rows = []
    positive_indices = []
    pairing_is_exact = len(pairs) == 4
    for fixed_signs, indices in sorted(pairs.items()):
        negative = [index for index in indices if signs[index][4] == -1]
        positive = [index for index in indices if signs[index][4] == 1]
        if len(negative) != 1 or len(positive) != 1:
            pairing_is_exact = False
            continue
        minus_index, plus_index = negative[0], positive[0]
        positive_indices.append(plus_index)
        even_equal = all(
            _character(signs[minus_index], masks[index])
            == _character(signs[plus_index], masks[index])
            for index in even_indices
        )
        odd_opposite = all(
            _character(signs[minus_index], masks[index])
            == -_character(signs[plus_index], masks[index])
            for index in odd_indices
        )
        pairing_is_exact &= even_equal and odd_opposite
        pair_rows.append(
            {
                "fixed_signs_without_i": list(fixed_signs),
                "negative_i_row": minus_index,
                "positive_i_row": plus_index,
                "even_characters_equal": even_equal,
                "odd_characters_opposite": odd_opposite,
            }
        )

    even_hadamard = sp.Matrix(
        [
            [_character(signs[row], masks[index]) for index in even_indices]
            for row in positive_indices
        ]
    )
    odd_hadamard = sp.Matrix(
        [
            [_character(signs[row], masks[index]) for index in odd_indices]
            for row in positive_indices
        ]
    )
    even_rows = [[int(value) for value in row] for row in even_hadamard.tolist()]
    odd_rows = [[int(value) for value in row] for row in odd_hadamard.tolist()]
    reduced_even_masks = {
        mask[:4] + mask[5:] for mask in (masks[index] for index in even_indices)
    }
    one_edge_masks = {
        (0, 0, 0, 0, 0),
        (0, 0, 1, 1, 1),
        (1, 1, 0, 1, 1),
        (1, 1, 1, 0, 0),
    }
    same_hadamard = even_hadamard == odd_hadamard
    hadamard_is_exact = even_hadamard.T * even_hadamard == 4 * sp.eye(
        4
    ) and odd_hadamard.T * odd_hadamard == 4 * sp.eye(4)
    passed = bool(
        pairing_is_exact
        and same_hadamard
        and hadamard_is_exact
        and reduced_even_masks == one_edge_masks
    )
    return {
        "experiment": "two-edge exact i-parity block reduction",
        "even_sector_masks": [list(masks[index]) for index in even_indices],
        "odd_sector_masks": [list(masks[index]) for index in odd_indices],
        "orientation_pair_rows": pair_rows,
        "pairing_is_exact": bool(pairing_is_exact),
        "even_hadamard": even_rows,
        "odd_hadamard": odd_rows,
        "even_and_odd_hadamards_identical": same_hadamard,
        "both_tables_are_exact_hadamard": hadamard_is_exact,
        "even_masks_reduce_to_one_edge_support": reduced_even_masks == one_edge_masks,
        "block_identity": "lambda_(r,+/-)=W_r*(E+/-O)",
        "interpretation": (
            "the eight margins are the eigenvalues of two commuting four-by-four "
            "group-circulants K_plus and K_minus"
        ),
        "passed": passed,
    }


def _factor_atlas_from_coefficient_report(
    coefficient_report: dict[str, object],
) -> dict[str, object]:
    """Derive exact endpoint and nested-flag factors for every sector."""

    if not verify_coefficient_report(coefficient_report):
        raise ValueError("coefficient report failed replay verification")
    variables = sp.symbols("a2 d2 e2 g2 i2 c2")
    variable_map = dict(zip(VARIABLE_ORDER, variables, strict=True))
    _a2, d2, e2, g2, i2, c2 = variables
    sector_reports = []
    for source in coefficient_report["sector_rows"]:
        mask = tuple(source["mask"])
        polynomial = sp.Poly(
            sum(
                sp.Rational(row["coefficient"])
                * sp.prod(
                    variables[axis] ** int(row["powers"][axis]) for axis in range(6)
                )
                for row in source["coefficient_rows"]
            ),
            *variables,
            domain=sp.QQ,
        )
        reduced = polynomial
        endpoint_multiplicities = {}
        for name, variable in variable_map.items():
            divisor = sp.Poly(1 - variable, *variables, domain=sp.QQ)
            multiplicity = 0
            while reduced.as_expr().subs(variable, 1) == 0:
                reduced = reduced.exquo(divisor)
                multiplicity += 1
            if multiplicity:
                endpoint_multiplicities[name] = multiplicity

        base = sp.Poly(reduced.as_expr().subs({i2: 0, c2: 0}), *variables, domain=sp.QQ)
        late = reduced - base
        correction = late
        nested_multiplicities = {}
        for name, variable in (("d2", d2), ("e2", e2), ("g2", g2)):
            divisor = sp.Poly(1 - variable, *variables, domain=sp.QQ)
            multiplicity = 0
            while (
                not correction.is_zero and correction.as_expr().subs(variable, 1) == 0
            ):
                correction = correction.exquo(divisor)
                multiplicity += 1
            if multiplicity:
                nested_multiplicities[name] = multiplicity

        endpoint_product = sp.Poly(
            sp.prod(
                (1 - variable_map[name]) ** multiplicity
                for name, multiplicity in endpoint_multiplicities.items()
            ),
            *variables,
            domain=sp.QQ,
        )
        nested_product = sp.Poly(
            sp.prod(
                (1 - variable_map[name]) ** multiplicity
                for name, multiplicity in nested_multiplicities.items()
            ),
            *variables,
            domain=sp.QQ,
        )
        recomposed = endpoint_product * (base + nested_product * correction)

        one_edge_base = sp.Poly(reduced.as_expr().subs(i2, 0), *variables, domain=sp.QQ)
        one_edge_difference = reduced - one_edge_base
        one_edge_i_quotient = one_edge_difference.exquo(
            sp.Poly(i2, *variables, domain=sp.QQ)
        )
        one_edge_correction = one_edge_i_quotient
        one_edge_nested_multiplicities = {}
        for name, variable in (("d2", d2), ("e2", e2), ("g2", g2)):
            divisor = sp.Poly(1 - variable, *variables, domain=sp.QQ)
            multiplicity = 0
            while (
                not one_edge_correction.is_zero
                and one_edge_correction.as_expr().subs(variable, 1) == 0
            ):
                one_edge_correction = one_edge_correction.exquo(divisor)
                multiplicity += 1
            if multiplicity:
                one_edge_nested_multiplicities[name] = multiplicity
        one_edge_nested_product = sp.Poly(
            sp.prod(
                (1 - variable_map[name]) ** multiplicity
                for name, multiplicity in one_edge_nested_multiplicities.items()
            ),
            *variables,
            domain=sp.QQ,
        )
        one_edge_recomposed = endpoint_product * (
            one_edge_base + i2 * one_edge_nested_product * one_edge_correction
        )
        base_rows = _polynomial_rows(base)
        correction_rows = _polynomial_rows(correction)
        one_edge_base_rows = _polynomial_rows(one_edge_base)
        one_edge_correction_rows = _polynomial_rows(one_edge_correction)
        found_endpoint_variables = tuple(endpoint_multiplicities)
        sector_reports.append(
            {
                "mask": list(mask),
                "source_coefficient_rows_sha256": source["coefficient_rows_sha256"],
                "source_nonzero_coefficient_count": len(polynomial.terms()),
                "endpoint_factor_multiplicities": endpoint_multiplicities,
                "endpoint_factors_match_slice_prediction": (
                    found_endpoint_variables == SLICE_ENDPOINT_PREDICTIONS[mask]
                ),
                "reduced_nonzero_coefficient_count": len(reduced.terms()),
                "reduced_multidegree": [
                    reduced.degree(variable) for variable in variables
                ],
                "base_nonzero_coefficient_count": len(base_rows),
                "base_coefficient_rows_sha256": _digest(base_rows),
                "nested_boundary_multiplicities": nested_multiplicities,
                "correction_nonzero_coefficient_count": len(correction_rows),
                "correction_multidegree": [
                    correction.degree(variable) for variable in variables
                ],
                "correction_coefficient_rows_sha256": _digest(correction_rows),
                "exact_recomposition": recomposed == polynomial,
                "one_edge_base_nonzero_coefficient_count": len(one_edge_base_rows),
                "one_edge_base_coefficient_rows_sha256": _digest(one_edge_base_rows),
                "one_edge_nested_boundary_multiplicities": (
                    one_edge_nested_multiplicities
                ),
                "one_edge_correction_nonzero_coefficient_count": len(
                    one_edge_correction_rows
                ),
                "one_edge_correction_multidegree": [
                    one_edge_correction.degree(variable) for variable in variables
                ],
                "one_edge_correction_coefficient_rows_sha256": _digest(
                    one_edge_correction_rows
                ),
                "exact_one_edge_recomposition": one_edge_recomposed == polynomial,
            }
        )
    all_share_nested = all(
        row["nested_boundary_multiplicities"] == {"d2": 1, "e2": 1, "g2": 1}
        for row in sector_reports
    )
    all_predictions_match = all(
        row["endpoint_factors_match_slice_prediction"] for row in sector_reports
    )
    all_recompose = all(row["exact_recomposition"] for row in sector_reports)
    all_share_one_edge_bridge = all(
        row["one_edge_nested_boundary_multiplicities"] == {"d2": 1, "e2": 1, "g2": 1}
        for row in sector_reports
    )
    all_one_edge_recompose = all(
        row["exact_one_edge_recomposition"] for row in sector_reports
    )
    parity_block = exact_i_parity_block_certificate()
    return {
        "experiment": "two-edge all-sector exact factor and nested-flag atlas",
        "source_sector_rows_sha256": coefficient_report["sector_rows_sha256"],
        "variable_order": list(VARIABLE_ORDER),
        "sector_count": len(sector_reports),
        "source_total_nonzero_coefficient_count": sum(
            row["source_nonzero_coefficient_count"] for row in sector_reports
        ),
        "endpoint_reduced_total_nonzero_coefficient_count": sum(
            row["reduced_nonzero_coefficient_count"] for row in sector_reports
        ),
        "all_endpoint_factors_match_exact_slice_predictions": all_predictions_match,
        "all_sectors_share_nested_D2E2G2_factor": all_share_nested,
        "universal_nested_identity": ("H_m=H_m|_(i2=c2=0)+(1-d2)(1-e2)(1-g2)R_m"),
        "all_sectors_share_exact_one_edge_bridge": all_share_one_edge_bridge,
        "universal_one_edge_bridge": ("H_m=H_m|_(i2=0)+i2(1-d2)(1-e2)(1-g2)T_m"),
        "i_parity_block_certificate": parity_block,
        "sector_reports_sha256": _digest(sector_reports),
        "sector_reports": sector_reports,
        "all_exact_recompositions": all_recompose,
        "all_exact_one_edge_recompositions": all_one_edge_recompose,
        "passed": bool(
            len(sector_reports) == 8
            and all_predictions_match
            and all_share_nested
            and all_recompose
            and all_share_one_edge_bridge
            and all_one_edge_recompose
            and parity_block["passed"]
        ),
    }


def factor_atlas(coefficients_path: Path) -> dict[str, object]:
    coefficient_report = json.loads(coefficients_path.read_text(encoding="utf-8"))
    return _factor_atlas_from_coefficient_report(coefficient_report)


def verify_factor_atlas_report(
    report: dict[str, object], coefficient_report: dict[str, object]
) -> bool:
    return report == _factor_atlas_from_coefficient_report(coefficient_report)


def _orthonormal_transverse_from_coefficient_report(
    coefficient_report: dict[str, object],
) -> dict[str, object]:
    """Certify the new residual is strictly determinant-decreasing at equality."""

    if not verify_coefficient_report(coefficient_report):
        raise ValueError("coefficient report failed replay verification")
    a2, d2, e2, g2, i2, c2 = sp.symbols("a2 d2 e2 g2 i2 c2")
    variables = (a2, d2, e2, g2, i2, c2)
    trivial = next(
        row for row in coefficient_report["sector_rows"] if row["mask"] == [0] * 6
    )
    polynomial = sp.Poly(
        sum(
            sp.Rational(row["coefficient"])
            * sp.prod(variables[axis] ** int(row["powers"][axis]) for axis in range(6))
            for row in trivial["coefficient_rows"]
        ),
        *variables,
        domain=sp.QQ,
    )
    restricted = sp.factor(polynomial.as_expr().subs({a2: 0, d2: 0, e2: 0, g2: 0}))
    margin_core = sp.factor((9 - c2) ** 2 - restricted)
    quotient = sp.factor(2 * margin_core / i2)
    quotient_polynomial = sp.Poly(quotient, i2, c2, domain=sp.QQ)
    degrees = tuple(quotient_polynomial.degree(variable) for variable in (i2, c2))
    bernstein_rows = []
    for first in range(degrees[0] + 1):
        for second in range(degrees[1] + 1):
            coefficient = sum(
                quotient_polynomial.coeff_monomial(i2**left * c2**right)
                * sp.binomial(first, left)
                / sp.binomial(degrees[0], left)
                * sp.binomial(second, right)
                / sp.binomial(degrees[1], right)
                for left in range(first + 1)
                for right in range(second + 1)
            )
            bernstein_rows.append(
                {"index": [first, second], "coefficient": str(coefficient)}
            )
    coefficients = [sp.Rational(row["coefficient"]) for row in bernstein_rows]
    masks, signs, _inverse, complements = _shared_setup()
    coefficient_rows = {
        tuple(row["mask"]): row["coefficient_rows"]
        for row in coefficient_report["sector_rows"]
    }

    def binary_polynomial_value(mask, values):
        return sum(
            sp.Rational(row["coefficient"])
            for row in coefficient_rows[mask]
            if all(value or power == 0 for value, power in zip(values, row["powers"]))
        )

    def binary_forced_factor(mask, values, *, first_order_i=False):
        for axis, (lower_bit, complement_bit) in enumerate(
            zip(mask, complements[mask], strict=True)
        ):
            if first_order_i and axis == 4:
                if lower_bit != 1:
                    return 0
                continue
            if lower_bit and values[axis] == 0:
                return 0
            if complement_bit and values[axis] == 1:
                return 0
        return 1

    even_masks = [mask for mask in masks if mask[4] == 0]
    odd_masks = [mask for mask in masks if mask[4] == 1]
    equality_rows = []
    for base_vertex in itertools.product((0, 1), repeat=5):
        values = base_vertex[:4] + (0, base_vertex[4])
        even_amplitudes = {
            mask: binary_polynomial_value(mask, values)
            * binary_forced_factor(mask, values)
            for mask in even_masks
        }
        odd_derivatives = {
            mask: binary_polynomial_value(mask, values)
            * binary_forced_factor(mask, values, first_order_i=True)
            for mask in odd_masks
        }
        for orientation_index, sign_row in enumerate(signs):
            base_margin = (9 - values[5]) ** 2 - sum(
                _character(sign_row, mask) * even_amplitudes[mask]
                for mask in even_masks
            )
            if base_margin != 0:
                continue
            odd_derivative = -sum(
                _character(sign_row, mask) * odd_derivatives[mask] for mask in odd_masks
            )
            equality_rows.append(
                {
                    "base_vertex": list(base_vertex),
                    "orientation_index": orientation_index,
                    "odd_first_derivative": str(sp.factor(odd_derivative)),
                }
            )
    all_vertex_derivatives_vanish = all(
        sp.Rational(row["odd_first_derivative"]) == 0 for row in equality_rows
    )
    expected_quotient = (
        c2**2 * (i2**2 - 4 * i2 + 5)
        + c2 * (-8 * i2**2 + 42 * i2 - 70)
        + 16 * i2**2
        - 104 * i2
        + 225
    )
    passed = bool(
        sp.expand(quotient - expected_quotient) == 0
        and degrees == (2, 2)
        and len(coefficients) == 9
        and min(coefficients) == 103
        and all(coefficient > 0 for coefficient in coefficients)
        and len(equality_rows) == 16
        and all_vertex_derivatives_vanish
    )
    return {
        "experiment": "two-edge orthonormal transverse stability theorem",
        "source_sector_rows_sha256": coefficient_report["sector_rows_sha256"],
        "slice": "a2=d2=e2=g2=0",
        "restricted_trivial_sector": str(restricted),
        "normalized_margin_factorization": str(margin_core),
        "full_margin_identity": ("target-det=(1-c2)^3*i2*P(i2,c2)/2"),
        "positive_quotient": str(quotient),
        "positive_quotient_multidegree": list(degrees),
        "bernstein_rows": bernstein_rows,
        "bernstein_rows_sha256": _digest(bernstein_rows),
        "minimum_bernstein_coefficient": str(min(coefficients)),
        "all_bernstein_coefficients_strictly_positive": all(
            coefficient > 0 for coefficient in coefficients
        ),
        "coordinate_base_vertices_audited": 32,
        "coordinate_orientation_margins_audited": 256,
        "coordinate_equality_rows": equality_rows,
        "coordinate_equality_row_count": len(equality_rows),
        "all_coordinate_equality_odd_derivatives_vanish": (
            all_vertex_derivatives_vanish
        ),
        "interpretation": (
            "the new residual strictly lowers the determinant away from i2=0 "
            "and the common Cayley boundary on the orthonormal base slice"
        ),
        "passed": passed,
    }


def orthonormal_transverse(coefficients_path: Path) -> dict[str, object]:
    coefficient_report = json.loads(coefficients_path.read_text(encoding="utf-8"))
    return _orthonormal_transverse_from_coefficient_report(coefficient_report)


def verify_orthonormal_transverse_report(
    report: dict[str, object], coefficient_report: dict[str, object]
) -> bool:
    return report == _orthonormal_transverse_from_coefficient_report(coefficient_report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    tile_parser = subparsers.add_parser("evaluate-tile")
    tile_parser.add_argument("--node-set", choices=tuple(NODE_SETS), required=True)
    tile_parser.add_argument("--tile-a", type=int, required=True)
    tile_parser.add_argument("--tile-d", type=int, required=True)
    tile_parser.add_argument("--workers", type=int, default=1)
    tile_parser.add_argument("--output", type=Path, required=True)
    reconstruct_parser = subparsers.add_parser("reconstruct")
    reconstruct_parser.add_argument("--tiles", nargs=25, type=Path, required=True)
    reconstruct_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--left", type=Path, required=True)
    compare_parser.add_argument("--right", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    holdout_parser = subparsers.add_parser("holdouts")
    holdout_parser.add_argument("--coefficients", type=Path, required=True)
    holdout_parser.add_argument("--workers", type=int, default=1)
    holdout_parser.add_argument("--output", type=Path, required=True)
    factor_parser = subparsers.add_parser("factor-atlas")
    factor_parser.add_argument("--coefficients", type=Path, required=True)
    factor_parser.add_argument("--output", type=Path, required=True)
    transverse_parser = subparsers.add_parser("orthonormal-transverse")
    transverse_parser.add_argument("--coefficients", type=Path, required=True)
    transverse_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.stage == "evaluate-tile":
        report = evaluate_tile(
            node_set=arguments.node_set,
            tile_a=arguments.tile_a,
            tile_d=arguments.tile_d,
            workers=arguments.workers,
        )
        _write_gzip_json(arguments.output, report)
    elif arguments.stage == "reconstruct":
        report = reconstruct(arguments.tiles)
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
    elif arguments.stage == "factor-atlas":
        report = factor_atlas(arguments.coefficients)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        report = orthonormal_transverse(arguments.coefficients)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"rows", "sector_rows", "source_tile_hashes"}
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit("shared-grid reconstruction stage failed")


if __name__ == "__main__":
    main()
