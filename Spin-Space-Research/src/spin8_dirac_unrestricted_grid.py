"""Crash-safe exact shared grids for the unrestricted Dirac--Gram sectors.

Each grid point evaluates sixteen orientation representatives with native
FLINT rational matrices, applies the exact Hadamard quotient, removes the
proved boundary and parity factors, and stores all sixteen residual margins.
The tile files are proof inputs; this module does not infer positivity.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sympy as sp
from flint import ctx, fmpq, fmpq_mat

from spin8_cayley_spectrum import (
    symbolic_query_projector,
    symbolic_triality_generators,
)
from spin8_dirac_edge import _character
from spin8_dirac_final_residual import exact_full_chart_sign_certificate
from spin8_dirac_star import rational_circle
from spin8_resource_limits import constrain_current_process

VARIABLE_ORDER = ("a2", "d2", "e2", "g2", "h2", "i2", "c2")
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

_COEFFICIENTS: dict[tuple[int, int, int], fmpq_mat] = {}
_FIXED: fmpq_mat | None = None
_PAIRS: tuple[tuple[fmpq, fmpq], ...] = ()
_MASKS: tuple[tuple[int, ...], ...] = ()
_COMPLEMENTS: dict[tuple[int, ...], tuple[int, ...]] = {}
_REPRESENTATIVES: tuple[tuple[int, ...], ...] = ()
_HADAMARD: tuple[tuple[int, ...], ...] = ()


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _fmpq(value: sp.Expr | int) -> fmpq:
    rational = sp.Rational(value)
    return fmpq(int(rational.p), int(rational.q))


def _fmpq_matrix(matrix: sp.Matrix) -> fmpq_mat:
    return fmpq_mat(matrix.rows, matrix.cols, [_fmpq(value) for value in matrix])


def _sector_metadata() -> tuple[
    tuple[tuple[int, ...], ...],
    dict[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    chart = exact_full_chart_sign_certificate()
    if not chart["passed"]:
        raise AssertionError("full-chart sign certificate failed")
    masks = tuple(tuple(row["lower_mask"]) for row in chart["chart_characters"])
    complements = {
        tuple(row["lower_mask"]): tuple(row["complement_mask"])
        for row in chart["chart_characters"]
    }

    from spin8_dirac_edge import exact_walsh_symmetry_certificate

    base = exact_walsh_symmetry_certificate()
    induced = {
        (t1, t2, t1 * t2, t4, t1 * t4, t2 * t4, t3 * t4)
        for action in base["triality_representation_actions"]
        for t1, t2, t3, t4 in [action["vector_signs"][1:5]]
    }
    unused = set(itertools.product((1, -1), repeat=7))
    representatives = []
    while unused:
        representative = min(unused)
        coset = {
            tuple(
                left * right for left, right in zip(representative, group, strict=True)
            )
            for group in induced
        }
        representatives.append(representative)
        unused -= coset
    representatives.sort()
    hadamard = tuple(
        tuple(_character(signs, mask) for signs in representatives) for mask in masks
    )
    if len(masks) != 16 or len(representatives) != 16:
        raise AssertionError("the full quotient must have sixteen sectors")
    return masks, complements, tuple(representatives), hadamard


def _worker_initialize(
    node_set: str,
    masks: tuple[tuple[int, ...], ...],
    complements: dict[tuple[int, ...], tuple[int, ...]],
    representatives: tuple[tuple[int, ...], ...],
    hadamard: tuple[tuple[int, ...], ...],
) -> None:
    global _COEFFICIENTS, _FIXED, _PAIRS, _MASKS, _COMPLEMENTS
    global _REPRESENTATIVES, _HADAMARD

    ctx.threads = 1
    generators = symbolic_triality_generators()
    basis = [[sp.Integer(row == column) for column in range(8)] for row in range(8)]
    needed = {0: (0,), 1: (0, 1), 2: (0, 1, 2, 3, 4)}
    coefficients: dict[tuple[int, int, int], fmpq_mat] = {}
    for view, indices in needed.items():
        diagonal = {
            index: symbolic_query_projector(view, basis[index], generators)
            for index in indices
        }
        for index in indices:
            coefficients[view, index, index] = _fmpq_matrix(diagonal[index])
        for left, right in itertools.combinations(indices, 2):
            vector = [basis[left][column] + basis[right][column] for column in range(8)]
            cross = (
                symbolic_query_projector(view, vector, generators)
                - diagonal[left]
                - diagonal[right]
            )
            coefficients[view, left, right] = _fmpq_matrix(cross)

    _COEFFICIENTS = coefficients
    _FIXED = coefficients[0, 0, 0] + coefficients[1, 0, 0]
    _PAIRS = tuple(
        (_fmpq(rational_circle(value)[0]), _fmpq(rational_circle(value)[1]))
        for value in NODE_SETS[node_set]
    )
    _MASKS = masks
    _COMPLEMENTS = complements
    _REPRESENTATIVES = representatives
    _HADAMARD = hadamard


def _projector(view: int, vector: list[tuple[int, fmpq]]) -> fmpq_mat:
    if _FIXED is None:
        raise RuntimeError("worker is not initialized")
    result = _FIXED * 0
    for offset, (left, left_value) in enumerate(vector):
        result = result + (left_value * left_value) * _COEFFICIENTS[view, left, left]
        for right, right_value in vector[offset + 1 :]:
            result = (
                result + (left_value * right_value) * _COEFFICIENTS[view, left, right]
            )
    return result


def _determinant_quotient(
    pairs: tuple[tuple[fmpq, fmpq], ...], signs: tuple[int, ...]
) -> fmpq:
    if _FIXED is None:
        raise RuntimeError("worker is not initialized")
    signed = tuple(
        (sign * lower, complement)
        for sign, (lower, complement) in zip(signs, pairs, strict=True)
    )
    (a, A), (d, D), (e, E), (g, G), (h, H), (i, I), (c, sine) = signed
    positive = [(0, a), (1, A)]
    first_negative = [(0, d), (1, D * e), (2, D * E)]
    second_negative = [
        (0, g),
        (1, G * h),
        (2, G * H * i),
        (3, G * H * I * c),
        (4, G * H * I * sine),
    ]
    information = (
        _FIXED
        + _projector(1, positive)
        + _projector(2, first_negative)
        + _projector(2, second_negative)
    )
    common = (A * D * E * G * H * I * sine) ** 6
    return 1024 * information.det() / common


def _residuals_at(index: tuple[int, ...]) -> list[str]:
    pairs = tuple(_PAIRS[position] for position in index)
    determinants = [_determinant_quotient(pairs, signs) for signs in _REPRESENTATIVES]
    sectors = [
        sum(
            (sign * determinant for sign, determinant in zip(row, determinants)),
            fmpq(0),
        )
        / 16
        for row in _HADAMARD
    ]
    target = (9 - pairs[-1][0] ** 2) ** 2
    residuals = []
    for sector_index, (mask, sector) in enumerate(zip(_MASKS, sectors, strict=True)):
        margin = target - sector if sector_index == 0 else -sector
        complement_mask = _COMPLEMENTS[mask]
        forced = fmpq(1)
        for pair, lower_bit, complement_bit in zip(
            pairs, mask, complement_mask, strict=True
        ):
            forced *= pair[0] ** lower_bit * pair[1] ** complement_bit
        residuals.append(str(margin / forced))
    return residuals


def _tile_worker(tile: tuple[int, int]) -> dict[str, object]:
    rows = []
    for tail in itertools.product(range(5), repeat=5):
        index = (*tile, *tail)
        rows.append(
            {"multi_index": list(index), "sector_residuals": _residuals_at(index)}
        )
    return {
        "tile": list(tile),
        "point_count": len(rows),
        "direct_determinant_count": len(rows) * 16,
        "rows_sha256": _digest(rows),
        "rows": rows,
    }


def _write_gzip_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    temporary.replace(path)


def _read_gzip_json(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def verify_tile(report: dict[str, object], node_set: str) -> bool:
    rows = report.get("rows")
    tile = report.get("tile")
    if not isinstance(rows, list) or not isinstance(tile, list) or len(tile) != 2:
        return False
    expected_indices = {
        (int(tile[0]), int(tile[1]), *tail)
        for tail in itertools.product(range(5), repeat=5)
    }
    try:
        observed = {tuple(row["multi_index"]) for row in rows}
        exact_rows = all(
            len(row["sector_residuals"]) == 16
            and all(sp.Rational(value).is_Rational for value in row["sector_residuals"])
            for row in rows
        )
    except (KeyError, TypeError, ValueError):
        return False
    masks, complements, representatives, _hadamard = _sector_metadata()
    return bool(
        report.get("experiment") == "unrestricted exact shared-grid tile"
        and report.get("node_set") == node_set
        and report.get("nodes") == [str(value) for value in NODE_SETS[node_set]]
        and report.get("sector_masks") == [list(mask) for mask in masks]
        and report.get("complement_masks")
        == {"".join(map(str, mask)): list(complements[mask]) for mask in masks}
        and report.get("orientation_representatives")
        == [list(row) for row in representatives]
        and len(rows) == len(observed) == len(expected_indices) == 3125
        and observed == expected_indices
        and exact_rows
        and report.get("rows_sha256") == _digest(rows)
        and report.get("point_count") == 3125
        and report.get("direct_determinant_count") == 50000
        and report.get("passed") is True
    )


def run_grid(*, node_set: str, output_dir: Path, workers: int) -> None:
    if node_set not in NODE_SETS:
        raise ValueError(f"unknown node set {node_set!r}")
    if not 1 <= workers <= 7:
        raise ValueError("workers must leave at least one logical core free")
    resource = constrain_current_process(workers=workers)
    masks, complements, representatives, hadamard = _sector_metadata()
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for tile in itertools.product(range(5), repeat=2):
        path = output_dir / f"{node_set}_tile_{tile[0]}_{tile[1]}.json.gz"
        if path.is_file():
            try:
                if verify_tile(_read_gzip_json(path), node_set):
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        pending.append((tile, path))
    print(
        json.dumps(
            {
                "node_set": node_set,
                "pending_tiles": len(pending),
                "workers": workers,
                "resource_contract": resource,
            }
        ),
        flush=True,
    )
    if not pending:
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_initialize,
        initargs=(node_set, masks, complements, representatives, hadamard),
    ) as pool:
        futures = {
            pool.submit(_tile_worker, tile): (tile, path) for tile, path in pending
        }
        for future in as_completed(futures):
            tile, path = futures[future]
            payload = future.result()
            payload.update(
                {
                    "experiment": "unrestricted exact shared-grid tile",
                    "node_set": node_set,
                    "nodes": [str(value) for value in NODE_SETS[node_set]],
                    "sector_masks": [list(mask) for mask in masks],
                    "complement_masks": {
                        "".join(map(str, mask)): list(complements[mask])
                        for mask in masks
                    },
                    "orientation_representatives": [
                        list(row) for row in representatives
                    ],
                    "passed": True,
                }
            )
            _write_gzip_json(path, payload)
            print(
                json.dumps(
                    {
                        "completed_tile": list(tile),
                        "path": path.as_posix(),
                        "rows_sha256": payload["rows_sha256"],
                    }
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-set", choices=sorted(NODE_SETS), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_grid_20260807"),
    )
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    run_grid(
        node_set=arguments.node_set,
        output_dir=arguments.output_dir,
        workers=arguments.workers,
    )


if __name__ == "__main__":
    main()
