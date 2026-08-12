"""Reconstruct the sixteen unrestricted residual polynomials exactly.

The input is one complete 25-tile shared grid produced by
``spin8_dirac_unrestricted_grid``.  Tensor-product interpolation is performed
with native FLINT rational matrices.  Every coefficient outside the
pre-certified multidegree box must vanish exactly.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import sympy as sp
from flint import ctx, fmpq, fmpq_mat

from spin8_dirac_final_residual import exact_full_multidegree_certificate
from spin8_dirac_star import rational_circle
from spin8_dirac_unrestricted_grid import (
    NODE_SETS,
    VARIABLE_ORDER,
    _read_gzip_json,
    verify_tile,
)
from spin8_resource_limits import constrain_current_process


def _mask_text(mask: tuple[int, ...]) -> str:
    return "".join(map(str, mask))


def _fmpq(value: str | sp.Expr) -> fmpq:
    numerator, denominator = sp.fraction(sp.Rational(value))
    return fmpq(int(numerator), int(denominator))


def _canonical_digest(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _write_gzip_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    temporary.replace(path)


def _inverse_vandermonde(node_set: str) -> fmpq_mat:
    squared = [rational_circle(value)[0] ** 2 for value in NODE_SETS[node_set]]
    matrix = fmpq_mat(
        5,
        5,
        [_fmpq(node**power) for node in squared for power in range(5)],
    )
    return matrix.inv()


def _transform_axis(data: list[fmpq], axis: int, inverse: fmpq_mat) -> list[fmpq]:
    size = 5**7
    stride = 5 ** (6 - axis)
    block = 5 * stride
    columns = size // 5
    entries = []
    for coordinate in range(5):
        for outer in range(size // block):
            base = outer * block + coordinate * stride
            entries.extend(data[base : base + stride])
    transformed = (inverse * fmpq_mat(5, columns, entries)).entries()
    result = [fmpq(0)] * size
    for coordinate in range(5):
        offset = coordinate * columns
        for outer in range(size // block):
            source = offset + outer * stride
            target = outer * block + coordinate * stride
            result[target : target + stride] = transformed[source : source + stride]
    return result


def _load_tiles(input_dir: Path, node_set: str) -> list[dict[str, object]]:
    paths = [
        input_dir / f"{node_set}_tile_{left}_{right}.json.gz"
        for left, right in itertools.product(range(5), repeat=2)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} grid tiles")
    reports = [_read_gzip_json(path) for path in paths]
    if not all(verify_tile(report, node_set) for report in reports):
        raise ValueError("at least one exact grid tile failed replay verification")
    return reports


def reconstruct(
    *, node_set: str, input_dir: Path, output_dir: Path, flint_threads: int
) -> dict[str, object]:
    if not 1 <= flint_threads <= 7:
        raise ValueError("FLINT threads must leave at least one logical core free")
    resource = constrain_current_process(workers=flint_threads)
    ctx.threads = flint_threads
    structure = exact_full_multidegree_certificate()
    if not structure["passed"]:
        raise AssertionError("the structural certificate failed")
    reports = _load_tiles(input_dir, node_set)
    masks = tuple(
        tuple(row["lower_mask"])
        for row in structure["full_chart_sign_certificate"]["chart_characters"]
    )
    bounds = {
        tuple(row["lower_mask"]): tuple(
            row["residual_polynomial_multidegree_upper_bound"]
        )
        for row in structure["sector_rows"]
    }
    inverse = _inverse_vandermonde(node_set)
    output_dir.mkdir(parents=True, exist_ok=True)
    sector_summaries = []
    for sector_index, mask in enumerate(masks):
        values = [fmpq(0)] * (5**7)
        for report in reports:
            for row in report["rows"]:
                index = tuple(int(value) for value in row["multi_index"])
                flat = sum(value * 5 ** (6 - axis) for axis, value in enumerate(index))
                values[flat] = _fmpq(row["sector_residuals"][sector_index])
        for axis in range(7):
            values = _transform_axis(values, axis, inverse)

        coefficient_rows = []
        out_of_bound = []
        for flat, value in enumerate(values):
            if value == 0:
                continue
            remaining = flat
            powers = []
            for axis in range(7):
                stride = 5 ** (6 - axis)
                powers.append(remaining // stride)
                remaining %= stride
            row = {"powers": powers, "coefficient": str(value)}
            coefficient_rows.append(row)
            if any(
                power > ceiling
                for power, ceiling in zip(powers, bounds[mask], strict=True)
            ):
                out_of_bound.append(row)
        observed_multidegree = [
            max(row["powers"][axis] for row in coefficient_rows) for axis in range(7)
        ]
        digest = _canonical_digest(coefficient_rows)
        payload = {
            "experiment": "unrestricted exact residual coefficient map",
            "node_set": node_set,
            "variable_order": list(VARIABLE_ORDER),
            "mask": list(mask),
            "degree_bound": list(bounds[mask]),
            "observed_multidegree": observed_multidegree,
            "nonzero_coefficient_count": len(coefficient_rows),
            "out_of_bound_nonzero_count": len(out_of_bound),
            "coefficient_rows_sha256": digest,
            "coefficient_rows": coefficient_rows,
            "passed": len(out_of_bound) == 0,
        }
        path = output_dir / f"{node_set}_sector_{_mask_text(mask)}.json.gz"
        _write_gzip_json(path, payload)
        sector_summaries.append(
            {
                key: payload[key]
                for key in (
                    "mask",
                    "degree_bound",
                    "observed_multidegree",
                    "nonzero_coefficient_count",
                    "out_of_bound_nonzero_count",
                    "coefficient_rows_sha256",
                    "passed",
                )
            }
            | {"path": path.as_posix()}
        )
        print(json.dumps(sector_summaries[-1]), flush=True)

    summary = {
        "experiment": "unrestricted exact shared-grid reconstruction",
        "node_set": node_set,
        "variable_order": list(VARIABLE_ORDER),
        "grid_point_count": 5**7,
        "direct_determinant_count": 16 * 5**7,
        "sector_count": len(sector_summaries),
        "sector_summaries": sector_summaries,
        "source_tile_hashes": [report["rows_sha256"] for report in reports],
        "resource_contract": resource,
        "passed": len(sector_summaries) == 16
        and all(row["passed"] for row in sector_summaries),
    }
    summary_path = output_dir / f"{node_set}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-set", choices=sorted(NODE_SETS), required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_grid_20260807"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_coefficients_20260807"),
    )
    parser.add_argument("--flint-threads", type=int, default=6)
    arguments = parser.parse_args()
    report = reconstruct(
        node_set=arguments.node_set,
        input_dir=arguments.input_dir,
        output_dir=arguments.output_dir,
        flint_threads=arguments.flint_threads,
    )
    print(json.dumps({"summary": report}, indent=2))
    if not report["passed"]:
        raise SystemExit("unrestricted reconstruction failed")


if __name__ == "__main__":
    main()
