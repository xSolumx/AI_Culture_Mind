"""Exact off-grid holdouts for the variable-Cayley one-edge reconstruction.

The original reconstruction artifact correctly retained ``holdouts_pending``
but an accompanying result document prematurely described the preregistered
8 by 32 holdout gate as complete.  This module closes that evidence gap without
rerunning either interpolation grid.  Every stored value is a direct exact
28 by 28 determinant, compared with the independently reconstructed sector
polynomials at a point used by neither grid.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sympy as sp

from spin8_dirac_edge import _character
from spin8_dirac_one_edge import SIGN_CHARACTERS, SIGNS
from spin8_dirac_one_edge_exact import (
    HOLDOUTS,
    VARIABLES,
    _determinant,
    polynomial_from_records,
)
from spin8_dirac_star import rational_circle


def _load_polynomials(report: dict[str, object]) -> dict[str, sp.Poly]:
    confirmation = report["confirmation"]
    return {
        name: polynomial_from_records(confirmation[name]["coefficients"])
        for name in ("F", "H1", "H2", "H3")
    }


def _predicted_components(
    parameters: tuple[sp.Rational, ...],
    polynomials: dict[str, sp.Poly],
) -> tuple[sp.Expr, dict[str, sp.Expr]]:
    pairs = tuple(rational_circle(value) for value in parameters)
    (a, A), (d, D), (e, E), (g, C), (cayley, sine) = pairs
    squared = tuple(pair[0] ** 2 for pair in pairs)
    substitutions = dict(zip(VARIABLES, squared, strict=True))
    values = {
        name: polynomial.as_expr().subs(substitutions)
        for name, polynomial in polynomials.items()
    }
    amplitudes = {
        "egc": A**2 * D**2 * e * E * g * C * cayley * sine**6 * values["H1"],
        "adgc": a * A**3 * d * D * E * g * C * cayley * sine**6 * values["H2"],
        "ade": a * A * d * D * e * sine**6 * values["H3"],
    }
    return values["F"], amplitudes


def _predicted_determinant(
    signs: tuple[int, ...], components: tuple[sp.Expr, dict[str, sp.Expr]]
) -> sp.Expr:
    invariant, amplitudes = components
    return sp.factor(
        invariant
        + sum(
            _character(signs, SIGN_CHARACTERS[name]) * amplitude
            for name, amplitude in amplitudes.items()
        )
    )


def _direct_worker(
    job: tuple[int, tuple[sp.Rational, ...], tuple[int, ...]],
) -> tuple[int, tuple[int, ...], str]:
    index, parameters, signs = job
    return index, signs, str(_determinant(parameters, signs))


def _value_digest(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def exact_holdout_certificate(
    reconstruction_path: Path, *, workers: int = 1
) -> dict[str, object]:
    source_bytes = reconstruction_path.read_bytes()
    reconstruction = json.loads(source_bytes)
    if not reconstruction["passed_reconstruction_gate"]:
        raise AssertionError("source reconstruction did not pass its frozen gate")
    polynomials = _load_polynomials(reconstruction)
    predicted_components = {
        index: _predicted_components(parameters, polynomials)
        for index, parameters in enumerate(HOLDOUTS)
    }
    jobs = [
        (index, parameters, signs)
        for index, parameters in enumerate(HOLDOUTS)
        for signs in SIGNS
    ]
    if workers == 1:
        direct_rows = map(_direct_worker, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        direct_rows = pool.map(_direct_worker, jobs, chunksize=4)

    rows: list[dict[str, object]] = []
    try:
        for index, signs, direct_text in direct_rows:
            direct = sp.Rational(direct_text)
            predicted = _predicted_determinant(signs, predicted_components[index])
            rows.append(
                {
                    "holdout_index": index,
                    "signs": list(signs),
                    "direct_determinant": direct_text,
                    "predicted_determinant": str(predicted),
                    "exact_match": direct == predicted,
                }
            )
    finally:
        if workers != 1:
            pool.shutdown()

    mismatch_count = sum(not row["exact_match"] for row in rows)
    return {
        "experiment": "variable-Cayley one-edge exact off-grid holdouts",
        "source_reconstruction_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "holdout_parameters": [
            [str(value) for value in parameters] for parameters in HOLDOUTS
        ],
        "sign_patterns_per_holdout": len(SIGNS),
        "exact_comparisons": len(rows),
        "mismatch_count": mismatch_count,
        "rows_sha256": _value_digest(rows),
        "rows": rows,
        "passed": len(rows) == len(HOLDOUTS) * len(SIGNS) and mismatch_count == 0,
    }


def verify_holdout_report(report: dict[str, object], reconstruction_path: Path) -> bool:
    source_bytes = reconstruction_path.read_bytes()
    if (
        report.get("source_reconstruction_sha256")
        != hashlib.sha256(source_bytes).hexdigest()
    ):
        return False
    rows = report.get("rows")
    if not isinstance(rows, list) or report.get("rows_sha256") != _value_digest(rows):
        return False
    reconstruction = json.loads(source_bytes)
    polynomials = _load_polynomials(reconstruction)
    predicted_components = {
        index: _predicted_components(parameters, polynomials)
        for index, parameters in enumerate(HOLDOUTS)
    }
    if len(rows) != len(HOLDOUTS) * len(SIGNS):
        return False
    expected_keys = set(itertools.product(range(len(HOLDOUTS)), SIGNS))
    observed_keys = {(int(row["holdout_index"]), tuple(row["signs"])) for row in rows}
    if observed_keys != expected_keys:
        return False
    for row in rows:
        index = int(row["holdout_index"])
        signs = tuple(int(value) for value in row["signs"])
        predicted = _predicted_determinant(signs, predicted_components[index])
        if sp.Rational(row["direct_determinant"]) != predicted:
            return False
        if sp.Rational(row["predicted_determinant"]) != predicted:
            return False
        if not row["exact_match"]:
            return False
    return (
        report.get("mismatch_count") == 0
        and report.get("exact_comparisons") == len(rows)
        and report.get("passed") is True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    report = exact_holdout_certificate(
        arguments.reconstruction, workers=arguments.workers
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "exact_comparisons": report["exact_comparisons"],
                "mismatch_count": report["mismatch_count"],
                "rows_sha256": report["rows_sha256"],
                "passed": report["passed"],
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit("exact off-grid holdout gate failed")


if __name__ == "__main__":
    main()
