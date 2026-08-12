"""Classify the equality set of the variable-Cayley one-edge theorem.

The published Duffy proof establishes nonnegativity.  This companion
certificate reads the exact determinant cache and classifies the zero support
of its Bernstein controls.  The classification is stronger than open-cube
positivity: it identifies every zero of the reduced determinant on the closed
five-cube.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp

from spin8_dirac_one_edge_positivity import (
    R,
    T,
    U,
    V,
    W,
    Y,
    Z,
    _integer_bernstein_tensor,
    _load_polynomials,
    _lower_duffy_power_tensor,
    _power_tensor_from_cache,
    _rational_bernstein_summary,
)
from spin8_resource_limits import constrain_current_process


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zero_rows(array: np.ndarray) -> list[tuple[int, ...]]:
    return [tuple(map(int, row)) for row in np.argwhere(array == 0)]


def _negative_rows(array: np.ndarray) -> list[tuple[int, ...]]:
    return [tuple(map(int, row)) for row in np.argwhere(array < 0)]


def _exceptional_face_certificate(reconstruction_path: Path) -> dict[str, object]:
    polynomials = _load_polynomials(reconstruction_path)
    even = polynomials["F"].as_expr()
    odd_one = polynomials["H1"].as_expr()
    target = (1 - Z) ** 3 * (9 - Z) ** 2
    reduced_center = sp.cancel((target - even) / (1 - Z) ** 3)
    first_square = sp.expand(
        (1 - U) ** 2 * (1 - V) ** 2 * R * (1 - R) * W * (1 - W) * Z * odd_one**2
    )
    face = sp.factor((reduced_center**2 - first_square).subs({U: 0, V: 0}))
    symmetric, remainder, mapping = sp.symmetrize(face, [R, W], formal=True)
    if remainder != 0:
        raise AssertionError("exceptional face is not symmetric in r,w")
    sum_variable, product_variable = mapping[0][0], mapping[1][0]
    lower_expression = symmetric.subs(
        {
            sum_variable: T,
            product_variable: T**2 * (1 - Y) / 4,
        }
    )
    upper_expression = symmetric.subs(
        {
            sum_variable: 2 - T,
            product_variable: ((2 - T) ** 2 - T**2 * Y) / 4,
        }
    )
    lower_summary, lower_controls = _rational_bernstein_summary(
        sp.Poly(sp.expand(lower_expression), T, Y, Z), (T, Y, Z)
    )
    upper_summary, upper_controls = _rational_bernstein_summary(
        sp.Poly(sp.expand(upper_expression), T, Y, Z), (T, Y, Z)
    )
    expected_lower_zeros = {
        (radial, angle, cayley)
        for radial in (0, 1)
        for angle in range(5)
        for cayley in range(5)
    } | {(2, 0, 4)}
    lower_zeros = set(_zero_rows(lower_controls))
    passed = bool(
        lower_zeros == expected_lower_zeros
        and not _negative_rows(lower_controls)
        and np.all(lower_controls[3] > 0)
        and np.all(lower_controls[8] > 0)
        and not _zero_rows(upper_controls)
        and not _negative_rows(upper_controls)
    )
    return {
        "face": "u=v=0",
        "symmetric_coordinates": (
            "lower: t=r+w, y=(r-w)^2/t^2; " "upper: t=2-r-w, y=(r-w)^2/t^2"
        ),
        "lower_chart": lower_summary,
        "lower_zero_support": ("radial Bernstein layers 0 and 1, plus index (2,0,4)"),
        "upper_chart": upper_summary,
        "upper_is_strictly_positive": True,
        "face_zero_set": "r=w=0",
        "passed": passed,
    }


def _full_lower_chart_certificate(cache_path: Path) -> dict[str, object]:
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    power, denominator = _power_tensor_from_cache(cache)
    duffy_power = _lower_duffy_power_tensor(power)
    controls, scale = _integer_bernstein_tensor(duffy_power)
    zeros = _zero_rows(controls)
    negatives = _negative_rows(controls)
    zero_rule = all(
        radial + first_tail + second_tail <= 3
        for radial, _angle, first_tail, second_tail, _cayley in zeros
    )
    expected_zero_count = sum(
        25 * 9 * sum(1 for r in range(13) for w in range(13) if t + r + w <= 3)
        for t in range(25)
    )
    exact_support = bool(
        len(zeros) == expected_zero_count
        and zero_rule
        and all(row[0] in (0, 1) for row in negatives)
        and np.all(controls[4] > 0)
        and np.all(controls[24] > 0)
    )
    return {
        "chart": "u=t*y, v=t*(1-y)",
        "degrees": [size - 1 for size in controls.shape],
        "zero_count": len(zeros),
        "zero_support_rule": "k_t+k_r+k_w<=3; k_y and k_z arbitrary",
        "negative_count": len(negatives),
        "negative_radial_layers": sorted({row[0] for row in negatives}),
        "radial_layer_4_strictly_positive": bool(np.all(controls[4] > 0)),
        "radial_layer_24_strictly_positive": bool(np.all(controls[24] > 0)),
        "common_integer_scale": str(denominator * scale),
        "passed": exact_support,
    }


def run(
    reconstruction_path: Path,
    cache_path: Path,
    assembled_path: Path,
    *,
    workers: int = 6,
) -> dict[str, object]:
    resource = constrain_current_process(workers=workers)
    assembled = json.loads(assembled_path.read_text(encoding="utf-8"))
    prior_theorem_contract = bool(
        assembled.get("theorem_proved")
        and assembled.get("determinant_proved_nonnegative")
        and assembled.get("determinant_cache_sha256") == _sha256(cache_path)
        and assembled["upper_triangle"]["controls"]["negative_count"] == 0
        and assembled["upper_triangle"]["controls"]["zero_count"] == 0
        and assembled["lower_triangle"]["boundary_layers"]["passed"]
    )
    face = _exceptional_face_certificate(reconstruction_path)
    lower = _full_lower_chart_certificate(cache_path)
    passed = bool(prior_theorem_contract and face["passed"] and lower["passed"])
    return {
        "theorem": "complete variable-Cayley one-edge equality classification",
        "source_reconstruction_sha256": _sha256(reconstruction_path),
        "determinant_cache_sha256": _sha256(cache_path),
        "assembled_theorem_sha256": _sha256(assembled_path),
        "prior_one_edge_theorem_contract_passed": prior_theorem_contract,
        "exceptional_face": face,
        "full_lower_duffy_chart": lower,
        "reduced_determinant_zero_set": "u=v=r=w=0",
        "complete_equality_set": "z=1 or (u,v,r,w)=(0,0,0,0)",
        "proof_logic": (
            "The upper Duffy chart is strictly positive. In the lower chart, "
            "the already-certified radial layers 0 and 1 are nonnegative; "
            "for 0<t<1 the strictly positive layer 4 is active, and at t=1 "
            "the strictly positive layer 24 is active. At t=0 the determinant "
            "is G0^2, whose exact symmetric-chart support vanishes only at "
            "r=w=0. Restoring det(K)=(1-z)^12 D adds z=1."
        ),
        "resource_contract": resource,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruction",
        type=Path,
        default=Path("artifacts/spin8_dirac_one_edge_exact_20260804.json"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/spin8_dirac_one_edge_determinant_cache_20260806.json"),
    )
    parser.add_argument(
        "--assembled",
        type=Path,
        default=Path("artifacts/spin8_dirac_one_edge_duffy_20260806.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        arguments.reconstruction,
        arguments.cache,
        arguments.assembled,
        workers=arguments.workers,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("one-edge equality classification failed")


if __name__ == "__main__":
    main()
