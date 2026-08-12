"""Independent FLINT checks for the Cayley and signed-star manuscripts.

The maintained theorem harnesses use SymPy to reconstruct their exact
polynomials.  This module transfers only rational coefficient maps into
python-flint, then independently performs the polynomial divisions,
differentiations, and Bernstein-basis transforms used by the two publication
certificates.  It therefore checks the exact arithmetic in a second computer
algebra system.  It does not independently derive the Spin(8) projectors.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from math import comb
from pathlib import Path

try:
    from flint import ctx, fmpq, fmpq_mpoly_ctx, fmpq_poly
except ImportError:  # pragma: no cover - exercised only in the minimal install
    ctx = fmpq = fmpq_mpoly_ctx = fmpq_poly = None

FLINT_AVAILABLE = fmpq_mpoly_ctx is not None


def _rational(text: str | int) -> fmpq:
    if isinstance(text, int):
        return fmpq(text)
    numerator, separator, denominator = text.partition("/")
    return fmpq(int(numerator), int(denominator)) if separator else fmpq(int(text))


def _from_records(context: object, records: list[dict[str, object]]) -> object:
    return context.from_dict(
        {
            tuple(int(power) for power in record["powers"]): _rational(
                str(record["coefficient"])
            )
            for record in records
        }
    )


def _exact_quotient(dividend: object, divisor: object) -> object:
    quotient, remainder = divmod(dividend, divisor)
    if remainder != 0:
        raise ArithmeticError("FLINT polynomial division left a nonzero remainder")
    return quotient


def _multidegree(polynomial: object, variable_count: int) -> tuple[int, ...]:
    coefficients = polynomial.to_dict()
    return tuple(
        int(max(powers[axis] for powers in coefficients))
        for axis in range(variable_count)
    )


def _bernstein_coefficients(polynomial: object) -> list[fmpq]:
    """Convert an FLINT power-basis polynomial on [0,1]^d exactly."""

    power = polynomial.to_dict()
    dimensions = len(next(iter(power)))
    degrees = _multidegree(polynomial, dimensions)
    coefficients: list[fmpq] = []
    for index in product(*(range(degree + 1) for degree in degrees)):
        value = fmpq(0)
        for powers, coefficient in power.items():
            if all(powers[axis] <= index[axis] for axis in range(dimensions)):
                weight = fmpq(1)
                for axis in range(dimensions):
                    weight *= fmpq(
                        comb(index[axis], powers[axis]),
                        comb(degrees[axis], powers[axis]),
                    )
                value += coefficient * weight
        coefficients.append(value)
    return coefficients


def _bernstein_summary(polynomial: object) -> dict[str, object]:
    coefficients = _bernstein_coefficients(polynomial)
    positives = [coefficient for coefficient in coefficients if coefficient > 0]
    degrees = _multidegree(polynomial, 4)
    return {
        "degrees": list(degrees),
        "coefficient_count": len(coefficients),
        "negative_count": int(
            sum(bool(coefficient < 0) for coefficient in coefficients)
        ),
        "zero_count": int(sum(bool(coefficient == 0) for coefficient in coefficients)),
        "minimum_positive_coefficient": str(min(positives)),
    }


def _cayley_certificate() -> dict[str, object]:
    # Coefficients are in ascending powers of z.
    one_minus_z = fmpq_poly([1, -1])
    nine_minus_z = fmpq_poly([9, -1])
    denominator = one_minus_z * nine_minus_z
    determinant = one_minus_z**3 * nine_minus_z**2 / 1024
    trace_inverse_numerator = fmpq_poly([387, -206, 11])
    trace_inverse_square_numerator = fmpq_poly([8883, 2676, 786, -76, 19])

    inverse_derivative_numerator = (
        trace_inverse_numerator.derivative() * denominator
        - trace_inverse_numerator * denominator.derivative()
    )
    expected_inverse_derivative_numerator = 96 * fmpq_poly([21, -6, 1])

    squared_denominator = denominator**2
    inverse_square_derivative_numerator = (
        trace_inverse_square_numerator.derivative() * squared_denominator
        - trace_inverse_square_numerator * squared_denominator.derivative()
    )
    sign_polynomial = fmpq_poly([12609, 336, -630, -8, -19])
    expected_inverse_square_derivative_numerator = 16 * denominator * sign_polynomial

    power = sign_polynomial.coeffs()
    degree = sign_polynomial.degree()
    sign_bernstein = [
        sum(
            (fmpq(comb(row, column), comb(degree, column)) * power[column])
            for column in range(row + 1)
        )
        for row in range(degree + 1)
    ]
    spectral_context = fmpq_mpoly_ctx.get(["c", "lambda"])
    c, eigenvalue = spectral_context.gens()
    block_zero = (
        -((eigenvalue - 1) ** 2)
        * (
            2 * c * eigenvalue
            - c
            - 2 * eigenvalue**3
            + 8 * eigenvalue**2
            - 6 * eigenvalue
            + 1
        )
        * (
            2 * c * eigenvalue
            - c
            + 2 * eigenvalue**3
            - 8 * eigenvalue**2
            + 6 * eigenvalue
            - 1
        )
        / 4
    )
    block_twin = (
        (c - 2 * eigenvalue**2 + 4 * eigenvalue - 1)
        * (c - 2 * eigenvalue**2 + 6 * eigenvalue - 3)
        * (c + 2 * eigenvalue**2 - 6 * eigenvalue + 3)
        * (c + 2 * eigenvalue**2 - 4 * eigenvalue + 1)
        / 16
    )
    endpoint_slopes = [
        block.derivative(0).subs({"c": 1, "lambda": 0})
        / (2 * block.derivative(1).subs({"c": 1, "lambda": 0}))
        for block in (block_zero, block_twin, block_twin)
    ]
    passed = bool(
        determinant == fmpq_poly([81, -261, 298, -138, 21, -1]) / 1024
        and inverse_derivative_numerator == expected_inverse_derivative_numerator
        and inverse_square_derivative_numerator
        == expected_inverse_square_derivative_numerator
        and sign_bernstein == [12609, 12693, 12672, 12544, 12288]
        and trace_inverse_numerator(0) / denominator(0) == 43
        and trace_inverse_square_numerator(0) / squared_denominator(0) == fmpq(329, 3)
        and endpoint_slopes == [fmpq(1, 8)] * 3
    )
    return {
        "determinant": str(determinant),
        "trace_inverse_derivative_numerator": str(inverse_derivative_numerator),
        "trace_inverse_square_derivative_numerator": str(
            inverse_square_derivative_numerator
        ),
        "inverse_square_sign_bernstein_coefficients": [
            str(coefficient) for coefficient in sign_bernstein
        ],
        "endpoint_small_eigenvalue_slopes_in_1_minus_z": [
            str(slope) for slope in endpoint_slopes
        ],
        "passed": passed,
    }


def _star_certificate(source_artifact: Path) -> dict[str, object]:
    source = json.loads(source_artifact.read_text(encoding="utf-8"))
    context = fmpq_mpoly_ctx.get(["u", "v", "w", "z"])
    u, v, w, z = context.gens()
    discovery = source["discovery_node_set"]
    confirmation = source["confirmation_node_set"]
    discovery_even = _from_records(context, discovery["even_coefficients"])
    discovery_odd = _from_records(context, discovery["odd_coefficients"])
    even = _from_records(context, confirmation["even_coefficients"])
    odd = _from_records(context, confirmation["odd_coefficients"])
    maps_match = discovery_even == even and discovery_odd == odd

    target = (1 - z) ** 3 * (9 - z) ** 2
    margin = target - even
    discriminant = margin**2 - u * v * w * (1 - u) * (1 - v) * (1 - w) * z * odd**2
    reduced_margin = _exact_quotient(margin, (1 - z) ** 3)
    reduced_odd = _exact_quotient(odd, (1 - u) * (v - w) * (1 - z) ** 3)
    reduced_discriminant = _exact_quotient(discriminant, (1 - z) ** 6)
    margin_summary = _bernstein_summary(reduced_margin)
    discriminant_summary = _bernstein_summary(reduced_discriminant)
    expected_margin = {
        "degrees": [3, 3, 3, 2],
        "coefficient_count": 192,
        "negative_count": 0,
        "zero_count": 3,
        "minimum_positive_coefficient": "32/3",
    }
    expected_discriminant = {
        "degrees": [6, 6, 6, 4],
        "coefficient_count": 1715,
        "negative_count": 0,
        "zero_count": 20,
        "minimum_positive_coefficient": "512/9",
    }
    exact_cubic_order_witnesses = {
        "u=1,v=w=z=0": even.subs({"u": 1, "v": 0, "w": 0, "z": 0}),
        "v=1,u=w=z=0": even.subs({"u": 0, "v": 1, "w": 0, "z": 0}),
        "w=1,u=v=z=0": even.subs({"u": 0, "v": 0, "w": 1, "z": 0}),
    }
    expected_order_witnesses = {
        "u=1,v=w=z=0": fmpq(25, 2),
        "v=1,u=w=z=0": fmpq(75, 2),
        "w=1,u=v=z=0": fmpq(75, 2),
    }
    passed = bool(
        maps_match
        and margin_summary == expected_margin
        and discriminant_summary == expected_discriminant
        and exact_cubic_order_witnesses == expected_order_witnesses
    )
    return {
        "source_artifact": source_artifact.name,
        "discovery_confirmation_maps_equal": maps_match,
        "reduced_orientation_polynomial": str(reduced_odd),
        "reduced_margin_bernstein": margin_summary,
        "reduced_discriminant_bernstein": discriminant_summary,
        "exact_cubic_order_witnesses": {
            key: str(value) for key, value in exact_cubic_order_witnesses.items()
        },
        "passed": passed,
    }


def run(*, source_artifact: Path, flint_threads: int = 6) -> dict[str, object]:
    if not FLINT_AVAILABLE:
        raise RuntimeError("python-flint is required; install the 'exact' extra")
    if not 1 <= flint_threads <= 7:
        raise ValueError("flint_threads must leave at least one CPU core free")
    ctx.threads = flint_threads
    cayley = _cayley_certificate()
    star = _star_certificate(source_artifact)
    return {
        "experiment": "independent FLINT publication-certificate cross-check",
        "flint_threads": flint_threads,
        "cayley_criteria": cayley,
        "signed_star_structure": star,
        "scope_boundary": (
            "FLINT independently checks rational polynomial arithmetic after "
            "the maintained coefficient maps have been supplied. It does not "
            "independently derive the Spin(8) projectors or the interpolation "
            "samples that produced those maps."
        ),
        "passed": bool(cayley["passed"] and star["passed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/spin8_dirac_star_20260804.json"),
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(
        source_artifact=arguments.source,
        flint_threads=arguments.threads,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("independent FLINT publication cross-check failed")


if __name__ == "__main__":
    main()
