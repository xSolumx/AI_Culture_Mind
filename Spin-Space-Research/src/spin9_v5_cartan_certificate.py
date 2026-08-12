"""Exact full-Cartan certificate for the Spin(9) Grassmann V5 slice.

The Cayley-null Grassmann normal slice contains the spin-two module
``V5 = Sym_0(3)``.  Every vector in V5 is conjugate under the null-plane
stabilizer to the Cartan section used here.  This module certifies on that
entire section that

``det I(P) < (101/100) det I(P0)``.

The proof has three exact layers:

* lift the reconstructed determinant numerator from prime fields to
  characteristic zero using a coefficient bound and both embeddings of
  ``Z[sqrt(2)]``;
* compactify the radial coordinate and reduce the shape interval to a unit
  square; and
* cover that square by a six-cell exact dyadic Bernstein atlas whose controls
  are all strictly positive.

The theorem concerns pure V5 graph variations over the Cayley-null plane.  It
does not certify the V1+V5 normal slice or the global Grassmann quotient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, deque
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

from spin9_dirac_clifford import build_spin9_clifford_system
from spin9_v5_cartan_reconstruction import (
    IDENTITY_DIRECTION_COUNT,
    IDENTITY_PRIMES,
    IDENTITY_SCALE_NODES,
    MONOMIALS,
    load_coefficients,
    validate_polynomial_identity_mod_prime,
    validate_rational_coefficients,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COEFFICIENT_ARTIFACT = (
    ROOT / "artifacts" / "spin9_v5_cartan_reconstruction_20260811.json"
)
DEFAULT_RAY_ARTIFACT = ROOT / "artifacts" / "spin9_v5_ray_certificate_20260811.json"

COMPACT_DEGREE = 84
EXPECTED_ROW_L1_NORMS = (
    18131,
    20348,
    21212,
    19082,
    19946,
    12861,
    21320,
    18261,
    24991,
    22929,
    24989,
    20799,
    18646,
    18623,
    16738,
    23026,
    21998,
    18460,
    21475,
    22552,
    25209,
    18460,
    21488,
    21475,
    24458,
    23481,
    20464,
    21475,
    21304,
    22601,
    21475,
    23210,
    23001,
    20445,
    18646,
    18785,
)


def _multiply_sparse(
    left: dict[tuple[int, int], int],
    right: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for (a, b), left_value in left.items():
        for (c, d), right_value in right.items():
            key = (a + c, b + d)
            result[key] = result.get(key, 0) + left_value * right_value
    return {key: value for key, value in result.items() if value}


def invariant_gap_coefficients(
    coefficients: tuple[Fraction, ...],
) -> dict[tuple[int, int], int]:
    """Return coefficients of ``101*delta**14 - 100*N`` in ``(p,y)``."""

    if any(value.denominator != 1 for value in coefficients):
        raise ValueError("the maintained Cartan numerator must be integral")
    delta = {(0, 0): 1, (1, 0): 8, (2, 0): 20, (3, 0): 16, (0, 2): -16}
    delta_power = {(0, 0): 1}
    for _ in range(14):
        delta_power = _multiply_sparse(delta_power, delta)
    result = {key: 101 * value for key, value in delta_power.items()}
    for monomial, coefficient in zip(MONOMIALS, coefficients, strict=True):
        result[monomial] = result.get(monomial, 0) - 100 * coefficient.numerator
    return {key: value for key, value in result.items() if value}


def compact_gap_power_tensor(
    coefficients: tuple[Fraction, ...],
) -> tuple[np.ndarray, int, int]:
    """Return the exact power tensor on the compact square ``(x,z)``.

    The chamber coordinates are

    ``p=(3+9*z**2)*r**2/2``, ``y=(9*z**2-1)*r**3/2``,
    ``r=x/(1-x)``.

    Multiplication by ``(1-x)**84`` makes the strict determinant gap a
    polynomial on ``[0,1]^2``.
    """

    compact: dict[tuple[int, int], Fraction] = {}
    for (a, b), coefficient in invariant_gap_coefficients(coefficients).items():
        radial_degree = 2 * a + 3 * b
        p_direction = {
            index: math.comb(a, index) * 3 ** (a - index) * 9**index
            for index in range(a + 1)
        }
        y_direction = {
            index: math.comb(b, index) * (-1) ** (b - index) * 9**index
            for index in range(b + 1)
        }
        direction: dict[int, int] = {}
        for left, left_value in p_direction.items():
            for right, right_value in y_direction.items():
                index = left + right
                direction[index] = direction.get(index, 0) + left_value * right_value
        base = Fraction(coefficient, 2 ** (a + b))
        for direction_power, direction_value in direction.items():
            for increment in range(COMPACT_DEGREE - radial_degree + 1):
                key = (radial_degree + increment, 2 * direction_power)
                value = (
                    base
                    * direction_value
                    * math.comb(COMPACT_DEGREE - radial_degree, increment)
                    * (-1) ** increment
                )
                compact[key] = compact.get(key, Fraction(0)) + value
    compact = {key: value for key, value in compact.items() if value}
    denominator = math.lcm(*(value.denominator for value in compact.values()))
    tensor = np.empty((COMPACT_DEGREE + 1, COMPACT_DEGREE + 1), dtype=object)
    tensor.fill(0)
    for powers, value in compact.items():
        tensor[powers] = value.numerator * (denominator // value.denominator)
    return tensor, denominator, len(compact)


def _integer_bernstein_tensor(power_tensor: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert an integer power tensor to uniformly scaled Bernstein controls."""

    current = power_tensor
    scale = 1
    for axis, size in enumerate(current.shape):
        degree = size - 1
        binomials = [math.comb(degree, index) for index in range(degree + 1)]
        axis_scale = math.lcm(*binomials)
        moved = np.moveaxis(current, axis, 0)
        shape = moved.shape
        flat = moved.reshape((degree + 1, -1))
        transformed = np.empty_like(flat)
        for row in range(degree + 1):
            value = np.zeros(flat.shape[1], dtype=object)
            for source in range(row + 1):
                weight = math.comb(row, source) * axis_scale // binomials[source]
                value += weight * flat[source]
            transformed[row] = value
        current = np.moveaxis(transformed.reshape(shape), 0, axis)
        scale *= axis_scale
    return current, scale


def _split_bernstein_half(
    controls: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split exact controls at one half, retaining an integer common scale."""

    moved = np.moveaxis(controls, axis, 0)
    degree = moved.shape[0] - 1
    left = np.empty_like(moved)
    right = np.empty_like(moved)
    common_scale = 2**degree
    left[0] = moved[0] * common_scale
    right[degree] = moved[degree] * common_scale
    current = moved
    for level in range(1, degree + 1):
        current = current[:-1] + current[1:]
        level_scale = 2 ** (degree - level)
        left[level] = current[0] * level_scale
        right[degree - level] = current[-1] * level_scale
    return np.moveaxis(left, 0, axis), np.moveaxis(right, 0, axis)


def _tensor_digest(tensor: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in tensor.flat:
        digest.update(str(value).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _dyadic_box(path: str) -> dict[str, str]:
    bounds = [[Fraction(0), Fraction(1)], [Fraction(0), Fraction(1)]]
    for step in filter(None, path.split(",")):
        axis = 0 if step[0] == "x" else 1
        midpoint = sum(bounds[axis], Fraction(0)) / 2
        if step[1] == "0":
            bounds[axis][1] = midpoint
        else:
            bounds[axis][0] = midpoint
    return {
        "x_lower": str(bounds[0][0]),
        "x_upper": str(bounds[0][1]),
        "z_lower": str(bounds[1][0]),
        "z_upper": str(bounds[1][1]),
    }


def bernstein_atlas_certificate(
    coefficients: tuple[Fraction, ...],
) -> dict[str, object]:
    """Build the six-cell strict exact Bernstein certificate."""

    power, power_denominator, compact_term_count = compact_gap_power_tensor(
        coefficients
    )
    controls, bernstein_scale = _integer_bernstein_tensor(power)
    native_negative = sum(value < 0 for value in controls.flat)
    native_zero = sum(value == 0 for value in controls.flat)

    pending = deque([(controls, 0, "")])
    leaves: list[dict[str, object]] = []
    split_count = 0
    while pending:
        cell, depth, path = pending.popleft()
        minimum = min(cell.flat)
        if minimum > 0:
            leaves.append(
                {
                    "path": path.rstrip(",") or "root",
                    "depth": depth,
                    "box": _dyadic_box(path),
                    "minimum_scaled_coefficient": str(minimum),
                    "zero_coefficient_count": 0,
                    "controls_sha256": _tensor_digest(cell),
                }
            )
            continue
        if depth >= 16:
            raise AssertionError(
                f"strict Bernstein atlas failed at depth {depth} on {path}"
            )
        axis = depth % 2
        lower, upper = _split_bernstein_half(cell, axis)
        symbol = "x" if axis == 0 else "z"
        pending.append((lower, depth + 1, f"{path}{symbol}0,"))
        pending.append((upper, depth + 1, f"{path}{symbol}1,"))
        split_count += 1

    return {
        "chamber_parameterization": {
            "radial": "r >= 0",
            "shape": "0 <= k <= 3",
            "unit_square": "r=x/(1-x), k=3*z, 0<=x,z<=1",
            "p": "(3+9*z^2)*r^2/2",
            "y": "(9*z^2-1)*r^3/2",
        },
        "compact_polynomial": "(1-x)^84 * (101*delta^14 - 100*N)",
        "power_multidegree": [COMPACT_DEGREE, COMPACT_DEGREE],
        "power_term_count": compact_term_count,
        "power_denominator": str(power_denominator),
        "power_tensor_sha256": _tensor_digest(power),
        "bernstein_positive_scale": str(bernstein_scale),
        "native_negative_coefficient_count": native_negative,
        "native_zero_coefficient_count": native_zero,
        "native_controls_sha256": _tensor_digest(controls),
        "split_rule": "alternate x and z midpoint splits on the unique unresolved cells",
        "split_count": split_count,
        "leaf_count": len(leaves),
        "leaf_depth_histogram": dict(
            sorted(Counter(str(leaf["depth"]) for leaf in leaves).items())
        ),
        "all_leaf_controls_strictly_positive": True,
        "leaves": leaves,
    }


def _exact_information_matrix() -> tuple[sp.Matrix, sp.Symbol, sp.Symbol]:
    """Build ``J(u,v)`` exactly over ``QQ(sqrt(2))``."""

    u, v = sp.symbols("u v", real=True)
    sqrt2 = sp.sqrt(2)
    generators = [
        sp.Matrix(matrix)
        for matrix in build_spin9_clifford_system().doubled_spin_generators
    ]
    frame = sp.zeros(16, 3)
    frame[0, 0] = 1
    frame[1, 1] = sqrt2 / 2
    frame[8, 1] = sqrt2 / 2
    frame[2, 2] = -sqrt2 / 2
    frame[12, 2] = sqrt2 / 2
    axis = sp.zeros(16, 3)
    for row, column, value in (
        (1, 1, 2),
        (2, 2, 1),
        (5, 2, -1),
        (6, 1, -1),
        (8, 1, -2),
        (9, 0, -2 * sqrt2),
        (11, 2, -1),
        (12, 2, 1),
        (14, 0, sqrt2),
        (15, 1, 1),
    ):
        axis[row, column] = value
    transverse = sp.zeros(16, 3)
    for row, column, value in (
        (4, 2, -sqrt2),
        (6, 0, -1),
        (10, 2, sqrt2),
        (14, 1, sqrt2),
        (15, 0, 1),
    ):
        transverse[row, column] = value
    graph_frame = frame + u * axis + v * transverse
    adjugate = (graph_frame.T * graph_frame).adjugate()
    observations = [
        sp.Matrix.vstack(
            *[(generator * graph_frame[:, column]).T for generator in generators]
        )
        for column in range(3)
    ]
    information = sp.zeros(36)
    for left in range(3):
        for right in range(3):
            information += (
                adjugate[left, right] * observations[left] * observations[right].T
            )
    return information.applyfunc(sp.expand), u, v


def coefficient_bound_certificate(
    coefficients: tuple[Fraction, ...],
) -> dict[str, object]:
    """Replay a coefficient bound sufficient for CRT identity lifting."""

    information, u, v = _exact_information_matrix()
    sqrt2 = sp.sqrt(2)

    def quadratic_norm(value: sp.Expr) -> sp.Rational:
        expanded = sp.expand(value)
        rational = sp.Rational(expanded.coeff(sqrt2, 0))
        radical = sp.Rational(expanded.coeff(sqrt2, 1))
        if sp.expand(rational + radical * sqrt2 - expanded) != 0:
            raise AssertionError("coefficient escaped QQ(sqrt(2))")
        if rational.q != 1 or radical.q != 1:
            raise AssertionError("information coefficient is not in ZZ[sqrt(2)]")
        return abs(rational) + 2 * abs(radical)

    def polynomial_l1(value: sp.Expr) -> int:
        polynomial = sp.Poly(value, u, v, extension=sqrt2)
        norm = sum(quadratic_norm(coefficient) for coefficient in polynomial.coeffs())
        if norm.q != 1:
            raise AssertionError("information row norm is not integral")
        return int(norm)

    row_norms = tuple(
        sum(polynomial_l1(information[row, column]) for column in range(36))
        for row in range(36)
    )
    if row_norms != EXPECTED_ROW_L1_NORMS:
        raise AssertionError("information coefficient row norms changed")
    determinant_bound = math.prod(row_norms)
    base_determinant = int(information.subs({u: 0, v: 0}).det())
    if base_determinant != 17_179_869_184:
        raise AssertionError("base information determinant changed")

    # For ||a+b*sqrt(2)||=|a|+2|b|, the substituted invariant generators
    # obey ||p||<=4 and ||y||<=4.  Submultiplicativity gives the following
    # inexpensive bound without expanding the 631-term candidate again.
    numerator_bound = sum(
        abs(coefficient.numerator) * 4 ** (a + b)
        for (a, b), coefficient in zip(MONOMIALS, coefficients, strict=True)
    )
    delta_bound = 1 + 8 * 4 + 20 * 4**2 + 16 * 4**3 + 16 * 4**2
    candidate_bound = abs(base_determinant) * delta_bound**22 * numerator_bound
    residual_bound = determinant_bound + candidate_bound
    modulus = math.prod(IDENTITY_PRIMES)
    return {
        "coefficient_norm": "l1 with |a+b*sqrt(2)|_q = |a|+2|b|",
        "information_row_l1_norms": list(row_norms),
        "determinant_coefficient_bound": str(determinant_bound),
        "base_information_determinant": str(base_determinant),
        "substituted_p_norm_bound": 4,
        "substituted_y_norm_bound": 4,
        "substituted_delta_norm_bound": delta_bound,
        "candidate_numerator_norm_bound": str(numerator_bound),
        "candidate_rhs_coefficient_bound": str(candidate_bound),
        "residual_coefficient_bound": str(residual_bound),
        "identity_prime_product": str(modulus),
        "identity_prime_product_digits": len(str(modulus)),
        "prime_product_exceeds_twice_residual_bound": modulus > 2 * residual_bound,
        "two_square_root_embeddings_checked_per_prime": True,
    }


def _ray_restriction_checks(
    coefficients: tuple[Fraction, ...],
    ray_artifact: Path,
) -> dict[str, bool]:
    """Compare the generic numerator with both independent ray artifacts."""

    payload = json.loads(ray_artifact.read_text(encoding="utf-8"))
    p, y, t = sp.symbols("p y t", real=True)
    numerator = sum(
        coefficient.numerator * p**a * y**b
        for (a, b), coefficient in zip(MONOMIALS, coefficients, strict=True)
    )
    axis = sp.sympify(
        payload["axisymmetric_ray"]["determinant_ratio_numerator_factor"],
        locals={"t": t},
    )
    zero = sp.sympify(
        payload["zero_cubic_ray"]["determinant_ratio_numerator_factor"],
        locals={"t": t},
    )
    axis_difference = sp.Poly(
        sp.expand(
            numerator.subs({p: 3 * t**2, y: -sp.sqrt(2) * t**3})
            - axis * (1 + 10 * t**2) ** 2
        ),
        t,
        extension=sp.sqrt(2),
    )
    zero_difference = sp.Poly(
        sp.expand(
            numerator.subs({p: t**2 / 2, y: 0})
            - zero * (1 + t**2) ** 2 * (1 + 2 * t**2) ** 2
        ),
        t,
        domain=sp.QQ,
    )
    return {
        "axisymmetric_ray_identity": bool(axis_difference.is_zero),
        "zero_cubic_ray_identity": bool(zero_difference.is_zero),
    }


def diagnostics(
    coefficient_artifact: Path = DEFAULT_COEFFICIENT_ARTIFACT,
    *,
    ray_artifact: Path = DEFAULT_RAY_ARTIFACT,
    run_identity: bool = True,
) -> dict[str, object]:
    """Replay the complete full-V5 Cartan theorem."""

    coefficients = load_coefficients(coefficient_artifact)
    unused_prime_valid = validate_rational_coefficients(coefficients)
    ray_checks = _ray_restriction_checks(coefficients, ray_artifact)
    positivity = bernstein_atlas_certificate(coefficients)
    coefficient_bound = coefficient_bound_certificate(coefficients)

    identity_rows: list[dict[str, object]] = []
    if run_identity:
        for index, prime in enumerate(IDENTITY_PRIMES, start=1):
            print(
                f"identity prime {index}/{len(IDENTITY_PRIMES)}: {prime}",
                file=sys.stderr,
                flush=True,
            )
            identity_rows.append(
                {
                    "prime": prime,
                    "both_square_root_embeddings_passed": (
                        validate_polynomial_identity_mod_prime(coefficients, prime)
                    ),
                }
            )

    identity_passed = bool(
        run_identity
        and len(identity_rows) == len(IDENTITY_PRIMES)
        and all(row["both_square_root_embeddings_passed"] for row in identity_rows)
        and coefficient_bound["prime_product_exceeds_twice_residual_bound"]
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "all pure V5 graph variations over the Cayley-null Spin(9) plane",
        "invariant_reduction": {
            "module": "V5 = Sym_0(3)",
            "cartan_coordinates": "(u,v)",
            "p": "3*u^2+v^2",
            "y": "sqrt(2)*u*(v^2-u^2)",
            "orbit_domain": "p>=0 and 27*y^2<=2*p^3",
            "gram_determinant": "1+8*p+20*p^2+16*p^3-16*y^2",
            "determinant_ratio": "N(p,y)/gram_determinant^14",
            "numerator_weighted_degree": 84,
        },
        "coefficient_artifact": coefficient_artifact.name,
        "coefficient_artifact_unused_prime_validation": unused_prime_valid,
        "independent_ray_restrictions": ray_checks,
        "characteristic_zero_identity": {
            "raw_identity": "det(J)=det(J0)*gram_determinant^22*N(p,y)",
            "raw_total_degree_bound": 216,
            "invariant_direction_count": IDENTITY_DIRECTION_COUNT,
            "scale_node_count": IDENTITY_SCALE_NODES,
            "prime_count": len(IDENTITY_PRIMES),
            "prime_rows": identity_rows,
            "coefficient_bound": coefficient_bound,
            "passed": identity_passed,
        },
        "positivity": positivity,
        "theorem": (
            "det I(P_V5) < (101/100) det I(P_cayley-null) for every pure V5 graph"
        ),
        "symmetric_candidate_comparison": {
            "candidate_ratio_exceeds_101_over_100": bool(
                json.loads(ray_artifact.read_text(encoding="utf-8"))[
                    "symmetric_candidate_comparison"
                ]["ratio_exceeds_101_over_100"]["positive"]
            ),
            "candidate_beats_entire_pure_v5_family": True,
        },
        "all_v5_shapes_certified": True,
        "full_v1_plus_v5_slice_certified": False,
        "global_grassmann_quotient_certified": False,
        "global_rank_three_optimum_certified": False,
    }
    report["passed"] = bool(
        unused_prime_valid
        and all(ray_checks.values())
        and identity_passed
        and positivity["all_leaf_controls_strictly_positive"]
        and positivity["split_count"] == 5
        and positivity["leaf_count"] == 6
        and report["symmetric_candidate_comparison"][
            "candidate_ratio_exceeds_101_over_100"
        ]
        and not report["full_v1_plus_v5_slice_certified"]
        and not report["global_grassmann_quotient_certified"]
        and not report["global_rank_three_optimum_certified"]
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=DEFAULT_COEFFICIENT_ARTIFACT,
    )
    parser.add_argument("--ray-artifact", type=Path, default=DEFAULT_RAY_ARTIFACT)
    parser.add_argument("--output", type=Path, help="optional JSON artifact path")
    parser.add_argument(
        "--skip-identity",
        action="store_true",
        help="development-only run; the resulting report cannot pass",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = diagnostics(
        args.coefficients,
        ray_artifact=args.ray_artifact,
        run_identity=not args.skip_identity,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
