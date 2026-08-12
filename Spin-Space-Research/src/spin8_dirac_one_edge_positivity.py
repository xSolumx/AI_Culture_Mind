"""Exact positivity certificate for the variable-Cayley one-edge theorem.

The determinant minor is difficult in the native tensor-product Bernstein
basis because its only bad controls lie next to the doubly-degenerate
``u=v=0`` face.  This module resolves that corner with two Duffy charts.

All coefficient signs in the final certificate are computed with integers.
Floating-point convolution is used nowhere in the proof path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp

from spin8_dirac_one_edge_exact import (
    VARIABLES,
    polynomial_from_records,
)

U, V, R, W, Z = VARIABLES
Y, T = sp.symbols("y t")


def _load_polynomials(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: polynomial_from_records(report["confirmation"][name]["coefficients"])
        for name in ("F", "H1", "H2", "H3")
    }


def _reduced_components(polynomials):
    F, H1, H2, H3 = (polynomials[name].as_expr() for name in ("F", "H1", "H2", "H3"))
    target = (1 - Z) ** 3 * (9 - Z) ** 2
    x = sp.expand(target - F)
    X = sp.cancel(x / (1 - Z) ** 3)
    P = sp.expand((1 - U) ** 2 * (1 - V) ** 2 * R * (1 - R) * W * (1 - W) * Z * H1**2)
    Q = sp.expand(U * (1 - U) ** 3 * V * (1 - V) * (1 - R) * W * (1 - W) * Z * H2**2)
    S = sp.expand(U * (1 - U) * V * (1 - V) * R * H3**2)
    C = sp.expand(
        U
        * (1 - U) ** 3
        * V
        * (1 - V) ** 2
        * R
        * (1 - R)
        * W
        * (1 - W)
        * Z
        * H1
        * H2
        * H3
    )
    return X, P, Q, S, C


def _reduced_determinant(polynomials):
    """Build the determinant with collected sparse polynomial products.

    Calling ``expand(X**4)`` asks SymPy to construct a multinomial expansion
    over every term of ``X`` before like monomials are collected.  Here ``X``
    has roughly 1,200 terms, so that path creates a combinatorial intermediate
    vastly larger than the final 203,978-term polynomial.  ``Poly`` products
    collect after each binary convolution and scale with the actual support.
    """

    F, H1, H2, H3 = (polynomials[name] for name in ("F", "H1", "H2", "H3"))
    target = sp.Poly((1 - Z) ** 3 * (9 - Z) ** 2, *VARIABLES)
    x = target - F
    X = x.exquo(sp.Poly((1 - Z) ** 3, *VARIABLES))
    P = (
        sp.Poly(
            (1 - U) ** 2 * (1 - V) ** 2 * R * (1 - R) * W * (1 - W) * Z,
            *VARIABLES,
        )
        * H1
        * H1
    )
    Q = (
        sp.Poly(
            U * (1 - U) ** 3 * V * (1 - V) * (1 - R) * W * (1 - W) * Z,
            *VARIABLES,
        )
        * H2
        * H2
    )
    S = sp.Poly(U * (1 - U) * V * (1 - V) * R, *VARIABLES) * H3 * H3
    C = (
        sp.Poly(
            U * (1 - U) ** 3 * V * (1 - V) ** 2 * R * (1 - R) * W * (1 - W) * Z,
            *VARIABLES,
        )
        * H1
        * H2
        * H3
    )
    return _sparse_tetrahedral_determinant(X, P, Q, S, C)


def _sparse_tetrahedral_determinant(X, P, Q, S, C):
    X2 = X * X
    diagonal = P + Q + S
    pairwise = P * Q + P * S + Q * S
    return (
        X2 * X2 - 2 * X2 * diagonal - 8 * X * C + P * P + Q * Q + S * S - 2 * pairwise
    )


def _integer_power_tensor(polynomial, variables):
    terms = polynomial.terms()
    denominator = math.lcm(*(int(sp.denom(coefficient)) for _, coefficient in terms))
    degrees = tuple(int(polynomial.degree(variable)) for variable in variables)
    tensor = np.empty(tuple(degree + 1 for degree in degrees), dtype=object)
    tensor.fill(0)
    for powers, coefficient in terms:
        tensor[powers] = int(coefficient * denominator)
    return tensor, denominator


def _complement_first_two(tensor):
    """Return coefficients of p(1-u, 1-v, ...)."""

    result = np.empty_like(tensor)
    result.fill(0)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            slab = tensor[i, j]
            for a in range(i + 1):
                for b in range(j + 1):
                    result[a, b] += (
                        (-1) ** (a + b) * math.comb(i, a) * math.comb(j, b) * slab
                    )
    return result


def _lower_duffy_power_tensor(tensor):
    """Apply u=t*y, v=t*(1-y) to an integer power tensor."""

    du = tensor.shape[0] - 1
    dv = tensor.shape[1] - 1
    trailing = tensor.shape[2:]
    degree = du + dv
    result = np.empty((degree + 1, degree + 1, *trailing), dtype=object)
    result.fill(0)
    for i in range(du + 1):
        for j in range(dv + 1):
            slab = tensor[i, j]
            for k in range(j + 1):
                result[i + j, i + k] += (-1) ** k * math.comb(j, k) * slab
    return result


def _integer_bernstein_tensor(power_tensor):
    """Convert a power tensor to Bernstein controls with one common scale.

    A global least-common-multiple per axis avoids rational object arrays.  If
    ``scale`` is returned, each actual Bernstein coefficient is exactly the
    corresponding integer divided by ``scale``.
    """

    current = power_tensor
    scale = 1
    for axis, size in enumerate(current.shape):
        degree = size - 1
        binomials = [math.comb(degree, k) for k in range(degree + 1)]
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


def _tensor_digest(tensor):
    digest = hashlib.sha256()
    for value in tensor.flat:
        digest.update(str(value).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _tensor_summary(tensor, denominator, *, direct_start=0):
    negative_by_first_axis = {}
    negative_count = 0
    zero_count = 0
    minimum = None
    minimum_positive = None
    direct_negative_count = 0
    for index, value in np.ndenumerate(tensor):
        if minimum is None or value < minimum:
            minimum = value
        if value < 0:
            negative_count += 1
            negative_by_first_axis[str(index[0])] = (
                negative_by_first_axis.get(str(index[0]), 0) + 1
            )
            if index[0] >= direct_start:
                direct_negative_count += 1
        elif value == 0:
            zero_count += 1
        elif minimum_positive is None or value < minimum_positive:
            minimum_positive = value
    return {
        "degrees": [size - 1 for size in tensor.shape],
        "coefficient_count": int(tensor.size),
        "negative_count": negative_count,
        "negative_by_first_axis": negative_by_first_axis,
        "direct_start": direct_start,
        "direct_negative_count": direct_negative_count,
        "zero_count": zero_count,
        "minimum": str(sp.Rational(minimum, denominator)),
        "minimum_positive": (
            str(sp.Rational(minimum_positive, denominator))
            if minimum_positive is not None
            else None
        ),
        "coefficients_sha256": _tensor_digest(tensor),
    }


def _rational_bernstein_summary(polynomial, variables):
    power, denominator = _integer_power_tensor(polynomial, variables)
    controls, scale = _integer_bernstein_tensor(power)
    return _tensor_summary(controls, denominator * scale), controls


def _symmetric_face_charts(G0):
    symmetric, remainder, mapping = sp.symmetrize(G0, [R, W], formal=True)
    if remainder != 0:
        raise AssertionError("u=v=0 face is not symmetric in r,w")
    s, p = mapping[0][0], mapping[1][0]
    expression = symmetric.subs({s: T, p: T**2 * (1 - Y) / 4})
    lower = sp.Poly(sp.expand(expression), T, Y, Z)
    expression = symmetric.subs({s: 2 - T, p: ((2 - T) ** 2 - T**2 * Y) / 4})
    upper = sp.Poly(sp.expand(expression), T, Y, Z)
    lower_summary, _ = _rational_bernstein_summary(lower, (T, Y, Z))
    upper_summary, _ = _rational_bernstein_summary(upper, (T, Y, Z))
    return {"lower_sum_chart": lower_summary, "upper_sum_chart": upper_summary}


def boundary_layer_certificate(polynomials):
    X, P, Q, S, C = _reduced_components(polynomials)
    G = sp.expand(X**2 - P)
    G0 = sp.factor(G.subs({U: 0, V: 0}))
    directional_derivative = Y * sp.diff(G, U).subs({U: 0, V: 0}) + (1 - Y) * sp.diff(
        G, V
    ).subs({U: 0, V: 0})
    first_factor = sp.Poly(sp.expand(G0 + directional_derivative / 12), Y, R, W, Z)
    first_summary, _ = _rational_bernstein_summary(first_factor, (Y, R, W, Z))
    charts = _symmetric_face_charts(G0)
    # In the compact tetrahedral determinant, the difference from G^2 is
    # built entirely from Q, S, and C.  Each has both value and first
    # derivatives zero on u=v=0.  This mechanically certifies that the first
    # two degree-24 Bernstein layers are exactly G0^2 and
    # G0*(G0 + directional_derivative/12), rather than trusting those formulas
    # as explanatory strings in the report.
    higher_components = (Q, S, C)
    remainder_zero_on_face = all(
        component.subs({U: 0, V: 0}) == 0 for component in higher_components
    )
    remainder_first_order_zero = all(
        derivative.subs({U: 0, V: 0}) == 0
        for component in higher_components
        for derivative in (sp.diff(component, U), sp.diff(component, V))
    )
    return {
        "duffy_layer_0_identity": "B0 = G0^2",
        "duffy_layer_1_identity": "B1 = G0 * (G0 + G0_prime/12)",
        "compact_remainder_zero_on_face": remainder_zero_on_face,
        "compact_remainder_first_order_zero_on_face": remainder_first_order_zero,
        "G0_symmetric_charts": charts,
        "G0_plus_directional_derivative_over_12": first_summary,
        "passed": (
            remainder_zero_on_face
            and remainder_first_order_zero
            and all(
                item["negative_count"] == 0
                for item in (*charts.values(), first_summary)
            )
        ),
    }


def _records_digest(records):
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_determinant_cache(reconstruction_path: Path, output: Path):
    """Build the large determinant once, then release SymPy before charts."""

    source_bytes = reconstruction_path.read_bytes()
    polynomials = _load_polynomials(reconstruction_path)
    determinant = _reduced_determinant(polynomials)
    records = [
        {"powers": list(powers), "coefficient": str(coefficient)}
        for powers, coefficient in determinant.terms()
    ]
    report = {
        "experiment": "variable-Cayley one-edge reduced determinant cache",
        "source_reconstruction_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "normalization": {
            "x_factor": "x=(1-z)^3 X",
            "determinant_factor": "det(K)=(1-z)^12 D",
            "reduced_determinant_terms": len(records),
        },
        "degrees": [int(determinant.degree(variable)) for variable in VARIABLES],
        "records_sha256": _records_digest(records),
        "coefficients": records,
    }
    output.write_text(
        json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return report


def _power_tensor_from_cache(cache):
    records = cache["coefficients"]
    if _records_digest(records) != cache["records_sha256"]:
        raise AssertionError("determinant-cache coefficient hash mismatch")
    denominator = math.lcm(
        *(int(sp.denom(sp.Rational(record["coefficient"]))) for record in records)
    )
    degrees = tuple(int(value) for value in cache["degrees"])
    tensor = np.empty(tuple(degree + 1 for degree in degrees), dtype=object)
    tensor.fill(0)
    for record in records:
        coefficient = sp.Rational(record["coefficient"])
        tensor[tuple(record["powers"])] = int(coefficient * denominator)
    return tensor, denominator


def certify_chart(cache_path: Path, output: Path, *, upper: bool):
    cache_bytes = cache_path.read_bytes()
    cache = json.loads(cache_bytes)
    power, denominator = _power_tensor_from_cache(cache)
    if upper:
        power = _complement_first_two(power)
    duffy_power = _lower_duffy_power_tensor(power)
    del power
    gc.collect()
    controls, scale = _integer_bernstein_tensor(duffy_power)
    del duffy_power
    gc.collect()
    summary = _tensor_summary(
        controls,
        denominator * scale,
        direct_start=0 if upper else 2,
    )
    report = {
        "experiment": "variable-Cayley one-edge exact Duffy chart",
        "determinant_cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
        "chart": ("1-u=t*y, 1-v=t*(1-y)" if upper else "u=t*y, v=t*(1-y)"),
        "domain": "u+v>=1" if upper else "u+v<=1",
        "controls": summary,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def certify_boundary(reconstruction_path: Path, output: Path):
    source_bytes = reconstruction_path.read_bytes()
    polynomials = _load_polynomials(reconstruction_path)
    F = polynomials["F"].as_expr()
    x = sp.expand((1 - Z) ** 3 * (9 - Z) ** 2 - F)
    if sp.rem(x, (1 - Z) ** 3, Z) != 0:
        raise AssertionError("x lacks the required (1-z)^3 factor")
    report = {
        "experiment": "variable-Cayley one-edge Duffy boundary layers",
        "source_reconstruction_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "boundary_layers": boundary_layer_certificate(polynomials),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _determinant_gate(lower_controls, upper_controls, boundary_layers):
    return (
        lower_controls["degrees"][0] == 24
        and lower_controls["direct_start"] == 2
        and lower_controls["direct_negative_count"] == 0
        and set(lower_controls["negative_by_first_axis"]).issubset({"0", "1"})
        and upper_controls["degrees"][0] == 24
        and upper_controls["direct_start"] == 0
        and upper_controls["negative_count"] == 0
        and boundary_layers["passed"]
    )


def _lower_order_gate(lower_order):
    return (
        lower_order["native_bernstein"]["x"]["negative_count"] == 0
        and lower_order["native_bernstein"]["x2_minus_q2"]["negative_count"] == 0
        and lower_order["native_bernstein"]["x2_minus_r2"]["negative_count"] == 0
        and lower_order["boundary_adapted_face"]["x2_minus_p2_proved"]
        and lower_order["cubic"]["proved_nonnegative"]
    )


def _holdout_gate(holdouts):
    return (
        holdouts["exact_comparisons"] == 256
        and holdouts["mismatch_count"] == 0
        and holdouts["passed"]
    )


def assemble(
    reconstruction_path: Path,
    cache_path: Path,
    lower_path: Path,
    upper_path: Path,
    boundary_path: Path,
    lower_order_path: Path,
    holdouts_path: Path,
):
    from spin8_dirac_one_edge_holdouts import verify_holdout_report

    source_bytes = reconstruction_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    cache_bytes = cache_path.read_bytes()
    cache_sha = hashlib.sha256(cache_bytes).hexdigest()
    cache = json.loads(cache_bytes)
    lower = json.loads(lower_path.read_bytes())
    upper = json.loads(upper_path.read_bytes())
    boundary = json.loads(boundary_path.read_bytes())
    lower_order_bytes = lower_order_path.read_bytes()
    lower_order = json.loads(lower_order_bytes)
    holdouts_bytes = holdouts_path.read_bytes()
    holdouts = json.loads(holdouts_bytes)
    if cache["source_reconstruction_sha256"] != source_sha:
        raise AssertionError("determinant cache is linked to another reconstruction")
    if lower["determinant_cache_sha256"] != cache_sha:
        raise AssertionError("lower chart is linked to another determinant cache")
    if upper["determinant_cache_sha256"] != cache_sha:
        raise AssertionError("upper chart is linked to another determinant cache")
    if boundary["source_reconstruction_sha256"] != source_sha:
        raise AssertionError("boundary certificate is linked to another reconstruction")
    if lower_order["source_reconstruction_sha256"] != source_sha:
        raise AssertionError(
            "lower-order certificate is linked to another reconstruction"
        )
    if holdouts["source_reconstruction_sha256"] != source_sha:
        raise AssertionError("holdout certificate is linked to another reconstruction")

    lower_controls = lower["controls"]
    upper_controls = upper["controls"]
    boundary_layers = boundary["boundary_layers"]
    determinant_passed = _determinant_gate(
        lower_controls, upper_controls, boundary_layers
    )
    lower_order_passed = _lower_order_gate(lower_order)
    holdouts_passed = _holdout_gate(holdouts) and verify_holdout_report(
        holdouts, reconstruction_path
    )
    return {
        "experiment": "variable-Cayley one-edge exact Duffy positivity certificate",
        "source_reconstruction_sha256": source_sha,
        "determinant_cache_sha256": cache_sha,
        "lower_order_certificate_sha256": hashlib.sha256(lower_order_bytes).hexdigest(),
        "exact_holdout_certificate_sha256": hashlib.sha256(holdouts_bytes).hexdigest(),
        "normalization": cache["normalization"],
        "lower_triangle": {
            "chart": lower["chart"],
            "domain": lower["domain"],
            "controls": lower_controls,
            "boundary_layers": boundary_layers,
        },
        "upper_triangle": {
            "chart": upper["chart"],
            "domain": upper["domain"],
            "controls": upper_controls,
        },
        "determinant_proved_nonnegative": determinant_passed,
        "lower_order_principal_minors_proved_nonnegative": lower_order_passed,
        "exact_off_grid_holdouts_passed": holdouts_passed,
        "theorem_proved": (
            determinant_passed and lower_order_passed and holdouts_passed
        ),
    }


def verify_assembled_report(
    report,
    reconstruction_path: Path,
    cache_path: Path,
    lower_order_path: Path,
    holdouts_path: Path,
):
    """Replay every lightweight acceptance predicate and cryptographic link.

    The million-control tensors are intentionally regenerated by the staged
    commands rather than embedded in the small assembled report.  This
    verifier does not pretend otherwise: it checks the published determinant
    cache, re-evaluates all stored exact holdout predictions, and recomputes
    every acceptance predicate represented in the assembled artifact.
    """

    from spin8_dirac_one_edge_holdouts import verify_holdout_report

    source_bytes = reconstruction_path.read_bytes()
    cache_bytes = cache_path.read_bytes()
    lower_order_bytes = lower_order_path.read_bytes()
    holdouts_bytes = holdouts_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    cache_sha = hashlib.sha256(cache_bytes).hexdigest()
    cache = json.loads(cache_bytes)
    lower_order = json.loads(lower_order_bytes)
    holdouts = json.loads(holdouts_bytes)
    if report.get("source_reconstruction_sha256") != source_sha:
        return False
    if report.get("determinant_cache_sha256") != cache_sha:
        return False
    if (
        report.get("lower_order_certificate_sha256")
        != hashlib.sha256(lower_order_bytes).hexdigest()
    ):
        return False
    if (
        report.get("exact_holdout_certificate_sha256")
        != hashlib.sha256(holdouts_bytes).hexdigest()
    ):
        return False
    if cache.get("source_reconstruction_sha256") != source_sha:
        return False
    if _records_digest(cache.get("coefficients", [])) != cache.get("records_sha256"):
        return False
    if report.get("normalization") != cache.get("normalization"):
        return False
    if not verify_holdout_report(holdouts, reconstruction_path):
        return False

    lower_triangle = report.get("lower_triangle", {})
    upper_triangle = report.get("upper_triangle", {})
    determinant_passed = _determinant_gate(
        lower_triangle.get("controls", {}),
        upper_triangle.get("controls", {}),
        lower_triangle.get("boundary_layers", {}),
    )
    lower_order_passed = _lower_order_gate(lower_order)
    holdouts_passed = _holdout_gate(holdouts)
    return (
        report.get("determinant_proved_nonnegative") is determinant_passed
        and report.get("lower_order_principal_minors_proved_nonnegative")
        is lower_order_passed
        and report.get("exact_off_grid_holdouts_passed") is holdouts_passed
        and report.get("theorem_proved")
        is (determinant_passed and lower_order_passed and holdouts_passed)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("determinant", "lower", "upper", "boundary", "assemble")
    )
    parser.add_argument("--reconstruction", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--lower", type=Path)
    parser.add_argument("--upper", type=Path)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--lower-order", type=Path)
    parser.add_argument("--holdouts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "determinant":
        report = build_determinant_cache(args.reconstruction, args.output)
    elif args.stage in {"lower", "upper"}:
        report = certify_chart(args.cache, args.output, upper=args.stage == "upper")
    elif args.stage == "boundary":
        report = certify_boundary(args.reconstruction, args.output)
    else:
        report = assemble(
            args.reconstruction,
            args.cache,
            args.lower,
            args.upper,
            args.boundary,
            args.lower_order,
            args.holdouts,
        )
    payload = json.dumps(report, indent=2) + "\n"
    if args.stage == "assemble":
        args.output.write_text(payload, encoding="utf-8")
    if args.stage == "determinant":
        print(
            json.dumps(
                {
                    "experiment": report["experiment"],
                    "source_reconstruction_sha256": report[
                        "source_reconstruction_sha256"
                    ],
                    "degrees": report["degrees"],
                    "term_count": len(report["coefficients"]),
                    "records_sha256": report["records_sha256"],
                },
                indent=2,
            )
        )
    else:
        print(payload)


if __name__ == "__main__":
    main()
