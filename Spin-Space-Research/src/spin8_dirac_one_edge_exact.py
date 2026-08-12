"""Exact reconstruction for the variable-Cayley one-edge Dirac--Gram gate."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp

from spin8_cayley_spectrum import symbolic_query_projector, symbolic_triality_generators
from spin8_dirac_edge import _character
from spin8_dirac_one_edge import SIGN_CHARACTERS, _symbolic_vector
from spin8_dirac_star import rational_circle

VARIABLES = sp.symbols("u v r w z")
U, V, R, W, Z = VARIABLES
SECTOR_ROWS = (
    (1, 1, 1, 1, 1),
    (1, 1, 1, 1, -1),
    (1, 1, -1, 1, 1),
    (1, 1, -1, 1, -1),
)
HADAMARD_INVERSE = (
    sp.Matrix(((1, 1, 1, 1), (1, -1, -1, 1), (1, -1, 1, -1), (1, 1, -1, -1))) / 4
)
NODE_SETS = {
    "discovery": (
        (sp.Rational(1, 10), sp.Rational(1, 5), sp.Rational(2, 5), sp.Rational(3, 5)),
        (
            sp.Rational(1, 12),
            sp.Rational(1, 6),
            sp.Rational(1, 3),
            sp.Rational(1, 2),
            sp.Rational(2, 3),
            sp.Rational(3, 4),
        ),
    ),
    "confirmation": (
        (
            sp.Rational(1, 11),
            sp.Rational(2, 11),
            sp.Rational(4, 11),
            sp.Rational(6, 11),
        ),
        (
            sp.Rational(1, 13),
            sp.Rational(2, 13),
            sp.Rational(3, 13),
            sp.Rational(5, 13),
            sp.Rational(7, 13),
            sp.Rational(9, 13),
        ),
    ),
}
EXPECTED_DEGREES = {
    "F": (3, 3, 3, 3, 5),
    "H1": (2, 2, 2, 2, 1),
    "H2": (1, 2, 2, 2, 1),
    "H3": (2, 2, 2, 3, 1),
}
HOLDOUTS = tuple(
    tuple(sp.Rational((2 * row + axis) % 9 + 1, 17 + axis) for axis in range(5))
    for row in range(8)
)


@lru_cache(maxsize=1)
def _context():
    generators = symbolic_triality_generators()
    basis = [[sp.Integer(i == j) for j in range(8)] for i in range(8)]
    fixed = symbolic_query_projector(
        0, basis[0], generators
    ) + symbolic_query_projector(1, basis[0], generators)
    return generators, basis, fixed


def _determinant(parameters, signs):
    generators, basis, fixed = _context()
    pairs = tuple(rational_circle(value) for value in parameters)
    (a, A), (d, D), (e, E), (g, C), (c, s) = pairs
    sa, sd, se, sg, sc = signs
    x2 = _symbolic_vector((sa * a, A), (0, 1), basis)
    x3 = _symbolic_vector((sd * d, D * se * e, D * E), (0, 1, 2), basis)
    q4 = _symbolic_vector((sc * c, s), (3, 4), basis)
    x4 = [sg * g * basis[0][i] + C * q4[i] for i in range(8)]
    info = fixed + symbolic_query_projector(1, x2, generators)
    info += symbolic_query_projector(2, x3, generators)
    info += symbolic_query_projector(2, x4, generators)
    delta = A**2 * D**2 * E**2 * C**2
    return sp.factor(1024 * info.det(method="domain-ge") / delta**3)


def _sectors(parameters):
    values = sp.Matrix([_determinant(parameters, signs) for signs in SECTOR_ROWS])
    F, P, Q, sector_R = [sp.factor(value) for value in HADAMARD_INVERSE * values]
    (a, A), (d, D), (e, E), (g, C), (c, s) = tuple(
        rational_circle(value) for value in parameters
    )
    amplitudes = (
        A**2 * D**2 * e * E * g * C * c * s**6,
        a * A**3 * d * D * E * g * C * c * s**6,
        a * A * d * D * e * s**6,
    )
    return (
        F,
        sp.factor(P / amplitudes[0]),
        sp.factor(Q / amplitudes[1]),
        sp.factor(sector_R / amplitudes[2]),
    )


def _grid_worker(job):
    indices, parameters = job
    return indices, tuple(str(value) for value in _sectors(parameters))


def _tensor_interpolate(values, nodes):
    current = values
    for axis in range(4, -1, -1):
        next_values = {}
        prefixes = itertools.product(*(range(len(nodes[i])) for i in range(axis)))
        for prefix in prefixes:
            next_values[prefix] = sp.interpolate(
                [
                    (nodes[axis][i], current[prefix + (i,)])
                    for i in range(len(nodes[axis]))
                ],
                VARIABLES[axis],
            )
        current = next_values
    return sp.Poly(current[()], *VARIABLES)


def _records(poly):
    return [
        {"powers": list(powers), "coefficient": str(coefficient)}
        for powers, coefficient in poly.terms()
    ]


def polynomial_from_records(records):
    expression = sp.Integer(0)
    for record in records:
        expression += sp.Rational(record["coefficient"]) * sp.prod(
            variable ** int(power)
            for variable, power in zip(VARIABLES, record["powers"], strict=True)
        )
    return sp.Poly(expression, *VARIABLES)


def tetrahedral_principal_polynomials(polynomials):
    F, H1, H2, H3 = (polynomials[name].as_expr() for name in ("F", "H1", "H2", "H3"))
    target = (1 - Z) ** 3 * (9 - Z) ** 2
    x = sp.expand(target - F)
    p2 = (
        (1 - U) ** 2
        * (1 - V) ** 2
        * R
        * (1 - R)
        * W
        * (1 - W)
        * Z
        * (1 - Z) ** 6
        * H1**2
    )
    q2 = (
        U
        * (1 - U) ** 3
        * V
        * (1 - V)
        * (1 - R)
        * W
        * (1 - W)
        * Z
        * (1 - Z) ** 6
        * H2**2
    )
    r2 = U * (1 - U) * V * (1 - V) * R * (1 - Z) ** 6 * H3**2
    pqr = (
        U
        * (1 - U) ** 3
        * V
        * (1 - V) ** 2
        * R
        * (1 - R)
        * W
        * (1 - W)
        * Z
        * (1 - Z) ** 9
        * H1
        * H2
        * H3
    )
    cubic = sp.expand(x**3 - x * (p2 + q2 + r2) - 2 * pqr)
    determinant = sp.expand(
        x**4
        - 2 * x**2 * (p2 + q2 + r2)
        - 8 * x * pqr
        + p2**2
        + q2**2
        + r2**2
        - 2 * (p2 * q2 + p2 * r2 + q2 * r2)
    )
    return {
        "x": sp.Poly(x, *VARIABLES),
        "x2_minus_p2": sp.Poly(x * x - p2, *VARIABLES),
        "x2_minus_q2": sp.Poly(x * x - q2, *VARIABLES),
        "x2_minus_r2": sp.Poly(x * x - r2, *VARIABLES),
        "cubic": sp.Poly(cubic, *VARIABLES),
        "determinant": sp.Poly(determinant, *VARIABLES),
    }


def bernstein_summary(polynomial):
    degrees = tuple(int(polynomial.degree(variable)) for variable in VARIABLES)
    coefficients = np.empty(tuple(degree + 1 for degree in degrees), dtype=object)
    coefficients.fill(sp.Integer(0))
    for powers, coefficient in polynomial.terms():
        coefficients[powers] = coefficient
    for axis, degree in enumerate(degrees):
        moved = np.moveaxis(coefficients, axis, 0)
        shape = moved.shape
        flat = moved.reshape((degree + 1, -1))
        transformed = np.empty_like(flat)
        for row in range(degree + 1):
            for column in range(flat.shape[1]):
                transformed[row, column] = sum(
                    sp.Rational(sp.binomial(row, source), sp.binomial(degree, source))
                    * flat[source, column]
                    for source in range(row + 1)
                )
        coefficients = np.moveaxis(transformed.reshape(shape), 0, axis)
    flat = list(coefficients.flat)
    negatives = [value for value in flat if value < 0]
    positives = [value for value in flat if value > 0]
    digest = hashlib.sha256(
        "\n".join(str(sp.factor(v)) for v in flat).encode()
    ).hexdigest()
    return {
        "degrees": list(degrees),
        "coefficient_count": len(flat),
        "negative_count": len(negatives),
        "zero_count": len(flat) - len(negatives) - len(positives),
        "minimum_coefficient": str(min(flat)),
        "minimum_positive": str(min(positives)) if positives else None,
        "coefficients_sha256": digest,
    }


def _hash(records):
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def reconstruct(node_set, workers):
    spatial, cayley = NODE_SETS[node_set]
    parameter_nodes = (spatial, spatial, spatial, spatial, cayley)
    square_nodes = tuple(
        tuple(sp.factor(rational_circle(value)[0] ** 2) for value in axis)
        for axis in parameter_nodes
    )
    jobs = []
    for indices in itertools.product(*(range(len(axis)) for axis in parameter_nodes)):
        jobs.append(
            (
                indices,
                tuple(parameter_nodes[i][index] for i, index in enumerate(indices)),
            )
        )
    raw = [{} for _ in range(4)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for indices, values in pool.map(_grid_worker, jobs, chunksize=1):
            for sector, value in enumerate(values):
                raw[sector][indices] = sp.Rational(value)
    polynomials = {
        name: _tensor_interpolate(raw[index], square_nodes)
        for index, name in enumerate(("F", "H1", "H2", "H3"))
    }
    return polynomials


def _symmetry_certificate():
    from spin8_dirac_edge import exact_walsh_symmetry_certificate

    base = exact_walsh_symmetry_certificate()
    induced = set()
    for action in base["triality_representation_actions"]:
        signs = action["vector_signs"]
        t1, t2, t3, t4 = signs[1], signs[2], signs[3], signs[4]
        induced.add((t1, t2, t1 * t2, t4, t3 * t4))
    annihilator = {
        mask
        for mask in itertools.product((0, 1), repeat=5)
        if all(_character(signs, mask) == 1 for signs in induced)
    }
    expected = set(SIGN_CHARACTERS.values())
    return {
        "induced_sign_group": [list(x) for x in sorted(induced)],
        "walsh_annihilator": [list(x) for x in sorted(annihilator)],
        "common_triality_conjugacy_verified": base["common_adjoint_conjugacy_verified"],
        "passed": annihilator == expected and len(induced) == 8,
    }


def _polynomial_report(polynomials):
    result = {}
    for name, poly in polynomials.items():
        records = _records(poly)
        result[name] = {
            "degrees": [int(poly.degree(v)) for v in VARIABLES],
            "term_count": len(records),
            "coefficients": records,
            "coefficients_sha256": _hash(records),
        }
    return result


def run(workers):
    symmetry = _symmetry_certificate()
    discovery = reconstruct("discovery", workers)
    confirmation = reconstruct("confirmation", workers)
    discovery_report = _polynomial_report(discovery)
    confirmation_report = _polynomial_report(confirmation)
    maps_match = all(
        discovery_report[name]["coefficients"]
        == confirmation_report[name]["coefficients"]
        for name in EXPECTED_DEGREES
    )
    degrees_match = all(
        tuple(confirmation_report[name]["degrees"]) == degree
        for name, degree in EXPECTED_DEGREES.items()
    )
    return {
        "experiment": "exact variable-Cayley one-edge reconstruction",
        "exact_symmetry": symmetry,
        "discovery": discovery_report,
        "confirmation": confirmation_report,
        "coefficient_maps_match": maps_match,
        "degrees_match": degrees_match,
        "holdouts_pending": True,
        "positivity_pending": True,
        "theorem_proved": False,
        "passed_reconstruction_gate": symmetry["passed"]
        and maps_match
        and degrees_match,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    report = run(args.workers)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "coefficient_maps_match",
                    "degrees_match",
                    "passed_reconstruction_gate",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
