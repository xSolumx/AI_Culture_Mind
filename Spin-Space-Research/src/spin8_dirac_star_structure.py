"""Exact structural compression of the signed star Dirac--Gram certificate."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import sympy as sp

from spin8_dirac_star import (
    U,
    V,
    VARIABLES,
    W,
    Z,
    bernstein_records,
    polynomial_from_records,
)


def _coefficient_hash(coefficients: list[str]) -> str:
    payload = json.dumps(coefficients, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bernstein_summary(polynomial: sp.Poly) -> dict[str, object]:
    degrees, coefficients = bernstein_records(polynomial)
    exact = [sp.Rational(value) for value in coefficients]
    indices = itertools.product(*(range(degree + 1) for degree in degrees))
    zero_multiindices = [
        list(index) for index, value in zip(indices, exact, strict=True) if value == 0
    ]
    positive = [value for value in exact if value > 0]
    return {
        "degrees": list(degrees),
        "coefficient_count": len(exact),
        "negative_count": sum(bool(value < 0) for value in exact),
        "zero_count": sum(bool(value == 0) for value in exact),
        "zero_multiindices": zero_multiindices,
        "minimum_positive_coefficient": str(min(positive)),
        "coefficients_sha256": _coefficient_hash(coefficients),
    }


def exact_star_structure_certificate(source_artifact: Path) -> dict[str, object]:
    """Verify forced factors and reduced strict-interior positivity exactly."""

    source = json.loads(source_artifact.read_text(encoding="utf-8"))
    discovery = source["discovery_node_set"]
    confirmation = source["confirmation_node_set"]
    discovery_even = polynomial_from_records(discovery["even_coefficients"])
    discovery_odd = polynomial_from_records(discovery["odd_coefficients"])
    even = polynomial_from_records(confirmation["even_coefficients"])
    odd = polynomial_from_records(confirmation["odd_coefficients"])
    maps_match = discovery_even == even and discovery_odd == odd

    target = (1 - Z) ** 3 * (9 - Z) ** 2
    margin = sp.factor(target - even.as_expr())
    discriminant = sp.factor(
        margin**2 - U * V * W * (1 - U) * (1 - V) * (1 - W) * Z * odd.as_expr() ** 2
    )

    margin_factor = (1 - Z) ** 3
    odd_factor = (1 - U) * (V - W) * (1 - Z) ** 3
    discriminant_factor = (1 - Z) ** 6
    reduced_margin = sp.cancel(margin / margin_factor)
    reduced_odd = sp.cancel(odd.as_expr() / odd_factor)
    reduced_discriminant = sp.cancel(discriminant / discriminant_factor)
    exact_divisions = {
        "margin_by_(1-z)^3": sp.denom(reduced_margin) == 1,
        "odd_by_(1-u)(v-w)(1-z)^3": sp.denom(reduced_odd) == 1,
        "discriminant_by_(1-z)^6": sp.denom(reduced_discriminant) == 1,
    }

    gram_factor = (1 - U) * (1 - V) * (1 - W)
    boundary_audit = {
        "orthonormal_face_has_zero_normalized_margin": (
            sp.expand(margin.subs({U: 0, V: 0, W: 0})) == 0
        ),
        "cayley_endpoint_has_zero_normalized_margin": (
            sp.expand(margin.subs({Z: 1})) == 0
        ),
        "gram_faces_have_nonzero_normalized_margin_generically": all(
            sp.expand(margin.subs({variable: 1})) != 0 for variable in (U, V, W)
        ),
        "gram_faces_annihilate_unnormalized_margin": all(
            sp.expand((gram_factor**3 * margin).subs({variable: 1})) == 0
            for variable in (U, V, W)
        ),
    }

    margin_summary = _bernstein_summary(sp.Poly(sp.expand(reduced_margin), *VARIABLES))
    discriminant_summary = _bernstein_summary(
        sp.Poly(sp.expand(reduced_discriminant), *VARIABLES)
    )
    strict_interior = (
        margin_summary["negative_count"] == 0
        and margin_summary["coefficient_count"] > margin_summary["zero_count"]
        and discriminant_summary["negative_count"] == 0
        and discriminant_summary["coefficient_count"]
        > discriminant_summary["zero_count"]
    )
    margin_expected = {
        "degrees": [3, 3, 3, 2],
        "coefficient_count": 192,
        "negative_count": 0,
        "zero_count": 3,
        "minimum_positive_coefficient": "32/3",
    }
    discriminant_expected = {
        "degrees": [6, 6, 6, 4],
        "coefficient_count": 1715,
        "negative_count": 0,
        "zero_count": 20,
        "minimum_positive_coefficient": "512/9",
    }
    margin_zero_support = [[0, 0, 0, cayley_index] for cayley_index in range(3)]
    discriminant_zero_support = [
        [left, middle, right, cayley_index]
        for left, middle, right in (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
        )
        for cayley_index in range(5)
    ]
    exact_equality_support = (
        margin_summary["zero_multiindices"] == margin_zero_support
        and discriminant_summary["zero_multiindices"] == discriminant_zero_support
    )
    passed = (
        maps_match
        and all(exact_divisions.values())
        and all(boundary_audit.values())
        and all(margin_summary[key] == value for key, value in margin_expected.items())
        and all(
            discriminant_summary[key] == value
            for key, value in discriminant_expected.items()
        )
        and exact_equality_support
        and strict_interior
    )
    return {
        "theorem": "reduced signed-star structure and strict interior inequality",
        "source_artifact": source_artifact.name,
        "discovery_confirmation_maps_equal": maps_match,
        "exact_factor_divisions": exact_divisions,
        "boundary_equality_audit": boundary_audit,
        "orientation_factor": "(1-u)(v-w)(1-z)^3",
        "reduced_orientation_polynomial": str(sp.factor(reduced_odd)),
        "reduced_margin_bernstein": margin_summary,
        "reduced_discriminant_bernstein": discriminant_summary,
        "exact_equality_support": exact_equality_support,
        "complete_normalized_equality_set": ("z=1 or (u,v,w)=(0,0,0)"),
        "strict_on_open_unit_box": strict_interior,
        "conclusion": (
            "The strengthened star-family inequality is strict for "
            "0<u,v,w,z<1; in particular, orientation sensitivity vanishes "
            "when u=1, v=w, or z=1."
        ),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/spin8_dirac_star_20260804.json"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = exact_star_structure_certificate(arguments.source)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("exact signed-star structural certificate failed")


if __name__ == "__main__":
    main()
