from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from gaussian_phase_lattice import (
    alferov_tangent_step,
    algebraic_phase,
    factor_gaussian_integer,
    free_phase_divisor,
    rational_circle_point,
    run_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "gaussian_phase_lattice_20260811.json"


def test_published_artifact_replays_exactly() -> None:
    stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert run_diagnostics() == stored
    assert stored["passed"] is True


def test_algebraic_phase_is_the_rational_circle_point() -> None:
    for denominator in (4, 7, 239, 268):
        point = rational_circle_point(denominator)
        assert point == algebraic_phase((Fraction(denominator), Fraction(1)))
        assert point[0] * point[0] + point[1] * point[1] == 1


def test_free_split_prime_divisor_cancels() -> None:
    terms = [(239, 1), (7, 2), (4, 2), (268, 2)]
    assert free_phase_divisor(terms) == {}
    for denominator, _ in terms:
        factorization = factor_gaussian_integer((denominator, 1))
        assert factorization["factors"]


def test_alferov_step_is_exact_but_only_has_the_claimed_bound() -> None:
    result = alferov_tangent_step(5, 12)
    assert result["nearest_cotangent_integer"] == 2
    assert result["reduced_remainder"] == "-2/29"
    assert result["tangent_subtraction_identity"] is True
    assert result["nearest_integer_numerator_bound"] is True
    with pytest.raises(ValueError, match="must be positive"):
        alferov_tangent_step(3, 1)
