"""Exact arithmetic contracts for real rational and quadratic coefficient fields.

The representation category used by the Schur and isotypic certificates is
still the category of finite-dimensional *real* representations.  This module
only records a smaller exact field containing the displayed matrix entries and
all certificate witnesses.  Linear equations are solved over that field; the
resulting basis is also a basis of the corresponding real solution space.

The first promoted extension is ``Q(sqrt(d))`` for positive rational ``d``.
It supplies exact membership, polynomial factorization, and order tests.  A
generic algebraic expression is rejected rather than silently handled through
floating-point signs or SymPy's expression domain.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from math import gcd

import sympy as sp
from sympy.polys.domains import AlgebraicField, RationalField
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed


@dataclass(frozen=True)
class ExactRealScalarField:
    """A declared exact ordered field used for certificate arithmetic."""

    primitive_element: sp.Expr | None = None

    def __post_init__(self) -> None:
        if self.primitive_element is None:
            return
        primitive = sp.sympify(self.primitive_element)
        square = sp.simplify(primitive**2)
        if not square.is_Rational or square <= 0:
            raise ValueError(
                "only real quadratic extensions Q(sqrt(d)) with rational d > 0 "
                "are currently certified"
            )
        if sp.sqrt(square).is_Rational:
            raise ValueError("the declared quadratic extension is actually rational")
        if sp.simplify(primitive - sp.sqrt(square)) != 0:
            raise ValueError("the primitive element must be the positive root sqrt(d)")
        object.__setattr__(self, "primitive_element", primitive)

    @property
    def domain(self) -> RationalField | AlgebraicField:
        if self.primitive_element is None:
            return sp.QQ
        return sp.QQ.algebraic_field(self.primitive_element)

    @property
    def degree(self) -> int:
        return 1 if self.primitive_element is None else 2

    @property
    def name(self) -> str:
        if self.primitive_element is None:
            return "Q"
        return f"Q({sp.sstr(self.primitive_element)})"

    @property
    def defining_polynomial(self) -> sp.Expr:
        variable = sp.Symbol("x")
        if self.primitive_element is None:
            return variable
        return sp.minpoly(self.primitive_element, variable)

    def contains(self, value: sp.Expr) -> bool:
        try:
            self.domain.from_sympy(sp.sympify(value))
        except (CoercionFailed, ValueError, TypeError):
            return False
        return True

    def contains_all(self, values: Iterable[sp.Expr]) -> bool:
        return all(self.contains(value) for value in values)

    def normalize(self, value: sp.Expr) -> sp.Expr:
        element = self.domain.from_sympy(sp.sympify(value))
        return sp.factor(self.domain.to_sympy(element))

    def polynomial(self, expression: sp.Expr, variable: sp.Symbol) -> sp.Poly:
        if self.primitive_element is None:
            return sp.Poly(expression, variable, domain=sp.QQ)
        return sp.Poly(expression, variable, extension=self.primitive_element)

    def factor_list(self, polynomial: sp.Poly) -> list[tuple[sp.Expr, int]]:
        variable = polynomial.gens[0]
        if self.primitive_element is None:
            return sp.factor_list(polynomial.as_expr(), variable)[1]
        return sp.factor_list(
            polynomial.as_expr(), variable, extension=self.primitive_element
        )[1]

    def sign(self, value: sp.Expr) -> int:
        """Return the exact sign in the declared real embedding."""

        element = self.domain.from_sympy(sp.sympify(value))
        if not element:
            return 0
        if self.primitive_element is None:
            rational = sp.Rational(self.domain.to_sympy(element))
            return 1 if rational > 0 else -1

        coefficients = [sp.Rational(value) for value in element.to_list()]
        if len(coefficients) == 1:
            coefficients.insert(0, sp.Rational(0))
        if len(coefficients) != 2:
            raise ValueError("quadratic field element did not have degree below two")
        radical_coefficient, rational_coefficient = coefficients
        if radical_coefficient == 0:
            return 1 if rational_coefficient > 0 else -1
        if rational_coefficient == 0:
            return 1 if radical_coefficient > 0 else -1
        if (rational_coefficient > 0) == (radical_coefficient > 0):
            return 1 if rational_coefficient > 0 else -1

        radicand = sp.Rational(self.primitive_element**2)
        comparison = rational_coefficient**2 - radical_coefficient**2 * radicand
        if comparison == 0:
            return 0
        comparison_sign = 1 if comparison > 0 else -1
        return comparison_sign if rational_coefficient > 0 else -comparison_sign

    def projective_matrix_normal_form(self, matrix: sp.Matrix) -> sp.Matrix:
        """Choose a deterministic exact representative of a matrix line."""

        candidate = sp.Matrix(matrix)
        if not self.contains_all(candidate):
            raise ValueError(f"matrix entries are not contained in {self.name}")
        if self.primitive_element is None:
            entries = [sp.Rational(entry) for entry in candidate]
            denominator = reduce(sp.ilcm, (int(entry.q) for entry in entries), 1)
            integers = [int(entry * denominator) for entry in entries]
            divisor = reduce(gcd, (abs(value) for value in integers), 0) or 1
            integers = [value // divisor for value in integers]
            first = next((value for value in integers if value), 1)
            if first < 0:
                integers = [-value for value in integers]
            return sp.Matrix(candidate.rows, candidate.cols, integers)

        first = next((entry for entry in candidate if entry != 0), None)
        if first is None:
            return sp.zeros(*candidate.shape)
        return candidate.applyfunc(lambda entry: self.normalize(entry / first))

    def nullspace(self, matrix: sp.Matrix) -> list[sp.Matrix]:
        """Return exact column nullspace vectors over the declared field.

        The rational path deliberately preserves SymPy's established Matrix
        basis so existing proof artifacts remain byte-stable.  Algebraic
        extensions use the sparse polynomial-domain implementation, avoiding
        expression-domain elimination and its severe scaling cost.
        """

        candidate = sp.Matrix(matrix)
        if self.primitive_element is None:
            return candidate.nullspace()
        domain_matrix = DomainMatrix.from_Matrix(candidate, fmt="sparse").convert_to(
            self.domain
        )
        row_basis = domain_matrix.nullspace().to_Matrix()
        vectors = [sp.Matrix(row) for row in row_basis.tolist()]
        if any(
            (candidate * vector).applyfunc(sp.simplify) != sp.zeros(candidate.rows, 1)
            for vector in vectors
        ):
            raise ValueError("polynomial-domain nullspace reconstruction failed")
        return vectors

    def rank(self, matrix: sp.Matrix) -> int:
        """Return exact rank through the polynomial-domain backend."""

        candidate = sp.Matrix(matrix)
        domain_matrix = DomainMatrix.from_Matrix(candidate, fmt="sparse").convert_to(
            self.domain
        )
        return int(domain_matrix.rank())


RATIONAL_FIELD = ExactRealScalarField()
Q_SQRT_TWO = ExactRealScalarField(sp.sqrt(2))


def field_from_extension(extension: sp.Expr | None) -> ExactRealScalarField:
    return RATIONAL_FIELD if extension is None else ExactRealScalarField(extension)


__all__ = [
    "Q_SQRT_TWO",
    "RATIONAL_FIELD",
    "ExactRealScalarField",
    "field_from_extension",
]
