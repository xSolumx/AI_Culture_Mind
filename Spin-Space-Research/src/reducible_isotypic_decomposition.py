"""Exact rational isotypic decomposition with explicit refusal boundaries.

For a completely reducible real representation ``rho`` supplied by rational
generator matrices, the simultaneous commutant is semisimple.  Reducibility is
therefore witnessed by nontrivial idempotents in that commutant.  This module
searches for such witnesses exactly, recursively splits the representation,
classifies every irreducible leaf with :mod:`schur_type_detector`, and groups
equivalent leaves by exact intertwiner spaces.

The resulting certificate verifies the real double-centralizer dimension law

    End_G(V_lambda ** m) = Mat_m(D_lambda),
    dim_R End_G(V_lambda ** m) = m**2 dim_R(D_lambda),

for ``D_lambda`` equal to ``R``, ``C``, or ``H``.  The implementation never
infers a decomposition from dimensions alone: projectors, restricted actions,
intertwiners, centrality, reconstruction, and dimension accounting are all
checked over exact arithmetic.

This is deliberately a certified rational-splitting layer, not an oracle for
all semisimple representations.  Some rational representations split over the
reals only after an algebraic field extension.  If the bounded deterministic
idempotent search cannot expose irreducible rational summands, the result is
``unresolved`` rather than a guessed decomposition.  Additional exact
commutant witnesses may be supplied by callers.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import sympy as sp
from exact_real_scalar_field import field_from_extension
from schur_type_detector import (
    SchurTypeCertificate,
    canonical_examples,
    detect_schur_type,
    exact_commutant_basis,
)
from schur_type_detector import certificate_json as schur_certificate_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "reducible_isotypic_decomposition_20260811.json"

DIVISION_DIMENSIONS = {"real": 1, "complex": 2, "quaternion": 4}


def _flatten(matrix: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(list(matrix))


def _matrix_json(matrix: sp.Matrix) -> list[list[str]]:
    return [
        [str(sp.factor(matrix[row, column])) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


def _validate_generators(
    generators: Iterable[sp.Matrix],
    scalar_extension: sp.Expr | None = None,
) -> list[sp.Matrix]:
    matrices = [sp.Matrix(generator) for generator in generators]
    if not matrices:
        raise ValueError("at least one representation generator is required")
    dimension = matrices[0].rows
    if dimension == 0 or matrices[0].cols != dimension:
        raise ValueError("representation generators must be nonempty square matrices")
    if any(matrix.shape != (dimension, dimension) for matrix in matrices):
        raise ValueError("all representation generators must have the same shape")
    field = field_from_extension(scalar_extension)
    if not field.contains_all(entry for matrix in matrices for entry in matrix):
        if scalar_extension is None:
            raise ValueError("exact rational representation generators are required")
        raise ValueError(
            f"exact representation generators over the declared field {field.name} "
            "are required"
        )
    return matrices


def _select_independent(matrices: Iterable[sp.Matrix]) -> list[sp.Matrix]:
    selected: list[sp.Matrix] = []
    current_rank = 0
    for matrix in matrices:
        candidate = sp.Matrix(matrix)
        if not any(candidate):
            continue
        design = sp.Matrix.hstack(
            *(_flatten(value) for value in (*selected, candidate))
        )
        new_rank = design.rank()
        if new_rank > current_rank:
            selected.append(candidate)
            current_rank = new_rank
    return selected


def exact_intertwiner_basis(
    source_generators: Iterable[sp.Matrix],
    target_generators: Iterable[sp.Matrix],
    *,
    scalar_extension: sp.Expr | None = None,
) -> list[sp.Matrix]:
    """Return exact maps ``T`` satisfying ``T source(g) = target(g) T``."""

    source = _validate_generators(source_generators, scalar_extension)
    target = _validate_generators(target_generators, scalar_extension)
    if len(source) != len(target):
        raise ValueError("source and target must have the same generator count")
    source_dimension = source[0].rows
    target_dimension = target[0].rows
    constraints = sp.Matrix.vstack(
        *(
            sp.kronecker_product(sp.eye(target_dimension), left.T)
            - sp.kronecker_product(right, sp.eye(source_dimension))
            for left, right in zip(source, target, strict=True)
        )
    )
    field = field_from_extension(scalar_extension)
    return [
        sp.Matrix(target_dimension, source_dimension, list(vector))
        for vector in field.nullspace(constraints)
    ]


def exact_algebra_center_basis(
    algebra_basis: Iterable[sp.Matrix],
    *,
    scalar_extension: sp.Expr | None = None,
) -> list[sp.Matrix]:
    """Return the exact center of a supplied matrix-algebra basis."""

    matrices = _select_independent(sp.Matrix(value) for value in algebra_basis)
    if not matrices:
        raise ValueError("at least one algebra basis matrix is required")
    dimension = matrices[0].rows
    if any(value.shape != (dimension, dimension) for value in matrices):
        raise ValueError("algebra basis matrices must be square with one shape")
    field = field_from_extension(scalar_extension)
    if not field.contains_all(entry for value in matrices for entry in value):
        if scalar_extension is None:
            raise ValueError("exact rational algebra basis matrices are required")
        raise ValueError(
            f"exact algebra basis matrices over the declared field {field.name} "
            "are required"
        )
    constraints = sp.Matrix.vstack(
        *(
            sp.Matrix.hstack(
                *(_flatten(left * right - right * left) for left in matrices)
            )
            for right in matrices
        )
    )
    field = field_from_extension(scalar_extension)
    return _select_independent(
        sum(
            (coefficient * matrix for coefficient, matrix in zip(vector, matrices)),
            sp.zeros(dimension),
        )
        for vector in field.nullspace(constraints)
    )


def _left_inverse(basis: sp.Matrix) -> sp.Matrix:
    """Return a deterministic rational left inverse of a full-column matrix."""

    rank = basis.cols
    pivot_rows = basis.T.rref()[1]
    if len(pivot_rows) != rank:
        raise ValueError("basis does not have full column rank")
    square = basis.extract(pivot_rows, range(rank))
    selector = sp.zeros(rank, basis.rows)
    for row, source_row in enumerate(pivot_rows):
        selector[row, source_row] = 1
    inverse = square.inv() * selector
    if inverse * basis != sp.eye(rank):
        raise ValueError("failed to construct an exact left inverse")
    return inverse


def _projector_coordinates(projector: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    columns = projector.columnspace()
    if not columns:
        raise ValueError("zero projector has no image coordinates")
    basis = sp.Matrix.hstack(*columns)
    return basis, _left_inverse(basis)


def _matrix_minimal_polynomial(
    matrix: sp.Matrix, scalar_extension: sp.Expr | None = None
) -> sp.Poly:
    """Compute the exact minimal polynomial by the first power dependence."""

    if matrix.rows != matrix.cols or matrix.rows == 0:
        raise ValueError("minimal polynomial requires a nonempty square matrix")
    variable = sp.Symbol("z")
    identity = sp.eye(matrix.rows)
    design = _flatten(identity)
    power = identity
    for degree in range(1, matrix.rows + 1):
        power = power * matrix
        target = _flatten(power)
        if design.row_join(target).rank() == design.rank():
            solution, parameters = design.gauss_jordan_solve(target)
            if parameters.rows:
                raise ValueError("minimal-polynomial coefficients were not unique")
            expression = variable**degree - sum(
                solution[index] * variable**index for index in range(degree)
            )
            polynomial = field_from_extension(scalar_extension).polynomial(
                expression, variable
            )
            if _evaluate_polynomial(polynomial, matrix) != sp.zeros(matrix.rows):
                raise ValueError("minimal-polynomial reconstruction failed")
            return polynomial.monic()
        design = design.row_join(target)
    raise ValueError("minimal polynomial was not found within the matrix dimension")


def _evaluate_polynomial(polynomial: sp.Poly, matrix: sp.Matrix) -> sp.Matrix:
    identity = sp.eye(matrix.rows)
    result = sp.zeros(matrix.rows)
    for coefficient in polynomial.all_coeffs():
        result = result * matrix + coefficient * identity
    return result


def _spectral_idempotents(
    matrix: sp.Matrix, scalar_extension: sp.Expr | None = None
) -> tuple[sp.Poly, list[sp.Matrix]]:
    """Return rational primary projectors from coprime minimal factors."""

    field = field_from_extension(scalar_extension)
    minimal = _matrix_minimal_polynomial(matrix, scalar_extension)
    variable = minimal.gens[0]
    factor_rows = field.factor_list(minimal)
    primary = [
        field.polynomial(factor**power, variable) for factor, power in factor_rows
    ]
    if len(primary) < 2:
        return minimal, []

    idempotents = []
    for factor in primary:
        quotient = minimal.exquo(factor)
        inverse = sp.invert(quotient, factor)
        if not isinstance(inverse, sp.Poly):
            inverse = field.polynomial(inverse, variable)
        residue = (quotient * inverse).rem(minimal)
        projector = _evaluate_polynomial(residue, matrix).applyfunc(sp.factor)
        if projector**2 != projector:
            raise ValueError("CRT polynomial did not produce an idempotent")
        idempotents.append(projector)
    if sum(idempotents, sp.zeros(matrix.rows)) != sp.eye(matrix.rows):
        raise ValueError("primary idempotents do not sum to the identity")
    if any(
        left * right != sp.zeros(matrix.rows)
        for index, left in enumerate(idempotents)
        for right in idempotents[index + 1 :]
    ):
        raise ValueError("primary idempotents are not orthogonal")
    return minimal, idempotents


def _coefficient_vectors(dimension: int) -> Iterator[tuple[int, ...]]:
    """Yield deterministic low-complexity commutant combinations."""

    seen: set[tuple[int, ...]] = set()

    def emit(vector: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        if any(vector) and vector not in seen:
            seen.add(vector)
            yield vector

    for index in range(dimension):
        vector = tuple(int(position == index) for position in range(dimension))
        yield from emit(vector)
    for support_size in range(2, min(4, dimension) + 1):
        for support in itertools.combinations(range(dimension), support_size):
            for signs in itertools.product((-1, 1), repeat=support_size):
                if signs[0] < 0:
                    continue
                vector = [0] * dimension
                for index, sign in zip(support, signs, strict=True):
                    vector[index] = sign
                yield from emit(tuple(vector))
    for left in range(dimension):
        for right in range(dimension):
            if left == right:
                continue
            vector = [0] * dimension
            vector[left] = 1
            vector[right] = 2
            yield from emit(tuple(vector))
    yield from emit(tuple(range(1, dimension + 1)))
    yield from emit(tuple((index + 1) ** 2 for index in range(dimension)))


@dataclass(frozen=True)
class SplitWitness:
    parent_rank: int
    child_ranks: tuple[int, int]
    coefficients: tuple[int, ...]
    minimal_polynomial: sp.Poly
    source: str


@dataclass(frozen=True)
class IrreducibleSummandCertificate:
    projector: sp.Matrix
    basis: sp.Matrix
    restricted_generators: tuple[sp.Matrix, ...]
    schur: SchurTypeCertificate
    exact_gates: dict[str, bool]


@dataclass(frozen=True)
class IsotypicBlockCertificate:
    projector: sp.Matrix
    summand_indices: tuple[int, ...]
    real_dimension: int
    irreducible_dimension: int
    multiplicity: int
    schur_type: str
    division_dimension: int
    commutant_dimension: int
    expected_commutant_dimension: int
    center_dimension: int
    expected_center_dimension: int
    pairwise_intertwiner_dimensions: tuple[tuple[int, ...], ...]
    reference_intertwiners: tuple[sp.Matrix, ...]
    aligned_basis: sp.Matrix
    aligned_commutant_basis: tuple[sp.Matrix, ...]
    aligned_center_basis: tuple[sp.Matrix, ...]
    exact_gates: dict[str, bool]


@dataclass(frozen=True)
class ReducibleIsotypicCertificate:
    real_dimension: int
    generator_count: int
    commutant_dimension: int
    completely_reducible_assumed: bool
    summands: tuple[IrreducibleSummandCertificate, ...]
    blocks: tuple[IsotypicBlockCertificate, ...]
    split_witnesses: tuple[SplitWitness, ...]
    candidates_examined: int
    exact_gates: dict[str, bool]
    unresolved_projector_ranks: tuple[int, ...]
    rejection_reason: str | None
    scalar_extension: sp.Expr | None = None

    @property
    def certified(self) -> bool:
        return bool(
            self.completely_reducible_assumed
            and not self.unresolved_projector_ranks
            and self.summands
            and self.blocks
            and all(self.exact_gates.values())
            and all(all(summand.exact_gates.values()) for summand in self.summands)
            and all(all(block.exact_gates.values()) for block in self.blocks)
        )


def _find_split(
    projector: sp.Matrix,
    commutant: Sequence[sp.Matrix],
    supplied_witnesses: Sequence[sp.Matrix],
    *,
    max_candidates: int,
    search_source: str = "commutant_search",
    scalar_extension: sp.Expr | None = None,
) -> tuple[sp.Matrix | None, SplitWitness | None, int]:
    basis, left_inverse = _projector_coordinates(projector)
    rank = basis.cols
    if rank == 1 or max_candidates == 0:
        return None, None, 0
    corner = _select_independent(
        left_inverse * projector * matrix * projector * basis for matrix in commutant
    )
    supplied = [
        left_inverse * projector * matrix * projector * basis
        for matrix in supplied_witnesses
    ]
    candidates: list[tuple[sp.Matrix, tuple[int, ...], str]] = [
        (matrix, (), "supplied") for matrix in supplied
    ]
    coefficients = _coefficient_vectors(len(corner))
    examined = 0
    while examined < max_candidates:
        if candidates:
            candidate, weights, source = candidates.pop(0)
        else:
            try:
                weights = next(coefficients)
            except StopIteration:
                break
            candidate = sum(
                (
                    weight * matrix
                    for weight, matrix in zip(weights, corner, strict=True)
                ),
                sp.zeros(rank),
            )
            source = search_source
        examined += 1
        minimal, idempotents = _spectral_idempotents(candidate, scalar_extension)
        proper = [value for value in idempotents if 0 < value.rank() < rank]
        if not proper:
            continue
        child = min(proper, key=lambda value: (value.rank(), tuple(value)))
        lifted = (basis * child * left_inverse * projector).applyfunc(sp.factor)
        complement = projector - lifted
        witness = SplitWitness(
            parent_rank=rank,
            child_ranks=(lifted.rank(), complement.rank()),
            coefficients=weights,
            minimal_polynomial=minimal,
            source=source,
        )
        return lifted, witness, examined
    return None, None, examined


def _recursive_split(
    projectors: Sequence[sp.Matrix],
    search_algebra: Sequence[sp.Matrix],
    supplied_witnesses: Sequence[sp.Matrix],
    *,
    max_candidates_per_leaf: int,
    search_source: str,
    scalar_extension: sp.Expr | None = None,
) -> tuple[list[sp.Matrix], list[SplitWitness], int]:
    pending = list(projectors)
    leaves: list[sp.Matrix] = []
    splits: list[SplitWitness] = []
    candidates_examined = 0
    while pending:
        projector = pending.pop()
        child, split, examined = _find_split(
            projector,
            search_algebra,
            supplied_witnesses,
            max_candidates=max_candidates_per_leaf,
            search_source=search_source,
            scalar_extension=scalar_extension,
        )
        candidates_examined += examined
        if child is None or split is None:
            leaves.append(projector)
            continue
        complement = (projector - child).applyfunc(sp.factor)
        splits.append(split)
        pending.extend((complement, child))
    return leaves, splits, candidates_examined


def _restrict_generators(
    generators: Sequence[sp.Matrix], projector: sp.Matrix
) -> tuple[sp.Matrix, sp.Matrix, tuple[sp.Matrix, ...]]:
    basis, left_inverse = _projector_coordinates(projector)
    restricted = tuple(left_inverse * generator * basis for generator in generators)
    return basis, left_inverse, restricted


def _corner_basis(
    projector: sp.Matrix, commutant: Sequence[sp.Matrix]
) -> list[sp.Matrix]:
    basis, left_inverse = _projector_coordinates(projector)
    return _select_independent(
        left_inverse * projector * matrix * projector * basis for matrix in commutant
    )


def _union_find_components(intertwiner_dimensions: list[list[int]]) -> list[list[int]]:
    size = len(intertwiner_dimensions)
    parent = list(range(size))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(size):
        for right in range(left + 1, size):
            if intertwiner_dimensions[left][right]:
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(size):
        groups.setdefault(find(index), []).append(index)
    return sorted(groups.values(), key=lambda group: group[0])


def decompose_reducible_representation(
    generators: Iterable[sp.Matrix],
    *,
    assume_completely_reducible: bool,
    splitting_witnesses: Iterable[sp.Matrix] = (),
    max_candidates_per_leaf: int = 4096,
    scalar_extension: sp.Expr | None = None,
) -> ReducibleIsotypicCertificate:
    """Return a fully checked isotypic decomposition or an unresolved result."""

    field = field_from_extension(scalar_extension)
    matrices = _validate_generators(generators, scalar_extension)
    dimension = matrices[0].rows
    if max_candidates_per_leaf < 0:
        raise ValueError("max_candidates_per_leaf must be nonnegative")
    witnesses = [sp.Matrix(value) for value in splitting_witnesses]
    if any(value.shape != (dimension, dimension) for value in witnesses):
        raise ValueError("splitting witnesses must match the representation dimension")
    if not field.contains_all(entry for value in witnesses for entry in value):
        if scalar_extension is None:
            raise ValueError("splitting witnesses must be exact rational matrices")
        raise ValueError(
            f"splitting witnesses must be exact matrices over {field.name}"
        )
    if any(
        value * generator != generator * value
        for value in witnesses
        for generator in matrices
    ):
        raise ValueError("splitting witnesses must lie in the exact commutant")
    commutant = exact_commutant_basis(matrices, scalar_extension=scalar_extension)
    commutant_center = exact_algebra_center_basis(
        commutant, scalar_extension=scalar_extension
    )
    commutant_gates = {
        "commutant_basis_commutes_exactly": all(
            basis * generator == generator * basis
            for basis in commutant
            for generator in matrices
        ),
        "commutant_center_basis_is_central_exactly": all(
            center * basis == basis * center
            for center in commutant_center
            for basis in commutant
        ),
    }
    if not assume_completely_reducible:
        return ReducibleIsotypicCertificate(
            real_dimension=dimension,
            generator_count=len(matrices),
            commutant_dimension=len(commutant),
            completely_reducible_assumed=False,
            summands=(),
            blocks=(),
            split_witnesses=(),
            candidates_examined=0,
            exact_gates=commutant_gates,
            unresolved_projector_ranks=(dimension,),
            rejection_reason=(
                "complete reducibility was not supplied, so commutant "
                "idempotents cannot certify a semisimple decomposition"
            ),
            scalar_extension=scalar_extension,
        )

    central_leaves, central_splits, central_candidates = _recursive_split(
        [sp.eye(dimension)],
        commutant_center,
        (),
        max_candidates_per_leaf=max_candidates_per_leaf,
        search_source="central_search",
        scalar_extension=scalar_extension,
    )
    leaves, multiplicity_splits, multiplicity_candidates = _recursive_split(
        central_leaves,
        commutant,
        witnesses,
        max_candidates_per_leaf=max_candidates_per_leaf,
        search_source="multiplicity_search",
        scalar_extension=scalar_extension,
    )
    split_rows = [*central_splits, *multiplicity_splits]
    candidates_examined = central_candidates + multiplicity_candidates

    leaves.sort(
        key=lambda value: (
            value.rank(),
            (
                tuple(value)
                if scalar_extension is None
                else tuple(sp.sstr(entry) for entry in value)
            ),
        )
    )
    summands: list[IrreducibleSummandCertificate] = []
    unresolved: list[int] = []
    for projector in leaves:
        basis, _, restricted = _restrict_generators(matrices, projector)
        schur = detect_schur_type(
            restricted,
            assume_completely_reducible=True,
            scalar_extension=scalar_extension,
        )
        gates = {
            "projector_is_idempotent": projector**2 == projector,
            "projector_commutes_with_generators": all(
                projector * generator == generator * projector for generator in matrices
            ),
            "basis_spans_projector_image": basis.rank() == projector.rank(),
            "restricted_action_reconstructs": all(
                generator * basis == basis * restricted_generator
                for generator, restricted_generator in zip(
                    matrices, restricted, strict=True
                )
            ),
            "leaf_has_division_commutant": schur.classified_irreducible,
        }
        summands.append(
            IrreducibleSummandCertificate(
                projector=projector,
                basis=basis,
                restricted_generators=restricted,
                schur=schur,
                exact_gates=gates,
            )
        )
        if not schur.classified_irreducible:
            unresolved.append(projector.rank())

    if unresolved:
        gates = {
            **commutant_gates,
            "central_stage_projectors_are_central": all(
                projector * value == value * projector
                for projector in central_leaves
                for value in commutant
            ),
            "central_stage_projectors_sum_to_identity": sum(
                central_leaves, sp.zeros(dimension)
            )
            == sp.eye(dimension),
            "leaf_projectors_sum_to_identity": sum(
                (value.projector for value in summands), sp.zeros(dimension)
            )
            == sp.eye(dimension),
            "all_leaves_irreducible": False,
        }
        return ReducibleIsotypicCertificate(
            real_dimension=dimension,
            generator_count=len(matrices),
            commutant_dimension=len(commutant),
            completely_reducible_assumed=True,
            summands=tuple(summands),
            blocks=(),
            split_witnesses=tuple(split_rows),
            candidates_examined=candidates_examined,
            exact_gates=gates,
            unresolved_projector_ranks=tuple(sorted(unresolved)),
            rejection_reason=(
                "bounded rational idempotent search left a nondivision leaf; "
                "supply an exact splitting witness or extend the scalar field"
                if scalar_extension is None
                else (
                    f"bounded idempotent search over {field.name} left a "
                    "nondivision leaf; supply an exact splitting witness or "
                    "extend the scalar field"
                )
            ),
            scalar_extension=scalar_extension,
        )

    intertwiner_dimensions = [
        [
            len(
                exact_intertwiner_basis(
                    source.restricted_generators,
                    target.restricted_generators,
                    scalar_extension=scalar_extension,
                )
            )
            for target in summands
        ]
        for source in summands
    ]
    groups = _union_find_components(intertwiner_dimensions)
    blocks: list[IsotypicBlockCertificate] = []
    for group in groups:
        first = summands[group[0]]
        assert first.schur.schur_type is not None
        schur_type = first.schur.schur_type
        division_dimension = DIVISION_DIMENSIONS[schur_type]
        projector = sum(
            (summands[index].projector for index in group),
            sp.zeros(dimension),
        ).applyfunc(sp.factor)
        multiplicity = len(group)
        corner_basis = _corner_basis(projector, commutant)
        corner_dimension = len(corner_basis)
        expected_corner = multiplicity**2 * division_dimension
        center_dimension = len(
            exact_algebra_center_basis(corner_basis, scalar_extension=scalar_extension)
        )
        expected_center = 2 if schur_type == "complex" else 1
        pairwise = tuple(
            tuple(intertwiner_dimensions[left][right] for right in group)
            for left in group
        )
        reference = first.restricted_generators
        reference_intertwiners = []
        aligned_embeddings = []
        for index in group:
            target = summands[index]
            if index == group[0]:
                intertwiner = sp.eye(first.projector.rank())
            else:
                intertwiner_space = exact_intertwiner_basis(
                    reference,
                    target.restricted_generators,
                    scalar_extension=scalar_extension,
                )
                invertible = [
                    value
                    for value in intertwiner_space
                    if value.rows == value.cols and value.det() != 0
                ]
                intertwiner = min(
                    invertible,
                    key=(
                        (lambda value: tuple(value))
                        if scalar_extension is None
                        else (lambda value: tuple(sp.sstr(entry) for entry in value))
                    ),
                )
            reference_intertwiners.append(intertwiner)
            aligned_embeddings.append(target.basis * intertwiner)
        aligned_basis = sp.Matrix.hstack(*aligned_embeddings)
        aligned_left_inverse = _left_inverse(aligned_basis)
        aligned_corner_basis = _select_independent(
            aligned_left_inverse * projector * value * projector * aligned_basis
            for value in commutant
        )
        aligned_center_basis = exact_algebra_center_basis(
            aligned_corner_basis, scalar_extension=scalar_extension
        )
        repeated_reference_actions = [
            sp.diag(*(generator for _ in range(multiplicity)))
            for generator in reference
        ]
        gates = {
            "summands_have_common_dimension": len(
                {summands[index].projector.rank() for index in group}
            )
            == 1,
            "summands_have_common_schur_type": len(
                {summands[index].schur.schur_type for index in group}
            )
            == 1,
            "equivalent_pair_intertwiner_dimension_matches_division_algebra": all(
                intertwiner_dimensions[left][right] == division_dimension
                for left in group
                for right in group
            ),
            "projector_is_central_in_full_commutant": all(
                projector * value == value * projector for value in commutant
            ),
            "corner_commutant_dimension_matches_double_centralizer": (
                corner_dimension == expected_corner
            ),
            "corner_center_dimension_matches_schur_type": (
                center_dimension == expected_center
            ),
            "reference_intertwiners_are_invertible": all(
                value.det() != 0 for value in reference_intertwiners
            ),
            "reference_intertwiners_align_actions": all(
                intertwiner * reference_generator == target_generator * intertwiner
                for index, intertwiner in zip(
                    group, reference_intertwiners, strict=True
                )
                for reference_generator, target_generator in zip(
                    reference,
                    summands[index].restricted_generators,
                    strict=True,
                )
            ),
            "aligned_basis_spans_isotypic_image": (
                aligned_basis.rank() == projector.rank()
            ),
            "aligned_commutant_basis_has_complete_dimension": (
                len(aligned_corner_basis) == expected_corner
            ),
            "aligned_commutant_basis_commutes_with_repeated_action": all(
                value * generator == generator * value
                for value in aligned_corner_basis
                for generator in repeated_reference_actions
            ),
            "aligned_center_basis_has_expected_dimension": (
                len(aligned_center_basis) == expected_center
            ),
            "aligned_center_basis_is_central": all(
                center * value == value * center
                for center in aligned_center_basis
                for value in aligned_corner_basis
            ),
        }
        blocks.append(
            IsotypicBlockCertificate(
                projector=projector,
                summand_indices=tuple(group),
                real_dimension=projector.rank(),
                irreducible_dimension=first.projector.rank(),
                multiplicity=multiplicity,
                schur_type=schur_type,
                division_dimension=division_dimension,
                commutant_dimension=corner_dimension,
                expected_commutant_dimension=expected_corner,
                center_dimension=center_dimension,
                expected_center_dimension=expected_center,
                pairwise_intertwiner_dimensions=pairwise,
                reference_intertwiners=tuple(reference_intertwiners),
                aligned_basis=aligned_basis,
                aligned_commutant_basis=tuple(aligned_corner_basis),
                aligned_center_basis=tuple(aligned_center_basis),
                exact_gates=gates,
            )
        )

    leaf_projectors = [summand.projector for summand in summands]
    basis_change = sp.Matrix.hstack(*(summand.basis for summand in summands))
    inverse_change = basis_change.inv() if basis_change.det() != 0 else None
    transformed = (
        [inverse_change * generator * basis_change for generator in matrices]
        if inverse_change is not None
        else []
    )
    offsets = [0]
    for summand in summands:
        offsets.append(offsets[-1] + summand.projector.rank())
    block_diagonal = bool(
        transformed
        and all(
            matrix[
                offsets[left] : offsets[left + 1], offsets[right] : offsets[right + 1]
            ]
            == sp.zeros(
                offsets[left + 1] - offsets[left],
                offsets[right + 1] - offsets[right],
            )
            for matrix in transformed
            for left in range(len(summands))
            for right in range(len(summands))
            if left != right
        )
    )
    aligned_global_basis = sp.Matrix.hstack(*(block.aligned_basis for block in blocks))
    aligned_inverse = (
        aligned_global_basis.inv() if aligned_global_basis.det() != 0 else None
    )
    aligned_actions_match = bool(
        aligned_inverse is not None
        and all(
            aligned_inverse * generator * aligned_global_basis
            == sp.diag(
                *(
                    sp.diag(
                        *(
                            summands[block.summand_indices[0]].restricted_generators[
                                generator_index
                            ]
                            for _ in range(block.multiplicity)
                        )
                    )
                    for block in blocks
                )
            )
            for generator_index, generator in enumerate(matrices)
        )
    )
    gates = {
        **commutant_gates,
        "central_stage_projectors_are_central": all(
            projector * value == value * projector
            for projector in central_leaves
            for value in commutant
        ),
        "central_stage_projectors_sum_to_identity": sum(
            central_leaves, sp.zeros(dimension)
        )
        == sp.eye(dimension),
        "leaf_projectors_are_pairwise_orthogonal": all(
            left * right == sp.zeros(dimension) and right * left == sp.zeros(dimension)
            for index, left in enumerate(leaf_projectors)
            for right in leaf_projectors[index + 1 :]
        ),
        "leaf_projectors_sum_to_identity": sum(leaf_projectors, sp.zeros(dimension))
        == sp.eye(dimension),
        "all_leaves_irreducible": all(
            summand.schur.classified_irreducible for summand in summands
        ),
        "basis_change_is_invertible": inverse_change is not None,
        "basis_change_block_diagonalizes_generators": block_diagonal,
        "aligned_isotypic_basis_is_invertible": aligned_inverse is not None,
        "aligned_coordinates_repeat_reference_actions": aligned_actions_match,
        "inequivalent_blocks_have_zero_intertwiner_space": all(
            intertwiner_dimensions[left][right] == 0
            for left in range(len(summands))
            for right in range(len(summands))
            if not any(left in group and right in group for group in groups)
        ),
        "isotypic_projectors_sum_to_identity": sum(
            (block.projector for block in blocks), sp.zeros(dimension)
        )
        == sp.eye(dimension),
        "global_commutant_dimension_matches_block_sum": len(commutant)
        == sum(block.expected_commutant_dimension for block in blocks),
        "global_commutant_center_dimension_matches_block_sum": len(
            exact_algebra_center_basis(commutant, scalar_extension=scalar_extension)
        )
        == sum(block.expected_center_dimension for block in blocks),
    }
    passed = bool(
        all(gates.values())
        and all(all(summand.exact_gates.values()) for summand in summands)
        and all(all(block.exact_gates.values()) for block in blocks)
    )
    return ReducibleIsotypicCertificate(
        real_dimension=dimension,
        generator_count=len(matrices),
        commutant_dimension=len(commutant),
        completely_reducible_assumed=True,
        summands=tuple(summands),
        blocks=tuple(blocks),
        split_witnesses=tuple(split_rows),
        candidates_examined=candidates_examined,
        exact_gates=gates,
        unresolved_projector_ranks=(),
        rejection_reason=(
            None if passed else "one or more exact decomposition gates failed"
        ),
        scalar_extension=scalar_extension,
    )


def aligned_isotypic_basis(
    certificate: ReducibleIsotypicCertificate,
) -> sp.Matrix:
    """Return the compiler-ready basis grouped by aligned isotypic copies."""

    if not certificate.certified:
        raise ValueError("a certified isotypic decomposition is required")
    basis = sp.Matrix.hstack(*(block.aligned_basis for block in certificate.blocks))
    if basis.shape != (certificate.real_dimension, certificate.real_dimension):
        raise ValueError("aligned isotypic basis has the wrong shape")
    if basis.det() == 0:
        raise ValueError("aligned isotypic basis is singular")
    return basis


def transform_to_isotypic_coordinates(
    generators: Iterable[sp.Matrix],
    certificate: ReducibleIsotypicCertificate,
) -> tuple[sp.Matrix, ...]:
    """Conjugate generators into the certificate's aligned block basis."""

    matrices = _validate_generators(generators, certificate.scalar_extension)
    if matrices[0].rows != certificate.real_dimension:
        raise ValueError("generator dimension does not match the certificate")
    if len(matrices) != certificate.generator_count:
        raise ValueError("generator count does not match the certificate")
    basis = aligned_isotypic_basis(certificate)
    inverse = basis.inv()
    return tuple(inverse * generator * basis for generator in matrices)


def _direct_product_sum(
    components: Sequence[Sequence[sp.Matrix]],
) -> list[sp.Matrix]:
    """Represent a direct-product algebra on a direct sum of its modules."""

    dimensions = [component[0].rows for component in components]
    output = []
    for active, component in enumerate(components):
        for generator in component:
            output.append(
                sp.diag(
                    *(
                        generator if index == active else sp.zeros(dimension)
                        for index, dimension in enumerate(dimensions)
                    )
                )
            )
    return output


def repeated_representation(
    generators: Iterable[sp.Matrix], multiplicity: int
) -> list[sp.Matrix]:
    if multiplicity < 1:
        raise ValueError("multiplicity must be positive")
    matrices = _validate_generators(generators)
    return [
        sp.diag(*(generator for _ in range(multiplicity))) for generator in matrices
    ]


def spin_two_generators() -> list[sp.Matrix]:
    """Return the rational spin-two action on ``Sym_0(3)``."""

    so3 = canonical_examples()["real_so3_vector"]
    basis = [
        sp.diag(1, -1, 0),
        sp.diag(0, 1, -1),
        sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
        sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    ]
    design = sp.Matrix.hstack(*(_flatten(value) for value in basis))
    output = []
    for generator in so3:
        columns = []
        for matrix in basis:
            image = generator * matrix - matrix * generator
            solution, parameters = design.gauss_jordan_solve(_flatten(image))
            if parameters.rows:
                raise ValueError("spin-two coordinates were not unique")
            columns.append(solution)
        output.append(sp.Matrix.hstack(*columns))
    return output


def cl3_spin3_fixture() -> list[sp.Matrix]:
    """Return ``2 V0 + 2 V1``, the Cl(3,0) conjugation isotypic type."""

    return [
        sp.diag(sp.zeros(2), generator, generator)
        for generator in canonical_examples()["real_so3_vector"]
    ]


def spin9_slice_fixture() -> list[sp.Matrix]:
    """Return the rational model ``V1 + 2 V5`` of the Spin(9) quotient slice."""

    return [
        sp.diag(sp.zeros(1), generator, generator)
        for generator in spin_two_generators()
    ]


def rationally_conjugate_representation(
    generators: Sequence[sp.Matrix],
) -> list[sp.Matrix]:
    dimension = generators[0].rows
    change = sp.eye(dimension)
    for row in range(dimension - 1):
        change[row, row + 1] = 1
    if dimension > 3:
        change[0, -1] = -1
    inverse = change.inv()
    return [change * generator * inverse for generator in generators]


def canonical_reducible_examples() -> dict[str, list[sp.Matrix]]:
    examples = canonical_examples()
    return {
        "doubled_real_so3": repeated_representation(examples["real_so3_vector"], 2),
        "doubled_complex_u1": repeated_representation(
            examples["complex_u1_realification"], 2
        ),
        "doubled_quaternion_su2": repeated_representation(
            examples["quaternion_su2_spinor_realification"], 2
        ),
        "mixed_real_complex_quaternion": _direct_product_sum(
            [
                examples["real_so3_vector"],
                examples["complex_u1_realification"],
                examples["quaternion_su2_spinor_realification"],
            ]
        ),
        "cl3_two_trivial_plus_two_vector": cl3_spin3_fixture(),
        "spin9_v1_plus_two_v5": spin9_slice_fixture(),
        "cl3_rational_conjugacy": rationally_conjugate_representation(
            cl3_spin3_fixture()
        ),
        "spin9_slice_rational_conjugacy": rationally_conjugate_representation(
            spin9_slice_fixture()
        ),
    }


def _split_witness_json(witness: SplitWitness) -> dict[str, object]:
    return {
        "parent_rank": witness.parent_rank,
        "child_ranks": list(witness.child_ranks),
        "coefficients": list(witness.coefficients),
        "minimal_polynomial": str(witness.minimal_polynomial.as_expr()),
        "source": witness.source,
    }


def certificate_json(certificate: ReducibleIsotypicCertificate) -> dict[str, object]:
    payload: dict[str, object] = {
        "real_dimension": certificate.real_dimension,
        "generator_count": certificate.generator_count,
        "commutant_dimension": certificate.commutant_dimension,
        "completely_reducible_assumed": certificate.completely_reducible_assumed,
        "certified": certificate.certified,
        "candidates_examined": certificate.candidates_examined,
        "unresolved_projector_ranks": list(certificate.unresolved_projector_ranks),
        "split_witnesses": [
            _split_witness_json(witness) for witness in certificate.split_witnesses
        ],
        "summands": [
            {
                "real_dimension": summand.projector.rank(),
                "projector": _matrix_json(summand.projector),
                "basis": _matrix_json(summand.basis),
                "schur": schur_certificate_json(summand.schur),
                "exact_gates": summand.exact_gates,
            }
            for summand in certificate.summands
        ],
        "isotypic_blocks": [
            {
                "summand_indices": list(block.summand_indices),
                "real_dimension": block.real_dimension,
                "irreducible_dimension": block.irreducible_dimension,
                "multiplicity": block.multiplicity,
                "schur_type": block.schur_type,
                "division_dimension": block.division_dimension,
                "commutant_dimension": block.commutant_dimension,
                "expected_commutant_dimension": block.expected_commutant_dimension,
                "center_dimension": block.center_dimension,
                "expected_center_dimension": block.expected_center_dimension,
                "pairwise_intertwiner_dimensions": [
                    list(row) for row in block.pairwise_intertwiner_dimensions
                ],
                "reference_intertwiners": [
                    _matrix_json(value) for value in block.reference_intertwiners
                ],
                "aligned_basis": _matrix_json(block.aligned_basis),
                "aligned_commutant_basis": [
                    _matrix_json(value) for value in block.aligned_commutant_basis
                ],
                "aligned_center_basis": [
                    _matrix_json(value) for value in block.aligned_center_basis
                ],
                "projector": _matrix_json(block.projector),
                "exact_gates": block.exact_gates,
            }
            for block in certificate.blocks
        ],
        "exact_gates": certificate.exact_gates,
        "rejection_reason": certificate.rejection_reason,
    }
    if certificate.scalar_extension is not None:
        field = field_from_extension(certificate.scalar_extension)
        payload["scalar_field"] = {
            "name": field.name,
            "degree": field.degree,
            "primitive_element": sp.sstr(field.primitive_element),
            "defining_polynomial": sp.sstr(field.defining_polynomial),
        }
    return payload


def diagnostics() -> dict[str, object]:
    examples = canonical_reducible_examples()
    expected = {
        "doubled_real_so3": [("real", 2, 3, 4)],
        "doubled_complex_u1": [("complex", 2, 2, 8)],
        "doubled_quaternion_su2": [("quaternion", 2, 4, 16)],
        "mixed_real_complex_quaternion": [
            ("real", 1, 3, 1),
            ("complex", 1, 2, 2),
            ("quaternion", 1, 4, 4),
        ],
        "cl3_two_trivial_plus_two_vector": [
            ("real", 2, 1, 4),
            ("real", 2, 3, 4),
        ],
        "spin9_v1_plus_two_v5": [
            ("real", 1, 1, 1),
            ("real", 2, 5, 4),
        ],
        "cl3_rational_conjugacy": [
            ("real", 2, 1, 4),
            ("real", 2, 3, 4),
        ],
        "spin9_slice_rational_conjugacy": [
            ("real", 1, 1, 1),
            ("real", 2, 5, 4),
        ],
    }
    certificates = {
        name: decompose_reducible_representation(
            generators,
            assume_completely_reducible=True,
        )
        for name, generators in examples.items()
    }
    observed = {
        name: sorted(
            (
                block.schur_type,
                block.multiplicity,
                block.irreducible_dimension,
                block.commutant_dimension,
            )
            for block in certificate.blocks
        )
        for name, certificate in certificates.items()
    }
    expected_sorted = {name: sorted(rows) for name, rows in expected.items()}
    missing_assumption = decompose_reducible_representation(
        examples["doubled_real_so3"],
        assume_completely_reducible=False,
    )
    bounded_refusal = decompose_reducible_representation(
        examples["doubled_real_so3"],
        assume_completely_reducible=True,
        max_candidates_per_leaf=0,
    )
    spin_two_casimir = -sum(
        (generator**2 for generator in spin_two_generators()), sp.zeros(5)
    )
    passed = bool(
        all(certificate.certified for certificate in certificates.values())
        and observed == expected_sorted
        and not missing_assumption.certified
        and not bounded_refusal.certified
        and bounded_refusal.unresolved_projector_ranks == (6,)
        and spin_two_casimir == 6 * sp.eye(5)
    )
    return {
        "schema_version": 1,
        "claim_scope": (
            "exact rational isotypic decomposition when bounded commutant-"
            "idempotent search exposes division-commutant leaves"
        ),
        "algorithm": [
            "solve the simultaneous rational commutant and its center",
            "split central sectors before multiplicity-space idempotents",
            "certify every recursive rational commutant idempotent",
            "classify every leaf as real, complex, or quaternionic Schur type",
            "group leaves by exact intertwiner spaces",
            "verify central isotypic projectors and m^2 dim(D) dimensions",
            "reconstruct an exact block-diagonal basis",
        ],
        "positive_controls": {
            name: certificate_json(certificate)
            for name, certificate in certificates.items()
        },
        "expected_block_signatures": {
            name: [list(row) for row in rows] for name, rows in expected_sorted.items()
        },
        "observed_block_signatures": {
            name: [list(row) for row in rows] for name, rows in observed.items()
        },
        "spin_two_casimir_equals_6_identity": spin_two_casimir == 6 * sp.eye(5),
        "missing_complete_reducibility_control": certificate_json(missing_assumption),
        "bounded_search_refusal_control": certificate_json(bounded_refusal),
        "trust_boundary": {
            "assumed": [
                "the supplied rational representation is completely reducible",
            ],
            "certified": [
                "all reported projectors and restricted actions",
                "irreducible R/C/H leaf types",
                "equivalence and inequivalence from intertwiner spaces",
                "central isotypic projectors",
                "double-centralizer dimension accounting",
            ],
            "not_claimed": [
                "termination for every rational semisimple representation",
                "automatic algebraic-field extension",
                "robust decomposition of noisy floating-point generators",
                "learned representation discovery or model superiority",
            ],
        },
        "passed": passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = diagnostics()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "IrreducibleSummandCertificate",
    "IsotypicBlockCertificate",
    "ReducibleIsotypicCertificate",
    "SplitWitness",
    "aligned_isotypic_basis",
    "canonical_reducible_examples",
    "certificate_json",
    "cl3_spin3_fixture",
    "decompose_reducible_representation",
    "diagnostics",
    "exact_algebra_center_basis",
    "exact_intertwiner_basis",
    "rationally_conjugate_representation",
    "repeated_representation",
    "spin9_slice_fixture",
    "spin_two_generators",
    "transform_to_isotypic_coordinates",
]
