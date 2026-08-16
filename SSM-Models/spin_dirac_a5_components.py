"""Exact low-dimensional Spin-component atlas for the binary icosahedral group.

The exact quaternion table from :mod:`spin_dirac_a5_cohomology` determines the
nine conjugacy classes of ``2.A5``.  Starting from the defining two-dimensional
character ``chi(g) = 2 Re(g)``, this module reconstructs all nine ordinary
irreducible characters, verifies their exact orthogonality and Frobenius--Schur
indicators, and checks the complete tensor-product ring.

Those character certificates classify real orthogonal representations through
the requested dimensions.  The standard theorem that ``2.A5`` is the universal
central extension of ``A5`` supplies the separate mod-2 lifting input:
``H1(2.A5; Z/2) = H2(2.A5; Z/2) = 0``.  Consequently every oriented real
representation lifts uniquely to Spin.  This theorem input is deliberately not
conflated with the characteristic-zero cochain contraction in
``spin_dirac_a5_cohomology.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spin_dirac_a5_cohomology import build_exact_group_table
from spin_dirac_a5_ladder import LADDER_DIMENSIONS
from spin_dirac_a5_rigidity import FIELD
from sympy import QQ, sqrt


@dataclass(frozen=True)
class RealIrrep:
    """One irreducible real module and its Schur-commutant type."""

    key: str
    character: str
    real_dimension: int
    division_algebra: str
    center_eigenvalue: int


CHARACTER_ORDER = (
    "1",
    "2",
    "2_prime",
    "3",
    "3_prime",
    "4_vector",
    "4_spin",
    "5",
    "6_spin",
)

REAL_IRREP_KEYS = (
    "1_R",
    "3_R",
    "3p_R",
    "4_R",
    "5_R",
    "2_H",
    "2p_H",
    "4_H",
    "6_H",
)


def _field_integer(value: Any) -> int:
    expression = FIELD.to_sympy(value)
    if expression.is_Integer is not True:
        raise AssertionError(f"expected an exact integer, received {expression}")
    return int(expression)


def _field_string(value: Any) -> str:
    return str(FIELD.to_sympy(value))


def _galois_conjugate(value: Any) -> Any:
    expression = FIELD.to_sympy(value)
    return FIELD.from_sympy(expression.subs(sqrt(5), -sqrt(5)))


def _sum_field(values: Any) -> Any:
    total = FIELD.zero
    for value in values:
        total += value
    return total


def inverse_indices(table: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Return the inverse of every exact table element."""

    identity = 0
    return tuple(
        next(
            candidate
            for candidate in range(len(table))
            if table[element][candidate] == identity
            and table[candidate][element] == identity
        )
        for element in range(len(table))
    )


def element_order(table: tuple[tuple[int, ...], ...], element: int) -> int:
    """Return one element order by exact repeated multiplication."""

    identity = 0
    value = identity
    for exponent in range(1, len(table) + 1):
        value = table[value][element]
        if value == identity:
            return exponent
    raise AssertionError("element order exceeded the group order")


def conjugacy_classes(
    table: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate exact conjugacy classes, with deterministic class ordering."""

    inverses = inverse_indices(table)
    unseen = set(range(len(table)))
    classes: list[tuple[int, ...]] = []
    while unseen:
        representative = min(unseen)
        conjugates = {
            table[table[group_element][representative]][inverses[group_element]]
            for group_element in range(len(table))
        }
        conjugacy_class = tuple(sorted(conjugates))
        classes.append(conjugacy_class)
        unseen.difference_update(conjugates)
    return tuple(classes)


def _derived_subgroup_size(table: tuple[tuple[int, ...], ...]) -> int:
    inverses = inverse_indices(table)
    commutators = {
        table[table[table[inverses[left]][inverses[right]]][left]][right]
        for left in range(len(table))
        for right in range(len(table))
    }
    reached = {0}
    queue = [0]
    offset = 0
    while offset < len(queue):
        current = queue[offset]
        offset += 1
        for generator in commutators:
            candidate = table[current][generator]
            if candidate not in reached:
                reached.add(candidate)
                queue.append(candidate)
    return len(reached)


def _center_indices(table: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(
        element
        for element in range(len(table))
        if all(
            table[element][other] == table[other][element]
            for other in range(len(table))
        )
    )


def complex_characters(
    group: tuple[tuple[Any, Any, Any, Any], ...],
) -> dict[str, tuple[Any, ...]]:
    """Construct all ordinary irreducible characters from the defining trace."""

    one = FIELD.one
    two = FIELD.convert(QQ(2))
    three = FIELD.convert(QQ(3))
    four = FIELD.convert(QQ(4))
    characters: dict[str, list[Any]] = {key: [] for key in CHARACTER_ORDER}
    for quaternion in group:
        defining = two * quaternion[0]
        defining_prime = _galois_conjugate(defining)
        three_dimensional = defining * defining - one
        three_prime = defining_prime * defining_prime - one
        spin_four = defining**3 - two * defining
        five_dimensional = defining**4 - three * defining * defining + one
        spin_six = defining**5 - four * defining**3 + three * defining
        vector_four = three_dimensional * three_prime - five_dimensional
        values = {
            "1": one,
            "2": defining,
            "2_prime": defining_prime,
            "3": three_dimensional,
            "3_prime": three_prime,
            "4_vector": vector_four,
            "4_spin": spin_four,
            "5": five_dimensional,
            "6_spin": spin_six,
        }
        for key in CHARACTER_ORDER:
            characters[key].append(values[key])
    return {key: tuple(values) for key, values in characters.items()}


def _inner_product(left: tuple[Any, ...], right: tuple[Any, ...]) -> Any:
    inverse_order = FIELD.convert(QQ(1, len(left)))
    return inverse_order * _sum_field(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def _mckay_graph_certificate(
    tensor_by_defining: dict[str, dict[str, int]],
) -> dict[str, object]:
    adjacency = {node: set(neighbors) for node, neighbors in tensor_by_defining.items()}
    symmetric = all(
        node in adjacency.get(neighbor, set())
        for node, neighbors in adjacency.items()
        for neighbor in neighbors
    )
    coefficients_are_one = all(
        coefficient == 1
        for decomposition in tensor_by_defining.values()
        for coefficient in decomposition.values()
    )
    loop_free = all(node not in neighbors for node, neighbors in adjacency.items())
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    reached = set()
    queue = [CHARACTER_ORDER[0]]
    while queue:
        current = queue.pop()
        if current in reached:
            continue
        reached.add(current)
        queue.extend(adjacency[current] - reached)
    branch_nodes = [
        node for node, neighbors in adjacency.items() if len(neighbors) == 3
    ]
    arm_lengths: list[int] = []
    if len(branch_nodes) == 1:
        branch = branch_nodes[0]
        for neighbor in adjacency[branch]:
            previous = branch
            current = neighbor
            length = 1
            while len(adjacency[current]) == 2:
                following = next(
                    item for item in adjacency[current] if item != previous
                )
                previous, current = current, following
                length += 1
            if len(adjacency[current]) != 1:
                arm_lengths = []
                break
            arm_lengths.append(length)
    arm_lengths.sort()
    checks = {
        "adjacency_is_symmetric": symmetric,
        "all_edge_multiplicities_are_one": coefficients_are_one,
        "graph_is_loop_free": loop_free,
        "graph_is_connected": len(reached) == len(CHARACTER_ORDER),
        "graph_is_a_tree": edge_count == len(CHARACTER_ORDER) - 1,
        "unique_trivalent_node": len(branch_nodes) == 1,
        "affine_e8_arm_lengths": arm_lengths == [1, 2, 5],
    }
    return {
        "type": "affine_E8",
        "nodes": len(CHARACTER_ORDER),
        "edges": edge_count,
        "branch_node": branch_nodes[0] if len(branch_nodes) == 1 else None,
        "arm_lengths": arm_lengths,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _class_records(
    group: tuple[tuple[Any, Any, Any, Any], ...],
    table: tuple[tuple[int, ...], ...],
    classes: tuple[tuple[int, ...], ...],
    characters: dict[str, tuple[Any, ...]],
    center_minus_one: int,
) -> tuple[list[dict[str, object]], dict[int, str]]:
    preliminary = []
    for conjugacy_class in classes:
        representative = conjugacy_class[0]
        preliminary.append(
            {
                "elements": conjugacy_class,
                "order": element_order(table, representative),
                "trace": characters["2"][representative],
            }
        )
    preliminary.sort(
        key=lambda record: (
            int(record["order"]),
            -float(FIELD.to_sympy(record["trace"]).evalf(30)),
        )
    )
    order_counts = Counter(int(record["order"]) for record in preliminary)
    order_offsets: Counter[int] = Counter()
    index_to_label: dict[int, str] = {}
    for record in preliminary:
        order = int(record["order"])
        suffix = chr(ord("A") + order_offsets[order])
        order_offsets[order] += 1
        label = f"{order}{suffix}"
        record["label"] = label
        for element in record["elements"]:
            index_to_label[element] = label
        if order_counts[order] == 1 and suffix != "A":
            raise AssertionError("a unique class received a non-A suffix")

    records: list[dict[str, object]] = []
    for record in preliminary:
        representative = record["elements"][0]
        center_partner = table[center_minus_one][representative]
        records.append(
            {
                "label": record["label"],
                "size": len(record["elements"]),
                "element_order": record["order"],
                "defining_trace": _field_string(record["trace"]),
                "minus_class": index_to_label[center_partner],
                "characters": {
                    key: _field_string(characters[key][representative])
                    for key in CHARACTER_ORDER
                },
            }
        )
    return records, index_to_label


def character_certificate() -> dict[str, object]:
    """Return exact class, character, Schur-type, and tensor certificates."""

    group, table = build_exact_group_table()
    classes = conjugacy_classes(table)
    characters = complex_characters(group)
    center = _center_indices(table)
    minus_one = next(
        index for index in center if index != 0 and element_order(table, index) == 2
    )
    class_records, index_to_label = _class_records(
        group, table, classes, characters, minus_one
    )

    class_constancy = all(
        all(
            characters[key][element] == characters[key][conjugacy_class[0]]
            for element in conjugacy_class
        )
        for key in CHARACTER_ORDER
        for conjugacy_class in classes
    )
    gram = [
        [
            _field_integer(_inner_product(characters[left], characters[right]))
            for right in CHARACTER_ORDER
        ]
        for left in CHARACTER_ORDER
    ]
    gram_is_identity = gram == [
        [int(row == column) for column in range(len(CHARACTER_ORDER))]
        for row in range(len(CHARACTER_ORDER))
    ]
    degrees = {key: _field_integer(characters[key][0]) for key in CHARACTER_ORDER}
    degree_square_sum = sum(degree * degree for degree in degrees.values())
    squares = tuple(table[element][element] for element in range(len(table)))
    indicators = {
        key: _field_integer(
            FIELD.convert(QQ(1, len(table)))
            * _sum_field(characters[key][square] for square in squares)
        )
        for key in CHARACTER_ORDER
    }
    center_eigenvalues = {
        key: _field_integer(characters[key][minus_one] / FIELD.convert(degrees[key]))
        for key in CHARACTER_ORDER
    }

    tensor_coefficients: dict[tuple[str, str], dict[str, int]] = {}
    tensor_passed = True
    maximum_tensor_coefficient = 0
    for left in CHARACTER_ORDER:
        for right in CHARACTER_ORDER:
            product = tuple(
                left_value * right_value
                for left_value, right_value in zip(
                    characters[left], characters[right], strict=True
                )
            )
            decomposition: dict[str, int] = {}
            for target in CHARACTER_ORDER:
                coefficient = _field_integer(
                    _inner_product(product, characters[target])
                )
                if coefficient:
                    decomposition[target] = coefficient
                    maximum_tensor_coefficient = max(
                        maximum_tensor_coefficient, coefficient
                    )
                tensor_passed &= coefficient >= 0
            reconstructed = tuple(
                _sum_field(
                    coefficient * characters[target][element]
                    for target, coefficient in decomposition.items()
                )
                for element in range(len(group))
            )
            tensor_passed &= reconstructed == product
            tensor_coefficients[(left, right)] = decomposition

    tensor_by_defining = {
        right: tensor_coefficients[("2", right)] for right in CHARACTER_ORDER
    }
    mckay_graph = _mckay_graph_certificate(tensor_by_defining)

    real_irreps = []
    real_key_by_character = {
        "1": "1_R",
        "3": "3_R",
        "3_prime": "3p_R",
        "4_vector": "4_R",
        "5": "5_R",
        "2": "2_H",
        "2_prime": "2p_H",
        "4_spin": "4_H",
        "6_spin": "6_H",
    }
    for character in CHARACTER_ORDER:
        indicator = indicators[character]
        if indicator not in (-1, 1):
            raise AssertionError("2.A5 unexpectedly produced a complex-type irrep")
        division_algebra = "R" if indicator == 1 else "H"
        real_dimension = (
            degrees[character] if indicator == 1 else 2 * degrees[character]
        )
        real_irreps.append(
            RealIrrep(
                key=real_key_by_character[character],
                character=character,
                real_dimension=real_dimension,
                division_algebra=division_algebra,
                center_eigenvalue=center_eigenvalues[character],
            )
        )
    real_irreps.sort(key=lambda irrep: REAL_IRREP_KEYS.index(irrep.key))

    order_four_class = next(
        record for record in class_records if record["element_order"] == 4
    )
    order_four_representative = next(
        element
        for element, label in index_to_label.items()
        if label == order_four_class["label"]
    )
    a5_spin_extension_exponents: dict[str, int] = {}
    for irrep in real_irreps:
        if irrep.center_eigenvalue != 1:
            continue
        trace = _field_integer(characters[irrep.character][order_four_representative])
        minus_eigen_dimension = (irrep.real_dimension - trace) // 2
        if minus_eigen_dimension % 2:
            raise AssertionError("an A5 involution had odd negative eigenspace")
        a5_spin_extension_exponents[irrep.key] = (minus_eigen_dimension // 2) % 2

    derived_order = _derived_subgroup_size(table)
    exact_checks = {
        "group_order_is_120": len(group) == 120,
        "center_is_order_two": len(center) == 2,
        "derived_subgroup_is_whole_group": derived_order == len(group),
        "nine_conjugacy_classes": len(classes) == 9,
        "class_sizes_sum_to_group_order": sum(len(cls) for cls in classes)
        == len(group),
        "characters_are_class_functions": class_constancy,
        "character_gram_is_identity": gram_is_identity,
        "degree_squares_sum_to_120": degree_square_sum == len(group),
        "frobenius_schur_types_are_real_or_quaternionic": all(
            indicator in (-1, 1) for indicator in indicators.values()
        ),
        "tensor_ring_has_nonnegative_integral_coefficients": tensor_passed,
        "defining_mckay_graph_is_affine_e8": mckay_graph["passed"],
        "real_irrep_keys_are_complete": tuple(irrep.key for irrep in real_irreps)
        == REAL_IRREP_KEYS,
        "a5_nontrivial_irreps_have_nontrivial_spin_extension": all(
            exponent == (key != "1_R")
            for key, exponent in a5_spin_extension_exponents.items()
        ),
    }
    return {
        "group_order": len(group),
        "center_indices": list(center),
        "minus_one_index": minus_one,
        "derived_subgroup_order": derived_order,
        "conjugacy_classes": class_records,
        "characters": [
            {
                "key": key,
                "complex_dimension": degrees[key],
                "frobenius_schur_indicator": indicators[key],
                "center_eigenvalue": center_eigenvalues[key],
                "values_by_class": {
                    record["label"]: record["characters"][key]
                    for record in class_records
                },
            }
            for key in CHARACTER_ORDER
        ],
        "character_gram": gram,
        "degree_square_sum": degree_square_sum,
        "real_irreps": [
            {
                "key": irrep.key,
                "character": irrep.character,
                "real_dimension": irrep.real_dimension,
                "division_algebra": irrep.division_algebra,
                "center_eigenvalue": irrep.center_eigenvalue,
            }
            for irrep in real_irreps
        ],
        "a5_spin_extension_exponents": a5_spin_extension_exponents,
        "tensor_by_defining_2": tensor_by_defining,
        "mckay_graph": mckay_graph,
        "tensor_ring": {
            "pairs_checked": len(CHARACTER_ORDER) ** 2,
            "maximum_coefficient": maximum_tensor_coefficient,
            "passed": bool(tensor_passed),
        },
        "exact_checks": exact_checks,
        "passed": all(exact_checks.values()),
    }


def _real_irreps_from_certificate(
    certificate: dict[str, object],
) -> tuple[RealIrrep, ...]:
    records = certificate["real_irreps"]
    if not isinstance(records, list):
        raise TypeError("real irrep certificate must be a list")
    return tuple(
        RealIrrep(
            key=str(record["key"]),
            character=str(record["character"]),
            real_dimension=int(record["real_dimension"]),
            division_algebra=str(record["division_algebra"]),
            center_eigenvalue=int(record["center_eigenvalue"]),
        )
        for record in records
    )


def _multiplicity_vectors(
    irreps: tuple[RealIrrep, ...], dimension: int
) -> tuple[tuple[int, ...], ...]:
    vectors: list[tuple[int, ...]] = []

    def visit(offset: int, remaining: int, prefix: tuple[int, ...]) -> None:
        if offset == len(irreps) - 1:
            irrep_dimension = irreps[offset].real_dimension
            if remaining % irrep_dimension == 0:
                vectors.append((*prefix, remaining // irrep_dimension))
            return
        irrep_dimension = irreps[offset].real_dimension
        for multiplicity in range(remaining // irrep_dimension + 1):
            visit(
                offset + 1,
                remaining - multiplicity * irrep_dimension,
                (*prefix, multiplicity),
            )

    visit(0, dimension, ())
    return tuple(vectors)


def _generating_function_coefficient(weights: tuple[int, ...], degree: int) -> int:
    """Coefficient of ``prod_i (1-x^weights[i])^-1`` through one degree."""

    coefficients = [0] * (degree + 1)
    coefficients[0] = 1
    for weight in weights:
        for offset in range(weight, degree + 1):
            coefficients[offset] += coefficients[offset - weight]
    return coefficients[degree]


def _centralizer_lie_dimension(irrep: RealIrrep, multiplicity: int) -> int:
    if irrep.division_algebra == "R":
        return multiplicity * (multiplicity - 1) // 2
    if irrep.division_algebra == "H":
        return multiplicity * (2 * multiplicity + 1)
    if irrep.division_algebra == "C":
        return multiplicity * multiplicity
    raise AssertionError(f"unknown division algebra {irrep.division_algebra}")


def component_record(
    irreps: tuple[RealIrrep, ...],
    multiplicities: tuple[int, ...],
    a5_extension_exponents: dict[str, int],
    dimension: int,
) -> dict[str, object]:
    """Classify one orthogonal isomorphism type and its Spin conjugacy orbit(s)."""

    multiplicity_map = {
        irrep.key: multiplicity
        for irrep, multiplicity in zip(irreps, multiplicities, strict=True)
    }
    nonzero_multiplicities = {
        key: value for key, value in multiplicity_map.items() if value
    }
    center_minus_dimension = sum(
        irrep.real_dimension * multiplicity
        for irrep, multiplicity in zip(irreps, multiplicities, strict=True)
        if irrep.center_eigenvalue == -1
    )
    centralizer_lie_dimension = sum(
        _centralizer_lie_dimension(irrep, multiplicity)
        for irrep, multiplicity in zip(irreps, multiplicities, strict=True)
    )
    orientation_reversing_centralizer = any(
        irrep.division_algebra == "R"
        and irrep.real_dimension % 2 == 1
        and multiplicity > 0
        for irrep, multiplicity in zip(irreps, multiplicities, strict=True)
    )
    spin_component_multiplicity = 1 if orientation_reversing_centralizer else 2
    a5_projecting = center_minus_dimension == 0
    a5_spin_extension_exponent = sum(
        multiplicity_map[key] * exponent
        for key, exponent in a5_extension_exponents.items()
    )

    if center_minus_dimension == 0:
        spin_center_sign = -1 if a5_spin_extension_exponent % 2 else 1
        center_images = ["-1" if spin_center_sign == -1 else "+1"]
        has_nontrivial_action = any(
            multiplicity
            for key, multiplicity in multiplicity_map.items()
            if key != "1_R"
        )
        if not has_nontrivial_action:
            kernel = "all_2.A5"
        elif spin_center_sign == 1:
            kernel = "center_order_2"
        else:
            kernel = "trivial"
    elif center_minus_dimension == dimension:
        spin_center_sign = None
        center_images = [f"+volume_{dimension}", f"-volume_{dimension}"]
        kernel = "trivial"
        if spin_component_multiplicity != 2:
            raise AssertionError("a fully center-negative type must orientation-split")
    else:
        spin_center_sign = None
        center_images = [f"grade_{center_minus_dimension}_involution"]
        kernel = "trivial"

    centralizer_factors = []
    for irrep, multiplicity in zip(irreps, multiplicities, strict=True):
        if not multiplicity:
            continue
        family = {
            "R": "O",
            "C": "U",
            "H": "Sp",
        }[irrep.division_algebra]
        centralizer_factors.append(f"{family}({multiplicity})[{irrep.key}]")

    component_id = "+".join(
        f"{multiplicity}*{irrep.key}"
        for irrep, multiplicity in zip(irreps, multiplicities, strict=True)
        if multiplicity
    )
    return {
        "id": component_id,
        "dimension": dimension,
        "multiplicities": nonzero_multiplicities,
        "a5_projecting": a5_projecting,
        "center_minus_eigenspace_dimension": center_minus_dimension,
        "center_plus_eigenspace_dimension": dimension - center_minus_dimension,
        "spin_center_scalar_sign": spin_center_sign,
        "spin_center_images_across_oriented_components": center_images,
        "homomorphism_kernel": kernel,
        "centralizer_factors_in_O": centralizer_factors,
        "centralizer_lie_dimension": centralizer_lie_dimension,
        "orbit_dimension": dimension * (dimension - 1) // 2 - centralizer_lie_dimension,
        "orientation_reversing_centralizer_exists": orientation_reversing_centralizer,
        "spin_conjugacy_components": spin_component_multiplicity,
    }


def component_atlas(
    character_report: dict[str, object],
    dimensions: tuple[int, ...] = LADDER_DIMENSIONS,
) -> dict[str, object]:
    """Enumerate all Spin-conjugacy components in the selected dimensions."""

    irreps = _real_irreps_from_certificate(character_report)
    extension_exponents = {
        str(key): int(value)
        for key, value in character_report["a5_spin_extension_exponents"].items()
    }
    by_dimension: dict[str, object] = {}
    all_passed = True
    for dimension in dimensions:
        records = [
            component_record(
                irreps,
                multiplicities,
                extension_exponents,
                dimension,
            )
            for multiplicities in _multiplicity_vectors(irreps, dimension)
        ]
        spin_components = sum(
            int(record["spin_conjugacy_components"]) for record in records
        )
        generating_function_types = _generating_function_coefficient(
            tuple(irrep.real_dimension for irrep in irreps), dimension
        )
        split_generating_function_types = _generating_function_coefficient(
            tuple(
                irrep.real_dimension
                for irrep in irreps
                if not (irrep.division_algebra == "R" and irrep.real_dimension % 2 == 1)
            ),
            dimension,
        )
        checks = {
            "every_record_has_requested_dimension": all(
                sum(
                    next(irrep.real_dimension for irrep in irreps if irrep.key == key)
                    * multiplicity
                    for key, multiplicity in record["multiplicities"].items()
                )
                == dimension
                for record in records
            ),
            "center_eigenspaces_partition_dimension": all(
                int(record["center_minus_eigenspace_dimension"])
                + int(record["center_plus_eigenspace_dimension"])
                == dimension
                for record in records
            ),
            "spin_component_multiplicity_is_one_or_two": all(
                record["spin_conjugacy_components"] in (1, 2) for record in records
            ),
            "a5_projecting_iff_center_minus_space_is_zero": all(
                bool(record["a5_projecting"])
                == (record["center_minus_eigenspace_dimension"] == 0)
                for record in records
            ),
            "type_count_matches_independent_generating_function": len(records)
            == generating_function_types,
            "orientation_split_count_matches_even_module_generating_function": sum(
                record["spin_conjugacy_components"] == 2 for record in records
            )
            == split_generating_function_types,
        }
        all_passed &= all(checks.values())
        by_dimension[str(dimension)] = {
            "orthogonal_isomorphism_types": len(records),
            "generating_function_type_count": generating_function_types,
            "spin_conjugacy_components": spin_components,
            "orientation_split_types": sum(
                record["spin_conjugacy_components"] == 2 for record in records
            ),
            "even_module_generating_function_split_count": split_generating_function_types,
            "a5_projecting_types": sum(record["a5_projecting"] for record in records),
            "a5_projecting_spin_components": sum(
                record["spin_conjugacy_components"]
                for record in records
                if record["a5_projecting"]
            ),
            "faithful_spin_components": sum(
                record["spin_conjugacy_components"]
                for record in records
                if record["homomorphism_kernel"] == "trivial"
            ),
            "factor_through_a5_spin_components": sum(
                record["spin_conjugacy_components"]
                for record in records
                if record["homomorphism_kernel"] == "center_order_2"
            ),
            "trivial_spin_components": sum(
                record["spin_conjugacy_components"]
                for record in records
                if record["homomorphism_kernel"] == "all_2.A5"
            ),
            "checks": checks,
            "components": records,
        }
    return {
        "dimensions": list(dimensions),
        "classification_basis": (
            "complete reducibility into certified real irreducibles, unique Spin "
            "lifting for the superperfect group 2.A5, and the O-to-SO centralizer "
            "orientation criterion"
        ),
        "by_dimension": by_dimension,
        "passed": bool(all_passed),
    }


def diagnostics() -> dict[str, object]:
    characters = character_certificate()
    atlas = component_atlas(characters)
    payload = {
        "schema_version": 1,
        "experiment": "exact 2.A5 character and low-dimensional Spin-component atlas",
        "field": "Q(sqrt(5))",
        "character_certificate": characters,
        "spin_lift_theorem_input": {
            "statement": (
                "2.A5 is the universal central extension of A5; hence it is "
                "superperfect and every oriented real representation lifts uniquely "
                "through Spin(n) -> SO(n)"
            ),
            "h1_with_z2": 0,
            "h2_with_z2": 0,
            "status": "standard theorem input, separated from exact table checks",
        },
        "component_atlas": atlas,
        "claim_scope": {
            "computer_assisted_exact": [
                "the 120-element table has nine conjugacy classes and is perfect",
                "the nine displayed characters are a complete orthonormal character table",
                "all Frobenius-Schur indicators and center eigenvalues",
                "all 81 tensor products have nonnegative integral decompositions",
                "all real-module multiplicity vectors in dimensions 3,8,9,10,11,12",
                "centralizer Lie dimensions and O-to-SO orientation splitting",
            ],
            "theorem_backed": [
                "the Schur multiplier of A5 has order two",
                "the exact binary group is the universal central extension 2.A5",
                "all oriented real 2.A5 representations lift uniquely to Spin",
            ],
            "not_claimed": [
                "an unrestricted Spin(8) Dirac-Gram inequality",
                "triality outside Spin(8)",
                "an ML or SSM advantage",
                "that characteristic-zero H2 proves the mod-2 Spin-lifting statement",
            ],
        },
    }
    payload["passed"] = bool(characters["passed"] and atlas["passed"])
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["certificate_sha256_without_self_hash"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = diagnostics()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
