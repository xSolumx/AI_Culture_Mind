"""Exact spinor and half-spinor branching over the global ``2.A5`` atlas.

This module takes every real orthogonal representation type classified by
``spin_dirac_a5_components.py`` and computes the restriction of the relevant
Spin(n) spinor module.  The construction is compositional in the Clifford
direct-sum rule.  Quaternionic irreducible blocks are derived from exact SU(2)
weights, not fitted from the final answer.

Every resulting branching is independently checked against the exterior
algebra identities for complex Clifford modules.  The reported invariant
spinors are representation-theoretic fixed vectors; no geometric Dirac
spectrum or kernel is inferred without an additional manifold and operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

from spin_dirac_a5_cohomology import build_exact_group_table
from spin_dirac_a5_components import (
    CHARACTER_ORDER,
    REAL_IRREP_KEYS,
    _inner_product,
    character_certificate,
    complex_characters,
    component_atlas,
)
from spin_dirac_a5_rigidity import FIELD
from sympy import QQ

Representation = dict[str, int]
FusionTable = dict[str, dict[str, Representation]]


def _field_integer(value: Any) -> int:
    expression = FIELD.to_sympy(value)
    if expression.is_Integer is not True:
        raise AssertionError(f"expected an integer multiplicity, got {expression}")
    return int(expression)


def _canonical_representation(representation: Representation) -> Representation:
    return {
        key: int(representation[key])
        for key in CHARACTER_ORDER
        if representation.get(key, 0)
    }


def _add_representations(*representations: Representation) -> Representation:
    result: Counter[str] = Counter()
    for representation in representations:
        result.update(representation)
    if any(multiplicity < 0 for multiplicity in result.values()):
        raise AssertionError("a representation sum acquired negative multiplicity")
    return _canonical_representation(dict(result))


def _scale_representation(
    representation: Representation, scalar: int
) -> Representation:
    if scalar < 0:
        raise ValueError("representation scaling must be nonnegative")
    return _canonical_representation(
        {key: scalar * multiplicity for key, multiplicity in representation.items()}
    )


def _decompose_character(
    values: tuple[Any, ...], characters: dict[str, tuple[Any, ...]]
) -> Representation:
    decomposition = {
        key: _field_integer(_inner_product(values, characters[key]))
        for key in CHARACTER_ORDER
    }
    if any(multiplicity < 0 for multiplicity in decomposition.values()):
        raise AssertionError(
            "a certified character decomposed with negative multiplicity"
        )
    result = _canonical_representation(decomposition)
    reconstructed = tuple(
        sum(
            (
                multiplicity * characters[key][element]
                for key, multiplicity in result.items()
            ),
            FIELD.zero,
        )
        for element in range(len(values))
    )
    if reconstructed != values:
        raise AssertionError("character decomposition did not reconstruct exactly")
    return result


def fusion_table(
    characters: dict[str, tuple[Any, ...]],
) -> FusionTable:
    """Return the complete exact ordinary representation ring."""

    return {
        left: {
            right: _decompose_character(
                tuple(
                    left_value * right_value
                    for left_value, right_value in zip(
                        characters[left], characters[right], strict=True
                    )
                ),
                characters,
            )
            for right in CHARACTER_ORDER
        }
        for left in CHARACTER_ORDER
    }


def _tensor_representations(
    left: Representation,
    right: Representation,
    fusion: FusionTable,
) -> Representation:
    result: Counter[str] = Counter()
    for left_key, left_multiplicity in left.items():
        for right_key, right_multiplicity in right.items():
            for target, target_multiplicity in fusion[left_key][right_key].items():
                result[target] += (
                    left_multiplicity * right_multiplicity * target_multiplicity
                )
    return _canonical_representation(dict(result))


def _representation_dimension(
    representation: Representation, degrees: dict[str, int]
) -> int:
    return sum(
        multiplicity * degrees[key] for key, multiplicity in representation.items()
    )


def _symmetric_power_values(
    defining: tuple[Any, ...], highest_weight: int
) -> tuple[Any, ...]:
    if highest_weight == 0:
        return tuple(FIELD.one for _ in defining)
    previous = tuple(FIELD.one for _ in defining)
    current = defining
    for _ in range(2, highest_weight + 1):
        following = tuple(
            defining_value * current_value - previous_value
            for defining_value, current_value, previous_value in zip(
                defining, current, previous, strict=True
            )
        )
        previous, current = current, following
    return current


def _su2_half_spin_symmetric_powers(
    vector_highest_weight: int,
) -> tuple[dict[int, int], dict[int, int]]:
    """Decompose both half-spin weight systems into SU(2) irreducibles."""

    if vector_highest_weight % 2 != 1:
        raise ValueError("a quaternionic SU(2) block needs odd highest weight")
    vector_weights = tuple(range(vector_highest_weight, -vector_highest_weight - 1, -2))
    weight_counters = (Counter(), Counter())
    for signs in product((1, -1), repeat=len(vector_weights)):
        parity = sum(sign < 0 for sign in signs) % 2
        numerator = sum(
            sign * weight for sign, weight in zip(signs, vector_weights, strict=True)
        )
        if numerator % 2:
            raise AssertionError("a spin weight was unexpectedly half-integral")
        weight_counters[parity][numerator // 2] += 1

    decompositions: list[dict[int, int]] = []
    for weights in weight_counters:
        remaining = Counter(weights)
        decomposition: dict[int, int] = {}
        while remaining:
            highest = max(remaining)
            multiplicity = remaining[highest]
            if multiplicity <= 0:
                raise AssertionError("SU(2) weight subtraction became nonpositive")
            decomposition[highest] = multiplicity
            for weight in range(highest, -highest - 1, -2):
                remaining[weight] -= multiplicity
                if remaining[weight] == 0:
                    del remaining[weight]
                elif remaining[weight] < 0:
                    raise AssertionError("invalid SU(2) highest-weight decomposition")
        decompositions.append(decomposition)
    return decompositions[0], decompositions[1]


def _restrict_su2_decomposition(
    decomposition: dict[int, int],
    defining: tuple[Any, ...],
    characters: dict[str, tuple[Any, ...]],
) -> Representation:
    result: Representation = {}
    for highest_weight, multiplicity in decomposition.items():
        restricted = _decompose_character(
            _symmetric_power_values(defining, highest_weight), characters
        )
        result = _add_representations(
            result, _scale_representation(restricted, multiplicity)
        )
    return result


def base_spinor_blocks(
    characters: dict[str, tuple[Any, ...]],
    fusion: FusionTable,
) -> dict[str, dict[str, object]]:
    """Build the spinor restriction of every irreducible real block."""

    one = {"1": 1}
    blocks: dict[str, dict[str, object]] = {
        "1_R": {"parity": "odd", "spin": one},
        "3_R": {"parity": "odd", "spin": {"2": 1}},
        "3p_R": {"parity": "odd", "spin": {"2_prime": 1}},
        "4_R": {
            "parity": "even",
            "plus": {"2": 1},
            "minus": {"2_prime": 1},
        },
        "5_R": {
            "parity": "odd",
            "spin": _decompose_character(
                _symmetric_power_values(characters["2"], 3), characters
            ),
        },
    }
    quaternionic_blocks = {
        "2_H": (1, "2"),
        "2p_H": (1, "2_prime"),
        "4_H": (3, "2"),
        "6_H": (5, "2"),
    }
    for key, (highest_weight, defining_key) in quaternionic_blocks.items():
        plus_su2, minus_su2 = _su2_half_spin_symmetric_powers(highest_weight)
        blocks[key] = {
            "parity": "even",
            "plus": _restrict_su2_decomposition(
                plus_su2, characters[defining_key], characters
            ),
            "minus": _restrict_su2_decomposition(
                minus_su2, characters[defining_key], characters
            ),
            "su2_vector_highest_weight": highest_weight,
            "su2_plus_decomposition": {
                str(weight): multiplicity
                for weight, multiplicity in sorted(plus_su2.items())
            },
            "su2_minus_decomposition": {
                str(weight): multiplicity
                for weight, multiplicity in sorted(minus_su2.items())
            },
        }

    vector_four = _tensor_representations({"2": 1}, {"2_prime": 1}, fusion)
    if vector_four != {"4_vector": 1}:
        raise AssertionError("the 4_R half-spin tensor did not recover its vector")
    if tuple(blocks) != REAL_IRREP_KEYS:
        raise AssertionError("base spinor blocks do not match the real irrep atlas")
    return blocks


def _direct_sum_spinors(
    left: dict[str, object],
    right: dict[str, object],
    fusion: FusionTable,
) -> dict[str, object]:
    left_even = left["parity"] == "even"
    right_even = right["parity"] == "even"
    if left_even and right_even:
        plus = _add_representations(
            _tensor_representations(left["plus"], right["plus"], fusion),
            _tensor_representations(left["minus"], right["minus"], fusion),
        )
        minus = _add_representations(
            _tensor_representations(left["plus"], right["minus"], fusion),
            _tensor_representations(left["minus"], right["plus"], fusion),
        )
        return {"parity": "even", "plus": plus, "minus": minus}
    if left_even and not right_even:
        return {
            "parity": "odd",
            "spin": _tensor_representations(
                _add_representations(left["plus"], left["minus"]),
                right["spin"],
                fusion,
            ),
        }
    if not left_even and right_even:
        return {
            "parity": "odd",
            "spin": _tensor_representations(
                left["spin"],
                _add_representations(right["plus"], right["minus"]),
                fusion,
            ),
        }
    product_spin = _tensor_representations(left["spin"], right["spin"], fusion)
    return {
        "parity": "even",
        "plus": product_spin,
        "minus": product_spin,
    }


def spinors_for_multiplicities(
    multiplicities: dict[str, int],
    blocks: dict[str, dict[str, object]],
    fusion: FusionTable,
) -> dict[str, object]:
    """Apply the exact Clifford direct-sum rule to one real representation."""

    state: dict[str, object] = {
        "parity": "even",
        "plus": {"1": 1},
        "minus": {},
    }
    for key in REAL_IRREP_KEYS:
        for _ in range(multiplicities.get(key, 0)):
            state = _direct_sum_spinors(state, blocks[key], fusion)
    return state


def _power_maps(
    table: tuple[tuple[int, ...], ...], maximum_power: int
) -> tuple[tuple[int, ...], ...]:
    maps = []
    for element in range(len(table)):
        powers = [0]
        current = 0
        for _ in range(maximum_power):
            current = table[current][element]
            powers.append(current)
        maps.append(tuple(powers))
    return tuple(maps)


def exterior_algebra_decompositions(
    vector_character: tuple[Any, ...],
    dimension: int,
    table: tuple[tuple[int, ...], ...],
    characters: dict[str, tuple[Any, ...]],
) -> tuple[Representation, Representation]:
    """Compute exact even/odd exterior characters using Newton identities."""

    power_maps = _power_maps(table, dimension)
    even_values = []
    odd_values = []
    for element in range(len(table)):
        elementary = [FIELD.one]
        for degree in range(1, dimension + 1):
            numerator = FIELD.zero
            for power in range(1, degree + 1):
                term = (
                    elementary[degree - power]
                    * vector_character[power_maps[element][power]]
                )
                numerator += term if power % 2 else -term
            elementary.append(FIELD.convert(QQ(1, degree)) * numerator)
        even_values.append(
            sum((elementary[k] for k in range(0, dimension + 1, 2)), FIELD.zero)
        )
        odd_values.append(
            sum((elementary[k] for k in range(1, dimension + 1, 2)), FIELD.zero)
        )
    return (
        _decompose_character(tuple(even_values), characters),
        _decompose_character(tuple(odd_values), characters),
    )


def _vector_character_for_component(
    multiplicities: dict[str, int],
    real_irreps: dict[str, dict[str, object]],
    characters: dict[str, tuple[Any, ...]],
) -> tuple[Any, ...]:
    coefficients: Counter[str] = Counter()
    for key, multiplicity in multiplicities.items():
        record = real_irreps[key]
        complexification_factor = 1 if record["division_algebra"] == "R" else 2
        coefficients[str(record["character"])] += multiplicity * complexification_factor
    return tuple(
        sum(
            (
                multiplicity * characters[key][element]
                for key, multiplicity in coefficients.items()
            ),
            FIELD.zero,
        )
        for element in range(len(next(iter(characters.values()))))
    )


def branching_atlas() -> dict[str, object]:
    """Compute and independently certify every requested spinor branching."""

    group, table = build_exact_group_table()
    characters = complex_characters(group)
    fusion = fusion_table(characters)
    character_report = character_certificate()
    components = component_atlas(character_report)
    blocks = base_spinor_blocks(characters, fusion)
    degrees = {
        record["key"]: int(record["complex_dimension"])
        for record in character_report["characters"]
    }
    real_irreps = {
        str(record["key"]): record for record in character_report["real_irreps"]
    }

    by_dimension: dict[str, object] = {}
    global_checks = []
    for dimension_text, dimension_atlas in components["by_dimension"].items():
        dimension = int(dimension_text)
        records = []
        for component in dimension_atlas["components"]:
            multiplicities = {
                str(key): int(value)
                for key, value in component["multiplicities"].items()
            }
            spinors = spinors_for_multiplicities(multiplicities, blocks, fusion)
            vector_character = _vector_character_for_component(
                multiplicities, real_irreps, characters
            )
            exterior_even, exterior_odd = exterior_algebra_decompositions(
                vector_character, dimension, table, characters
            )
            checks: dict[str, bool]
            if dimension % 2:
                spin = spinors["spin"]
                expected_dimension = 2 ** ((dimension - 1) // 2)
                checks = {
                    "spinor_dimension": _representation_dimension(spin, degrees)
                    == expected_dimension,
                    "clifford_even_identity": _tensor_representations(
                        spin, spin, fusion
                    )
                    == exterior_even,
                    "odd_hodge_character_matches_even": exterior_odd == exterior_even,
                }
                record = {
                    "id": component["id"],
                    "dimension": dimension,
                    "spinor": spin,
                    "invariant_spinors": spin.get("1", 0),
                    "exterior_even": exterior_even,
                    "checks": checks,
                }
            else:
                plus = spinors["plus"]
                minus = spinors["minus"]
                expected_half_dimension = 2 ** (dimension // 2 - 1)
                same_chiral_branching = plus == minus
                orientation_split = component["spin_conjugacy_components"] == 2
                checks = {
                    "plus_dimension": _representation_dimension(plus, degrees)
                    == expected_half_dimension,
                    "minus_dimension": _representation_dimension(minus, degrees)
                    == expected_half_dimension,
                    "clifford_even_identity": _add_representations(
                        _tensor_representations(plus, plus, fusion),
                        _tensor_representations(minus, minus, fusion),
                    )
                    == exterior_even,
                    "clifford_odd_identity": _add_representations(
                        _tensor_representations(plus, minus, fusion),
                        _tensor_representations(minus, plus, fusion),
                    )
                    == exterior_odd,
                    "unsplit_type_has_equal_chiral_branching": orientation_split
                    or same_chiral_branching,
                }
                record = {
                    "id": component["id"],
                    "dimension": dimension,
                    "orientation_A": {"plus": plus, "minus": minus},
                    "orientation_B": (
                        {"plus": minus, "minus": plus} if orientation_split else None
                    ),
                    "orientation_split": orientation_split,
                    "chirality_character_distinguishes_orientations": (
                        orientation_split and not same_chiral_branching
                    ),
                    "invariant_half_spinors_orientation_A": {
                        "plus": plus.get("1", 0),
                        "minus": minus.get("1", 0),
                    },
                    "exterior_even": exterior_even,
                    "exterior_odd": exterior_odd,
                    "checks": checks,
                }
            global_checks.extend(checks.values())
            records.append(record)

        by_dimension[dimension_text] = {
            "orthogonal_types": len(records),
            "spin_conjugacy_components": dimension_atlas["spin_conjugacy_components"],
            "types_with_invariant_spinors": sum(
                (
                    record.get("invariant_spinors", 0) > 0
                    if dimension % 2
                    else any(record["invariant_half_spinors_orientation_A"].values())
                )
                for record in records
            ),
            "orientation_split_types": sum(
                bool(record.get("orientation_split", False)) for record in records
            ),
            "orientation_splits_distinguished_by_chiral_character": sum(
                bool(
                    record.get("chirality_character_distinguishes_orientations", False)
                )
                for record in records
            ),
            "records": records,
        }

    base_block_report = {
        key: {
            field: value
            for field, value in block.items()
            if field
            in {
                "parity",
                "spin",
                "plus",
                "minus",
                "su2_vector_highest_weight",
                "su2_plus_decomposition",
                "su2_minus_decomposition",
            }
        }
        for key, block in blocks.items()
    }
    report = {
        "schema_version": 1,
        "experiment": "exact spinor branching over the global 2.A5 Spin atlas",
        "field": "Q(sqrt(5))",
        "base_spinor_blocks": base_block_report,
        "by_dimension": by_dimension,
        "verification": {
            "components_checked": sum(
                record["orthogonal_types"] for record in by_dimension.values()
            ),
            "clifford_identity_method": (
                "Newton exterior characters versus independently composed spinors"
            ),
            "all_checks_pass": all(global_checks),
        },
        "claim_scope": {
            "computer_assisted_exact": [
                "spinor or half-spinor irreducible multiplicities for all 245 orthogonal types",
                "SU(2)-weight derivation of all quaternionic base-block spinors",
                "independent even/odd exterior-algebra Clifford identities",
                "invariant-spinor and chirality-character counts",
            ],
            "not_claimed": [
                "a geometric Dirac spectrum without a manifold and Dirac operator",
                "that invariant spinors are automatically zero modes on an arbitrary geometry",
                "an ML or SSM advantage",
            ],
        },
        "passed": all(global_checks),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["certificate_sha256_without_self_hash"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = branching_atlas()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
