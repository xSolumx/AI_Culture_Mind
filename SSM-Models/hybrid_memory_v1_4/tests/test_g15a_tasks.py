"""Contracts for the prospective G15A task distribution."""

from __future__ import annotations

import torch
from spin8_triality import SPIN8_PAIRS

from hybrid_memory_v1_4.g15a_tasks import (
    CENTER_STEPS,
    NO_SYMMETRY_VALUE_START,
    OFF_TORUS_PAIRS,
    QUERY_TOKEN,
    SYMMETRY_TARGET_START,
    generate_no_symmetry_batch,
    generate_symmetry_batch,
)


def test_g15a_batches_are_deterministic_final_only_and_disjoint() -> None:
    first = generate_symmetry_batch(40, 64, seed=2131)
    second = generate_symmetry_batch(40, 64, seed=2131)
    assert first.fingerprint() == second.fingerprint()
    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(first.coordinates, second.coordinates)
    assert bool((first.token_ids[:, -1] == QUERY_TOKEN).all())
    assert not bool((first.token_ids[:, -1] == first.targets).any())
    assert int(first.targets.min()) == SYMMETRY_TARGET_START
    assert int(first.targets.max()) == SYMMETRY_TARGET_START + 9

    no_symmetry = generate_no_symmetry_batch(32, 256, seed=2131)
    assert not bool(no_symmetry.coordinates.any())
    assert torch.equal(no_symmetry.token_ids[:, 0], no_symmetry.targets)
    assert int(no_symmetry.targets.min()) == NO_SYMMETRY_VALUE_START


def test_off_torus_classes_are_invisible_to_fixed_t4_but_center_is_visible() -> None:
    batch = generate_symmetry_batch(100, 64, seed=2137)
    commuting = torch.zeros(28, dtype=torch.bool)
    for pair in ((0, 1), (2, 3), (4, 5), (6, 7)):
        commuting[SPIN8_PAIRS.index(pair)] = True
    for pair in OFF_TORUS_PAIRS:
        assert not bool(commuting[SPIN8_PAIRS.index(pair)])

    off_torus = batch.labels < len(OFF_TORUS_PAIRS)
    identity = batch.labels == 8
    center = batch.labels == 9
    assert not bool(batch.coordinates[off_torus][..., commuting].any())
    assert not bool(batch.coordinates[identity].any())
    assert bool(batch.coordinates[center][..., commuting].any())
    center_nonzero_steps = (batch.coordinates[center].abs().sum(dim=(-1, -2)) > 0).sum(
        dim=-1
    )
    assert bool((center_nonzero_steps == CENTER_STEPS).all())
