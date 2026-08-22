from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from spin_delta_capability_gate import (
    QUERY_TOKEN,
    VALUE_OFFSET,
    WRITE_TOKEN,
    overwrite_retrieval_batch,
)
from summarize_spin_delta_capability import EXPECTED_SEEDS, summarize


def test_overwrite_batch_target_is_latest_queried_value() -> None:
    generator = torch.Generator().manual_seed(202_608_53)
    inputs, targets = overwrite_retrieval_batch(64, 12, generator=generator)
    assert inputs.shape == (64, 38)
    assert torch.all(inputs[:, 0:36:3] == WRITE_TOKEN)
    assert torch.all(inputs[:, -2] == QUERY_TOKEN)
    keys = inputs[:, 1:36:3]
    values = inputs[:, 2:36:3]
    query = inputs[:, -1]
    positions = torch.arange(12).expand(64, -1)
    latest = torch.where(keys == query[:, None], positions, -1).max(dim=1).values
    assert torch.equal(targets, values.gather(1, latest[:, None]).squeeze(1))
    assert torch.all(targets >= VALUE_OFFSET)


def test_overwrite_batch_is_reproducible_and_covers_both_keys() -> None:
    left = overwrite_retrieval_batch(
        32, 8, generator=torch.Generator().manual_seed(73)
    )
    right = overwrite_retrieval_batch(
        32, 8, generator=torch.Generator().manual_seed(73)
    )
    assert torch.equal(left[0], right[0]) and torch.equal(left[1], right[1])
    keys = left[0][:, 1:24:3]
    assert torch.all(keys.min(dim=1).values == 0)
    assert torch.all(keys.max(dim=1).values == 1)


def test_oracle_slots_mark_only_value_writes_and_final_query() -> None:
    inputs, _, oracle = overwrite_retrieval_batch(
        16,
        8,
        generator=torch.Generator().manual_seed(79),
        return_oracle=True,
    )
    assert torch.equal(oracle[:, 2:24:3, 0], inputs[:, 1:24:3])
    assert torch.equal(oracle[:, -1, 1], inputs[:, -1])
    write_mask = oracle[..., 0] >= 0
    query_mask = oracle[..., 1] >= 0
    assert torch.all(write_mask.sum(dim=1) == 8)
    assert torch.all(query_mask.sum(dim=1) == 1)


def _artifact(seed: int, baseline: float, candidate: float) -> dict[str, object]:
    metrics = lambda accuracy: {
        str(writes): {"accuracy": accuracy, "nats_per_query": 0.1}
        for writes in (8, 16, 32)
    }
    return {
        "stage": "spin_delta_overwrite_capability",
        "variant_order": ["independent_v1_2", "spin_delta"],
        "config": {"seed": seed},
        "pairing": {
            "common_parameters_bitwise_equal": True,
            "maximum_absolute_logit_difference": 1.0e-6,
        },
        "implementation_sha256": {"same": "hash"},
        "task": {"same": "task"},
        "results": [
            {"final": metrics(baseline)},
            {"final": metrics(candidate)},
        ],
    }


def test_capability_summary_enforces_absolute_and_differential_gates(
    tmp_path: Path,
) -> None:
    paths = []
    for index, seed in enumerate(EXPECTED_SEEDS):
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(json.dumps(_artifact(seed, 0.82, 0.93 + 0.01 * index)))
        paths.append(path)
    report = summarize(paths)
    assert report["candidate_capability_pass"] is True
    assert report["differential_advantage_pass"] is True


def test_capability_summary_rejects_pairing_failure(tmp_path: Path) -> None:
    paths = []
    for seed in EXPECTED_SEEDS:
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(json.dumps(_artifact(seed, 0.82, 0.94)))
        paths.append(path)
    payload = json.loads(paths[0].read_text())
    payload["pairing"]["maximum_absolute_logit_difference"] = 2.1e-6
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="pairing residual"):
        summarize(paths)
