from __future__ import annotations

import json
from pathlib import Path

import pytest

from summarize_spin_delta_oracle_gate import EXPECTED_SEEDS, summarize


def _artifact(seed: int, learned: float, oracle: float) -> dict[str, object]:
    metrics = lambda accuracy: {
        str(writes): {"accuracy": accuracy} for writes in (8, 16, 32)
    }
    return {
        "stage": "spin_delta_oracle_address_intervention",
        "variant_order": ["learned_addresses", "oracle_addresses"],
        "config": {"seed": seed},
        "pairing": {"all_parameters_bitwise_equal": True},
        "implementation_sha256": {"same": "hash"},
        "intervention": {"same": "intervention"},
        "results": [
            {"final": metrics(learned)},
            {"final": metrics(oracle)},
        ],
    }


def _write(tmp_path: Path, learned: float, oracle: float) -> list[Path]:
    paths = []
    for seed in EXPECTED_SEEDS:
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(json.dumps(_artifact(seed, learned, oracle)))
        paths.append(path)
    return paths


def test_oracle_summary_accepts_capacity_and_rescue(tmp_path: Path) -> None:
    report = summarize(_write(tmp_path, 0.85, 0.97))
    assert report["oracle_capacity_pass"] is True
    assert report["address_inference_bottleneck_pass"] is True


def test_oracle_summary_rejects_parameter_mismatch(tmp_path: Path) -> None:
    paths = _write(tmp_path, 0.85, 0.97)
    payload = json.loads(paths[1].read_text())
    payload["pairing"]["all_parameters_bitwise_equal"] = False
    paths[1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="parameter mismatch"):
        summarize(paths)
