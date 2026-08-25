"""Fail-closed contracts for the G15A operational cohort."""

from __future__ import annotations

import torch

from hybrid_memory_v1_4.g15a_spin_dirac_cohort import (
    ARM_SPECS,
    EVALUATION_LENGTHS,
    QUALITY_SEEDS,
    _adjudicate,
    _inner_conjugation_residual,
    _model_config,
    _oracle_semantic_ladder,
    quality_config,
)
from hybrid_memory_v1_4.model import HybridMemoryLM, parameter_count


def test_frozen_g15a_arm_set_and_parameter_tensors_are_exactly_matched() -> None:
    assert ARM_SPECS == {
        "I": ("identity", "identity"),
        "I+C": ("identity", "clifford"),
        "C": ("commuting_so2", "clifford"),
        "S": ("spin8", "clifford"),
    }
    config = quality_config()
    models = {arm: HybridMemoryLM(_model_config(arm, config)) for arm in ARM_SPECS}
    counts = {arm: parameter_count(model) for arm, model in models.items()}
    shapes = {
        arm: {
            name: tuple(parameter.shape) for name, parameter in model.named_parameters()
        }
        for arm, model in models.items()
    }
    assert len(set(counts.values())) == 1
    assert all(value == shapes["I"] for value in shapes.values())
    assert all(
        model.config.spin_dirac_gate_mode == "equivariant_scalar"
        for model in models.values()
    )
    assert all(model.config.spin_dirac_bound_values for model in models.values())


def _fake_seed_report(
    *, spin_symmetry: float, comparator_symmetry: float, spin_no_symmetry: float
) -> dict[str, object]:
    arms = {}
    for arm in ARM_SPECS:
        symmetry = spin_symmetry if arm == "S" else comparator_symmetry
        no_symmetry = spin_no_symmetry if arm == "S" else 1.0
        arms[arm] = {
            "symmetry": {
                "evaluation": {
                    str(length): {"accuracy": symmetry} for length in EVALUATION_LENGTHS
                },
                "inner_conjugation_max_abs_residual_float64": 1e-12,
            },
            "no_symmetry": {
                "evaluation": {
                    str(length): {"accuracy": no_symmetry}
                    for length in EVALUATION_LENGTHS
                }
            },
        }
    return {"seed": 1, "arms": arms}


def test_adjudicator_requires_per_seed_margin_and_no_symmetry_noninferiority() -> None:
    passing = _adjudicate(
        [
            _fake_seed_report(
                spin_symmetry=1.0, comparator_symmetry=0.9, spin_no_symmetry=0.99
            )
        ]
    )
    assert passing["passed"] is True
    insufficient_margin = _adjudicate(
        [
            _fake_seed_report(
                spin_symmetry=0.91, comparator_symmetry=0.9, spin_no_symmetry=1.0
            )
        ]
    )
    assert insufficient_margin["passed"] is False
    retrieval_regression = _adjudicate(
        [
            _fake_seed_report(
                spin_symmetry=1.0, comparator_symmetry=0.9, spin_no_symmetry=0.98
            )
        ]
    )
    assert retrieval_regression["passed"] is False


def test_quality_protocol_and_oracle_ladder_are_frozen() -> None:
    config = quality_config()
    assert config.seeds == QUALITY_SEEDS == (2131, 2137, 2141)
    assert config.training_updates == 300
    assert config.training_batch_size == 16
    assert config.dtype == "float32"
    ladder = _oracle_semantic_ladder()
    assert ladder["passed"] is True
    assert _inner_conjugation_residual("S", torch.ones(2), seed=2131) <= 1e-9
