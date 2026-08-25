"""Contracts for the prospectively frozen G15A conditional controls."""

from __future__ import annotations

from hybrid_memory_v1_4.g15a_conditional_controls import _adjudicate
from hybrid_memory_v1_4.g15a_spin_dirac_cohort import EVALUATION_LENGTHS


def _evaluation(accuracy: float) -> dict[str, dict[str, float]]:
    return {str(length): {"accuracy": accuracy} for length in EVALUATION_LENGTHS}


def _reports(
    *,
    identity_accuracy: float,
    broken_accuracy: float,
    broken_inner_residual: float = 1e-12,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    primary = {
        "seed_reports": [
            {
                "seed": 1,
                "arms": {
                    "S": {
                        "symmetry": {"evaluation": _evaluation(1.0)},
                        "no_symmetry": {
                            "evaluation": _evaluation(1.0),
                            "training_schedule_sha256": "paired",
                        },
                    }
                },
            }
        ]
    }
    controls = []
    for arm, accuracy, residual in (
        ("S+identity-read", identity_accuracy, 1e-12),
        ("S-broken", broken_accuracy, broken_inner_residual),
    ):
        controls.append(
            (
                arm,
                {
                    "symmetry": {
                        "evaluation": _evaluation(accuracy),
                        "inner_conjugation_max_abs_residual_float64": residual,
                    },
                    "no_symmetry": {
                        "evaluation": _evaluation(1.0),
                        "training_schedule_sha256": "paired",
                    },
                },
            )
        )
    return primary, [{"seed": 1, "arms": dict(controls)}]


def test_conditional_adjudication_requires_per_seed_attribution_margins() -> None:
    primary, controls = _reports(identity_accuracy=0.97, broken_accuracy=0.97)
    passing = _adjudicate(primary, controls)
    assert passing["clifford_read_contribution_supported"] is True
    assert passing["shared_triality_coupling_contribution_supported"] is True
    primary, controls = _reports(identity_accuracy=1.0, broken_accuracy=1.0)
    tied = _adjudicate(primary, controls)
    assert tied["clifford_read_contribution_supported"] is False
    assert tied["shared_triality_coupling_contribution_supported"] is False
    assert "triality-specific attribution fails" in tied["decision"]


def test_broken_coupling_residual_is_diagnostic_not_an_integrity_gate() -> None:
    primary, controls = _reports(
        identity_accuracy=1.0,
        broken_accuracy=0.2,
        broken_inner_residual=0.08,
    )
    result = _adjudicate(primary, controls)
    row = result["per_seed"][0]
    assert row["checks"]["conditional_inner_replays_at_most_1e_9_diagnostic"] is False
    assert result["integrity_passed"] is True
    assert result["shared_triality_coupling_contribution_supported"] is True
