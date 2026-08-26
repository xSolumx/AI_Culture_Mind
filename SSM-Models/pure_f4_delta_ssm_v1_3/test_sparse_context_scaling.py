from __future__ import annotations

from .benchmark_sparse_context_scaling import _scaling_verdict


def _report(length: int, candidate_ms: float = 10.0) -> dict[str, object]:
    batch = 4096 // length

    def row(
        milliseconds: float,
        peak: int,
        parameters: int,
        model_d_model: int,
    ) -> dict[str, object]:
        return {
            "timing": {"median_ms": milliseconds},
            "maximum_peak_allocated_bytes": peak,
            "parameters": [parameters],
            "model_d_model": [model_d_model],
            "any_dense_sequence_action": False,
        }

    return {
        "config": {
            "variants": [
                "e6_primitive_dead",
                "e6_primitive_event",
                "mamba2_official",
            ],
            "batch_size": batch,
            "sequence_length": length,
            "d_model": 126,
            "mamba_d_model": 140,
        },
        "git": {"revision": "abc", "dirty": False},
        "summary": {
            "e6_primitive_dead": row(10.0, 100, 679_866, 126),
            "e6_primitive_event": row(candidate_ms, 100, 679_866, 126),
            "mamba2_official": row(9.0, 90, 682_160, 140),
        },
    }


def test_scaling_verdict_passes_only_when_every_context_passes() -> None:
    reports = [_report(length) for length in (128, 256, 512, 1024, 2048, 4096)]
    verdict = _scaling_verdict(reports)
    assert verdict["fixed_token_scaling_pass"] is True
    assert verdict["mamba_competitive_all_contexts_pass"] is True

    reports[-1]["summary"]["e6_primitive_event"]["timing"]["median_ms"] = 21.0
    verdict = _scaling_verdict(reports)
    assert verdict["fixed_token_scaling_pass"] is False
    assert verdict["mamba_competitive_all_contexts_pass"] is False


def test_scaling_verdict_fails_on_dirty_or_parameter_mismatch() -> None:
    reports = [_report(128), _report(256)]
    reports[0]["git"]["dirty"] = True
    assert _scaling_verdict(reports)["fixed_token_scaling_pass"] is False

    reports[0]["git"]["dirty"] = False
    reports[1]["summary"]["mamba2_official"]["parameters"] = [700_000]
    verdict = _scaling_verdict(reports)
    assert verdict["parameter_match_pass"] is False
    assert verdict["fixed_token_scaling_pass"] is False
