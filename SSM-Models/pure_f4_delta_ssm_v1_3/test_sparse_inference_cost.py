from __future__ import annotations

from .benchmark_sparse_inference_cost import _verdict


def _arm(milliseconds: float, cache_bytes: int, parameters: int) -> dict[str, object]:
    return {
        "parameters": [parameters],
        "decode": {"median_ms": milliseconds},
        "prefill": {"median_ms": milliseconds * 10},
        "cache_building_prefill": {"median_ms": milliseconds * 12},
        "maximum_cache_bytes": cache_bytes,
    }


def test_inference_verdict_keeps_decode_and_prefill_gates_separate() -> None:
    summary = {
        "e6_primitive_dead": _arm(1.0, 100, 679_866),
        "e6_primitive_event": _arm(1.1, 100, 679_866),
        "mamba2_official": _arm(1.0, 120, 682_160),
    }
    verdict = _verdict(summary)
    assert verdict["streaming_decode_pass"] is True
    assert verdict["bulk_prefill_pass"] is True

    summary["e6_primitive_event"]["prefill"]["median_ms"] = 20.0
    verdict = _verdict(summary)
    assert verdict["streaming_decode_pass"] is True
    assert verdict["bulk_prefill_pass"] is False


def test_inference_verdict_fails_parameter_mismatch() -> None:
    summary = {
        "e6_primitive_dead": _arm(1.0, 100, 679_866),
        "e6_primitive_event": _arm(1.0, 100, 679_866),
        "mamba2_official": _arm(1.0, 100, 700_000),
    }
    verdict = _verdict(summary)
    assert verdict["parameter_match_pass"] is False
    assert verdict["streaming_decode_pass"] is False


def test_inference_verdict_rejects_nonpositive_parameter_count() -> None:
    summary = {
        "e6_primitive_dead": _arm(1.0, 100, 0),
        "e6_primitive_event": _arm(1.0, 100, 0),
        "mamba2_official": _arm(1.0, 100, 0),
    }
    verdict = _verdict(summary)
    assert verdict["streaming_decode_pass"] is False
    assert verdict["reason"] == (
        "parameter counts are invalid or inconsistent across cycles"
    )
