from __future__ import annotations

from .external_baselines import BASELINES, GITHUB_SOURCES, SCHEMA_VERSION


def test_external_baseline_manifest_is_pinned_and_claim_separated() -> None:
    assert SCHEMA_VERSION == 1
    ids = [baseline["id"] for baseline in BASELINES]
    assert len(ids) == len(set(ids))
    assert {
        "falcon_mamba_7b",
        "mamba3_siso_187m",
        "mamba3_mimo_187m",
        "gka_primed_qwen3_8b",
        "gdn_primed_qwen3_8b",
        "jamba_v0_1",
        "jamba2_3b",
    } == set(ids)
    for baseline in BASELINES:
        assert len(baseline["revision"]) == 40
        assert baseline["weight_bytes"] > 0
        assert baseline["comparison_tiers"]
        assert baseline["boundary"]
    for source in GITHUB_SOURCES.values():
        assert source["url"].startswith("https://github.com/")
        assert len(source["revision"]) == 40
