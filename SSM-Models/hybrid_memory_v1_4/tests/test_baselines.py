"""Registry, fail-closed, and static product-key baseline contracts."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hybrid_memory_v1_4.baselines as baselines_module
from delta_product_reference import DeltaProductReferenceModel
from hybrid_memory_v1_4.baselines import (
    BASELINE_NAMES,
    BASELINE_REGISTRY,
    BaselineAvailability,
    BaselineSpec,
    BaselineUnavailableError,
    ProductKeyMemory,
    baseline_availability,
    baseline_metadata,
    build_baseline,
)
from hybrid_memory_v1_4.fla_adapter import DeltaRuleAdapter

EXPECTED_NAMES = (
    "delta_product_reference",
    "fla_delta_semantic",
    "fla_delta_fused",
    "mamba2_official",
    "product_key_static",
)


def test_registry_is_explicit_frozen_and_uses_repository_implementations() -> None:
    assert BASELINE_NAMES == EXPECTED_NAMES
    assert tuple(BASELINE_REGISTRY) == EXPECTED_NAMES
    assert all(isinstance(spec, BaselineSpec) for spec in BASELINE_REGISTRY.values())
    assert all(spec.name == name for name, spec in BASELINE_REGISTRY.items())

    with pytest.raises(FrozenInstanceError):
        BASELINE_REGISTRY["product_key_static"].official = True  # type: ignore[misc]

    delta = build_baseline(
        "delta_product_reference",
        device="cpu",
        dtype=torch.float64,
        input_vocab_size=17,
        output_size=13,
        hidden_size=8,
        num_heads=2,
        num_householder=2,
        intermediate_size=16,
    )
    semantic = build_baseline(
        "fla_delta_semantic",
        device="cpu",
        dtype=torch.float64,
        heads=2,
        key_dimension=4,
        value_dimension=3,
    )
    assert isinstance(delta, DeltaProductReferenceModel)
    assert isinstance(semantic, DeltaRuleAdapter)
    assert semantic.backend == "semantic_recurrent"
    with pytest.raises(TypeError, match="fixes backend"):
        build_baseline(
            "fla_delta_semantic",
            heads=1,
            key_dimension=2,
            value_dimension=2,
            backend="fla_chunk",
        )


def test_availability_is_structured_and_unknown_names_fail() -> None:
    report = baseline_availability("product_key_static", "cpu", torch.float64)
    assert isinstance(report, BaselineAvailability)
    assert report.available and bool(report)
    assert report.name == "product_key_static"
    assert report.device == "cpu"
    assert report.dtype == "float64"
    assert report.as_dict()["reasons"] == ()
    assert "available" in report.summary()
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_availability("automatic", "cpu", torch.float32)


@pytest.mark.parametrize("name", ("fla_delta_fused", "mamba2_official"))
def test_official_baselines_fail_closed_without_cpu_substitution(name: str) -> None:
    report = baseline_availability(name, "cpu", torch.float32)
    assert not report
    assert report.reasons
    with pytest.raises(BaselineUnavailableError) as raised:
        build_baseline(name, device="cpu", dtype=torch.float32)
    assert isinstance(raised.value, RuntimeError)
    assert raised.value.availability == report
    assert raised.value.as_dict() == report.as_dict()
    assert name in str(raised.value)
    assert "unavailable" in str(raised.value)


def test_missing_mamba_optional_dependency_is_reported_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        baselines_module, "_device_and_dtype_reasons", lambda device, dtype: []
    )

    def missing_components() -> object:
        raise ImportError("simulated missing mamba_ssm")

    monkeypatch.setattr(baselines_module, "_mamba2_components", missing_components)
    report = baseline_availability("mamba2_official", "cuda", torch.float16)
    assert not report
    assert report.reasons == ("official fused mamba_ssm Mamba-2 is unavailable",)
    assert report.detail == "ImportError: simulated missing mamba_ssm"
    with pytest.raises(BaselineUnavailableError) as raised:
        build_baseline("mamba2_official", device="cuda", dtype=torch.float16)
    assert raised.value.availability == report


def _brute_product_key_topk(
    memory: ProductKeyMemory, query: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first, second = query.split(memory.subkey_dim, dim=-1)
    first_scores = first @ memory.subkeys[0].T
    second_scores = second @ memory.subkeys[1].T
    all_scores = (first_scores.unsqueeze(-1) + second_scores.unsqueeze(-2)).flatten(
        start_dim=-2
    )
    scores, indices = all_scores.topk(memory.top_k, dim=-1)
    output = torch.sum(
        torch.softmax(scores, dim=-1).unsqueeze(-1) * memory.values[indices],
        dim=-2,
    )
    return output, scores, indices


@pytest.mark.parametrize("leading_shape", [(), (5,), (2, 3), (2, 1, 4)])
def test_product_key_cartesian_topk_matches_exhaustive_lookup(
    leading_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(20260824)
    memory = ProductKeyMemory(6, 4, num_subkeys=5, top_k=4).double()
    query = torch.randn(*leading_shape, 6, dtype=torch.float64)

    expected, expected_scores, expected_indices = _brute_product_key_topk(memory, query)
    scores, indices = memory.topk(query)
    actual = memory(query)
    assert actual.shape == (*leading_shape, 4)
    assert scores.shape == (*leading_shape, 4)
    assert indices.shape == (*leading_shape, 4)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(indices, expected_indices)


def test_product_key_gradients_reach_query_both_subkeys_and_values() -> None:
    torch.manual_seed(20260825)
    memory = ProductKeyMemory(8, 3, num_subkeys=6, top_k=4).double()
    query = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    output = memory(query)
    weights = torch.linspace(0.2, 1.1, output.numel(), dtype=torch.float64)
    (output * weights.reshape_as(output)).sum().backward()

    assert query.grad is not None and torch.count_nonzero(query.grad) > 0
    for table in memory.subkeys:
        assert table.grad is not None
        assert torch.count_nonzero(table.grad) > 0
        assert torch.isfinite(table.grad).all()
    assert memory.values.grad is not None
    assert torch.count_nonzero(memory.values.grad) > 0
    assert torch.isfinite(memory.values.grad).all()


def test_product_key_is_explicitly_static_parameter_memory_with_zero_state() -> None:
    memory = ProductKeyMemory(8, 3, num_subkeys=4, top_k=3)
    assert memory.memory_kind == "static_parameter_memory"
    assert not memory.supports_episode_writes
    assert memory.state_scalars == 0
    assert memory.state_bytes(torch.float16, batch_size=999) == 0
    assert "cannot store associations introduced" in ProductKeyMemory.__doc__

    metadata = baseline_metadata(
        "product_key_static", memory, batch_size=999, dtype=torch.float16
    )
    expected_parameters = 2 * 4 * 4 + 4**2 * 3
    assert metadata["parameter_count"] == expected_parameters
    assert metadata["fixed_cache_bytes"] == 0
    assert metadata["variable_cache_bytes"] == 0
    assert metadata["total_cache_bytes"] == 0
    assert "no per-episode writes" in str(metadata["claim_boundary"])


def test_common_metadata_preserves_implementation_and_claim_boundaries() -> None:
    for name, spec in BASELINE_REGISTRY.items():
        metadata = baseline_metadata(name)
        assert metadata["name"] == name
        assert metadata["official"] is spec.official
        assert metadata["fused"] is spec.fused
        assert metadata["reference"] is spec.reference
        assert metadata["claim_boundary"] == spec.claim_boundary
        assert metadata["claim_boundary"]
        assert metadata["parameter_count"] is None

    delta = build_baseline(
        "delta_product_reference",
        input_vocab_size=11,
        output_size=7,
        hidden_size=8,
        num_heads=2,
        num_householder=2,
        intermediate_size=16,
    )
    delta_metadata = baseline_metadata(
        "delta_product_reference", delta, batch_size=3, dtype=torch.float32
    )
    assert delta_metadata["parameter_count"] == sum(
        parameter.numel() for parameter in delta.parameters()
    )
    assert delta_metadata["fixed_cache_bytes"] == 3 * 2 * 4 * 4 * 4
    assert delta_metadata["variable_cache_bytes"] == 0

    assert BASELINE_REGISTRY["fla_delta_fused"].official
    assert BASELINE_REGISTRY["fla_delta_fused"].fused
    assert not BASELINE_REGISTRY["fla_delta_fused"].reference
    assert BASELINE_REGISTRY["mamba2_official"].official
    assert not BASELINE_REGISTRY["delta_product_reference"].official
    assert BASELINE_REGISTRY["delta_product_reference"].reference


def test_guarded_cuda_fla_registry_run() -> None:
    report = baseline_availability("fla_delta_fused", "cuda", torch.float16)
    if not report:
        pytest.skip(report.summary())
    adapter = build_baseline(
        "fla_delta_fused",
        device="cuda",
        dtype=torch.float16,
        heads=1,
        key_dimension=32,
        value_dimension=32,
        chunk_size=16,
    )
    assert isinstance(adapter, DeltaRuleAdapter)
    assert adapter.backend == "fla_chunk"
    q = torch.randn(1, 16, 1, 32, device="cuda", dtype=torch.float16)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn_like(q)
    beta = torch.sigmoid(torch.randn(1, 16, 1, device="cuda", dtype=torch.float16))
    output, state = adapter(q, k, v, beta, output_final_state=True)
    assert output.shape == v.shape
    assert state is not None and state.shape == (1, 1, 32, 32)


def test_guarded_cuda_official_mamba2_registry_run() -> None:
    report = baseline_availability("mamba2_official", "cuda", torch.float16)
    if not report:
        pytest.skip(report.summary())
    model = build_baseline(
        "mamba2_official",
        device="cuda",
        dtype=torch.float16,
        vocab_size=32,
        d_model=64,
        num_layers=1,
        d_state=16,
        headdim=32,
    )
    tokens = torch.randint(0, 32, (1, 16), device="cuda")
    output = model(tokens)
    assert output["logits"].shape == (1, 16, 32)
