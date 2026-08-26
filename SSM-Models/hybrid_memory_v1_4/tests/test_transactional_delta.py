"""Prospective Phase-0 contracts for G15B-T transactional delta memory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_memory_v1_4.g15be_phase0_qualification import qualify as qualify_g15be
from hybrid_memory_v1_4.g15bt_phase0_qualification import qualify as qualify_g15bt
from hybrid_memory_v1_4.model import (
    CausalDepthwiseConv1d,
    GatedDeltaState,
    HybridMemoryConfig,
    HybridMemoryLM,
    parameter_count,
)
from hybrid_memory_v1_4.transactional_delta import (
    TransactionalDeltaConfig,
    TransactionalDeltaMemory,
)


def _model_config(controller_mode: str = "history") -> HybridMemoryConfig:
    return HybridMemoryConfig(
        vocab_size=73,
        model_dim=16,
        layer_plan=("transactional_delta",),
        gated_delta_heads=2,
        gated_delta_key_dim=5,
        gated_delta_value_dim=6,
        transactional_controller_mode=controller_mode,  # type: ignore[arg-type]
        use_local_conv=True,
        conv_kernel=4,
        expansion=2,
        dropout=0.0,
        tie_embeddings=False,
    )


def _assert_delta_states_close(
    actual: tuple[object, ...],
    expected: tuple[object, ...],
    *,
    rtol: float = 1e-11,
    atol: float = 1e-11,
) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert isinstance(left, GatedDeltaState)
        assert isinstance(right, GatedDeltaState)
        torch.testing.assert_close(left.memory, right.memory, rtol=rtol, atol=atol)
        torch.testing.assert_close(
            left.convolution, right.convolution, rtol=rtol, atol=atol
        )


def test_config_rejects_invalid_controller_and_requires_history_cache() -> None:
    with pytest.raises(ValueError, match="controller_mode"):
        TransactionalDeltaConfig(8, controller_mode="future")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initial_commit_strength"):
        TransactionalDeltaConfig(8, initial_commit_strength=1.0)
    with pytest.raises(ValueError, match="initial_erase_strength"):
        TransactionalDeltaConfig(8, initial_erase_strength=0.0)
    with pytest.raises(ValueError, match="effective_edit_gate_mode"):
        TransactionalDeltaConfig(
            8,
            effective_edit_gate_mode="ratio",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="local convolution"):
        HybridMemoryConfig(
            model_dim=8,
            layer_plan=("transactional_delta",),
            use_local_conv=False,
        )
    with pytest.raises(ValueError, match="kernel"):
        HybridMemoryConfig(
            model_dim=8,
            layer_plan=("transactional_delta",),
            use_local_conv=True,
            conv_kernel=1,
        )


def test_full_and_history_arms_are_exactly_parameter_and_state_matched() -> None:
    torch.manual_seed(2381)
    full = HybridMemoryLM(_model_config("full"))
    torch.manual_seed(2381)
    history = HybridMemoryLM(_model_config("history"))
    assert parameter_count(full) == parameter_count(history)
    assert sum(p.numel() for p in full.parameters() if p.requires_grad) == sum(
        p.numel() for p in history.parameters() if p.requires_grad
    )
    assert {
        name: tuple(parameter.shape) for name, parameter in full.named_parameters()
    } == {
        name: tuple(parameter.shape) for name, parameter in history.named_parameters()
    }
    assert full.state_capacity_bytes(3, torch.float32) == history.state_capacity_bytes(
        3, torch.float32
    )
    assert full.state_dict().keys() == history.state_dict().keys()


def test_product_and_additive_effective_edits_are_matched_at_initialization() -> None:
    torch.manual_seed(2481)
    product = TransactionalDeltaMemory(
        TransactionalDeltaConfig(
            12,
            heads=3,
            key_dim=4,
            value_dim=4,
            controller_mode="full",
            effective_edit_gate_mode="product",
        )
    )
    torch.manual_seed(2481)
    additive = TransactionalDeltaMemory(
        TransactionalDeltaConfig(
            12,
            heads=3,
            key_dim=4,
            value_dim=4,
            controller_mode="full",
            effective_edit_gate_mode="logit_additive",
        )
    )
    assert product.state_dict().keys() == additive.state_dict().keys()
    for name, tensor in product.state_dict().items():
        assert torch.equal(tensor, additive.state_dict()[name]), name

    inputs = torch.randn(2, 7, 12)
    product_controls = product._controls(inputs, inputs)
    additive_controls = additive._controls(inputs, inputs)
    torch.testing.assert_close(
        product_controls[3] * product_controls[4],
        additive_controls[4],
        rtol=0.0,
        atol=2e-8,
    )
    torch.testing.assert_close(
        product_controls[3] * product_controls[5],
        additive_controls[5],
        rtol=0.0,
        atol=2e-8,
    )


def test_product_and_residual_delta_are_functionally_matched_at_initialization() -> None:
    torch.manual_seed(2481)
    product = TransactionalDeltaMemory(
        TransactionalDeltaConfig(
            12,
            heads=3,
            key_dim=4,
            value_dim=4,
            controller_mode="full",
            effective_edit_gate_mode="product",
        )
    ).double()
    torch.manual_seed(2481)
    residual = TransactionalDeltaMemory(
        TransactionalDeltaConfig(
            12,
            heads=3,
            key_dim=4,
            value_dim=4,
            controller_mode="full",
            effective_edit_gate_mode="residual_delta",
        )
    ).double()
    assert product.state_dict().keys() == residual.state_dict().keys()
    for name, tensor in product.state_dict().items():
        assert torch.equal(tensor, residual.state_dict()[name]), name

    inputs = torch.randn(2, 7, 12, dtype=torch.float64)
    product_controls = product._controls(inputs, inputs)
    residual_controls = residual._controls(inputs, inputs)
    product_erase = product_controls[3] * product_controls[4]
    product_write = product_controls[3] * product_controls[5]
    torch.testing.assert_close(
        product_erase.expand_as(residual_controls[4]),
        residual_controls[4],
        rtol=0.0,
        atol=1e-8,
    )
    torch.testing.assert_close(
        product_write,
        residual_controls[5],
        rtol=0.0,
        atol=1e-8,
    )
    product_output, product_state = product(
        inputs, inputs, scan_mode="recurrent"
    )
    residual_output, residual_state = residual(
        inputs, inputs, scan_mode="recurrent"
    )
    torch.testing.assert_close(product_output, residual_output, rtol=0.0, atol=1e-8)
    torch.testing.assert_close(product_state, residual_state, rtol=0.0, atol=1e-8)


@pytest.mark.parametrize(
    "gate_mode", ("product", "logit_additive", "residual_delta")
)
def test_effective_edit_gate_modes_remain_bounded_and_contracting(
    gate_mode: str,
) -> None:
    torch.manual_seed(2482)
    layer = TransactionalDeltaMemory(
        TransactionalDeltaConfig(
            12,
            heads=3,
            key_dim=4,
            value_dim=4,
            controller_mode="full",
            effective_edit_gate_mode=gate_mode,  # type: ignore[arg-type]
        )
    ).double()
    inputs = torch.randn(2, 11, 12, dtype=torch.float64)
    controls = layer._controls(inputs, inputs)
    transition, injection = layer._transitions(*controls[1:], None)
    assert torch.isfinite(transition).all()
    assert torch.isfinite(injection).all()
    spectral_maximum = torch.linalg.matrix_norm(transition, ord=2).max().detach()
    assert float(spectral_maximum) <= 1.0 + 1e-12
    _, _, diagnostics = layer(
        inputs, inputs, scan_mode="recurrent", return_diagnostics=True
    )
    assert diagnostics["effective_edit_gate_mode"] == gate_mode
    for name in ("effective_erase_strength", "effective_write_strength"):
        gate = diagnostics[name]
        assert isinstance(gate, torch.Tensor)
        assert bool(((gate > 0.0) & (gate < 1.0)).all())


def test_residual_delta_recurrent_and_parallel_execution_agree() -> None:
    torch.manual_seed(2484)
    layer = TransactionalDeltaMemory(
        TransactionalDeltaConfig(
            12,
            heads=3,
            key_dim=3,
            value_dim=5,
            controller_mode="full",
            effective_edit_gate_mode="residual_delta",
        )
    ).double()
    inputs = torch.randn(2, 13, 12, dtype=torch.float64)
    initial = torch.randn(2, 3, 3, 5, dtype=torch.float64)
    valid_mask = torch.tensor(
        [
            [
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
            ],
            [
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
                False,
                True,
            ],
        ]
    )
    recurrent_output, recurrent_state, recurrent_diagnostics = layer(
        inputs,
        inputs,
        initial,
        valid_mask=valid_mask,
        scan_mode="recurrent",
        return_diagnostics=True,
    )
    parallel_output, parallel_state, parallel_diagnostics = layer(
        inputs,
        inputs,
        initial,
        valid_mask=valid_mask,
        scan_mode="parallel",
        return_diagnostics=True,
    )
    torch.testing.assert_close(
        recurrent_output, parallel_output, rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(
        recurrent_state, parallel_state, rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(
        recurrent_diagnostics["state_norm"],
        parallel_diagnostics["state_norm"],
        rtol=1e-11,
        atol=1e-11,
    )
    strength = recurrent_diagnostics["residual_delta_strength"]
    assert isinstance(strength, torch.Tensor)
    torch.testing.assert_close(
        strength,
        recurrent_diagnostics["effective_erase_strength"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        strength,
        recurrent_diagnostics["effective_write_strength"],
        rtol=0.0,
        atol=0.0,
    )


def test_strict_history_is_current_invariant_and_prior_sensitive() -> None:
    torch.manual_seed(2382)
    convolution = CausalDepthwiseConv1d(6, 4).double()
    inputs = torch.randn(2, 8, 6, dtype=torch.float64)
    full, history, _ = convolution.full_and_strict_history(inputs)

    current_changed = inputs.clone()
    current_changed[:, 5] += torch.randn(2, 6, dtype=torch.float64) * 7.0
    changed_full, changed_history, _ = convolution.full_and_strict_history(
        current_changed
    )
    assert torch.count_nonzero(changed_full[:, 5] - full[:, 5]) > 0
    assert torch.equal(changed_history[:, 5], history[:, 5])

    prior_changed = inputs.clone()
    prior_changed[:, 4] += torch.randn(2, 6, dtype=torch.float64) * 7.0
    _, prior_history, _ = convolution.full_and_strict_history(prior_changed)
    assert torch.count_nonzero(prior_history[:, 5] - history[:, 5]) > 0


def test_history_edit_controls_and_affine_update_are_current_invariant() -> None:
    torch.manual_seed(2383)
    layer = TransactionalDeltaMemory(
        TransactionalDeltaConfig(
            12, heads=2, key_dim=4, value_dim=5, controller_mode="history"
        )
    ).double()
    full = torch.randn(2, 1, 12, dtype=torch.float64)
    changed_full = full + torch.randn_like(full) * 5.0
    history = torch.randn(2, 1, 12, dtype=torch.float64)
    controls = layer._controls(full, history)
    changed_controls = layer._controls(changed_full, history)
    assert torch.count_nonzero(controls[0] - changed_controls[0]) > 0
    for original, changed in zip(controls[1:], changed_controls[1:], strict=True):
        assert torch.equal(original, changed)

    transition, injection = layer._transitions(*controls[1:], None)
    changed_transition, changed_injection = layer._transitions(
        *changed_controls[1:], None
    )
    assert torch.equal(transition, changed_transition)
    assert torch.equal(injection, changed_injection)
    initial = torch.randn(2, 2, 4, 5, dtype=torch.float64)
    _, final = layer._recurrent_states(transition, injection, initial)
    _, changed_final = layer._recurrent_states(
        changed_transition, changed_injection, initial
    )
    assert torch.equal(final, changed_final)


def test_symmetric_erase_overwrites_and_transitions_are_contractions() -> None:
    layer = TransactionalDeltaMemory(
        TransactionalDeltaConfig(4, heads=1, key_dim=2, value_dim=2)
    ).double()
    key = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float64)
    value = torch.tensor([[[[2.0, -3.0]], [[7.0, 11.0]]]], dtype=torch.float64)
    scalar = torch.ones(1, 2, 1, 1, dtype=torch.float64)
    write = torch.ones(1, 2, 1, 2, dtype=torch.float64)
    retention = torch.ones(1, 2, 1, 2, dtype=torch.float64)
    transition, injection = layer._transitions(
        key, value, scalar, scalar, write, retention, None
    )
    states, final = layer._recurrent_states(
        transition, injection, torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    )
    torch.testing.assert_close(states[0, 0, 0, 0], value[0, 0, 0])
    torch.testing.assert_close(final[0, 0, 0], value[0, 1, 0])

    torch.manual_seed(2384)
    random_key = torch.nn.functional.normalize(
        torch.randn(3, 9, 1, 2, dtype=torch.float64), dim=-1
    )
    random_value = torch.randn(3, 9, 1, 2, dtype=torch.float64)
    commit = torch.rand(3, 9, 1, 1, dtype=torch.float64)
    erase = torch.rand(3, 9, 1, 1, dtype=torch.float64)
    random_write = torch.rand(3, 9, 1, 2, dtype=torch.float64)
    random_retention = torch.rand(3, 9, 1, 2, dtype=torch.float64)
    random_transition, random_injection = layer._transitions(
        random_key,
        random_value,
        commit,
        erase,
        random_write,
        random_retention,
        None,
    )
    assert torch.isfinite(random_transition).all()
    assert torch.isfinite(random_injection).all()
    spectral_norm = torch.linalg.matrix_norm(random_transition, ord=2)
    assert float(spectral_norm.max()) <= 1.0 + 1e-12


def test_memory_fp64_parallel_recurrent_and_arbitrary_chunks_match() -> None:
    torch.manual_seed(2385)
    layer = TransactionalDeltaMemory(
        TransactionalDeltaConfig(
            16, heads=2, key_dim=5, value_dim=6, controller_mode="history"
        )
    ).double()
    full = torch.randn(2, 19, 16, dtype=torch.float64)
    history = torch.randn(2, 19, 16, dtype=torch.float64)
    recurrent, recurrent_state = layer(full, history, scan_mode="recurrent")
    parallel, parallel_state = layer(full, history, scan_mode="parallel")
    pieces = []
    state = None
    for start, stop in ((0, 2), (2, 9), (9, 14), (14, 19)):
        output, state = layer(
            full[:, start:stop],
            history[:, start:stop],
            state,
            scan_mode="parallel",
        )
        pieces.append(output)
    torch.testing.assert_close(recurrent, parallel, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(recurrent_state, parallel_state, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        recurrent, torch.cat(pieces, dim=1), rtol=1e-11, atol=1e-11
    )
    assert state is not None
    torch.testing.assert_close(recurrent_state, state, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("controller_mode", ("full", "history"))
def test_full_lm_fp64_scan_chunk_step_mask_and_diagnostics(
    controller_mode: str,
) -> None:
    torch.manual_seed(2386)
    model = HybridMemoryLM(_model_config(controller_mode)).double().eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 13))
    recurrent = model(tokens, delta_scan_mode="recurrent", return_diagnostics=True)
    parallel = model(tokens, delta_scan_mode="parallel")
    torch.testing.assert_close(
        parallel["logits"], recurrent["logits"], rtol=3e-10, atol=3e-11
    )
    _assert_delta_states_close(parallel["states"], recurrent["states"])
    diagnostics = recurrent["diagnostics"][0]
    assert diagnostics["kind"] == "transactional_delta"
    assert diagnostics["controller_mode"] == controller_mode

    pieces = []
    state = None
    for start, stop in ((0, 3), (3, 4), (4, 9), (9, 13)):
        chunk = model(tokens[:, start:stop], state, delta_scan_mode="parallel")
        pieces.append(chunk["logits"])
        state = chunk["states"]
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), recurrent["logits"], rtol=3e-10, atol=3e-11
    )
    assert state is not None
    _assert_delta_states_close(state, recurrent["states"])

    step_logits = []
    state = None
    for position in range(tokens.shape[1]):
        logits, state = model.step(tokens[:, position], state)
        step_logits.append(logits[:, None])
    torch.testing.assert_close(
        torch.cat(step_logits, dim=1), recurrent["logits"], rtol=3e-10, atol=3e-11
    )
    assert state is not None
    _assert_delta_states_close(state, recurrent["states"])

    masked_tokens = tokens[:, :6]
    valid_mask = torch.tensor(
        [[True, True, False, True, False, True], [True, False, True, True, True, False]]
    )
    masked_full = model(
        masked_tokens, valid_mask=valid_mask, delta_scan_mode="recurrent"
    )
    masked_steps = None
    for position in range(masked_tokens.shape[1]):
        _, masked_steps = model.step(
            masked_tokens[:, position], masked_steps, valid_mask=valid_mask[:, position]
        )
    assert masked_steps is not None
    _assert_delta_states_close(masked_steps, masked_full["states"])

    prefix = model(tokens[:, :4], delta_scan_mode="recurrent")
    frozen = model(
        tokens[:, 4:5],
        prefix["states"],
        valid_mask=torch.zeros(2, 1, dtype=torch.bool),
        delta_scan_mode="recurrent",
    )
    _assert_delta_states_close(frozen["states"], prefix["states"], rtol=0.0, atol=0.0)


def test_real_lm_loss_reaches_every_declared_transactional_path() -> None:
    torch.manual_seed(2387)
    model = HybridMemoryLM(_model_config("history")).double().train()
    tokens = torch.randint(0, model.config.vocab_size, (3, 17))
    output = model(tokens, delta_scan_mode="parallel")
    state = output["states"][0]
    assert isinstance(state, GatedDeltaState)
    loss = output["logits"].square().mean() + state.memory.square().mean()
    loss.backward()
    mixer = model.blocks[0].mixer
    assert isinstance(mixer, TransactionalDeltaMemory)
    parameters = dict(model.named_parameters())
    for name in (
        "embedding.weight",
        "blocks.0.local_conv.conv.weight",
        "blocks.0.mixer.query_projection.weight",
        "blocks.0.mixer.key_projection.weight",
        "blocks.0.mixer.value_projection.weight",
        "blocks.0.mixer.commit_projection.weight",
        "blocks.0.mixer.erase_projection.weight",
        "blocks.0.mixer.write_projection.weight",
        "blocks.0.mixer.decay_projection.weight",
        "blocks.0.mixer.output_gate.weight",
        "blocks.0.mixer.output_projection.weight",
        "lm_head.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name

    report = model.state_byte_report(output["states"])
    assert report["layers"][0]["kind"] == "transactional_delta"
    assert report["layers"][0]["capacity_components"]["memory"] > 0


def test_phase0_qualification_harness_passes_on_semantic_cpu_path() -> None:
    report = qualify_g15bt(torch.device("cpu"))
    assert report["passed"] is True
    assert report["matched_arms"]["passed"] is True
    assert report["causality_and_contraction"]["passed"] is True
    assert report["fp64_execution"]["passed"] is True
    assert report["fp32_execution"]["passed"] is True
    assert report["gradient_reach"]["passed"] is True


def test_effective_edit_phase0_harness_passes_on_semantic_cpu_path() -> None:
    report = qualify_g15be(torch.device("cpu"))
    assert report["passed"] is True
    assert report["matched_arms"]["passed"] is True
    assert report["bounded_update"]["passed"] is True
    assert all(
        row["passed"]
        for arm in report["execution"].values()
        for row in arm.values()
    )
    assert all(row["passed"] for row in report["gradient_reach"].values())
