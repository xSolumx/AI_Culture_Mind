from pathlib import Path

import torch
from torch.nn import functional as F

from mamba3_reference import Mamba3ReferenceLM, mamba3_affine_scan
from validate_quality import validate_report
from vision_benchmark import (
    Mamba2VisionClassifier,
    Mamba3VisionClassifier,
    SpinorVisionClassifier,
    patchify,
)

from spinor_delta_ssm import (
    DepthwiseCausalMix,
    Spin3IsotypicLinear,
    SpinorDeltaLM,
    geometric_product,
    normalized_rotor,
    rotor_coefficients_from_bivector,
    rotor_affine_scan,
    rotor_sandwich,
    rotor_sandwich_fast,
)


def test_geometric_identity_and_rotor_norm():
    x = torch.randn(2, 3, 8)
    one = torch.zeros(2, 3, 8)
    one[..., 0] = 1
    assert torch.allclose(geometric_product(x, one), x)
    rotor = normalized_rotor(torch.randn(2, 3, 4))
    assert torch.allclose(rotor.norm(dim=-1), torch.ones(2, 3), atol=1e-6)
    assert torch.allclose(
        rotor_sandwich(rotor, x).norm(dim=-1), x.norm(dim=-1), atol=2e-5
    )


def test_model_shape_and_gradients():
    torch.manual_seed(0)
    model = SpinorDeltaLM(vocab_size=64, channels=4, layers=2)
    tokens = torch.randint(0, 64, (2, 12))
    logits = model(tokens)
    assert logits.shape == (2, 12, 64)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flatten())
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_isotypic_decoder_shape_and_gradients():
    torch.manual_seed(11)
    model = SpinorDeltaLM(vocab_size=64, channels=6, layers=2, decoder_channels=3)
    tokens = torch.randint(0, 64, (2, 12))
    logits = model(tokens)
    assert logits.shape == (2, 12, 64)
    logits.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_streaming_chunk_parity():
    torch.manual_seed(12)
    model = SpinorDeltaLM(vocab_size=64, channels=4, layers=2, decoder_channels=2)
    model.eval()
    tokens = torch.randint(0, 64, (2, 19))
    full, _ = model(tokens, return_states=True)
    first, states = model(tokens[:, :7], return_states=True)
    second, states = model(tokens[:, 7:], recurrent_states=states, return_states=True)
    streamed = torch.cat((first, second), dim=1)
    assert torch.allclose(full, streamed, atol=3e-5, rtol=3e-5)


def test_local_causal_mixer_identity_and_streaming_parity():
    torch.manual_seed(13)
    mixer = DepthwiseCausalMix(channels=3, kernel_size=4)
    inputs = torch.randn(2, 11, 3, 8)
    mixed, history = mixer(inputs)
    assert torch.allclose(mixed, inputs, atol=1e-6, rtol=1e-6)
    first, history = mixer(inputs[:, :5])
    second, history = mixer(inputs[:, 5:], history)
    assert torch.allclose(
        torch.cat((first, second), dim=1), mixed, atol=2e-6, rtol=2e-6
    )


def test_local_mixer_model_streaming_parity():
    torch.manual_seed(14)
    model = SpinorDeltaLM(
        vocab_size=64, channels=4, layers=2, decoder_channels=2, local_kernel=4
    )
    model.eval()
    tokens = torch.randint(0, 64, (2, 19))
    full, _ = model(tokens, return_states=True)
    first, states = model(tokens[:, :7], return_states=True)
    second, states = model(tokens[:, 7:], recurrent_states=states, return_states=True)
    assert torch.allclose(full, torch.cat((first, second), dim=1), atol=3e-5, rtol=3e-5)


def test_affine_scan_matches_recurrence():
    torch.manual_seed(1)
    batch, length, channels = 2, 17, 3
    decay = torch.rand(batch, length, channels) * 0.4 + 0.5
    rotors = normalized_rotor(torch.randn(batch, length, channels, 4))
    drive = torch.randn(batch, length, channels, 8) * 0.1
    initial = torch.randn(batch, channels, 8)
    fast, final = rotor_affine_scan(decay, rotors, drive, initial)
    state = initial
    slow = []
    for t in range(length):
        state = decay[:, t, :, None] * rotor_sandwich(rotors[:, t], state) + drive[:, t]
        slow.append(state)
    slow = torch.stack(slow, dim=1)
    assert torch.allclose(fast, slow, atol=2e-5, rtol=2e-5)
    assert torch.allclose(final, slow[:, -1], atol=2e-5, rtol=2e-5)


def test_isotypic_equivariance_and_compact_scan():
    torch.manual_seed(3)
    x = torch.randn(2, 5, 3, 8)
    frame = normalized_rotor(torch.randn(2, 5, 4)).unsqueeze(-2)
    layer = Spin3IsotypicLinear(3, 4)
    assert torch.allclose(
        layer(rotor_sandwich(frame, x)), rotor_sandwich(frame, layer(x)), atol=2e-5
    )
    decay = torch.rand(2, 17, 3) * 0.4 + 0.5
    rotors = rotor_coefficients_from_bivector(torch.randn(2, 17, 3, 3))
    drive = torch.randn(2, 17, 3, 8) * 0.1
    initial = torch.randn(2, 3, 8)
    fast, _ = rotor_affine_scan(decay, rotors, drive, initial)
    state = initial
    slow = []
    for position in range(17):
        state = (
            decay[:, position, :, None]
            * rotor_sandwich_fast(rotors[:, position], state)
            + drive[:, position]
        )
        slow.append(state)
    assert torch.allclose(fast, torch.stack(slow, dim=1), atol=2e-5, rtol=2e-5)


def test_mamba3_affine_scan_matches_trapezoidal_recurrence():
    torch.manual_seed(21)
    batch, length, heads, state_width = 2, 19, 3, 5
    decay = torch.rand(batch, length, heads) * 0.4 + 0.5
    coefficient = torch.rand(batch, length, heads) * 0.2
    drive = torch.randn(batch, length, heads, 2, state_width)
    fast = mamba3_affine_scan(decay, coefficient, drive)
    state = torch.zeros(batch, heads, state_width)
    previous = torch.zeros_like(state)
    slow = []
    for position in range(length):
        state = (
            decay[:, position, :, None] * state
            + coefficient[:, position, :, None] * previous
            + drive[:, position, :, 0]
        )
        slow.append(state)
        previous = drive[:, position, :, 1]
    assert torch.allclose(fast, torch.stack(slow, dim=1), atol=2e-5, rtol=2e-5)


def test_mamba3_reference_shape_and_gradients():
    torch.manual_seed(22)
    model = Mamba3ReferenceLM(
        vocab_size=64, d_model=38, layers=2, d_state=8, headdim=19, mimo_rank=2
    )
    tokens = torch.randint(0, 64, (2, 11))
    logits = model(tokens)
    assert logits.shape == (2, 11, 64)
    F.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_saved_quality_artifact_contract():
    artifact = Path(__file__).resolve().parent / "results" / "quality_v2_300.json"
    summary = validate_report(artifact)
    assert summary["rows"] == 2
    assert summary["seeds"] == [0, 1]
    assert summary["parameter_counts"] == [674322]


def test_vision_patch_models_shape_and_gradients():
    torch.manual_seed(23)
    images = torch.rand(2, 3, 32, 32)
    assert patchify(images, 4).shape == (2, 64, 48)
    models = (
        SpinorVisionClassifier(48, 10, 44, 2, 2, 65),
        Mamba2VisionClassifier(48, 10, 160, 2, 65),
        Mamba3VisionClassifier(48, 10, 152, 2, 65),
    )
    for model in models:
        logits = model(images, 4)
        assert logits.shape == (2, 10)
        logits.square().mean().backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
