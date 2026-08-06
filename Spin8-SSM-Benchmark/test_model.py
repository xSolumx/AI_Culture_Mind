import torch

from spinor_delta_ssm import (
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
