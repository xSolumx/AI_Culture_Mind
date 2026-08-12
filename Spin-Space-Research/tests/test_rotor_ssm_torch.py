"""PyTorch algebra, streaming, and gradient tests."""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from rotor_ssm_torch import (
        GA_DIM,
        GASSMLanguageModel,
        SelectiveRotorSSM,
        geometric_product,
        reversion,
        rotor_from_bivector,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchRotorSSMTests(unittest.TestCase):
    def basis(self, index: int):
        return torch.nn.functional.one_hot(torch.tensor(index), GA_DIM).float()

    def test_algebra_and_unit_rotors(self) -> None:
        one, e1, e2 = self.basis(0), self.basis(1), self.basis(2)
        e12 = self.basis(4)
        torch.testing.assert_close(geometric_product(e1, e1), one)
        torch.testing.assert_close(geometric_product(e1, e2), e12)
        torch.testing.assert_close(geometric_product(e2, e1), -e12)
        rotors = rotor_from_bivector(torch.randn(7, 3))
        torch.testing.assert_close(
            geometric_product(rotors, reversion(rotors)),
            one.expand_as(rotors),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_sequence_chunk_and_token_streaming_match(self) -> None:
        torch.manual_seed(0)
        model = GASSMLanguageModel(
            vocab_size=24,
            channels=3,
            num_layers=2,
            dropout_rate=0.0,
        ).eval()
        tokens = torch.arange(18).reshape(2, 9) % 24
        full_logits, full_states = model(tokens, return_recurrent_states=True)
        first_logits, states = model(tokens[:, :4], return_recurrent_states=True)
        second_logits, states = model(
            tokens[:, 4:], states, return_recurrent_states=True
        )
        chunked = torch.cat([first_logits, second_logits], dim=1)

        stream_logits = []
        stream_states = None
        for position in range(tokens.shape[1]):
            logits, stream_states = model(
                tokens[:, position : position + 1],
                stream_states,
                return_recurrent_states=True,
            )
            stream_logits.append(logits)
        streamed = torch.cat(stream_logits, dim=1)
        torch.testing.assert_close(full_logits, chunked, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(full_logits, streamed, rtol=1e-5, atol=1e-5)
        for expected, chunk_state, stream_state in zip(
            full_states, states, stream_states
        ):
            torch.testing.assert_close(expected, chunk_state, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(expected, stream_state, rtol=1e-5, atol=1e-5)

    def test_cuda_forward_backward_when_available(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        model = GASSMLanguageModel(
            vocab_size=64, channels=4, num_layers=1, dropout_rate=0.0
        ).cuda()
        tokens = torch.randint(0, 64, (4, 16), device="cuda")
        loss = model(tokens).square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))

    def test_rotor_controller_starts_at_identity_but_receives_gradients(self) -> None:
        torch.manual_seed(4)
        layer = SelectiveRotorSSM(channels=3)
        inputs = torch.randn(2, 6, 3, GA_DIM)
        _, rotors, _ = layer.transitions(inputs)
        identity = self.basis(0).expand_as(rotors)
        torch.testing.assert_close(rotors, identity, rtol=1e-6, atol=1e-6)

        outputs, _ = layer(inputs)
        outputs[:, 1:].square().mean().backward()
        gradient = layer.rotor_control.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_initial_half_lives_and_uniform_contraction(self) -> None:
        layer = SelectiveRotorSSM(
            channels=3, min_half_life=4.0, max_half_life=16.0
        )
        inputs = torch.zeros(2, 5, 3, GA_DIM)
        decay, _, _ = layer.transitions(inputs)
        expected = torch.pow(
            torch.tensor(0.5),
            1.0 / torch.tensor([4.0, 8.0, 16.0]),
        )
        torch.testing.assert_close(decay[0, 0], expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(bool(torch.all(decay < 1.0)))


if __name__ == "__main__":
    unittest.main()
