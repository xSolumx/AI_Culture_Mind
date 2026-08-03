"""Load-bearing tests for the isolated SpinorModel overhaul."""

from __future__ import annotations

import math
import tempfile
import types
import unittest
from pathlib import Path

import torch

from overhauled.algebra import (
    GA_DIM,
    RotorAffineTransition,
    apply_transition,
    associative_scan,
    compose_transitions,
    geometric_product,
    reversion,
    rotor_from_bivector,
    rotor_sandwich,
)
from overhauled.model import SpinorSSMConfig, SpinorSSMLanguageModel
from overhauled.train import Vocabulary, load_checkpoint, save_checkpoint


def random_transition(
    shape: tuple[int, ...], *, generator: torch.Generator
) -> RotorAffineTransition:
    bivector = torch.randn(*shape, 3, generator=generator, dtype=torch.float64)
    rotor = rotor_from_bivector(bivector)
    retention = torch.rand(*shape, generator=generator, dtype=torch.float64)
    drive = torch.randn(*shape, GA_DIM, generator=generator, dtype=torch.float64)
    return RotorAffineTransition(retention, rotor, drive)


class AlgebraAndScanTests(unittest.TestCase):
    def test_basis_reversion_and_rotor_tangent(self) -> None:
        basis = torch.eye(GA_DIM, dtype=torch.float64)
        torch.testing.assert_close(geometric_product(basis[1], basis[1]), basis[0])
        torch.testing.assert_close(geometric_product(basis[1], basis[2]), basis[4])
        torch.testing.assert_close(geometric_product(basis[2], basis[1]), -basis[4])
        left = torch.arange(1, 9, dtype=torch.float64)
        right = torch.arange(8, 0, -1, dtype=torch.float64)
        torch.testing.assert_close(
            reversion(geometric_product(left, right)),
            geometric_product(reversion(right), reversion(left)),
        )

        bivector = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        state = torch.tensor(
            [0.0, 1.0, 2.0, -1.0, 0.5, -0.2, 0.3, 0.0],
            dtype=torch.float64,
        )
        weights = torch.arange(1, 9, dtype=torch.float64)
        output = (rotor_sandwich(rotor_from_bivector(bivector), state) * weights).sum()
        output.backward()
        self.assertGreater(float(bivector.grad.norm()), 1e-8)

    def test_transition_composition_is_associative_and_matches_application(self) -> None:
        generator = torch.Generator().manual_seed(1)
        first = random_transition((2, 3), generator=generator)
        second = random_transition((2, 3), generator=generator)
        third = random_transition((2, 3), generator=generator)
        left = compose_transitions(third, compose_transitions(second, first))
        right = compose_transitions(compose_transitions(third, second), first)
        torch.testing.assert_close(left.retention, right.retention, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(left.rotor, right.rotor, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(left.drive, right.drive, rtol=1e-12, atol=1e-12)

        state = torch.randn(2, 3, GA_DIM, generator=generator, dtype=torch.float64)
        sequential = apply_transition(
            third, apply_transition(second, apply_transition(first, state))
        )
        torch.testing.assert_close(apply_transition(left, state), sequential)

    def test_parallel_scan_matches_sequential_recurrence(self) -> None:
        generator = torch.Generator().manual_seed(2)
        transition = random_transition((3, 17, 4), generator=generator)
        initial = torch.randn(3, 4, GA_DIM, generator=generator, dtype=torch.float64)
        parallel = apply_transition(associative_scan(transition), initial[:, None])
        state = initial
        recurrent = []
        for position in range(17):
            state = apply_transition(
                RotorAffineTransition(
                    transition.retention[:, position],
                    transition.rotor[:, position],
                    transition.drive[:, position],
                ),
                state,
            )
            recurrent.append(state)
        torch.testing.assert_close(
            parallel, torch.stack(recurrent, dim=1), rtol=2e-11, atol=2e-11
        )


class LanguageModelContractTests(unittest.TestCase):
    def make_model(self) -> SpinorSSMLanguageModel:
        torch.manual_seed(3)
        return SpinorSSMLanguageModel(
            SpinorSSMConfig(
                vocab_size=23,
                channels=3,
                num_layers=2,
                expansion=2,
                dropout=0.0,
                max_half_life=256.0,
            )
        ).double()

    def test_full_chunk_and_token_streaming_are_equivalent(self) -> None:
        model = self.make_model().eval()
        tokens = torch.randint(1, 23, (2, 13), generator=torch.Generator().manual_seed(4))
        full, full_state = model(
            tokens, return_recurrent_states=True, backend="parallel"
        )

        first, state = model(
            tokens[:, :5], return_recurrent_states=True, backend="parallel"
        )
        second, chunk_state = model(
            tokens[:, 5:], state, return_recurrent_states=True, backend="parallel"
        )
        chunked = torch.cat((first, second), dim=1)

        state = model.initial_states(tokens.shape[0], dtype=torch.float64)
        streamed = []
        for position in range(tokens.shape[1]):
            logits, state = model.step(tokens[:, position], state)
            streamed.append(logits)
        streamed_logits = torch.stack(streamed, dim=1)

        torch.testing.assert_close(full, chunked, rtol=2e-10, atol=2e-10)
        torch.testing.assert_close(full, streamed_logits, rtol=2e-10, atol=2e-10)
        for expected, actual in zip(full_state, chunk_state):
            torch.testing.assert_close(expected, actual, rtol=2e-10, atol=2e-10)
        for expected, actual in zip(full_state, state):
            torch.testing.assert_close(expected, actual, rtol=2e-10, atol=2e-10)
        self.assertEqual(model.recurrent_state_scalars, 2 * 3 * GA_DIM)

    def test_initialized_retention_matches_requested_half_lives(self) -> None:
        model = self.make_model().eval()
        block = model.blocks[0].ssm
        inputs = torch.randn(2, 5, 3, GA_DIM, dtype=torch.float64)
        retention = block.transitions(inputs).retention
        expected_half_lives = torch.logspace(
            math.log10(model.config.min_half_life),
            math.log10(model.config.max_half_life),
            model.config.channels,
            dtype=torch.float64,
        )
        expected = torch.exp(-math.log(2.0) / expected_half_lives)
        torch.testing.assert_close(
            retention,
            expected.expand_as(retention),
            rtol=2e-7,
            atol=2e-7,
        )

    def test_convex_innovation_parameterization_obeys_bibo_bound(self) -> None:
        model = self.make_model().eval()
        ssm = model.blocks[0].ssm
        inputs = torch.randn(2, 128, 3, GA_DIM, dtype=torch.float64)
        transition = ssm.transitions(inputs)
        effective_candidate = transition.drive / (
            1.0 - transition.retention
        ).unsqueeze(-1)
        bound = effective_candidate.norm(dim=-1).amax(dim=1)
        sequence = apply_transition(
            associative_scan(transition),
            torch.zeros(2, 1, 3, GA_DIM, dtype=torch.float64),
        )
        observed = sequence.norm(dim=-1).amax(dim=1)
        self.assertTrue(torch.all(observed <= bound + 1e-10))

    def test_padding_is_an_identity_transition_and_forward_is_causal(self) -> None:
        model = self.make_model().eval()
        prefix = torch.tensor([[2, 3, 4]])
        padded = torch.tensor([[2, 3, 4, 0, 0]])
        _, prefix_state = model(prefix, return_recurrent_states=True)
        _, padded_state = model(
            padded,
            attention_mask=torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool),
            return_recurrent_states=True,
        )
        for expected, actual in zip(prefix_state, padded_state):
            torch.testing.assert_close(expected, actual, rtol=2e-10, atol=2e-10)

        first = torch.tensor([[2, 3, 4, 5]])
        second = torch.tensor([[2, 3, 4, 6]])
        first_logits = model(first)
        second_logits = model(second)
        torch.testing.assert_close(
            first_logits[:, :3], second_logits[:, :3], rtol=2e-10, atol=2e-10
        )

    def test_backward_reaches_identity_rotor_controller(self) -> None:
        model = self.make_model().train()
        tokens = torch.randint(1, 23, (3, 7), generator=torch.Generator().manual_seed(5))
        targets = torch.randint(1, 23, (3, 7), generator=torch.Generator().manual_seed(6))
        loss = torch.nn.functional.cross_entropy(model(tokens).flatten(0, 1), targets.flatten())
        loss.backward()
        for block in model.blocks:
            gradient = block.ssm.rotor_strength.weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.norm()), 0.0)

    def test_generation_primes_once_then_uses_single_token_steps(self) -> None:
        model = self.make_model().eval()
        lengths = []
        original = model.forward

        def recording_forward(instance, token_ids, *args, **kwargs):
            lengths.append(token_ids.shape[1])
            return original(token_ids, *args, **kwargs)

        model.forward = types.MethodType(recording_forward, model)
        prompt = torch.tensor([[2, 3, 4, 5]])
        generated = model.generate(prompt, max_new_tokens=3)
        self.assertEqual(generated.shape, (1, 7))
        self.assertEqual(lengths[0], 4)
        self.assertTrue(all(length == 1 for length in lengths[1:]))

    def test_checkpoint_round_trip_is_exact(self) -> None:
        model = self.make_model().eval()
        with torch.no_grad():
            model.output_bias[0] = 0.123456789012345
        vocabulary = Vocabulary(("<pad>", "<unk>", "<bos>", "<eos>", *map(str, range(19))))
        tokens = torch.tensor([[2, 4, 5, 3]])
        expected = model(tokens)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, vocabulary, metadata={"test": True})
            loaded, loaded_vocabulary, metadata = load_checkpoint(
                path, device=torch.device("cpu")
            )
        loaded = loaded.eval()
        self.assertEqual(loaded.output_bias.dtype, torch.float64)
        torch.testing.assert_close(loaded(tokens), expected, rtol=0, atol=0)
        self.assertEqual(loaded_vocabulary, vocabulary)
        self.assertEqual(metadata, {"test": True})


if __name__ == "__main__":
    unittest.main()
