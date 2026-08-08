"""Contract tests for the preregistered v2.1 transport ladder."""

from __future__ import annotations

import unittest

import torch
from run_transport_ablation_v2 import (
    ACTION_KEYS,
    Q8_TABLE,
    PredictionConfig,
    associative_recall_batch,
    make_language_model,
)
from transport_ablation_v2 import (
    FAMILY_NAMES,
    MatchedTransportLanguageModel,
    MatchedTransportSSM,
    apply_transport,
    compose_affine_transitions,
    identity_action,
    transport_affine_scan,
    transport_recurrent_scan,
)


class TransportAlgebraTests(unittest.TestCase):
    def test_every_action_is_norm_nonexpansive(self) -> None:
        torch.manual_seed(300)
        inputs = torch.randn(2, 7, 3, 8, dtype=torch.float64)
        for family in FAMILY_NAMES:
            with self.subTest(family=family):
                layer = MatchedTransportSSM(3, family).double()
                _, actions, _ = layer.transitions(inputs)
                state = torch.randn(2, 7, 3, 8, dtype=torch.float64)
                transformed = apply_transport(family, actions, state)
                self.assertLessEqual(
                    float(
                        (transformed.norm(dim=-1) - state.norm(dim=-1)).max().detach()
                    ),
                    1e-11,
                )

    def test_composition_is_associative_and_matches_application(self) -> None:
        torch.manual_seed(301)
        inputs = torch.randn(2, 5, 2, 8, dtype=torch.float64)
        state = torch.randn(2, 2, 8, dtype=torch.float64)
        for family in FAMILY_NAMES:
            with self.subTest(family=family):
                layer = MatchedTransportSSM(2, family).double()
                decay, actions, drive = layer.transitions(inputs)
                transitions = tuple(
                    (decay[:, i], actions[:, i], drive[:, i]) for i in range(3)
                )
                a, b, c = transitions
                left = compose_affine_transitions(
                    family, c, compose_affine_transitions(family, b, a)
                )
                right = compose_affine_transitions(
                    family, compose_affine_transitions(family, c, b), a
                )
                for left_value, right_value in zip(left, right):
                    torch.testing.assert_close(
                        left_value, right_value, rtol=1e-11, atol=1e-11
                    )
                sequential = state
                for transition in transitions:
                    step_decay, step_action, step_drive = transition
                    sequential = (
                        step_decay.unsqueeze(-1)
                        * apply_transport(family, step_action, sequential)
                        + step_drive
                    )
                composed_decay, composed_action, composed_drive = left
                composed = (
                    composed_decay.unsqueeze(-1)
                    * apply_transport(family, composed_action, state)
                    + composed_drive
                )
                torch.testing.assert_close(composed, sequential, rtol=1e-11, atol=1e-11)


class TransportModelTests(unittest.TestCase):
    def test_parallel_recurrent_and_chunked_paths_match(self) -> None:
        torch.manual_seed(302)
        tokens = torch.arange(22).reshape(2, 11) % 31
        for family in FAMILY_NAMES:
            with self.subTest(family=family):
                model = (
                    MatchedTransportLanguageModel(31, 2, 1, family, dropout_rate=0)
                    .double()
                    .eval()
                )
                full, full_state = model(tokens, return_recurrent_states=True)
                recurrent, recurrent_state = model(
                    tokens,
                    return_recurrent_states=True,
                    scan_mode="recurrent",
                )
                first, cache = model(tokens[:, :4], return_recurrent_states=True)
                second, cache = model(
                    tokens[:, 4:], cache, return_recurrent_states=True
                )
                torch.testing.assert_close(full, recurrent, rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(
                    full,
                    torch.cat((first, second), dim=1),
                    rtol=1e-10,
                    atol=1e-10,
                )
                torch.testing.assert_close(
                    full_state[0], recurrent_state[0], rtol=1e-10, atol=1e-10
                )
                torch.testing.assert_close(
                    full_state[0], cache[0], rtol=1e-10, atol=1e-10
                )

    def test_long_arbitrary_inputs_remain_in_unit_ball(self) -> None:
        torch.manual_seed(303)
        inputs = 1e5 * torch.randn(2, 1024, 2, 8, dtype=torch.float64)
        for family in FAMILY_NAMES:
            with self.subTest(family=family):
                layer = MatchedTransportSSM(2, family).double()
                states, _ = layer(inputs, scan_mode="recurrent")
                self.assertLessEqual(
                    float(states.norm(dim=-1).max().detach()), 1 + 2e-10
                )

    def test_action_parameters_receive_finite_gradients(self) -> None:
        torch.manual_seed(304)
        inputs = torch.randn(2, 9, 2, 8)
        for family in FAMILY_NAMES[1:]:
            with self.subTest(family=family):
                layer = MatchedTransportSSM(2, family)
                outputs, _ = layer(inputs)
                outputs[:, 1:].square().mean().backward()
                if family == "rotor":
                    parameters = layer.rotor_control.parameters()
                elif family == "fixed_rotor":
                    parameters = (layer.fixed_bivector,)
                else:
                    parameters = layer.action_control.parameters()
                gradients = [
                    parameter.grad
                    for parameter in parameters
                    if parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_identity_intervention_is_exact(self) -> None:
        torch.manual_seed(305)
        layer = MatchedTransportSSM(2, "rotor").double()
        inputs = torch.randn(2, 13, 2, 8, dtype=torch.float64)
        decay, actions, drive = layer.transitions(inputs, force_identity=True)
        expected = identity_action(inputs, "rotor")
        torch.testing.assert_close(actions, expected, rtol=0, atol=0)
        parallel, _ = transport_affine_scan("rotor", decay, actions, drive)
        recurrent, _ = transport_recurrent_scan("rotor", decay, actions, drive)
        torch.testing.assert_close(parallel, recurrent, rtol=1e-12, atol=1e-12)

    def test_cuda_all_families_forward_backward(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        torch.manual_seed(306)
        tokens = torch.randint(0, 31, (2, 17), device="cuda")
        for family in FAMILY_NAMES:
            with self.subTest(family=family):
                model = MatchedTransportLanguageModel(31, 2, 1, family).cuda()
                loss = model(tokens).square().mean()
                loss.backward()
                self.assertTrue(
                    all(
                        bool(torch.isfinite(parameter.grad).all())
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    )
                )


class RunnerProtocolTests(unittest.TestCase):
    def test_q8_table_has_expected_noncommutative_group_law(self) -> None:
        self.assertEqual(int(Q8_TABLE[2, 4]), 6)  # i*j=k
        self.assertEqual(int(Q8_TABLE[4, 2]), 7)  # j*i=-k
        self.assertEqual(int(Q8_TABLE[2, 2]), 1)  # i*i=-1
        for left in range(8):
            for middle in range(8):
                for right in range(8):
                    self.assertEqual(
                        int(Q8_TABLE[Q8_TABLE[left, middle], right]),
                        int(Q8_TABLE[left, Q8_TABLE[middle, right]]),
                    )

    def test_associative_recall_target_matches_encoded_query(self) -> None:
        inputs, targets = associative_recall_batch(
            16, 64, torch.Generator().manual_seed(307)
        )
        self.assertTrue(bool((inputs[:, -2] == 272).all()))
        for row in range(inputs.shape[0]):
            pairs = {
                int(inputs[row, position]): int(inputs[row, position + 1]) - 256
                for position in range(0, inputs.shape[1] - 2, 2)
            }
            self.assertEqual(pairs[int(inputs[row, -1])], int(targets[row]))

    def test_common_initialization_is_identical_across_families(self) -> None:
        config = PredictionConfig(channels=2, layers=1)
        identity, _ = make_language_model("identity", 2, 308, config)
        reference = identity.state_dict()
        for family in FAMILY_NAMES[1:]:
            with self.subTest(family=family):
                model, _ = make_language_model(family, 2, 308, config)
                state = model.state_dict()
                for key, value in reference.items():
                    if any(marker in key for marker in ACTION_KEYS):
                        continue
                    if key in state and state[key].shape == value.shape:
                        torch.testing.assert_close(state[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
