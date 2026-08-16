"""Acceptance gates for the maintained Pure Spin(8) SSM."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
from pure_spin8_ssm import __version__
from pure_spin8_ssm.torch_backend import (
    PureSpin8CausalLM,
    PureSpin8Config,
    PureSpin8SSMLayer,
    Spin8AffineTransition,
    apply_spin8_affine,
    compose_spin8_affine,
    hillis_steele_spin8_scan,
    recurrent_spin8_scan,
    spin8_factorized_actions,
    spin8_group_actions,
    work_efficient_spin8_scan,
)
from spin8_triality import (
    SPIN8_BIVECTOR_DIM,
    SPIN8_DIM,
    TRIALITY_REPRESENTATIONS,
    torch_triality_generators,
)
from spin8_triality_lift import triality_bind, triality_tensor


class PureSpin8AlgebraTests(unittest.TestCase):
    def test_factorized_center_and_orthogonality(self) -> None:
        dtype = torch.float64
        generators = torch_triality_generators(dtype=dtype)
        coordinates = torch.zeros(5, SPIN8_BIVECTOR_DIM, dtype=dtype)
        coordinates[:, 0] = torch.linspace(0.0, 2.0 * math.pi, 5, dtype=dtype)
        actions = spin8_factorized_actions(
            coordinates, generators, TRIALITY_REPRESENTATIONS
        )
        identity = torch.eye(SPIN8_DIM, dtype=dtype)
        orthogonality = actions.transpose(-1, -2) @ actions
        self.assertLess(float((orthogonality - identity).abs().max()), 1e-12)
        self.assertLess(float((actions[-1, 0] - identity).abs().max()), 1e-12)
        self.assertLess(float((actions[-1, 1] + identity).abs().max()), 1e-12)
        self.assertLess(float((actions[-1, 2] + identity).abs().max()), 1e-12)

    def test_one_plane_factorized_equals_exponential(self) -> None:
        torch.manual_seed(1)
        dtype = torch.float64
        generators = torch_triality_generators(dtype=dtype)
        coordinates = torch.zeros(7, SPIN8_BIVECTOR_DIM, dtype=dtype)
        coordinates[:, 11] = torch.randn(7, dtype=dtype)
        factorized = spin8_group_actions(
            coordinates,
            generators,
            TRIALITY_REPRESENTATIONS,
            mode="factorized",
        )
        exponential = spin8_group_actions(
            coordinates,
            generators,
            TRIALITY_REPRESENTATIONS,
            mode="exponential",
        )
        self.assertLess(float((factorized - exponential).abs().max()), 2e-15)

    def test_triality_tensor_is_equivariant_for_factorized_products(self) -> None:
        torch.manual_seed(2)
        dtype = torch.float64
        coordinates = 0.3 * torch.randn(SPIN8_BIVECTOR_DIM, dtype=dtype)
        actions = spin8_factorized_actions(
            coordinates,
            torch_triality_generators(dtype=dtype),
            TRIALITY_REPRESENTATIONS,
        )
        vector_action, positive_action, negative_action = actions
        positive = torch.randn(13, SPIN8_DIM, dtype=dtype)
        negative = torch.randn(13, SPIN8_DIM, dtype=dtype)
        rho = triality_tensor(dtype=dtype)
        expected = torch.einsum(
            "ij,bj->bi", vector_action, triality_bind(positive, negative, rho)
        )
        actual = triality_bind(
            torch.einsum("ij,bj->bi", positive_action, positive),
            torch.einsum("ij,bj->bi", negative_action, negative),
            rho,
        )
        self.assertLess(float((actual - expected).abs().max()), 3e-14)


class PureSpin8ScanTests(unittest.TestCase):
    @staticmethod
    def random_transition(
        batch: int, length: int, channels: int, *, dtype: torch.dtype
    ) -> Spin8AffineTransition:
        coordinates = 0.1 * torch.randn(
            batch, length, channels, SPIN8_BIVECTOR_DIM, dtype=dtype
        )
        action = spin8_factorized_actions(
            coordinates,
            torch_triality_generators(dtype=dtype),
            TRIALITY_REPRESENTATIONS,
        )
        return Spin8AffineTransition(
            scale=torch.sigmoid(torch.randn(batch, length, channels, dtype=dtype)),
            action=action,
            drive=0.05
            * torch.randn(
                batch,
                length,
                channels,
                len(TRIALITY_REPRESENTATIONS),
                SPIN8_DIM,
                dtype=dtype,
            ),
        )

    def test_affine_associativity(self) -> None:
        torch.manual_seed(3)
        transition = self.random_transition(1, 3, 2, dtype=torch.float64)

        def at(index: int) -> Spin8AffineTransition:
            return Spin8AffineTransition(
                transition.scale[:, index],
                transition.action[:, index],
                transition.drive[:, index],
            )

        left = compose_spin8_affine(at(2), compose_spin8_affine(at(1), at(0)))
        right = compose_spin8_affine(compose_spin8_affine(at(2), at(1)), at(0))
        self.assertLess(float((left.scale - right.scale).abs().max()), 1e-15)
        self.assertLess(float((left.action - right.action).abs().max()), 2e-15)
        self.assertLess(float((left.drive - right.drive).abs().max()), 2e-16)

    def test_work_efficient_hillis_and_recurrent_match(self) -> None:
        torch.manual_seed(4)
        transition = self.random_transition(2, 31, 2, dtype=torch.float64)
        initial = torch.randn(2, 2, 3, SPIN8_DIM, dtype=torch.float64)
        recurrent, _ = recurrent_spin8_scan(transition, initial)
        work = apply_spin8_affine(
            work_efficient_spin8_scan(transition), initial[:, None]
        )
        hillis = apply_spin8_affine(
            hillis_steele_spin8_scan(transition), initial[:, None]
        )
        self.assertLess(float((work - recurrent).abs().max()), 2e-15)
        self.assertLess(float((hillis - recurrent).abs().max()), 2e-15)

    def test_work_efficient_gradients_match_recurrent(self) -> None:
        torch.manual_seed(5)
        layer = PureSpin8SSMLayer(6, channels=1).double()
        inputs_a = torch.randn(2, 9, 6, dtype=torch.float64, requires_grad=True)
        inputs_b = inputs_a.detach().clone().requires_grad_(True)
        work, _ = layer(inputs_a, scan_mode="work_efficient", return_raw_states=True)
        loss_work = work.square().sum()
        gradients_work = torch.autograd.grad(
            loss_work,
            (inputs_a, layer.coefficient_controller.weight),
            retain_graph=True,
        )
        recurrent, _ = layer(inputs_b, scan_mode="recurrent", return_raw_states=True)
        loss_recurrent = recurrent.square().sum()
        gradients_recurrent = torch.autograd.grad(
            loss_recurrent, (inputs_b, layer.coefficient_controller.weight)
        )
        self.assertLess(float((work - recurrent).detach().abs().max()), 1e-14)
        for actual, expected in zip(gradients_work, gradients_recurrent):
            self.assertLess(float((actual - expected).abs().max()), 2e-12)


class PureSpin8LayerAndModelTests(unittest.TestCase):
    def test_bound_mask_and_cache_continuation(self) -> None:
        torch.manual_seed(6)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = PureSpin8SSMLayer(7, channels=2).float().to(device)
        inputs = torch.randn(3, 257, 7, device=device)
        mask = torch.ones(3, 257, dtype=torch.bool, device=device)
        mask[1, 233:] = False
        full, final = layer(
            inputs,
            valid_mask=mask,
            scan_mode="work_efficient",
            return_raw_states=True,
        )
        first, cache = layer(
            inputs[:, :113], scan_mode="work_efficient", return_raw_states=True
        )
        second, continued = layer(
            inputs[:, 113:],
            cache,
            valid_mask=mask[:, 113:],
            scan_mode="work_efficient",
            return_raw_states=True,
        )
        chunked = torch.cat((first, second), dim=1)
        self.assertLess(float((chunked - full).detach().abs().max()), 2e-6)
        self.assertLess(float((continued - final).detach().abs().max()), 2e-6)
        self.assertLessEqual(
            float(torch.linalg.vector_norm(full, dim=-1).detach().max()), 1.00001
        )
        self.assertEqual(layer.cache_scalars, 48)

    def test_model_shapes_backward_and_checkpoint_roundtrip(self) -> None:
        torch.manual_seed(7)
        config = PureSpin8Config(vocab_size=19, d_model=24, num_layers=2, channels=1)
        model = PureSpin8CausalLM(config)
        token_ids = torch.randint(0, config.vocab_size, (3, 11))
        result = model(token_ids, labels=token_ids)
        self.assertEqual(result["logits"].shape, (3, 11, config.vocab_size))
        self.assertEqual(len(result["states"]), config.num_layers)
        self.assertEqual(model.cache_scalars, 48)
        result["loss"].backward()
        gradient = model.backbone.blocks[0].ssm.coefficient_controller.weight.grad
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.norm()), 0.0)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pure_spin8.pt"
            model.save_checkpoint(checkpoint, metadata={"test": True})
            loaded = PureSpin8CausalLM.load_checkpoint(checkpoint)
            loaded_logits = loaded(token_ids)["logits"]
        self.assertEqual(__version__, "1.0.0")
        self.assertLess(
            float((loaded_logits - result["logits"].detach()).detach().abs().max()),
            1e-7,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_backward_is_finite(self) -> None:
        torch.manual_seed(8)
        layer = PureSpin8SSMLayer(5, channels=1).cuda()
        inputs = torch.randn(2, 17, 5, device="cuda", requires_grad=True)
        outputs, _ = layer(inputs, scan_mode="work_efficient")
        outputs.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(inputs.grad).all()))
        self.assertGreater(float(inputs.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
