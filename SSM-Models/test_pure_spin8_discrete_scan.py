"""Contracts for compiled discrete-token Pure Spin(8) inference."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from pure_spin8_ssm.discrete_scan import (
    CompiledSpin8TokenTracker,
    discrete_spin8_scan,
    eager_discrete_spin8_scan,
    triton_is_available,
)

ROOT = Path(__file__).resolve().parent
BENCHMARK_ARTIFACT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_compiled_token_scan_seed1.json"
)
BENCHMARK_ARTIFACT_SHA256 = (
    "aa19ba66e5e2d17967f189c4744a1f1c165e0181b11c999f6cd0e4329dd6fb55"
)


def problem(device: str, *, batch: int = 3, length: int = 7):
    generator = torch.Generator(device="cpu").manual_seed(20_260_817)
    raw = torch.randn(8, 3, 8, 8, generator=generator)
    table = torch.linalg.qr(raw).Q.to(device)
    tokens = torch.randint(8, (batch, length), generator=generator).to(device)
    initial = torch.randn(batch, 3, 8, generator=generator).to(device)
    return table, tokens, initial


class PureSpin8DiscreteScanTests(unittest.TestCase):
    def test_cpu_auto_matches_eager_and_preserves_table_gradients(self) -> None:
        table, tokens, initial = problem("cpu")
        table.requires_grad_(True)
        initial.requires_grad_(True)
        expected = eager_discrete_spin8_scan(table, tokens, initial)
        actual = discrete_spin8_scan(table, tokens, initial, backend="auto")
        self.assertTrue(torch.equal(actual, expected))
        table_gradient, initial_gradient = torch.autograd.grad(
            actual.square().mean(), (table, initial)
        )
        self.assertTrue(torch.isfinite(table_gradient).all())
        self.assertTrue(torch.isfinite(initial_gradient).all())

    def test_token_range_is_checked_by_default(self) -> None:
        table, tokens, initial = problem("cpu")
        tokens[0, 0] = table.shape[0]
        with self.assertRaisesRegex(ValueError, "outside the action table"):
            discrete_spin8_scan(table, tokens, initial)

    def test_compiled_tracker_checkpoint_roundtrip(self) -> None:
        table, tokens, initial = problem("cpu", batch=2)
        tracker = CompiledSpin8TokenTracker(
            table,
            initial[0],
            representations=("vector", "positive", "negative"),
            metadata={"source": "unit-test"},
        )
        expected, expected_final = tracker(tokens, backend="eager")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compiled.pt"
            tracker.save_checkpoint(path)
            loaded = CompiledSpin8TokenTracker.load_checkpoint(path)
        actual, actual_final = loaded(tokens, backend="eager")
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_final, expected_final))
        self.assertEqual(loaded.metadata, {"source": "unit-test"})
        self.assertEqual(loaded.recurrent_state_scalars, 24)

    def test_frozen_model_level_benchmark_and_checkpoint(self) -> None:
        payload = BENCHMARK_ARTIFACT.read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(), BENCHMARK_ARTIFACT_SHA256
        )
        report = json.loads(payload)
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["pure_spin8_version"], "1.1.0")
        self.assertGreaterEqual(
            report["aggregate"]["triton_speedup_vs_source_dynamic_range"][0],
            30.0,
        )
        self.assertLessEqual(
            report["aggregate"]["maximum_grid_source_vs_triton_abs_error"],
            5e-5,
        )
        checkpoint_record = report["compiled_checkpoint"]
        checkpoint = Path(checkpoint_record["path"])
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            checkpoint_record["sha256"],
        )
        loaded = CompiledSpin8TokenTracker.load_checkpoint(checkpoint)
        self.assertEqual(tuple(loaded.action_table.shape), (8, 3, 8, 8))
        self.assertEqual(loaded.recurrent_state_scalars, 24)

    @unittest.skipUnless(
        triton_is_available(), "requires the optional Triton CUDA path"
    )
    def test_cuda_triton_forward_and_initial_gradient_match_eager(self) -> None:
        for length in (1, 5, 17):
            with self.subTest(length=length):
                table, tokens, initial_base = problem("cuda", length=length)
                weights = torch.randn_like(
                    eager_discrete_spin8_scan(table, tokens, initial_base)
                )

                def value_and_gradient(
                    backend: str,
                    initial_base: torch.Tensor = initial_base,
                    table: torch.Tensor = table,
                    tokens: torch.Tensor = tokens,
                    weights: torch.Tensor = weights,
                ):
                    initial = initial_base.clone().requires_grad_(True)
                    states = discrete_spin8_scan(
                        table,
                        tokens,
                        initial,
                        backend=backend,
                        validate_token_range=False,
                    )
                    (gradient,) = torch.autograd.grad(
                        (states * weights).sum(), initial
                    )
                    return states.detach(), gradient.detach()

                eager_states, eager_gradient = value_and_gradient("eager")
                triton_states, triton_gradient = value_and_gradient("triton")
                self.assertTrue(
                    torch.allclose(
                        triton_states, eager_states, atol=4e-6, rtol=3e-6
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        triton_gradient, eager_gradient, atol=8e-6, rtol=4e-6
                    )
                )

    @unittest.skipUnless(
        triton_is_available(), "requires the optional Triton CUDA path"
    )
    def test_explicit_triton_rejects_trainable_table(self) -> None:
        table, tokens, initial = problem("cuda")
        table.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "action-table gradients"):
            discrete_spin8_scan(table, tokens, initial, backend="triton")


if __name__ == "__main__":
    unittest.main()
