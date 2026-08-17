import unittest

import torch

import mixed_monomial_golden_parallel_chunk_scan as parallel
import mixed_monomial_golden_triton_chunk_recurrence as fused


def problem(
    device: str,
    *,
    batch: int = 3,
    chunks: int = 5,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(20_260_818)
    table = (
        torch.randn(7, 4, 7, 24, 8, generator=generator) / 4
    ).to(device)
    left = torch.randint(7, (batch, chunks), generator=generator).to(device)
    middle = torch.randint(4, (batch, chunks), generator=generator).to(device)
    right = torch.randint(7, (batch, chunks), generator=generator).to(device)
    initial = torch.randn(batch, 8, generator=generator).to(device)
    return table, left, middle, right, initial


class MixedMonomialGoldenTritonChunkRecurrenceTests(unittest.TestCase):
    def test_cpu_auto_path_matches_eager_and_preserves_table_gradients(self) -> None:
        table, left, middle, right, initial = problem("cpu")
        table.requires_grad_(True)
        initial.requires_grad_(True)
        expected = fused.eager_indexed_chunk_recurrence(
            table, left, middle, right, initial
        )
        actual = fused.indexed_chunk_recurrence(
            table, left, middle, right, initial, backend="auto"
        )
        self.assertTrue(torch.equal(actual, expected))
        table_gradient, initial_gradient = torch.autograd.grad(
            actual.square().mean(), (table, initial)
        )
        self.assertEqual(table_gradient.shape, table.shape)
        self.assertEqual(initial_gradient.shape, initial.shape)
        self.assertTrue(torch.isfinite(table_gradient).all())
        self.assertTrue(torch.isfinite(initial_gradient).all())

    def test_eager_recurrence_matches_parallel_endpoint_scan(self) -> None:
        table, left, middle, right, initial = problem(
            "cpu", batch=2, chunks=7
        )
        selected = table[left, middle, right]
        endpoint = selected[..., 16:24, :]
        recurrent = fused.eager_indexed_chunk_recurrence(
            table, left, middle, right, initial
        )
        scanned = parallel.compiled_parallel_states(
            endpoint, selected, initial
        )
        self.assertTrue(
            torch.allclose(recurrent, scanned, atol=3e-6, rtol=3e-6)
        )

    @unittest.skipUnless(
        fused.triton_is_available(), "requires the optional Triton CUDA path"
    )
    def test_cuda_fused_forward_and_initial_gradient_match_eager(self) -> None:
        for chunks in (1, 3, 8):
            with self.subTest(chunks=chunks):
                table, left, middle, right, initial_base = problem(
                    "cuda", chunks=chunks
                )
                weights = torch.randn(
                    3,
                    3 * chunks,
                    8,
                    generator=torch.Generator().manual_seed(110 + chunks),
                ).cuda()

                def value_and_gradient(
                    backend: fused.ChunkRecurrenceBackend,
                    initial_base: torch.Tensor = initial_base,
                    table: torch.Tensor = table,
                    left: torch.Tensor = left,
                    middle: torch.Tensor = middle,
                    right: torch.Tensor = right,
                    weights: torch.Tensor = weights,
                ):
                    initial = initial_base.clone().requires_grad_(True)
                    states = fused.indexed_chunk_recurrence(
                        table,
                        left,
                        middle,
                        right,
                        initial,
                        backend=backend,
                    )
                    (gradient,) = torch.autograd.grad(
                        (states * weights).sum(), initial
                    )
                    return states.detach(), gradient.detach()

                eager_states, eager_gradient = value_and_gradient("eager")
                triton_states, triton_gradient = value_and_gradient("triton")
                self.assertTrue(
                    torch.allclose(
                        eager_states,
                        triton_states,
                        atol=4e-6,
                        rtol=3e-6,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        eager_gradient,
                        triton_gradient,
                        atol=8e-6,
                        rtol=3e-6,
                    )
                )

    @unittest.skipUnless(
        fused.triton_is_available(), "requires the optional Triton CUDA path"
    )
    def test_cuda_fused_recurrence_matches_parallel_scan_and_gradient(self) -> None:
        table, left, middle, right, initial_base = problem(
            "cuda", batch=4, chunks=6
        )
        selected = table[left, middle, right]
        endpoint = selected[..., 16:24, :]
        weights = torch.randn(
            4,
            18,
            8,
            generator=torch.Generator().manual_seed(121),
        ).cuda()

        def value_and_gradient(recurrent: bool):
            initial = initial_base.clone().requires_grad_(True)
            if recurrent:
                states = fused.triton_indexed_chunk_recurrence(
                    table, left, middle, right, initial
                )
            else:
                states = parallel.compiled_parallel_states(
                    endpoint, selected, initial
                )
            (gradient,) = torch.autograd.grad(
                (states * weights).sum() / states.numel(), initial
            )
            return states.detach(), gradient.detach()

        scanned_states, scanned_gradient = value_and_gradient(False)
        recurrent_states, recurrent_gradient = value_and_gradient(True)
        self.assertTrue(
            torch.allclose(
                scanned_states, recurrent_states, atol=8e-6, rtol=5e-6
            )
        )
        self.assertTrue(
            torch.allclose(
                scanned_gradient, recurrent_gradient, atol=5e-6, rtol=5e-6
            )
        )

    @unittest.skipUnless(
        fused.triton_is_available(), "requires the optional Triton CUDA path"
    )
    def test_explicit_fused_path_rejects_trainable_table(self) -> None:
        table, left, middle, right, initial = problem("cuda")
        table.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "table gradients"):
            fused.triton_indexed_chunk_recurrence(
                table, left, middle, right, initial
            )


if __name__ == "__main__":
    unittest.main()
