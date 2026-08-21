import unittest

import torch

import mixed_monomial_golden_parallel_chunk_scan as parallel
import mixed_monomial_golden_triton_local_prefix as fused


def problem(
    device: str,
    *,
    batch: int = 3,
    chunks: int = 5,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(20_260_817)
    table = torch.randn(7, 4, 7, 24, 8, generator=generator).to(device)
    left = torch.randint(7, (batch, chunks), generator=generator).to(device)
    middle = torch.randint(4, (batch, chunks), generator=generator).to(device)
    right = torch.randint(7, (batch, chunks), generator=generator).to(device)
    incoming = torch.randn(
        batch, chunks, 8, generator=generator
    ).to(device)
    return table, left, middle, right, incoming


class MixedMonomialGoldenTritonLocalPrefixTests(unittest.TestCase):
    def test_cpu_auto_path_matches_eager_and_preserves_table_gradients(self) -> None:
        table, left, middle, right, incoming = problem("cpu")
        table.requires_grad_(True)
        incoming.requires_grad_(True)
        expected = fused.eager_indexed_local_prefix_states(
            table, left, middle, right, incoming
        )
        actual = fused.indexed_local_prefix_states(
            table, left, middle, right, incoming, backend="auto"
        )
        self.assertTrue(torch.equal(actual, expected))
        table_gradient, incoming_gradient = torch.autograd.grad(
            actual.square().mean(), (table, incoming)
        )
        self.assertEqual(table_gradient.shape, table.shape)
        self.assertEqual(incoming_gradient.shape, incoming.shape)
        self.assertTrue(torch.isfinite(table_gradient).all())
        self.assertTrue(torch.isfinite(incoming_gradient).all())

    def test_input_contract_rejects_mismatched_shapes_and_backends(self) -> None:
        table, left, middle, right, incoming = problem("cpu")
        with self.assertRaisesRegex(ValueError, "incoming"):
            fused.indexed_local_prefix_states(
                table, left, middle, right, incoming[:, :-1]
            )
        with self.assertRaisesRegex(ValueError, "backend"):
            fused.indexed_local_prefix_states(
                table,
                left,
                middle,
                right,
                incoming,
                backend="unknown",  # type: ignore[arg-type]
            )

    def test_integrated_indexed_eager_path_matches_preselected_path(self) -> None:
        table, left_index, middle_index, right_index, _ = problem(
            "cpu", batch=2, chunks=6
        )
        local = table[left_index, middle_index, right_index]
        endpoint = local[..., 16:24, :]
        initial_base = torch.randn(
            2,
            8,
            generator=torch.Generator().manual_seed(81),
            dtype=table.dtype,
        )
        weights = torch.randn(
            2,
            18,
            8,
            generator=torch.Generator().manual_seed(82),
            dtype=table.dtype,
        )

        def value_and_gradient(indexed: bool) -> tuple[torch.Tensor, torch.Tensor]:
            initial = initial_base.clone().requires_grad_(True)
            if indexed:
                states = parallel.compiled_parallel_indexed_states(
                    endpoint,
                    table,
                    left_index,
                    middle_index,
                    right_index,
                    initial,
                    local_backend="eager",
                )
            else:
                states = parallel.compiled_parallel_states(
                    endpoint, local, initial
                )
            (gradient,) = torch.autograd.grad((states * weights).sum(), initial)
            return states.detach(), gradient.detach()

        expected_states, expected_gradient = value_and_gradient(False)
        actual_states, actual_gradient = value_and_gradient(True)
        self.assertTrue(torch.equal(actual_states, expected_states))
        self.assertTrue(torch.equal(actual_gradient, expected_gradient))

    @unittest.skipUnless(
        fused.triton_is_available(), "requires the optional Triton CUDA path"
    )
    def test_cuda_fused_forward_and_incoming_gradient_match_eager(self) -> None:
        table, left, middle, right, incoming_base = problem("cuda")
        weights = torch.randn(
            3,
            15,
            8,
            generator=torch.Generator().manual_seed(91),
        ).cuda()

        def value_and_gradient(backend: fused.LocalPrefixBackend):
            incoming = incoming_base.clone().requires_grad_(True)
            states = fused.indexed_local_prefix_states(
                table,
                left,
                middle,
                right,
                incoming,
                backend=backend,
            )
            (gradient,) = torch.autograd.grad((states * weights).sum(), incoming)
            return states.detach(), gradient.detach()

        eager_states, eager_gradient = value_and_gradient("eager")
        triton_states, triton_gradient = value_and_gradient("triton")
        self.assertTrue(
            torch.allclose(eager_states, triton_states, atol=2e-6, rtol=2e-6)
        )
        self.assertTrue(
            torch.allclose(
                eager_gradient, triton_gradient, atol=4e-6, rtol=2e-6
            )
        )

    @unittest.skipUnless(
        fused.triton_is_available(), "requires the optional Triton CUDA path"
    )
    def test_cuda_integrated_scan_initial_gradient_matches_eager(self) -> None:
        table, left_index, middle_index, right_index, _ = problem(
            "cuda", batch=2, chunks=5
        )
        local = table[left_index, middle_index, right_index]
        endpoint = local[..., 16:24, :]
        initial_base = torch.randn(
            2,
            8,
            generator=torch.Generator().manual_seed(101),
        ).cuda()
        weights = torch.randn(
            2,
            15,
            8,
            generator=torch.Generator().manual_seed(102),
        ).cuda()

        def value_and_gradient(backend: fused.LocalPrefixBackend):
            initial = initial_base.clone().requires_grad_(True)
            states = parallel.compiled_parallel_indexed_states(
                endpoint,
                table,
                left_index,
                middle_index,
                right_index,
                initial,
                local_backend=backend,
            )
            (gradient,) = torch.autograd.grad((states * weights).sum(), initial)
            return states.detach(), gradient.detach()

        eager_states, eager_gradient = value_and_gradient("eager")
        triton_states, triton_gradient = value_and_gradient("triton")
        self.assertTrue(
            torch.allclose(eager_states, triton_states, atol=2e-5, rtol=2e-5)
        )
        self.assertTrue(
            torch.allclose(
                eager_gradient, triton_gradient, atol=2e-5, rtol=2e-5
            )
        )

    @unittest.skipUnless(
        fused.triton_is_available(), "requires the optional Triton CUDA path"
    )
    def test_explicit_fused_path_rejects_trainable_table(self) -> None:
        table, left, middle, right, incoming = problem("cuda")
        table.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "table gradients"):
            fused.triton_indexed_local_prefix_states(
                table, left, middle, right, incoming
            )


if __name__ == "__main__":
    unittest.main()
