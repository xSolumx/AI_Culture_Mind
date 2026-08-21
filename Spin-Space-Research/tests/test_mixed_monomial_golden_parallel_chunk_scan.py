import unittest

import torch

from mixed_monomial_golden_parallel_chunk_scan import (
    compile_chunk_operators,
    compiled_parallel_from_primitives,
    compiled_parallel_states,
    primitive_parallel_states,
    primitive_recurrent_states,
    scan_composition_counts,
)


def stable_matrices(
    batch: int, chunks: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    identity = torch.eye(8, dtype=torch.float64).reshape(1, 1, 8, 8)
    values = []
    for _ in range(3):
        perturbation = 0.03 * torch.randn(
            batch, chunks, 8, 8, generator=generator, dtype=torch.float64
        )
        values.append(identity + perturbation)
    return values[0], values[1], values[2]


class MixedMonomialGoldenParallelChunkScanTests(unittest.TestCase):
    def test_both_parallel_trees_match_recurrence_and_compiled_scan(self) -> None:
        for chunks in (1, 3, 5, 8):
            with self.subTest(chunks=chunks):
                left, middle, right = stable_matrices(2, chunks, 100 + chunks)
                initial = torch.randn(
                    2,
                    8,
                    generator=torch.Generator().manual_seed(200 + chunks),
                    dtype=torch.float64,
                )
                recurrent = primitive_recurrent_states(
                    left, middle, right, initial
                )
                for backend in ("work_efficient", "hillis_steele"):
                    primitive = primitive_parallel_states(
                        left,
                        middle,
                        right,
                        initial,
                        backend=backend,
                    )
                    compiled = compiled_parallel_from_primitives(
                        left,
                        middle,
                        right,
                        initial,
                        backend=backend,
                    )
                    self.assertTrue(
                        torch.allclose(recurrent, primitive, atol=2e-12, rtol=2e-12)
                    )
                    self.assertTrue(
                        torch.allclose(recurrent, compiled, atol=2e-12, rtol=2e-12)
                    )

    def test_precompiled_operator_path_matches_differentiable_compilation(self) -> None:
        left, middle, right = stable_matrices(3, 7, 301)
        initial = torch.randn(
            3,
            8,
            generator=torch.Generator().manual_seed(302),
            dtype=torch.float64,
        )
        endpoint, local = compile_chunk_operators(left, middle, right)
        precompiled = compiled_parallel_states(endpoint, local, initial)
        direct = compiled_parallel_from_primitives(left, middle, right, initial)
        self.assertTrue(torch.equal(precompiled, direct))

    def test_forward_and_all_input_gradients_match_recurrent_oracle(self) -> None:
        base = stable_matrices(2, 4, 401)
        initial_base = torch.randn(
            2,
            8,
            generator=torch.Generator().manual_seed(402),
            dtype=torch.float64,
        )
        weights = torch.randn(
            2,
            12,
            8,
            generator=torch.Generator().manual_seed(403),
            dtype=torch.float64,
        )

        def value_and_gradients(compiled: bool):
            left, middle, right = (
                value.clone().requires_grad_(True) for value in base
            )
            initial = initial_base.clone().requires_grad_(True)
            states = (
                compiled_parallel_from_primitives(
                    left, middle, right, initial
                )
                if compiled
                else primitive_recurrent_states(
                    left, middle, right, initial
                )
            )
            loss = (states * weights).sum() + 0.01 * states.square().sum()
            gradients = torch.autograd.grad(
                loss, (left, middle, right, initial)
            )
            return states, gradients

        recurrent_states, recurrent_gradients = value_and_gradients(False)
        compiled_states, compiled_gradients = value_and_gradients(True)
        self.assertTrue(
            torch.allclose(
                recurrent_states, compiled_states, atol=2e-12, rtol=2e-12
            )
        )
        for recurrent, compiled in zip(
            recurrent_gradients, compiled_gradients, strict=True
        ):
            self.assertTrue(
                torch.allclose(recurrent, compiled, atol=5e-11, rtol=5e-11)
            )

    def test_chunking_reduces_parallel_tree_products(self) -> None:
        counts = scan_composition_counts(64)
        self.assertEqual(counts["primitive_length"], 192)
        self.assertEqual(counts["compiled_chunk_length"], 64)
        self.assertEqual(counts["primitive_work_efficient_products"], 766)
        self.assertEqual(counts["compiled_work_efficient_products"], 190)
        self.assertLess(
            counts["compiled_hillis_steele_products"],
            counts["primitive_hillis_steele_products"],
        )


if __name__ == "__main__":
    unittest.main()
