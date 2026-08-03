"""Fast tests for the maintained tensor-only SpinorModel implementation."""

import unittest

try:
    import torch
except ImportError:  # The repository's JAX-only environment does not include PyTorch.
    torch = None

if torch is not None:
    from geometric_layers import GA_DIM, geometric_product_ga3, reversion
    from spinor_llm import SpinorLLM


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GeometricAlgebraTests(unittest.TestCase):
    def basis(self, index: int):
        return torch.nn.functional.one_hot(torch.tensor(index), GA_DIM).float()

    def test_basis_identities(self) -> None:
        one, e1, e2, e3 = (self.basis(index) for index in range(4))
        e12, e13, e23, e123 = (self.basis(index) for index in range(4, 8))
        torch.testing.assert_close(geometric_product_ga3(e1, e1), one)
        torch.testing.assert_close(geometric_product_ga3(e1, e2), e12)
        torch.testing.assert_close(geometric_product_ga3(e2, e1), -e12)
        torch.testing.assert_close(geometric_product_ga3(e1, e3), e13)
        torch.testing.assert_close(geometric_product_ga3(e2, e3), e23)
        torch.testing.assert_close(geometric_product_ga3(e12, e3), e123)

    def test_reversion_reverses_product(self) -> None:
        left = torch.arange(1, 9, dtype=torch.float32)
        right = torch.arange(8, 0, -1, dtype=torch.float32)
        actual = reversion(geometric_product_ga3(left, right))
        expected = geometric_product_ga3(reversion(right), reversion(left))
        torch.testing.assert_close(actual, expected)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelSmokeTests(unittest.TestCase):
    def test_forward_backward_and_causality(self) -> None:
        torch.manual_seed(0)
        model = SpinorLLM(
            vocab_size=16,
            num_layers=1,
            num_heads=2,
            dropout_rate=0.0,
        )
        model.eval()
        first = torch.tensor([[2, 3, 4]])
        second = torch.tensor([[2, 3, 5]])
        first_logits = model(first)
        second_logits = model(second)
        self.assertEqual(first_logits.shape, (1, 3, 16))
        torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2])

        model.train()
        loss = model(first).sum()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
