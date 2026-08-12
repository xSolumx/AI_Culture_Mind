from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from spin9_clifford_memory import (
    diagnostics,
    spin9_bind,
    spin9_hopf,
    spin9_involutions,
    spin9_unbind,
)


class Spin9CliffordMemoryTests(unittest.TestCase):
    def test_memory_boundary_diagnostics(self) -> None:
        report = diagnostics()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["state_scalars"]["direct_slots"], 64)
        self.assertEqual(report["state_scalars"]["spin9_bound_slots"], 64)
        self.assertFalse(
            report["claim_boundary"]["spin9_specific_capacity_advantage_established"]
        )

    def test_bind_unbind_and_hopf_identity_have_gradients(self) -> None:
        generator = torch.Generator().manual_seed(940_001)
        involutions = spin9_involutions(dtype=torch.float64)
        address = F.normalize(
            torch.randn(5, 9, generator=generator, dtype=torch.float64), dim=-1
        ).requires_grad_(True)
        value = F.normalize(
            torch.randn(5, 16, generator=generator, dtype=torch.float64), dim=-1
        ).requires_grad_(True)
        bound = spin9_bind(address, value, involutions)
        recovered = spin9_unbind(address, bound, involutions)
        hopf = spin9_hopf(value, involutions)
        loss = recovered.square().mean() + hopf.square().mean()
        gradients = torch.autograd.grad(loss, (address, value))
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        self.assertLess(float((recovered - value).detach().abs().max()), 2e-14)


if __name__ == "__main__":
    unittest.main()
