"""Backend-parity and mathematical-contract tests for the pure rotor SSM."""

from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import torch

import pure_rotor_ssm.jax_backend as jx
import pure_rotor_ssm.torch_backend as pt


class PureAlgebraTests(unittest.TestCase):
    def test_specialized_rotor_operations_match_dense_geometric_product(self) -> None:
        rng = np.random.default_rng(100)
        bivectors = rng.normal(size=(3, 5, 3))
        other_bivectors = rng.normal(size=(3, 5, 3))
        multivectors = rng.normal(size=(3, 5, 8))

        tq = pt.rotor_from_bivector(torch.tensor(bivectors, dtype=torch.float64))
        tr = pt.rotor_from_bivector(
            torch.tensor(other_bivectors, dtype=torch.float64)
        )
        tx = torch.tensor(multivectors, dtype=torch.float64)
        torch.testing.assert_close(
            pt.rotor_product(tq, tr),
            pt.geometric_product(tq, tr),
            rtol=1e-13,
            atol=1e-13,
        )
        torch.testing.assert_close(
            pt.rotor_sandwich(tq, tx),
            pt.geometric_product(
                pt.geometric_product(tq, tx), pt.reversion(tq)
            ),
            rtol=1e-13,
            atol=1e-13,
        )

        jq = jx.rotor_from_bivector(jnp.asarray(bivectors))
        jr = jx.rotor_from_bivector(jnp.asarray(other_bivectors))
        jmv = jnp.asarray(multivectors)
        np.testing.assert_allclose(
            jx.rotor_product(jq, jr),
            jx.geometric_product(jq, jr),
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            jx.rotor_sandwich(jq, jmv),
            jx.geometric_product(
                jx.geometric_product(jq, jmv), jx.reversion(jq)
            ),
            rtol=1e-13,
            atol=1e-13,
        )

    def test_cross_backend_algebra_parity_and_even_closure(self) -> None:
        rng = np.random.default_rng(101)
        bivectors = rng.normal(size=(4, 7, 3))
        multivectors = rng.normal(size=(4, 7, 8))
        tq = pt.rotor_from_bivector(torch.tensor(bivectors, dtype=torch.float64))
        jq = jx.rotor_from_bivector(jnp.asarray(bivectors))
        tx = torch.tensor(multivectors, dtype=torch.float64)
        jx_values = jnp.asarray(multivectors)
        np.testing.assert_allclose(tq.numpy(), np.asarray(jq), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            pt.rotor_sandwich(tq, tx).numpy(),
            np.asarray(jx.rotor_sandwich(jq, jx_values)),
            rtol=1e-13,
            atol=1e-13,
        )
        products = pt.rotor_product(tq[:, :1], tq[:, 1:])
        self.assertEqual(float(products[..., [1, 2, 3, 7]].abs().max()), 0.0)
        torch.testing.assert_close(
            products.square().sum(dim=-1),
            torch.ones_like(products[..., 0]),
            rtol=1e-13,
            atol=1e-13,
        )

    def test_smooth_features_have_finite_zero_jacobians(self) -> None:
        jax_jacobian = jax.jacfwd(jx.spin3_invariant_features)(
            jnp.zeros((8,), dtype=jnp.float64)
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(jax_jacobian))))
        np.testing.assert_array_equal(
            np.asarray(jax_jacobian[jnp.asarray([2, 3, 4])]), np.zeros((3, 8))
        )

        values = torch.zeros(8, dtype=torch.float64, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(
            pt.spin3_invariant_features, values
        )
        self.assertTrue(bool(torch.isfinite(jacobian).all()))
        torch.testing.assert_close(jacobian[2:], torch.zeros_like(jacobian[2:]))


class PureTransitionTests(unittest.TestCase):
    @staticmethod
    def _transition_arrays(seed: int = 102, length: int = 31):
        rng = np.random.default_rng(seed)
        decay = rng.uniform(0.65, 0.999, size=(2, length, 3))
        rotors = rng.normal(size=(2, length, 3, 3))
        drive = rng.normal(size=(2, length, 3, 8)) * 0.02
        initial = rng.normal(size=(2, 3, 8))
        return decay, rotors, drive, initial

    def test_associativity_and_parallel_recurrent_parity(self) -> None:
        decay, bivectors, drive, initial = self._transition_arrays()
        td = torch.tensor(decay, dtype=torch.float64)
        tq = pt.rotor_from_bivector(torch.tensor(bivectors, dtype=torch.float64))
        tb = torch.tensor(drive, dtype=torch.float64)
        ti = torch.tensor(initial, dtype=torch.float64)
        parallel, parallel_final = pt.rotor_affine_scan(td, tq, tb, ti)
        recurrent, recurrent_final = pt.rotor_recurrent_scan(td, tq, tb, ti)
        torch.testing.assert_close(parallel, recurrent, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            parallel_final, recurrent_final, rtol=1e-12, atol=1e-12
        )

        transitions = tuple(
            (td[:, index], tq[:, index], tb[:, index]) for index in range(3)
        )
        a, b, c = transitions
        left = pt.compose_transitions(c, pt.compose_transitions(b, a))
        right = pt.compose_transitions(pt.compose_transitions(c, b), a)
        for left_value, right_value in zip(left, right):
            torch.testing.assert_close(
                left_value, right_value, rtol=1e-12, atol=1e-12
            )

        jd, jb, ji = jnp.asarray(decay), jnp.asarray(drive), jnp.asarray(initial)
        jq = jx.rotor_from_bivector(jnp.asarray(bivectors))
        jparallel, jparallel_final = jx.rotor_affine_scan(jd, jq, jb, ji)
        jrecurrent, jrecurrent_final = jx.rotor_recurrent_scan(jd, jq, jb, ji)
        np.testing.assert_allclose(jparallel, jrecurrent, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            jparallel_final, jrecurrent_final, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            parallel.numpy(), np.asarray(jparallel), rtol=1e-12, atol=1e-12
        )

    def test_hard_bound_under_long_arbitrary_additive_inputs(self) -> None:
        torch.manual_seed(103)
        layer = pt.SelectiveRotorSSM(
            channels=2, min_half_life=2, max_half_life=64
        ).double()
        inputs = 1e4 * torch.randn(2, 4096, 2, 8, dtype=torch.float64)
        states, final_state = layer(inputs, scan_mode="recurrent")
        self.assertLessEqual(
            float(states.norm(dim=-1).max().detach()), 1 + 1e-12
        )
        self.assertLessEqual(
            float(final_state.norm(dim=-1).max().detach()), 1 + 1e-12
        )

    def test_padding_is_an_identity_transition(self) -> None:
        torch.manual_seed(104)
        layer = pt.SelectiveRotorSSM(channels=2).double()
        inputs = torch.randn(2, 12, 2, 8, dtype=torch.float64)
        valid = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1] * 12],
            dtype=torch.bool,
        )
        decay, rotors, drive = layer.transitions(inputs, valid)
        torch.testing.assert_close(decay[0, 4:], torch.ones_like(decay[0, 4:]))
        torch.testing.assert_close(drive[0, 4:], torch.zeros_like(drive[0, 4:]))
        expected_identity = pt.identity_rotor(rotors[0, 4:])
        torch.testing.assert_close(rotors[0, 4:], expected_identity)
        states, final_state = pt.rotor_affine_scan(decay, rotors, drive)
        torch.testing.assert_close(
            states[0, 4:], states[0, 3].expand_as(states[0, 4:]),
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(final_state[0], states[0, 3])

    def test_model_cache_matches_full_sequence_in_both_backends(self) -> None:
        tokens = np.arange(22, dtype=np.int64).reshape(2, 11) % 29

        torch.manual_seed(105)
        tmodel = pt.GASSMLanguageModel(
            vocab_size=29, channels=2, num_layers=2, dropout_rate=0
        ).double().eval()
        ttokens = torch.from_numpy(tokens)
        full, full_states = tmodel(ttokens, return_recurrent_states=True)
        first, cache = tmodel(ttokens[:, :6], return_recurrent_states=True)
        second, cache = tmodel(
            ttokens[:, 6:], cache, return_recurrent_states=True
        )
        torch.testing.assert_close(
            full, torch.cat((first, second), dim=1), rtol=1e-12, atol=1e-12
        )
        for expected, actual in zip(full_states, cache):
            torch.testing.assert_close(expected, actual, rtol=1e-12, atol=1e-12)

        jmodel = jx.GASSMLanguageModel(
            vocab_size=29,
            channels=2,
            num_layers=2,
            dropout_rate=0,
            dtype=jnp.float64,
        )
        jtokens = jnp.asarray(tokens)
        variables = jmodel.init(jax.random.PRNGKey(105), jtokens, training=False)
        jfull, jfull_states = jmodel.apply(
            variables, jtokens, training=False, return_recurrent_states=True
        )
        jfirst, jcache = jmodel.apply(
            variables,
            jtokens[:, :6],
            training=False,
            return_recurrent_states=True,
        )
        jsecond, jcache = jmodel.apply(
            variables,
            jtokens[:, 6:],
            jcache,
            training=False,
            return_recurrent_states=True,
        )
        np.testing.assert_allclose(
            jfull, jnp.concatenate((jfirst, jsecond), axis=1), rtol=1e-12, atol=1e-12
        )
        for expected, actual in zip(jfull_states, jcache):
            np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)

    def test_cuda_parallel_scan_forward_backward(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        torch.manual_seed(106)
        layer = pt.SelectiveRotorSSM(channels=8).cuda()
        inputs = torch.randn(4, 257, 8, 8, device="cuda", requires_grad=True)
        parallel, final_state = layer(inputs, scan_mode="parallel")
        recurrent, recurrent_final = layer(inputs, scan_mode="recurrent")
        torch.testing.assert_close(parallel, recurrent, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(
            final_state, recurrent_final, rtol=2e-4, atol=2e-5
        )
        (parallel.square().mean() + final_state.square().mean()).backward()
        self.assertTrue(bool(torch.isfinite(inputs.grad).all()))

    def test_cuda_dispatch_matches_specialized_sandwich(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        torch.manual_seed(107)
        rotors = pt.rotor_from_bivector(torch.randn(3, 19, 4, 3, device="cuda"))
        values = torch.randn(3, 19, 4, 8, device="cuda")
        selected = pt.rotor_sandwich(rotors, values)
        specialized = pt.specialized_rotor_sandwich(rotors, values)
        dense = pt.geometric_product(
            pt.geometric_product(rotors, values), pt.reversion(rotors)
        )
        torch.testing.assert_close(selected, dense, rtol=0, atol=0)
        torch.testing.assert_close(
            selected, specialized, rtol=2e-5, atol=2e-6
        )


if __name__ == "__main__":
    unittest.main()
