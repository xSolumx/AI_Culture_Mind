"""Fast correctness and smoke tests for the canonical JAX implementation."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from GALib import (
    GA_DIM,
    GradeLinear,
    geometric_product,
    normalized_rotor,
    reversion,
    rotor_from_bivector,
    rotor_sandwich,
)
from ga_ssm import (
    GASSMBlock,
    GASSMLanguageModel,
    GATransformerLM,
    create_train_state,
    data_generator,
    rotor_affine_scan,
    rotor_recurrent_scan,
    train_step,
)


def basis(index: int) -> jax.Array:
    return jax.nn.one_hot(index, GA_DIM)


class GeometricAlgebraTests(unittest.TestCase):
    def test_basis_identities(self) -> None:
        one, e1, e2, e3 = (basis(index) for index in range(4))
        e12, e13, e23, e123 = (basis(index) for index in range(4, 8))

        np.testing.assert_allclose(geometric_product(one, e1), e1)
        np.testing.assert_allclose(geometric_product(e1, e1), one)
        np.testing.assert_allclose(geometric_product(e1, e2), e12)
        np.testing.assert_allclose(geometric_product(e2, e1), -e12)
        np.testing.assert_allclose(geometric_product(e1, e3), e13)
        np.testing.assert_allclose(geometric_product(e2, e3), e23)
        np.testing.assert_allclose(geometric_product(e12, e3), e123)

    def test_reversion_reverses_products(self) -> None:
        left = jnp.arange(1, 9, dtype=jnp.float32)
        right = jnp.arange(8, 0, -1, dtype=jnp.float32)
        expected = geometric_product(reversion(right), reversion(left))
        actual = reversion(geometric_product(left, right))
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_batched_product_uses_last_axis(self) -> None:
        values = jnp.stack([basis(1), basis(2)])
        result = geometric_product(values, basis(1))
        self.assertEqual(result.shape, (2, GA_DIM))

    def test_bivector_exponential_produces_unit_rotors(self) -> None:
        bivectors = jax.random.normal(jax.random.PRNGKey(4), (2, 5, 3))
        rotors = rotor_from_bivector(bivectors)
        products = geometric_product(rotors, reversion(rotors))
        expected = jnp.broadcast_to(basis(0), products.shape)
        np.testing.assert_allclose(products, expected, rtol=1e-5, atol=1e-5)

    def test_grade_linear_commutes_with_rotor_action(self) -> None:
        inputs = jax.random.normal(jax.random.PRNGKey(5), (2, 4, 3, GA_DIM))
        frame_rotor = rotor_from_bivector(jnp.asarray([0.3, -0.2, 0.1]))
        transformed_inputs = rotor_sandwich(frame_rotor, inputs)
        layer = GradeLinear(in_channels=3, out_channels=5)
        parameters = layer.init(jax.random.PRNGKey(6), inputs)
        outputs = layer.apply(parameters, inputs)
        transformed_outputs = layer.apply(parameters, transformed_inputs)
        expected = rotor_sandwich(frame_rotor, outputs)
        np.testing.assert_allclose(
            transformed_outputs, expected, rtol=2e-5, atol=2e-5
        )


class ModelSmokeTests(unittest.TestCase):
    def test_forward_and_training_step(self) -> None:
        model = GATransformerLM(
            vocab_size=16,
            num_layers=1,
            num_heads=2,
            ffn_features=4,
            max_len=6,
            dropout_rate=0.0,
            gradient_checkpointing=False,
        )
        state = create_train_state(model, jax.random.PRNGKey(0), 1e-3, (2, 6))
        batch = {
            "inputs": jnp.arange(12, dtype=jnp.int32).reshape(2, 6) % 16,
            "targets": (jnp.arange(12, dtype=jnp.int32).reshape(2, 6) + 1) % 16,
        }
        logits = state.apply_fn({"params": state.params}, batch["inputs"], training=False)
        self.assertEqual(logits.shape, (2, 6, 16))

        next_state, loss = train_step(state, batch, jax.random.PRNGKey(1))
        self.assertEqual(int(next_state.step), 1)
        self.assertTrue(np.isfinite(float(loss)))

    def test_data_generator_aligns_next_tokens(self) -> None:
        batches = list(
            data_generator(
                np.arange(17), 4, 2, jax.random.PRNGKey(0), shuffle=False
            )
        )
        self.assertEqual(len(batches), 2)
        np.testing.assert_array_equal(
            np.asarray(batches[0]["targets"]),
            np.asarray(batches[0]["inputs"]) + 1,
        )

    def test_rotor_affine_parallel_scan_matches_recurrence(self) -> None:
        key = jax.random.PRNGKey(7)
        decay = jax.random.uniform(key, (2, 9, 3), minval=0.7, maxval=0.99)
        rotor_parameters = jax.random.normal(jax.random.PRNGKey(8), (2, 9, 3, 4))
        rotors = normalized_rotor(rotor_parameters)
        drive = jax.random.normal(jax.random.PRNGKey(9), (2, 9, 3, GA_DIM))
        initial = jax.random.normal(jax.random.PRNGKey(10), (2, 3, GA_DIM))

        parallel, final_state = rotor_affine_scan(decay, rotors, drive, initial)
        state = initial
        sequential = []
        for index in range(decay.shape[1]):
            state = (
                decay[:, index, :, None]
                * rotor_sandwich(rotors[:, index], state)
                + drive[:, index]
            )
            sequential.append(state)
        sequential = jnp.stack(sequential, axis=1)
        np.testing.assert_allclose(parallel, sequential, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(final_state, sequential[:, -1], rtol=2e-5, atol=2e-5)

        recurrent, recurrent_final = rotor_recurrent_scan(
            decay, rotors, drive, initial
        )
        np.testing.assert_allclose(parallel, recurrent, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            final_state, recurrent_final, rtol=2e-5, atol=2e-5
        )

    def test_long_parallel_scan_remains_close_to_recurrence(self) -> None:
        length = 2048
        keys = jax.random.split(jax.random.PRNGKey(11), 4)
        decay = jax.random.uniform(
            keys[0], (1, length, 4), minval=0.995, maxval=0.9999
        )
        rotors = rotor_from_bivector(
            0.02 * jax.random.normal(keys[1], (1, length, 4, 3))
        )
        drive = 0.01 * jax.random.normal(keys[2], (1, length, 4, GA_DIM))
        initial = jax.random.normal(keys[3], (1, 4, GA_DIM))

        parallel, parallel_final = rotor_affine_scan(
            decay, rotors, drive, initial
        )
        recurrent, recurrent_final = rotor_recurrent_scan(
            decay, rotors, drive, initial
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(parallel))))
        np.testing.assert_allclose(parallel, recurrent, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            parallel_final, recurrent_final, rtol=1e-4, atol=1e-5
        )

    def test_ssm_language_model_is_causal_and_trainable(self) -> None:
        model = GASSMLanguageModel(
            vocab_size=16,
            channels=2,
            num_layers=1,
            expansion=2,
            max_len=12,
            dropout_rate=0.0,
        )
        state = create_train_state(model, jax.random.PRNGKey(20), 1e-3, (2, 8))
        first = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]])
        second = first.at[:, 5:].set(jnp.asarray([[10, 11, 12], [11, 12, 13]]))
        first_logits = state.apply_fn({"params": state.params}, first, training=False)
        second_logits = state.apply_fn({"params": state.params}, second, training=False)
        self.assertEqual(first_logits.shape, (2, 8, 16))
        np.testing.assert_allclose(first_logits[:, :5], second_logits[:, :5], rtol=1e-5, atol=1e-5)

        batch = {"inputs": first, "targets": (first + 1) % 16}
        next_state, loss = train_step(state, batch, jax.random.PRNGKey(21))
        self.assertEqual(int(next_state.step), 1)
        self.assertTrue(np.isfinite(float(loss)))

    def test_block_is_spin3_equivariant(self) -> None:
        inputs = jax.random.normal(jax.random.PRNGKey(30), (2, 7, 2, GA_DIM))
        initial = jax.random.normal(jax.random.PRNGKey(31), (2, 2, GA_DIM))
        frame_rotor = rotor_from_bivector(jnp.asarray([0.2, 0.4, -0.1]))
        block = GASSMBlock(channels=2, expansion=2, dropout_rate=0.0)
        parameters = block.init(
            jax.random.PRNGKey(32), inputs, initial, training=False
        )
        outputs, final_state = block.apply(
            parameters, inputs, initial, training=False
        )
        transformed_outputs, transformed_final = block.apply(
            parameters,
            rotor_sandwich(frame_rotor, inputs),
            rotor_sandwich(frame_rotor, initial),
            training=False,
        )
        np.testing.assert_allclose(
            transformed_outputs,
            rotor_sandwich(frame_rotor, outputs),
            rtol=5e-5,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            transformed_final,
            rotor_sandwich(frame_rotor, final_state),
            rtol=5e-5,
            atol=5e-5,
        )

    def test_full_parallel_chunked_and_token_streaming_are_equivalent(self) -> None:
        model = GASSMLanguageModel(
            vocab_size=24,
            channels=3,
            num_layers=2,
            expansion=2,
            dropout_rate=0.0,
        )
        tokens = jnp.arange(18, dtype=jnp.int32).reshape(2, 9) % 24
        state = create_train_state(model, jax.random.PRNGKey(40), 1e-3, (2, 9))

        full_logits, full_states = state.apply_fn(
            {"params": state.params},
            tokens,
            training=False,
            return_recurrent_states=True,
            scan_mode="parallel",
        )

        first_logits, chunk_states = state.apply_fn(
            {"params": state.params},
            tokens[:, :4],
            training=False,
            return_recurrent_states=True,
            scan_mode="parallel",
        )
        second_logits, chunk_states = state.apply_fn(
            {"params": state.params},
            tokens[:, 4:],
            chunk_states,
            training=False,
            return_recurrent_states=True,
            scan_mode="parallel",
        )
        chunked_logits = jnp.concatenate([first_logits, second_logits], axis=1)

        streaming_states = None
        streaming_logits = []
        for position in range(tokens.shape[1]):
            step_logits, streaming_states = state.apply_fn(
                {"params": state.params},
                tokens[:, position : position + 1],
                streaming_states,
                training=False,
                return_recurrent_states=True,
                scan_mode="recurrent",
            )
            streaming_logits.append(step_logits)
        streaming_logits = jnp.concatenate(streaming_logits, axis=1)

        np.testing.assert_allclose(full_logits, chunked_logits, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(full_logits, streaming_logits, rtol=1e-4, atol=1e-4)
        for expected, chunked, streamed in zip(
            full_states, chunk_states, streaming_states
        ):
            np.testing.assert_allclose(expected, chunked, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(expected, streamed, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
