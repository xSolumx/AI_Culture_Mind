"""Canonical JAX/Flax geometric-algebra language-model experiment.

The numbered ``GA-SSM-*`` files are retained as historical prototypes.  New
work should import from this module because it is side-effect free: datasets,
profilers, checkpoints, and training are only activated by :func:`main`.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen import remat
from flax.training import checkpoints, train_state

from GALib import (
    GA_DIM,
    GradeLinear,
    GeometricChannelMix,
    GeometricDense,
    geometric_product,
    grade_invariants,
    reversion,
    rotor_from_bivector,
    rotor_sandwich,
)


ArrayBatch = dict[str, jax.Array]


class GeometricRMSNorm(nn.Module):
    """RMS normalization with one scalar gain per multivector channel."""

    channels: int
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        if inputs.shape[-2:] != (self.channels, GA_DIM):
            raise ValueError("unexpected multivector channel shape")
        rms = jnp.sqrt(jnp.mean(jnp.square(inputs), axis=(-2, -1), keepdims=True) + self.epsilon)
        gain = self.param("gain", nn.initializers.ones, (self.channels, 1), jnp.float32)
        return inputs / rms * gain.astype(self.dtype)


class GeometricGatedFFN(nn.Module):
    """Channel-expanding GA mixer gated only by invariant channel energies."""

    channels: int
    expansion: int = 2
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        hidden_channels = self.channels * self.expansion
        hidden = GradeLinear(
            self.channels, hidden_channels, dtype=self.dtype
        )(inputs)
        invariant_features = grade_invariants(hidden).reshape(
            *hidden.shape[:-2], hidden_channels * 4
        )
        gates = nn.sigmoid(
            nn.Dense(hidden_channels, dtype=self.dtype, param_dtype=jnp.float32)(
                invariant_features
            )
        )
        hidden = hidden * gates[..., None]
        return GradeLinear(
            hidden_channels, self.channels, dtype=self.dtype
        )(hidden)


def rotor_transition_step(
    state: jax.Array,
    decay: jax.Array,
    rotor: jax.Array,
    drive: jax.Array,
) -> jax.Array:
    """Apply one stable damped-rotor affine state transition."""
    return decay[..., None] * rotor_sandwich(rotor, state) + drive


def rotor_affine_scan(
    decay: jax.Array,
    rotors: jax.Array,
    drive: jax.Array,
    initial_state: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Parallel scan of ``h_t = d_t R_t h_(t-1) R_t~ + b_t``.

    ``decay`` has shape ``(B, L, C)`` and both multivector arrays have shape
    ``(B, L, C, 8)``.  The composed transition remains a damped rotor-affine
    map, so prefix states can be computed with an associative scan.
    """
    if rotors.shape != drive.shape or rotors.shape[:-1] != decay.shape:
        raise ValueError("decay, rotors, and drive have incompatible shapes")
    if rotors.shape[-1] != GA_DIM:
        raise ValueError(f"rotors and drive must end in {GA_DIM} components")

    def compose(left, right):
        left_decay, left_rotor, left_drive = left
        right_decay, right_rotor, right_drive = right
        composed_rotor = geometric_product(right_rotor, left_rotor)
        transported_drive = rotor_sandwich(right_rotor, left_drive)
        composed_drive = right_drive + right_decay[..., None] * transported_drive
        return right_decay * left_decay, composed_rotor, composed_drive

    cumulative_decay, cumulative_rotor, cumulative_drive = jax.lax.associative_scan(
        compose, (decay, rotors, drive), axis=1
    )
    if initial_state is None:
        states = cumulative_drive
    else:
        if initial_state.shape != drive.shape[:1] + drive.shape[2:]:
            raise ValueError("initial_state must have shape (B, C, 8)")
        carried = rotor_sandwich(cumulative_rotor, initial_state[:, None, :, :])
        states = cumulative_decay[..., None] * carried + cumulative_drive
    return states, states[:, -1]


def rotor_recurrent_scan(
    decay: jax.Array,
    rotors: jax.Array,
    drive: jax.Array,
    initial_state: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Sequential scan used for streaming and as the parallel-scan oracle."""
    if rotors.shape != drive.shape or rotors.shape[:-1] != decay.shape:
        raise ValueError("decay, rotors, and drive have incompatible shapes")
    if initial_state is None:
        initial_state = jnp.zeros_like(drive[:, 0])

    def step(state, transition):
        step_decay, step_rotor, step_drive = transition
        next_state = rotor_transition_step(
            state, step_decay, step_rotor, step_drive
        )
        return next_state, next_state

    final_state, time_major_states = jax.lax.scan(
        step,
        initial_state,
        (
            jnp.moveaxis(decay, 1, 0),
            jnp.moveaxis(rotors, 1, 0),
            jnp.moveaxis(drive, 1, 0),
        ),
    )
    return jnp.moveaxis(time_major_states, 0, 1), final_state


class SelectiveRotorSSM(nn.Module):
    """Input-selective, stable multivector state-space layer."""

    channels: int
    min_half_life: float = 4.0
    max_half_life: float = 2048.0
    minimum_step_size: float = 1e-2
    minimum_decay_rate: float = 1e-4
    max_rotor_angle: float = np.pi / 2
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        initial_state: jax.Array | None = None,
        *,
        scan_mode: str = "parallel",
    ) -> tuple[jax.Array, jax.Array]:
        batch_size, _, channels, mv_dim = inputs.shape
        if channels != self.channels or mv_dim != GA_DIM:
            raise ValueError("unexpected SelectiveRotorSSM input shape")

        invariants = grade_invariants(inputs).reshape(
            batch_size, inputs.shape[1], self.channels * 4
        )
        step_control = nn.Dense(
            self.channels,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(invariants)
        step_size = self.minimum_step_size + nn.softplus(step_control)

        if self.min_half_life <= 0 or self.max_half_life < self.min_half_life:
            raise ValueError("half-life bounds must be positive and ordered")
        if self.minimum_step_size <= 0 or self.minimum_decay_rate <= 0:
            raise ValueError("step-size and decay-rate floors must be positive")
        expected_initial_step = self.minimum_step_size + np.log(2.0)
        slowest_initial_rate = np.log(2.0) / (
            self.max_half_life * expected_initial_step
        )
        if self.minimum_decay_rate >= slowest_initial_rate:
            raise ValueError(
                "minimum_decay_rate is too large for the requested half-lives"
            )

        def rate_initializer(key, shape, dtype):
            half_lives = jnp.exp(
                jnp.linspace(
                    jnp.log(self.min_half_life),
                    jnp.log(self.max_half_life),
                    shape[0],
                    dtype=dtype,
                )
            )
            expected_step = self.minimum_step_size + nn.softplus(
                jnp.asarray(0.0, dtype)
            )
            target_rates = jnp.log(2.0) / (half_lives * expected_step)
            free_rates = target_rates - self.minimum_decay_rate
            return jnp.log(jnp.expm1(free_rates))

        log_rates = self.param(
            "log_rates",
            rate_initializer,
            (self.channels,),
            jnp.float32,
        )
        rates = self.minimum_decay_rate + nn.softplus(log_rates).astype(self.dtype)
        decay = jnp.exp(-step_size * rates)

        rotor_source = GradeLinear(
            self.channels, self.channels, use_bias=False, dtype=self.dtype
        )(inputs)
        rotor_strength = jnp.tanh(
            nn.Dense(
                self.channels,
                dtype=self.dtype,
                param_dtype=jnp.float32,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(invariants)
        )
        rotors = rotor_from_bivector(
            rotor_source[..., 4:7] * rotor_strength[..., None],
            self.max_rotor_angle,
        )

        projected_inputs = GradeLinear(
            self.channels, self.channels, dtype=self.dtype
        )(inputs)
        # Compute 1-exp(-2 Delta lambda) without subtracting two nearly equal
        # floating-point values when the learned half-life is long.
        injection_variance = -jnp.expm1(-2.0 * step_size * rates)
        injection = jnp.sqrt(
            jnp.maximum(injection_variance, jnp.finfo(self.dtype).tiny)
        )
        drive = injection[..., None] * projected_inputs
        if scan_mode == "parallel":
            return rotor_affine_scan(decay, rotors, drive, initial_state)
        if scan_mode == "recurrent":
            return rotor_recurrent_scan(decay, rotors, drive, initial_state)
        raise ValueError("scan_mode must be 'parallel' or 'recurrent'")


class GASSMBlock(nn.Module):
    channels: int
    expansion: int = 2
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        initial_state: jax.Array | None = None,
        *,
        training: bool,
        scan_mode: str = "parallel",
    ) -> tuple[jax.Array, jax.Array]:
        normalized = GeometricRMSNorm(self.channels, dtype=self.dtype)(inputs)
        sequence, final_state = SelectiveRotorSSM(
            self.channels, dtype=self.dtype
        )(normalized, initial_state, scan_mode=scan_mode)
        sequence = nn.Dropout(self.dropout_rate)(sequence, deterministic=not training)
        outputs = inputs + sequence

        normalized = GeometricRMSNorm(self.channels, dtype=self.dtype)(outputs)
        feed_forward = GeometricGatedFFN(
            self.channels, self.expansion, self.dtype
        )(normalized)
        feed_forward = nn.Dropout(self.dropout_rate)(
            feed_forward, deterministic=not training
        )
        return outputs + feed_forward, final_state


class GASSMLanguageModel(nn.Module):
    """Language model built around selective damped-rotor state transitions."""

    vocab_size: int
    channels: int = 8
    num_layers: int = 4
    expansion: int = 2
    max_len: int = 512
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        token_ids: jax.Array,
        recurrent_states: Sequence[jax.Array] | None = None,
        *,
        training: bool,
        return_recurrent_states: bool = False,
        scan_mode: str = "parallel",
    ) -> jax.Array | tuple[jax.Array, tuple[jax.Array, ...]]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if recurrent_states is None:
            layer_states: tuple[jax.Array | None, ...] = (None,) * self.num_layers
        else:
            if len(recurrent_states) != self.num_layers:
                raise ValueError("one recurrent state is required per model layer")
            layer_states = tuple(recurrent_states)

        embeddings = self.param(
            "token_embeddings",
            nn.initializers.normal(stddev=0.02),
            (self.vocab_size, self.channels, GA_DIM),
            jnp.float32,
        ).astype(self.dtype)
        outputs = embeddings[token_ids]
        outputs = nn.Dropout(self.dropout_rate)(outputs, deterministic=not training)
        final_states = []
        for layer_index in range(self.num_layers):
            outputs, final_state = GASSMBlock(
                self.channels, self.expansion, self.dropout_rate, self.dtype
            )(
                outputs,
                layer_states[layer_index],
                training=training,
                scan_mode=scan_mode,
            )
            final_states.append(final_state)
        outputs = GeometricRMSNorm(self.channels, dtype=self.dtype)(outputs)

        # scalar(x * reverse(embedding)) equals the coefficient inner product in
        # Euclidean GA, allowing an efficient, weight-tied geometric decoder.
        logits = jnp.einsum("blci,vci->blv", outputs, embeddings)
        logits /= jnp.sqrt(jnp.asarray(self.channels * GA_DIM, self.dtype))
        vocabulary_bias = self.param(
            "vocabulary_bias", nn.initializers.zeros, (self.vocab_size,), jnp.float32
        )
        logits = (logits + vocabulary_bias).astype(jnp.float32)
        if return_recurrent_states:
            return logits, tuple(final_states)
        return logits


def initialize_recurrent_states(
    model: GASSMLanguageModel,
    batch_size: int,
) -> tuple[jax.Array, ...]:
    """Create zero-valued per-layer states for streaming inference."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return tuple(
        jnp.zeros((batch_size, model.channels, GA_DIM), dtype=model.dtype)
        for _ in range(model.num_layers)
    )


def data_generator(
    token_array: Sequence[int] | np.ndarray,
    seq_len: int,
    batch_size: int,
    key: jax.Array,
    *,
    shuffle: bool = True,
) -> Iterator[ArrayBatch]:
    """Yield deterministic, non-overlapping next-token batches."""
    if seq_len < 1 or batch_size < 1:
        raise ValueError("seq_len and batch_size must be positive")

    tokens = np.asarray(token_array, dtype=np.int32)
    num_sequences = (tokens.size - 1) // seq_len
    if num_sequences < batch_size:
        return

    usable = num_sequences * seq_len
    inputs = tokens[:usable].reshape(num_sequences, seq_len)
    targets = tokens[1 : usable + 1].reshape(num_sequences, seq_len)
    order = np.arange(num_sequences)
    if shuffle:
        order = np.asarray(jax.random.permutation(key, num_sequences))

    for offset in range(0, num_sequences - batch_size + 1, batch_size):
        indices = order[offset : offset + batch_size]
        yield {
            "inputs": jnp.asarray(inputs[indices]),
            "targets": jnp.asarray(targets[indices]),
        }


class PositionalRotors(nn.Module):
    """Learn a fixed-size table of unit rotors and slice it per sequence."""

    max_len: int
    num_heads: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, sequence_length: int) -> jax.Array:
        if sequence_length > self.max_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_len {self.max_len}"
            )
        params = self.param(
            "rotor_params",
            nn.initializers.normal(stddev=0.1),
            (self.max_len, self.num_heads, 4),
            jnp.float32,
        )[:sequence_length].astype(self.dtype)
        params /= jnp.maximum(jnp.linalg.norm(params, axis=-1, keepdims=True), 1e-6)
        scalar, e12, e13, e23 = jnp.split(params, 4, axis=-1)
        zeros = jnp.zeros_like(scalar)
        return jnp.concatenate(
            [scalar, zeros, zeros, zeros, e12, e13, e23, zeros], axis=-1
        )


class GeometricAttention(nn.Module):
    """Causal multi-head attention with GA query/key similarity."""

    num_heads: int
    max_len: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        batch_size, length, mv_dim = inputs.shape
        if mv_dim != GA_DIM:
            raise ValueError(f"inputs must contain {GA_DIM}-component multivectors")

        rotors = PositionalRotors(self.max_len, self.num_heads, self.dtype)(length)
        rotors = rotors[None, ...]
        expanded = inputs[:, :, None, :]
        rotated = geometric_product(rotors, expanded)
        rotated = geometric_product(rotated, reversion(rotors))

        qkv = GeometricDense(features=3, dtype=self.dtype)(rotated)
        queries, keys, values = jnp.moveaxis(qkv, -2, 0)
        queries = jnp.transpose(queries, (0, 2, 1, 3))
        keys = jnp.transpose(keys, (0, 2, 1, 3))
        values = jnp.transpose(values, (0, 2, 1, 3))

        products = geometric_product(
            queries[:, :, :, None, :],
            reversion(keys)[:, :, None, :, :],
        )
        scores = products[..., 0] / jnp.sqrt(jnp.asarray(GA_DIM, self.dtype))
        causal_mask = jnp.tril(jnp.ones((1, 1, length, length), dtype=bool))
        scores = jnp.where(causal_mask, scores, jnp.finfo(self.dtype).min)
        weights = nn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, values)
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        attended = attended.reshape(batch_size, length, self.num_heads * GA_DIM)
        return nn.Dense(GA_DIM, dtype=self.dtype, param_dtype=jnp.float32)(attended)


class GeometricFFN(nn.Module):
    """A compact operator-bank feed-forward block."""

    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        hidden = GeometricDense(self.features, activation=nn.gelu, dtype=self.dtype)(inputs)
        pooled = hidden.mean(axis=-2)
        return GeometricDense(1, dtype=self.dtype)(pooled).squeeze(-2)


class RotorResidual(nn.Module):
    """Apply a learned token-wise rotor followed by a residual connection."""

    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        params = nn.Dense(4, dtype=self.dtype, param_dtype=jnp.float32)(inputs)
        params /= jnp.maximum(jnp.linalg.norm(params, axis=-1, keepdims=True), 1e-6)
        scalar, e12, e13, e23 = jnp.split(params, 4, axis=-1)
        zeros = jnp.zeros_like(scalar)
        rotors = jnp.concatenate(
            [scalar, zeros, zeros, zeros, e12, e13, e23, zeros], axis=-1
        )
        rotated = geometric_product(rotors, inputs)
        rotated = geometric_product(rotated, reversion(rotors))
        return inputs + rotated


class GeometricTransformerBlock(nn.Module):
    num_heads: int
    ffn_features: int
    max_len: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = True

    @nn.compact
    def __call__(self, inputs: jax.Array, *, training: bool) -> jax.Array:
        attention_cls = remat(GeometricAttention) if self.gradient_checkpointing else GeometricAttention
        ffn_cls = remat(GeometricFFN) if self.gradient_checkpointing else GeometricFFN

        attention = attention_cls(self.num_heads, self.max_len, self.dtype)(inputs)
        attention = nn.Dropout(self.dropout_rate)(attention, deterministic=not training)
        outputs = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(inputs + attention)

        feed_forward = ffn_cls(self.ffn_features, self.dtype)(outputs)
        feed_forward = nn.Dropout(self.dropout_rate)(feed_forward, deterministic=not training)
        return nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(
            outputs + feed_forward
        )


class GATransformerLM(nn.Module):
    vocab_size: int
    num_layers: int = 2
    num_heads: int = 4
    ffn_features: int = 64
    max_len: int = 128
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = True

    @nn.compact
    def __call__(self, token_ids: jax.Array, *, training: bool) -> jax.Array:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if token_ids.shape[1] > self.max_len:
            raise ValueError("input sequence exceeds the configured max_len")

        outputs = nn.Embed(
            num_embeddings=self.vocab_size,
            features=GA_DIM,
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(token_ids)
        outputs = nn.Dropout(self.dropout_rate)(outputs, deterministic=not training)
        outputs = RotorResidual(self.dtype)(outputs)
        for _ in range(self.num_layers):
            outputs = GeometricTransformerBlock(
                self.num_heads,
                self.ffn_features,
                self.max_len,
                self.dropout_rate,
                self.dtype,
                self.gradient_checkpointing,
            )(outputs, training=training)

        vocabulary_multivectors = GeometricDense(
            self.vocab_size, use_bias=False, dtype=self.dtype
        )(outputs)
        return vocabulary_multivectors[..., 0].astype(jnp.float32)


def create_train_state(
    model: nn.Module,
    key: jax.Array,
    learning_rate: float | optax.Schedule,
    batch_shape: tuple[int, int],
) -> train_state.TrainState:
    variables = model.init(
        {"params": key, "dropout": key},
        jnp.ones(batch_shape, dtype=jnp.int32),
        training=False,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=optimizer
    )


@jax.jit
def train_step(
    state: train_state.TrainState, batch: ArrayBatch, dropout_key: jax.Array
) -> tuple[train_state.TrainState, jax.Array]:
    def loss_fn(params: dict) -> jax.Array:
        logits = state.apply_fn(
            {"params": params},
            batch["inputs"],
            training=True,
            rngs={"dropout": dropout_key},
        )
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["targets"]
        ).mean()

    loss, gradients = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=gradients), loss


@jax.jit
def eval_step(state: train_state.TrainState, batch: ArrayBatch) -> jax.Array:
    logits = state.apply_fn(
        {"params": state.params}, batch["inputs"], training=False
    )
    return optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["targets"]
    ).mean()


def evaluate_model(
    state: train_state.TrainState,
    batches: Iterator[ArrayBatch],
    limit: int | None = None,
) -> float:
    losses: list[float] = []
    for index, batch in enumerate(batches):
        if limit is not None and index >= limit:
            break
        losses.append(float(eval_step(state, batch)))
    if not losses:
        raise ValueError("evaluation produced no complete batches")
    return float(np.mean(losses))


def sample_text(
    state: train_state.TrainState,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    seed: int = 42,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    generated = list(tokenizer.encode(prompt))
    if not generated:
        raise ValueError("prompt must encode to at least one token")
    key = jax.random.PRNGKey(seed)
    vocabulary_size = int(tokenizer.n_vocab)
    effective_top_k = min(top_k, vocabulary_size)

    prompt_tokens = jnp.asarray(generated, dtype=jnp.int32)[None, :]
    logits, recurrent_states = state.apply_fn(
        {"params": state.params},
        prompt_tokens,
        training=False,
        return_recurrent_states=True,
        scan_mode="parallel",
    )

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :] / temperature
        values, indices = jax.lax.top_k(next_logits, effective_top_k)
        key, sample_key = jax.random.split(key)
        selected = jax.random.categorical(sample_key, values)
        next_token = int(indices[0, selected[0]])
        generated.append(next_token)
        logits, recurrent_states = state.apply_fn(
            {"params": state.params},
            jnp.asarray([[next_token]], dtype=jnp.int32),
            recurrent_states,
            training=False,
            return_recurrent_states=True,
            scan_mode="recurrent",
        )
    return tokenizer.decode(generated)


@dataclass(frozen=True)
class TrainingConfig:
    seq_len: int = 64
    batch_size: int = 32
    epochs: int = 10
    warmup_steps: int = 500
    peak_learning_rate: float = 3e-4
    end_learning_rate: float = 3e-5
    checkpoint_dir: Path = Path(__file__).resolve().parent / "ga_transformer_checkpoints"
    seed: int = 0
    channels: int = 8
    num_layers: int = 4
    expansion: int = 2


def train(config: TrainingConfig) -> train_state.TrainState:
    """Load WikiText-2 and run the configured training experiment."""
    import tiktoken
    from datasets import load_dataset

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    separator = "<|endoftext|>"

    def encode_split(name: str) -> np.ndarray:
        text = f" {separator} ".join(
            row["text"] for row in dataset[name] if row["text"].strip()
        )
        return np.asarray(
            tokenizer.encode(text, allowed_special={separator}), dtype=np.int32
        )

    train_tokens = encode_split("train")
    validation_tokens = encode_split("validation")
    batches_per_epoch = (train_tokens.size - 1) // (
        config.seq_len * config.batch_size
    )
    if batches_per_epoch < 1:
        raise ValueError("training split is too small for one complete batch")
    total_steps = config.epochs * batches_per_epoch
    warmup_steps = min(config.warmup_steps, max(total_steps - 1, 0))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.peak_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, 1),
        end_value=config.end_learning_rate,
    )

    model = GASSMLanguageModel(
        vocab_size=tokenizer.n_vocab,
        channels=config.channels,
        num_layers=config.num_layers,
        expansion=config.expansion,
        max_len=config.seq_len,
    )
    key = jax.random.PRNGKey(config.seed)
    key, init_key, data_key, dropout_key = jax.random.split(key, 4)
    state = create_train_state(
        model, init_key, schedule, (config.batch_size, config.seq_len)
    )
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = checkpoints.restore_checkpoint(config.checkpoint_dir, target=state)

    for epoch in range(config.epochs):
        epoch_key = jax.random.fold_in(data_key, epoch)
        batches = data_generator(
            train_tokens, config.seq_len, config.batch_size, epoch_key
        )
        for batch_index, batch in enumerate(batches, start=1):
            dropout_key, step_key = jax.random.split(dropout_key)
            state, loss = train_step(state, batch, step_key)
            if int(state.step) % 100 == 0:
                print(
                    f"epoch={epoch + 1} batch={batch_index}/{batches_per_epoch} "
                    f"step={int(state.step)} loss={float(loss):.4f}"
                )

        validation_batches = data_generator(
            validation_tokens,
            config.seq_len,
            config.batch_size,
            jax.random.PRNGKey(0),
            shuffle=False,
        )
        validation_loss = evaluate_model(state, validation_batches)
        print(f"epoch={epoch + 1} validation_loss={validation_loss:.4f}")
        checkpoints.save_checkpoint(
            config.checkpoint_dir,
            state,
            step=int(state.step),
            keep=3,
            overwrite=True,
        )

    print(
        sample_text(
            state,
            tokenizer,
            "Artificial intelligence is",
        )
    )
    return state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--profile-port", type=int)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.profile_port is not None:
        jax.profiler.start_server(args.profile_port)
    train(
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            channels=args.channels,
            num_layers=args.layers,
        )
    )


if __name__ == "__main__":
    main()
