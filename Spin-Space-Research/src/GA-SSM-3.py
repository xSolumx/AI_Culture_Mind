import os
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tiktoken
from datasets import load_dataset
from flax.training import checkpoints, train_state
import jax.random as random
import jax.profiler

# THE MOST ROBUST IMPORT FOR CHECKPOINTING ACROSS JAX VERSIONS:
# Corrected import for remat policy in Flax
from flax.linen import remat  # Use flax.linen.remat directly

# Set environment variable for bfloat16 mixed precision policy for dot products
# This often helps in ensuring dot products are done in bfloat16 when using mixed precision.
# os.environ['XLA_FLAGS'] = '--xla_cpu_enable_bfloat16_dot_with_int8_by_default=true'


# === GA(3,0) Algebra Primitives ===
# These operations are inherently element-wise or simple arithmetic,
# JAX's JIT will optimize them. We'll rely on global dtype policy or
# explicit casting for bfloat16 computation.


@jax.jit
def geometric_product_ga3(a, b):
    # a,b: (..., 8)
    s1, x1, y1, z1, xy1, xz1, yz1, xyz1 = jnp.split(a, 8, axis=-1)
    s2, x2, y2, z2, xy2, xz2, yz2, xyz2 = jnp.split(b, 8, axis=-1)

    s = (
        s1 * s2
        + x1 * x2
        + y1 * y2
        + z1 * z2
        - xy1 * xy2
        - xz1 * xz2
        - yz1 * yz2
        - xyz1 * xyz2
    )
    x = (
        s1 * x2
        + x1 * s2
        - y1 * xy2
        + xy1 * y2
        - z1 * xz2
        + xz1 * z2
        - yz1 * xyz2
        - xyz1 * yz2
    )
    y = (
        s1 * y2
        + y1 * s2
        + x1 * xy2
        - xy1 * x2
        - z1 * yz2
        + yz1 * z2
        + xz1 * xyz2
        + xyz1 * xz2
    )
    z = (
        s1 * z2
        + z1 * s2
        + x1 * xz2
        - xz1 * x2
        + y1 * yz2
        - yz1 * y2
        - xy1 * xyz2
        - xyz1 * xy2
    )
    xy = (
        s1 * xy2
        + xy1 * s2
        + x1 * y2
        - y1 * x2
        + z1 * xyz2
        + xyz1 * z2
        + xz1 * yz2
        - yz1 * xz2
    )
    xz = (
        s1 * xz2
        + xz1 * s2
        + x1 * z2
        - z1 * x2
        - y1 * xyz2
        - xyz1 * y2
        - xy1 * yz2
        + yz1 * xy2
    )
    yz = (
        s1 * yz2
        + yz1 * s2
        + y1 * z2
        - z1 * y2
        + x1 * xyz2
        + xyz1 * x2
        + xy1 * xz2
        - xz1 * xy2
    )
    xyz = (
        s1 * xyz2
        + xyz1 * s2
        + x1 * yz2
        - y1 * xz2
        + z1 * xy2
        + xy1 * z2
        - xz1 * y2
        + yz1 * x2
    )
    return jnp.concatenate([s, x, y, z, xy, xz, yz, xyz], axis=-1)


@jax.jit
def reversion(mv):
    signs = jnp.array(
        [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], dtype=mv.dtype
    )  # Ensure signs match input dtype
    return mv * signs


# === Tokenizer and Data Utils ===


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def pack_token_sequence(token_array, seq_len, batch_size):
    num_sequences = len(token_array) // seq_len
    trimmed = token_array[: num_sequences * seq_len]
    reshaped = trimmed.reshape((num_sequences, seq_len))
    num_batches = num_sequences // batch_size
    batched = reshaped[: num_batches * batch_size]
    return batched.reshape((num_batches, batch_size, seq_len))


# --- Efficient Data Generator ---
def data_generator(token_array, seq_len, batch_size, is_training=True):
    num_sequences = len(token_array) // seq_len
    trimmed = token_array[: num_sequences * seq_len]
    reshaped = trimmed.reshape((num_sequences, seq_len))
    num_batches = num_sequences // batch_size
    # Shuffle for training data
    if is_training:
        idx = np.arange(num_batches * batch_size)
        np.random.shuffle(idx)
        reshaped = reshaped[idx].reshape(num_batches, batch_size, seq_len)

    for i in range(num_batches):
        b = reshaped[i]  # Already batched
        # Targets are simply inputs shifted by one position
        inputs = b[:, None, :]  # Add T=1 dim
        targets = np.roll(b, -1, axis=-1)[:, None, :]  # Add T=1 dim
        yield {"inputs": jnp.array(inputs), "targets": jnp.array(targets)}


# --- Geometric Dense Layer ---


class GeometricDense(nn.Module):
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32  # Added dtype argument

    @nn.compact
    def __call__(self, x):
        # x: (..., H, 8)
        *leading_dims, H, _ = x.shape
        kernel = self.param(
            "kernel", nn.initializers.lecun_normal(), (self.features, 8), self.dtype
        )

        # vmapped geometric product: kernel (features,8), x (...,H,8)
        def apply_kernel(w):
            # w: (8,), broadcast geometric product over x's leading dims + H
            # Ensure operations are done in the desired dtype
            return geometric_product_ga3(
                w.astype(x.dtype), x
            )  # Cast kernel weight to input dtype

        # Map over features dim
        y = jax.vmap(apply_kernel)(kernel)  # (features, ..., H, 8)

        # Rearrange axes: currently (features, ..., H, 8) to (..., H, features, 8)
        n_leading = len(leading_dims)
        axes_order = list(range(1, n_leading + 1)) + [n_leading + 1, 0, n_leading + 2]
        y = jnp.transpose(y, axes_order)

        if self.use_bias:
            bias = self.param(
                "bias", nn.initializers.zeros, (self.features, 8), self.dtype
            )
            # Broadcast bias over leading dims and H
            bias = bias.reshape((1,) * (len(y.shape) - 3) + bias.shape)
            y = y + bias

        return y  # (..., H, features, 8)


# --- Geometric Feed Forward ---


def mv_relu(mv):
    return nn.relu(mv)


class GeometricFFN(nn.Module):
    ffn_dim: int
    dtype: jnp.dtype = jnp.float32  # Added dtype argument

    @nn.compact
    def __call__(self, x):
        # x: (..., 8)
        hidden = GeometricDense(features=self.ffn_dim, dtype=self.dtype)(
            x
        )  # (..., ffn_dim, 8)
        hidden_act = mv_relu(hidden)
        mean_hidden = jnp.mean(hidden_act, axis=-2)
        output_mv = GeometricDense(features=1, dtype=self.dtype)(mean_hidden).squeeze(
            -2
        )
        return output_mv


# --- GA Rotor positional encoding (learned multi-head rotors) ---


class PositionalRotors(nn.Module):
    max_len: int
    num_heads: int
    dtype: jnp.dtype = jnp.float32  # Added dtype argument

    @nn.compact
    def __call__(self):
        # Learn rotor params: (max_len, num_heads, 4) = scalar + b12 + b13 + b23
        rotor_params = self.param(
            "pos_rotor_params",
            nn.initializers.normal(stddev=0.1),
            (self.max_len, self.num_heads, 4),
            self.dtype,  # Initialize params in desired dtype
        )
        # Normalize to unit rotor
        norm = jnp.linalg.norm(rotor_params, axis=-1, keepdims=True) + 1e-8
        rotor_params = rotor_params / norm

        s = rotor_params[..., 0:1]  # scalar part
        b12 = rotor_params[..., 1:2]
        b13 = rotor_params[..., 2:3]
        b23 = rotor_params[..., 3:4]

        zeros = jnp.zeros_like(s)

        # Compose full rotor multivector: [s,0,0,0,b12,b13,b23,0]
        rotors = jnp.concatenate(
            [s, zeros, zeros, zeros, b12, b13, b23, zeros], axis=-1
        )
        return rotors  # (max_len, num_heads, 8)


# --- Geometric Attention with positional rotors applied ---


class GeometricAttention(nn.Module):
    num_heads: int
    dtype: jnp.dtype = jnp.float32  # Added dtype argument

    @nn.compact
    def __call__(self, x):
        # x: (B, T, L, 8)
        B, T, L, mv_dim = x.shape
        head_dim = mv_dim  # 8

        # Get positional rotors: (L, H, 8)
        pos_rotors = PositionalRotors(
            max_len=L, num_heads=self.num_heads, dtype=self.dtype
        )()
        R = pos_rotors[None, None, :, :, :]  # (1,1,L,H,8)
        R_inv = reversion(R)

        x_exp = jnp.expand_dims(x, axis=3)  # (B,T,L,1,8)
        x_exp = jnp.broadcast_to(x_exp, (B, T, L, self.num_heads, 8))  # (B,T,L,H,8)

        # Apply sandwich product for positional encoding
        x_rot = geometric_product_ga3(R, x_exp)
        x_rot = geometric_product_ga3(x_rot, R_inv)  # (B,T,L,H,8)

        # Project QKV: features=3 (Q,K,V), keep heads dim separate
        qkv = GeometricDense(features=3, dtype=self.dtype)(x_rot)  # (B,T,L,H,3,8)

        # Split Q,K,V along features dim (axis=-2)
        q, k, v = jnp.split(qkv, 3, axis=-2)  # each (B,T,L,H,1,8)

        # Remove singleton features dim
        q = jnp.squeeze(q, axis=-2)  # (B,T,L,H,8)
        k = jnp.squeeze(k, axis=-2)
        v = jnp.squeeze(v, axis=-2)

        # Transpose to (B, T, H, L, 8)
        q = jnp.transpose(q, (0, 1, 3, 2, 4))
        k = jnp.transpose(k, (0, 1, 3, 2, 4))
        v = jnp.transpose(v, (0, 1, 3, 2, 4))

        k_rev = reversion(k)

        # Broadcast dims for geometric product attention scores
        q_exp = q[..., :, None, :]  # (B,T,H,L,1,8)
        k_rev_exp = k_rev[..., None, :, :]  # (B,T,H,1,L,8)

        gp = geometric_product_ga3(q_exp, k_rev_exp)  # (B,T,H,L,L,8)

        attn_scores = gp[..., 0] / jnp.sqrt(head_dim).astype(
            x.dtype
        )  # scalar part. Ensure division is same dtype

        causal_mask = nn.make_causal_mask(jnp.ones((B, L), dtype=bool))
        causal_mask = causal_mask.reshape(B, 1, 1, L, L)
        mask = jnp.broadcast_to(causal_mask, (B, T, self.num_heads, L, L))

        attn_scores = jnp.where(
            mask, attn_scores, jnp.array(-1e9, dtype=attn_scores.dtype)
        )  # Ensure -1e9 has correct dtype
        attn_weights = nn.softmax(attn_scores, axis=-1)

        out = jnp.einsum("bthql,bthld->bthqd", attn_weights, v)  # (B,T,H,L,8)

        out = jnp.transpose(out, (0, 1, 3, 2, 4))  # (B,T,L,H,8)

        # Merge head and multivector dim for final projection
        out_reshaped = out.reshape(B, T, L, self.num_heads * 8)

        out_proj = nn.Dense(features=8, dtype=self.dtype)(out_reshaped)  # (B,T,L,8)
        return out_proj


# --- GASSM (rotor residual block) ---


class GASSM(nn.Module):
    dtype: jnp.dtype = jnp.float32  # Added dtype argument

    @nn.compact
    def __call__(self, x):
        rotor_params = nn.Dense(features=4, dtype=self.dtype)(x)  # (B,T,L,4)
        norm = jnp.linalg.norm(rotor_params, axis=-1, keepdims=True)
        rotor_params = rotor_params / (norm + 1e-6)

        s, b12, b13, b23 = jnp.split(rotor_params, 4, axis=-1)

        # Initialize with zeros of the correct dtype
        rotors = jnp.zeros_like(x, dtype=self.dtype)
        rotors = rotors.at[..., 0].set(s.squeeze(-1))
        rotors = rotors.at[..., 4].set(b12.squeeze(-1))
        rotors = rotors.at[..., 5].set(b13.squeeze(-1))
        rotors = rotors.at[..., 6].set(b23.squeeze(-1))

        R_inv = reversion(rotors)

        x_rot = geometric_product_ga3(rotors, x)
        x_rot = geometric_product_ga3(x_rot, R_inv)

        return x_rot + x


# --- Transformer Block ---


class GeometricTransformerBlock(nn.Module):
    num_heads: int
    ffn_dim: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, *, training: bool):
        # Apply checkpointing to the attention and FFN sub-blocks within the transformer block
        # Correct usage: apply remat to the Module *class* and then instantiate it.
        # This creates a new Module (or a wrapper around it) that is checkpointed.
        attn_output = remat(GeometricAttention)(
            num_heads=self.num_heads, dtype=self.dtype
        )(x)
        attn_output = nn.Dropout(self.dropout_rate)(
            attn_output, deterministic=not training
        )
        x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)(x + attn_output)

        ffn_output = remat(GeometricFFN)(ffn_dim=self.ffn_dim, dtype=self.dtype)(x)
        ffn_output = nn.Dropout(self.dropout_rate)(
            ffn_output, deterministic=not training
        )
        x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)(x + ffn_output)
        return x


# --- Final Transformer LM ---


class GATransformerLM(nn.Module):
    vocab_size: int
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 64
    max_len: int = 64
    dtype: jnp.dtype = jnp.float32  # Added dtype argument

    @nn.compact
    def __call__(self, x, *, training: bool):
        # x: (B,T,L)

        # Embed tokens (B,T,L,8) - Embeddings should be created in the desired dtype
        x = nn.Embed(
            num_embeddings=self.vocab_size,
            features=8,
            param_dtype=self.dtype,
            dtype=self.dtype,
        )(x)

        x = nn.Dropout(0.1)(x, deterministic=not training)

        x = GASSM(dtype=self.dtype)(x)

        for _ in range(self.num_layers):
            x = GeometricTransformerBlock(
                num_heads=self.num_heads, ffn_dim=self.ffn_dim, dtype=self.dtype
            )(x, training=training)

        # Output projection: (B,T,L,vocab_size,8)
        out = GeometricDense(
            features=self.vocab_size, use_bias=False, dtype=self.dtype
        )(x)

        logits = out[..., 0]  # scalar part for logits
        logits = jnp.mean(logits, axis=1)  # collapse T dim

        return logits


# --- Training utils ---


def create_train_state(model, key, learning_rate_fn, batch_shape, model_dtype):
    # Initialize parameters with the specified model_dtype
    params = model.init(key, jnp.ones(batch_shape, dtype=jnp.int32), training=False)[
        "params"
    ]
    # Keep optimizer state in float32 for stability
    tx = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=learning_rate_fn)
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch, dropout_key):
    def loss_fn(params):
        # Cast inputs to the model's dtype if they come in as float32
        # For integer inputs (tokens), this is not needed until after embedding.
        logits = state.apply_fn(
            {"params": params},
            batch["inputs"],
            training=True,
            rngs={"dropout": dropout_key},
        )
        # Ensure loss computation is stable. optax.softmax_cross_entropy_with_integer_labels
        # is generally robust to mixed precision inputs.
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            batch["targets"].reshape(-1),
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # Optax handles casting gradients to float32 for parameter updates if needed
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn({"params": state.params}, batch["inputs"], training=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]),
        batch["targets"].reshape(-1),
    ).mean()
    return loss


def evaluate_model(state, val_generator_fn):
    total_loss = 0.0
    num_batches = 0
    for (
        batch
    ) in val_generator_fn():  # Call the generator function to get a fresh generator
        total_loss += eval_step(state, batch)
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0


def sample_text(
    state, tokenizer, prompt, max_len=50, temperature=1.0, top_k=50, seed=42
):
    key = random.PRNGKey(seed)
    tokens = jnp.array(tokenizer.encode(prompt))[None, None, :]  # (1,1,L)
    for _ in range(max_len):
        # Logits come out in model_dtype, but should be stable enough for sampling
        logits = state.apply_fn(
            {"params": state.params}, tokens, training=False
        )  # (1,L,vocab)
        last_logits = logits[:, -1, :] / temperature
        top_k_vals, top_k_indices = jax.lax.top_k(last_logits, k=top_k)
        key, subkey = random.split(key)
        # Use softmax for sampling.
        # This can be done in full precision if there are numerical issues with bfloat16 probs.
        # For simple top_k sampling with temperature, bfloat16 should be fine.
        sampled_index = random.categorical(subkey, top_k_vals)
        next_token = jnp.take_along_axis(top_k_indices, sampled_index[:, None], axis=-1)
        next_token = next_token[:, None, :]  # add T dim = 1
        tokens = jnp.concatenate([tokens, next_token], axis=2)  # concat over L
    return tokenizer.decode(tokens[0, 0, :].tolist())


def main():
    SEQ_LEN = 64
    BATCH_SIZE = 16
    T = 1  # Number of sequences per item in the batch (should be 1 for current model)
    NUM_EPOCHS = 10  # Increased for more meaningful training
    # LEARNING_RATE = 3e-4 # Replaced by schedule
    WARMUP_STEPS = 500  # Number of steps for linear warmup
    # CHECKPOINT_DIR = "ga_transformer_checkpoints" # Defined below with os.makedirs

    # PROFILEING
    jax.profiler.start_server(9999)

    # --- Mixed Precision Setting ---
    MODEL_DTYPE = jnp.bfloat16  # Change to jnp.float32 for full precision

    tokenizer = get_tokenizer()
    VOCAB_SIZE = tokenizer.n_vocab

    print("Loading and preparing dataset...")
    raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Filter out empty strings and join
    train_texts = [ex["text"] for ex in raw_dataset["train"] if ex["text"].strip()]
    val_texts = [ex["text"] for ex in raw_dataset["validation"] if ex["text"].strip()]

    # Add a beginning-of-sequence token (e.g., a special token or a newline as a separator)
    # Using ' ' or '<|endoftext|>' if tokenizer supports it. Here, newline is already common.
    # For GPT-2 tokenizer, <|endoftext|> (token 50256) is often used as BOS/EOS
    # Joining with space is a simple way to combine the texts into a single stream.
    train_tokens = np.array(tokenizer.encode(" ".join(train_texts)))
    val_tokens = np.array(tokenizer.encode(" ".join(val_texts)))

    # Calculate total training steps for learning rate schedule
    # Need to instantiate the generator once to get the number of batches
    # Create a *new* generator for this count to avoid exhausting the main one.
    num_train_sequences = len(train_tokens) // SEQ_LEN
    num_train_batches = num_train_sequences // BATCH_SIZE
    TOTAL_STEPS = NUM_EPOCHS * num_train_batches
    print(f"Total training steps: {TOTAL_STEPS}")

    # Prepare data generators
    train_data_gen = lambda: data_generator(
        train_tokens, SEQ_LEN, BATCH_SIZE, is_training=True
    )
    val_data_gen = lambda: data_generator(
        val_tokens, SEQ_LEN, BATCH_SIZE, is_training=False
    )

    model = GATransformerLM(
        vocab_size=VOCAB_SIZE,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        max_len=SEQ_LEN,
        dtype=MODEL_DTYPE,
    )

    rng = random.PRNGKey(0)
    dropout_key = random.PRNGKey(1)

    # --- Learning Rate Schedule ---
    # Linear warmup followed by cosine decay
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=3e-4,  # Peak learning rate
        warmup_steps=WARMUP_STEPS,
        decay_steps=TOTAL_STEPS,
        end_value=0.0,
    )
    learning_rate_fn = schedule_fn

    state = create_train_state(
        model, rng, learning_rate_fn, (BATCH_SIZE, T, SEQ_LEN), MODEL_DTYPE
    )

    # --- Checkpointing ---
    CHECKPOINT_DIR = "ga_transformer_checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Ensure directory exists
    print(f"Loading checkpoint from {CHECKPOINT_DIR}...")

    # Attempt to restore the entire state object if it exists
    restored_state = checkpoints.restore_checkpoint(CHECKPOINT_DIR, target=state)
    if (
        restored_state == state
    ):  # restore_checkpoint returns the target object if loaded, otherwise target remains unchanged
        print("No checkpoint found. Starting from scratch.")
    else:
        state = restored_state  # Assign the restored state
        print(f"Checkpoint loaded from step {state.step}.")

    print(f"Starting training with model dtype: {MODEL_DTYPE}")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        # Iterate over the training data generator
        for batch_idx, batch in enumerate(train_data_gen()):
            dropout_key, subkey = random.split(dropout_key)
            state, loss = train_step(state, batch, subkey)
            if (state.step % 100 == 0) or (
                batch_idx == num_train_batches - 1
            ):  # Also print last batch of epoch
                current_lr = learning_rate_fn(state.step)
                print(
                    f"  Batch {batch_idx+1}/{num_train_batches}, Step: {state.step}, Loss: {loss.item():.4f}, LR: {current_lr.item():.6f}"
                )

        val_loss = evaluate_model(state, val_data_gen)
        print(f"Validation loss: {val_loss.item():.4f}")

        # Save checkpoint after each epoch
        checkpoints.save_checkpoint(CHECKPOINT_DIR, state, state.step, keep=3)
        print(f"Checkpoint saved at step {state.step}")

    prompt = "Artificial intelligence is"
    print("\nSampling text:")
    # Sampling doesn't require dropout, so we don't need to pass a dropout_key
    print(sample_text(state, tokenizer, prompt, max_len=50))


if __name__ == "__main__":
    main()
