# GA-SSM Transformer Decoder with Tokenization and LM-style Training

# Requirements: flax, jax, optax, numpy, datasets, tiktoken
# pip install flax jax optax numpy datasets tiktoken

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from datasets import load_dataset
import tiktoken
from functools import partial
import flax.serialization
import os
import jax.random as random
from typing import Tuple


# === GA(3,0) Algebra Primitives ===


@jax.jit
def geometric_product(a, b):
    # a, b shape (..., 8) multivectors
    # Basis order: [1, e1, e2, e3, e12, e13, e23, e123]
    def gp_single(a, b):
        a = a[:8]  # Ensure 8D
        b = b[:8]
        a0, a1, a2, a3, a4, a5, a6, a7 = a
        b0, b1, b2, b3, b4, b5, b6, b7 = b

        return jnp.array(
            [
                a0 * b0
                + a1 * b1
                + a2 * b2
                + a3 * b3
                - a4 * b4
                - a5 * b5
                - a6 * b6
                - a7 * b7,
                a0 * b1
                + a1 * b0
                - a2 * b4
                - a3 * b5
                + a4 * b2
                + a5 * b3
                - a6 * b7
                - a7 * b6,
                a0 * b2
                + a2 * b0
                + a1 * b4
                - a3 * b6
                - a4 * b1
                + a5 * b7
                + a6 * b3
                + a7 * b5,
                a0 * b3
                + a3 * b0
                + a1 * b5
                + a2 * b6
                - a4 * b7
                - a5 * b1
                - a6 * b2
                - a7 * b4,
                a0 * b4
                + a4 * b0
                + a1 * b2
                - a2 * b1
                + a3 * b7
                + a5 * b6
                - a6 * b5
                + a7 * b3,
                a0 * b5
                + a5 * b0
                + a1 * b3
                - a3 * b1
                - a2 * b7
                - a4 * b6
                + a6 * b4
                - a7 * b2,
                a0 * b6
                + a6 * b0
                + a2 * b3
                - a3 * b2
                + a1 * b7
                + a4 * b5
                - a5 * b4
                + a7 * b1,
                a0 * b7
                + a7 * b0
                + a1 * b6
                - a2 * b5
                + a3 * b4
                + a4 * b3
                - a5 * b2
                + a6 * b1,
            ]
        )

    return jax.vmap(gp_single)(a, b) if a.ndim == 2 else gp_single(a, b)


@jax.jit
def reversion(mv):
    signs = jnp.array([1, 1, 1, 1, -1, -1, -1, -1])
    return mv * signs


@jax.jit
def rotor(theta):
    e12 = jnp.array([0, 0, 0, 0, 1, 0, 0, 0])
    norm = jnp.linalg.norm(e12)
    e12_unit = e12 / norm
    return jnp.cos(theta / 2) * jnp.eye(1, 8, 0)[0] - jnp.sin(theta / 2) * e12_unit


# === Tokenization ===

enc = tiktoken.get_encoding("gpt2")


def tokenize(text):
    return enc.encode(text)


def decode(tokens):
    return enc.decode(tokens)


# === Dataset Pre-packing ===


def pack_token_sequence(token_array, seq_len, batch_size):
    total_tokens = len(token_array)
    num_chunks = total_tokens // (seq_len * batch_size)
    usable_tokens = num_chunks * seq_len * batch_size

    trimmed = token_array[:usable_tokens]
    reshaped = trimmed.reshape((batch_size, -1))  # (B, total_seq)
    packed = []

    for i in range(0, reshaped.shape[1], seq_len):
        chunk = reshaped[:, i : i + seq_len]
        if chunk.shape[1] == seq_len:
            packed.append(chunk)

    packed = jnp.stack(packed)  # (num_batches, B, L)
    packed = jnp.swapaxes(packed, 0, 1)  # (B, num_batches, L)
    packed = packed.reshape(-1, seq_len)  # (B * num_batches, L)

    return packed


# Load and tokenize text
raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
text_data = "\n\n".join(example["text"] for example in raw_dataset["train"])
tokens = tokenize(text_data)
tokens = np.array(tokens[:1000000])  # Truncate for demo

SEQ_LEN = 64
BATCH_SIZE = 32
VOCAB_SIZE = enc.n_vocab

# Pack tokens
packed = pack_token_sequence(tokens, SEQ_LEN, BATCH_SIZE)
inputs = packed
targets = jnp.roll(inputs, shift=-1, axis=-1)
num_batches = len(inputs) // BATCH_SIZE


# Create batches
num_batches = len(tokens) // (SEQ_LEN * BATCH_SIZE)
inputs = tokens[: num_batches * SEQ_LEN * BATCH_SIZE].reshape(
    num_batches, BATCH_SIZE, SEQ_LEN
)
targets = jnp.roll(inputs, shift=-1, axis=-1)

# === GA Embedding ===


class GAEmbed(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        emb = nn.Embed(num_embeddings=VOCAB_SIZE, features=8)(x)
        return emb


class GeometricDense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        # x: (..., 8)
        # Learn a multivector weight per output feature
        w = self.param("kernel", nn.initializers.lecun_normal(), (self.features, 8))
        # This is a simplification. A true geometric linear layer is more complex.
        # This example performs a geometric product with a set of learned weights.
        # For a simple projection, a standard nn.Dense is often more practical.
        # However, for maintaining structure:

        # Define a function for one feature output
        def apply_gp(inp, weight):
            # inp: (8,), weight: (8,)
            return geometric_product(weight, inp)

        # Apply for each output feature
        # This is computationally intensive and needs optimization
        # For now, let's just use a standard Dense layer for projection.
        # But a more advanced model could replace all Dense layers with this concept.
        return nn.Dense(self.features)(x)


# === GA-SSM Cell ===


class GASSM(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x shape: (B, L, 8)

        # Project input to get a dynamic rotation angle per token
        # This makes the rotation context-dependent
        thetas = nn.Dense(features=1)(x).squeeze(-1)  # (B, L)

        # Create rotors for each token
        e12 = jnp.array([0, 0, 0, 0, 1, 0, 0, 0])
        rotors = (
            jnp.cos(thetas / 2)[..., None] * jnp.eye(1, 8, 0)
            - jnp.sin(thetas / 2)[..., None] * e12
        )

        R_inv = reversion(rotors)

        # Define a single rotation for vmap
        def rotate(v, R, R_inv):
            return geometric_product(geometric_product(R, v), R_inv)

        # vmap over batch and sequence length
        x_rot = jax.vmap(jax.vmap(rotate))(x, rotors, R_inv)

        return x_rot + x  # Removed input_scale for simplicity


# === Transformer Decoder ===


class TransformerBlock(nn.Module):
    hidden_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool):  # Pass training flag
        attn_output = nn.SelfAttention(num_heads=4, qkv_features=self.hidden_dim)(x)
        attn_output = nn.Dropout(self.dropout_rate)(
            attn_output, deterministic=not training
        )
        x = x + attn_output

        # Add LayerNorm
        x = nn.LayerNorm()(x)

        ffn_output = nn.Dense(self.hidden_dim * 4)(x)
        ffn_output = nn.relu(ffn_output)
        ffn_output = nn.Dense(self.hidden_dim)(ffn_output)
        ffn_output = nn.Dropout(self.dropout_rate)(
            ffn_output, deterministic=not training
        )
        x = x + ffn_output

        # Add LayerNorm
        x = nn.LayerNorm()(x)
        return x


class GAPositionalEncoding(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        # x shape: (B, L, 8)
        seq_len = x.shape[1]

        # Learnable bivectors for position encoding
        # One for each position up to a max length
        pos_bivectors = self.param(
            "pos_bivectors",
            nn.initializers.normal(stddev=0.02),
            (seq_len, 3),  # e.g., for e12, e13, e23 components
        )

        # Create full bivector multivectors
        full_bivectors = jnp.zeros((seq_len, 8))
        full_bivectors = full_bivectors.at[:, 4:7].set(pos_bivectors)

        # Create rotors from bivectors R = exp(-B/2) approx 1 - B/2
        # For simplicity, we can use a small angle approximation
        # or just use it as an additive feature for now.
        # A more robust implementation would use the exponential map.
        pos_rotors = jnp.cos(0.5) * jnp.eye(1, 8, 0) - jnp.sin(0.5) * full_bivectors

        # Apply the rotation to each token embedding
        # This is a simplified example; vmap would be needed
        # For simplicity, let's just add it for now.
        pos_embedding = nn.Dense(8)(full_bivectors)  # Project to embed dim

        return x + pos_embedding[None, :, :]


# In GATransformerLM:
class GATransformerLM(nn.Module):
    # ... (attributes)
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = GAEmbed(self.embed_dim)(x)
        x = GAPositionalEncoding(self.embed_dim)(x)  # ADD THIS
        x = GASSM()(x)
        x = nn.Dense(self.hidden_dim)(x)
        for _ in range(self.num_layers):
            x = TransformerBlock(self.hidden_dim)(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits


# === Training ===

model = GATransformerLM(vocab_size=VOCAB_SIZE)
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32))


@jax.jit
def cross_entropy_loss(logits, targets):
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return loss.mean()


GRAD_CLIP_NORM = 1.0

# In the training setup section
num_epochs = 200
steps_per_epoch = num_batches
total_steps = num_epochs * steps_per_epoch
warmup_steps = 1000

# Create the scheduler
lr_scheduler = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=warmup_steps,
    decay_steps=total_steps - warmup_steps,
    end_value=1e-5,
)

# Chain it with Adam (or AdamW) and clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP_NORM),
    optax.adamw(learning_rate=lr_scheduler),  # Use AdamW for weight decay
)

# Initialize opt_state as usual
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, x, y, rng):
    def loss_fn(p):
        logits = model.apply(p, x, rngs={"dropout": rng})
        loss = cross_entropy_loss(logits, y)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


##=================== TRAINING ===============##


# Load validation data
val_text_data = "\n\n".join(example["text"] for example in raw_dataset["validation"])
val_tokens = tokenize(val_text_data)
val_tokens = np.array(val_tokens[:5000])  # Smaller val set for speed

val_num_batches = len(val_tokens) // (SEQ_LEN * BATCH_SIZE)
val_inputs = val_tokens[: val_num_batches * SEQ_LEN * BATCH_SIZE].reshape(
    val_num_batches, BATCH_SIZE, SEQ_LEN
)
val_targets = jnp.roll(val_inputs, shift=-1, axis=-1)


@jax.jit
def eval_step(params, x, y):
    logits = model.apply(params, x)
    return cross_entropy_loss(logits, y)


def evaluate(params):
    total_loss = 0.0
    for batch in range(val_num_batches):
        x_batch = jnp.array(val_inputs[batch])
        y_batch = jnp.array(val_targets[batch])
        loss = eval_step(params, x_batch, y_batch)
        total_loss += loss
    avg_loss = total_loss / val_num_batches
    return avg_loss


def sample(params, prompt, max_len=50, temperature=1.0, top_k=50, key_seed=42):
    tokens = jnp.array(tokenize(prompt))[None, :]  # (1, L)
    key = random.PRNGKey(key_seed)

    for _ in range(max_len):
        logits = model.apply(params, tokens)  # (1, seq_len, vocab_size)
        logits = logits[:, -1, :] / temperature  # last token logits, scaled

        # Top-k filtering
        top_k_indices = jnp.argsort(logits, axis=-1)[:, -top_k:]
        top_k_logits = jnp.take_along_axis(logits, top_k_indices, axis=-1)

        # Sample from top-k logits
        key, subkey = random.split(key)
        sampled_index = random.categorical(subkey, top_k_logits, axis=-1)
        next_token = jnp.take_along_axis(top_k_indices, sampled_index[:, None], axis=-1)

        tokens = jnp.concatenate([tokens, next_token], axis=1)

    return decode(tokens[0].tolist())


# Checkpointing helpers
CHECKPOINT_PATH = "checkpoint.msgpack"


def save_checkpoint(params, opt_state):
    data = {"params": params, "opt_state": opt_state}
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(flax.serialization.to_bytes(data))


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return None

    # Initialize model and opt_state to get the correct structure for deserialization
    print("Initializing dummy model for checkpoint loading...")
    key = jax.random.PRNGKey(0)
    dummy_params = model.init(key, jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32))
    dummy_opt_state = optimizer.init(dummy_params)
    target_struct = {"params": dummy_params, "opt_state": dummy_opt_state}

    with open(CHECKPOINT_PATH, "rb") as f:
        data = flax.serialization.from_bytes(target_struct, f.read())
    print("Checkpoint loaded successfully.")
    return data


loaded = load_checkpoint()
if loaded:
    params, opt_state = loaded["params"], loaded["opt_state"]
else:
    # Initialize from scratch
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32))
    opt_state = optimizer.init(params)
    print("No checkpoint found, initializing new model.")

# === Main Training Loop ===

# === Dataset Pre-packing ===

# Use the function as intended
# (B * num_batches, L)
packed_train = pack_token_sequence(tokens, SEQ_LEN, BATCH_SIZE)
train_inputs = packed_train
train_targets = jnp.roll(train_inputs, shift=-1, axis=-1)
num_train_batches = len(train_inputs) // BATCH_SIZE

packed_val = pack_token_sequence(val_tokens, SEQ_LEN, BATCH_SIZE)
val_inputs = packed_val
val_targets = jnp.roll(val_inputs, shift=-1, axis=-1)
num_val_batches = len(val_inputs) // BATCH_SIZE


# === Main Training Loop ===
def train(num_epochs=200):
    # ...
    # Iterate through pre-packed batches
    for epoch in range(num_epochs):
        # Create a shuffled view for the epoch
        key, subkey = random.split(key)
        perm = random.permutation(subkey, len(train_inputs))
        shuffled_inputs = train_inputs[perm]
        shuffled_targets = train_targets[perm]

        for i in range(num_train_batches):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            x_batch = shuffled_inputs[start:end]
            y_batch = shuffled_targets[start:end]
            key, subkey = jax.random.split(key)
            params, opt_state, loss = train_step(
                params, opt_state, x_batch, y_batch, subkey
            )

            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss:.4f}")

        val_loss = evaluate(params)
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(params, opt_state)
            print(f"Checkpoint saved at epoch {epoch} with val loss {val_loss:.4f}")

        sample_text = sample(params, "I am a")
        print(f"Sample generation: {sample_text}\n")


if __name__ == "__main__":
    train()
