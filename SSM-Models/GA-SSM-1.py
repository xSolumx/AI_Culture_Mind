import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from functools import partial
from typing import List, Callable
import tiktoken

# === TOKENIZER ===
enc = tiktoken.get_encoding("gpt2")


def tokenize(text: str) -> List[int]:
    return enc.encode(text)


def decode(tokens: List[int]) -> str:
    return enc.decode(tokens)

# === GEOMETRIC ALGEBRA (GA) ===
# Multivectors in GA(3,0): [1, e1, e2, e3, e12, e23, e31, e123]

basis_names = ["1", "e1", "e2", "e3", "e12", "e23", "e31", "e123"]


def zero_mv():
    return jnp.zeros(8)


def encode_symbol_to_ga(symbol: str) -> jnp.ndarray:
    # Domain-specific encodings
    if symbol.lower() in ["dog", "cat", "mouse"]:
        return jnp.array([1, 1, 0, 0, 0, 0, 0, 0])  # animal: e1
    elif symbol.lower() in ["chases", "eats", "loves"]:
        return jnp.array([1, 0, 1, 0, 0, 0, 0, 0])  # verb: e2
    elif symbol.lower() in ["not", "and", "or"]:
        return jnp.array([1, 0, 0, 1, 0, 0, 0, 0])  # logical op: e3
    else:
        return zero_mv()


# === MORPHISMS ===
class Morphism:
    def __call__(self, x):
        raise NotImplementedError


class Rotate(Morphism):
    def __call__(self, x):
        rotor = jnp.array([1, 0, 0, 0, 0.5, 0.5, 0, 0])
        return x * rotor


class Scale(Morphism):
    def __call__(self, x):
        return x * 2.0


class Negate(Morphism):
    def __call__(self, x):
        return -x


class ReflectX(Morphism):
    def __call__(self, x):
        reflect = jnp.array([1, -1, 1, 1, -1, 1, 1, 1])
        return x * reflect


class CompositeMorphism(Morphism):
    def __init__(self, morphisms: List[Morphism]):
        self.morphisms = morphisms

    def __call__(self, x):
        for m in self.morphisms:
            x = m(x)
        return x


# === GA + SSM + MORPHISM PIPELINE ===
def symbolic_to_latent(symbols: List[str], morphism: Morphism) -> jnp.ndarray:
    latent = jnp.zeros(8)
    for sym in symbols:
        vec = encode_symbol_to_ga(sym)
        latent += vec
    latent = morphism(latent)
    return latent


# === TRANSFORMER DECODER ===
class TransformerDecoder(nn.Module):
    vocab_size: int
    num_layers: int = 2
    num_heads: int = 4
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, tokens, latent, train=False):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim)(tokens)
        x += latent[:self.hidden_dim]  # Inject GA+SSM vector as initial bias
        for _ in range(self.num_layers):
            x = nn.SelfAttention(num_heads=self.num_heads)(x)
            x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits


# === DECODING (GREEDY + BEAM SEARCH) ===
def generate_sentence(params, model, latent, beam=False, max_len=10, num_beams=3):
    bos = enc.encode("<|endoftext|>")[0]
    beams = [(jnp.array([bos]), 0.0)]  # (token_seq, score)
    for _ in range(max_len):
        new_beams = []
        for tokens, score in beams:
            logits = model.apply(params, tokens[None, :], latent[None, :])[0, -1]
            probs = jax.nn.log_softmax(logits)
            topk_idx = jnp.argsort(-probs)[:num_beams]
            for i in topk_idx:
                new_seq = jnp.concatenate([tokens, jnp.array([i])])
                new_score = score + probs[i].item()
                new_beams.append((new_seq, new_score))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:num_beams]
    best_seq = beams[0][0]
    return decode(np.array(best_seq.tolist()))


# === SYNTHETIC DATASET ===
class SyntheticDataset:
    def __init__(self, size=1000, max_len=5):
        self.symbol_pool = {
            "animal": ["dog", "cat", "mouse"],
            "verb": ["chases", "eats", "loves"],
            "logic": ["not", "and", "or"]
        }
        self.size = size
        self.max_len = max_len
        self.data = [self._generate_sample() for _ in range(size)]

    def _generate_sample(self):
        num_tokens = np.random.randint(2, self.max_len)
        symbols = []
        for _ in range(num_tokens):
            cat = np.random.choice(list(self.symbol_pool.keys()))
            symbol = np.random.choice(self.symbol_pool[cat])
            symbols.append(symbol)
        text = " ".join(symbols)
        return {"symbols": symbols, "text": text}

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


# === COLLATE + TOKENIZER ===
def collate_batch(batch):
    texts = [item["text"] for item in batch]
    symbol_lists = [item["symbols"] for item in batch]

    # Tokenize and pad
    tokenized = [tokenize(text) for text in texts]
    max_len = max(len(toks) for toks in tokenized)
    tokens_padded = np.zeros((len(batch), max_len), dtype=int)
    masks = np.zeros_like(tokens_padded)
    for i, toks in enumerate(tokenized):
        tokens_padded[i, :len(toks)] = toks
        masks[i, :len(toks)] = 1

    # GA+Morphism latent generation
    morphism = CompositeMorphism([Rotate(), Scale(), Negate()])
    latents = jnp.stack([symbolic_to_latent(syms, morphism) for syms in symbol_lists])

    return jnp.array(tokens_padded), jnp.array(masks), latents


# === LOSS FUNCTION ===
def cross_entropy_loss(logits, targets, mask):
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.sum(loss * mask) / jnp.sum(mask)


# === TRAIN STEP ===
@jax.jit
def train_step(optimizer, params, batch, model):
    tokens, mask, latents = batch
    targets = tokens

    def loss_fn(p):
        logits = model.apply(p, tokens, latents)
        return cross_entropy_loss(logits, targets, mask)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, optimizer.state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


# === MAIN TRAIN LOOP ===
def train():
    # Config
    vocab_size = 50257
    batch_size = 16
    num_epochs = 3
    learning_rate = 1e-3

    # Init
    dataset = SyntheticDataset(size=512)
    model = TransformerDecoder(vocab_size=vocab_size)
    dummy_tokens = jnp.array([[0]])
    dummy_latent = jnp.zeros((1, 8))
    params = model.init(jax.random.PRNGKey(0), dummy_tokens, dummy_latent)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Epochs
    for epoch in range(num_epochs):
        np.random.shuffle(dataset.data)
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset.data[i:i+batch_size]
            batch = collate_batch(batch_data)
            params, opt_state, loss = train_step(optimizer, params, batch, model)

            if i % 100 == 0:
                print(f"Epoch {epoch} Iter {i}: Loss {loss:.4f}")

    return model, params


if __name__ == "__main__":
    model, params = train()

    # Inference demo
    symbols = ["dog", "eats", "mouse"]
    morphism = CompositeMorphism([Rotate(), Scale(), Negate()])
    latent = symbolic_to_latent(symbols, morphism)
    sentence = generate_sentence(params, model, latent, beam=True)

    print("Input symbols:", symbols)
    print("Decoded sentence:", sentence)
