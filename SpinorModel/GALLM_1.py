import torch
import torch.nn as nn
import math
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from torch.utils.data import Dataset, DataLoader


# ---------- GA operations ----------
def geometric_product(a, b):
    # Basic implementation for GA product in low dimensions (extendable)
    dot = (a * b).sum(dim=-1, keepdim=True)
    wedge = a.unsqueeze(-1) * b.unsqueeze(-2) - b.unsqueeze(-1) * a.unsqueeze(-2)
    return torch.cat([dot, wedge.reshape(*dot.shape[:-1], -1)], dim=-1)


class GALinear(nn.Module):
    def __init__(self, in_features=8, out_features=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


# ---------- GA Positional Encoding ----------
class GAPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('encoding', encoding)  # register buffer for device moving

    def forward(self, x):
        return x + self.encoding[:, : x.size(1)].to(x.device)


# ---------- GA Multi-head Attention ----------
class GAMultiHeadAttention(nn.Module):
    def __init__(self, d_model=32, heads=4):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.heads = heads
        self.d_head = d_model // heads
        self.q_lin = nn.ModuleList(
            [GALinear(d_model, self.d_head) for _ in range(heads)]
        )
        self.k_lin = nn.ModuleList(
            [GALinear(d_model, self.d_head) for _ in range(heads)]
        )
        self.v_lin = nn.ModuleList(
            [GALinear(d_model, self.d_head) for _ in range(heads)]
        )
        self.out_lin = GALinear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, _ = x.size()
        all_heads = []

        for i in range(self.heads):
            Q = self.q_lin[i](x)  # [B, S, d_head]
            K = self.k_lin[i](x)  # [B, S, d_head]
            V = self.v_lin[i](x)  # [B, S, d_head]

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, S, S]

            if mask is not None:
                # mask shape: [B, S, S] with 1 for allowed positions, 0 for masked
                scores = scores.masked_fill(mask == 0, float("-inf"))

            attn = torch.softmax(scores, dim=-1)  # [B, S, S]
            out = torch.matmul(attn, V)  # [B, S, d_head]

            all_heads.append(out)

        concat = torch.cat(all_heads, dim=-1)  # [B, S, d_model]
        return self.out_lin(concat)


# ---------- GA Transformer Layer ----------
class GALayer(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.attn = GAMultiHeadAttention(d_model=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            GALinear(d_model, d_model * 4),
            nn.ReLU(),
            GALinear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


# ---------- GA Transformer Model ----------
class GATransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, layers=4, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = GAPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([GALayer(d_model) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)

        # Output layer: weight tied to token_embed weights
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.token_embed.weight

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, attention_mask=None):
        """
        x: [B, S] token indices
        attention_mask: [B, S] with 1 for tokens, 0 for padding
        """
        B, S = x.shape

        # Embed tokens + positions
        x = self.token_embed(x) * math.sqrt(self.d_model)  # scale embedding
        x = self.pos_embed(x)

        # Build causal mask (prevent attention to future tokens)
        causal_mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool)).unsqueeze(0).expand(B, -1, -1)

        # Combine with attention_mask for padding: 
        # attention_mask: [B,S] -> [B,1,S] to broadcast on keys dimension
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1).expand(-1, S, -1)  # [B,S,S]
            combined_mask = causal_mask & (padding_mask.bool())
        else:
            combined_mask = causal_mask

        # Convert to int mask (1 or 0)
        combined_mask = combined_mask.int()

        for layer in self.layers:
            x = layer(x, combined_mask)

        x = self.norm(x)
        logits = self.out(x)  # [B, S, vocab_size]

        return logits


# ---------- Tokenizer Setup ----------
def train_tokenizer_from_text(corpus_file, vocab_size=256):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train([corpus_file], trainer)
    return tokenizer


# ---------- Dataset ----------
class TextDataset(Dataset):
    def __init__(self, corpus, tokenizer, pad_id):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.pad_id = pad_id

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        # Add BOS and EOS tokens around each sequence
        encoded = self.tokenizer.encode(self.corpus[idx])
        ids = [self.tokenizer.token_to_id("<BOS>")] + encoded.ids + [self.tokenizer.token_to_id("<EOS>")]
        return ids


def collate_fn(batch, pad_id=0):
    max_len = max(len(ids) for ids in batch)
    input_ids = []
    attention_mask = []
    for ids in batch:
        padded = ids + [pad_id] * (max_len - len(ids))
        mask = [1] * len(ids) + [0] * (max_len - len(ids))
        input_ids.append(padded)
        attention_mask.append(mask)
    return torch.tensor(input_ids), torch.tensor(attention_mask)


def train_model(model, tokenizer, corpus, epochs=50, batch_size=2, lr=0.01):
    pad_id = tokenizer.token_to_id("<PAD>")
    if pad_id is None or pad_id < 0:
        raise ValueError("PAD token not found or invalid in tokenizer vocab!")

    dataset = TextDataset(corpus, tokenizer, pad_id)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)

            # Shift targets for causal LM (predict next token)
            targets = input_ids[:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous()

            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)

            loss = loss_fn(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.6f}")


# ---------- Decode Output ----------
def decode_output(tokenizer, logits, stop_token="<EOS>"):
    """
    Greedy decode from logits, stopping at stop_token if found.
    logits: [1, S, V]
    """
    ids = torch.argmax(logits, dim=-1)[0].tolist()
    decoded_tokens = []
    stop_id = tokenizer.token_to_id(stop_token)
    for i in ids:
        if i == stop_id:
            break
        decoded_tokens.append(i)
    return tokenizer.decode(decoded_tokens)


# ---------- Run ----------
if __name__ == "__main__":
    corpus_file = Path(__file__).with_name("corpus.txt")
    # Make sure corpus is large enough in practice
    corpus = [
        "This is an example.",
        "Geometric algebra meets attention.",
        "Spinors and tensors combined.",
        "Add more diverse sentences here for better tokenizer training.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "This script uses geometric algebra and transformers.",
    ]
    if not corpus_file.exists() or corpus_file.stat().st_size == 0:
        corpus_file.write_text("\n".join(corpus), encoding="utf-8")

    print("Training BPE tokenizer...")
    tokenizer = train_tokenizer_from_text(str(corpus_file), vocab_size=512)
    print("Tokenizer vocab size:", tokenizer.get_vocab_size())

    model = GATransformer(vocab_size=tokenizer.get_vocab_size())
    train_model(model, tokenizer, corpus, epochs=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Example generation
    test_input_text = "Hello, I am "
    test_input = tokenizer.encode(test_input_text)
    input_ids = torch.tensor([[tokenizer.token_to_id("<BOS>")] + test_input.ids]).to(device)

    model.eval()
    with torch.no_grad():
        output_logits = model(input_ids)

    decoded = decode_output(tokenizer, output_logits)
    print("Decoded output:", decoded)
