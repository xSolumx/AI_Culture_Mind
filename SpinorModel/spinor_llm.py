"""Small, side-effect-free spinor language-model experiment in PyTorch."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

try:
    from .geometric_layers import GA_DIM, SpinorTransformerBlock
except ImportError:  # Support ``python SpinorModel/spinor_llm.py``.
    from geometric_layers import GA_DIM, SpinorTransformerBlock


DEFAULT_CORPUS = (
    "hello world",
    "spinor networks are interesting",
    "geometric algebra is powerful",
    "world of geometric algebra",
    "networks are powerful",
)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_vocabulary(corpus: Sequence[str]) -> tuple[dict[str, int], dict[int, str]]:
    words = sorted(set(" ".join(corpus).split()))
    vocabulary = [PAD_TOKEN, UNK_TOKEN, *words]
    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    return word_to_index, {index: word for word, index in word_to_index.items()}


class PrefixDataset(Dataset):
    """Every proper sentence prefix paired with its next token."""

    def __init__(self, corpus: Sequence[str], word_to_index: dict[str, int]):
        self.examples: list[tuple[list[int], int]] = []
        unknown = word_to_index[UNK_TOKEN]
        for sentence in corpus:
            token_ids = [word_to_index.get(word, unknown) for word in sentence.split()]
            self.examples.extend(
                (token_ids[:index], token_ids[index])
                for index in range(1, len(token_ids))
            )
        if not self.examples:
            raise ValueError("corpus must contain at least one multi-token sentence")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.examples[index]


def collate_prefixes(
    examples: list[tuple[list[int], int]], pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(prefix) for prefix, _ in examples)
    token_ids = torch.full((len(examples), max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    targets = torch.empty(len(examples), dtype=torch.long)
    for row, (prefix, target) in enumerate(examples):
        length = len(prefix)
        token_ids[row, :length] = torch.tensor(prefix)
        attention_mask[row, :length] = True
        targets[row] = target
    return token_ids, attention_mask, targets


class SpinorEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ga_embedding_dim: int = GA_DIM):
        super().__init__()
        if ga_embedding_dim != GA_DIM:
            raise ValueError(f"ga_embedding_dim must equal {GA_DIM}")
        self.embedding = nn.Embedding(vocab_size, ga_embedding_dim, padding_idx=0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class SpinorLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ga_embedding_dim: int = GA_DIM,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.embedding_layer = SpinorEmbedding(vocab_size, ga_embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            SpinorTransformerBlock(ga_embedding_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        )
        self.output_linear = nn.Linear(ga_embedding_dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if attention_mask is None:
            attention_mask = token_ids.ne(0)
        outputs = self.embedding_layer(token_ids)
        for block in self.transformer_blocks:
            outputs = block(outputs, attention_mask)
        return self.output_linear(outputs)


def final_token_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=-1)
    if torch.any(lengths == 0):
        raise ValueError("every sequence must contain at least one non-padding token")
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, lengths - 1]


def train_spinor_llm(
    model: SpinorLLM,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    *,
    device: torch.device | None = None,
) -> list[float]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[float] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for token_ids, attention_mask, targets in dataloader:
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            sequence_logits = model(token_ids, attention_mask)
            loss = criterion(final_token_logits(sequence_logits, attention_mask), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
        mean_loss = total_loss / len(dataloader)
        history.append(mean_loss)
        print(f"epoch={epoch + 1} loss={mean_loss:.4f}")
    return history


def generate_text(
    model: SpinorLLM,
    prompt: str,
    word_to_index: dict[str, int],
    index_to_word: dict[int, str],
    *,
    max_new_tokens: int = 5,
    context_window: int = 32,
) -> str:
    unknown = word_to_index[UNK_TOKEN]
    generated_words = prompt.split()
    generated_ids = [word_to_index.get(word, unknown) for word in generated_words]
    if not generated_ids:
        raise ValueError("prompt must contain at least one token")
    device = next(model.parameters()).device
    model.eval()
    for _ in range(max_new_tokens):
        context = torch.tensor([generated_ids[-context_window:]], device=device)
        with torch.no_grad():
            next_id = int(model(context)[:, -1].argmax(dim=-1).item())
        next_word = index_to_word.get(next_id, UNK_TOKEN)
        if next_word in {PAD_TOKEN, UNK_TOKEN}:
            break
        generated_ids.append(next_id)
        generated_words.append(next_word)
    return " ".join(generated_words)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)
    word_to_index, index_to_word = build_vocabulary(DEFAULT_CORPUS)
    dataset = PrefixDataset(DEFAULT_CORPUS, word_to_index)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_prefixes,
    )
    model = SpinorLLM(len(word_to_index))
    train_spinor_llm(model, dataloader, args.epochs, args.learning_rate)
    print(generate_text(model, "geometric", word_to_index, index_to_word))


if __name__ == "__main__":
    main()
