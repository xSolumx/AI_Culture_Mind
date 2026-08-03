"""Training, evaluation, checkpointing, and streaming generation harness."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from .model import SpinorSSMConfig, SpinorSSMLanguageModel

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        required = (PAD, UNK, BOS, EOS)
        if self.tokens[:4] != required:
            raise ValueError(f"vocabulary must begin with {required}")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary tokens must be unique")

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, text: str, *, boundaries: bool = True) -> list[int]:
        lookup = self.token_to_id
        result = [lookup.get(token, lookup[UNK]) for token in text.split()]
        return [lookup[BOS], *result, lookup[EOS]] if boundaries else result

    def decode(self, token_ids: list[int]) -> str:
        ignored = {PAD, BOS}
        words = []
        for token_id in token_ids:
            token = self.tokens[token_id] if 0 <= token_id < len(self.tokens) else UNK
            if token == EOS:
                break
            if token not in ignored:
                words.append(token)
        return " ".join(words)


def build_vocabulary(lines: list[str], *, minimum_frequency: int = 1) -> Vocabulary:
    if minimum_frequency < 1:
        raise ValueError("minimum_frequency must be positive")
    counts = Counter(token for line in lines for token in line.split())
    lexical = sorted(token for token, count in counts.items() if count >= minimum_frequency)
    return Vocabulary((PAD, UNK, BOS, EOS, *lexical))


class CausalWindowDataset(Dataset):
    """Overlapping next-token windows that never cross line boundaries."""

    def __init__(
        self,
        lines: list[str],
        vocabulary: Vocabulary,
        *,
        context_length: int,
        stride: int | None = None,
    ) -> None:
        if context_length < 2:
            raise ValueError("context_length must be at least two")
        stride = stride or max(1, context_length // 2)
        if stride < 1:
            raise ValueError("stride must be positive")
        self.examples: list[tuple[list[int], list[int]]] = []
        for line in lines:
            encoded = vocabulary.encode(line)
            for start in range(0, max(1, len(encoded) - 1), stride):
                window = encoded[start : start + context_length + 1]
                if len(window) >= 2:
                    self.examples.append((window[:-1], window[1:]))
                if start + context_length + 1 >= len(encoded):
                    break
        if not self.examples:
            raise ValueError("corpus produced no causal examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.examples[index]


def collate_causal(
    examples: list[tuple[list[int], list[int]]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(len(inputs) for inputs, _ in examples)
    inputs = torch.zeros(len(examples), length, dtype=torch.long)
    targets = torch.full(
        (len(examples), length), IGNORE_INDEX, dtype=torch.long
    )
    mask = torch.zeros(len(examples), length, dtype=torch.bool)
    for row, (source, target) in enumerate(examples):
        size = len(source)
        inputs[row, :size] = torch.tensor(source)
        targets[row, :size] = torch.tensor(target)
        mask[row, :size] = True
    return inputs, mask, targets


@torch.no_grad()
def evaluate(
    model: SpinorSSMLanguageModel,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    labels = 0
    for token_ids, mask, targets in loader:
        token_ids, mask, targets = (
            tensor.to(device) for tensor in (token_ids, mask, targets)
        )
        logits = model(token_ids, attention_mask=mask)
        loss_sum += float(
            nn.functional.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
        )
        valid = targets.ne(IGNORE_INDEX)
        correct += int((logits.argmax(dim=-1)[valid] == targets[valid]).sum())
        labels += int(valid.sum())
    return {
        "loss": loss_sum / labels,
        "perplexity": float(torch.exp(torch.tensor(loss_sum / labels))),
        "accuracy": correct / labels,
        "labels": labels,
    }


def train(
    model: SpinorSSMLanguageModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.01,
) -> list[dict[str, float | int]]:
    if epochs < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid optimizer schedule")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        labels = 0
        for token_ids, mask, targets in train_loader:
            token_ids, mask, targets = (
                tensor.to(device) for tensor in (token_ids, mask, targets)
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids, attention_mask=mask)
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            valid = int(targets.ne(IGNORE_INDEX).sum())
            (loss / valid).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            labels += valid
        validation = evaluate(model, validation_loader, device=device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / labels,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
        }
        history.append(row)
        print(json.dumps(row))
    return history


def save_checkpoint(
    path: Path,
    model: SpinorSSMLanguageModel,
    vocabulary: Vocabulary,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "spinor-ssm-overhauled-v1",
            "config": asdict(model.config),
            "vocabulary": list(vocabulary.tokens),
            "model": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(
    path: Path, *, device: torch.device
) -> tuple[SpinorSSMLanguageModel, Vocabulary, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "spinor-ssm-overhauled-v1":
        raise ValueError("unsupported checkpoint format")
    config = SpinorSSMConfig(**payload["config"])
    state_dict = payload["model"]
    saved_dtype = next(
        tensor.dtype for tensor in state_dict.values() if tensor.is_floating_point()
    )
    model = SpinorSSMLanguageModel(config).to(device=device, dtype=saved_dtype)
    model.load_state_dict(state_dict, strict=True)
    return model, Vocabulary(tuple(payload["vocabulary"])), payload["metadata"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "corpus.txt",
    )
    parser.add_argument("--output", type=Path, default=Path("spinor_ssm_v1.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    lines = [line.strip() for line in args.corpus.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    vocabulary = build_vocabulary(lines)
    dataset = CausalWindowDataset(
        lines, vocabulary, context_length=args.context_length
    )
    validation_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - validation_size
    if train_size < 1:
        raise ValueError("corpus is too small for a train/validation split")
    split_generator = torch.Generator().manual_seed(args.seed)
    train_data, validation_data = random_split(
        dataset, (train_size, validation_size), generator=split_generator
    )
    loader_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        collate_fn=collate_causal,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_causal,
    )
    config = SpinorSSMConfig(
        vocab_size=len(vocabulary.tokens),
        channels=args.channels,
        num_layers=args.layers,
    )
    model = SpinorSSMLanguageModel(config)
    history = train(
        model,
        train_loader,
        validation_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    save_checkpoint(
        args.output,
        model,
        vocabulary,
        metadata={"history": history, "seed": args.seed},
    )
    prompt = torch.tensor(
        [vocabulary.encode("geometric algebra", boundaries=False)], device=device
    )
    generated = model.generate(prompt, max_new_tokens=16, stop_ids={3})
    print(vocabulary.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
