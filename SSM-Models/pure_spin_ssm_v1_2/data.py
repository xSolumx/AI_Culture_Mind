"""Deterministic natural byte streams for matched causal-LM experiments."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

TINY_SHAKESPEARE_REVISION = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    f"{TINY_SHAKESPEARE_REVISION}/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_SHA256 = (
    "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
)


def _split_tiny_shakespeare(payload: bytes) -> dict[str, bytes]:
    """Return disjoint chronological train/validation/test byte slices."""

    train_stop = int(0.90 * len(payload))
    validation_stop = int(0.95 * len(payload))
    return {
        "train": payload[:train_stop],
        "validation": payload[train_stop:validation_stop],
        "test": payload[validation_stop:],
    }


def tiny_shakespeare_bytes(
    split: str,
    *,
    offline: bool = False,
    cache_root: Path | None = None,
) -> tuple[torch.Tensor, str]:
    """Load pinned Tiny Shakespeare with a deterministic disjoint 90/5/5 split."""

    if split not in {"train", "validation", "test"}:
        raise ValueError("Tiny Shakespeare split must be train, validation, or test")
    root = cache_root
    if root is None:
        root = Path(
            os.environ.get(
                "PURE_SPIN_V12_DATA_CACHE",
                Path.home() / ".cache" / "pure_spin_ssm_v1_2",
            )
        )
    path = root / "tiny_shakespeare" / f"input-{TINY_SHAKESPEARE_REVISION}.txt"
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"cached Tiny Shakespeare file not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != TINY_SHAKESPEARE_SHA256:
        raise RuntimeError(
            f"Tiny Shakespeare hash mismatch: expected {TINY_SHAKESPEARE_SHA256}, "
            f"got {digest}"
        )
    selected = _split_tiny_shakespeare(payload)[split]
    array = np.frombuffer(selected, dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(array), hashlib.sha256(selected).hexdigest()


def _cached_wikitext(split: str, cache_root: Path | None):
    from datasets import Dataset

    if cache_root is None:
        huggingface_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        root = huggingface_home / "datasets"
    else:
        root = cache_root
    matches = tuple(root.glob(f"**/wikitext-2-raw-v1/**/wikitext-{split}.arrow"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one cached WikiText-2 {split} shard")
    return Dataset.from_file(str(matches[0]))


def wikitext_bytes(
    split: str, *, offline: bool = False, cache_root: Path | None = None
) -> tuple[np.ndarray, str]:
    from datasets import load_dataset

    dataset = (
        _cached_wikitext(split, cache_root)
        if offline
        else load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    )
    payload = "\n\n".join(row["text"] for row in dataset if row["text"].strip()).encode()
    return np.frombuffer(payload, dtype=np.uint8).astype(np.int64), hashlib.sha256(payload).hexdigest()


def random_batch(
    stream: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        stream.numel() - sequence_length - 1,
        (batch_size,),
        generator=generator,
    )
    offsets = torch.arange(sequence_length + 1)
    sample = stream[starts[:, None] + offsets]
    return sample[:, :-1], sample[:, 1:]


__all__ = [
    "TINY_SHAKESPEARE_REVISION",
    "TINY_SHAKESPEARE_SHA256",
    "TINY_SHAKESPEARE_URL",
    "random_batch",
    "tiny_shakespeare_bytes",
    "wikitext_bytes",
]
