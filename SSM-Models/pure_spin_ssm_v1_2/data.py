"""Deterministic natural byte streams for matched causal-LM experiments."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset


def _cached_wikitext(split: str, cache_root: Path | None) -> Dataset:
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
