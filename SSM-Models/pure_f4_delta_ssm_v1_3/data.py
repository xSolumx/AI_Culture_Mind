"""Pinned, deterministic Tiny Shakespeare streams for v1.3 experiments."""

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
    """Load a pinned Tiny Shakespeare revision with a deterministic 90/5/5 split."""

    if split not in {"train", "validation", "test"}:
        raise ValueError("Tiny Shakespeare split must be train, validation, or test")
    root = cache_root
    if root is None:
        root = Path(
            os.environ.get(
                "PURE_F4_DATA_CACHE",
                Path.home() / ".cache" / "pure_f4_delta_ssm_v1_3",
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
            f"Tiny Shakespeare hash mismatch: expected {TINY_SHAKESPEARE_SHA256}, got {digest}"
        )
    selected = _split_tiny_shakespeare(payload)[split]
    array = np.frombuffer(selected, dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(array), hashlib.sha256(selected).hexdigest()


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
]
