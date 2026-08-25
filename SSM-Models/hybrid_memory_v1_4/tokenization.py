"""Lossless natural-text tokenization with raw-byte accounting.

Raw UTF-8 bytes remain the reference protocol.  The fitted alternative is a
Hugging Face ByteLevel BPE tokenizer trained on training text only.  Every
encoded stream carries the number of original UTF-8 bytes represented by each
token, allowing token cross entropy to remain comparable as bits per raw byte.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


@dataclass(frozen=True)
class EncodedText:
    """Token ids plus exact original-byte accounting for each token."""

    token_ids: torch.Tensor
    token_byte_lengths: torch.Tensor
    raw_byte_count: int
    text_sha256: str

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 1 or self.token_byte_lengths.ndim != 1:
            raise ValueError("encoded token tensors must be one-dimensional")
        if self.token_ids.shape != self.token_byte_lengths.shape:
            raise ValueError("token ids and byte lengths must have equal shape")
        if self.token_ids.dtype != torch.long:
            raise TypeError("token ids must use torch.long")
        if self.token_byte_lengths.dtype != torch.long:
            raise TypeError("token byte lengths must use torch.long")
        if bool((self.token_byte_lengths <= 0).any()):
            raise ValueError("every token must account for at least one raw byte")
        if int(self.token_byte_lengths.sum()) != self.raw_byte_count:
            raise ValueError("per-token byte accounting does not match raw text")

    @property
    def token_count(self) -> int:
        return self.token_ids.numel()

    @property
    def bytes_per_token(self) -> float:
        return self.raw_byte_count / self.token_count


class LosslessTextTokenizer(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> EncodedText: ...

    def decode(self, token_ids: list[int]) -> str: ...

    def serialized(self) -> str: ...


class RawByteTokenizer:
    """Unfitted UTF-8 byte tokenizer used by all earlier natural-text gates."""

    name = "raw_utf8_bytes"
    vocab_size = 256

    def encode(self, text: str) -> EncodedText:
        raw = text.encode("utf-8")
        return EncodedText(
            token_ids=torch.tensor(list(raw), dtype=torch.long),
            token_byte_lengths=torch.ones(len(raw), dtype=torch.long),
            raw_byte_count=len(raw),
            text_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def decode(self, token_ids: list[int]) -> str:
        if any(token < 0 or token >= self.vocab_size for token in token_ids):
            raise ValueError("raw byte token id is outside 0..255")
        return bytes(token_ids).decode("utf-8")

    def serialized(self) -> str:
        return json.dumps(
            {"type": type(self).__name__, "encoding": "UTF-8", "vocab_size": 256},
            sort_keys=True,
        )


class ByteLevelBPETokenizer:
    """Lossless fitted ByteLevel BPE with no special or unknown tokens."""

    name = "bytelevel_bpe"

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._tokenizer.decoder = decoders.ByteLevel()

    @classmethod
    def train(
        cls,
        training_text: str,
        *,
        vocab_size: int,
        min_frequency: int = 2,
    ) -> ByteLevelBPETokenizer:
        if vocab_size < 256:
            raise ValueError("ByteLevel BPE vocab_size must be at least 256")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True,
        )
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
            special_tokens=[],
        )
        tokenizer.train_from_iterator([training_text], trainer=trainer)
        result = cls(tokenizer)
        encoded = result.encode(training_text)
        if result.decode(encoded.token_ids.tolist()) != training_text:
            raise RuntimeError("trained ByteLevel BPE failed training-text round trip")
        return result

    @classmethod
    def from_serialized(cls, serialized: str) -> ByteLevelBPETokenizer:
        return cls(Tokenizer.from_str(serialized))

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> EncodedText:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        if not encoding.ids:
            raise ValueError("cannot encode empty text")
        # ByteLevel maps each one of the 256 possible bytes to exactly one
        # Unicode alphabet character before BPE merging.  A merged vocabulary
        # piece therefore represents exactly ``len(piece)`` original bytes.
        # This is more reliable than Encoding.offsets: upstream offsets are
        # character-oriented and can overlap for multi-byte Unicode input.
        byte_lengths = [len(piece) for piece in encoding.tokens]
        raw = text.encode("utf-8")
        if sum(byte_lengths) != len(raw):
            raise RuntimeError("ByteLevel vocabulary pieces do not cover every raw byte")
        result = EncodedText(
            token_ids=torch.tensor(encoding.ids, dtype=torch.long),
            token_byte_lengths=torch.tensor(byte_lengths, dtype=torch.long),
            raw_byte_count=len(raw),
            text_sha256=hashlib.sha256(raw).hexdigest(),
        )
        if self.decode(result.token_ids.tolist()) != text:
            raise RuntimeError("ByteLevel BPE failed exact text round trip")
        return result

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def serialized(self) -> str:
        return self._tokenizer.to_str(pretty=True)


def tokenizer_fingerprint(tokenizer: LosslessTextTokenizer) -> str:
    """Content hash of the complete tokenizer contract."""

    return hashlib.sha256(tokenizer.serialized().encode("utf-8")).hexdigest()


def tokenizer_report(
    tokenizer: LosslessTextTokenizer,
    train: EncodedText,
    validation: EncodedText,
) -> dict[str, Any]:
    train_round_trip = hashlib.sha256(
        tokenizer.decode(train.token_ids.tolist()).encode("utf-8")
    ).hexdigest()
    validation_round_trip = hashlib.sha256(
        tokenizer.decode(validation.token_ids.tolist()).encode("utf-8")
    ).hexdigest()
    return {
        "name": tokenizer.name,
        "runtime_class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "vocab_size": tokenizer.vocab_size,
        "sha256": tokenizer_fingerprint(tokenizer),
        "train": {
            "raw_bytes": train.raw_byte_count,
            "tokens": train.token_count,
            "bytes_per_token": train.bytes_per_token,
            "round_trip": train_round_trip == train.text_sha256,
        },
        "validation": {
            "raw_bytes": validation.raw_byte_count,
            "tokens": validation.token_count,
            "bytes_per_token": validation.bytes_per_token,
            "round_trip": validation_round_trip == validation.text_sha256,
        },
    }


__all__ = [
    "ByteLevelBPETokenizer",
    "EncodedText",
    "LosslessTextTokenizer",
    "RawByteTokenizer",
    "tokenizer_fingerprint",
    "tokenizer_report",
]
