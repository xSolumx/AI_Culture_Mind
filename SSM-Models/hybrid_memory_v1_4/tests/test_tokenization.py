from __future__ import annotations

import pytest
import torch

from hybrid_memory_v1_4.tokenization import (
    ByteLevelBPETokenizer,
    EncodedText,
    RawByteTokenizer,
    tokenizer_fingerprint,
)

TEXT = "Once upon a time, Sol saw wind. 🌞\n\nSol saw wind again.\n"


def test_raw_byte_tokenizer_is_exact_and_accounts_for_utf8() -> None:
    tokenizer = RawByteTokenizer()
    encoded = tokenizer.encode(TEXT)
    assert encoded.raw_byte_count == len(TEXT.encode("utf-8"))
    assert encoded.token_count == encoded.raw_byte_count
    assert tokenizer.decode(encoded.token_ids.tolist()) == TEXT
    assert tokenizer.vocab_size == 256


def test_bytelevel_bpe_round_trip_offsets_and_serialization() -> None:
    training = TEXT * 20 + "The wind carried the same small story onward.\n" * 20
    tokenizer = ByteLevelBPETokenizer.train(training, vocab_size=320)
    encoded = tokenizer.encode(TEXT)
    assert tokenizer.decode(encoded.token_ids.tolist()) == TEXT
    assert int(encoded.token_byte_lengths.sum()) == len(TEXT.encode("utf-8"))
    assert encoded.token_count < RawByteTokenizer().encode(TEXT).token_count
    restored = ByteLevelBPETokenizer.from_serialized(tokenizer.serialized())
    restored_encoding = restored.encode(TEXT)
    torch.testing.assert_close(restored_encoding.token_ids, encoded.token_ids)
    assert tokenizer_fingerprint(restored) == tokenizer_fingerprint(tokenizer)


def test_encoded_text_rejects_incorrect_byte_accounting() -> None:
    with pytest.raises(ValueError, match="does not match"):
        EncodedText(
            token_ids=torch.tensor([1, 2]),
            token_byte_lengths=torch.tensor([1, 1]),
            raw_byte_count=3,
            text_sha256="x",
        )


def test_bytelevel_bpe_rejects_vocab_below_byte_alphabet() -> None:
    with pytest.raises(ValueError, match="at least 256"):
        ByteLevelBPETokenizer.train(TEXT, vocab_size=255)
