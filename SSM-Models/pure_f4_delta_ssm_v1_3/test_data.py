from __future__ import annotations

import hashlib

from .data import _split_tiny_shakespeare


def test_tiny_shakespeare_splits_are_disjoint_and_reconstruct_source() -> None:
    payload = bytes(range(100))
    pieces = _split_tiny_shakespeare(payload)

    assert len(pieces["train"]) == 90
    assert len(pieces["validation"]) == 5
    assert len(pieces["test"]) == 5
    assert b"".join(pieces[name] for name in ("train", "validation", "test")) == payload
    assert len({hashlib.sha256(piece).digest() for piece in pieces.values()}) == 3
