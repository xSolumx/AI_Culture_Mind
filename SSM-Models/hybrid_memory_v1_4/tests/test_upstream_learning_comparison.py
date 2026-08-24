from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_memory_v1_4.tasks import generate_mqar_batch
from hybrid_memory_v1_4.upstream_learning_comparison import (
    _gather_positions,
    externally_observable_losses,
)


def test_external_write_and_retrieval_losses_use_only_causal_positions() -> None:
    batch = generate_mqar_batch(2, 4, 3, 24, seed=17)
    vocab_size = batch.schema.vocabulary.vocab_size
    logits = torch.full((2, 24, vocab_size), -20.0)
    write_positions = batch.metadata["stored_key_positions"]
    stored_values = batch.metadata["stored_values"]
    assert isinstance(write_positions, torch.Tensor)
    assert isinstance(stored_values, torch.Tensor)
    logits.scatter_(
        2,
        batch.query_positions.unsqueeze(-1),
        0.0,
    )
    logits.scatter_(
        2,
        write_positions.unsqueeze(-1),
        0.0,
    )
    query_logits = _gather_positions(logits, batch.query_positions)
    write_logits = _gather_positions(logits, write_positions)
    query_logits.scatter_(2, batch.targets.unsqueeze(-1), 20.0)
    write_logits.scatter_(2, stored_values.unsqueeze(-1), 20.0)
    logits.scatter_(
        1,
        batch.query_positions.unsqueeze(-1).expand(-1, -1, vocab_size),
        query_logits,
    )
    logits.scatter_(
        1,
        write_positions.unsqueeze(-1).expand(-1, -1, vocab_size),
        write_logits,
    )
    retrieval, reconstruction = externally_observable_losses(logits, batch)
    assert float(retrieval) < 1e-6
    assert float(reconstruction) < 1e-6
