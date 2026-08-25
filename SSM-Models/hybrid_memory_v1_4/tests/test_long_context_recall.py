from __future__ import annotations

import torch

from hybrid_memory_v1_4.long_context_recall import (
    _continuation_log_probability,
    _recall_pair,
)
from hybrid_memory_v1_4.model import HybridMemoryConfig, HybridMemoryLM
from hybrid_memory_v1_4.tokenization import ByteLevelBPETokenizer, RawByteTokenizer


def test_recall_pair_has_exact_distance_and_equal_counterfactual_length() -> None:
    filler = "held out natural text " * 200
    for distance in (128, 256, 512, 1024):
        pair = _recall_pair(filler, distance=distance, example=3)
        assert pair["distance_raw_bytes"] == distance
        assert len(pair["matching_prompt"].encode()) == len(
            pair["mismatched_prompt"].encode()
        )
        assert pair["supported_value"] != pair["counterfactual_value"]


def test_recall_continuation_is_a_prefix_stable_token_boundary() -> None:
    filler = "held out natural text " * 100
    pair = _recall_pair(filler, distance=128, example=0)
    tokenizers = (
        RawByteTokenizer(),
        ByteLevelBPETokenizer.train(filler, vocab_size=320),
    )
    for tokenizer in tokenizers:
        prompt = tokenizer.encode(pair["matching_prompt"]).token_ids
        full = tokenizer.encode(
            pair["matching_prompt"] + pair["continuation"]
        ).token_ids
        assert full[: prompt.numel()].tolist() == prompt.tolist()


def test_continuation_scorer_returns_finite_value() -> None:
    torch.manual_seed(5)
    tokenizer = RawByteTokenizer()
    model = HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=256,
            model_dim=16,
            layer_plan=("attention",),
            attention_heads=2,
            expansion=1,
            use_local_conv=False,
        )
    )
    score, prompt_tokens, continuation_tokens = _continuation_log_probability(
        model,
        tokenizer,
        "The secret word was",
        " lion",
        torch.device("cpu"),
    )
    assert torch.isfinite(torch.tensor(score))
    assert prompt_tokens > 0
    assert continuation_tokens == 5
