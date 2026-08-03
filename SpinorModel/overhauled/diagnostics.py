"""Reproducible correctness and reference-throughput comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from SpinorModel.spinor_llm import SpinorLLM
from SpinorModel.overhauled.model import SpinorSSMConfig, SpinorSSMLanguageModel


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def tokens_per_second(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    repeats: int,
    backend: str | None = None,
) -> float:
    kwargs = {} if backend is None else {"backend": backend}
    model.eval()
    for _ in range(2):
        model(tokens, **kwargs)
    if tokens.is_cuda:
        torch.cuda.synchronize(tokens.device)
    start = time.perf_counter()
    for _ in range(repeats):
        model(tokens, **kwargs)
    if tokens.is_cuda:
        torch.cuda.synchronize(tokens.device)
    return tokens.numel() * repeats / (time.perf_counter() - start)


def run(
    *,
    device: torch.device,
    long_length: int = 2048,
    benchmark_length: int = 512,
    repeats: int = 5,
    seed: int = 0,
) -> dict[str, object]:
    torch.manual_seed(seed)
    vocabulary_size = 512
    original = SpinorLLM(
        vocabulary_size, num_layers=2, num_heads=2, dropout_rate=0.0
    ).to(device)
    matched = SpinorSSMLanguageModel(
        SpinorSSMConfig(
            vocab_size=vocabulary_size,
            channels=1,
            num_layers=2,
            dropout=0.0,
        )
    ).to(device)
    research_default = SpinorSSMLanguageModel(
        SpinorSSMConfig(vocab_size=vocabulary_size, dropout=0.0)
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(seed + 1)
    benchmark_tokens = torch.randint(
        1,
        vocabulary_size,
        (2, benchmark_length),
        generator=generator,
        device=device,
    )
    long_model = SpinorSSMLanguageModel(
        SpinorSSMConfig(
            vocab_size=64,
            channels=2,
            num_layers=1,
            dropout=0.0,
            max_half_life=2048.0,
        )
    ).to(device).eval()
    long_tokens = torch.randint(
        1, 64, (2, long_length), generator=generator, device=device
    )
    with torch.no_grad():
        parallel, parallel_state = long_model(
            long_tokens, return_recurrent_states=True, backend="parallel"
        )
        recurrent, recurrent_state = long_model(
            long_tokens, return_recurrent_states=True, backend="recurrent"
        )

    return {
        "seed": seed,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": str(next(long_model.parameters()).dtype),
        "long_length": long_length,
        "benchmark_batch": 2,
        "benchmark_length": benchmark_length,
        "benchmark_repeats": repeats,
        "original_parameters": parameter_count(original),
        "width_matched_overhaul_parameters": parameter_count(matched),
        "width_matched_overhaul_state_scalars": matched.recurrent_state_scalars,
        "research_default_parameters": parameter_count(research_default),
        "research_default_state_scalars": research_default.recurrent_state_scalars,
        "length_parallel_recurrent_logit_max_error": float(
            (parallel - recurrent).abs().max()
        ),
        "length_parallel_recurrent_state_max_error": max(
            float((left - right).abs().max())
            for left, right in zip(parallel_state, recurrent_state)
        ),
        "original_reference_tokens_per_second": tokens_per_second(
            original, benchmark_tokens, repeats=repeats
        ),
        "overhaul_reference_parallel_tokens_per_second": tokens_per_second(
            matched, benchmark_tokens, repeats=repeats, backend="parallel"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--long-length", type=int, default=2048)
    parser.add_argument("--benchmark-length", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    report = run(
        device=torch.device(device_name),
        long_length=args.long_length,
        benchmark_length=args.benchmark_length,
        repeats=args.repeats,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
