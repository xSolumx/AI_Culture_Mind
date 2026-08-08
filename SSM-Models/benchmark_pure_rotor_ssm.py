"""Reproducible CUDA systems benchmark for pure rotor SSM v2."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from pure_rotor_ssm import __version__ as MODEL_VERSION
from pure_rotor_ssm.torch_backend import (
    GASSMLanguageModel,
    geometric_product,
    reversion,
    rotor_from_bivector,
    specialized_rotor_sandwich,
)


def timed_cuda(callable_, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return timings


def summarize(timings: list[float], tokens: int) -> dict[str, float]:
    median_ms = statistics.median(timings)
    ordered = sorted(timings)
    p95_ms = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "tokens_per_second": tokens * 1000.0 / median_ms,
    }


def load_or_create_model(args, device: torch.device):
    if args.checkpoint is None:
        config = {
            "vocab_size": args.vocab_size,
            "channels": args.channels,
            "num_layers": args.layers,
            "expansion": args.expansion,
            "dropout_rate": 0.0,
        }
        checkpoint_meta = None
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint["model_config"]
        checkpoint_meta = {
            "path": str(args.checkpoint),
            "model_version": checkpoint["model_version"],
            "variant": checkpoint["variant"],
        }
    model = GASSMLanguageModel(**config).to(device)
    if args.checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, config, checkpoint_meta


def model_benchmark(model, lengths, batch_size, warmup, iterations, device):
    results = []
    for length in lengths:
        tokens = torch.randint(
            0, model.vocab_size, (batch_size, length), device=device
        )
        for scan_mode in ("parallel", "recurrent"):
            model.eval()

            def inference(tokens=tokens, scan_mode=scan_mode):
                with torch.no_grad():
                    model(tokens, scan_mode=scan_mode)

            torch.cuda.reset_peak_memory_stats(device)
            inference_times = timed_cuda(inference, warmup, iterations)
            inference_result = summarize(
                inference_times, batch_size * length
            )
            inference_result["peak_memory_mib"] = (
                torch.cuda.max_memory_allocated(device) / 2**20
            )

            model.train()

            def forward_backward(tokens=tokens, scan_mode=scan_mode):
                model.zero_grad(set_to_none=True)
                logits = model(tokens, scan_mode=scan_mode)
                logits.float().square().mean().backward()

            torch.cuda.reset_peak_memory_stats(device)
            train_times = timed_cuda(forward_backward, warmup, iterations)
            train_result = summarize(train_times, batch_size * length)
            train_result["peak_memory_mib"] = (
                torch.cuda.max_memory_allocated(device) / 2**20
            )
            results.append(
                {
                    "batch_size": batch_size,
                    "sequence_length": length,
                    "scan_mode": scan_mode,
                    "inference": inference_result,
                    "forward_backward": train_result,
                }
            )
    return results


def rotor_kernel_benchmark(args, device):
    shape = (args.kernel_batch_size, args.kernel_length, args.channels)
    bivectors = torch.randn(*shape, 3, device=device)
    multivectors = torch.randn(*shape, 8, device=device)
    rotors = rotor_from_bivector(bivectors)

    def specialized():
        return specialized_rotor_sandwich(rotors, multivectors)

    def dense():
        return geometric_product(
            geometric_product(rotors, multivectors), reversion(rotors)
        )

    specialized_times = timed_cuda(specialized, args.warmup, args.iterations)
    dense_times = timed_cuda(dense, args.warmup, args.iterations)
    with torch.no_grad():
        error = float((specialized() - dense()).abs().max())
    specialized_median = statistics.median(specialized_times)
    dense_median = statistics.median(dense_times)
    return {
        "shape": list(shape) + [8],
        "selected_cuda_path": "dense_geometric_product",
        "specialized_median_ms": specialized_median,
        "dense_median_ms": dense_median,
        "dense_speedup_over_specialized": specialized_median / dense_median,
        "max_absolute_error": error,
    }


def parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(part) for part in value.split(","))
    if not lengths or any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be comma-separated positives")
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lengths", type=parse_lengths, default=(64, 256))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--kernel-batch-size", type=int, default=16)
    parser.add_argument("--kernel-length", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.warmup < 1 or args.iterations < 1 or args.batch_size < 1:
        raise ValueError("warmup, iterations, and batch size must be positive")

    torch.manual_seed(200)
    torch.cuda.manual_seed_all(200)
    device = torch.device("cuda")
    model, config, checkpoint_meta = load_or_create_model(args, device)
    report = {
        "model_version": MODEL_VERSION,
        "checkpoint": checkpoint_meta,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_config": config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "model_results": model_benchmark(
            model,
            args.lengths,
            args.batch_size,
            args.warmup,
            args.iterations,
            device,
        ),
        "rotor_kernel": rotor_kernel_benchmark(args, device),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
