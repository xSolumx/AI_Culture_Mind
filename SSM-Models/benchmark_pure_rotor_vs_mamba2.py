"""Matched byte-level benchmark for Pure Rotor SSM, its identity ablation, and Mamba-2.

This is an experiment runner, not part of :mod:`pure_rotor_ssm`.  It compares
the maintained model with the publicly deployed/selective Mamba-2 architecture
through Hugging Face Transformers on the identical byte stream.  On Windows it
may use Transformers' unfused reference path; that fact is written to every
report and must not be confused with the official fused ``mamba_ssm`` kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from pure_rotor_ssm import __version__ as PURE_ROTOR_VERSION
from pure_rotor_ssm.torch_backend import GASSMLanguageModel
from transformers import Mamba2Config, Mamba2ForCausalLM


@dataclass(frozen=True)
class BenchmarkConfig:
    """Small, parameter-near configuration for an architectural screening run."""

    steps: int = 300
    batch_size: int = 32
    sequence_length: int = 128
    validation_batches: int = 20
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    rotor_channels: int = 20
    rotor_layers: int = 3
    rotor_expansion: int = 2
    mamba_hidden_size: int = 96
    mamba_layers: int = 2
    mamba_heads: int = 4
    mamba_head_dim: int = 48
    mamba_state_size: int = 16
    seed: int = 0


def cached_wikitext_split(split: str, cache_directory: Path | None = None) -> Dataset:
    """Load a cached dataset Arrow shard without contacting the Hub."""

    root = cache_directory or Path.home() / ".cache" / "huggingface" / "datasets"
    candidates = tuple(
        root.glob(f"wikitext/wikitext-2-raw-v1/*/*/wikitext-{split}.arrow")
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one cached WikiText-2 {split!r} Arrow shard under {root}; "
            "omit --offline to allow a download or pass --dataset-cache-directory"
        )
    return Dataset.from_file(str(candidates[0]))


def wiki_bytes(
    split: str,
    *,
    offline: bool = False,
    cache_directory: Path | None = None,
) -> np.ndarray:
    """Return the immutable UTF-8 byte stream used by every candidate."""

    dataset = (
        cached_wikitext_split(split, cache_directory)
        if offline
        else load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    )
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)


def fixed_validation_batches(
    tokens: np.ndarray, config: BenchmarkConfig
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    required = (
        config.validation_batches * config.batch_size * (config.sequence_length + 1)
    )
    if tokens.size < required:
        raise ValueError("validation byte stream is too short for the configuration")
    data = tokens[:required].reshape(
        config.validation_batches, config.batch_size, config.sequence_length + 1
    )
    return [
        (torch.from_numpy(batch[:, :-1]), torch.from_numpy(batch[:, 1:]))
        for batch in data
    ]


def random_batch(
    tokens: torch.Tensor, config: BenchmarkConfig, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.numel() <= config.sequence_length:
        raise ValueError("training byte stream is too short for the configuration")
    starts = torch.randint(
        0,
        tokens.numel() - config.sequence_length - 1,
        (config.batch_size,),
        generator=generator,
    )
    offsets = torch.arange(config.sequence_length + 1)
    sequences = tokens[starts[:, None] + offsets]
    return sequences[:, :-1], sequences[:, 1:]


def mamba2_model(config: BenchmarkConfig) -> Mamba2ForCausalLM:
    """Construct the public Mamba-2 architecture without pretrained weights."""

    if config.mamba_hidden_size * 2 != config.mamba_heads * config.mamba_head_dim:
        raise ValueError("Mamba-2 expand=2 requires 2*hidden_size = heads*head_dim")
    return Mamba2ForCausalLM(
        Mamba2Config(
            vocab_size=256,
            hidden_size=config.mamba_hidden_size,
            state_size=config.mamba_state_size,
            num_hidden_layers=config.mamba_layers,
            num_heads=config.mamba_heads,
            head_dim=config.mamba_head_dim,
            expand=2,
            conv_kernel=4,
            n_groups=1,
            tie_word_embeddings=True,
            use_cache=False,
        )
    )


def pure_rotor_model(
    config: BenchmarkConfig, *, max_rotor_angle: float
) -> GASSMLanguageModel:
    return GASSMLanguageModel(
        vocab_size=256,
        channels=config.rotor_channels,
        num_layers=config.rotor_layers,
        expansion=config.rotor_expansion,
        dropout_rate=0.0,
        max_rotor_angle=max_rotor_angle,
    )


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def identity_ablation_metadata(model: GASSMLanguageModel) -> dict[str, int | bool]:
    """Expose the known raw-versus-effective capacity issue of this ablation."""

    disabled = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".ssm.rotor_control." in name
    )
    raw = parameter_count(model)
    return {
        "raw_parameter_count": raw,
        "disabled_rotor_controller_parameter_count": disabled,
        "effective_parameter_count_if_rotation_is_fixed_identity": raw - disabled,
        "raw_parameter_match_is_not_effective_capacity_match": True,
    }


def decoder_tying_metadata(
    pure_rotor: GASSMLanguageModel, mamba2: Mamba2ForCausalLM
) -> dict[str, bool]:
    """Record a capacity-relevant decoder choice instead of assuming parity."""

    mamba_input = mamba2.get_input_embeddings()
    mamba_output = mamba2.get_output_embeddings()
    return {
        "pure_rotor_input_output_embeddings_tied": True,
        "mamba2_input_output_embeddings_tied": (
            mamba_input.weight.data_ptr() == mamba_output.weight.data_ptr()
        ),
    }


def build_models(config: BenchmarkConfig) -> dict[str, torch.nn.Module]:
    """Build all candidates and reject a misleadingly mismatched comparison."""

    models: dict[str, torch.nn.Module] = {
        "pure_rotor": pure_rotor_model(config, max_rotor_angle=math.pi),
        "identity_rotation_ablation": pure_rotor_model(config, max_rotor_angle=0.0),
        "mamba2_transformers": mamba2_model(config),
    }
    rotor_parameters = parameter_count(models["pure_rotor"])
    mamba_parameters = parameter_count(models["mamba2_transformers"])
    relative_gap = abs(rotor_parameters - mamba_parameters) / max(
        rotor_parameters, mamba_parameters
    )
    if relative_gap > 0.05:
        raise ValueError(
            "configured Pure Rotor and Mamba-2 parameter counts differ by more than 5%"
        )
    return models


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def logits_for(
    name: str,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    rotor_scan_mode: str,
) -> torch.Tensor:
    if name == "mamba2_transformers":
        return model(input_ids=inputs, use_cache=False).logits
    return model(inputs, scan_mode=rotor_scan_mode)


@torch.no_grad()
def evaluate(
    name: str,
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    rotor_scan_mode: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = logits_for(name, model, inputs, rotor_scan_mode)
        total_loss += float(
            torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="sum"
            )
        )
        total_tokens += targets.numel()
    return total_loss / total_tokens


def checkpoint_payload(
    name: str, model: torch.nn.Module, config: BenchmarkConfig, result: dict
) -> dict:
    return {
        "format_version": 1,
        "candidate": name,
        "pure_rotor_model_version": PURE_ROTOR_VERSION,
        "benchmark_config": asdict(config),
        "metrics": result,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }


def train_one(
    name: str,
    make_model: Callable[[], torch.nn.Module],
    train_tokens: torch.Tensor,
    validation: list[tuple[torch.Tensor, torch.Tensor]],
    config: BenchmarkConfig,
    device: torch.device,
    rotor_scan_mode: str,
    checkpoint_directory: Path | None,
) -> dict:
    seed_everything(config.seed)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    initial_loss = evaluate(name, model, validation, device, rotor_scan_mode)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    loss_samples: dict[str, float] = {}
    start = time.perf_counter()
    for step in range(1, config.steps + 1):
        inputs, targets = random_batch(train_tokens, config, generator)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = logits_for(name, model, inputs, rotor_scan_mode)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == config.steps:
            loss_samples[str(step)] = float(loss.detach())
            print(
                f"{name} seed={config.seed} step={step}/{config.steps} "
                f"loss={loss_samples[str(step)]:.5f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final_loss = evaluate(name, model, validation, device, rotor_scan_mode)
    result = {
        "name": name,
        "parameters": parameter_count(model),
        "initial_validation_nll": initial_loss,
        "final_validation_nll": final_loss,
        "final_validation_bits_per_byte": final_loss / math.log(2.0),
        "final_train_loss": loss_samples[str(config.steps)],
        "loss_samples": loss_samples,
        "last_preclip_gradient_norm": float(gradient_norm),
        "elapsed_seconds": elapsed,
        "tokens_per_second": config.steps
        * config.batch_size
        * config.sequence_length
        / elapsed,
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else 0.0
        ),
    }
    if checkpoint_directory is not None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        path = checkpoint_directory / f"{name}_seed{config.seed}_step{config.steps}.pt"
        torch.save(checkpoint_payload(name, model, config, result), path)
        result["checkpoint"] = str(path)
        result["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rotor-scan-mode",
        choices=("parallel", "schur_parallel", "recurrent"),
        default="parallel",
    )
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="load cached Arrow shards directly without contacting the Hub",
    )
    parser.add_argument(
        "--dataset-cache-directory",
        type=Path,
        help="Hugging Face datasets cache root used with --offline",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.steps, args.batch_size, args.sequence_length, args.validation_batches)
        < 1
    ):
        raise ValueError(
            "steps, batch size, sequence length, and validation batches must be positive"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = BenchmarkConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        validation_batches=args.validation_batches,
    )
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    device = torch.device(args.device)
    train_array = wiki_bytes(
        "train", offline=args.offline, cache_directory=args.dataset_cache_directory
    )
    validation_array = wiki_bytes(
        "validation", offline=args.offline, cache_directory=args.dataset_cache_directory
    )
    train_tokens = torch.from_numpy(train_array)
    validation = fixed_validation_batches(validation_array, config)
    # Construct once only to report the fixed matched configuration, then train
    # freshly seeded candidates in ``train_one``.
    model_shapes = build_models(config)
    counts = {name: parameter_count(model) for name, model in model_shapes.items()}
    identity_metadata = identity_ablation_metadata(
        model_shapes["identity_rotation_ablation"]
    )
    decoder_metadata = decoder_tying_metadata(
        model_shapes["pure_rotor"], model_shapes["mamba2_transformers"]
    )
    results = []
    for seed in seeds:
        run_config = BenchmarkConfig(**{**asdict(config), "seed": seed})
        factories = {
            "pure_rotor": lambda config=run_config: pure_rotor_model(
                config, max_rotor_angle=math.pi
            ),
            "identity_rotation_ablation": lambda config=run_config: pure_rotor_model(
                config, max_rotor_angle=0.0
            ),
            "mamba2_transformers": lambda config=run_config: mamba2_model(config),
        }
        for name, factory in factories.items():
            results.append(
                train_one(
                    name,
                    factory,
                    train_tokens,
                    validation,
                    run_config,
                    device,
                    args.rotor_scan_mode,
                    args.checkpoint_directory,
                )
            )
    report = {
        "experiment": "pure rotor SSM versus Mamba-2 byte-level comparison",
        "status": "completed empirical run; insufficient for a language-model superiority claim",
        "pure_rotor_model_version": PURE_ROTOR_VERSION,
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "config": asdict(config),
        "rotor_scan_mode": args.rotor_scan_mode,
        "integrity": {
            "shared_utf8_byte_vocabulary": True,
            "shared_random_windows_per_seed": True,
            "candidate_initialized_after_seed": True,
            "mamba2_backend": "huggingface_transformers",
            "mamba_ssm_extension_importable": importlib.util.find_spec("mamba_ssm")
            is not None,
            "mamba2_fused_kernel_claimed": False,
            "identity_rotation_ablation": identity_metadata,
            "decoder_tying": decoder_metadata,
            "dataset_offline_mode": args.offline,
            "parameter_counts": counts,
        },
        "data": {
            "dataset": "wikitext/wikitext-2-raw-v1",
            "encoding": "UTF-8 bytes",
            "train_bytes": int(train_array.size),
            "validation_bytes": int(validation_array.size),
            "train_sha256": hashlib.sha256(train_array.tobytes()).hexdigest(),
            "validation_sha256": hashlib.sha256(validation_array.tobytes()).hexdigest(),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
