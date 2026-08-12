"""Train rotor and identity-transition SSMs on WikiText-2 bytes with CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from rotor_ssm_torch import GASSMLanguageModel


@dataclass(frozen=True)
class ExperimentConfig:
    steps: int = 200
    validation_batches: int = 20
    batch_size: int = 32
    seq_len: int = 64
    channels: int = 8
    layers: int = 2
    expansion: int = 2
    learning_rate: float = 3e-3
    seed: int = 0


def wiki_bytes(split: str) -> np.ndarray:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)


def fixed_validation_batches(
    tokens: np.ndarray, config: ExperimentConfig
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    required = config.validation_batches * config.batch_size * (config.seq_len + 1)
    data = tokens[:required].reshape(
        config.validation_batches, config.batch_size, config.seq_len + 1
    )
    return [
        (torch.from_numpy(batch[:, :-1]), torch.from_numpy(batch[:, 1:]))
        for batch in data
    ]


def random_batch(
    tokens: torch.Tensor,
    config: ExperimentConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        0,
        tokens.numel() - config.seq_len - 1,
        (config.batch_size,),
        generator=generator,
    )
    offsets = torch.arange(config.seq_len + 1)
    sequences = tokens[starts[:, None] + offsets]
    return sequences[:, :-1], sequences[:, 1:]


@torch.no_grad()
def evaluate(
    model: GASSMLanguageModel,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for inputs, targets in batches:
        logits = model(inputs.to(device))
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.to(device).flatten()
        )
        losses.append(float(loss))
    return float(np.mean(losses))


def run_variant(
    name: str,
    max_rotor_angle: float,
    train_tokens: torch.Tensor,
    validation: list[tuple[torch.Tensor, torch.Tensor]],
    config: ExperimentConfig,
    device: torch.device,
) -> dict:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = GASSMLanguageModel(
        vocab_size=256,
        channels=config.channels,
        num_layers=config.layers,
        expansion=config.expansion,
        dropout_rate=0.0,
        max_rotor_angle=max_rotor_angle,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    initial_loss = evaluate(model, validation, device)
    losses = []
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model.train()
    loss_samples = {}
    for step in range(1, config.steps + 1):
        inputs, targets = random_batch(train_tokens, config, generator)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 50 == 0:
            loss_samples[str(step)] = losses[-1]
            print(f"{name} step={step}/{config.steps} train_loss={losses[-1]:.4f}")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    validation_loss = evaluate(model, validation, device)
    diagnostic_inputs = validation[0][0].to(device)
    outputs = model.token_embeddings[diagnostic_inputs]
    transition_diagnostics = []
    with torch.no_grad():
        for block in model.blocks:
            normalized = block.norm1(outputs)
            decay, rotors, _ = block.ssm.transitions(normalized)
            angles = 2.0 * torch.acos(rotors[..., 0].clamp(-1.0, 1.0))
            transition_diagnostics.append(
                {
                    "mean_rotor_angle_radians": float(angles.mean()),
                    "p95_rotor_angle_radians": float(torch.quantile(angles, 0.95)),
                    "max_rotor_angle_radians": float(angles.max()),
                    "mean_decay": float(decay.mean()),
                    "min_decay": float(decay.min()),
                    "max_decay": float(decay.max()),
                    "rotor_control_weight_l2": float(block.ssm.rotor_control.weight.norm()),
                }
            )
            outputs, _ = block(outputs)
    return {
        "name": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_validation_loss": initial_loss,
        "final_validation_loss": validation_loss,
        "final_validation_bits_per_byte": validation_loss / math.log(2.0),
        "final_train_loss": losses[-1],
        "mean_last_20_train_loss": float(np.mean(losses[-20:])),
        "elapsed_seconds": elapsed,
        "steps_per_second": config.steps / elapsed,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "loss_samples": loss_samples,
        "transition_diagnostics": transition_diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this controlled local experiment")
    device = torch.device("cuda")
    config = ExperimentConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        channels=args.channels,
        layers=args.layers,
        seed=args.seed,
    )
    train_array = wiki_bytes("train")
    validation_array = wiki_bytes("validation")
    train_tokens = torch.from_numpy(train_array)
    validation = fixed_validation_batches(validation_array, config)
    results = [
        run_variant(
            "selective_rotor",
            math.pi / 2,
            train_tokens,
            validation,
            config,
            device,
        ),
        run_variant(
            "identity_rotation_ablation",
            0.0,
            train_tokens,
            validation,
            config,
            device,
        ),
    ]
    report = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "config": asdict(config),
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
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
