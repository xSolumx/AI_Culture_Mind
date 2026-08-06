"""Matched byte-level WikiText-2 benchmark for SpinorDeltaLM and Mamba-2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import Mamba2Config, Mamba2ForCausalLM

from spinor_delta_ssm import SpinorDeltaLM


@dataclass(frozen=True)
class Config:
    steps: int = 300
    batch_size: int = 16
    seq_len: int = 256
    # The 42x8 state uses a 20x8 (160-scalar) equivariant decoder bottleneck;
    # this keeps both parameter count and dominant logits FLOPs near Mamba-2.
    channels: int = 42
    decoder_channels: int = 20
    layers: int = 4
    expansion: int = 2
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0


def wiki_bytes(split: str) -> np.ndarray:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)


def validation_batches(tokens: np.ndarray, config: Config):
    required = 20 * config.batch_size * (config.seq_len + 1)
    data = tokens[:required].reshape(20, config.batch_size, config.seq_len + 1)
    return [
        (torch.from_numpy(batch[:, :-1]), torch.from_numpy(batch[:, 1:]))
        for batch in data
    ]


def random_batch(tokens: torch.Tensor, config: Config, generator: torch.Generator):
    starts = torch.randint(
        0,
        tokens.numel() - config.seq_len - 1,
        (config.batch_size,),
        generator=generator,
    )
    offsets = torch.arange(config.seq_len + 1)
    sequences = tokens[starts[:, None] + offsets]
    return sequences[:, :-1], sequences[:, 1:]


class CausalLogitsWrapper(torch.nn.Module):
    def __init__(self, model, is_mamba: bool):
        super().__init__()
        self.model = model
        self.is_mamba = is_mamba

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.is_mamba:
            return self.model(input_ids=inputs).logits
        return self.model(inputs)


def prepare_execution(
    model, is_mamba: bool, config: Config, device: torch.device, jit: bool
):
    execution = CausalLogitsWrapper(model, is_mamba).to(device)
    if not jit:
        return execution
    example = torch.zeros(
        config.batch_size, config.seq_len, dtype=torch.long, device=device
    )
    # Static-shape tracing removes Python dispatch and launches one CUDA graph
    # per tensorized scan round. Parameters remain shared with `model`, so the
    # original optimizer still updates the traced module's weights.
    return torch.jit.trace(execution, (example,), check_trace=False)


@torch.no_grad()
def evaluate(execution, batches, device: torch.device) -> float:
    execution.eval()
    losses = []
    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = execution(inputs)
        losses.append(
            float(
                torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), targets.flatten()
                )
            )
        )
    return float(np.mean(losses))


def make_mamba(config: Config) -> Mamba2ForCausalLM:
    # hidden_size * expand must equal num_heads * head_dim.
    return Mamba2ForCausalLM(
        Mamba2Config(
            vocab_size=256,
            hidden_size=160,
            state_size=16,
            num_hidden_layers=config.layers,
            num_heads=5,
            head_dim=64,
            expand=2,
            conv_kernel=4,
            n_groups=1,
            tie_word_embeddings=True,
            use_cache=False,
        )
    )


def train_one(
    name, make_model, train_tokens, validation, config, device, is_mamba, jit
):
    # Construct the model only after seeding.  Constructing it in the caller
    # made the advertised seed fail to control initialization, and Mamba then
    # inherited RNG state from the preceding Spinor model.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = make_model()
    print(f"starting {name} seed={config.seed}", flush=True)
    model = model.to(device)
    execution = prepare_execution(model, is_mamba, config, device, jit)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    initial = evaluate(execution, validation, device)
    losses = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    execution.train()
    for step in range(config.steps):
        inputs, targets = random_batch(train_tokens, config, generator)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = execution(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(
                f"{name} seed={config.seed} step={step + 1}/{config.steps}", flush=True
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final = evaluate(execution, validation, device)
    return {
        "name": name,
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_loss": initial,
        "final_loss": final,
        "perplexity": math.exp(final),
        "bits_per_byte": final / math.log(2.0),
        "final_train_loss": losses[-1],
        "mean_last_20_train_loss": float(np.mean(losses[-20:])),
        "elapsed_seconds": elapsed,
        "tokens_per_second": config.steps
        * config.batch_size
        * config.seq_len
        / elapsed,
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else 0.0
        ),
        "jit_trace": jit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    config = Config(steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len)
    train_array = wiki_bytes("train")
    valid_array = wiki_bytes("validation")
    train_tokens = torch.from_numpy(train_array)
    validation = validation_batches(valid_array, config)
    results = []
    for seed in (int(value) for value in args.seeds.split(",") if value.strip()):
        run_config = Config(**{**asdict(config), "seed": seed})
        results.append(
            train_one(
                "spinor_isotypic_delta",
                lambda: SpinorDeltaLM(
                    256,
                    run_config.channels,
                    run_config.layers,
                    run_config.expansion,
                    decoder_channels=run_config.decoder_channels,
                ),
                train_tokens,
                validation,
                run_config,
                device,
                False,
                not args.no_jit,
            )
        )
        results.append(
            train_one(
                "mamba2_transformers",
                lambda: make_mamba(run_config),
                train_tokens,
                validation,
                run_config,
                device,
                True,
                not args.no_jit,
            )
        )
    report = {
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "config": asdict(config),
        "training_tokens_per_run": config.steps * config.batch_size * config.seq_len,
        "integrity": {
            "model_initialized_after_seed": True,
            "python_numpy_torch_cuda_seeded": True,
            "mamba_fused_extension_available": False,
            "rotor_execution": "tensor_cuda_associative_scan",
        },
        "data": {
            "dataset": "wikitext/wikitext-2-raw-v1",
            "encoding": "UTF-8 bytes",
            "train_bytes": int(train_array.size),
            "validation_bytes": int(valid_array.size),
            "train_sha256": hashlib.sha256(train_array.tobytes()).hexdigest(),
            "validation_sha256": hashlib.sha256(valid_array.tobytes()).hexdigest(),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
