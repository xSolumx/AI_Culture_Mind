"""Local Spinor quality benchmark with the corrected decoder scaling."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from benchmark import Config, train_one, validation_batches, wiki_bytes
from spinor_delta_ssm import SpinorDeltaLM


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
    config = Config(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        local_kernel=0,
        decoder_scale=1.0,
    )
    train_array = wiki_bytes("train")
    valid_array = wiki_bytes("validation")
    train_tokens = torch.from_numpy(train_array)
    validation = validation_batches(valid_array, config)
    results = []
    for seed_text in args.seeds.split(","):
        seed = int(seed_text.strip())
        run_config = Config(**{**asdict(config), "seed": seed})
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        result = train_one(
            "spinor_isotypic_delta_v2",
            lambda: SpinorDeltaLM(
                256,
                run_config.channels,
                run_config.layers,
                run_config.expansion,
                decoder_channels=run_config.decoder_channels,
                local_kernel=run_config.local_kernel,
                decoder_scale=run_config.decoder_scale,
            ),
            train_tokens,
            validation,
            run_config,
            device,
            "native",
            not args.no_jit,
        )
        result["seed"] = seed
        results.append(result)
    report = {
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "training_tokens_per_run": config.steps * config.batch_size * config.seq_len,
        "integrity": {
            "model_initialized_after_seed": True,
            "python_numpy_torch_cuda_seeded": True,
            "local_kernel": config.local_kernel,
            "decoder_scale": config.decoder_scale,
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
