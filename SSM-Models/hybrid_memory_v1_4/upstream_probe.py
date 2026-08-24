"""Reproducible probes for real upstream v1.4 comparison implementations."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .baselines import BASELINE_NAMES, baseline_availability, build_baseline
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.baselines import (  # type: ignore[no-redef]
        BASELINE_NAMES,
        baseline_availability,
        build_baseline,
    )

FLASHRT_REPOSITORY = "flashrt/gated-delta-attention"
FLASHRT_REVISION = "892f725c92033f8daf3de1329e1bba05b2747a39"
STATE_SPACES_REPOSITORY = "state-spaces/mamba2-130m"
STATE_SPACES_REVISION = "3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49"


def _environment() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capability = (
        torch.cuda.get_device_capability(device) if device.type == "cuda" else None
    )
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "compute_capability": capability,
    }


def _timed_forward(model: torch.nn.Module, tokens: torch.Tensor) -> dict[str, Any]:
    with torch.inference_mode():
        for _ in range(2):
            output = model(tokens, use_cache=False)
        if tokens.device.type == "cuda":
            torch.cuda.synchronize(tokens.device)
        started = time.perf_counter()
        for _ in range(10):
            output = model(tokens, use_cache=False)
        if tokens.device.type == "cuda":
            torch.cuda.synchronize(tokens.device)
    logits = output.logits if hasattr(output, "logits") else output["logits"]
    elapsed = time.perf_counter() - started
    return {
        "shape": list(logits.shape),
        "finite": bool(torch.isfinite(logits).all()),
        "mean_ms": elapsed * 100.0,
        "tokens_per_second": tokens.numel() * 10 / elapsed,
    }


def probe_native_transformers() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    availability = {
        name: baseline_availability(name, device, torch.float32).as_dict()
        for name in BASELINE_NAMES
    }
    common = {
        "vocab_size": 197,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "tie_word_embeddings": True,
        "use_cache": False,
        "pad_token_id": 0,
        "eos_token_id": 1,
    }
    mamba = build_baseline(
        "transformers_mamba2",
        device=device,
        dtype=torch.float32,
        **common,
        state_size=16,
        expand=2,
        head_dim=16,
        num_heads=8,
        n_groups=4,
        conv_kernel=4,
        chunk_size=64,
    )
    olmo = build_baseline(
        "transformers_olmo_hybrid",
        device=device,
        dtype=torch.float32,
        **common,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        max_position_embeddings=1024,
    )
    tokens = torch.randint(0, 197, (2, 128), device=device)
    models = {}
    for name, model in (
        ("transformers_mamba2", mamba),
        ("transformers_olmo_hybrid", olmo),
    ):
        models[name] = {
            "type": type(model).__name__,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "probe": _timed_forward(model, tokens),
        }

    sdpa: dict[str, Any] = {
        "flash_compiled": bool(torch.backends.cuda.is_flash_attention_available())
        if device.type == "cuda"
        else False,
        "flash_usable": False,
        "memory_efficient_usable": False,
    }
    if device.type == "cuda":
        q = torch.randn(2, 4, 128, 16, device=device)
        params = torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, True, False)
        sdpa["flash_usable"] = bool(torch.backends.cuda.can_use_flash_attention(params))
        sdpa["memory_efficient_usable"] = bool(
            torch.backends.cuda.can_use_efficient_attention(params)
        )
    return {
        "probe": "native_transformers",
        "environment": _environment(),
        "availability": availability,
        "models": models,
        "sdpa": sdpa,
        "timing_claim": "within-process smoke only; not a matched performance claim",
    }


def probe_fla() -> dict[str, Any]:
    import fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    if not torch.cuda.is_available():
        raise RuntimeError("FLA probe requires CUDA")
    rows = []
    for length in (128, 512, 2048):
        batch, heads, dimension = 2, 4, 16
        query = torch.randn(
            batch, length, heads, dimension, device="cuda", dtype=torch.float16
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        log_decay = -torch.nn.functional.softplus(
            torch.randn(batch, length, heads, device="cuda", dtype=torch.float32)
        )
        beta = torch.sigmoid(torch.randn_like(log_decay))
        for _ in range(2):
            output, state = chunk_gated_delta_rule(
                query,
                key,
                value,
                log_decay,
                beta,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            output, state = chunk_gated_delta_rule(
                query,
                key,
                value,
                log_decay,
                beta,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "length": length,
                "output_shape": list(output.shape),
                "state_shape": list(state.shape),
                "finite": bool(
                    torch.isfinite(output).all() and torch.isfinite(state).all()
                ),
                "mean_ms": elapsed * 50.0,
                "tokens_per_second": batch * length * 20 / elapsed,
            }
        )
    return {
        "probe": "fla_gated_delta_operator",
        "environment": _environment(),
        "fla_version": getattr(fla, "__version__", None),
        "rows": rows,
        "claim_boundary": "actual FLA operator smoke; not a complete language model",
        "timing_claim": "warm intra-operator timing in one WSL environment only",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_pretrained_mamba(model_path: Path) -> dict[str, Any]:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    if not torch.cuda.is_available():
        raise RuntimeError("pretrained Mamba-2 probe requires CUDA")
    weight_path = model_path / "pytorch_model.bin"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    started = time.perf_counter()
    model = MambaLMHeadModel.from_pretrained(
        str(model_path), device="cuda", dtype=torch.float16
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    rows = []
    for length in (128, 512):
        tokens = torch.randint(0, 50277, (1, length), device="cuda")
        with torch.inference_mode():
            for _ in range(2):
                output = model(tokens).logits
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(5):
                output = model(tokens).logits
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "length": length,
                "shape": list(output.shape),
                "finite": bool(torch.isfinite(output).all()),
                "mean_ms": elapsed * 200.0,
                "tokens_per_second": length * 5 / elapsed,
            }
        )
    return {
        "probe": "pretrained_mamba2",
        "environment": _environment(),
        "repository": STATE_SPACES_REPOSITORY,
        "revision": STATE_SPACES_REVISION,
        "weight_sha256": _sha256(weight_path),
        "weight_bytes": weight_path.stat().st_size,
        "model_type": type(model).__name__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "load_seconds": load_seconds,
        "rows": rows,
        "claim_boundary": "actual pretrained checkpoint inference; no MQAR adaptation",
        "timing_claim": "warm WSL inference smoke only; not cross-environment matched",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "probe", choices=("native-transformers", "fla", "pretrained-mamba")
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.probe == "native-transformers":
        report = probe_native_transformers()
    elif args.probe == "fla":
        report = probe_fla()
    else:
        if args.model_path is None:
            parser.error("--model-path is required for pretrained-mamba")
        report = probe_pretrained_mamba(args.model_path)
    report["upstream_pins"] = {
        FLASHRT_REPOSITORY: FLASHRT_REVISION,
        STATE_SPACES_REPOSITORY: STATE_SPACES_REVISION,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
