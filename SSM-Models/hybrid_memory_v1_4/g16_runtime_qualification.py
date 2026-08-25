"""Qualify the frozen G16 arms on the exact SM75 runtime without quality metrics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .frontier_shootout import (
    ARMS,
    MODEL_SEED,
    PREREGISTRATION,
    _build_optimizer,
    _forward_logits,
    _git,
    _mamba_source_provenance,
    _olmo_runtime_provenance,
    build_model,
)
from .natural_text_frontier import _sha256

BATCH_SIZE = 2
SEQUENCE_LENGTH = 64


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _paired_tokens() -> torch.Tensor:
    generator = torch.Generator().manual_seed(MODEL_SEED)
    return torch.randint(
        0,
        512,
        (BATCH_SIZE, SEQUENCE_LENGTH),
        generator=generator,
        dtype=torch.long,
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _qualify_arm(
    arm: str,
    tokens: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, metadata = build_model(arm, device)
    model.train()
    optimizer, optimizer_metadata = _build_optimizer(arm, model)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    logits = _forward_logits(arm, model, tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        tokens[:, 1:].reshape(-1),
    )
    logits_finite = bool(torch.isfinite(logits).all())
    loss_finite = bool(torch.isfinite(loss))
    loss.backward()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    gradients = [parameter.grad for parameter in trainable]
    complete_gradients = all(gradient is not None for gradient in gradients)
    finite_gradients = complete_gradients and all(
        bool(torch.isfinite(gradient).all())
        for gradient in gradients
        if gradient is not None
    )
    optimizer.step()
    finite_parameters_after_step = all(
        bool(torch.isfinite(parameter).all()) for parameter in trainable
    )
    torch.cuda.synchronize(device)
    passed = (
        logits_finite
        and loss_finite
        and complete_gradients
        and finite_gradients
        and finite_parameters_after_step
    )
    result = {
        "arm": arm,
        "passed": passed,
        "model": metadata,
        "optimizer": optimizer_metadata,
        "input_shape": list(tokens.shape),
        "logits_shape": list(logits.shape),
        "logits_finite": logits_finite,
        "loss_finite": loss_finite,
        "trainable_tensor_count": len(trainable),
        "gradient_tensor_count": sum(gradient is not None for gradient in gradients),
        "complete_gradients": complete_gradients,
        "finite_gradients": finite_gradients,
        "finite_parameters_after_step": finite_parameters_after_step,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "wall_seconds": time.perf_counter() - started,
    }
    del optimizer, model, logits, loss, gradients, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(
    *,
    output: Path,
    mamba_source_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("G16 qualification requires CUDA")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError("G16 qualification requires exact SM75")
    git_commit, git_status = _git()
    if git_status:
        raise RuntimeError("G16 qualification requires a clean committed worktree")
    mamba = _mamba_source_provenance(mamba_source_root)
    olmo = _olmo_runtime_provenance()
    tokens = _paired_tokens().to(device)
    token_digest = _tensor_digest(tokens)
    rows = [_qualify_arm(arm, tokens, device=device) for arm in ARMS]
    passed = all(row["passed"] for row in rows)
    report = {
        "schema_version": 1,
        "stage": "G16-runtime-qualification",
        "claim_status": "runtime qualification only; no training-quality metric",
        "passed": passed,
        "arms": rows,
        "pairing": {
            "model_seed": MODEL_SEED,
            "input_shape": [BATCH_SIZE, SEQUENCE_LENGTH],
            "input_sha256": token_digest,
        },
        "mamba_source": mamba,
        "olmo_runtime": olmo,
        "protocol": str(PREREGISTRATION),
        "protocol_sha256": _sha256(PREREGISTRATION),
        "source": str(Path(__file__).resolve()),
        "source_sha256": _sha256(Path(__file__).resolve()),
        "git_commit_at_start": git_commit,
        "git_status_at_start": git_status,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "explicit_nonclaims": [
            "not a language-quality result",
            "not a speed ranking",
            "not a G16 training artifact",
            "not a scaling result",
        ],
    }
    _atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mamba-source-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run(
        output=args.output,
        mamba_source_root=args.mamba_source_root,
        device=torch.device(args.device),
    )
    print(args.output)
    print(json.dumps({"passed": report["passed"], "arms": report["arms"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
