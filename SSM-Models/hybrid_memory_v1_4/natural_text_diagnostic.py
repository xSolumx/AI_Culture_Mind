"""Post-hoc causal mixer diagnostic for the retained G11 v1.4.5 model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .natural_text_screen import (
        SNAPSHOT_SHA256,
        _batch,
        _build_model,
        _evaluate,
        _load_streams,
        _sha256,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.natural_text_screen import (  # type: ignore[no-redef]
        SNAPSHOT_SHA256,
        _batch,
        _build_model,
        _evaluate,
        _load_streams,
        _sha256,
    )

MODEL_NAME = "hybrid_v1_4_5"
CONTROL_BATCHES = 8


def _evaluate_ablation(
    model: torch.nn.Module,
    validation: torch.Tensor,
    device: torch.device,
    disabled_block: int | None,
) -> dict[str, float | int | bool]:
    scales = [block.residual_scale.detach().clone() for block in model.blocks]
    if disabled_block is not None:
        model.blocks[disabled_block].residual_scale.data.fill_(-30.0)
    result = _evaluate(MODEL_NAME, model, validation, device)
    for block, scale in zip(model.blocks, scales, strict=True):
        block.residual_scale.data.copy_(scale)
    return result


def _control_summary(
    model: torch.nn.Module,
    validation: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    write_sum: torch.Tensor | None = None
    retention_sum: torch.Tensor | None = None
    read_norm = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for batch_index in range(CONTROL_BATCHES):
            inputs, _ = _batch(
                validation,
                namespace="g11-validation",
                batch_index=batch_index,
                batch_size=16,
                device=device,
            )
            output = model(inputs, return_diagnostics=True)
            diagnostic = output["diagnostics"][0]
            write = diagnostic["write_strength"].float()
            retention = diagnostic["retention"].float()
            read = diagnostic["read"].float()
            per_head_write = write.sum(dim=(0, 1))
            per_head_retention = retention.sum(dim=(0, 1))
            write_sum = (
                per_head_write if write_sum is None else write_sum + per_head_write
            )
            retention_sum = (
                per_head_retention
                if retention_sum is None
                else retention_sum + per_head_retention
            )
            read_norm += float(read.norm(dim=-1).sum())
            count += write.shape[0] * write.shape[1]
    assert write_sum is not None and retention_sum is not None
    return {
        "batches": CONTROL_BATCHES,
        "scored_input_bytes": count,
        "mean_write_strength_per_head": (write_sum / count).tolist(),
        "mean_retention_per_head": (retention_sum / count).tolist(),
        "mean_read_norm": read_norm / count,
        "residual_scales": [
            {
                "raw": float(block.residual_scale.detach()),
                "sigmoid": float(torch.sigmoid(block.residual_scale).detach()),
            }
            for block in model.blocks
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    started = time.perf_counter()
    _, validation, snapshot_report = _load_streams(args.snapshot)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload.get("model_name") != MODEL_NAME:
        raise ValueError("checkpoint is not the G11 v1.4.5 model")
    if payload.get("snapshot_sha256") != SNAPSHOT_SHA256:
        raise ValueError("checkpoint does not name the frozen G11 snapshot")
    model = _build_model(MODEL_NAME, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    full = _evaluate_ablation(model, validation, device, None)
    without_gated_delta = _evaluate_ablation(model, validation, device, 0)
    without_attention = _evaluate_ablation(model, validation, device, 1)
    report = {
        "schema_version": 1,
        "claim_status": "post-hoc causal mixer diagnostic on frozen G11 model",
        "model_name": MODEL_NAME,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "snapshot": str(args.snapshot),
        "snapshot_sha256": _sha256(args.snapshot),
        "dataset_hub_sha": snapshot_report["hub_sha_at_snapshot"],
        "ablations": {
            "full": full,
            "without_gated_delta": without_gated_delta,
            "without_attention": without_attention,
        },
        "bits_per_byte_increase_without_gated_delta": (
            without_gated_delta["bits_per_byte"] - full["bits_per_byte"]
        ),
        "bits_per_byte_increase_without_attention": (
            without_attention["bits_per_byte"] - full["bits_per_byte"]
        ),
        "controls": _control_summary(model, validation, device),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "post-hoc ablation of one frozen single-seed TinyStories model; "
            "not a preregistered architecture comparison or general quality claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps(report["ablations"], sort_keys=True))


if __name__ == "__main__":
    main()
