"""Measure temporal query-control observability from final retrieval loss."""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark import file_sha256, package_version, seed_all
from spin_delta_capability_gate import overwrite_retrieval_batch
from spin_delta_causal_router_gate import build_candidate
from spin_delta_write_curriculum_gate import state_digest

ROOT = Path(__file__).resolve().parent
SEEDS = (761, 769, 773)
DEPTHS = (1, 2)
MODES = ("hard_fallback", "soft_query_event", "authoritative_query")
WRITES = 8
BATCH_SIZE = 64
PROTOCOL = "SPIN_DELTA_TEMPORAL_OBSERVABILITY_PREREGISTRATION.md"


@dataclass(frozen=True)
class AuditConfig:
    init_seed: int
    data_seed: int
    layers: int
    d_model: int = 64
    router_width: int = 32
    router_kernel_size: int = 3
    router_temperature: float = 1.0


def position_roles(length: int) -> list[str]:
    if length != 3 * WRITES + 2:
        raise ValueError("unexpected overwrite-retrieval length")
    roles = []
    for index in range(length):
        if index == length - 2:
            roles.append("query_marker")
        elif index == length - 1:
            roles.append("final_query_key")
        else:
            roles.append(("write_marker", "write_key", "write_value")[index % 3])
    return roles


def _controls(model, routing, mode: str):
    controls = routing.controls
    if mode == "hard_fallback":
        return controls
    if mode == "soft_query_event":
        event = torch.sigmoid(
            routing.query_event_logits / model.router.temperature
        ).unsqueeze(-1)
    elif mode == "authoritative_query":
        event = torch.ones_like(routing.query_event_logits).unsqueeze(-1)
    else:
        raise ValueError(mode)
    return torch.cat((controls[..., :3], event, controls[..., 4:]), dim=-1)


def _gradient_or_zero(gradient, reference):
    return torch.zeros_like(reference) if gradient is None else gradient


def _aligned_mass(gradient: torch.Tensor, desired_sign: torch.Tensor) -> float:
    absolute = gradient.abs()
    total = absolute.sum()
    if float(total) == 0.0:
        return 0.0
    aligned = absolute[(gradient * desired_sign) > 0].sum()
    return float(aligned / total)


def probe_path(model, inputs, target, oracle, mode: str):
    model.zero_grad(set_to_none=True)
    routing = model.router(inputs)
    controls = _controls(model, routing, mode)
    result = model.core(
        inputs,
        scan_mode="raw_cuda_delta",
        delta_router_controls=controls,
    )
    logits = result["logits"][:, -1]
    loss = F.cross_entropy(logits, target)
    event_gradient, slot_gradient = torch.autograd.grad(
        loss,
        (routing.query_event_logits, routing.query_slot_logits),
        allow_unused=True,
    )
    event_gradient = _gradient_or_zero(event_gradient, routing.query_event_logits)
    slot_gradient = _gradient_or_zero(slot_gradient, routing.query_slot_logits)
    event_abs = event_gradient.abs().mean(dim=0)
    slot_norm = slot_gradient.norm(dim=-1).mean(dim=0)
    final_target = oracle[:, -1, 1]
    correct = slot_gradient[:, -1].gather(1, final_target[:, None]).squeeze(1)
    wrong = slot_gradient[:, -1].gather(1, (1 - final_target)[:, None]).squeeze(1)
    roles = position_roles(inputs.shape[1])
    role_metrics = {}
    for role in dict.fromkeys(roles):
        indices = [index for index, value in enumerate(roles) if value == role]
        role_metrics[role] = {
            "event_gradient_mean_absolute": float(event_abs[indices].mean()),
            "slot_gradient_mean_norm": float(slot_norm[indices].mean()),
        }
    desired_nonfinal = torch.ones_like(event_gradient[:, :-1])
    desired_final = -torch.ones_like(event_gradient[:, -1:])
    return {
        "mode": mode,
        "loss": float(loss.detach()),
        "hard_query_event_maximum": float(routing.controls[..., 3].detach().max()),
        "event_gradient_mean_absolute_by_position": event_abs.detach().tolist(),
        "slot_gradient_mean_norm_by_position": slot_norm.detach().tolist(),
        "nonfinal_event_gradient_maximum_absolute": float(event_abs[:-1].max()),
        "nonfinal_slot_gradient_maximum_norm": float(slot_norm[:-1].max()),
        "final_event_gradient_mean_absolute": float(event_abs[-1]),
        "final_slot_gradient_mean_norm": float(slot_norm[-1]),
        "nonfinal_event_off_aligned_mass": _aligned_mass(
            event_gradient[:, :-1], desired_nonfinal
        ),
        "final_event_on_aligned_mass": _aligned_mass(
            event_gradient[:, -1:], desired_final
        ),
        "final_correct_slot_descent_margin": float((wrong - correct).mean()),
        "role_metrics": role_metrics,
        "finite": bool(
            torch.isfinite(logits).all()
            and torch.isfinite(event_gradient).all()
            and torch.isfinite(slot_gradient).all()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows = []
    for seed in SEEDS:
        generator = torch.Generator().manual_seed(1_330_000 + seed)
        inputs, target, oracle = overwrite_retrieval_batch(
            BATCH_SIZE,
            WRITES,
            generator=generator,
            device=device,
            return_oracle=True,
        )
        for depth in DEPTHS:
            config = AuditConfig(seed, 1_330_000 + seed, depth)
            seed_all(seed)
            model = build_candidate(config).to(device)
            source_digest = state_digest(model)
            paths = [probe_path(model, inputs, target, oracle, mode) for mode in MODES]
            if state_digest(model) != source_digest:
                raise RuntimeError("read-only Jacobian probe changed model state")
            rows.append(
                {
                    "seed": seed,
                    "depth": depth,
                    "source_state_sha256": source_digest,
                    "paths": paths,
                }
            )
    implementation_paths = (
        Path(__file__),
        ROOT / "spin_delta_router.py",
        ROOT / "model.py",
        ROOT / "spin_delta_scan.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "stage": "spin_delta_temporal_observability",
        "protocol": PROTOCOL,
        "seeds": list(SEEDS),
        "depths": list(DEPTHS),
        "modes": list(MODES),
        "writes": WRITES,
        "batch_size": BATCH_SIZE,
        "position_roles": position_roles(3 * WRITES + 2),
        "rows": rows,
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "triton": package_version("triton", "triton-windows"),
        },
        "implementation_sha256": {
            path.relative_to(ROOT).as_posix(): file_sha256(path)
            for path in implementation_paths
        },
    }
    if not all(
        path["finite"]
        and all(
            math.isfinite(path[name])
            for name in (
                "loss",
                "nonfinal_event_gradient_maximum_absolute",
                "nonfinal_slot_gradient_maximum_norm",
                "final_event_gradient_mean_absolute",
                "final_slot_gradient_mean_norm",
                "nonfinal_event_off_aligned_mass",
                "final_event_on_aligned_mass",
                "final_correct_slot_descent_margin",
            )
        )
        for row in rows
        for path in row["paths"]
    ):
        raise FloatingPointError("nonfinite temporal observability metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
