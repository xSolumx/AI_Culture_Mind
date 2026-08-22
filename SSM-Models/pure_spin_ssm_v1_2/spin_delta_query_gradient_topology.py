"""Audit hard, soft, and authoritative Spin-Delta query gradient paths."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark import file_sha256, package_version, seed_all
from spin_delta_capability_gate import overwrite_retrieval_batch
from spin_delta_causal_router_gate import build_candidate
from spin_delta_label_free_curriculum import LabelFreeConfig
from spin_delta_write_curriculum_gate import state_digest

ROOT = Path(__file__).resolve().parent
SEEDS = (691, 701, 709)
MODES = ("hard_fallback", "soft_query_event", "authoritative_query")
PROTOCOL = "SPIN_DELTA_QUERY_GRADIENT_TOPOLOGY_PREREGISTRATION.md"


def _straight_through_binary(logit: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logit)
    hard = (probability >= 0.5).to(probability)
    return hard.detach() - probability.detach() + probability


def _straight_through_slot(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits, dim=-1)
    hard = F.one_hot(probability.argmax(dim=-1), 2).to(probability)
    return hard.detach() - probability.detach() + probability


def local_audit(mode: str) -> dict[str, object]:
    event_logit = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    slot_logits = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    internal_query = torch.tensor([0.35, 0.65], dtype=torch.float64)
    slot = _straight_through_slot(slot_logits)
    if mode == "hard_fallback":
        event = _straight_through_binary(event_logit)
    elif mode == "soft_query_event":
        event = torch.sigmoid(event_logit)
    elif mode == "authoritative_query":
        event = torch.ones_like(event_logit)
    else:
        raise ValueError(mode)
    query = event * slot + (1.0 - event) * internal_query
    score = query @ torch.tensor([0.8, -0.3], dtype=torch.float64)
    loss = (score - 0.17).square()
    event_gradient, slot_gradient = torch.autograd.grad(
        loss, (event_logit, slot_logits), allow_unused=True
    )
    event_value = float(event.detach())
    return {
        "mode": mode,
        "event_value": event_value,
        "score": float(score.detach()),
        "loss": float(loss.detach()),
        "event_logit_gradient": (
            0.0 if event_gradient is None else float(event_gradient)
        ),
        "slot_gradient": slot_gradient.detach().tolist(),
        "slot_gradient_norm": float(slot_gradient.detach().norm()),
    }


def _clone(source, config, device):
    seed_all(config.init_seed)
    model = build_candidate(config).to(device)
    model.load_state_dict(source.state_dict())
    return model


def _head_gradient_norm(model, rows: tuple[int, ...]) -> float:
    weight = model.router.output.weight.grad[list(rows)]
    bias = model.router.output.bias.grad[list(rows)]
    return float(torch.sqrt(weight.square().sum() + bias.square().sum()))


def full_model_path(mode, source, config, inputs, target, device):
    model = _clone(source, config, device)
    model.train()
    model.zero_grad(set_to_none=True)
    routing = model.router(inputs)
    controls = routing.controls
    if mode == "soft_query_event":
        soft_query = torch.sigmoid(routing.query_event_logits).unsqueeze(-1)
        controls = torch.cat(
            (controls[..., :3], soft_query, controls[..., 4:6]), dim=-1
        )
    elif mode == "authoritative_query":
        query_on = torch.ones_like(routing.query_event_logits).unsqueeze(-1)
        controls = torch.cat((controls[..., :3], query_on, controls[..., 4:6]), dim=-1)
    elif mode != "hard_fallback":
        raise ValueError(mode)
    result = model.core(
        inputs,
        scan_mode="raw_cuda_delta",
        delta_router_controls=controls,
    )
    logits = result["logits"][:, -1]
    loss = F.cross_entropy(logits, target)
    loss.backward()
    return {
        "mode": mode,
        "loss": float(loss.detach()),
        "final_query_event_mean": float(controls[:, -1, 3].detach().mean()),
        "query_event_gradient_norm": _head_gradient_norm(model, (3,)),
        "query_slot_gradient_norm": _head_gradient_norm(model, (4, 5)),
        "finite_logits": bool(torch.isfinite(logits).all()),
        "finite_router_gradients": bool(
            torch.isfinite(model.router.output.weight.grad).all()
            and torch.isfinite(model.router.output.bias.grad).all()
        ),
        "logits": logits.detach(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    local = [local_audit(mode) for mode in MODES]
    rows = []
    for seed in SEEDS:
        config = LabelFreeConfig(seed, 1_270_000 + seed)
        seed_all(seed)
        source = build_candidate(config).to(device)
        source_sha256 = state_digest(source)
        generator = torch.Generator().manual_seed(1_290_000 + seed)
        inputs, target = overwrite_retrieval_batch(
            128,
            2,
            generator=generator,
            device=device,
        )
        paths = [
            full_model_path(mode, source, config, inputs, target, device)
            for mode in MODES
        ]
        hard_logits = paths[0].pop("logits")
        for path in paths[1:]:
            logits = path.pop("logits")
            path["maximum_absolute_logit_change_from_hard"] = float(
                (logits - hard_logits).abs().max()
            )
        paths[0]["maximum_absolute_logit_change_from_hard"] = 0.0
        if any(
            state_digest(_clone(source, config, device)) != source_sha256
            for _ in range(2)
        ):
            raise RuntimeError("full-model probe clones differ")
        rows.append(
            {
                "seed": seed,
                "source_state_sha256": source_sha256,
                "paths": paths,
            }
        )
    implementation_paths = (
        Path(__file__),
        ROOT / "spin_delta_router.py",
        ROOT / "spin_delta_label_free_curriculum.py",
        ROOT / "model.py",
        ROOT / "spin_delta_scan.py",
        ROOT / "raw_cuda.py",
        ROOT / "csrc" / "spin_scan.cpp",
        ROOT / "csrc" / "spin_scan_cuda.cu",
    )
    report = {
        "schema_version": 1,
        "stage": "spin_delta_query_gradient_topology",
        "protocol": PROTOCOL,
        "seeds": list(SEEDS),
        "modes": list(MODES),
        "local_float64": local,
        "full_model": rows,
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
        math.isfinite(value)
        for row in rows
        for path in row["paths"]
        for value in (
            path["loss"],
            path["query_event_gradient_norm"],
            path["query_slot_gradient_norm"],
            path["maximum_absolute_logit_change_from_hard"],
        )
    ):
        raise FloatingPointError("nonfinite audit metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
