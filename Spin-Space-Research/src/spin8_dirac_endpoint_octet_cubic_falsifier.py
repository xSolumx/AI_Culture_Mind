"""Bounded GPU falsifier for the adjacent-endpoint Schur cubic.

This evaluates the reconstructed Walsh amplitudes in float64 and forms the
Klein-four convolution directly.  A negative candidate has no theorem status
until it is reconstructed and checked in exact arithmetic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spin8_dirac_endpoint_octet import H0, H1, SURVIVING
from spin8_dirac_endpoint_octet_falsifier import (
    EndpointEvaluator,
    _atomic_json,
    _sha256,
)


def _xor(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a ^ b for a, b in zip(left, right, strict=True))


class CubicEvaluator(EndpointEvaluator):
    def __init__(self, coefficient_dir: Path, device: torch.device):
        super().__init__(coefficient_dir, device)
        self.amplitude_index = {mask: index for index, mask in enumerate(SURVIVING)}

    def cubic(self, point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _margins, amplitudes = self.margins(point)
        coefficients = []
        for target in H0:
            subgroup = torch.zeros(
                point.shape[0], dtype=point.dtype, device=point.device
            )
            coset = torch.zeros_like(subgroup)
            for left in H0:
                subgroup += (
                    amplitudes[:, self.amplitude_index[left]]
                    * amplitudes[:, self.amplitude_index[_xor(left, target)]]
                )
            for left in H1:
                coset += (
                    amplitudes[:, self.amplitude_index[left]]
                    * amplitudes[:, self.amplitude_index[_xor(left, target)]]
                )
            coefficients.append(subgroup - coset)
        z0, z1, z2, z3 = coefficients
        cubic = z0**3 - z0 * (z1**2 + z2**2 + z3**2) + 2 * z1 * z2 * z3
        return cubic, z0


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    seed: int,
    random_points: int,
    starts: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    evaluator = CubicEvaluator(coefficient_dir, device)
    generator = torch.Generator(device=device).manual_seed(seed)

    random_minimum = float("inf")
    random_raw = None
    random_point = None
    batch_size = 64 if device.type == "cuda" else 8
    with torch.no_grad():
        remaining = random_points
        while remaining:
            count = min(batch_size, remaining)
            points = torch.rand(
                count, 5, generator=generator, dtype=torch.float64, device=device
            )
            cubic, z0 = evaluator.cubic(points)
            ratio = cubic / z0.abs().clamp_min(1e-30) ** 3
            value, row = ratio.min(dim=0)
            if float(value) < random_minimum:
                random_minimum = float(value)
                random_raw = float(cubic[row])
                random_point = points[row].detach().cpu().tolist()
            remaining -= count

    logits = torch.empty(starts, 5, dtype=torch.float64, device=device).uniform_(
        -4.0, 4.0, generator=generator
    )
    logits.requires_grad_(True)
    optimizer = torch.optim.Adam((logits,), lr=learning_rate)
    best_ratio = float("inf")
    best_raw = None
    best_point = None
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        points = logits.sigmoid()
        cubic, z0 = evaluator.cubic(points)
        ratio = cubic / z0.abs().clamp_min(1e-30) ** 3
        ratio.mean().backward()
        optimizer.step()
        with torch.no_grad():
            value, row = ratio.min(dim=0)
            if float(value) < best_ratio:
                best_ratio = float(value)
                best_raw = float(cubic[row])
                best_point = points[row].detach().cpu().tolist()
            if step % 25 == 0 or step == steps - 1:
                history.append({"step": step, "minimum_ratio": float(value)})

    report = {
        "experiment": "adjacent endpoint octet cubic GPU falsifier",
        "evidence_class": "floating-point counterexample search; not a proof",
        "domain": "(ud,ue,ug,ui,y) in [0,1]^5",
        "seed": seed,
        "random_screen": {
            "point_count": random_points,
            "minimum_normalized_cubic": random_minimum,
            "raw_cubic": random_raw,
            "point": random_point,
        },
        "gradient_search": {
            "starts": starts,
            "steps": steps,
            "learning_rate": learning_rate,
            "minimum_normalized_cubic": best_ratio,
            "raw_cubic": best_raw,
            "point": best_point,
            "history": history,
        },
        "candidate_counterexample_found": bool(
            min(random_minimum, best_ratio) < -1e-10
        ),
        "exact_followup_required_for_any_negative": True,
        "input_sha256": evaluator.input_hashes,
        "runtime": {
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
    }
    _atomic_json(output, report)
    report["artifact_sha256"] = _sha256(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficient-dir",
        type=Path,
        default=Path("artifacts/spin8_dirac_unrestricted_coefficients_20260807"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--random-points", type=int, default=4096)
    parser.add_argument("--starts", type=int, default=32)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    if arguments.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    report = run(
        arguments.coefficient_dir,
        output=arguments.output,
        seed=arguments.seed,
        random_points=arguments.random_points,
        starts=arguments.starts,
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        device=device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
