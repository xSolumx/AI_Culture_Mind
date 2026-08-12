"""GPU falsifier for the eight adjacent-endpoint Dirac--Gram margins.

This program evaluates the exact reconstructed Walsh polynomials in float64.
It can find a candidate counterexample, but a negative candidate must still be
rationalized and checked in exact arithmetic before theorem status changes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from fractions import Fraction
from pathlib import Path

import torch

from spin8_dirac_endpoint_octet import SURVIVING
from spin8_dirac_final_residual import exact_full_chart_sign_certificate
from spin8_dirac_unrestricted_grid import _sector_metadata


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_polynomials(coefficient_dir: Path, device: torch.device):
    rows = []
    hashes = {}
    for mask in SURVIVING:
        path = coefficient_dir / f"alpha_sector_{''.join(map(str, mask))}.json.gz"
        hashes[path.name] = _sha256(path)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            coefficient_rows = json.load(handle)["coefficient_rows"]
        powers = torch.tensor(
            [row["powers"] for row in coefficient_rows],
            dtype=torch.int64,
            device=device,
        )
        coefficients = torch.tensor(
            [float(4 * Fraction(row["coefficient"])) for row in coefficient_rows],
            dtype=torch.float64,
            device=device,
        )
        rows.append((powers, coefficients))
    return tuple(rows), hashes


def _sign_matrix(device: torch.device) -> torch.Tensor:
    masks, _complements, _representatives, hadamard = _sector_metadata()
    index = {mask: row for row, mask in enumerate(masks)}
    patterns = sorted(
        {
            tuple(hadamard[index[mask]][column] for mask in SURVIVING)
            for column in range(16)
        }
    )
    if len(patterns) != 8:
        raise AssertionError("endpoint sign quotient did not yield eight characters")
    return torch.tensor(patterns, dtype=torch.float64, device=device)


class EndpointEvaluator:
    def __init__(self, coefficient_dir: Path, device: torch.device):
        chart = exact_full_chart_sign_certificate()
        if not chart["passed"]:
            raise AssertionError("full-chart sign certificate failed")
        self.masks = tuple(tuple(row["lower_mask"]) for row in chart["chart_characters"])
        self.complements = {
            tuple(row["lower_mask"]): tuple(row["complement_mask"])
            for row in chart["chart_characters"]
        }
        self.polynomials, self.input_hashes = _load_polynomials(
            coefficient_dir, device
        )
        self.signs = _sign_matrix(device)
        self.device = device

    @staticmethod
    def _residual(values: torch.Tensor, powers: torch.Tensor, coefficients: torch.Tensor):
        monomials = torch.ones(
            values.shape[0], powers.shape[0], dtype=values.dtype, device=values.device
        )
        for axis in range(7):
            monomials = monomials * values[:, axis : axis + 1].pow(
                powers[None, :, axis]
            )
        return monomials @ coefficients

    def margins(self, point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ud, ue, ug, ui, y = point.unbind(dim=1)
        zeros = torch.zeros_like(ud)
        ones = torch.ones_like(ud)
        uh = 1 - y.square()
        values = torch.stack((zeros, ud, ue, ug, uh, ui, ones), dim=1)
        amplitudes = []
        for mask, (powers, coefficients) in zip(
            SURVIVING, self.polynomials, strict=True
        ):
            residual = self._residual(values, powers, coefficients)
            forced_square = torch.ones_like(residual)
            complement = self.complements[mask]
            for axis in range(7):
                if mask[axis]:
                    forced_square = forced_square * values[:, axis]
                if complement[axis]:
                    forced_square = forced_square * (1 - values[:, axis])
            amplitudes.append(forced_square.clamp_min(0).sqrt() * residual)
        amplitude_matrix = torch.stack(amplitudes, dim=1)
        return amplitude_matrix @ self.signs.T, amplitude_matrix


def run(
    coefficient_dir: Path,
    *,
    output: Path,
    seed: int,
    random_points: int,
    starts_per_orientation: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    evaluator = EndpointEvaluator(coefficient_dir, device)

    generator = torch.Generator(device=device).manual_seed(seed)
    random_minimum = float("inf")
    random_point = None
    random_orientation = None
    batch_size = 64 if device.type == "cuda" else 8
    with torch.no_grad():
        remaining = random_points
        while remaining:
            count = min(batch_size, remaining)
            points = torch.rand(
                count, 5, generator=generator, dtype=torch.float64, device=device
            )
            margins, amplitudes = evaluator.margins(points)
            normalized = margins / amplitudes[:, :1].abs().clamp_min(1e-30)
            value, flat = normalized.reshape(-1).min(dim=0)
            if float(value) < random_minimum:
                row = int(flat) // 8
                orientation = int(flat) % 8
                random_minimum = float(value)
                random_point = points[row].detach().cpu().tolist()
                random_orientation = orientation
            remaining -= count

    orientation_count = evaluator.signs.shape[0]
    assigned = torch.arange(orientation_count, device=device).repeat_interleave(
        starts_per_orientation
    )
    logits = torch.empty(
        assigned.shape[0], 5, dtype=torch.float64, device=device
    ).uniform_(-4.0, 4.0, generator=generator)
    logits.requires_grad_(True)
    optimizer = torch.optim.Adam((logits,), lr=learning_rate)
    best_ratio = float("inf")
    best_margin = None
    best_mean = None
    best_point = None
    best_orientation = None
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        points = logits.sigmoid()
        margins, amplitudes = evaluator.margins(points)
        selected = margins[torch.arange(assigned.shape[0], device=device), assigned]
        means = amplitudes[:, 0].abs().clamp_min(1e-30)
        ratios = selected / means
        ratios.mean().backward()
        optimizer.step()
        with torch.no_grad():
            value, row = ratios.min(dim=0)
            if float(value) < best_ratio:
                best_ratio = float(value)
                best_margin = float(selected[row])
                best_mean = float(means[row])
                best_point = points[row].detach().cpu().tolist()
                best_orientation = int(assigned[row])
            if step % 25 == 0 or step == steps - 1:
                history.append({"step": step, "minimum_ratio": float(value)})

    signs = evaluator.signs.detach().cpu().to(torch.int64).tolist()
    report = {
        "experiment": "adjacent endpoint octet GPU falsifier",
        "evidence_class": "floating-point counterexample search; not a proof",
        "domain": "(ud,ue,ug,ui,y) in [0,1]^5",
        "seed": seed,
        "random_screen": {
            "point_count": random_points,
            "minimum_normalized_margin": random_minimum,
            "point": random_point,
            "orientation_index": random_orientation,
            "orientation_signs": signs[random_orientation],
        },
        "gradient_search": {
            "starts_per_orientation": starts_per_orientation,
            "steps": steps,
            "learning_rate": learning_rate,
            "minimum_normalized_margin": best_ratio,
            "raw_margin": best_margin,
            "mean_amplitude": best_mean,
            "point": best_point,
            "orientation_index": best_orientation,
            "orientation_signs": signs[best_orientation],
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
    parser.add_argument("--starts-per-orientation", type=int, default=8)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    if arguments.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    report = run(
        arguments.coefficient_dir,
        output=arguments.output,
        seed=arguments.seed,
        random_points=arguments.random_points,
        starts_per_orientation=arguments.starts_per_orientation,
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        device=device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
