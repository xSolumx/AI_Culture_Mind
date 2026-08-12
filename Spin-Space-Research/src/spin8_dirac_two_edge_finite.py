"""Exact radical elimination for the complete finite second Dirac edge.

After the common Hadamard reduction, each paired orientation margin has the
form

    A(x) + sqrt(1-x) B(x)
      +/- sqrt(x) (C(x) + sqrt(1-x) D(x)),

where ``x=i**2``.  Setting ``y=sqrt(1-x)`` shows that positivity of both signs
is exactly equivalent to two ordinary polynomial inequalities on ``y in
[0,1]``.  No relaxation or repeated squaring is used.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Iterable
from pathlib import Path

import psutil
import sympy as sp
import torch

from spin8_dirac_two_edge_kernel import (
    EVEN_MASKS,
    HADAMARD,
    ODD_MASKS,
    SectorPolynomial,
    TorchChannels,
    _exact_forced,
    load_sectors,
)
from spin8_resource_limits import constrain_current_process


def _core_polynomial(
    sector: SectorPolynomial,
    point: tuple[sp.Expr, ...],
    x: sp.Expr,
) -> sp.Expr:
    six_point = point[:4] + (x, point[4])
    return sp.factor(
        sum(
            coefficient
            * sp.prod(
                six_point[axis] ** powers[axis] for axis in range(6)
            )
            for powers, coefficient in sector.terms
        )
    )


def exact_finite_components(
    sectors: dict[tuple[int, ...], SectorPolynomial],
    point: tuple[sp.Expr, ...],
    channel: int,
    y: sp.Expr,
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Return ``A,B,C,D,L,S`` for one channel at a fixed base point."""

    if channel not in range(4):
        raise ValueError("channel must be between zero and three")
    x = 1 - y**2
    even = [
        _exact_forced(sectors[mask], point)
        * _core_polynomial(sectors[mask], point, x)
        for mask in EVEN_MASKS
    ]
    odd = [
        _exact_forced(sectors[mask], point)
        * _core_polynomial(sectors[mask], point, x)
        for mask in ODD_MASKS
    ]
    row = HADAMARD[channel]
    target = (9 - point[-1]) ** 2
    a_part = target - row[0] * even[0] - row[3] * even[3]
    b_part = -row[1] * even[1] - row[2] * even[2]
    c_part = -row[0] * odd[0] - row[3] * odd[3]
    d_part = -row[1] * odd[1] - row[2] * odd[2]
    center = sp.factor(a_part + y * b_part)
    odd_drive = sp.factor(c_part + y * d_part)
    square_margin = sp.factor(center**2 - (1 - y**2) * odd_drive**2)
    return tuple(
        map(sp.factor, (a_part, b_part, c_part, d_part, center, square_margin))
    )


def exact_radical_elimination_certificate(
    sectors: dict[tuple[int, ...], SectorPolynomial],
) -> dict[str, object]:
    """Mechanically establish the degree-six/degree-twelve reduction."""

    y = sp.symbols("y", nonnegative=True)
    point = (
        sp.Rational(25, 169),
        sp.Rational(9, 25),
        sp.Rational(49, 289),
        sp.Rational(64, 361),
        sp.Rational(81, 441),
    )
    structural_i_degrees = {
        "even": [max(powers[4] for powers, _ in sectors[mask].terms) for mask in EVEN_MASKS],
        "odd": [max(powers[4] for powers, _ in sectors[mask].terms) for mask in ODD_MASKS],
    }
    structural_degrees_match = structural_i_degrees == {
        "even": [3, 2, 1, 2],
        "odd": [2, 1, 1, 2],
    }
    rows = []
    passed = True
    for channel in range(4):
        a_part, b_part, c_part, d_part, center, square_margin = (
            exact_finite_components(sectors, point, channel, y)
        )
        degrees = {
            "A_in_y": int(sp.Poly(a_part, y).degree()),
            "B_in_y": int(sp.Poly(b_part, y).degree()),
            "C_in_y": int(sp.Poly(c_part, y).degree()),
            "D_in_y": int(sp.Poly(d_part, y).degree()),
            "center_L": int(sp.Poly(center, y).degree()),
            "square_margin_S": int(sp.Poly(square_margin, y).degree()),
        }
        # Check both signs at a disjoint rational y without introducing an
        # algebraic extension: choose y=3/5, hence sqrt(1-y^2)=4/5.
        y0 = sp.Rational(3, 5)
        x_root = sp.Rational(4, 5)
        evaluated = center.subs(y, y0)
        odd_drive = c_part + y * d_part
        predicted_pair = (
            sp.factor(evaluated + x_root * odd_drive.subs(y, y0)),
            sp.factor(evaluated - x_root * odd_drive.subs(y, y0)),
        )
        x0 = 1 - y0**2
        amplitudes = []
        for mask in EVEN_MASKS + ODD_MASKS:
            sector = sectors[mask]
            amplitude = _exact_forced(sector, point) * _core_polynomial(
                sector, point, x0
            )
            amplitude *= x_root ** sector.mask[4]
            amplitude *= y0 ** sector.complement[4]
            amplitudes.append(amplitude)
        even_amplitudes = amplitudes[:4]
        odd_amplitudes = amplitudes[4:]
        target = (9 - point[-1]) ** 2
        direct_pair = tuple(
            sp.factor(
                target
                - sum(
                    HADAMARD[channel][index]
                    * (even_amplitudes[index] + sign * odd_amplitudes[index])
                    for index in range(4)
                )
            )
            for sign in (1, -1)
        )
        row_passed = bool(
            predicted_pair == direct_pair
            and degrees["center_L"] <= 6
            and degrees["square_margin_S"] <= 12
        )
        passed &= row_passed
        rows.append(
            {
                "channel": channel,
                "degrees": degrees,
                "exact_pair_check_at_y_3_over_5": list(map(str, predicted_pair)),
                "direct_pair_matches": predicted_pair == direct_pair,
                "passed": row_passed,
            }
        )
    return {
        "experiment": "exact finite-edge one-squaring radical elimination",
        "base_squared_point": list(map(str, point)),
        "paired_margin_form": (
            "L(y) +/- sqrt(1-y^2) R(y), with L=A+yB and R=C+yD"
        ),
        "equivalent_polynomial_conditions": [
            "L(y) >= 0",
            "S(y)=L(y)^2-(1-y^2)R(y)^2 >= 0",
        ],
        "maximum_center_degree": 6,
        "maximum_square_margin_degree": 12,
        "structural_i2_degrees": structural_i_degrees,
        "structural_degrees_match": structural_degrees_match,
        "one_squaring_is_equivalent_when_center_is_nonnegative": True,
        "channel_rows": rows,
        "scope_boundary": (
            "This proves the algebraic reduction and degree bounds. It does not "
            "prove L or S nonnegative over the base five-cube."
        ),
        "passed": bool(passed and structural_degrees_match),
    }


def torch_finite_certificates(
    evaluator: TorchChannels,
    points: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the exact-equivalent center and squared-margin numerically."""

    points = points.to(device=evaluator.device, dtype=evaluator.dtype)
    y = y.to(device=evaluator.device, dtype=evaluator.dtype).reshape(-1, 1)
    x = 1 - y.square()

    def stripped(mask: tuple[int, ...]) -> torch.Tensor:
        _powers, _coefficients, lower, complement, _i_comp = evaluator.rows[mask]
        core = sum(
            evaluator._core(mask, points, degree) * x[:, 0] ** degree
            for degree in range(4)
        )
        forced = torch.prod(
            points.clamp_min(0).pow(lower / 2)
            * (1 - points).clamp_min(0).pow(complement / 2),
            dim=-1,
        )
        return forced * core

    even = torch.stack([stripped(mask) for mask in EVEN_MASKS], dim=-1)
    odd = torch.stack([stripped(mask) for mask in ODD_MASKS], dim=-1)
    hadamard = evaluator.hadamard
    target = (9 - points[:, -1]).square().unsqueeze(-1)
    a_part = target - (
        even[:, [0, 3]] @ hadamard[:, [0, 3]].T
    )
    b_part = -(even[:, [1, 2]] @ hadamard[:, [1, 2]].T)
    c_part = -(odd[:, [0, 3]] @ hadamard[:, [0, 3]].T)
    d_part = -(odd[:, [1, 2]] @ hadamard[:, [1, 2]].T)
    center = a_part + y * b_part
    odd_drive = c_part + y * d_part
    square_margin = center.square() - (1 - y.square()) * odd_drive.square()
    return center, square_margin


def _sample_six_cube(
    count: int,
    random: torch.Generator,
    device: torch.device,
    *,
    face_axis: int | None = None,
    face_value: int | None = None,
) -> torch.Tensor:
    uniform_count = count // 2
    uniform = torch.rand(uniform_count, 6, generator=random, dtype=torch.float64)
    concentration = torch.full((count - uniform_count, 6), 0.25, dtype=torch.float64)
    left = torch._standard_gamma(concentration, generator=random)
    right = torch._standard_gamma(concentration, generator=random)
    boundary = left / (left + right)
    values = torch.cat((uniform, boundary), dim=0).to(device)
    if face_axis is not None:
        values[:, face_axis] = float(face_value)
    return values


def numerical_finite_falsifier(
    evaluator: TorchChannels,
    *,
    seed: int,
    samples_per_region: int,
    batch_size: int,
) -> dict[str, object]:
    """Falsify the two exact-equivalent polynomial conditions on all faces."""

    random = torch.Generator(device="cpu")
    random.manual_seed(seed)
    names = ("a2", "d2", "e2", "g2", "c2", "y")
    regions: list[tuple[str, int | None, int | None]] = [("full", None, None)]
    regions.extend(
        (f"{names[axis]}={value}", axis, value)
        for axis in range(6)
        for value in (0, 1)
    )
    rows = []
    total_negative_center = 0
    total_negative_square = 0
    for name, axis, value in regions:
        center_minimum = torch.full(
            (4,), torch.inf, dtype=evaluator.dtype, device=evaluator.device
        )
        square_minimum = torch.full(
            (4,), torch.inf, dtype=evaluator.dtype, device=evaluator.device
        )
        center_point = torch.zeros(
            4, 6, dtype=evaluator.dtype, device=evaluator.device
        )
        square_point = torch.zeros_like(center_point)
        region_negative_center = 0
        region_negative_square = 0
        remaining = samples_per_region
        while remaining:
            count = min(batch_size, remaining)
            values = _sample_six_cube(
                count,
                random,
                evaluator.device,
                face_axis=axis,
                face_value=value,
            )
            center, square = torch_finite_certificates(
                evaluator, values[:, :5], values[:, 5]
            )
            center_values, center_indices = center.min(dim=0)
            square_values, square_indices = square.min(dim=0)
            for channel in range(4):
                if center_values[channel] < center_minimum[channel]:
                    center_minimum[channel] = center_values[channel]
                    center_point[channel] = values[center_indices[channel]]
                if square_values[channel] < square_minimum[channel]:
                    square_minimum[channel] = square_values[channel]
                    square_point[channel] = values[square_indices[channel]]
            region_negative_center += int((center < -1e-9).sum())
            region_negative_square += int((square < -1e-8).sum())
            remaining -= count
        total_negative_center += region_negative_center
        total_negative_square += region_negative_square
        rows.append(
            {
                "region": name,
                "samples": samples_per_region,
                "minimum_center_by_channel": center_minimum.tolist(),
                "center_minimizer_points": center_point.tolist(),
                "minimum_square_margin_by_channel": square_minimum.tolist(),
                "square_margin_minimizer_points": square_point.tolist(),
                "negative_center_count_below_minus_1e_minus_9": (
                    region_negative_center
                ),
                "negative_square_count_below_minus_1e_minus_8": (
                    region_negative_square
                ),
            }
        )
    return {
        "seed": seed,
        "samples_per_region": samples_per_region,
        "region_count": len(regions),
        "total_samples": samples_per_region * len(regions),
        "batch_size": batch_size,
        "negative_center_count": total_negative_center,
        "negative_square_margin_count": total_negative_square,
        "regions": rows,
        "status": "falsification evidence only; no theorem promotion",
    }


def run(
    coefficients: Path,
    output: Path,
    *,
    seed: int,
    samples_per_region: int,
    batch_size: int,
    workers: int,
) -> dict[str, object]:
    resource = constrain_current_process(workers=workers)
    sectors = load_sectors(coefficients)
    exact = exact_radical_elimination_certificate(sectors)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    evaluator = TorchChannels(sectors, device)
    numerical = numerical_finite_falsifier(
        evaluator,
        seed=seed,
        samples_per_region=samples_per_region,
        batch_size=batch_size,
    )
    report = {
        "experiment": "finite two-edge exact reduction and CUDA falsifier",
        "coefficient_artifact": str(coefficients),
        "exact_reduction": exact,
        "numerical_falsifier": numerical,
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "resource_contract": resource,
        },
        "theorem_promoted": False,
        "passed_exact_reduction": exact["passed"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(arguments: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--samples-per-region", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(arguments)
    report = run(
        args.coefficients,
        args.output,
        seed=args.seed,
        samples_per_region=args.samples_per_region,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed_exact_reduction": report["passed_exact_reduction"],
                "negative_center_count": report["numerical_falsifier"][
                    "negative_center_count"
                ],
                "negative_square_margin_count": report["numerical_falsifier"][
                    "negative_square_margin_count"
                ],
            }
        )
    )
    if not report["passed_exact_reduction"]:
        raise SystemExit("finite-edge radical elimination failed")


if __name__ == "__main__":
    main()
