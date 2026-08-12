"""Boundary-kernel falsifier for the second Dirac--Gram residual edge.

The exact all-sector reconstruction reduces the first-order two-edge question
to four scalar implications

    lambda_r(u, v, r, w, z) = 0  =>  mu_r(u, v, r, w, z) = 0.

Here ``lambda`` is an eigenvalue of the proved one-edge orientation matrix and
``mu`` is the matching Hadamard eigenvalue of the odd first derivative in the
new Cholesky coordinate.  A single point with zero ``lambda`` and nonzero
``mu`` is an exact local counterexample to two-edge positivity.  Numerical
search in this module is only a falsifier; candidates must be rationalized and
replayed by :func:`exact_channels` before they count.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import psutil
import sympy as sp
import torch

from spin8_dirac_two_edge_shared_reconstruct import (
    VARIABLE_ORDER,
    _shared_setup,
    verify_coefficient_report,
)
from spin8_resource_limits import constrain_current_process

BASE_VARIABLE_ORDER = ("a2", "d2", "e2", "g2", "c2")
EVEN_MASKS = (
    (0, 0, 0, 0, 0, 0),
    (0, 0, 1, 1, 0, 1),
    (1, 1, 0, 1, 0, 1),
    (1, 1, 1, 0, 0, 0),
)
ODD_MASKS = (
    (0, 1, 0, 1, 1, 0),
    (0, 1, 1, 0, 1, 1),
    (1, 0, 0, 0, 1, 1),
    (1, 0, 1, 1, 1, 0),
)
HADAMARD = (
    (1, 1, -1, -1),
    (1, -1, 1, -1),
    (1, -1, -1, 1),
    (1, 1, 1, 1),
)


@dataclass(frozen=True)
class SectorPolynomial:
    mask: tuple[int, ...]
    complement: tuple[int, ...]
    terms: tuple[tuple[tuple[int, ...], sp.Rational], ...]


def load_sectors(path: Path) -> dict[tuple[int, ...], SectorPolynomial]:
    """Load and integrity-check the exact all-sector coefficient artifact."""

    report = json.loads(path.read_text(encoding="utf-8"))
    if not verify_coefficient_report(report):
        raise ValueError("two-edge coefficient report failed exact replay")
    _masks, _signs, _inverse, complements = _shared_setup()
    sectors = {}
    for row in report["sector_rows"]:
        mask = tuple(row["mask"])
        terms = tuple(
            (
                tuple(int(power) for power in item["powers"]),
                sp.Rational(item["coefficient"]),
            )
            for item in row["coefficient_rows"]
        )
        sectors[mask] = SectorPolynomial(mask, complements[mask], terms)
    expected = set(EVEN_MASKS) | set(ODD_MASKS)
    if set(sectors) != expected:
        raise ValueError("coefficient report has unexpected Walsh support")
    return sectors


def _exact_core(sector: SectorPolynomial, point: tuple[sp.Expr, ...]) -> sp.Expr:
    six_point = point[:4] + (sp.Integer(0), point[4])
    return sp.factor(
        sum(
            coefficient
            * sp.prod(
                six_point[axis] ** powers[axis] for axis in range(len(VARIABLE_ORDER))
            )
            for powers, coefficient in sector.terms
            if powers[4] == 0
        )
    )


def _exact_core_i2_derivative(
    sector: SectorPolynomial, point: tuple[sp.Expr, ...]
) -> sp.Expr:
    six_point = point[:4] + (sp.Integer(0), point[4])
    return sp.factor(
        sum(
            coefficient
            * sp.prod(
                six_point[axis] ** powers[axis]
                for axis in range(len(VARIABLE_ORDER))
                if axis != 4
            )
            for powers, coefficient in sector.terms
            if powers[4] == 1
        )
    )


def _exact_forced(
    sector: SectorPolynomial, point: tuple[sp.Expr, ...]
) -> sp.Expr:
    """Return the i-independent forced monomial at i=0.

    For an odd sector this is the derivative of its forced monomial with
    respect to the signed lower coordinate ``i`` at zero.
    """

    six_point = point[:4] + (sp.Integer(0), point[4])
    result = sp.Integer(1)
    for axis, (lower_bit, complement_bit) in enumerate(
        zip(sector.mask, sector.complement, strict=True)
    ):
        if axis == 4:
            continue
        squared = six_point[axis]
        result *= squared ** sp.Rational(lower_bit, 2)
        result *= (1 - squared) ** sp.Rational(complement_bit, 2)
    return result


def exact_channels(
    sectors: dict[tuple[int, ...], SectorPolynomial],
    point: tuple[sp.Expr, ...],
) -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    """Evaluate the four exact one-edge margins and odd derivatives."""

    if len(point) != 5:
        raise ValueError("point must follow (a2,d2,e2,g2,c2)")
    even = sp.Matrix(
        [
            _exact_forced(sectors[mask], point)
            * _exact_core(sectors[mask], point)
            for mask in EVEN_MASKS
        ]
    )
    odd = sp.Matrix(
        [
            _exact_forced(sectors[mask], point)
            * _exact_core(sectors[mask], point)
            for mask in ODD_MASKS
        ]
    )
    hadamard = sp.Matrix(HADAMARD)
    target = (9 - point[-1]) ** 2
    lambdas = sp.ones(4, 1) * target - hadamard * even
    mus = -(hadamard * odd)
    return tuple(map(sp.factor, lambdas)), tuple(map(sp.factor, mus))


def exact_even_i2_curvatures(
    sectors: dict[tuple[int, ...], SectorPolynomial],
    point: tuple[sp.Expr, ...],
) -> tuple[sp.Expr, ...]:
    """Return coefficients of physical ``i**2`` in the four margins."""

    derivatives = []
    for mask in EVEN_MASKS:
        sector = sectors[mask]
        core = _exact_core(sector, point)
        core_derivative = _exact_core_i2_derivative(sector, point)
        forced = _exact_forced(sector, point)
        derivatives.append(
            forced * (core_derivative - sp.Rational(sector.complement[4], 2) * core)
        )
    curvatures = -(sp.Matrix(HADAMARD) * sp.Matrix(derivatives))
    return tuple(map(sp.factor, curvatures))


def exact_quadratic_schur_counterexample(
    sectors: dict[tuple[int, ...], SectorPolynomial],
) -> dict[str, object]:
    """Reject a global quadratic-truncation certificate with exact arithmetic.

    Every squared coordinate below comes from a rational point on the unit
    circle, so all forced square roots are rational and the witness contains
    no algebraic-number sign ambiguity.
    """

    point = (
        sp.Rational(25, 169),
        sp.Rational(16, 25),
        sp.Rational(25, 169),
        sp.Rational(16, 25),
        sp.Rational(1600, 1681),
    )
    lambdas, mus = exact_channels(sectors, point)
    curvatures = exact_even_i2_curvatures(sectors, point)
    residuals = tuple(
        sp.factor(4 * lam * curvature - mu**2)
        for lam, mu, curvature in zip(lambdas, mus, curvatures, strict=True)
    )
    passed = bool(
        all(value > 0 for value in lambdas)
        and all(value < 0 for value in curvatures)
        and all(value < 0 for value in residuals)
    )
    return {
        "experiment": "exact rejection of global quadratic Schur certificate",
        "squared_coordinate_order": list(BASE_VARIABLE_ORDER),
        "squared_point": list(map(str, point)),
        "rational_circle_parameters": ["1/5", "1/2", "1/5", "1/2", "4/5"],
        "lambdas": list(map(str, lambdas)),
        "odd_derivatives": list(map(str, mus)),
        "i2_curvatures": list(map(str, curvatures)),
        "four_lambda_nu_minus_mu_squared": list(map(str, residuals)),
        "decimal_residuals": [float(value) for value in residuals],
        "falsified_claim": (
            "the unmodified quadratic Schur residual 4*lambda*nu-mu^2 is "
            "globally nonnegative on the one-edge cube"
        ),
        "scope_boundary": (
            "This does not falsify the full two-edge determinant inequality. "
            "Higher powers of i2 remain present and can restore positivity."
        ),
        "passed": passed,
    }


def exact_local_kernel_certificate(
    sectors: dict[tuple[int, ...], SectorPolynomial],
) -> dict[str, object]:
    """Certify the complete quadratic jet at the one-edge equality line.

    The physical transverse coordinates are ``a,d,e,g`` and ``z=c^2``.  The
    certificate is extracted from the maintained exact coefficient maps, not
    inserted as an assumed formula.  It also closes the sole quadratic null
    direction at ``z=1`` by evaluating that path to all orders.
    """

    u, v, r, w, z, s = sp.symbols("u v r w z s", nonnegative=True)
    point = (u, v, r, w, z)
    origin = {u: 0, v: 0, r: 0, w: 0}
    trivial = _exact_core(sectors[EVEN_MASKS[0]], point)
    diagonal = tuple(
        sp.factor(-sp.diff(trivial, variable).subs(origin))
        for variable in (u, v, r, w)
    )
    even_eg = sp.factor(
        sp.sqrt(z) * _exact_core(sectors[EVEN_MASKS[1]], point).subs(origin)
    )
    odd_dg = sp.factor(
        _exact_core(sectors[ODD_MASKS[0]], point).subs(origin)
    )
    odd_de = sp.factor(
        sp.sqrt(z) * _exact_core(sectors[ODD_MASKS[1]], point).subs(origin)
    )
    odd_a = sp.factor(
        sp.sqrt(z) * _exact_core(sectors[ODD_MASKS[2]], point).subs(origin)
    )
    expected_diagonal = (
        5 * (z - 9) * (z - 5) / 2,
        2 * (z - 9) * (z - 3),
        2 * (z - 9) * (z - 3),
        2 * (z - 9) * (z - 3),
    )
    expected_even_eg = 8 * sp.sqrt(z) * (z - 9)
    expected_odd_dg = -(z - 23) * (z - 9)
    extracted_formulas_match = bool(
        all(sp.factor(left - right) == 0 for left, right in zip(diagonal, expected_diagonal, strict=True))
        and sp.factor(even_eg - expected_even_eg) == 0
        and sp.factor(odd_dg - expected_odd_dg) == 0
        and odd_de == 0
        and odd_a == 0
    )
    transverse_determinant = sp.factor(
        diagonal[2] * diagonal[3] - (even_eg / 2) ** 2
    )
    expected_determinant = 4 * (z - 9) ** 3 * (z - 1)
    endpoint_lambdas, endpoint_mus = exact_channels(sectors, (0, 0, s, s, 1))
    equality_curvatures = exact_even_i2_curvatures(sectors, (0, 0, 0, 0, z))
    expected_curvature = 5 * (z - 9) * (z - 5) / 2
    expected_endpoint = (
        -64 * s * (s - 2),
        -64 * s**2 * (s**2 - 2),
        -64 * s**2 * (s**2 - 2),
        -64 * s * (s - 2),
    )
    endpoint_path_matches = bool(
        all(
            sp.factor(left - right) == 0
            for left, right in zip(endpoint_lambdas, expected_endpoint, strict=True)
        )
        and all(value == 0 for value in endpoint_mus)
    )
    equality_curvatures_match = all(
        sp.factor(value - expected_curvature) == 0 for value in equality_curvatures
    )
    rows = []
    for channel, row in enumerate(HADAMARD):
        rows.append(
            {
                "channel": channel,
                "eg_character": row[1],
                "lambda_quadratic": str(
                    diagonal[0] * sp.Symbol("a") ** 2
                    + diagonal[1] * sp.Symbol("d") ** 2
                    + diagonal[2] * sp.Symbol("e") ** 2
                    + diagonal[3] * sp.Symbol("g") ** 2
                    - row[1]
                    * even_eg
                    * sp.Symbol("e")
                    * sp.Symbol("g")
                ),
                "mu_quadratic": str(
                    (z - 23)
                    * (z - 9)
                    * sp.Symbol("d")
                    * sp.Symbol("g")
                ),
            }
        )
    passed = bool(
        extracted_formulas_match
        and sp.factor(transverse_determinant - expected_determinant) == 0
        and endpoint_path_matches
        and equality_curvatures_match
    )
    return {
        "experiment": "exact quadratic boundary-kernel jet",
        "physical_coordinates": ["a", "d", "e", "g"],
        "cayley_squared_coordinate": "z",
        "lambda_diagonal_coefficients": list(map(str, diagonal)),
        "lambda_eg_amplitude": str(even_eg),
        "mu_dg_amplitude_before_margin_sign": str(odd_dg),
        "mu_de_amplitude": str(odd_de),
        "mu_a_amplitude": str(odd_a),
        "channel_rows": rows,
        "eg_block_determinant": str(transverse_determinant),
        "eg_block_determinant_nonnegative_on_unit_interval": True,
        "quadratic_form_positive_definite_for_z_less_than_1": True,
        "only_quadratic_degeneracy": (
            "z=1 in channels 1 and 2 along equal e,g magnitudes"
        ),
        "odd_quadratic_vanishes_on_degenerate_tangent": True,
        "new_edge_quadratic_curvatures_on_equality_line": list(
            map(str, equality_curvatures)
        ),
        "new_edge_matches_a_direction_stiffness": equality_curvatures_match,
        "endpoint_equal_magnitude_lambdas": list(map(str, endpoint_lambdas)),
        "endpoint_equal_magnitude_mus": list(map(str, endpoint_mus)),
        "endpoint_null_is_lifted_quartically": endpoint_path_matches,
        "scope_boundary": (
            "This is an exact local certificate at the orthonormal equality line. "
            "It does not classify every global equality point of the one-edge theorem."
        ),
        "passed": passed,
    }


class TorchChannels:
    """Sparse float64 CUDA evaluator of the exact scalar reduction."""

    def __init__(
        self,
        sectors: dict[tuple[int, ...], SectorPolynomial],
        device: torch.device,
    ) -> None:
        self.device = device
        self.dtype = torch.float64
        self.hadamard = torch.tensor(HADAMARD, dtype=self.dtype, device=device)
        self.rows = {}
        for mask, sector in sectors.items():
            powers_by_i_degree = {degree: [] for degree in range(4)}
            coefficients_by_i_degree = {degree: [] for degree in range(4)}
            for six_powers, coefficient in sector.terms:
                if six_powers[4] in powers_by_i_degree:
                    powers_by_i_degree[six_powers[4]].append(
                        six_powers[:4] + six_powers[5:]
                    )
                    coefficients_by_i_degree[six_powers[4]].append(float(coefficient))
            self.rows[mask] = (
                tuple(
                    torch.tensor(
                        powers_by_i_degree[degree], dtype=torch.long, device=device
                    )
                    for degree in range(4)
                ),
                tuple(
                    torch.tensor(
                        coefficients_by_i_degree[degree],
                        dtype=self.dtype,
                        device=device,
                    )
                    for degree in range(4)
                ),
                torch.tensor(
                    sector.mask[:4] + sector.mask[5:],
                    dtype=self.dtype,
                    device=device,
                ),
                torch.tensor(
                    sector.complement[:4] + sector.complement[5:],
                    dtype=self.dtype,
                    device=device,
                ),
                sector.complement[4],
            )

    def _core(
        self, mask: tuple[int, ...], points: torch.Tensor, i_degree: int
    ) -> torch.Tensor:
        powers_by_degree, coefficients_by_degree, _lower, _complement, _i_comp = (
            self.rows[mask]
        )
        powers = powers_by_degree[i_degree]
        coefficients = coefficients_by_degree[i_degree]
        if powers.shape[0] == 0:
            return torch.zeros(points.shape[0], dtype=self.dtype, device=self.device)
        maximum_degree = int(powers.max())
        power_table = torch.stack(
            [points**degree for degree in range(maximum_degree + 1)], dim=-1
        )
        values = torch.ones(
            points.shape[0], powers.shape[0], dtype=self.dtype, device=self.device
        )
        for axis in range(5):
            values *= power_table[:, axis, powers[:, axis]]
        return values @ coefficients

    def _amplitude(self, mask: tuple[int, ...], points: torch.Tensor) -> torch.Tensor:
        _powers, _coefficients, lower, complement, _i_comp = self.rows[mask]
        core = self._core(mask, points, 0)
        forced = torch.prod(
            points.clamp_min(0).pow(lower / 2)
            * (1 - points).clamp_min(0).pow(complement / 2),
            dim=-1,
        )
        return forced * core

    def _even_i2_derivative(
        self, mask: tuple[int, ...], points: torch.Tensor
    ) -> torch.Tensor:
        _powers, _coefficients, lower, complement, i_complement = self.rows[mask]
        core = self._core(mask, points, 0)
        core_derivative = self._core(mask, points, 1)
        forced = torch.prod(
            points.clamp_min(0).pow(lower / 2)
            * (1 - points).clamp_min(0).pow(complement / 2),
            dim=-1,
        )
        return forced * (core_derivative - (i_complement / 2) * core)

    def __call__(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points = points.to(device=self.device, dtype=self.dtype)
        even = torch.stack(
            [self._amplitude(mask, points) for mask in EVEN_MASKS], dim=-1
        )
        odd = torch.stack(
            [self._amplitude(mask, points) for mask in ODD_MASKS], dim=-1
        )
        target = (9 - points[:, -1]).square().unsqueeze(-1)
        lambdas = target - even @ self.hadamard.T
        mus = -(odd @ self.hadamard.T)
        return lambdas, mus

    def curvatures(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate coefficients of physical ``i**2`` in all four margins."""

        points = points.to(device=self.device, dtype=self.dtype)
        derivatives = torch.stack(
            [self._even_i2_derivative(mask, points) for mask in EVEN_MASKS], dim=-1
        )
        return -(derivatives @ self.hadamard.T)


def _sample_points(
    count: int,
    random: torch.Generator,
    device: torch.device,
    *,
    face_axis: int | None = None,
    face_value: int | None = None,
) -> torch.Tensor:
    # A 50/50 uniform/Beta(1/4,1/4) mixture sees both the interior and thin
    # boundary layers.  Beta sampling itself is performed on CPU because the
    # CUDA generator API does not expose a stable seeded beta sampler.
    uniform_count = count // 2
    uniform = torch.rand(uniform_count, 5, generator=random, dtype=torch.float64)
    concentration = torch.full((count - uniform_count, 5), 0.25, dtype=torch.float64)
    # Inverse-CDF is unavailable; Gamma ratios give exact Beta samples.
    left = torch._standard_gamma(concentration, generator=random)
    right = torch._standard_gamma(concentration, generator=random)
    boundary = left / (left + right)
    points = torch.cat((uniform, boundary), dim=0).to(device)
    if face_axis is not None:
        points[:, face_axis] = float(face_value)
    return points


def numerical_falsifier(
    evaluator: TorchChannels,
    *,
    seed: int,
    samples_per_region: int,
    batch_size: int,
) -> dict[str, object]:
    """Search the interior and every coordinate face without theorem promotion."""

    random = torch.Generator(device="cpu")
    random.manual_seed(seed)
    regions: list[tuple[str, int | None, int | None]] = [("full", None, None)]
    regions.extend(
        (f"{BASE_VARIABLE_ORDER[axis]}={value}", axis, value)
        for axis in range(5)
        for value in (0, 1)
    )
    rows = []
    global_candidates = []
    for region_index, (name, axis, value) in enumerate(regions):
        minima = torch.full((4,), math.inf, dtype=torch.float64, device=evaluator.device)
        curvature_minima = torch.full(
            (4,), math.inf, dtype=torch.float64, device=evaluator.device
        )
        schur_minima = torch.full(
            (4,), math.inf, dtype=torch.float64, device=evaluator.device
        )
        best_ratio = torch.zeros(4, dtype=torch.float64, device=evaluator.device)
        best_points = torch.zeros(4, 5, dtype=torch.float64, device=evaluator.device)
        candidate_count = 0
        remaining = samples_per_region
        while remaining:
            count = min(batch_size, remaining)
            points = _sample_points(
                count,
                random,
                evaluator.device,
                face_axis=axis,
                face_value=value,
            )
            lambdas, mus = evaluator(points)
            curvatures = evaluator.curvatures(points)
            schur = 4 * lambdas * curvatures - mus.square()
            minima = torch.minimum(minima, lambdas.min(dim=0).values)
            curvature_minima = torch.minimum(
                curvature_minima, curvatures.min(dim=0).values
            )
            schur_minima = torch.minimum(schur_minima, schur.min(dim=0).values)
            ratio = mus.abs() / torch.sqrt(lambdas.clamp_min(0) + 1e-24)
            values, indices = ratio.max(dim=0)
            improved = values > best_ratio
            best_ratio[improved] = values[improved]
            for channel in torch.where(improved)[0].tolist():
                best_points[channel] = points[indices[channel]]
            suspicious = (lambdas.abs() < 1e-10) & (mus.abs() > 1e-7)
            candidate_count += int(suspicious.sum())
            if suspicious.any():
                hit_rows, hit_channels = torch.where(suspicious)
                for hit, channel in zip(hit_rows[:16].tolist(), hit_channels[:16].tolist(), strict=True):
                    global_candidates.append(
                        {
                            "region": name,
                            "channel": channel,
                            "point": points[hit].tolist(),
                            "lambda": float(lambdas[hit, channel]),
                            "mu": float(mus[hit, channel]),
                        }
                    )
            remaining -= count
        rows.append(
            {
                "region": name,
                "samples": samples_per_region,
                "minimum_lambda_by_channel": minima.tolist(),
                "minimum_i2_curvature_by_channel": curvature_minima.tolist(),
                "minimum_quadratic_schur_residual_by_channel": schur_minima.tolist(),
                "maximum_abs_mu_over_sqrt_lambda_by_channel": best_ratio.tolist(),
                "ratio_maximizer_points": best_points.tolist(),
                "numerical_candidate_count": candidate_count,
            }
        )
    return {
        "seed": seed,
        "samples_per_region": samples_per_region,
        "total_samples": samples_per_region * len(regions),
        "batch_size": batch_size,
        "regions": rows,
        "numerical_candidates": global_candidates,
        "candidate_count": sum(row["numerical_candidate_count"] for row in rows),
        "status": "falsification evidence only; exact replay required",
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    evaluator = TorchChannels(sectors, device)
    local_certificate = exact_local_kernel_certificate(sectors)
    schur_counterexample = exact_quadratic_schur_counterexample(sectors)
    # Exact load-bearing anchors are replayed before the numerical campaign.
    anchor_rows = []
    for z in (sp.Rational(0), sp.Rational(1, 5), sp.Rational(4, 5), sp.Rational(1)):
        lambdas, mus = exact_channels(sectors, (0, 0, 0, 0, z))
        anchor_rows.append(
            {
                "point": ["0", "0", "0", "0", str(z)],
                "lambdas": list(map(str, lambdas)),
                "mus": list(map(str, mus)),
            }
        )
    numerical = numerical_falsifier(
        evaluator,
        seed=seed,
        samples_per_region=samples_per_region,
        batch_size=batch_size,
    )
    report = {
        "experiment": "two-edge boundary-kernel scalar falsifier",
        "coefficient_artifact": str(coefficients),
        "variable_order": list(BASE_VARIABLE_ORDER),
        "identity": "lambda_r=0 implies mu_r=0 is necessary for two-edge PSD",
        "exact_orthonormal_anchor_rows": anchor_rows,
        "exact_local_kernel_certificate": local_certificate,
        "exact_quadratic_schur_counterexample": schur_counterexample,
        "numerical": numerical,
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "resource_contract": resource,
        },
        "passed_exact_anchor": bool(
            local_certificate["passed"]
            and schur_counterexample["passed"]
            and all(
                all(value == "0" for value in row["lambdas"] + row["mus"])
                for row in anchor_rows
            )
        ),
        "theorem_promoted": False,
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
    print(json.dumps({
        "output": str(args.output),
        "candidate_count": report["numerical"]["candidate_count"],
        "total_samples": report["numerical"]["total_samples"],
        "passed_exact_anchor": report["passed_exact_anchor"],
    }, indent=2))


if __name__ == "__main__":
    main()
