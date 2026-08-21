"""Isotypic-to-silicon compiler IR, version 2.1.1.

The exact Programme 01 certificate describes mathematical blocks.  This module
turns only certified block signatures into a runtime plan and keeps three
facts separate:

* Schur type and multiplicity are algebraic facts;
* shared action across copies is a model/runtime fact;
* kernel choice is empirical hardware policy.

Conflating the last two is invalid: channels with independently selected group
elements are not a Tensor-Core multiplicity axis even when their representation
types happen to match.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from pure_spin8_ssm import __compiler_version__

COMPILER_VERSION = __compiler_version__
SchurType = Literal["real", "complex", "quaternion"]
KernelBackend = Literal["eager", "triton_scalar", "triton_tensor_core"]
DIVISION_DIMENSIONS: dict[SchurType, int] = {
    "real": 1,
    "complex": 2,
    "quaternion": 4,
}


@dataclass(frozen=True)
class IsotypicBlock:
    """One aligned ``V tensor_D D^m`` block."""

    name: str
    schur_type: SchurType
    irreducible_real_dimension: int
    multiplicity: int
    shared_action: bool
    scalar_field: str = "R"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("isotypic block name must be nonempty")
        if self.schur_type not in DIVISION_DIMENSIONS:
            raise ValueError("unknown Schur type")
        if self.irreducible_real_dimension < 1 or self.multiplicity < 1:
            raise ValueError("block dimensions must be positive")

    @property
    def real_dimension(self) -> int:
        return self.irreducible_real_dimension * self.multiplicity

    @property
    def expected_commutant_dimension(self) -> int:
        return self.multiplicity**2 * DIVISION_DIMENSIONS[self.schur_type]


@dataclass(frozen=True)
class RuntimeShape:
    batch_size: int
    sequence_length: int
    dtype: str
    device: str
    training: bool
    emit_every_prefix: bool = True

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.sequence_length < 1:
            raise ValueError("runtime dimensions must be positive")
        if self.dtype not in ("float16", "float32", "float64"):
            raise ValueError("unsupported runtime dtype")


@dataclass(frozen=True)
class HardwareTarget:
    name: str
    backend: str
    compute_capability: tuple[int, int] | None
    tensor_cores: bool
    tf32: bool

    @classmethod
    def current(cls, device: torch.device | str) -> HardwareTarget:
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return cls(
                name="cpu",
                backend=device.type,
                compute_capability=None,
                tensor_cores=False,
                tf32=False,
            )
        index = device.index if device.index is not None else torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(index)
        return cls(
            name=torch.cuda.get_device_name(index),
            backend="cuda",
            compute_capability=capability,
            tensor_cores=capability >= (7, 0),
            tf32=capability >= (8, 0),
        )


@dataclass(frozen=True)
class BlockSchedule:
    block: IsotypicBlock
    backend: KernelBackend
    accumulation_dtype: str
    tile_multiplicity: int | None
    profile_speedup: float | None
    reason: str


@dataclass(frozen=True)
class IsotypicExecutionPlan:
    compiler_version: str
    runtime: RuntimeShape
    hardware: HardwareTarget
    schedules: tuple[BlockSchedule, ...]
    certificate_sha256: str | None
    profile_sha256: str | None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint
        return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blocks_from_exact_certificate(
    path: str | Path,
    control: str,
    *,
    shared_action: bool,
) -> tuple[IsotypicBlock, ...]:
    """Load certified signatures without trusting stored descriptive prose."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise ValueError("the exact isotypic certificate is not passing")
    expected = payload.get("expected_block_signatures", {}).get(control)
    observed = payload.get("observed_block_signatures", {}).get(control)
    if expected is None or observed is None or expected != observed:
        raise ValueError("expected and observed exact block signatures disagree")
    blocks = []
    for index, row in enumerate(observed):
        schur_type, multiplicity, irreducible_dimension, commutant_dimension = row
        block = IsotypicBlock(
            name=f"{control}:{index}",
            schur_type=schur_type,
            irreducible_real_dimension=int(irreducible_dimension),
            multiplicity=int(multiplicity),
            shared_action=shared_action,
            scalar_field="certificate-declared",
        )
        if block.expected_commutant_dimension != int(commutant_dimension):
            raise ValueError("certificate row violates the double-centralizer law")
        blocks.append(block)
    return tuple(blocks)


def spin8_triality_blocks(
    multiplicity: int, *, shared_action: bool
) -> tuple[IsotypicBlock, ...]:
    """Return the three pairwise-inequivalent real Spin(8) blocks."""

    return tuple(
        IsotypicBlock(
            name=name,
            schur_type="real",
            irreducible_real_dimension=8,
            multiplicity=multiplicity,
            shared_action=shared_action,
            scalar_field="R",
        )
        for name in ("8v", "8s+", "8s-")
    )


def _profile_lookup(
    profile: dict[str, object] | None,
    runtime: RuntimeShape,
    block: IsotypicBlock,
) -> float | None:
    if profile is None:
        return None
    hardware = profile.get("hardware", {})
    if not isinstance(hardware, dict):
        return None
    rows = profile.get("rows", [])
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if (
            row.get("batch_size") == runtime.batch_size
            and row.get("sequence_length") == runtime.sequence_length
            and row.get("isotypic_multiplicity") == block.multiplicity
        ):
            return float(row["tensor_core_speedup_vs_scalar"])
    return None


def compile_isotypic_plan(
    blocks: tuple[IsotypicBlock, ...],
    runtime: RuntimeShape,
    hardware: HardwareTarget,
    *,
    certificate_path: str | Path | None = None,
    hardware_profile_path: str | Path | None = None,
    tensor_core_margin: float = 1.05,
) -> IsotypicExecutionPlan:
    """Compile algebraic blocks under conservative executable contracts."""

    if not blocks:
        raise ValueError("at least one isotypic block is required")
    profile = None
    profile_hash = None
    if hardware_profile_path is not None:
        profile_path = Path(hardware_profile_path)
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile_hash = sha256(profile_path)
        recorded_hardware = profile.get("hardware", {})
        recorded_capability = recorded_hardware.get("compute_capability", [])
        profile_compatible = (
            profile.get("passed") is True
            and profile.get("compiler_version") == COMPILER_VERSION
            and recorded_hardware.get("gpu") == hardware.name
            and tuple(recorded_capability) == hardware.compute_capability
        )
        if not profile_compatible:
            profile = None

    schedules = []
    for block in blocks:
        profile_speedup = _profile_lookup(profile, runtime, block)
        if hardware.backend != "cuda":
            backend: KernelBackend = "eager"
            accumulation = runtime.dtype
            tile = None
            reason = "no CUDA target"
        elif block.schur_type != "real" or block.irreducible_real_dimension != 8:
            backend = "eager"
            accumulation = runtime.dtype
            tile = None
            reason = "v2.1.1 has no compiled complex/quaternionic or non-8D kernel"
        elif runtime.dtype == "float32":
            backend = "triton_scalar"
            accumulation = "float32"
            tile = None
            reason = "full-gradient register recurrence is the maintained CUDA path"
        elif runtime.dtype == "float16":
            tensor_eligible = (
                not runtime.training
                and block.shared_action
                and hardware.tensor_cores
                and block.multiplicity >= 16
            )
            if (
                tensor_eligible
                and profile_speedup is not None
                and profile_speedup >= tensor_core_margin
            ):
                backend = "triton_tensor_core"
                accumulation = "float32"
                tile = 16
                reason = "exact hardware-profile cell clears the dispatch margin"
            else:
                backend = "triton_scalar"
                accumulation = "float32"
                tile = None
                reasons = []
                if runtime.training:
                    reasons.append("Tensor-Core recurrence is inference-only")
                if not block.shared_action:
                    reasons.append("actions differ across copies")
                if block.multiplicity < 16:
                    reasons.append("multiplicity does not fill one tile")
                if profile_speedup is None:
                    reasons.append("no exact device/shape profile cell")
                elif profile_speedup < tensor_core_margin:
                    reasons.append("profile does not clear the speed margin")
                reason = "; ".join(reasons) or "Tensor-Core eligibility failed"
        else:
            backend = "eager"
            accumulation = runtime.dtype
            tile = None
            reason = "float64 remains an eager correctness path"
        schedules.append(
            BlockSchedule(
                block=block,
                backend=backend,
                accumulation_dtype=accumulation,
                tile_multiplicity=tile,
                profile_speedup=profile_speedup,
                reason=reason,
            )
        )

    return IsotypicExecutionPlan(
        compiler_version=COMPILER_VERSION,
        runtime=runtime,
        hardware=hardware,
        schedules=tuple(schedules),
        certificate_sha256=(
            sha256(Path(certificate_path)) if certificate_path is not None else None
        ),
        profile_sha256=profile_hash,
    )


__all__ = [
    "COMPILER_VERSION",
    "BlockSchedule",
    "HardwareTarget",
    "IsotypicBlock",
    "IsotypicExecutionPlan",
    "RuntimeShape",
    "blocks_from_exact_certificate",
    "compile_isotypic_plan",
    "spin8_triality_blocks",
]
