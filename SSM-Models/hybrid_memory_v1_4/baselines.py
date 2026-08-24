"""Fail-closed registry for matched memory baselines.

Registry names select one implementation only. In particular, an unavailable
official fused implementation raises :class:`BaselineUnavailableError`; it is
never replaced by a semantic or unfused implementation.

``ProductKeyMemory`` is a static learned parameter-memory lookup. Its values
are model parameters shared by every example. It has no recurrent state and
cannot write or retain per-episode associations from an input sequence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from delta_product_reference import (
    DeltaProductReferenceLayer,
    DeltaProductReferenceModel,
)

from .fla_adapter import DeltaRuleAdapter, fla_available

Factory = Callable[..., nn.Module]
AvailabilityHook = Callable[[torch.device, torch.dtype], "BaselineAvailability"]


@dataclass(frozen=True)
class BaselineAvailability:
    """Serializable result of checking one exact registry implementation."""

    name: str
    available: bool
    device: str
    dtype: str
    reasons: tuple[str, ...] = ()
    detail: str | None = None

    def __bool__(self) -> bool:
        return self.available

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def summary(self) -> str:
        status = "available" if self.available else "unavailable"
        explanation = "; ".join(self.reasons) if self.reasons else "all checks passed"
        if self.detail:
            explanation = f"{explanation}; {self.detail}"
        return (
            f"baseline {self.name!r} {status} "
            f"(device={self.device!r}, dtype={self.dtype!r}): {explanation}"
        )


class BaselineUnavailableError(RuntimeError):
    """Runtime failure carrying the exact failed availability report."""

    def __init__(self, availability: BaselineAvailability) -> None:
        self.availability = availability
        self.report = availability
        super().__init__(availability.summary())

    def as_dict(self) -> dict[str, object]:
        return self.availability.as_dict()


@dataclass(frozen=True)
class BaselineSpec:
    """One immutable registry entry and its comparison boundary."""

    name: str
    factory: Factory
    availability_hook: AvailabilityHook
    official: bool
    fused: bool
    reference: bool
    claim_boundary: str
    fixed_cache_bytes: int | None = None
    variable_cache_bytes: int | None = None

    def availability(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> BaselineAvailability:
        requested_device, requested_dtype = _normalize_request(device, dtype)
        return self.availability_hook(requested_device, requested_dtype)

    def build(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        requested_device, requested_dtype = _normalize_request(device, dtype)
        report = self.availability_hook(requested_device, requested_dtype)
        if not report:
            raise BaselineUnavailableError(report)
        module = self.factory(**kwargs)
        module.to(device=requested_device, dtype=requested_dtype)
        return module

    def metadata(
        self,
        module: nn.Module | None = None,
        *,
        batch_size: int = 1,
        dtype: torch.dtype | None = None,
    ) -> dict[str, object]:
        return _metadata(self, module, batch_size=batch_size, dtype=dtype)


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _dtype_element_size(dtype: torch.dtype) -> int:
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    try:
        probe = torch.empty((), dtype=dtype)
    except (RuntimeError, TypeError) as error:
        raise TypeError("dtype must have fixed-width elements") from error
    if not probe.is_floating_point():
        raise TypeError("dtype must be floating point")
    return probe.element_size()


class ProductKeyMemory(nn.Module):
    """Static product-key parameter memory, not episodic recurrent memory.

    The query is split in half and scored against two independent subkey
    tables. The Cartesian product of each table's best subkeys is formed, the
    final global top-k pairs are selected, and their learned value parameters
    are combined with a softmax over pair scores. Arbitrary leading query
    dimensions are preserved.

    This module has no write path. It cannot store associations introduced in
    the current episode; training can only change the shared subkeys and value
    parameters.
    """

    memory_kind = "static_parameter_memory"
    supports_episode_writes = False
    claim_boundary = (
        "Static learned parameter memory only; it has no per-episode writes or "
        "episodic recurrent memory."
    )

    def __init__(
        self,
        query_dim: int,
        value_dim: int,
        num_subkeys: int,
        top_k: int = 4,
    ) -> None:
        super().__init__()
        self.query_dim = _positive_integer("query_dim", query_dim)
        self.value_dim = _positive_integer("value_dim", value_dim)
        self.num_subkeys = _positive_integer("num_subkeys", num_subkeys)
        self.top_k = _positive_integer("top_k", top_k)
        if self.query_dim % 2:
            raise ValueError("query_dim must be even for two product subkeys")
        if self.top_k > self.num_subkeys**2:
            raise ValueError("top_k cannot exceed num_subkeys squared")

        self.subkey_dim = self.query_dim // 2
        self.subkeys = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.num_subkeys, self.subkey_dim)),
                nn.Parameter(torch.empty(self.num_subkeys, self.subkey_dim)),
            ]
        )
        self.values = nn.Parameter(torch.empty(self.num_subkeys**2, self.value_dim))
        self.reset_parameters()

    @property
    def state_scalars(self) -> int:
        return 0

    def state_bytes(
        self,
        dtype: torch.dtype = torch.float32,
        *,
        batch_size: int = 1,
    ) -> int:
        _positive_integer("batch_size", batch_size)
        _dtype_element_size(dtype)
        return 0

    def reset_parameters(self) -> None:
        for table in self.subkeys:
            nn.init.normal_(table, mean=0.0, std=self.subkey_dim**-0.5)
        nn.init.normal_(self.values, mean=0.0, std=0.02)

    def _validate_query(self, query: torch.Tensor) -> None:
        if not isinstance(query, torch.Tensor):
            raise TypeError("query must be a torch tensor")
        if query.ndim < 1 or query.shape[-1] != self.query_dim:
            raise ValueError(f"query must have shape (..., {self.query_dim})")
        if not query.is_floating_point():
            raise TypeError("query must use a floating-point dtype")
        if query.device != self.values.device or query.dtype != self.values.dtype:
            raise ValueError("query must match the memory parameter device and dtype")

    def topk(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final pair scores and flattened value-table indices."""

        self._validate_query(query)
        first_query, second_query = query.split(self.subkey_dim, dim=-1)
        first_scores = first_query @ self.subkeys[0].T
        second_scores = second_query @ self.subkeys[1].T

        subkey_k = min(self.top_k, self.num_subkeys)
        first_scores, first_indices = first_scores.topk(subkey_k, dim=-1)
        second_scores, second_indices = second_scores.topk(subkey_k, dim=-1)
        candidate_scores = (
            first_scores.unsqueeze(-1) + second_scores.unsqueeze(-2)
        ).flatten(start_dim=-2)
        candidate_indices = (
            first_indices.unsqueeze(-1) * self.num_subkeys
            + second_indices.unsqueeze(-2)
        ).flatten(start_dim=-2)

        scores, selected = candidate_scores.topk(self.top_k, dim=-1)
        indices = candidate_indices.gather(-1, selected)
        return scores, indices

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        scores, indices = self.topk(query)
        weights = torch.softmax(scores, dim=-1)
        selected_values = self.values[indices]
        return torch.sum(weights.unsqueeze(-1) * selected_values, dim=-2)

    def extra_repr(self) -> str:
        return (
            f"query_dim={self.query_dim}, value_dim={self.value_dim}, "
            f"num_subkeys={self.num_subkeys}, top_k={self.top_k}, "
            "memory_kind='static_parameter_memory'"
        )


StaticProductKeyMemory = ProductKeyMemory


def _normalize_request(
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> tuple[torch.device, torch.dtype]:
    try:
        requested_device = torch.device("cpu" if device is None else device)
    except (RuntimeError, TypeError) as error:
        raise TypeError("device must identify a torch device") from error
    requested_dtype = torch.float32 if dtype is None else dtype
    if not isinstance(requested_dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    return requested_device, requested_dtype


def _device_and_dtype_reasons(
    device: torch.device,
    dtype: torch.dtype,
) -> list[str]:
    reasons: list[str] = []
    try:
        floating = torch.empty((), dtype=dtype).is_floating_point()
    except (RuntimeError, TypeError):
        floating = False
    if not floating:
        reasons.append(f"a floating-point dtype is required, got {dtype}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            reasons.append("CUDA is not available")
        elif device.index is not None and device.index >= torch.cuda.device_count():
            reasons.append(f"CUDA device index {device.index} does not exist")
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            reasons.append("MPS is not available")
    elif device.type != "cpu":
        reasons.append(f"unsupported device type {device.type!r}")
    return reasons


def _local_availability(name: str) -> AvailabilityHook:
    def check(device: torch.device, dtype: torch.dtype) -> BaselineAvailability:
        reasons = tuple(_device_and_dtype_reasons(device, dtype))
        return BaselineAvailability(
            name=name,
            available=not reasons,
            device=str(device),
            dtype=str(dtype).removeprefix("torch."),
            reasons=reasons,
        )

    return check


def _fla_fused_availability(
    device: torch.device,
    dtype: torch.dtype,
) -> BaselineAvailability:
    report = fla_available(device=device, dtype=dtype)
    return BaselineAvailability(
        name="fla_delta_fused",
        available=report.available,
        device=report.device,
        dtype=report.dtype,
        reasons=report.reasons,
        detail=None if report.version is None else f"FLA version {report.version}",
    )


def _mamba2_components() -> tuple[type[nn.Module], Callable[[], tuple[bool, str]]]:
    # This repository wrapper imports mamba_ssm only inside its probe/constructor.
    from pure_spin_ssm_v1_2.mamba2_baseline import (
        OfficialMamba2LM,
        fused_mamba2_available,
    )

    return OfficialMamba2LM, fused_mamba2_available


def _mamba2_availability(
    device: torch.device,
    dtype: torch.dtype,
) -> BaselineAvailability:
    reasons = _device_and_dtype_reasons(device, dtype)
    if device.type != "cuda":
        reasons.append(f"official fused Mamba-2 requires CUDA, got {device}")
    detail: str | None = None
    if not reasons:
        try:
            _, probe = _mamba2_components()
            available, detail = probe()
        except (ImportError, RuntimeError, OSError) as error:
            available = False
            detail = f"{type(error).__name__}: {error}"
        if not available:
            reasons.append("official fused mamba_ssm Mamba-2 is unavailable")
    return BaselineAvailability(
        name="mamba2_official",
        available=not reasons,
        device=str(device),
        dtype=str(dtype).removeprefix("torch."),
        reasons=tuple(reasons),
        detail=detail,
    )


def _semantic_delta_factory(**kwargs: Any) -> nn.Module:
    if "backend" in kwargs:
        raise TypeError("fla_delta_semantic fixes backend='semantic_recurrent'")
    return DeltaRuleAdapter(backend="semantic_recurrent", **kwargs)


def _fused_delta_factory(**kwargs: Any) -> nn.Module:
    if "backend" in kwargs:
        raise TypeError("fla_delta_fused fixes backend='fla_chunk'")
    return DeltaRuleAdapter(backend="fla_chunk", **kwargs)


def _mamba2_factory(**kwargs: Any) -> nn.Module:
    model_type, _ = _mamba2_components()
    return model_type(**kwargs)


_PRODUCT_KEY_BOUNDARY = (
    "Static learned parameter memory only; no per-episode writes, associations, "
    "or episodic recurrent state."
)

BASELINE_REGISTRY: dict[str, BaselineSpec] = {
    "delta_product_reference": BaselineSpec(
        name="delta_product_reference",
        factory=DeltaProductReferenceModel,
        availability_hook=_local_availability("delta_product_reference"),
        official=False,
        fused=False,
        reference=True,
        claim_boundary=(
            "Repository PyTorch architecture reference, not the official fused "
            "DeltaProduct implementation and not a fused-kernel timing baseline."
        ),
        variable_cache_bytes=0,
    ),
    "fla_delta_semantic": BaselineSpec(
        name="fla_delta_semantic",
        factory=_semantic_delta_factory,
        availability_hook=_local_availability("fla_delta_semantic"),
        official=False,
        fused=False,
        reference=True,
        claim_boundary=(
            "Repository exact semantic DeltaRule operator adapter; not an official "
            "FLA kernel, complete language model, or fused timing baseline."
        ),
        variable_cache_bytes=0,
    ),
    "fla_delta_fused": BaselineSpec(
        name="fla_delta_fused",
        factory=_fused_delta_factory,
        availability_hook=_fla_fused_availability,
        official=True,
        fused=True,
        reference=False,
        claim_boundary=(
            "Pinned official FLA 0.5.2 DeltaRule operator adapter only; it is not "
            "a complete matched language model."
        ),
        variable_cache_bytes=0,
    ),
    "mamba2_official": BaselineSpec(
        name="mamba2_official",
        factory=_mamba2_factory,
        availability_hook=_mamba2_availability,
        official=True,
        fused=True,
        reference=False,
        claim_boundary=(
            "Repository wrapper around official fused mamba_ssm Mamba-2; valid only "
            "for environments passing the exact availability probe."
        ),
    ),
    "product_key_static": BaselineSpec(
        name="product_key_static",
        factory=ProductKeyMemory,
        availability_hook=_local_availability("product_key_static"),
        official=False,
        fused=False,
        reference=True,
        claim_boundary=_PRODUCT_KEY_BOUNDARY,
        fixed_cache_bytes=0,
        variable_cache_bytes=0,
    ),
}

BASELINE_NAMES = tuple(BASELINE_REGISTRY)


def get_baseline_spec(name: str) -> BaselineSpec:
    if not isinstance(name, str):
        raise TypeError("baseline name must be a string")
    try:
        return BASELINE_REGISTRY[name]
    except KeyError as error:
        raise ValueError(
            f"unknown baseline {name!r}; expected one of {BASELINE_NAMES}"
        ) from error


def baseline_availability(
    name: str,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> BaselineAvailability:
    """Check one exact implementation without selecting a substitute."""

    return get_baseline_spec(name).availability(device=device, dtype=dtype)


def build_baseline(
    name: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Build one exact registry implementation or fail with its report."""

    return get_baseline_spec(name).build(device=device, dtype=dtype, **kwargs)


def parameter_count(module: nn.Module) -> int:
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    return sum(parameter.numel() for parameter in module.parameters())


def _module_dtype(module: nn.Module) -> torch.dtype | None:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.dtype
    buffer = next(module.buffers(), None)
    return None if buffer is None else buffer.dtype


def _fixed_cache_bytes(
    spec: BaselineSpec,
    module: nn.Module | None,
    *,
    batch_size: int,
    dtype: torch.dtype,
) -> int | None:
    if spec.fixed_cache_bytes is not None:
        return spec.fixed_cache_bytes
    if module is None:
        return None
    state_bytes = getattr(module, "state_bytes", None)
    if callable(state_bytes):
        return int(state_bytes(dtype, batch_size=batch_size))
    if isinstance(module, DeltaProductReferenceModel):
        return batch_size * module.recurrent_state_scalars * _dtype_element_size(dtype)
    if isinstance(module, DeltaProductReferenceLayer):
        scalars = module.num_heads * module.head_dim**2
        return batch_size * scalars * _dtype_element_size(dtype)
    return None


def _metadata(
    spec: BaselineSpec,
    module: nn.Module | None,
    *,
    batch_size: int,
    dtype: torch.dtype | None,
) -> dict[str, object]:
    _positive_integer("batch_size", batch_size)
    if module is not None and not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module or None")
    effective_dtype = (
        dtype
        or (_module_dtype(module) if module is not None else None)
        or torch.float32
    )
    _dtype_element_size(effective_dtype)
    fixed = _fixed_cache_bytes(
        spec, module, batch_size=batch_size, dtype=effective_dtype
    )
    variable = spec.variable_cache_bytes
    total = None if fixed is None or variable is None else fixed + variable
    return {
        "name": spec.name,
        "parameter_count": None if module is None else parameter_count(module),
        "fixed_cache_bytes": fixed,
        "variable_cache_bytes": variable,
        "total_cache_bytes": total,
        "official": spec.official,
        "fused": spec.fused,
        "reference": spec.reference,
        "claim_boundary": spec.claim_boundary,
    }


def baseline_metadata(
    name: str,
    module: nn.Module | None = None,
    *,
    batch_size: int = 1,
    dtype: torch.dtype | None = None,
) -> dict[str, object]:
    """Return comparison metadata, preserving unknown cache values as ``None``."""

    return get_baseline_spec(name).metadata(module, batch_size=batch_size, dtype=dtype)


__all__ = [
    "BASELINE_NAMES",
    "BASELINE_REGISTRY",
    "BaselineAvailability",
    "BaselineSpec",
    "BaselineUnavailableError",
    "DeltaProductReferenceLayer",
    "DeltaProductReferenceModel",
    "ProductKeyMemory",
    "StaticProductKeyMemory",
    "baseline_availability",
    "baseline_metadata",
    "build_baseline",
    "get_baseline_spec",
    "parameter_count",
]
