"""Fail-closed adapters for the official FLA DeltaRule operators.

This is an operator adapter and semantic correctness reference. It does not
register an FLA or Hugging Face model. FLA is imported only when availability
is queried or an FLA backend is explicitly selected.

The supported fused API is the repository's pinned
``flash-linear-attention==0.5.2`` contract: sequence-major tensors with shapes
``q, k: (B, L, H, K)``, ``v: (B, L, H, V)``, and ``beta: (B, L, H)``.
The semantic paths narrowly adapt the exact recurrence and affine scan from
``Spin-Space-Research/src/schurscan_delta_memory.py`` to add a head axis.
"""

from __future__ import annotations

import importlib.metadata
import math
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Literal

import torch

DeltaRuleBackend = Literal[
    "semantic_recurrent",
    "semantic_parallel",
    "fla_chunk",
    "fla_recurrent",
]

DELTA_RULE_BACKENDS: tuple[DeltaRuleBackend, ...] = (
    "semantic_recurrent",
    "semantic_parallel",
    "fla_chunk",
    "fla_recurrent",
)
FLA_BACKENDS = frozenset(("fla_chunk", "fla_recurrent"))
SUPPORTED_FLA_VERSION = "0.5.2"
_FLA_DTYPES = (torch.float16, torch.bfloat16)

FlaOperator = Callable[..., tuple[torch.Tensor, torch.Tensor | None]]


@dataclass(frozen=True)
class FlaAvailability:
    """Serializable report for the pinned FLA operator contract."""

    installed: bool
    version: str | None
    device: str
    dtype: str
    reasons: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.installed and not self.reasons

    def __bool__(self) -> bool:
        return self.available

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def summary(self) -> str:
        status = "available" if self.available else "unavailable"
        details = "; ".join(self.reasons) if self.reasons else "all checks passed"
        return (
            f"FLA {status} (installed={self.installed}, version={self.version!r}, "
            f"device={self.device!r}, dtype={self.dtype!r}): {details}"
        )


def _distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("flash-linear-attention", "fla-core"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _import_fla_operators() -> tuple[FlaOperator, FlaOperator]:
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

    return chunk_delta_rule, fused_recurrent_delta_rule


def fla_available(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> FlaAvailability:
    """Report whether official FLA 0.5.2 DeltaRule ops can run as requested.

    The package and operators are checked lazily. Omitting arguments checks the
    adapter's default fused contract, CUDA with ``torch.float16``.
    """

    requested_device = torch.device("cuda" if device is None else device)
    requested_dtype = torch.float16 if dtype is None else dtype
    if not isinstance(requested_dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")

    versions = _distribution_versions()
    installed = bool(versions)
    version = versions.get("flash-linear-attention") or versions.get("fla-core")
    reasons: list[str] = []
    if not installed:
        reasons.append("flash-linear-attention/fla-core is not installed")
    else:
        mismatches = {
            name: found
            for name, found in versions.items()
            if found != SUPPORTED_FLA_VERSION
        }
        if mismatches:
            rendered = ", ".join(
                f"{name}={found}" for name, found in mismatches.items()
            )
            reasons.append(
                f"unsupported FLA version ({rendered}); expected {SUPPORTED_FLA_VERSION}"
            )

    if not sys.platform.startswith("linux"):
        reasons.append("the pinned FLA CUDA tier is supported only on Linux")
    if requested_device.type != "cuda":
        reasons.append(f"FLA DeltaRule requires a CUDA device, got {requested_device}")
    elif not torch.cuda.is_available():
        reasons.append("CUDA is not available")
    elif (
        requested_device.index is not None
        and requested_device.index >= torch.cuda.device_count()
    ):
        reasons.append(f"CUDA device index {requested_device.index} does not exist")
    if requested_dtype not in _FLA_DTYPES:
        reasons.append(
            "FLA DeltaRule requires torch.float16 or torch.bfloat16, "
            f"got {requested_dtype}"
        )

    if not reasons:
        try:
            _import_fla_operators()
        except Exception as error:  # noqa: BLE001 - third-party import failures vary.
            reasons.append(
                f"fla.ops.delta_rule import failed: {type(error).__name__}: {error}"
            )

    return FlaAvailability(
        installed=installed,
        version=version,
        device=str(requested_device),
        dtype=str(requested_dtype).removeprefix("torch."),
        reasons=tuple(reasons),
    )


def require_fla(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[FlaOperator, FlaOperator]:
    """Return official DeltaRule operators or raise with a structured report."""

    report = fla_available(device=device, dtype=dtype)
    if not report:
        raise RuntimeError(report.summary())
    try:
        return _import_fla_operators()
    except Exception as error:  # Guard against an import race after the report.
        raise RuntimeError(
            f"FLA became unavailable after validation: {type(error).__name__}: {error}"
        ) from error


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def delta_rule_state_scalars(
    heads: int,
    key_dimension: int,
    value_dimension: int,
) -> int:
    """Logical recurrent-state scalars per sequence."""

    return (
        _positive_integer("heads", heads)
        * _positive_integer("key_dimension", key_dimension)
        * _positive_integer("value_dimension", value_dimension)
    )


def delta_rule_state_bytes(
    heads: int,
    key_dimension: int,
    value_dimension: int,
    dtype: torch.dtype = torch.float32,
    *,
    batch_size: int = 1,
) -> int:
    """Logical recurrent-state bytes, including the requested batch size."""

    _positive_integer("batch_size", batch_size)
    try:
        element_size = torch.empty((), dtype=dtype).element_size()
    except (RuntimeError, TypeError) as error:
        raise ValueError("dtype must have fixed-width torch elements") from error
    return (
        batch_size
        * delta_rule_state_scalars(heads, key_dimension, value_dimension)
        * element_size
    )


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    tensors = {"q": q, "k": k, "v": v, "beta": beta}
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise TypeError("q, k, v, and beta must be torch tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or beta.ndim != 3:
        raise ValueError(
            "q, k, v, beta must have shapes (B,L,H,K), (B,L,H,K), "
            "(B,L,H,V), and (B,L,H)"
        )
    if q.shape != k.shape or q.shape[:3] != v.shape[:3] or q.shape[:3] != beta.shape:
        raise ValueError(
            "q, k, v, beta must have shapes (B,L,H,K), (B,L,H,K), "
            "(B,L,H,V), and (B,L,H) with matching B,L,H"
        )
    batch, length, heads, key_dimension = q.shape
    value_dimension = v.shape[-1]
    if min(batch, length, heads, key_dimension, value_dimension) < 1:
        raise ValueError("B, L, H, K, and V must all be positive")
    if not q.dtype.is_floating_point:
        raise TypeError("q, k, v, and beta must use a floating-point dtype")
    for name, tensor in tensors.items():
        if tensor.dtype != q.dtype:
            raise ValueError(
                f"{name} dtype {tensor.dtype} does not match q dtype {q.dtype}"
            )
        if tensor.device != q.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match q device {q.device}"
            )
    if initial_state is not None:
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("initial_state must be a torch tensor or None")
        expected = (batch, heads, key_dimension, value_dimension)
        if initial_state.shape != expected:
            raise ValueError(f"initial_state must have shape {expected}")
        if initial_state.dtype != q.dtype or initial_state.device != q.device:
            raise ValueError("initial_state must match the input dtype and device")
    return batch, length, heads, key_dimension, value_dimension


def _initial_state(
    q: torch.Tensor,
    value_dimension: int,
    initial_state: torch.Tensor | None,
) -> torch.Tensor:
    if initial_state is not None:
        return initial_state
    return q.new_zeros(q.shape[0], q.shape[2], q.shape[3], value_dimension)


def _semantic_recurrent_states(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state
    states = []
    for position in range(k.shape[1]):
        key = k[:, position]
        residual = v[:, position] - torch.einsum("bhk,bhkv->bhv", key, state)
        state = state + (
            beta[:, position, :, None, None]
            * key[..., :, None]
            * residual[..., None, :]
        )
        states.append(state)
    return torch.stack(states, dim=1), state


def _semantic_parallel_states(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive Hillis-Steele scan of the exact affine delta transitions."""

    key_dimension = k.shape[-1]
    identity = torch.eye(key_dimension, dtype=k.dtype, device=k.device).reshape(
        1, 1, 1, key_dimension, key_dimension
    )
    linear = identity - (beta[..., None, None] * k[..., :, None] * k[..., None, :])
    drive = beta[..., None, None] * k[..., :, None] * v[..., None, :]

    offset = 1
    while offset < k.shape[1]:
        composed_linear = linear[:, offset:] @ linear[:, :-offset]
        composed_drive = linear[:, offset:] @ drive[:, :-offset] + drive[:, offset:]
        linear = torch.cat((linear[:, :offset], composed_linear), dim=1)
        drive = torch.cat((drive[:, :offset], composed_drive), dim=1)
        offset *= 2

    states = linear @ initial_state[:, None] + drive
    return states, states[:, -1]


def delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    *,
    backend: DeltaRuleBackend = "semantic_recurrent",
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply an exact semantic or official fused DeltaRule operator.

    FLA backends never fall back. Selecting one either invokes the official
    operator or raises ``RuntimeError`` with an availability report.
    """

    if backend not in DELTA_RULE_BACKENDS:
        raise ValueError(
            f"unknown DeltaRule backend {backend!r}; expected one of {DELTA_RULE_BACKENDS}"
        )
    if not isinstance(scale, Real) or isinstance(scale, bool):
        raise TypeError("scale must be a real number")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    if not isinstance(output_final_state, bool):
        raise TypeError("output_final_state must be a bool")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size not in (16, 32, 64) and backend == "fla_chunk":
        raise ValueError("FLA 0.5.2 chunk_size must be 16, 32, or 64")

    _, _, _, _, value_dimension = _validate_inputs(q, k, v, beta, initial_state)
    if backend == "semantic_recurrent":
        state = _initial_state(q, value_dimension, initial_state)
        states, final_state = _semantic_recurrent_states(k, v, beta, state)
        output = scale * torch.einsum("blhk,blhkv->blhv", q, states)
        return output, final_state if output_final_state else None
    if backend == "semantic_parallel":
        state = _initial_state(q, value_dimension, initial_state)
        states, final_state = _semantic_parallel_states(k, v, beta, state)
        output = scale * torch.einsum("blhk,blhkv->blhv", q, states)
        return output, final_state if output_final_state else None

    chunk_operator, recurrent_operator = require_fla(device=q.device, dtype=q.dtype)
    operator = chunk_operator if backend == "fla_chunk" else recurrent_operator
    kwargs: dict[str, object] = {
        "scale": scale,
        "initial_state": initial_state,
        "output_final_state": output_final_state,
    }
    if backend == "fla_chunk":
        kwargs["chunk_size"] = chunk_size
    result = operator(q, k, v, beta, **kwargs)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("official FLA DeltaRule returned an invalid result")
    output, final_state = result
    expected_output = (*q.shape[:3], value_dimension)
    if output.shape != expected_output:
        raise RuntimeError(
            f"official FLA DeltaRule returned output shape {tuple(output.shape)}, "
            f"expected {expected_output}"
        )
    expected_state = (q.shape[0], q.shape[2], q.shape[3], value_dimension)
    if output_final_state and (
        final_state is None or final_state.shape != expected_state
    ):
        actual = None if final_state is None else tuple(final_state.shape)
        raise RuntimeError(
            f"official FLA DeltaRule returned final state shape {actual}, "
            f"expected {expected_state}"
        )
    return output, final_state


class DeltaRuleAdapter(torch.nn.Module):
    """Shape-bound operator adapter, not FLA/HF model registration."""

    def __init__(
        self,
        heads: int,
        key_dimension: int,
        value_dimension: int,
        *,
        backend: DeltaRuleBackend = "semantic_recurrent",
        scale: float = 1.0,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.heads = _positive_integer("heads", heads)
        self.key_dimension = _positive_integer("key_dimension", key_dimension)
        self.value_dimension = _positive_integer("value_dimension", value_dimension)
        if backend not in DELTA_RULE_BACKENDS:
            raise ValueError(f"unknown DeltaRule backend {backend!r}")
        self.backend = backend
        self.scale = scale
        self.chunk_size = chunk_size

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return (self.heads, self.key_dimension, self.value_dimension)

    @property
    def state_scalars(self) -> int:
        return delta_rule_state_scalars(*self.state_shape)

    def state_bytes(
        self,
        dtype: torch.dtype = torch.float32,
        *,
        batch_size: int = 1,
    ) -> int:
        return delta_rule_state_bytes(
            *self.state_shape, dtype=dtype, batch_size=batch_size
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if q.ndim == 4 and (
            q.shape[2:] != (self.heads, self.key_dimension)
            or v.ndim != 4
            or v.shape[2:] != (self.heads, self.value_dimension)
        ):
            raise ValueError(
                "input H/K/V dimensions do not match the adapter state shape "
                f"{self.state_shape}"
            )
        return delta_rule(
            q,
            k,
            v,
            beta,
            backend=self.backend,
            scale=self.scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=self.chunk_size,
        )


FlaDeltaRuleAdapter = DeltaRuleAdapter


__all__ = [
    "DELTA_RULE_BACKENDS",
    "FLA_BACKENDS",
    "SUPPORTED_FLA_VERSION",
    "DeltaRuleAdapter",
    "DeltaRuleBackend",
    "FlaAvailability",
    "FlaDeltaRuleAdapter",
    "delta_rule",
    "delta_rule_state_bytes",
    "delta_rule_state_scalars",
    "fla_available",
    "require_fla",
]
