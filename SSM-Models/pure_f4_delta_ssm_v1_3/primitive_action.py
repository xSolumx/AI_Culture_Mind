"""Exact primitive-subgroup transport for the Albert representation.

The existing direct exceptional action evaluates ``exp(sum_a theta_a G_a)``
with a dense 27 by 27 matrix exponential at every token.  This module exposes
the complementary canonical-coordinate chart

``exp(theta_{F-1} G_{F-1}) ... exp(theta_0 G_0)``.

Every generator produced by :mod:`albert` splits into disconnected real blocks
of size at most three.  Compact F4 blocks are evaluated with Rodrigues' exact
formula; the symmetric complement of E6(-26) is evaluated in a fixed real
eigenbasis.  Consequently no per-token dense matrix exponential is needed.

``primitive_product_reference`` is the portable differentiable oracle.  The
CUDA entry point uses a source-built SM75 kernel with a hand-written reverse
recurrence.  It reconstructs intermediate states with exact inverse subgroup
actions, so backward stores only the final transported value rather than one
state per primitive factor.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.cpp_extension import load

from .albert import ALBERT_DIM, build_albert_algebra

PrimitiveAlgebra = Literal["f4", "e6"]
_BLOCK_SIZE = 3
_BLOCK_COUNT = ALBERT_DIM // _BLOCK_SIZE
_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PrimitiveMetadata:
    """Block data for an ordered basis of one-parameter subgroups."""

    algebra: PrimitiveAlgebra
    generators: torch.Tensor
    permutations: torch.Tensor
    kinds: torch.Tensor
    local_generators: torch.Tensor
    local_generator_squares: torch.Tensor
    frequencies: torch.Tensor
    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor
    operator_norms: torch.Tensor
    content_mask: torch.Tensor

    @property
    def factor_count(self) -> int:
        return int(self.generators.shape[0])


def _connected_components(generator: np.ndarray, tolerance: float) -> list[list[int]]:
    adjacency = (np.abs(generator) > tolerance) | (
        np.abs(generator.T) > tolerance
    )
    remaining = set(range(ALBERT_DIM))
    components: list[list[int]] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack = [root]
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = set(np.flatnonzero(adjacency[node]).tolist()) & remaining
            remaining.difference_update(neighbours)
            stack.extend(sorted(neighbours, reverse=True))
        components.append(sorted(component))
    return components


def _pack_components(components: list[list[int]]) -> np.ndarray:
    """Pack disconnected components into nine independent 3D blocks."""

    if any(len(component) > _BLOCK_SIZE for component in components):
        raise ValueError("primitive generator has a connected block larger than 3")
    bins: list[list[int]] = []
    for component in sorted(components, key=lambda item: (-len(item), item[0])):
        for block in bins:
            if len(block) + len(component) <= _BLOCK_SIZE:
                block.extend(component)
                break
        else:
            bins.append(list(component))
    if len(bins) != _BLOCK_COUNT or any(len(block) != _BLOCK_SIZE for block in bins):
        raise AssertionError("the 27D primitive blocks did not pack into nine triples")
    permutation = np.asarray([index for block in bins for index in block], dtype=np.int32)
    if sorted(permutation.tolist()) != list(range(ALBERT_DIM)):
        raise AssertionError("primitive block packing is not a permutation")
    return permutation


@lru_cache(maxsize=2)
def build_primitive_metadata(algebra: PrimitiveAlgebra) -> PrimitiveMetadata:
    """Derive the exact small-block description from the maintained bank."""

    data = build_albert_algebra()
    if algebra == "f4":
        generators = np.asarray(data.f4, dtype=np.float64)
    elif algebra == "e6":
        generators = np.asarray(data.e6, dtype=np.float64)
    else:  # pragma: no cover - Literal plus public runtime validation
        raise ValueError(f"unsupported primitive algebra {algebra!r}")

    factor_count = generators.shape[0]
    permutations = np.empty((factor_count, ALBERT_DIM), dtype=np.int32)
    kinds = np.empty(factor_count, dtype=np.int32)
    local = np.zeros(
        (factor_count, _BLOCK_COUNT, _BLOCK_SIZE, _BLOCK_SIZE), dtype=np.float64
    )
    local_square = np.zeros_like(local)
    frequencies = np.zeros((factor_count, _BLOCK_COUNT), dtype=np.float64)
    eigenvectors = np.zeros_like(local)
    eigenvalues = np.zeros(
        (factor_count, _BLOCK_COUNT, _BLOCK_SIZE), dtype=np.float64
    )
    operator_norms = np.linalg.norm(generators, ord=2, axis=(-2, -1))

    tolerance = 2e-11
    for factor, generator in enumerate(generators):
        skew_residual = float(np.max(np.abs(generator + generator.T)))
        symmetric_residual = float(np.max(np.abs(generator - generator.T)))
        if skew_residual <= tolerance:
            kind = 0
        elif symmetric_residual <= tolerance:
            kind = 1
        else:
            raise ValueError(
                f"generator {factor} is neither skew nor symmetric: "
                f"{skew_residual=:.3e}, {symmetric_residual=:.3e}"
            )
        kinds[factor] = kind
        permutation = _pack_components(_connected_components(generator, tolerance))
        permutations[factor] = permutation
        for block in range(_BLOCK_COUNT):
            indices = permutation[
                block * _BLOCK_SIZE : (block + 1) * _BLOCK_SIZE
            ]
            local_generator = generator[np.ix_(indices, indices)]
            local[factor, block] = local_generator
            local_square[factor, block] = local_generator @ local_generator
            if kind == 0:
                frequency_square = max(
                    0.0, -0.5 * float(np.trace(local_generator @ local_generator))
                )
                frequencies[factor, block] = frequency_square**0.5
                eigenvectors[factor, block] = np.eye(_BLOCK_SIZE)
            else:
                values, vectors = np.linalg.eigh(local_generator)
                eigenvalues[factor, block] = values
                eigenvectors[factor, block] = vectors

        reconstructed = np.zeros_like(generator)
        for block in range(_BLOCK_COUNT):
            indices = permutation[
                block * _BLOCK_SIZE : (block + 1) * _BLOCK_SIZE
            ]
            reconstructed[np.ix_(indices, indices)] = local[factor, block]
        residual = float(np.max(np.abs(reconstructed - generator)))
        if residual > tolerance:
            raise AssertionError(
                f"primitive block decomposition changed generator {factor}: {residual}"
            )

    return PrimitiveMetadata(
        algebra=algebra,
        generators=torch.from_numpy(generators.copy()),
        permutations=torch.from_numpy(permutations),
        kinds=torch.from_numpy(kinds),
        local_generators=torch.from_numpy(local),
        local_generator_squares=torch.from_numpy(local_square),
        frequencies=torch.from_numpy(frequencies),
        eigenvectors=torch.from_numpy(eigenvectors),
        eigenvalues=torch.from_numpy(eigenvalues),
        operator_norms=torch.from_numpy(operator_norms),
        content_mask=torch.from_numpy(kinds == 1),
    )


def _apply_one_reference(
    values: torch.Tensor,
    angle: torch.Tensor,
    factor: int,
    metadata: PrimitiveMetadata,
) -> torch.Tensor:
    permutation = metadata.permutations[factor].to(values.device, dtype=torch.long)
    inverse = torch.argsort(permutation)
    blocked = values.index_select(-1, permutation).reshape(
        *values.shape[:-1], _BLOCK_COUNT, _BLOCK_SIZE
    )
    if int(metadata.kinds[factor]) == 0:
        generator = metadata.local_generators[factor].to(values)
        square = metadata.local_generator_squares[factor].to(values)
        frequency = metadata.frequencies[factor].to(values)
        first = torch.einsum("bij,...hbj->...hbi", generator, blocked)
        second = torch.einsum("bij,...hbj->...hbi", square, blocked)
        argument = angle[..., None] * frequency
        nonzero = frequency.abs() > 1e-12
        linear = torch.where(
            nonzero,
            torch.sin(argument)
            / torch.where(nonzero, frequency, torch.ones_like(frequency)),
            angle[..., None].expand_as(argument),
        )
        quadratic = torch.where(
            nonzero,
            (1.0 - torch.cos(argument))
            / torch.where(nonzero, frequency.square(), torch.ones_like(frequency)),
            0.5 * angle[..., None].square().expand_as(argument),
        )
        transformed = (
            blocked
            + linear[..., None, :, None] * first
            + quadratic[..., None, :, None] * second
        )
    else:
        vectors = metadata.eigenvectors[factor].to(values)
        eigenvalues = metadata.eigenvalues[factor].to(values)
        modal = torch.einsum("bim,...hbi->...hbm", vectors, blocked)
        modal = modal * torch.exp(angle[..., None, None, None] * eigenvalues)
        transformed = torch.einsum("bim,...hbm->...hbi", vectors, modal)
    flattened = transformed.flatten(start_dim=-2)
    return flattened.index_select(-1, inverse)


def primitive_product_reference(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    algebra: PrimitiveAlgebra = "e6",
) -> torch.Tensor:
    """Apply the ordered primitive product with ordinary PyTorch autograd.

    ``values`` ends in ``(copies,27)`` and ``coordinates`` has the same leading
    dimensions followed by the primitive-factor dimension.  Factor zero acts
    first.  The implementation never calls ``torch.matrix_exp``.
    """

    metadata = build_primitive_metadata(algebra)
    if values.ndim < 2 or values.shape[-1] != ALBERT_DIM:
        raise ValueError("values must end in (copies,27)")
    if coordinates.shape[:-1] != values.shape[:-2]:
        raise ValueError("coordinate leading dimensions must match values")
    if coordinates.shape[-1] != metadata.factor_count:
        raise ValueError(
            f"{algebra} primitive coordinates require {metadata.factor_count} factors"
        )
    if values.device != coordinates.device or values.dtype != coordinates.dtype:
        raise ValueError("values and coordinates must share device and dtype")
    result = values
    for factor in range(metadata.factor_count):
        result = _apply_one_reference(
            result, coordinates[..., factor], factor, metadata
        )
    return result


def dense_primitive_product_oracle(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    algebra: PrimitiveAlgebra = "e6",
) -> torch.Tensor:
    """Slow dense-matrix-exponential oracle used only for qualification."""

    metadata = build_primitive_metadata(algebra)
    if coordinates.shape[-1] != metadata.factor_count:
        raise ValueError("coordinate count does not match primitive bank")
    result = values
    generators = metadata.generators.to(values)
    for factor in range(metadata.factor_count):
        action = torch.matrix_exp(
            coordinates[..., factor, None, None] * generators[factor]
        )
        result = result @ action.transpose(-1, -2)
    return result


def _extension_name() -> str:
    sources = (
        _ROOT / "primitive_action_bindings.cpp",
        _ROOT / "primitive_action_cuda.cu",
    )
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    digest.update(torch.__version__.encode())
    digest.update(str(torch.version.cuda).encode())
    digest.update(platform.python_version().encode())
    digest.update(os.name.encode())
    return f"pure_exceptional_primitive_sm75_{digest.hexdigest()[:16]}"


def _cuda_dependency_include_paths() -> list[str]:
    nvidia_root = Path(torch.__file__).resolve().parent.parent / "nvidia"
    candidates = (
        nvidia_root / "cublas" / "include",
        nvidia_root / "cusparse" / "include",
        nvidia_root / "cusolver" / "include",
        nvidia_root / "cudnn" / "include",
        nvidia_root / "cuda_runtime" / "include",
    )
    return [str(path) for path in candidates if path.is_dir()]


@lru_cache(maxsize=1)
def _cuda_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("primitive CUDA transport requires CUDA")
    if torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("primitive CUDA transport is qualified only for SM75")
    original_path = os.environ.get("PATH", "")
    # Do not resolve the venv's Python symlink: ninja lives beside that symlink,
    # not beside the base interpreter it targets.
    executable_bin = str(Path(sys.executable).parent)
    if shutil.which("ninja") is None and (Path(executable_bin) / "ninja").is_file():
        os.environ["PATH"] = executable_bin + os.pathsep + original_path
    try:
        return load(
            name=_extension_name(),
            sources=[
                str(_ROOT / "primitive_action_bindings.cpp"),
                str(_ROOT / "primitive_action_cuda.cu"),
            ],
            extra_include_paths=_cuda_dependency_include_paths(),
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                "-gencode=arch=compute_75,code=sm_75",
            ],
            extra_cflags=["-O3"],
            verbose=False,
        )
    finally:
        os.environ["PATH"] = original_path


class _PrimitiveProductCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, coordinates: torch.Tensor, *metadata):
        extension = _cuda_extension()
        output = extension.forward(values.contiguous(), coordinates.contiguous(), *metadata)
        ctx.save_for_backward(coordinates.contiguous(), output, *metadata)
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        coordinates, output, *metadata = ctx.saved_tensors
        value_gradient, coordinate_gradient = _cuda_extension().backward(
            coordinates,
            output,
            output_gradient.contiguous(),
            *metadata,
        )
        return value_gradient, coordinate_gradient, *(None for _ in metadata)


def primitive_event_layout(
    length: int,
    event_stride: int,
    event_phase: int = 0,
    position_offset: int = 0,
) -> tuple[int, int]:
    """Return ``(first_local_position,event_count)`` for a sequence chunk."""

    if length < 1 or event_stride < 1 or position_offset < 0:
        raise ValueError("length/stride must be positive and offset nonnegative")
    if not 0 <= event_phase < event_stride:
        raise ValueError("event_phase must lie in [0,event_stride)")
    if position_offset <= event_phase:
        first_global = event_phase
    else:
        distance = position_offset - event_phase
        first_global = event_phase + (
            (distance + event_stride - 1) // event_stride
        ) * event_stride
    first_local = first_global - position_offset
    count = (
        0
        if first_local >= length
        else 1 + (length - 1 - first_local) // event_stride
    )
    return first_local, count


def primitive_delta_recurrence_reference(
    retention: torch.Tensor,
    write_key: torch.Tensor,
    erase_key: torch.Tensor,
    write_value: torch.Tensor,
    initial_state: torch.Tensor,
    query: torch.Tensor,
    event_coordinates: torch.Tensor,
    algebra: PrimitiveAlgebra = "e6",
    *,
    event_stride: int,
    event_phase: int = 0,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Portable oracle for retention/erase/event-action/write/read semantics."""

    if retention.ndim != 3:
        raise ValueError("retention must be (B,L,H)")
    batch, length, heads = retention.shape
    if write_key.ndim != 4 or write_key.shape != erase_key.shape:
        raise ValueError("keys must share shape (B,L,R,H)")
    rank = write_key.shape[2]
    if write_key.shape != (batch, length, rank, heads):
        raise ValueError("key shape mismatch")
    if write_value.shape != (batch, length, rank, ALBERT_DIM):
        raise ValueError("write value must be (B,L,R,27)")
    if initial_state.shape != (batch, heads, ALBERT_DIM):
        raise ValueError("initial state must be (B,H,27)")
    if query.shape != retention.shape:
        raise ValueError("query must match retention")
    first_local, event_count = primitive_event_layout(
        length, event_stride, event_phase, position_offset
    )
    factor_count = build_primitive_metadata(algebra).factor_count
    if event_coordinates.shape != (batch, event_count, factor_count):
        raise ValueError("event coordinate shape does not match event layout")
    tensors = (
        retention,
        write_key,
        erase_key,
        write_value,
        initial_state,
        query,
        event_coordinates,
    )
    if any(tensor.device != retention.device for tensor in tensors):
        raise ValueError("all recurrence tensors must share a device")
    if any(tensor.dtype != retention.dtype for tensor in tensors):
        raise ValueError("all recurrence tensors must share a dtype")
    state = initial_state
    reads = []
    event = 0
    for position in range(length):
        retained = retention[:, position, :, None] * state
        projections = torch.einsum(
            "brh,bhv->brv", erase_key[:, position], retained
        )
        state = retained - torch.einsum(
            "brh,brv->bhv", write_key[:, position], projections
        )
        if position >= first_local and (position - first_local) % event_stride == 0:
            state = primitive_product_reference(
                state, event_coordinates[:, event], algebra
            )
            event += 1
        state = state + torch.einsum(
            "brh,brv->bhv", write_key[:, position], write_value[:, position]
        )
        reads.append(torch.einsum("bh,bhv->bv", query[:, position], state))
    if event != event_count:
        raise AssertionError("event recurrence consumed the wrong coordinate count")
    return torch.stack(reads, dim=1), state


class _PrimitiveDeltaCuda(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        retention,
        write_key,
        erase_key,
        write_value,
        initial_state,
        query,
        event_coordinates,
        *metadata_and_layout,
    ):
        metadata = metadata_and_layout[:-2]
        event_stride, first_event_local = metadata_and_layout[-2:]
        reads, final_state, states = _cuda_extension().delta_forward(
            retention.contiguous(),
            write_key.contiguous(),
            erase_key.contiguous(),
            write_value.contiguous(),
            initial_state.contiguous(),
            query.contiguous(),
            event_coordinates.contiguous(),
            *metadata,
            event_stride,
            first_event_local,
        )
        ctx.save_for_backward(
            retention,
            write_key,
            erase_key,
            write_value,
            initial_state,
            query,
            event_coordinates,
            states,
            *metadata,
        )
        ctx.event_stride = event_stride
        ctx.first_event_local = first_event_local
        ctx.metadata_count = len(metadata)
        return reads, final_state

    @staticmethod
    def backward(ctx, read_gradient, final_gradient):
        saved = ctx.saved_tensors
        recurrence = saved[:7]
        states = saved[7]
        metadata = saved[8:]
        if read_gradient is None:
            retention = recurrence[0]
            read_gradient = retention.new_zeros(
                retention.shape[0], retention.shape[1], ALBERT_DIM
            )
        if final_gradient is None:
            final_gradient = torch.zeros_like(recurrence[4])
        gradients = _cuda_extension().delta_backward(
            *recurrence,
            states,
            read_gradient.contiguous(),
            final_gradient.contiguous(),
            *metadata,
            ctx.event_stride,
            ctx.first_event_local,
        )
        return (*gradients, *(None for _ in metadata), None, None)


def primitive_delta_recurrence_cuda(
    retention: torch.Tensor,
    write_key: torch.Tensor,
    erase_key: torch.Tensor,
    write_value: torch.Tensor,
    initial_state: torch.Tensor,
    query: torch.Tensor,
    event_coordinates: torch.Tensor,
    action: "PrimitiveExceptionalAction",
    *,
    event_stride: int,
    event_phase: int = 0,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SM75 sparse-event Delta recurrence returning reads and final state."""

    if retention.device.type != "cuda" or retention.dtype != torch.float32:
        raise ValueError("primitive Delta CUDA requires CUDA FP32 tensors")
    batch, length, _ = retention.shape
    first_local, event_count = primitive_event_layout(
        length, event_stride, event_phase, position_offset
    )
    if event_coordinates.shape != (batch, event_count, action.coordinate_dim):
        raise ValueError("event coordinate shape does not match event layout")
    metadata = (
        action.permutations,
        action.kinds,
        action.local_generators,
        action.local_generator_squares,
        action.frequencies,
        action.eigenvectors,
        action.eigenvalues,
    )
    if any(tensor.device != retention.device for tensor in metadata):
        raise ValueError("move PrimitiveExceptionalAction to the recurrence device")
    if any(tensor.dtype != torch.float32 for tensor in metadata[2:]):
        raise ValueError("cast PrimitiveExceptionalAction floating buffers to FP32")
    return _PrimitiveDeltaCuda.apply(
        retention,
        write_key,
        erase_key,
        write_value,
        initial_state,
        query,
        event_coordinates,
        *metadata,
        event_stride,
        first_local,
    )


def _primitive_product_cuda_with_metadata(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    leading = coordinates.shape[:-1]
    copies = values.shape[-2]
    flat_values = values.reshape(-1, copies, ALBERT_DIM)
    flat_coordinates = coordinates.reshape(-1, coordinates.shape[-1])
    output = _PrimitiveProductCuda.apply(flat_values, flat_coordinates, *tensors)
    return output.reshape(*leading, copies, ALBERT_DIM)


def primitive_product_cuda(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    algebra: PrimitiveAlgebra = "e6",
) -> torch.Tensor:
    """Apply the ordered primitive product with the exact SM75 CUDA backend."""

    metadata = build_primitive_metadata(algebra)
    if values.device.type != "cuda" or coordinates.device.type != "cuda":
        raise ValueError("primitive CUDA inputs must be CUDA tensors")
    if values.dtype != torch.float32 or coordinates.dtype != torch.float32:
        raise ValueError("primitive CUDA transport uses FP32 state and coordinates")
    if values.ndim < 2 or values.shape[-1] != ALBERT_DIM:
        raise ValueError("values must end in (copies,27)")
    if coordinates.shape[:-1] != values.shape[:-2]:
        raise ValueError("coordinate leading dimensions must match values")
    if coordinates.shape[-1] != metadata.factor_count:
        raise ValueError("coordinate count does not match primitive bank")
    tensors = (
        metadata.permutations.to(device=values.device),
        metadata.kinds.to(device=values.device),
        metadata.local_generators.to(values),
        metadata.local_generator_squares.to(values),
        metadata.frequencies.to(values),
        metadata.eigenvectors.to(values),
        metadata.eigenvalues.to(values),
    )
    return _primitive_product_cuda_with_metadata(values, coordinates, tensors)


class PrimitiveExceptionalAction(nn.Module):
    """Module wrapper for the canonical primitive F4/E6 chart."""

    geometry = "canonical_product"
    representation_dim = ALBERT_DIM

    def __init__(self, algebra: PrimitiveAlgebra = "e6", *, backend: str = "auto"):
        super().__init__()
        if algebra not in {"f4", "e6"}:
            raise ValueError("primitive action supports f4 or e6")
        if backend not in {"auto", "reference", "cuda"}:
            raise ValueError("backend must be auto, reference, or cuda")
        self.algebra = algebra
        self.backend = backend
        metadata = build_primitive_metadata(algebra)
        self.coordinate_dim = metadata.factor_count
        for name in (
            "generators",
            "permutations",
            "kinds",
            "local_generators",
            "local_generator_squares",
            "frequencies",
            "eigenvectors",
            "eigenvalues",
            "operator_norms",
            "content_mask",
        ):
            tensor = getattr(metadata, name).clone()
            if tensor.is_floating_point():
                tensor = tensor.to(torch.get_default_dtype())
            self.register_buffer(name, tensor, persistent=True)

    def log_operator_norm_bound(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Additive log-norm bound for the ordered noncompact primitives."""

        if coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError("coordinate count does not match primitive bank")
        if self.algebra == "f4":
            return coordinates.new_zeros(coordinates.shape[:-1])
        weights = self.operator_norms.to(coordinates) * self.content_mask.to(
            coordinates
        )
        return (coordinates.abs() * weights).sum(dim=-1)

    def _metadata(self) -> PrimitiveMetadata:
        return PrimitiveMetadata(
            algebra=self.algebra,
            generators=self.generators,
            permutations=self.permutations,
            kinds=self.kinds,
            local_generators=self.local_generators,
            local_generator_squares=self.local_generator_squares,
            frequencies=self.frequencies,
            eigenvectors=self.eigenvectors,
            eigenvalues=self.eigenvalues,
            operator_norms=self.operator_norms,
            content_mask=self.content_mask,
        )

    def forward(self, values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        if values.ndim < 2 or values.shape[-1] != ALBERT_DIM:
            raise ValueError("values must end in (copies,27)")
        if coordinates.shape[:-1] != values.shape[:-2]:
            raise ValueError("coordinate leading dimensions must match values")
        if values.device != coordinates.device or values.dtype != coordinates.dtype:
            raise ValueError("values and coordinates must share device and dtype")
        use_cuda = self.backend == "cuda" or (
            self.backend == "auto"
            and sys.platform == "linux"
            and values.device.type == "cuda"
            and values.dtype == torch.float32
            and torch.cuda.get_device_capability(values.device) == (7, 5)
        )
        if use_cuda:
            if values.dtype != torch.float32 or coordinates.dtype != torch.float32:
                raise ValueError("primitive CUDA transport uses FP32 state and coordinates")
            if coordinates.shape[-1] != self.coordinate_dim:
                raise ValueError("coordinate count does not match primitive bank")
            tensors = (
                self.permutations,
                self.kinds,
                self.local_generators,
                self.local_generator_squares,
                self.frequencies,
                self.eigenvectors,
                self.eigenvalues,
            )
            if any(tensor.device != values.device for tensor in tensors):
                raise ValueError("move PrimitiveExceptionalAction to the input device")
            if any(
                tensor.dtype != torch.float32
                for tensor in tensors[2:]
            ):
                raise ValueError("cast PrimitiveExceptionalAction floating buffers to FP32")
            return _primitive_product_cuda_with_metadata(values, coordinates, tensors)
        metadata = self._metadata()
        if coordinates.shape[-1] != metadata.factor_count:
            raise ValueError("coordinate count does not match primitive bank")
        result = values
        for factor in range(metadata.factor_count):
            result = _apply_one_reference(
                result, coordinates[..., factor], factor, metadata
            )
        return result


__all__ = [
    "PrimitiveExceptionalAction",
    "PrimitiveMetadata",
    "build_primitive_metadata",
    "dense_primitive_product_oracle",
    "primitive_delta_recurrence_cuda",
    "primitive_delta_recurrence_reference",
    "primitive_event_layout",
    "primitive_product_cuda",
    "primitive_product_reference",
]
