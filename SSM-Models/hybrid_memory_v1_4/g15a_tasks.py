"""Deterministic G15A supplied-coordinate and no-symmetry task generators."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from spin8_triality import SPIN8_PAIRS

FILLER_TOKEN = 0
WRITE_TOKEN = 1
QUERY_TOKEN = 2
SYMMETRY_TARGET_START = 3
SYMMETRY_CLASSES = 10
NO_SYMMETRY_VALUE_START = 13
NO_SYMMETRY_CLASSES = 8
VOCAB_SIZE = NO_SYMMETRY_VALUE_START + NO_SYMMETRY_CLASSES
MINIMUM_LENGTH = 64
CENTER_STEPS = 26
CENTER_STEP_ANGLE = 2.0 * math.pi / CENTER_STEPS
OFF_TORUS_STEP_ANGLE = 0.20
OFF_TORUS_STEPS = 4
OFF_TORUS_PAIRS = (
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 5),
    (3, 6),
    (3, 7),
)


@dataclass(frozen=True)
class G15ABatch:
    """One final-only classification batch with external Spin coordinates."""

    token_ids: torch.Tensor
    coordinates: torch.Tensor
    targets: torch.Tensor
    labels: torch.Tensor
    task: str

    def __post_init__(self) -> None:
        if self.task not in ("symmetry", "no_symmetry"):
            raise ValueError("unknown G15A task")
        if self.token_ids.ndim != 2 or self.token_ids.dtype != torch.long:
            raise ValueError("token_ids must be int64 with shape (batch,length)")
        batch, length = self.token_ids.shape
        if batch < 1 or length < MINIMUM_LENGTH:
            raise ValueError("G15A batches require nonempty batch and length >= 64")
        if self.coordinates.shape[:2] != (batch, length):
            raise ValueError("coordinates must match token batch and length")
        if self.coordinates.ndim != 4 or self.coordinates.shape[-1] != 28:
            raise ValueError("coordinates must have shape (batch,length,heads,28)")
        if not self.coordinates.is_floating_point():
            raise TypeError("coordinates must use a floating-point dtype")
        if self.targets.shape != (batch,) or self.targets.dtype != torch.long:
            raise ValueError("targets must be int64 with shape (batch,)")
        if self.labels.shape != (batch,) or self.labels.dtype != torch.long:
            raise ValueError("labels must be int64 with shape (batch,)")
        if not bool(torch.isfinite(self.coordinates).all()):
            raise ValueError("coordinates must be finite")
        if float(self.coordinates.abs().max()) > 0.25:
            raise ValueError("coordinates exceed the G15 bounded chart")
        if not bool((self.token_ids[:, -1] == QUERY_TOKEN).all()):
            raise ValueError("the final scored token must be the query marker")

    @property
    def batch_size(self) -> int:
        return self.token_ids.shape[0]

    @property
    def length(self) -> int:
        return self.token_ids.shape[1]

    def to(self, device: torch.device | str) -> G15ABatch:
        return G15ABatch(
            token_ids=self.token_ids.to(device),
            coordinates=self.coordinates.to(device),
            targets=self.targets.to(device),
            labels=self.labels.to(device),
            task=self.task,
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.task.encode())
        for tensor in (
            self.token_ids,
            self.coordinates,
            self.targets,
            self.labels,
        ):
            contiguous = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(contiguous.shape)).encode())
            digest.update(contiguous.numpy().tobytes())
        return digest.hexdigest()


def _balanced_labels(batch_size: int, classes: int, *, seed: int) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    offset = int(torch.randint(classes, (), generator=generator))
    labels = (torch.arange(batch_size) + offset).remainder(classes)
    return labels[torch.randperm(batch_size, generator=generator)]


def generate_symmetry_batch(
    batch_size: int,
    length: int,
    *,
    seed: int,
    heads: int = 1,
    dtype: torch.dtype = torch.float32,
) -> G15ABatch:
    """Generate the ten-class off-torus plus central-sign tracking task.

    Classes 0--7 apply four bounded increments in eight different Spin(7)
    stabilizer planes that are absent from the fixed ``SO(2)^4`` torus.
    Class 8 is identity. Class 9 composes 26 bounded increments to the central
    ``2*pi`` element in plane (0,1). Tokens are identical across labels, so the
    supplied group coordinates are the only action-class signal.
    """

    if length < MINIMUM_LENGTH:
        raise ValueError("symmetry task requires length >= 64")
    if heads < 1:
        raise ValueError("heads must be positive")
    labels = _balanced_labels(batch_size, SYMMETRY_CLASSES, seed=seed)
    tokens = torch.full((batch_size, length), FILLER_TOKEN, dtype=torch.long)
    tokens[:, 0] = WRITE_TOKEN
    tokens[:, -1] = QUERY_TOKEN
    coordinates = torch.zeros(batch_size, length, heads, 28, dtype=dtype)
    for row, label in enumerate(labels.tolist()):
        if label < len(OFF_TORUS_PAIRS):
            index = SPIN8_PAIRS.index(OFF_TORUS_PAIRS[label])
            coordinates[row, 1 : 1 + OFF_TORUS_STEPS, :, index] = OFF_TORUS_STEP_ANGLE
        elif label == SYMMETRY_CLASSES - 1:
            index = SPIN8_PAIRS.index((0, 1))
            coordinates[row, 1 : 1 + CENTER_STEPS, :, index] = CENTER_STEP_ANGLE
    targets = SYMMETRY_TARGET_START + labels
    return G15ABatch(tokens, coordinates, targets, labels, "symmetry")


def generate_no_symmetry_batch(
    batch_size: int,
    length: int,
    *,
    seed: int,
    heads: int = 1,
    dtype: torch.dtype = torch.float32,
) -> G15ABatch:
    """Generate delayed one-value recall with identically zero transport."""

    if length < MINIMUM_LENGTH:
        raise ValueError("no-symmetry task requires length >= 64")
    if heads < 1:
        raise ValueError("heads must be positive")
    labels = _balanced_labels(batch_size, NO_SYMMETRY_CLASSES, seed=seed)
    values = NO_SYMMETRY_VALUE_START + labels
    tokens = torch.full((batch_size, length), FILLER_TOKEN, dtype=torch.long)
    tokens[:, 0] = values
    tokens[:, -1] = QUERY_TOKEN
    coordinates = torch.zeros(batch_size, length, heads, 28, dtype=dtype)
    return G15ABatch(tokens, coordinates, values.clone(), labels, "no_symmetry")


__all__ = [
    "CENTER_STEPS",
    "CENTER_STEP_ANGLE",
    "FILLER_TOKEN",
    "MINIMUM_LENGTH",
    "NO_SYMMETRY_CLASSES",
    "NO_SYMMETRY_VALUE_START",
    "OFF_TORUS_PAIRS",
    "OFF_TORUS_STEPS",
    "OFF_TORUS_STEP_ANGLE",
    "QUERY_TOKEN",
    "SYMMETRY_CLASSES",
    "SYMMETRY_TARGET_START",
    "VOCAB_SIZE",
    "WRITE_TOKEN",
    "G15ABatch",
    "generate_no_symmetry_batch",
    "generate_symmetry_batch",
]
