"""Geometry-aware optimizer components for Hybrid Memory v1.4 research.

The maintained model does not silently replace AdamW.  This module exposes an
explicit experimental composite:

* Muon for eligible two-dimensional hidden-layer matrices;
* scalar-second-moment AdamW for small memory-control tensors; and
* ordinary AdamW for embeddings, norms, biases, and convolutional tensors.

The scalar second moment is invariant under an orthogonal change of coordinates
within one parameter tensor.  That property directly targets the optimizer
failure isolated by the repository's Spin(8)/SO(8) chart audit.  It is not a
claim that the composite improves language modelling; frozen experiments must
establish that separately.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class OptimizerGroupReport:
    """Auditable description of one disjoint optimizer parameter group."""

    optimizer: str
    role: str
    names: tuple[str, ...]
    tensors: int
    parameters: int
    learning_rate: float
    weight_decay: float
    second_moment: str | None


@dataclass(frozen=True)
class OptimizerPartition:
    """Disjoint parameter lists and their stable names."""

    muon: tuple[tuple[str, nn.Parameter], ...]
    scalar_adamw: tuple[tuple[str, nn.Parameter], ...]
    adamw_decay: tuple[tuple[str, nn.Parameter], ...]
    adamw_no_decay: tuple[tuple[str, nn.Parameter], ...]

    @property
    def named_groups(self) -> tuple[tuple[tuple[str, nn.Parameter], ...], ...]:
        return (
            self.muon,
            self.scalar_adamw,
            self.adamw_decay,
            self.adamw_no_decay,
        )


_CONTROL_MARKERS = (
    "coordinate_projection",
    "erase_projection",
    "write_projection",
    "decay_projection",
    "residual_scale",
)
_NO_DECAY_MARKERS = (
    "embedding",
    "embed_tokens",
    "word_embeddings",
    "lm_head",
    "norm",
)


def partition_optimizer_parameters(model: nn.Module) -> OptimizerPartition:
    """Partition every trainable parameter exactly once by semantic role.

    Muon is deliberately limited to hidden-layer matrices.  Memory write and
    decay controllers retain their tensor identity and receive one scalar
    second moment per tensor.  Embeddings, normalizers, biases, and scalars use
    ordinary coordinatewise AdamW without decay; remaining non-matrix tensors
    (principally depthwise convolution kernels) use decoupled weight decay.
    """

    groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        "muon": [],
        "scalar_adamw": [],
        "adamw_decay": [],
        "adamw_no_decay": [],
    }
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity in seen:
            raise ValueError(f"trainable parameter {name!r} is exposed more than once")
        seen.add(identity)
        lowered = name.lower()
        if any(marker in lowered for marker in _CONTROL_MARKERS):
            destination = "scalar_adamw"
        elif (
            parameter.ndim == 2
            and min(parameter.shape) >= 2
            and not any(marker in lowered for marker in _NO_DECAY_MARKERS)
        ):
            destination = "muon"
        elif (
            parameter.ndim < 2
            or lowered.endswith(".bias")
            or any(marker in lowered for marker in _NO_DECAY_MARKERS)
        ):
            destination = "adamw_no_decay"
        else:
            destination = "adamw_decay"
        groups[destination].append((name, parameter))

    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if seen != expected:
        raise RuntimeError(
            "optimizer partition does not cover every trainable parameter"
        )
    return OptimizerPartition(
        muon=tuple(groups["muon"]),
        scalar_adamw=tuple(groups["scalar_adamw"]),
        adamw_decay=tuple(groups["adamw_decay"]),
        adamw_no_decay=tuple(groups["adamw_no_decay"]),
    )


class ScalarSecondMomentAdamW(torch.optim.Optimizer):
    """AdamW with one rotation-invariant second moment per parameter tensor.

    The first moment has the same shape as the gradient and therefore transforms
    covariantly.  The second moment is the mean squared gradient, a scalar that
    is unchanged by an orthogonal coordinate transform.  Decoupled weight decay
    also preserves that covariance.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"invalid learning rate: {lr}")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid beta2: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"invalid weight decay: {weight_decay}")
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError(
                        "ScalarSecondMomentAdamW does not support sparse gradients"
                    )
                if torch.is_complex(parameter) or torch.is_complex(gradient):
                    raise RuntimeError(
                        "ScalarSecondMomentAdamW does not support complex parameters"
                    )
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros(
                        (), device=parameter.device, dtype=parameter.dtype
                    )
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                mean_square = gradient.square().mean()
                exp_avg_sq.mul_(beta2).add_(mean_square, alpha=1.0 - beta2)

                learning_rate = group["lr"]
                weight_decay = group["weight_decay"]
                if weight_decay:
                    parameter.mul_(1.0 - learning_rate * weight_decay)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denominator.add_(group["eps"])
                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-learning_rate / bias_correction1,
                )
        return loss


class BlockScalarSecondMomentAdamW(ScalarSecondMomentAdamW):
    """AdamW with one scalar second moment per final-axis vector block.

    For a token-coordinate table shaped ``(tokens, coordinates)``, the first
    moment remains coordinatewise while the second moment has shape
    ``(tokens, 1)``.  This preserves covariance under one shared orthogonal
    coordinate change and equivariance under token-row permutations without
    forcing unrelated sparse token rows to share an adaptive scale.
    """

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError(
                        "BlockScalarSecondMomentAdamW does not support sparse gradients"
                    )
                if torch.is_complex(parameter) or torch.is_complex(gradient):
                    raise RuntimeError(
                        "BlockScalarSecondMomentAdamW does not support complex parameters"
                    )
                state = self.state[parameter]
                second_moment_shape = (
                    () if parameter.ndim == 0 else (*parameter.shape[:-1], 1)
                )
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros(
                        second_moment_shape,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                mean_square = (
                    gradient.square()
                    if gradient.ndim == 0
                    else gradient.square().mean(dim=-1, keepdim=True)
                )
                exp_avg_sq.mul_(beta2).add_(mean_square, alpha=1.0 - beta2)

                learning_rate = group["lr"]
                weight_decay = group["weight_decay"]
                if weight_decay:
                    parameter.mul_(1.0 - learning_rate * weight_decay)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denominator.add_(group["eps"])
                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-learning_rate / bias_correction1,
                )
        return loss


class HarmonicMuonAdamW:
    """A checkpointable composite optimizer with an explicit group contract."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
    ) -> None:
        muon_class = getattr(torch.optim, "Muon", None)
        if muon_class is None:
            raise RuntimeError("HarmonicMuonAdamW requires torch.optim.Muon")
        self.partition = partition_optimizer_parameters(model)
        if not self.partition.muon:
            raise ValueError("HarmonicMuonAdamW found no eligible hidden matrices")

        self.muon = muon_class(
            [parameter for _, parameter in self.partition.muon],
            lr=lr,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            nesterov=True,
            ns_steps=muon_ns_steps,
            adjust_lr_fn="match_rms_adamw",
        )
        self.scalar_adamw = (
            ScalarSecondMomentAdamW(
                [parameter for _, parameter in self.partition.scalar_adamw],
                lr=lr,
                betas=betas,
                weight_decay=0.0,
            )
            if self.partition.scalar_adamw
            else None
        )
        adamw_groups = []
        if self.partition.adamw_decay:
            adamw_groups.append(
                {
                    "params": [
                        parameter for _, parameter in self.partition.adamw_decay
                    ],
                    "weight_decay": weight_decay,
                    "role": "non_matrix_decay",
                }
            )
        if self.partition.adamw_no_decay:
            adamw_groups.append(
                {
                    "params": [
                        parameter for _, parameter in self.partition.adamw_no_decay
                    ],
                    "weight_decay": 0.0,
                    "role": "embedding_norm_bias",
                }
            )
        self.adamw = (
            torch.optim.AdamW(adamw_groups, lr=lr, betas=betas)
            if adamw_groups
            else None
        )
        self._optimizers = tuple(
            optimizer
            for optimizer in (self.muon, self.scalar_adamw, self.adamw)
            if optimizer is not None
        )
        self._hyperparameters = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "muon_momentum": muon_momentum,
            "muon_ns_steps": muon_ns_steps,
            "muon_adjust_lr_fn": "match_rms_adamw",
        }

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [
            group for optimizer in self._optimizers for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self._optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self._optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "optimizer": type(self).__name__,
            "hyperparameters": self._hyperparameters,
            "partition": self.partition_report(),
            "muon": self.muon.state_dict(),
            "scalar_adamw": (
                None if self.scalar_adamw is None else self.scalar_adamw.state_dict()
            ),
            "adamw": None if self.adamw is None else self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("optimizer") != type(self).__name__:
            raise ValueError("optimizer checkpoint type mismatch")
        self.muon.load_state_dict(state_dict["muon"])
        scalar_state = state_dict.get("scalar_adamw")
        if (self.scalar_adamw is None) != (scalar_state is None):
            raise ValueError("scalar optimizer checkpoint partition mismatch")
        if self.scalar_adamw is not None:
            self.scalar_adamw.load_state_dict(scalar_state)
        adamw_state = state_dict.get("adamw")
        if (self.adamw is None) != (adamw_state is None):
            raise ValueError("AdamW checkpoint partition mismatch")
        if self.adamw is not None:
            self.adamw.load_state_dict(adamw_state)

    def partition_report(self) -> list[dict[str, Any]]:
        lr = self._hyperparameters["lr"]
        weight_decay = self._hyperparameters["weight_decay"]
        rows = (
            OptimizerGroupReport(
                optimizer="torch.optim.Muon",
                role="hidden_matrices",
                names=tuple(name for name, _ in self.partition.muon),
                tensors=len(self.partition.muon),
                parameters=sum(p.numel() for _, p in self.partition.muon),
                learning_rate=lr,
                weight_decay=weight_decay,
                second_moment=None,
            ),
            OptimizerGroupReport(
                optimizer="ScalarSecondMomentAdamW",
                role="memory_controls",
                names=tuple(name for name, _ in self.partition.scalar_adamw),
                tensors=len(self.partition.scalar_adamw),
                parameters=sum(p.numel() for _, p in self.partition.scalar_adamw),
                learning_rate=lr,
                weight_decay=0.0,
                second_moment="one scalar per parameter tensor",
            ),
            OptimizerGroupReport(
                optimizer="torch.optim.AdamW",
                role="non_matrix_decay",
                names=tuple(name for name, _ in self.partition.adamw_decay),
                tensors=len(self.partition.adamw_decay),
                parameters=sum(p.numel() for _, p in self.partition.adamw_decay),
                learning_rate=lr,
                weight_decay=weight_decay,
                second_moment="coordinatewise",
            ),
            OptimizerGroupReport(
                optimizer="torch.optim.AdamW",
                role="embedding_norm_bias",
                names=tuple(name for name, _ in self.partition.adamw_no_decay),
                tensors=len(self.partition.adamw_no_decay),
                parameters=sum(p.numel() for _, p in self.partition.adamw_no_decay),
                learning_rate=lr,
                weight_decay=0.0,
                second_moment="coordinatewise",
            ),
        )
        return [asdict(row) for row in rows if row.tensors]


def build_optimizer(
    model: nn.Module,
    name: str,
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer | HarmonicMuonAdamW:
    """Build an explicitly named optimizer for matched experiments."""

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "harmonic_muon_adamw":
        return HarmonicMuonAdamW(model, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


__all__ = [
    "BlockScalarSecondMomentAdamW",
    "HarmonicMuonAdamW",
    "OptimizerGroupReport",
    "OptimizerPartition",
    "ScalarSecondMomentAdamW",
    "build_optimizer",
    "partition_optimizer_parameters",
]
