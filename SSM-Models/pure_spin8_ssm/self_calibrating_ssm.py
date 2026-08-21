"""Self-calibrating continuous Spin(8) recurrence.

The layer consumes seven ordered vector-probe images and one lift-odd sign per
time step.  It reconstructs an oriented vector action, lifts it to the full
triality tuple, shares that physical action across all isotypic memory copies,
and executes the schedule selected by compiler v2.1.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from spin8_triality import (
    SPIN8_DIM,
    TRIALITY_REPRESENTATIONS,
    torch_triality_generators,
)
from torch import nn
from torch.nn import functional as F

from pure_spin8_ssm.compiler import (
    HardwareTarget,
    IsotypicExecutionPlan,
    RuntimeShape,
    compile_isotypic_plan,
    spin8_triality_blocks,
)
from pure_spin8_ssm.continuous_scan import (
    ContinuousBackend,
    continuous_spin8_scan,
)
from pure_spin8_ssm.self_calibration import spin8_actions_from_seven_probes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARDWARE_PROFILE = (
    ROOT
    / "experiments"
    / "artifacts"
    / "spin8_compiler_v211_rtx2070s_20260821.json"
)
ProjectionMode = Literal["none", "qr", "polar"]


class SelfCalibratingSpin8SSMLayer(nn.Module):
    """Continuous shared-action memory compiled from a seven-probe interface."""

    def __init__(
        self,
        *,
        channels: int = 1,
        projection: ProjectionMode = "qr",
        hardware_profile: str | Path | None = DEFAULT_HARDWARE_PROFILE,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if projection not in ("none", "qr", "polar"):
            raise ValueError("unknown frame projection")
        self.channels = channels
        self.representations = TRIALITY_REPRESENTATIONS
        self.projection = projection
        self.hardware_profile = (
            None if hardware_profile is None else str(Path(hardware_profile))
        )
        initial = torch.randn(channels, len(self.representations), SPIN8_DIM)
        self.initial_state = nn.Parameter(F.normalize(initial, dim=-1))
        self.register_buffer(
            "generators",
            torch_triality_generators(self.representations),
            persistent=True,
        )

    @property
    def cache_scalars(self) -> int:
        return self.channels * len(self.representations) * SPIN8_DIM

    def initial_cache(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return self.initial_state.to(reference).unsqueeze(0).expand(batch_size, -1, -1, -1)

    def compile_plan(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
        training: bool,
    ) -> IsotypicExecutionPlan:
        dtype_name = str(dtype).removeprefix("torch.")
        profile = (
            self.hardware_profile
            if self.hardware_profile is not None
            and Path(self.hardware_profile).is_file()
            else None
        )
        return compile_isotypic_plan(
            spin8_triality_blocks(self.channels, shared_action=True),
            RuntimeShape(
                batch_size=batch_size,
                sequence_length=sequence_length,
                dtype=dtype_name,
                device=device.type,
                training=training,
            ),
            HardwareTarget.current(device),
            hardware_profile_path=profile,
        )

    def _plan_backend(self, plan: IsotypicExecutionPlan) -> ContinuousBackend:
        backends = {schedule.backend for schedule in plan.schedules}
        if len(backends) != 1:
            raise RuntimeError("triality blocks require one common recurrence backend")
        backend = backends.pop()
        return backend

    def forward(
        self,
        probes: torch.Tensor,
        lift_sign: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        scale: torch.Tensor | None = None,
        drive: torch.Tensor | None = None,
        backend: ContinuousBackend = "auto",
        execution_dtype: torch.dtype | None = None,
        return_plan: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, IsotypicExecutionPlan
    ]:
        if probes.ndim != 4 or probes.shape[-2:] != (7, 8):
            raise ValueError("probes must have shape (batch,length,7,8)")
        batch, length = probes.shape[:2]
        if lift_sign.shape != (batch, length):
            raise ValueError("lift_sign must have shape (batch,length)")
        generators = self.generators.to(probes)
        project: bool | Literal["qr", "polar"] = (
            False if self.projection == "none" else self.projection
        )
        actions, _ = spin8_actions_from_seven_probes(
            probes,
            generators,
            self.representations,
            lift_sign=lift_sign,
            project=project,
        )
        if execution_dtype is None:
            execution_dtype = probes.dtype
        actions = actions.to(dtype=execution_dtype)
        if scale is None:
            scale = torch.ones(
                batch,
                length,
                self.channels,
                dtype=execution_dtype,
                device=probes.device,
            )
        else:
            scale = scale.to(dtype=execution_dtype)
        if drive is None:
            drive = torch.zeros(
                batch,
                length,
                self.channels,
                len(self.representations),
                SPIN8_DIM,
                dtype=execution_dtype,
                device=probes.device,
            )
        else:
            drive = drive.to(dtype=execution_dtype)
        if state is None:
            state = self.initial_cache(batch, actions)
        else:
            state = state.to(dtype=execution_dtype)
        plan = self.compile_plan(
            batch_size=batch,
            sequence_length=length,
            dtype=execution_dtype,
            device=probes.device,
            training=(
                torch.is_grad_enabled()
                and any(
                    tensor.requires_grad
                    for tensor in (probes, scale, drive, state)
                )
            ),
        )
        selected_backend = self._plan_backend(plan) if backend == "auto" else backend
        outputs = continuous_spin8_scan(
            actions,
            scale,
            drive,
            state,
            backend=selected_backend,
        )
        result = (outputs, outputs[:, -1])
        return (*result, plan) if return_plan else result


__all__ = ["SelfCalibratingSpin8SSMLayer"]
