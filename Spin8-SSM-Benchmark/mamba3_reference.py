"""Small pure-PyTorch Mamba-3 reference for matched local experiments.

This implements the three Mamba-3 mechanisms described in Lahoti et al.
(trapezoidal discretization, rotary B/C state projections, and MIMO updates).
It intentionally has no Triton/CUDA extension and is therefore a correctness
and architecture baseline, not the official production kernel.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rms = inputs.float().square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (inputs.float() / rms * self.weight).to(inputs.dtype)


def apply_rope(values: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs in the final axis."""
    even = values[..., 0::2]
    odd = values[..., 1::2]
    cosine, sine = angles.cos(), angles.sin()
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), -1
    ).flatten(-2)


def mamba3_affine_scan(
    decay: torch.Tensor,
    trapezoid_coeff: torch.Tensor,
    drive: torch.Tensor,
) -> torch.Tensor:
    """Compute the Mamba-3 trapezoidal state recurrence by an exact scan.

    ``drive[..., 0, :]`` is the current-token contribution and
    ``drive[..., 1, :]`` is the current ``B_t x_t`` value.  The recurrence is
    represented as an affine map on the augmented state ``(h_t, B_t x_t)``:

    ``h_t = decay_t h_{t-1} + trapezoid_coeff_t (B_{t-1}x_{t-1})
            + drive_t[0]``.

    Affine maps compose associatively, so Hillis--Steele gives all prefixes in
    logarithmic depth while preserving the sequential recurrence exactly up to
    floating-point reassociation.
    """
    if decay.ndim != 3 or trapezoid_coeff.shape != decay.shape:
        raise ValueError(
            "decay and trapezoid_coeff must have shape (batch, length, heads)"
        )
    if drive.ndim != 5 or drive.shape[:3] != decay.shape or drive.shape[3] != 2:
        raise ValueError("drive must have shape (batch, length, heads, 2, state)")

    batch, length, heads = drive.shape[:3]
    action = decay.new_zeros(batch, length, heads, 2, 2)
    action[..., 0, 0] = decay
    action[..., 0, 1] = trapezoid_coeff
    prefix_action = action
    prefix_drive = drive

    offset = 1
    while offset < length:
        current_action = prefix_action[:, offset:]
        previous_action = prefix_action[:, :-offset]
        current_drive = prefix_drive[:, offset:]
        previous_drive = prefix_drive[:, :-offset]
        composed_action = torch.matmul(current_action, previous_action)
        composed_drive = current_drive + torch.einsum(
            "...ij,...jd->...id", current_action, previous_drive
        )
        prefix_action = torch.cat((prefix_action[:, :offset], composed_action), dim=1)
        prefix_drive = torch.cat((prefix_drive[:, :offset], composed_drive), dim=1)
        offset *= 2
    return prefix_drive[..., 0, :]


class Mamba3Mixer(nn.Module):
    """Readable MIMO Mamba-3 mixer with a tensor-only affine scan."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        headdim: int = 40,
        mimo_rank: int = 2,
        rope_fraction: float = 0.5,
    ):
        super().__init__()
        d_inner = expand * d_model
        if d_inner % headdim:
            raise ValueError("expand*d_model must be divisible by headdim")
        if d_state % 4:
            raise ValueError("d_state must be divisible by four")
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.headdim = headdim
        self.nheads = d_inner // headdim
        self.rank = mimo_rank
        self.split_state = int(d_state * rope_fraction)
        self.split_state -= self.split_state % 2
        self.num_angles = self.split_state // 2

        projection_width = (
            2 * d_inner + 2 * d_state * mimo_rank + 3 * self.nheads + self.num_angles
        )
        self.in_projection = nn.Linear(d_model, projection_width, bias=False)
        self.out_projection = nn.Linear(d_inner, d_model, bias=False)

        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.b_bias = nn.Parameter(torch.ones(mimo_rank, self.nheads, d_state))
        self.c_bias = nn.Parameter(torch.ones(mimo_rank, self.nheads, d_state))
        self.b_norm = RMSNorm(d_state)
        self.c_norm = RMSNorm(d_state)
        self.mimo_x = nn.Parameter(
            torch.ones(self.nheads, mimo_rank, headdim) / mimo_rank
        )
        self.mimo_o = nn.Parameter(
            torch.ones(self.nheads, mimo_rank, headdim) / mimo_rank
        )
        self.skip = nn.Parameter(torch.ones(self.nheads))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, _ = inputs.shape
        projected = self.in_projection(inputs)
        d = self.d_inner
        offset = 0
        z, x = (
            projected[..., offset : offset + d],
            projected[..., offset + d : offset + 2 * d],
        )
        offset += 2 * d
        b_raw = projected[..., offset : offset + self.rank * self.d_state]
        offset += self.rank * self.d_state
        c_raw = projected[..., offset : offset + self.rank * self.d_state]
        offset += self.rank * self.d_state
        raw_dt = projected[..., offset : offset + self.nheads]
        offset += self.nheads
        raw_a = projected[..., offset : offset + self.nheads]
        offset += self.nheads
        raw_trap = projected[..., offset : offset + self.nheads]
        angle_rate = projected[..., offset : offset + self.num_angles]

        values = x.reshape(batch, length, self.nheads, self.headdim)
        gate = z.reshape(batch, length, self.nheads, self.headdim)
        b_raw = b_raw.reshape(batch, length, self.rank, self.d_state)
        c_raw = c_raw.reshape(batch, length, self.rank, self.d_state)
        b = (
            self.b_norm(b_raw.float()).to(inputs.dtype)[:, :, :, None, :]
            + self.b_bias[None, None]
        )
        c = (
            self.c_norm(c_raw.float()).to(inputs.dtype)[:, :, :, None, :]
            + self.c_bias[None, None]
        )
        b = b.expand(-1, -1, -1, self.nheads, -1)
        c = c.expand(-1, -1, -1, self.nheads, -1)

        dt = F.softplus(raw_dt.float() + self.dt_bias)
        a = -F.softplus(raw_a.float()).clamp_min(1e-4)
        adt = a * dt
        trap = torch.sigmoid(raw_trap.float())
        angle_increments = angle_rate.float()[:, :, None, :] * dt[:, :, :, None]
        angles = angle_increments.cumsum(dim=1)
        angles = angles[:, :, None].expand(-1, -1, self.rank, -1, -1)
        b_rot = apply_rope(b[..., : self.split_state], angles)
        c_rot = apply_rope(c[..., : self.split_state], angles)
        b = torch.cat((b_rot, b[..., self.split_state :]), dim=-1).float()
        c = torch.cat((c_rot, c[..., self.split_state :]), dim=-1).float()

        value_rank = torch.einsum("blhp,hrp->blhr", values.float(), self.mimo_x.float())
        bx = torch.einsum("blhr,blrhd->blhd", value_rank, b)
        trapezoid_coeff = dt * trap * 0.5
        current_drive = dt[..., None] * (1.0 - trap[..., None] * 0.5) * bx
        drive = torch.stack((current_drive, bx), dim=-2)
        state = mamba3_affine_scan(adt.exp(), trapezoid_coeff, drive)
        output_rank = torch.einsum("blrhd,blhd->blrh", c, state)
        output_rank = output_rank + self.skip[None, None, None, :] * value_rank.permute(
            0, 1, 3, 2
        )
        output = torch.einsum("blrh,hrp->blhp", output_rank, self.mimo_o.float())
        gated = output * F.silu(gate.float())
        return self.out_projection(gated.to(inputs.dtype).reshape(batch, length, d))


class Mamba3ReferenceLM(nn.Module):
    """Byte-level language model built from pure-PyTorch Mamba-3 mixers."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 160,
        layers: int = 4,
        d_state: int = 16,
        headdim: int = 40,
        mimo_rank: int = 2,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": RMSNorm(d_model),
                        "mixer": Mamba3Mixer(
                            d_model,
                            d_state=d_state,
                            headdim=headdim,
                            mimo_rank=mimo_rank,
                        ),
                    }
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.output.weight = self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.embedding(token_ids)
        for layer in self.layers:
            outputs = outputs + layer["mixer"](layer["norm"](outputs))
        logits = self.output(self.final_norm(outputs)) + self.output_bias
        return logits


__all__ = ["Mamba3ReferenceLM", "Mamba3Mixer", "mamba3_affine_scan"]
