"""Profile one Pure Spin v1.2 training step on the active CUDA device."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from model import PureSpinSSMV12, PureSpinV12Config


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20_260_821)
    model = PureSpinSSMV12(
        PureSpinV12Config(group_schedule=(3, 4, 6, 8))
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    inputs = torch.randint(0, 256, (8, 256), device="cuda")
    targets = torch.randint(0, 256, (8, 256), device="cuda")

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, scan_mode="raw_cuda_factorized")["logits"]
        F.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
        optimizer.step()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profile:
        for _ in range(10):
            step()
    torch.cuda.synchronize()
    print(profile.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=30
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
