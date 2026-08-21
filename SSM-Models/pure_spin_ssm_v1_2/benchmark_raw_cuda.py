"""Compare raw CUDA, Triton, and PyTorch Spin(8) materialized recurrences."""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from pure_spin8_ssm.continuous_scan import continuous_spin8_scan
from raw_cuda import raw_cuda_spin8_scan


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed(fn, repeats):
    for _ in range(20):
        output = fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return output, statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/raw_cuda_scan.json")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(19)
    b, length, channels, reps = args.batch, args.length, args.channels, 3
    matrices = torch.randn(b, length, reps, 8, 8, device="cuda", dtype=torch.float32)
    action = torch.linalg.qr(matrices).Q.to(dtype).contiguous()
    scale = torch.sigmoid(torch.randn(b, length, channels, device="cuda", dtype=dtype))
    drive = torch.randn(b, length, channels, reps, 8, device="cuda", dtype=dtype) * 0.01
    initial = torch.randn(b, channels, reps, 8, device="cuda", dtype=dtype) * 0.01
    with torch.inference_mode():
        reference, eager_us = timed(
            lambda: continuous_spin8_scan(
                action, scale, drive, initial, backend="eager"
            ),
            max(3, args.repeats // 10),
        )
        triton, triton_us = timed(
            lambda: continuous_spin8_scan(
                action, scale, drive, initial, backend="triton_scalar"
            ),
            args.repeats,
        )
        raw, raw_us = timed(
            lambda: raw_cuda_spin8_scan(action, scale, drive, initial),
            args.repeats,
        )
    atol = 3e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(triton, reference, atol=atol, rtol=atol)
    torch.testing.assert_close(raw, reference, atol=atol, rtol=atol)
    report = {
        "schema_version": 1,
        "claim_scope": "forward materialized-action recurrence only; not end-to-end training",
        "shape": [b, length, channels, reps, 8],
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timing": {
            "method": "CUDA events on the current stream",
            "warmup_repetitions": 20,
            "timed_repetitions": args.repeats,
        },
        "implementation_sha256": {
            path.relative_to(ROOT).as_posix(): file_sha256(path)
            for path in (
                Path(__file__),
                ROOT / "raw_cuda.py",
                ROOT / "csrc" / "spin_scan.cpp",
                ROOT / "csrc" / "spin_scan_cuda.cu",
            )
        },
        "median_microseconds": {
            "pytorch": eager_us,
            "triton": triton_us,
            "raw_cuda": raw_us,
        },
        "max_abs_error": {
            "triton": float((triton - reference).abs().max()),
            "raw_cuda": float((raw - reference).abs().max()),
        },
        "raw_cuda_has_backward": False,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
