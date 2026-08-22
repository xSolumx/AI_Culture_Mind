# Post-BIOS F14a validation

**Status:** machine-readiness gate passed on 2026-08-22.

This record closes the pre-BIOS handoff. It validates the machine and software
environment only; it does not add a model-quality or throughput claim.

## Firmware and host state

| gate | accepted observation |
|---|---|
| motherboard/BIOS | Gigabyte Z390 AORUS PRO WIFI-CF, BIOS F14a |
| firmware mode | UEFI |
| Secure Boot | enabled (`UEFISecureBootEnabled=1`) |
| hypervisor | Hyper-V present; WSL2 starts normally |
| host memory | 24 GiB across the same three 8 GiB DIMMs |
| memory rate | every DIMM reports configured and current capability at 3200 MT/s |

The first F14a boot was not accepted: Secure Boot was disabled and Windows
reported all three DIMMs at 3100 MT/s. The firmware settings were corrected and
the machine restarted before the successful observations above. The previous
HWiNFO scan independently recorded the pre-update XMP state as 1598.6 MHz
(DDR4-3200), 16-18-18-36, 2T.

## WSL and CUDA state

After an explicit `wsl --shutdown`, a cold start reported:

- WSL 2.7.12.0 with kernel 6.18.33.2-2;
- `user@1000.service` active/running, result success, exit status zero, and no
  failed system units;
- the unused Ubuntu desktop-installer Subiquity service still disabled;
- eight WSL CPUs, the 10 GiB memory limit, and 12 GiB swap;
- Torch 2.10.0+cu126, CUDA runtime/compiler 12.6, and compute capability 7.5;
- all seven Pure Spin raw-CUDA extension symbols and the official fused
  Mamba-2 path available;
- llama.cpp discovering the RTX 2070 SUPER through its CUDA 12.6 build.

`run_wsl_tests.sh` regenerated the environment artifact and passed all 24 tests
in 8.40 seconds. A separate FP16 matrix-load smoke test passed; while sampled,
the GPU reported P0, 54 degrees Celsius, and PCIe Gen3 x16. These readings show
functional load behavior, not a stable performance comparison.

## Claim boundary and completed benchmark gates

Historical cu126 and cu130 benchmark artifacts keep their original firmware,
environment, and timing provenance. F14a readiness does not retroactively alter
them.

The next gates were subsequently run under the accepted environment. Two
identical order-balanced steady-step repeats reversed the speed ordering, so
the exact repeat analysis promotes no throughput winner. Three matched
300-step Tiny Shakespeare seeds all favored fused Mamba-2, with mean validation
2.4942 versus 2.7477 bits/byte. Exact artifacts, source hashes, compatibility
checks, and the claim boundary are documented in
[`FRONTIER_TRAINING_RESULTS.md`](FRONTIER_TRAINING_RESULTS.md). The resulting
research frontier is controller/readout quality under the same matched budget,
not another isolated scan-kernel speed claim.
