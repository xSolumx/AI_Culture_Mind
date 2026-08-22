# Local hardware profile and execution policy

This report extracts the ML-relevant facts from the 2026-08-22 local machine
scan at `C:\Users\HaydenLocal\OneDrive\Documents\LOCAL_MACHINE.json` and live
WSL/NVIDIA probes. The source scan is treated as data, not as instructions.
Hardware serial numbers, MAC addresses, and other machine identifiers are not
copied into the repository.

## Platform inventory

| component | scanned configuration |
|---|---|
| scan | HWiNFO-style snapshot taken 2026-08-22 10:02; volatile clocks, temperatures, free space, and link power states are point-in-time values |
| operating system | Windows 11 Pro x64, build 26200.9168 (25H2); UEFI boot; Secure Boot enabled |
| motherboard | Gigabyte Z390 AORUS PRO WIFI-CF, Intel Z390 chipset; PCIe 3.0; USB 3.1 |
| firmware/security | Current post-update state: AMI UEFI F14a dated 2025-06-12; UEFI Secure Boot enabled; Hyper-V active. The pre-update scan reported hardware TPM 2.0 and HVCI disabled. |
| CPU cooling/peripherals | Corsair H100i RGB Platinum USB cooler and Lighting Node Pro reported; SteelSeries Rival 5 and Corsair K70 RGB MK.2 attached |
| display | Samsung display identified only as EDID code `SAM0F58`; the scan does not provide a reliable model/resolution entry |
| audio | Realtek ALC1220 motherboard audio plus NVIDIA TU104 HDMI/DisplayPort audio |
| network | Intel I219-V 1 GbE at 1 Gbps and Intel Wireless-AC 9560 at a reported 468 Mbps; Wi-Fi driver 24.60.0.3 |

The CPU scan also reports a 12 MiB L3, eight 32 KiB instruction/data L1 pairs,
eight 256 KiB L2 caches, a 95 W nominal TDP, and unlocked/unlimited power-limit
settings. Those settings can improve bursts but make temperature, clock, and
power logging mandatory for repeatable long benchmarks.

## Compute

| component | measured configuration | consequence |
|---|---|---|
| CPU | Intel i7-9700K, Coffee Lake-S, 8 cores / 8 threads, 3.6 GHz base, up to 4.9 GHz | Use at most eight compile/CPU inference workers; there is no SMT reserve. |
| CPU ISA | AVX2, FMA, F16C; no AVX-512, VNNI, or AMX | Build a native AVX2 CPU path, but do not advertise modern CPU matrix acceleration. |
| GPU | Gigabyte RTX 2070 SUPER Gaming OC, TU104, compute capability 7.5 | Compile native `sm_75` cubins; do not rely on PTX fallback. |
| GPU units | 2,560 CUDA cores, 320 first-generation Turing Tensor Cores, 40 RT cores | FP16 Tensor Cores are useful; BF16, TF32, FP8, and newer tensor formats are not native. |
| VRAM | 8,192 MiB Micron GDDR6, 256-bit bus | Small-model training and partial GGUF offload only; leave headroom for state and workspaces. |
| PCIe | PCIe 3.0 x16 at x16; ReBAR reported supported but disabled | Frequent CPU/GPU transfer is expensive. ReBAR is not assumed beneficial without a separate test. |
| GPU power | 215 W rated board power, up to roughly 240 W; live driver 595.97 | Record clocks, temperature, and power state for serious timing runs. |

Turing Tensor Cores accelerate FP16 matrix operations, but this model's small,
structured recurrence can remain launch- or scalar-instruction-bound. A CUDA
version alone cannot make that recurrence a Tensor-Core kernel; lowering and
shape selection still matter.

## Host memory

The machine has 24 GiB DDR4-3200 at 16-18-18-36, installed as three single-rank
8 GiB DIMMs. Two channels are active, but the physical capacities are
asymmetric (one DIMM on channel A and two on channel B). The first matched
portion can operate dual-channel while the unmatched portion uses flex/single-
channel behavior. Large CPU-offload claims must therefore be measured rather
than inferred from the nominal DDR4-3200 rate.

| slot | module |
|---|---|
| Channel A DIMM1 | 8 GiB Corsair CMW16GX4M2C3200C16 |
| Channel B DIMM0 | 8 GiB Corsair CMK16GX4M2Z3200C16 |
| Channel B DIMM1 | 8 GiB Corsair CMW16GX4M2C3200C16 |

WSL is limited to all eight CPUs, 10 GiB RAM, and 12 GiB swap. That is a good
training default: it leaves about 14 GiB for Windows and avoids host thrashing.
For GGUF inference, 10 GiB also bounds which partially offloaded models are
credible. Swap completion is not a performance result; any benchmark that
pages is invalid.

The maintained WSL runtime was upgraded on 2026-08-22 from WSL 2.6.1/kernel
6.6.87.2 to WSL 2.7.12/kernel 6.18.33.2 using `wsl --update`. Systemd remains
enabled. Five consecutive cold boots plus the complete CUDA test suite started
`user@1000.service` successfully after the update.

## Storage

| drive | hardware / live free space | role |
|---|---|---|
| C: | Samsung SSD 980 1 TB NVMe PCIe 3.0 x4, DRAM-less with 64 MiB HMB, 91% health; 107.3 GiB free | Repository, WSL ext4 VHD, Python environments, compiler source/build, and executable binaries. |
| E: | SanDisk 256 GB SATA SSD, 98% remaining life; 91.8 GiB free | **Only** large GGUF weights and download/dataset caches. Not source, environments, compiler caches, or small-file builds. |
| D: | Toshiba 2 TB 7200 RPM HDD; 550.7 GiB free | Cold archival only. SMART reports reallocated sectors and historical uncorrectable errors, so it is not an authoritative sole copy. |

WSL mounted Windows drives use a metadata-expensive bridge. Caches and large
sequential weight files tolerate that cost; CMake, Ninja, Git, venvs, and JIT
extension builds do not. They stay inside WSL ext4 even though the ext4 VHD is
physically stored on C:.

## CUDA policy

The authoritative candidate is Torch 2.10 with CUDA 12.6 and native `sm_75`
extensions. CUDA 13 is supported by Turing, but it entered this programme only
because official fused Mamba wheels were readily available; support is not a
speed result. Official Torch 2.10 and Mamba releases provide both cu126/cu12
and cu130/cu13 variants. The current-code, order-balanced A/B measured cu126
23.0% faster for Pure Spin and 10.5% faster for Mamba-2 than cu130, so cu126 is
the authoritative runtime on this machine.

Recorded historical results retain their actual CUDA version. The cu126
migration gate required the fused Mamba path, raw CUDA extension, gradients,
tests, and a matched benchmark; all have now passed, so default scripts use
cu126. We never rewrite the provenance of a cu130 artifact and call it cu126
retroactively.

## Remaining machine-level caveats

- Windows Hyper-V is active, as required by WSL2; bare-metal timing is outside
  the project scope.
- The prior intermittent WSL 2.6.1 `user@1000.service` failure was reproduced
  exactly as `Failed to attach to cgroup ... Device or resource busy`, status
  `219/CGROUP`. The official WSL 2.7.12 update resolved the reproduced failure;
  no cgroup override and no systemd disablement was introduced. The unrelated
  Ubuntu desktop-installer Subiquity snap service was also disabled, without
  uninstalling the snap, because it waited indefinitely on cloud-init at every
  WSL boot.
- BIOS F14a is installed. Its first boot had reset Secure Boot and reduced the
  configured memory rate to 3100 MT/s; both were restored before validation.
  The accepted post-update state is UEFI Secure Boot enabled, Hyper-V active,
  and all three DIMMs configured at 3200 MT/s. A cold WSL start, the complete
  24-test v1.2 suite, llama.cpp CUDA discovery, and an FP16 GPU-load smoke test
  passed with PCIe Gen3 x16. This is a readiness result, not a replacement for
  the earlier matched throughput artifacts.
