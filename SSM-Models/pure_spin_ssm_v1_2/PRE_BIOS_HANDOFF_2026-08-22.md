# Pre-BIOS environment handoff

**Frozen:** 2026-08-22, before the planned Gigabyte Z390 AORUS PRO WIFI
F12-to-F14a firmware update.

This is a machine-readiness record, not a model result. No new architecture or
benchmark claim was introduced in this preparation pass.

## WSL user-session repair

The earlier warning was real but intermittent. WSL 2.6.1/kernel 6.6.87.2 first
started `user@1000.service` successfully, then reproduced the failure on a
normal later invocation:

```text
Failed to attach to cgroup /user.slice/user-1000.slice/user@1000.service:
Device or resource busy
Failed at step CGROUP spawning /lib/systemd/systemd
status=219/CGROUP
```

The repair deliberately preserved systemd and avoided cgroup overrides:

1. `wsl --update` upgraded the host component to WSL 2.7.12.0 and the kernel
   to 6.18.33.2-2.
2. The unrelated `ubuntu-desktop-installer.subiquity-server` snap service was
   stopped and disabled. It had started on every WSL boot, consumed about
   142 MiB, and waited indefinitely for cloud-init. The snap was not removed.
3. Five consecutive `wsl --shutdown` cold-start cycles returned
   `Result=success`, `ExecMainStatus=0`, `ActiveState=active`, and
   `SubState=running`. The fifth-boot journal contained no 219/CGROUP or failed
   user-manager entry.

The service change is reversible if the graphical Ubuntu installer is ever
actually needed:

```bash
sudo snap start --enable ubuntu-desktop-installer.subiquity-server
```

## Prepared execution environment

| item | prepared state |
|---|---|
| WSL | 2.7.12.0, kernel 6.18.33.2, systemd enabled |
| resource limit | 8 CPUs, 10 GiB RAM, 12 GiB swap |
| venv | `/home/local/.venvs/pure-spin-v12-torch210-cu126` |
| Python/Torch | Python 3.10.12, Torch 2.10.0+cu126 |
| CUDA | runtime 12.6, nvcc 12.6.85, native `sm_75` |
| fused baseline | Mamba-SSM 2.3.2.post1, causal-conv1d 1.7.0 |
| local build/cache | venv, CUDA builds, Torch extensions, Triton and Inductor on WSL ext4 |
| E: data/cache | pip, Hugging Face, Tiny Shakespeare, and future large GGUF downloads only |
| llama.cpp | pinned source/build on WSL ext4; CUDA 12.6 device discovery passed |

After the WSL update and cold-start campaign, `run_wsl_tests.sh` regenerated
`artifacts/wsl_environment.json`, loaded all seven raw-CUDA extension symbols,
loaded the official fused Mamba-2 path, and passed all 24 tests. llama.cpp also
reported the RTX 2070 SUPER as compute capability 7.5. No model weight was
downloaded.

## Firmware handoff

Gigabyte's official rev. 1.0 support page lists F14a as a 6.80 MB image dated
2025-06-11 with checksum `40F6` and security-fix release notes:
<https://www.gigabyte.com/Motherboard/Z390-AORUS-PRO-WIFI-rev-10/support>

Before flashing:

- verify the physical PCB revision is rev. 1.0 and use only its matching
  official image;
- retain the downloaded archive and verify the vendor-displayed checksum;
- record the current XMP, virtualization, Secure Boot/TPM, boot-order, fan/pump,
  CPU power-limit, PCIe, and ReBAR settings because firmware defaults may
  change them;
- ensure the Windows recovery key is available if device encryption or
  BitLocker is enabled. Its state could not be read from this non-elevated
  session, so it is explicitly unverified.

After the restart, do not resume timing claims until all of these pass:

```powershell
Get-CimInstance Win32_BIOS | Select-Object SMBIOSBIOSVersion, ReleaseDate
wsl.exe --version
wsl.exe -d Ubuntu -- bash -lc 'systemctl status user@1000.service --no-pager'
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/c/Users/HaydenLocal/Programming/AI_Culture_Mind/SSM-Models/pure_spin_ssm_v1_2 && source wsl_env.sh && bash run_wsl_tests.sh'
```

Then verify that RAM is again operating at DDR4-3200, virtualization still
allows WSL2 to start, Secure Boot/TPM state is expected, the GPU negotiates
PCIe 3.0 x16 under load, and cooling/power settings are stable. A post-update
hardware snapshot should supersede—never silently rewrite—the F12 snapshot.
