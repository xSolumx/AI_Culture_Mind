# AI_Culture_Mind storage audit

**Measured:** 2026-08-22 on the local C:/E:/WSL installation
**Mutation boundary:** read-only accounting; no environment, cache, Git object,
or benchmark datum was deleted

All values below are allocated bytes reported by `du -B1` or exact file sizes,
converted to GiB using 2^30 bytes. They are point-in-time measurements.

## Checkout on C:

| Component | Bytes | GiB | Status |
|---|---:|---:|---|
| Entire checkout | 17,049,337,856 | 15.88 | observed |
| `.git` | 9,951,162,368 | 9.27 | includes 6.20 GiB of verified garbage/cruft |
| root Windows `.venv` | 2,995,998,720 | 2.79 | referenced by repo documentation |
| `SSM-Models/.venv-cuda` | 3,270,234,112 | 3.05 | referenced by two holonomy runners |
| `Spin8-SSM-Benchmark/data/hf` | 285,941,760 | 0.27 | reproducible CIFAR/plain-text parquet cache |
| all remaining source, docs, small artifacts, and working files | 546,000,896 | 0.51 | retained project content |

The apparent 16 GB repository is therefore only about 0.51 GiB of ordinary
working content after subtracting Git storage, two embedded environments, and
the explicit dataset cache.

## WSL environments stored physically inside the C: WSL VHD

| Environment | Bytes | GiB | Status |
|---|---:|---:|---|
| `/home/local/.venvs/pure-spin-v12` | 6,442,508,288 | 6.00 | obsolete Torch 2.6/cu124/Mamba 2.2.6; removed post-audit |
| `/home/local/.venvs/pure-spin-v12-torch210` | 6,408,286,208 | 5.97 | cu130 control; removed after the cu126 A/B and llama.cpp migration |
| `/home/local/.venvs/pure-spin-v12-torch210-cu126` | 7,660,822,320 | 7.13 | current authoritative v1.2 environment |
| `/home/local/.venvs/schurscan-fla` | 5,115,138,048 | 4.76 | separate Programme 2 environment; retain |
| `/home/local/.venvs/schurscan-fla-smoke` | 30,814,208 | 0.03 | separate smoke environment |

The initial audit found distinct cu124 and cu130 Pure Spin stacks. Both have now
been superseded by the validated cu126 stack, and current scripts reference
only `pure-spin-v12-torch210-cu126`. The table retains the deleted sizes to make
the reclamation auditable rather than erasing the pre-cleanup state.

## Caches and wheels

| Location | Bytes | GiB | Attribution |
|---|---:|---:|---|
| WSL `~/.cache/pip` | 18,104,000,512 | 16.86 | shared; 18.1 GB HTTP cache, zero built wheels |
| Windows pip cache | 9,567,092,736 | 8.91 | shared; 7.65 GB HTTP plus 1.91 GB built wheels |
| Windows Hugging Face cache | 2,629,128,192 | 2.45 | mostly plant/vision models from another repo |
| WSL `~/.triton` | 338,137,088 | 0.32 | shared compiled-kernel cache |
| old v1.2 data cache | 22,630,400 | 0.02 | superseded by E:-backed cache |
| old Torch extension cache | 14,995,456 | 0.01 | superseded by E:-backed cache |

Only 15.5 MB of the Windows Hugging Face total is the duplicated old WikiText
dataset namespace; the 2.31+ GiB Grounding DINO data belongs to the plant-vision
work and must not be charged to or removed for this programme.

## Git-specific waste

`git count-objects -vH` found:

- seven interrupted `tmp_obj_*` files: 3,215,513,946 bytes (3.00 GiB);
- one 3,445,129,588-byte (3.21 GiB) cruft pack containing 335 objects not
  reachable from any ref or reflog;
- that cruft pack is dominated by a 3,133.3 MiB GGUF blob whose on-disk object
  occupies 2,995.2 MiB;
- the three ordinary pack indexes share only two object IDs, so the live packs
  are not wholesale duplicates.

The 6.20 GiB Git reclaim estimate is therefore evidence-backed garbage/cruft,
not a proposal to rewrite live history.

## E: policy and current footprint

`wsl_env.sh` now fails closed unless `/mnt/e` is writable and directs only pip,
Hugging Face, dataset, and model downloads to
`E:\AI_Culture_Mind_Large\pure_spin_ssm_v1_2`. Latency-sensitive Torch,
Triton, Inductor, and CUDA compiler caches were moved back to WSL ext4 after a
measured Plan 9 small-file stall.

| E:-backed component | Bytes |
|---|---:|
| reproducible pip/download cache | 1,728,153,044 |
| superseded Triton compiler cache | 144,348,784 |
| superseded Torch-extension cache | 14,698,566 |
| pinned data cache | 1,115,394 |
| external model configs/cards, no weights | 389,703 |
| other empty/metadata cache entries | 5,698 |
| total | 1,888,711,189 |

No Falcon, Jamba, GKA, GDN, or Mamba-3 weight file was downloaded.

## Reclaim tiers

1. **Already reclaimed:** about 12.2 GiB from the cu124 and cu130 Pure Spin
   environments plus the old cu130 llama.cpp build. These are generated and
   recoverable only by reinstall/rebuild.
2. **Still reclaimable but not deleted:** 6.20 GiB of verified Git
   garbage/cruft. Removing it is a separate repository-maintenance operation.
3. **Reproducible shared caches:** a further 25.77 GiB across WSL and Windows
   pip. Purging or moving these loses no source but may slow unrelated projects
   until packages are downloaded again.
4. **Retain:** the authoritative cu126 venv, SchurScan venv, both referenced
   Windows venvs, and plant-model Hugging Face cache. The former cu130 control
   venv is no longer in this tier: its reproducible environment and benchmark
   provenance are recorded, and the generated environment has been removed.

Cleaning files inside the WSL ext4 filesystem reduces logical WSL usage but may
not immediately shrink the 45.84 GiB `ext4.vhdx` on C:. Returning those bytes
to Windows requires a separate, deliberate WSL shutdown/compaction operation.
That system-level mutation was not performed by this audit.

## Post-audit CUDA migration

The unreferenced 6.1 GiB Torch 2.6/cu124/Mamba 2.2.6 environment was removed;
it is recoverable only by reinstalling the recorded tuple. Its storage role is
now occupied by the validated Torch 2.10/cu126 environment. A minimal official
CUDA 12.6 compiler installation adds 228 MiB and avoids the full toolkit.

A current-code order-balanced A/B measured cu126 23.0% faster than cu130 for
Pure Spin and 10.5% faster for Mamba-2. The pinned llama.cpp source was then
rebuilt for `sm_75` with CUDA 12.6 and its runtime linkage and GPU discovery
were verified. The superseded cu130 venv (5.97 GiB) and old cu130 llama.cpp
build (0.25 GiB) were removed. Together with the earlier cu124 venv removal,
this reclaimed about 12.2 GiB of generated WSL content; recovery requires a
reinstall/rebuild, while the committed cu130 JSON evidence remains intact.

The E:-backed pip cache is now 1.61 GiB because it contains the reproducible
cu126 wheel downloads. Two superseded compiler-cache directories currently
remain on E: (`torch_extensions`, 14.0 MiB; `triton`, 137.7 MiB). They are not
used by the active configuration and are safe to delete, but the attempted
recursive removal was blocked by the local command policy, so this audit does
not falsely report them as reclaimed. Active compiler caches live on WSL ext4.
