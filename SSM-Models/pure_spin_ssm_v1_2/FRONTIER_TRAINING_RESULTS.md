# Pure Spin v1.2 frontier training backend

**Status:** implemented and gradient-validated; current throughput claims are
separated into isolated-kernel, steady-step, and end-to-end measurements

**Historical WikiText natural-data artifact (pre-reconstruction backend):**
[`artifacts/wikitext2_byte_spin_ladder_3seed_summary.json`](artifacts/wikitext2_byte_spin_ladder_3seed_summary.json)

**Current guarded-backward artifacts:**
[`artifacts/raw_cuda_factorized_training_guarded_reconstruct_rtx2070s_20260821.json`](artifacts/raw_cuda_factorized_training_guarded_reconstruct_rtx2070s_20260821.json)
and
[`artifacts/steady_step_spin_ladder_guarded_reconstruct_vs_mamba2_wsl.json`](artifacts/steady_step_spin_ladder_guarded_reconstruct_vs_mamba2_wsl.json)

## What is implemented

The training backend now has two raw CUDA schedules:

1. `raw_cuda_controller` fuses coefficient-controller dot products, all 28
   ordered Spin(8) factors, the recurrent update, and exact reverse mode into
   one warp per `(batch, channel)`.
2. `raw_cuda_factorized` lets the dense controller execute as a parallel GEMM,
   then runs the noncommutative ordered factors and recurrence in a raw CUDA
   warp. Its backward reconstructs the rotated state from the affine output,
   then reconstructs factor inputs using orthogonality instead of storing 28
   intermediate states.

The second schedule is faster on the RTX 2070 SUPER. The first schedule placed
only 16 warps on the GPU at the benchmark shape and made each warp compute 28
serial controller dot products. Splitting at the coordinate boundary exposes
the controller to the GPU's dense-matrix machinery and reserves the custom
kernel for the algebra that GEMM cannot express.

Both schedules use PyTorch's current CUDA stream and return gradients for every
trainable input in their scope. Output and full-gradient parity are checked
against the maintained Triton implementations. The coordinate backend accepts
any exact generator subset with between 1 and 28 factors.

The new reverse pass uses

\[
R_t h_{t-1} = \frac{h_t-d_t}{a_t}
\]

on the ordinary nonzero-retention path. For zero or tiny float32 `a_t`, it
falls back to replaying the forward factors. This guard matters: sigmoid is
strictly positive over the reals, but finite-precision sigmoid can underflow.
The test suite explicitly places both zero and `1e-9` retentions inside a
sequence and checks output plus all gradients against Triton.

At `(B=8, L=256, C=2)`, the final guarded 28-coordinate
forward+backward kernel took 3,700.9 microseconds versus 12,286.5 for Triton, a
69.9% latency reduction. It is also 10.4% below the original raw kernel's
4,132.5 microseconds. This microbenchmark excludes the controller GEMM.

## Exact nested-group model

The promoted four-block configuration is

\[
Spin(3)\hookrightarrow Spin(4)\hookrightarrow Spin(6)\hookrightarrow Spin(8),
\]

with respectively `3, 6, 15, 28` plane generators. The implementation selects
all coordinate planes supported on the first `n` axes, so each stage is a
closed Lie subalgebra and an actual subgroup restriction of the same Spin(8)
triality representation. It is not a prefix or a heuristic pruning mask.

The useful special isomorphisms are:

- `SU(1)` is the trivial identity group and supplies no trainable action;
- `Spin(3) ~= SU(2)`;
- `Spin(4) ~= SU(2) x SU(2)`;
- `Spin(6) ~= SU(4)`;
- `Spin(8)` supplies the full `8v + 8+ + 8-` triality action.

All blocks retain the same 24-scalar-per-channel triality state. The residual
stream maps the restricted representations between stages. A dedicated test
proves that the Spin(3) kernel equals the full Spin(8) kernel with the other 25
plane coordinates set exactly to zero, for outputs and gradients.

## Matched natural-data result

The table in this section is the last fully provenance-captured three-seed
WikiText comparison before guarded backward reconstruction. It is preserved as
historical evidence; it must not be relabelled as a Tiny Shakespeare result or
as a timing result for the current CUDA source.

Both candidates used the same immutable WikiText-2 UTF-8 byte streams, batches,
optimizer, 300 steps, batch size 8, sequence length 256, RTX 2070 SUPER, Torch
2.10.0+cu130, and seeds 17, 29, and 43. Pure Spin has 626,516 parameters and
official fused Mamba-2 has 623,740, a 0.44% gap.

| seed | Spin ladder bpb | Mamba-2 bpb | Spin tokens/s | Mamba-2 tokens/s | Mamba/Spin |
|---:|---:|---:|---:|---:|---:|
| 17 | 2.718 | 2.500 | 80,967 | 86,264 | 1.065x |
| 29 | 2.732 | 2.470 | 82,596 | 88,597 | 1.073x |
| 43 | 2.747 | 2.500 | 78,529 | 89,111 | 1.135x |
| mean | **2.732** | **2.490** | **80,697** | **87,991** | **1.091x** |

Relative to the first three-seed Triton controller result, mean Pure Spin
throughput rose from 14,987 to 80,697 tokens/s, or 5.38x. Relative to the first
raw controller implementation measured in the final WSL environment at seed
17, the ladder rose from 23,564 to 80,967 tokens/s, or 3.44x.

This historical run does not beat Mamba-2: its mean throughput remains 9.1%
lower and its mean
validation result remains worse by 0.242 bits/byte. It also makes no comparison
claim against Jamba or Falcon-Mamba; those are differently scaled pretrained
systems, not matched local controls. The current external programme also pins
Mamba-3 SISO/MIMO, GKA, and GDN, but no speed or quality row is promoted until
its separately isolated implementation passes the same matched-run contract.

## Current CUDA 12.6 steady-step result

The current implementation was also measured with four alternating execution
cycles: Spin then Mamba, followed by Mamba then Spin, repeated twice. Every
model/cycle used ten warmup steps and five ten-step CUDA-event windows. The
timed region includes forward, backward, gradient clipping, and AdamW; it
excludes data loading, host-to-device transfer, and validation.

| environment | Spin median | Mamba-2 median | Mamba / Spin |
|---|---:|---:|---:|
| Torch 2.10 cu126 | **70,615** | 70,255 | 0.995x |
| Torch 2.10 cu130 control | 57,434 | **63,598** | 1.107x |

These are current-code A/B runs with identical source hashes and protocol.
cu126 improves Spin throughput by 23.0% and Mamba-2 throughput by 10.5% over
cu130 on this machine. Within cu126, the 0.51% Spin lead is too small relative
to observed cycle variation to support a model-speed superiority claim. It
does establish cu126 as the better local runtime. Exact artifacts are
[`artifacts/steady_step_spin_ladder_vs_mamba2_cu126.json`](artifacts/steady_step_spin_ladder_vs_mamba2_cu126.json)
and
[`artifacts/steady_step_spin_ladder_vs_mamba2_cu130_current.json`](artifacts/steady_step_spin_ladder_vs_mamba2_cu130_current.json).

### Post-F14a repeatability gate

The same cu126 order-balanced protocol was rerun twice after the accepted F14a
firmware, Secure Boot, and DDR4-3200 validation. The ordering reversed:

| repeat | Spin median | Mamba-2 median | Spin / Mamba |
|---|---:|---:|---:|
| F14a run 1 | 68,328 | **70,318** | 0.9717x |
| F14a run 2 | **73,626** | 72,479 | 1.0158x |

Across runs, the geometric-mean Spin/Mamba ratio is 0.9935x. Across the eight
paired execution cycles, ratios range from 0.9228x to 1.0487x, with median
0.9960x. Pooling model cycle medians yields a different small ordering
(1.0051x), demonstrating that sub-percent point estimates are not resolved by
this protocol. The promoted verdict is therefore **throughput ordering
unresolved at observed repeatability**. Neither run is selected as the speed
claim.

Bracketing telemetry also differed: run 1 moved from 56 to 60 degrees Celsius,
P0, and 52.67 to 77.42 W; run 2 moved from 53 degrees Celsius/P3/40.90 W to 58
degrees Celsius/P0/77.06 W. Both ended at 1,980 MHz core, 7,300 MHz memory, and
PCIe Gen3 x16. These are pre/post samples rather than per-window traces, so they
support the unresolved verdict but cannot apportion the cycle variance.

The compatibility-checking analysis and exact inputs are:

- [`artifacts/steady_step_spin_ladder_vs_mamba2_cu126_f14a_repeatability.json`](artifacts/steady_step_spin_ladder_vs_mamba2_cu126_f14a_repeatability.json)
- [`artifacts/steady_step_spin_ladder_vs_mamba2_cu126_f14a.json`](artifacts/steady_step_spin_ladder_vs_mamba2_cu126_f14a.json)
- [`artifacts/steady_step_spin_ladder_vs_mamba2_cu126_f14a_repeat.json`](artifacts/steady_step_spin_ladder_vs_mamba2_cu126_f14a_repeat.json)

This closes the current low-level speed gate as a tie at the resolution of the
local measurement. Architectural quality, not another scan-only timing, is the
next model frontier.

## Current Tiny Shakespeare result

The first complete post-migration run used seed 17, 300 optimizer steps, batch
size 8, sequence length 256, the pinned disjoint Tiny Shakespeare byte splits,
and the current `raw_cuda_hybrid` Spin ladder. Both candidates were constructed
before training and used the same batches and validation windows.

| run | Spin bpb | Mamba-2 bpb | Spin tok/s | Mamba-2 tok/s |
|---|---:|---:|---:|---:|
| cu126 first | 2.7263 | **2.4606** | **67,259** | 53,386 |
| cu126 repeat | 2.7263 | **2.4606** | 56,713 | **56,877** |
| preserved cu130 | 2.7263 | **2.4606** | 42,831 | **49,133** |

The parameter gap is 0.443%. Mamba-2 improves validation by 0.2657 bits/byte
and uses about 22% less peak CUDA memory (118.0 versus 151.6 MB). Sequential
timing varies enough to reverse the speed ordering, which is why the
order-balanced table above is authoritative for throughput. These remain
single-seed convergence results, but they directly falsify any claim that the
present v1.2 wins matched Shakespeare quality. The cu126 artifacts are
[`artifacts/shakespeare_byte_spin_ladder_vs_mamba2_seed17_300_wsl_cu126.json`](artifacts/shakespeare_byte_spin_ladder_vs_mamba2_seed17_300_wsl_cu126.json)
and
[`artifacts/shakespeare_byte_spin_ladder_vs_mamba2_seed17_300_wsl_cu126_repeat.json`](artifacts/shakespeare_byte_spin_ladder_vs_mamba2_seed17_300_wsl_cu126_repeat.json).

### Post-F14a three-seed quality gate

The accepted environment was then run at seeds 17, 29, and 43 with identical
data, 300-step budget, optimizer, model construction, and 0.443% parameter gap.
Sequential training timers are deliberately excluded from this summary; speed
belongs to the order-balanced gate above.

| seed | Pure Spin bpb | fused Mamba-2 bpb | Spin minus Mamba |
|---:|---:|---:|---:|
| 17 | 2.7263 | **2.4606** | +0.2657 |
| 29 | 2.7433 | **2.5023** | +0.2410 |
| 43 | 2.7736 | **2.5196** | +0.2539 |
| mean | 2.7477 | **2.4942** | +0.2535 |

Mamba-2 wins all three recorded seeds. Sample standard deviations are 0.0239
bpb for Pure Spin and 0.0304 bpb for Mamba-2; the seedwise deficit remains
positive throughout. Peak allocated CUDA memory is also unchanged at
151,556,608 bytes for Pure Spin versus 117,968,896 for Mamba-2. This is a
matched small-model, short-budget empirical result, not a general theorem, but
it robustly closes the present v1.2 quality claim in Mamba-2's favor.

The provenance-checked summary is
[`artifacts/shakespeare_byte_spin_ladder_vs_mamba2_3seed_wsl_cu126_f14a_summary.json`](artifacts/shakespeare_byte_spin_ladder_vs_mamba2_3seed_wsl_cu126_f14a_summary.json),
with each source artifact and SHA-256 embedded. The result identifies the next
research target sharply: improve the selective controller/readout or state
allocation while preserving the exact bounded Spin action and matched budget.

## Triality-invariant amplitude readout gate

The first theory-driven quality intervention transferred v1.3's amplitude
lesson without importing its failed dense exceptional transport. The candidate
kept the normalized triality direction and appended, per channel, the three
sector log-energies and bounded cubic triality contraction. Tests verify the
four scalars are invariant under the shared Spin(8) action and respond to
positive state rescaling.

The candidate adds 4,096 parameters (0.654%) and was frozen in commit
`34c2ff8` before seeds 71, 73, and 79 were trained.

| seed | direction bpb | invariant bpb | improvement |
|---:|---:|---:|---:|
| 71 | 2.71058 | **2.69516** | +0.01543 |
| 73 | 2.72186 | **2.72010** | +0.00177 |
| 79 | 2.72772 | **2.72510** | +0.00262 |
| mean | 2.72006 | **2.71345** | +0.00661 |

It won all three seeds and never regressed, but failed the separately required
mean-improvement threshold of +0.0100 bpb. Direction therefore remains the
default. The favorable effect closes only about 2.6% of the established mean
gap to Mamba-2 and does not justify a speed gate or default change. See
[`TRIALITY_INVARIANT_READOUT_RESULTS.md`](TRIALITY_INVARIANT_READOUT_RESULTS.md)
and
[`artifacts/shakespeare_triality_readout_gate_summary_cu126_f14a.json`](artifacts/shakespeare_triality_readout_gate_summary_cu126_f14a.json).

## Channel-mixer falsification

Two custom non-exponential candidates were implemented:

- `sol_bounded_quadratic`, using a separate value and bounded rational gate;
- `sol_self_gate`, using one projected stream and the map
  `z * (1 + z / sqrt(1 + z^2))`.

Six seed-17 ablations tested fewer parameters, matched parameters, matched
depth, and reallocation of saved gate parameters into model width. None beat
SwiGLU on the matched quality/speed criteria. The closest small model used
30.6% fewer parameters and 8.6% less peak memory, but degraded validation by
0.026 bpb. SwiGLU therefore remains the maintained default.

## Reproduction

```bash
source wsl_env.sh
bash run_wsl_tests.sh
python benchmark.py --dataset tiny_shakespeare --offline \
  --steps 300 --batch-size 8 --sequence-length 256 --seed 17 \
  --spin-backend raw_cuda_hybrid \
  --spin-group-schedule 3 4 6 8

python benchmark_steady_step.py \
  --cycles 4 --windows 5 --steps-per-window 10 --warmup-steps 10 \
  --output artifacts/steady_step_spin_ladder_guarded_reconstruct_vs_mamba2_wsl.json

python analyze_steady_step_repeats.py \
  artifacts/steady_step_spin_ladder_vs_mamba2_cu126_f14a.json \
  artifacts/steady_step_spin_ladder_vs_mamba2_cu126_f14a_repeat.json \
  --output artifacts/steady_step_spin_ladder_vs_mamba2_cu126_f14a_repeatability.json

python summarize_matched_runs.py \
  artifacts/shakespeare_byte_spin_ladder_vs_mamba2_seed{17,29,43}_300_wsl_cu126_f14a.json \
  --output artifacts/shakespeare_byte_spin_ladder_vs_mamba2_3seed_wsl_cu126_f14a_summary.json
```

To replay the historical dataset, request
`--dataset wikitext2_legacy` explicitly and retain the original stream hashes.
The current default is pinned Tiny Shakespeare. External checkpoint metadata
and exact source revisions are audited by
`python external_baselines/audit.py --live`; that command downloads no weights.

The semantic chunk-parallel associative affine compiler is now implemented and
documented in
[`CHUNK_PARALLEL_COMPILER_RESULTS.md`](CHUNK_PARALLEL_COMPILER_RESULTS.md).
It reduces eager matrix-composition work and wins the materialized forward
control, but remains much slower than the factor-direct raw kernel. The same
work also produced `raw_cuda_hybrid`: isotypic-split forward plus packed
backward, measuring 1.54% faster than packed raw CUDA on order-balanced
complete steps. The next speed frontier is therefore fused factor-to-chunk
compilation, not another eager tree. The next quality frontier remains
controller and readout design; the hardware result does not erase the observed
0.242 bpb gap.
