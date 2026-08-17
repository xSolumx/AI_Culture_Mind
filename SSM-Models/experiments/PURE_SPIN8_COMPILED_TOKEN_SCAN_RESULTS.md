# Pure Spin(8) Compiled Token-Scan Results

Implementation and execution: **2026-08-17T03:40--03:47+02:00**

## Outcome

Pure Spin(8) v1.1 adds a model-level compiler for finite latent-action
dictionaries. The validated seed-1 router is evaluated once for each of its
eight tokens, producing a frozen `[8,3,8,8]` faithful triality action table.
One Triton program per `(batch, representation)` then keeps the eight-scalar
state in registers and emits every prefix.

Across the frozen eight-cell grid, the compiled Triton tracker is
**30.7x--67.1x faster** than the original dynamic token router, 28-factor
triality action construction, and work-efficient prefix scan. Median speedup is
**42.75x**. The maximum source-versus-compiled output difference is `1.32e-5`,
and all held-out center/identity signatures remain correct.

## What was compiled

The source is the passing fresh seed-1 model from the latent-increment
validation. Compilation removes no recurrent state and introduces no learned
approximation: it caches the already-learned action of each finite token. The
runtime still carries all 24 faithful `(8v,8s+,8s-)` state scalars.

The compiled checkpoint has zero trainable parameters because it is an
inference artifact. It records and hashes the source experiment artifact,
source learned checkpoint, action table, initial state, representation order,
and compiler metadata. Training or adapting the dictionary continues to use
the original differentiable router and eager action construction.

## RTX 2070 SUPER measurements

All calls emit every prefix in float32 and are CUDA-synchronized. Timings
exclude one-time checkpoint load and action compilation.

| Batch | Length | Dynamic source | Compiled Triton | Speedup | Dynamic peak | Compiled peak |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | `4,690 us` | `117.8 us` | `39.8x` | `9.5 MB` | `8.6 MB` |
| 1 | 128 | `6,963 us` | `117.7 us` | `59.2x` | `16.4 MB` | `8.6 MB` |
| 1 | 1,024 | `8,992 us` | `284.4 us` | `31.6x` | `73.3 MB` | `8.9 MB` |
| 8 | 16 | `4,888 us` | `159.2 us` | `30.7x` | `16.4 MB` | `8.6 MB` |
| 8 | 128 | `7,117 us` | `155.9 us` | `45.7x` | `73.3 MB` | `8.9 MB` |
| 8 | 1,024 | `18,277 us` | `319.3 us` | `57.2x` | `509.6 MB` | `11.0 MB` |
| 32 | 16 | `4,878 us` | `158.7 us` | `30.7x` | `39.9 MB` | `8.7 MB` |
| 32 | 128 | `10,360 us` | `154.4 us` | `67.1x` | `259.1 MB` | `9.8 MB` |

The eager compiled recurrence is intentionally a correctness/training oracle,
not the fast path: its Python launch per token makes it slower than the dynamic
parallel path at long lengths.

## Correctness gates

- CUDA forward parity and custom initial-state-gradient parity pass at lengths
  1, 5, and 17 in unit tests.
- Source-versus-compiled maximum error over all forced relation schedules is
  `2.56e-6`.
- Source-versus-compiled maximum error over the timing grid is `1.32e-5`, below
  the frozen `5e-5` tolerance.
- Center classification, center-row correctness, and identity-row correctness
  remain 1.0 on every L16/L64/L128 early/late relation split.
- The compiled checkpoint independently rehashes and reloads exactly.

## Artifacts

Benchmark artifact:
`artifacts/pure_spin8_compiled_token_scan_seed1.json`<br>
SHA-256:
`aa19ba66e5e2d17967f189c4744a1f1c165e0181b11c999f6cd0e4329dd6fb55`

Compiled checkpoint:
`../checkpoints/pure_spin8_compiled_token_scan/compiled_latent_pure_spin8_seed1.pt`<br>
SHA-256:
`40d71f4e93b957e2ece9b308c02b95c58c46b0e55c9d2f60969e9924a0aa2305`

Action table plus initial-state tensor hash:
`1bc5982e0bb684a77c371021232e48acd6a4082e1908817569497f13ff86065c`

Implementation:
[`pure_spin8_ssm/discrete_scan.py`](../pure_spin8_ssm/discrete_scan.py)<br>
Runner:
[`benchmark_pure_spin8_compiled_token_scan.py`](../benchmark_pure_spin8_compiled_token_scan.py)

## Claim boundary

This is a real fused, model-level, frozen-dictionary inference result on the
recorded workstation. It is not a parallel-prefix kernel: sequence depth is
serial inside each program. It does not provide action-table gradients, fused
training, continuous-observation routing, a fused-Mamba comparison, or a
hardware-independent speed theorem.
