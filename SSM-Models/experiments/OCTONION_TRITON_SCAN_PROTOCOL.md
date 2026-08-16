# Fused Octonion Recurrence WSL/Triton Protocol

Protocol frozen: **2026-08-16T20:56:25+02:00**

## Scope

Benchmark the optional fused left-octonion recurrence in the available Ubuntu
WSL2 CUDA environment. This is a systems continuation of the already frozen
operator-lift theorem/audit, not a replacement experiment.

Environment discovered before freezing:

- Ubuntu WSL2;
- Python 3.10.12;
- PyTorch 2.4.0+cu121;
- Triton 3.0.0;
- NVIDIA GeForce RTX 2070 SUPER, compute capability 7.5, 8 GiB;
- host driver 595.97.

## Frozen implementations

Compare:

1. `triton_fused_recurrent`: one program per batch/lane, eight state scalars
   retained in registers across the sequence chunk, with a fused reverse
   recurrence;
2. `pytorch_raw_recurrent`: the transparent per-token Python/PyTorch oracle;
3. `pytorch_work_efficient_operator`: the ordered Blelloch-style `8 by 8`
   operator tree used for parallel training.

All paths normalize the same input token octonions and implement the same
left-parenthesized state update `h_t = u_t h_(t-1)`. Do not compare a raw
octonion prefix collapsed to one octonion: that is algebraically invalid and
is tested as a falsifier elsewhere.

## Frozen shapes and timing

- dtype: float32;
- batch: 8;
- lanes: 4;
- lengths: 128, 512, 1024, 4096;
- warmups: 5;
- repeats: 20;
- CUDA Events with synchronization;
- report minimum, median, mean, population standard deviation, p20, and p80.

Forward timing includes token normalization and output materialization. Time
the raw PyTorch recurrent oracle only through length 512 to avoid turning a
known launch-overhead control into the dominant run cost. Time full
forward/backward at lengths 128, 512, and 1024 for Triton and the work-efficient
operator path; include the raw recurrent backward control through length 512.
Backward timing uses fresh differentiable inputs and a fixed dense output
cotangent on every repeat.

Record forward and gradient discrepancies before interpreting timing. Required
gates:

- all three WSL unit tests pass;
- fused versus raw forward maximum absolute error below `2e-5` at length 127;
- token and initial-state gradient errors below `2e-4` absolute at length 31;
- fused length-4096 unit-norm drift below `2e-4`;
- no nonfinite output or gradient; and
- compile time excluded by warmup.

## Claim boundary

A passing result establishes a fused differentiable WSL/CUDA implementation of
the compact recurrence and its local timing against two eager PyTorch paths.
It does not establish a language-model result, superiority over production
Mamba/Delta kernels, a general fused associative matrix scan, or portability
beyond the recorded Triton/CUDA environment.
