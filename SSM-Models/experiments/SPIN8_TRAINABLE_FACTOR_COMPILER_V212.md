# Trainable Spin(8) factor compiler v2.1.2

## Result

The continuous Pure Spin(8) layer no longer has to construct and store three
`8 x 8` action matrices at every token. Its factorized chart is already an
ordered product of 28 orthogonal plane factors. Version 2.1.2 applies those
factors directly to each recurrent eight-vector.

The reverse pass exploits orthogonality. Starting from the rotated state, it
applies inverse factors in reverse order to recover each pre-factor state and
transport the adjoint. This computes gradients for all 28 coordinates, scale,
drive, and initial state in linear factor work without storing a 28-state tape.
The coordinate gradient propagates through the learned controller.

A second backend goes further: it evaluates the controller's linear map inside
the Triton recurrence and atomically returns controller weight, bias, and input
gradients. Thus the coefficient controller, triality factor application, and
recurrent scan are one full-forward/full-backward CUDA primitive.

## Why maximal fusion is not the selected lowering

The three triality representations share the same 28 coordinates. Full fusion
recomputes those coordinates in each representation program and atomically
reduces three controller-gradient contributions. The staged direct-factor path
uses the accelerator's matrix multiplication once, then reuses the coordinate
tensor across `8v`, `8s+`, and `8s-`.

The measured compiler decision is therefore to preserve that reuse boundary.
Full fusion is a verified backend, not the default on this device. This is an
example where algebraic sharing determines the profitable fusion boundary.

## RTX 2070 SUPER audit

Artifact:
[`spin8_factorized_training_rtx2070s_20260821.json`](artifacts/spin8_factorized_training_rtx2070s_20260821.json),
SHA-256 `cc9057fe40ed65851f1c4a23be078c5467b96b0329f0c287492e2bc5dbf3e80a`.

All timings are full FP32 forward and backward, including input, initial-state,
retention, write, drive, RMSNorm, and coefficient-controller gradients. They do
not include an optimizer step.

| `(B,L,C,input)` | staged direct / eager | fully fused / eager | fully fused / staged | eager transient | staged transient | fused transient |
|---|---:|---:|---:|---:|---:|---:|
| `(4,128,4,16)` | `11.615x` | `8.694x` | `0.748x` | 150.8 MB | 2.02 MB | 1.58 MB |
| `(8,128,8,16)` | `14.833x` | `11.694x` | `0.788x` | 600.9 MB | 6.90 MB | 6.09 MB |
| `(4,512,4,16)` | `14.916x` | `10.747x` | `0.720x` | 601.1 MB | 7.03 MB | 6.28 MB |

Maximum forward error is below `4.77e-7`. Maximum full-model gradient error is
below `7.63e-5`. Controller weight and bias gradients are finite and nonzero in
every row. Timings are interleaved by backend and transient allocation is
measured relative to the synchronized live baseline.

`compiled_controller` requests maximal fusion. `compiled_factorized` requests
the staged direct-factor path. `compiled_auto` invokes the v2.1.2 cost model;
an exact matching hardware/model-shape profile selects the lowest measured
median. In the absence of a matching profile, CUDA FP32 conservatively keeps
the reusable controller result and avoids materialized actions.

## Relationship to external work

[Givens coordinate descent](https://arxiv.org/abs/1312.0624) and
[quasi-Givens orthogonal fine-tuning](https://arxiv.org/abs/2404.04316) provide
precedents for efficient learned orthogonal factors. The compiler result here
is narrower and different: the same Spin(8) coordinate drives three inequivalent
representations inside a recurrent scan, and the reverse kernel exploits that
specific shared-coordinate structure.

[Mamba](https://arxiv.org/abs/2312.00752) motivates coupling scan algebra to
hardware implementation. No Mamba throughput or task-quality comparison is
made by this audit.

## Claim boundary

Established: exact action and full-gradient parity, learned controller
gradients, maintained-layer masking, full controller/action/scan fusion, a
faster reuse-preserving staged lowering, profile-driven selection, and bounded
local CUDA timing/memory evidence.

Open: FP16/BF16 training, Tensor-Core controller lowering, representations
outside canonical full triality, optimizer fusion, checkpointed/chunked long
training, cross-device profiles, natural-task quality, and comparison with
optimized modern SSM training.
