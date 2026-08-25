# G16 SM75 trained-frontier shootout results

**Executed:** 2026-08-25 | **Start commit:** `5796a851df02e1d2c878dabe3e16246dfecd2fd6` | **Start status:** clean | **Runtime:** WSL2, RTX 2070 SUPER, SM75, PyTorch 2.9.0+cu128

**Artifact:** [`g16_frontier_shootout_sm75_2026-08-25.json`](artifacts/g16_frontier_shootout_sm75_2026-08-25.json)

**Artifact SHA-256:** `76323bb4b3a87705ac66e77bdd1c056f2a4cbb6bf7b5597386f4953187f2cac7`

## Bottom line

Official fused Mamba-2 wins the prospectively frozen one-seed development
cohort on ordinary TinyStories compression at every reported context. At 4,096
tokens it reaches `1.48571` BPRB, beating v1.4.5 by `0.09764` BPRB. It is also
substantially faster and smaller in measured peak CUDA allocation on this exact
SM75 runtime.

The local semantic GDN2 edit-law candidate does not improve this shell: it is
`0.02121` BPRB worse than v1.4.5 at 4,096 and slower in every phase. The actual
Transformers OLMo Hybrid is `0.03077` BPRB worse than v1.4.5, although its real
FLA/SDPA runtime is the fastest measured arm.

All four arms fail the learned-recall gate. OLMo's nominally best 8,192-byte
mean gain is only `0.003786` nats and is explicitly non-passing. The G16 result
therefore rejects the current v1.4.5/GDN2 ordinary-compression frontier at this
bounded scale without identifying a successful long-range factual-memory
model.

## Integrity and matching

All frozen integrity checks pass:

- four eligible real implementations completed all 1,000 optimizer updates;
- parameters are within `0.291%` of the 124,534-parameter reference;
- every arm consumed the same SHA-256-bound target-token stream;
- all losses, gradients, parameters, evaluation values, and recall values are
  finite;
- the run started from a clean committed tree after the prospective phase-one
  train-mode repair;
- Mamba is the official fused source at revision `e9594ce1...`;
- OLMo is Transformers 5.15.1 plus FLA 0.5.2 using
  `chunk_gated_delta_rule`, not a compatibility approximation.

Each arm received 4,096,000 target tokens: 200 updates at each context in
`256 -> 512 -> 1,024 -> 2,048 -> 4,096`, with 4,096 targets per update and
ordinary next-token cross-entropy only. The frozen 751,721-token training
snapshot is necessarily revisited to reach that budget; validation remains
separate and the repetition is paired across arms.

The cohort is parameter- and target-matched. It is not optimizer-matched or
step-time-matched: each architecture uses its declared native optimizer path,
and quality per wall-clock compute is a separate conclusion from quality per
fixed token budget.

## Final ordinary compression

| Arm | Params | BPRB 256 | 512 | 1,024 | 2,048 | 4,096 | Delta vs v1.4.5 at 4,096 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1.4.5 | 124,534 | 1.60333 | 1.59215 | 1.58867 | 1.58539 | 1.58335 | 0.00000 |
| local GDN2 | 124,414 | 1.64000 | 1.62346 | 1.61473 | 1.60819 | 1.60457 | +0.02121 |
| official fused Mamba-2 | 124,172 | **1.49410** | **1.48951** | **1.48764** | **1.48613** | **1.48571** | **-0.09764** |
| Transformers OLMo Hybrid | 124,376 | 1.68019 | 1.64804 | 1.62796 | 1.61637 | 1.61413 | +0.03077 |

The Mamba-2 lead is consistent across the entire context ladder in this one
seed. This is much stronger than selecting one favorable endpoint, but it is
still development evidence on one small snapshot and one GPU.

## Exact local systems measurements

Median synchronized update seconds after ten within-phase warmups:

| Arm | L256 | L512 | L1,024 | L2,048 | L4,096 | Peak training CUDA bytes |
|---|---:|---:|---:|---:|---:|---:|
| v1.4.5 | 0.08820 | 0.09889 | 0.10853 | 0.11490 | 0.12528 | 1,590,665,216 |
| local GDN2 | 0.10764 | 0.11890 | 0.13137 | 0.13874 | 0.15002 | 2,078,491,648 |
| official fused Mamba-2 | 0.02434 | 0.03611 | 0.03662 | 0.03378 | **0.02392** | 368,763,392 |
| Transformers OLMo Hybrid | **0.01592** | **0.01585** | **0.01634** | **0.01696** | 0.02509 | **322,308,096** |

These are real intra-cohort measurements on the named SM75 environment. They
are not hardware-general kernel rankings. Fixed `batch x length = 4,096`
changes batch shape across phases, and the external models use different
fused kernels from the local reference implementations.

## Long-range recall remains negative

| Arm | Mean matching-minus-counterfactual gain at 8,192 raw bytes (nats) | Frozen recall gate |
|---|---:|---|
| v1.4.5 | 0.0000117 | fail |
| local GDN2 | -0.0003030 | fail |
| official fused Mamba-2 | 0.0000010 | fail |
| Transformers OLMo Hybrid | 0.0037861 | fail |

The OLMo value is the numerical maximum, not a learned-recall result. All four
are effectively insensitive at the longest distance under the frozen gate.

## Research decision

1. Do not promote local GDN2 over v1.4.5 from edit-law elegance; it lost both
   compression and exact local step cost in the matched shell.
2. Treat official fused Mamba-2 as the ordinary-compression reference that the
   next 1.4/1.4.5 successor must beat, not merely match an older local fallback.
3. Keep G15B separate. G16 shows that ordinary next-token training did not
   produce reliable long-range binding in any arm; the interleaved controller
   gate must directly establish content address/edit/query capability before
   returning to natural text.
4. If G15B passes, the scientifically defensible language model is a hybrid in
   which a commissioned content-edit memory earns its place by recall ablation,
   while Mamba-2 remains the ordinary-compression control.
5. Repeat only surviving designs across fresh seeds and larger snapshots before
   making a model-family or scaling claim.

## Claim boundary

This is a completed prospectively frozen single-seed trained development
shootout. It rejects current local candidates at the tested scale; it does not
prove universal Mamba-2 superiority, optimizer optimality, a scaling law,
general language quality, or hardware-general efficiency. The retained
checkpoints are research artifacts in the recorded WSL run directory, not
released pretrained models.
