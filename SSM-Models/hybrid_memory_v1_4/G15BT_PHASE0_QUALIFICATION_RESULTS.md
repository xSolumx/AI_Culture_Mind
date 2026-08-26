# G15B-T Phase-0 qualification results

**Frozen protocol:**
[`G15BT_TRANSACTIONAL_DELTA_PROTOCOL_2026-08-26.md`](G15BT_TRANSACTIONAL_DELTA_PROTOCOL_2026-08-26.md)
**Exact artifact:**
[`artifacts/g15bt_phase0_qualification_sm75_2026-08-26.json`](artifacts/g15bt_phase0_qualification_sm75_2026-08-26.json)
**SHA-256:**
`0b4683ad3b66f7dc010e03737550873cd695d46ade897f21e094d38f4ece2438`
**Execution commit:** `86372b82edce95bdd17627e779418e9399a70ae8`
**Status:** passed implementation qualification; Phase 1 authorized
prospectively; no learning or model-quality result

## Bottom line

G15B-T Phase 0 passed every frozen implementation, matching, causality,
contraction, numerical-parity, gradient-reach, provenance, and exact-SM75
gate. The full-view control `F` and strict-history transactional arm `T` are
genuinely matched at 38,082 total/active parameters and 5,632 bytes of FP32
streaming state capacity for batch two.

The pass authorizes only the already prospective Phase-1 constructed training
screen. No G15B-T model has been trained, no learned overwrite result exists,
and the architecture is not promoted. The formally failed R5 and R5-S
retained-checkpoint results remain unchanged and historical; G15B-T is a fresh
architectural pivot rather than a retrospective repair.

## Exact execution

- NVIDIA GeForce RTX 2070 SUPER, exact compute capability `(7,5)`;
- Python 3.11.16;
- PyTorch 2.9.0+cu128 with CUDA runtime 12.8;
- clean Git status at quality-run start;
- evidentiary artifact from commit
  `86372b82edce95bdd17627e779418e9399a70ae8`;
- elapsed qualification time `1.802395058` seconds;
- peak allocated CUDA memory `23,233,536` bytes.

The artifact binds the protocol, qualification harness, model shell, and
transactional-delta implementation by SHA-256.

## Matched arms

| Contract | Full-view `F` | Strict-history `T` | Result |
|---|---:|---:|---|
| Total parameters | 38,082 | 38,082 | exact |
| Active parameters | 38,082 | 38,082 | exact |
| Batch-2 FP32 state capacity | 5,632 bytes | 5,632 bytes | exact |
| State-dict keys | same | same | pass |
| Named parameter shapes | same | same | pass |
| Full plus history views computed | yes | yes | matched graph contract |

This is architecture qualification, not measured training-step parity. Phase 1
must still record update time and peak memory under paired training schedules.

## Causality and contraction

At a fixed position, changing the current token produced a nonzero full-view
effect while leaving the strict-history view exactly unchanged:

- current-to-history maximum residual: `0`;
- full-view current-token effect: `6.393168940986013`;
- prior-token strict-history effect: `6.290625270674364`;
- edit controls, affine transition, injection, and post-update memory state:
  bit-identical under the current-token intervention.

All transition and injection tensors were finite. The maximum measured
transition spectral norm was `0.9995000000000012`, below the frozen
nonexpansive allowance of `1 + 1e-6`.

## Numerical execution contracts

| Execution contract | FP64 maximum | FP32 maximum | Discrete result |
|---|---:|---:|---|
| Model logits across parallel/recurrent/chunk/step views | `2.775558e-16` | `8.940697e-8` | predictions exact |
| Convolution or memory state across all declared paths | `1.665335e-16` | `5.960464e-8` | pass |

The FP64 bound was `1e-10`; the FP32 logit bound was `5e-4`. Full-sequence,
arbitrary chunk, token-step, masked-hole, recurrent, and parallel paths all
passed without fallback or architecture substitution.

## Gradient reach and test evidence

A real loss produced finite nonzero gradients for every declared path:
embedding, local-history convolution, query, key, value, commit, scalar erase,
channelwise write, retention, output gate, output projection, and LM head. The
smallest recorded nonzero maximum gradient was `2.604216e-10` on a retention
projection; embedding and LM-head maxima were `3.389165e-3` and
`2.380732e-3`.

The final Windows suite reported 405 passed and 6 skipped. Twenty focused WSL
tests passed, including the new transactional implementation and Phase-0
harness. The final post-harness full WSL suite reported 411 passed in 60.09
seconds.

## Decision and next boundary

The artifact's frozen decision is:

> `authorize prospective G15B-T Phase-1 constructed training`

Phase 1 must use fresh seeds 2381, 2383, and 2389 and the frozen matched
`F/T/T-AUX` schedule. Only a passing `T` arm with causal-use interventions may
advance to a separately frozen natural-text/scaling protocol. `T-AUX` remains
an unequal-objective diagnostic and cannot promote the architecture.

## Nonclaims

Phase 0 does not establish successful optimization, learned commit timing,
overwrite quality, generic association, natural-text robustness, longer-
context recall, tokenizer or optimizer superiority, efficiency, a Spin
advantage, or model-family promotion. Equal declared state capacity is not a
measured throughput result. R5 and R5-S remain failed under their own frozen
adjudications.
