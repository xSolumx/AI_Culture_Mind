# G15B-D Phase-0 qualification results

**Frozen protocol:**
[`G15BD_RESIDUAL_DELTA_PROTOCOL_2026-08-26.md`](G15BD_RESIDUAL_DELTA_PROTOCOL_2026-08-26.md)

**Qualification harness:**
[`g15bd_phase0_qualification.py`](g15bd_phase0_qualification.py)

**Exact artifact:**
[`artifacts/g15bd_phase0_qualification_sm75_2026-08-26.json`](artifacts/g15bd_phase0_qualification_sm75_2026-08-26.json)

**SHA-256:**
`44a8556b60db7cb8c5e1edc239255dc510b62f03960e4514a49b645e79123921`

**Execution commit:** `549e6d98d0bebc35fad32daa498486fd075aa906`

**Status:** pass; authorize only prospective Phase-1 constructed training

## Bottom line

G15B-D passed every frozen Phase-0 matching, coupling, direct-formula,
contraction, numerical-parity, gradient, provenance, and exact-SM75 gate. The
matched product `P` and coupled residual-delta `D` arms each have 22,161
total/active parameters, 1,408 FP32 state bytes, and the same initialized
parameter hash.

This is an implementation qualification, not a learning result. It authorizes
only the already prospective constructed Phase-1 `P/D` cohort. No trained
G15B-D model, natural-text, causal-use, scaling, optimizer, geometry, or model
promotion claim follows.

## Exact execution

The evidentiary artifact ran from a clean tree under WSL2 on an NVIDIA GeForce
RTX 2070 SUPER with exact compute capability `(7,5)`, Python 3.11.16, PyTorch
2.9.0+cu128, and CUDA 12.8. Qualification took `2.1162893` seconds and peaked
at 32,148,480 CUDA bytes.

The initialized matching contract passes:

- `P/D` total and active parameters: `22,161 / 22,161` each;
- `P/D` FP32 recurrent state: `1,408` bytes each;
- initialized parameter hashes: identical;
- effective-gate maximum residual: `1.862645e-9`;
- coupled `D` erase/write residual: exactly `0`;
- initial model-logit residual: `2.980232e-8`;
- initial state residual: `3.259629e-9`;
- initial predictions: exact.

## Coupling and contraction

The independent direct residual-delta construction agrees exactly with the
compiled affine law:

- direct transition residual: `0`;
- direct injection residual: `0`;
- coupling residual: `0`.

The maximum measured `D` transition spectral norm is
`0.9995000000000026`, inside the frozen nonexpansive bound. All recorded
strengths, transition tensors, injections, states, reads, and logits are
finite.

## Numerical execution contracts

Maximum FP64 residuals are:

| Surface | Maximum residual |
|---|---:|
| model logits | `2.775558e-16` |
| recurrent state | `1.110223e-16` |
| memory read | `3.469447e-18` |

Maximum FP32 residuals are:

| Surface | Maximum residual |
|---|---:|
| model logits | `8.940697e-8` |
| recurrent state | `5.960464e-8` |
| memory read | `1.396984e-9` |

All compared predictions are exact. Full, parallel, recurrent, arbitrary
chunk, token-step, masked, and compact-valid execution satisfy their frozen
contracts.

## Gradient and decision boundary

A real language-model loss produces finite nonzero gradients for every
declared query, key, value, shared-edit, erase, write, retention, readout,
convolution, embedding, and LM path.

The frozen decision is:

> authorize only prospective G15B-D Phase-1 constructed training

Phase 0 does not show that residual delta learns overwrite, improves on `P`,
uses only valid events, transfers to natural text, scales to longer contexts,
or benefits from geometry. Those claims remain untested.
