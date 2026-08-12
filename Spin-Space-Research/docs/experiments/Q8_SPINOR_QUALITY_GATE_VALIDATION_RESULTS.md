# Q8 spinor discovery/retraction/quality-gate validation

Date completed: 2026-08-03. Seeds 10--19 were untouched while the `0.10`
manifold-distance rule was selected on seeds 0--9.

## Prospective result

The complete write-free pipeline passes every preregistered gate in all ten
fresh validation seeds:

```text
unconstrained tangent training
    -> joint four-token Q8 frame retraction on every channel
    -> label-free decoder gating by pre-retraction manifold distance
```

| Variant on seeds 10--19 | Standard dense pass | Exact algebra pass |
|---|---:|---:|
| Raw learned spinor | 8/10 | 0/10 whole-model |
| All-channel joint retraction | 8/10 | 10/10 |
| Retracted + frozen quality gate | **10/10** | **10/10** |

Raw seeds 12 and 17 fail behaviorally. The joint retraction makes their action
families exact but, as prospectively anticipated, does not by itself remove
decoder interference from formerly nuisance channels. The quality gate removes
only those decoder columns and rescues both without labels or retraining.

## Frozen rule generalizes

The validation-cohort projection distances remain separated around the frozen
threshold:

| Channel class under the fixed rule | Count | RMS range |
|---|---:|---:|
| Retained (`<=0.10`) | 24/40 | 0.000594--0.02636 |
| Rejected (`>0.10`) | 16/40 | 0.13088--0.69754 |

No threshold lies close to an observed validation channel. Every seed retains
two or three channels. The rule sees only action-family distance to the exact
Q8 manifold; it never sees endpoints, predictions, margins, decoder gradients,
or channel-ablation accuracy.

## Full dense and long gates

- All 460 seed/length cells in the 46-length matched odd/even dense sweep are
  100% pair-member and 100% both-members-correct.
- All 40 seed/length cells at base lengths 4095, 4096, 16383, and 16384 are
  100% pair-member and 100% both-members-correct.
- Central state separation is `2.00006`--`2.00080`, the numerical sign-flip
  distance.
- Maximum full/chunk/token logit discrepancy is `2.86e-6`; recurrent state
  parity is exact.

## Exact mechanism

Across all ten retracted validation checkpoints:

| Metric | Range |
|---|---:|
| Whole-model homomorphism RMS | `1.55e-7`--`1.68e-7` |
| Whole-model homomorphism max | `3.06e-7`--`4.24e-7` |
| Inverse-pair antipodality RMS | `7.55e-8` |
| Generator-square-to-`-1` RMS | `7.90e-8`--`8.89e-8` |
| Generator anticommutator RMS | `9.84e-8`--`1.24e-7` |

Thus this is not merely a decoder-level long-sequence pass. The persistent
state transition is numerically faithful to the Q8 spinor relations within the
reported float32 residuals, and the decoder uses only channels that SGD placed
near that shared manifold before retraction. “Exact” refers to the abstract
compiled group action, not to zero residual in its floating-point evaluation.

## What is proved—and what is not

The result validates a reliable controlled pipeline for discovering, compiling,
and using noncommutative center-faithful recurrent actions from endpoint-only
supervision. In this ten-seed cohort, raw SGD alone was not 10/10 reliable; the
exact compiler and representation-quality gate were essential components of
the validated pipeline. This finite cohort is evidence about the tested
protocol, not a theorem about every optimizer or budget.

This does not yet prove a language-model advantage, a Spin(8) triality benefit,
or superiority to a fully optimized generic orthogonal baseline across seeds.
The GA and Householder comparison is currently a controlled seed-0 separation.
Q8 is the adversarial center-fidelity gate that justifies proceeding to the
next representation, not the final application benchmark.

Artifacts:

- `q8_spinor_center_validation_seeds10_19.json`
- `q8_spinor_joint_retraction_validation_seeds10_19.json`
- `q8_spinor_quality_gate_validation_seeds10_19.json`
- `q8_spinor_quality_gate_validation_dense_seeds10_19.json`
- `q8_spinor_quality_gate_validation_long_seeds10_19.json`
- `q8_spinor_quality_gated_validation_seeds10_19_checkpoints/`
- `Q8_SPINOR_QUALITY_GATE_PREREGISTRATION.md`
