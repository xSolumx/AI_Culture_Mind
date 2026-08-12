# Spin(8) masked cross-representation completion results

Date executed: 2026-08-03. Protocol frozen in
`SPIN8_MASKED_COMPLETION_PREREGISTRATION.md`. Raw artifact:
`spin8_masked_completion_seeds0_2.json`.

## Frozen-gate result

The exact oracle and all three learned bilinear seeds pass. The MLP and linear
families do not approach the held-out completion threshold.

| Family | Parameters | Seeds | Worst mean cosine across all test cells | Worst MSE |
|---|---:|---:|---:|---:|
| exact triality | 0 | oracle | `1.0` | `9.87e-27` |
| learned bilinear | 512 | 3 | `0.999845` | `1.17e-4` |
| two-layer MLP | 608 | 3 | `0.165`--`0.342` | `0.114`--`0.163` |
| linear | 136 | 3 | `0.0010`--`0.0032` | `0.152`--`0.155` |

The bilinear tensors independently converge to cosine
`0.999906`, `0.999911`, and `0.999916` with the fixed octonionic triality
tensor. Their final training MSE values are `1.32e-5`, `1.18e-5`, and
`1.16e-5`.

The oracle was evaluated in float64 because the preregistered `1e-12` MSE gate
tests the algebra, not accumulated float32 transport error. Learned families
remain float32. An earlier diagnostic smoke exposed this distinction before
the full cohort; the gate was not loosened.

## What this establishes

For a unit positive spinor key, triality is a reversible binding map:

```text
s- -> V,       V[i] = s-^T rho[i] s+,
s- = sum_i V[i] rho[i] s+.
```

An unconstrained bilinear learner can rediscover this exact tensor from short
two-generator training words and then complete the masked negative spinor
under an unseen third generator through length 512. A similarly sized generic
MLP does not acquire the multiplicative law under the same budget.

## Post-result identifiability audit

The 64 bilinear input features in the training distribution have full numerical
rank and condition number `2.439`. Thus this cohort identifies the entire
tensor; it is not evidence that the Spin(8) prior extrapolates an
underdetermined law. The result supports bilinear/triality inductive bias over
the MLP, but the stronger symmetry-sample-efficiency claim needs a deliberately
rank-deficient training orbit.

## Claim boundary

The transport actions are supplied by the exact shared Spin(8) algebra. This
is a cross-representation binding gate, not a learned controller, memory
capacity, attention replacement, or language-model result.
