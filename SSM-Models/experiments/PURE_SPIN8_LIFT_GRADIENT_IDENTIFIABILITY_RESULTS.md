# Pure Spin(8) lift-gradient identifiability certificate

Status: **PASS**, exact optimizer-trace certificate on frozen seeds 4--6<br>
Recorded: **2026-08-17T08:28:34.954333+02:00**

## Question

Does the adaptive-calibration advantage over the independent `SO(8)^3` control
reflect poor optimization, or is an unsupervised representation-specific
parameter block outside the loss graph?

## Architecture result

The independent control maps a shared nine-unit observation trunk into 84
coordinates arranged as three disjoint 28-coordinate heads:

`(c_v, c_+, c_-)`.

The adaptive endpoint loss reads the full final vector block and one addressed
positive-spinor scalar. It never reads the negative-spinor prediction.
Consequently:

- the vector and positive head rows receive direct data gradients;
- the negative-specific 28-coordinate head receives exactly zero data
  gradient for every batch;
- the generic shared observation trunk still receives gradients, so negative
  predictions can change indirectly and are not frozen;
- the shared Pure Spin(8) router has one 28-coordinate head, and all 28 rows
  receive data gradients because the same bivector coordinates generate every
  triality view.

This is a parameter-identifiability statement, not merely a loss comparison.

## Automatic gradient audit

The exact first frozen batch was regenerated for each seed. Every one of the
252 vector-head and 252 positive-head weights in the independent control has a
nonzero gradient; all 252 negative-head weights and all 28 negative-head biases
have bitwise-zero gradient. All 108 shared-trunk weights have nonzero gradient.
All 28 rows of the shared Spin(8) coordinate head have nonzero gradient.

| Seed | independent vector-head L2 | positive-head L2 | negative-head L2 | shared Spin(8) minimum row L2 |
|---:|---:|---:|---:|---:|
| 4 | `0.128459` | `0.176374` | **`0.0`** | `0.006930` |
| 5 | `0.051650` | `0.062117` | **`0.0`** | `0.006542` |
| 6 | `0.069622` | `0.074306` | **`0.0`** | `0.008871` |

## Full optimizer-trace certificate

The frozen optimizer is AdamW with learning rate `0.003`, weight decay
`0.0001`, and 2,000 updates. A parameter slice with zero data gradient is
therefore multiplied only by

`(1 - 0.003 * 0.0001)^2000 = 0.9994001798741531`.

For each seed, the validator reconstructed the exact candidate initialization,
repeated the float32 decay operation 2,000 times, and compared it with the
strictly rehashed final checkpoint.

| Seed | negative-weight residual from decay-only | negative-bias residual | vector residual | positive residual |
|---:|---:|---:|---:|---:|
| 4 | **`0.0`** | **`0.0`** | `1.415161` | `1.327333` |
| 5 | **`0.0`** | **`0.0`** | `1.923446` | `1.524824` |
| 6 | **`0.0`** | **`0.0`** | `1.583675` | `0.673523` |

The negative-specific weights change from initialization by at most
`8.9407e-5`, exactly the prescribed decay. The vector and positive blocks differ
from their decay-only counterfactual by orders of magnitude, confirming direct
data updates.

## Consequence for the earlier result

The independent negative-view gap cannot be repaired by merely extending the
same adaptive-loss training budget: no number of identical updates introduces
a direct negative-head gradient. A longer run could still improve the shared
trunk and the directly supervised positive block, so the seed-6 positive-bit
optimization caveat remains. It is no longer accurate to treat the
negative-view result as an undifferentiated optimization failure.

That matched falsifier was executed with a **single shared latent coordinate
head**, the same 24-scalar state and router initialization, and independently
trainable positive/negative Spin(8) alignments. The full-supervision capability
gate passes, but the pre-frozen universal all-view dominance gate fails on two
directly supervised seed-7 vector-L128 cells. Preserving the failure, correct
triality alignment wins every action, spinor-L128, and completely hidden
negative-L128 comparison. See
[`PURE_SPIN8_SCRAMBLED_ALIGNMENT_RESULTS.md`](PURE_SPIN8_SCRAMBLED_ALIGNMENT_RESULTS.md).

## Claim boundary

This certificate proves direct gradient separation for the implemented model
and frozen loss, and exactly explains the final negative-head parameter trace.
It does not prove a global optimizer theorem, say that the negative prediction
is constant, resolve positive-channel optimization, or by itself establish
that exact triality rather than generic coefficient sharing causes transfer.
The executed scrambled-alignment control supplies bounded evidence for the
cross-view spinor stratum, while explicitly failing universal all-view
dominance.

## Artifact

- certificate SHA-256:
  `dee7c22e94bd627704609b7cad58939532e11c583a483e0e73987149f9339ab5`

Reproduce with
[`analyze_pure_spin8_lift_gradient_identifiability.py`](../analyze_pure_spin8_lift_gradient_identifiability.py).
