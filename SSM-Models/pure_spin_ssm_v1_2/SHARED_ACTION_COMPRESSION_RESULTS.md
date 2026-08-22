# Shared-action compression result

**Closed:** 2026-08-22

## Verdict

The shared-action identity recurrence is not quality-noninferior to established
Pure Spin v1.2 on the frozen Tiny Shakespeare gate. It remains a correct
compiler control, but it is not promoted and its conditional throughput gate
was not run.

| seed | independent v1.2 bpb | shared action bpb | improvement |
|---:|---:|---:|---:|
| 179 | **2.683078** | 2.696556 | -0.013478 |
| 181 | **2.650261** | 2.695317 | -0.045056 |
| 191 | **2.698908** | 2.712834 | -0.013926 |
| mean | **2.677416** | 2.701569 | -0.024153 |

Positive improvement is independent minus shared-action. The candidate lost
all three seeds. Mean regression exceeded the frozen `0.0100` bpb
non-inferiority margin, although no individual seed crossed the `0.0500` hard
regression bound. Because quality failed, the order-balanced speed campaign was
not authorized. Sequential timers in the training artifacts are diagnostic
only and also happened to favor the established model in all three runs.

## Interpretation

The 6,708 removed controller parameters are only 1.071% of the established
model, but tying them removes an independent sequence of Spin actions from each
block. The consistent quality loss is evidence that the second controller
provides useful dynamical diversity rather than redundant parameterization at
this scale and training horizon.

Together with the recurrent-mixing gate, this closes the simplest Schur-style
route:

- sharing the Spin action makes multiplicity mixing algebraically closed but
  loses quality;
- adding learned `SO(2)` mixing inside that shared-action family does not
  recover the loss and has unfavorable mean effect within its own paired gate;
- therefore neither shared action nor recurrent multiplicity mixing belongs in
  maintained v1.2.

The more credible future direction must preserve independently learned channel
actions. Any cross-channel recurrence then needs either a general block-affine
compiler or an explicitly transported/gauge-covariant low-rank coupling; it
cannot inherit the cheap Kronecker closure of the shared-action candidate.

The machine-readable verdict is
[`artifacts/shared_action_compression_summary.json`](artifacts/shared_action_compression_summary.json),
and the frozen protocol is
[`SHARED_ACTION_COMPRESSION_PREREGISTRATION.md`](SHARED_ACTION_COMPRESSION_PREREGISTRATION.md).
