# Spin-Delta causal low-entropy router results

**Decision:** router identification passed; autonomous retrieval capacity and
robust rescue failed.

## Frozen result

The candidate learned a causal width-three token router with balanced
train-time event/slot supervision. Evaluation supplied tokens only. Its hard
forward decisions identified every write event, query event, write slot, and
query slot correctly in every one of the 36 measured seed/length/metric cells.

| Seed | Variant | W=8 | W=16 | W=32 |
|---:|---|---:|---:|---:|
| 449 | learned continuous | **93.99%** | **93.53%** | **94.24%** |
| 449 | causal discrete auxiliary | 78.81% | 78.91% | 78.86% |
| 457 | learned continuous | 88.01% | 87.92% | 88.01% |
| 457 | causal discrete auxiliary | **98.39%** | **98.46%** | **99.07%** |
| 461 | learned continuous | **99.78%** | **99.80%** | **99.83%** |
| 461 | causal discrete auxiliary | 99.10% | 99.10% | 98.95% |

Three-seed means were:

| Writes | Continuous | Causal discrete | Candidate change |
|---:|---:|---:|---:|
| 8 | 93.93% | 92.10% | -1.83 points |
| 16 | 93.75% | 92.15% | -1.60 points |
| 32 | 94.03% | 92.29% | -1.73 points |

The exact machine decision is
`artifacts/spin_delta_causal_router/summary.json`.

## Interpretation

The causal receptive-field hypothesis was correct but insufficient. A tiny
router can infer the complete synthetic grammar robustly and extrapolate that
identification from 8 to 32 writes. Therefore the final seed variance cannot
be attributed to failure to recognize write/query events or binary slots.

Seed 449 is the decisive falsifier: routing is exactly correct at evaluation,
yet retrieval remains near 79%. Seed 457 shows the same architecture can reach
99%, while seed 461 shows both old and new controllers can solve the task. The
failure has moved downstream from final address identification to the joint
optimization trajectory. Early hard-routing errors train the recurrence under
a changing, occasionally destructive write process; later-perfect routing
does not guarantee recovery within the frozen budget.

This explains why the oracle intervention was uniformly successful: its
controls were exact from the first optimizer step. The next controlled test is
phase separation, not a larger router. Train the router to its grammar labels
while leaving the core untouched, freeze it, then train the pristine recurrent
core. Success would localize the remaining failure to co-adaptation; failure
would expose a difference between internally produced straight-through
controls and the oracle path or a deeper optimization defect.

## Boundaries

The candidate adds 11,494 parameters and uses task-grammar labels during
training. It is not parameter matched, self-supervised, or a natural-language
model. Perfect synthetic routing is not evidence of event discovery on
Shakespeare. The result neither promotes Spin-Delta nor authorizes a speed
comparison.

Candidate training took 26.77--29.58 seconds versus 18.89--30.24 seconds for
the fixed-order continuous control. Peak allocation was 146,964,480 versus
145,654,784 bytes. These are diagnostics only.
