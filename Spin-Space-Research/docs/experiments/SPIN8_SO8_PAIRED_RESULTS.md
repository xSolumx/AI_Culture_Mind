# Positive-half-spin versus generic SO(8) paired cohort: results

Date completed: 2026-08-03.

The preregistered five-seed cohort completed without protocol failures. Both
families used 4,488 trainable parameters, including 448 action parameters, and
had identical initial functions under each paired seed. All streaming checks
passed and every final curriculum batch reached 100% accuracy.

## Raw composition result

| Family | Final curriculum fit | Dense gate | Mean dense minimum | Mean homomorphism RMS |
|---|---:|---:|---:|---:|
| Positive half-spin | 5/5 | 0/5 | 0.000684 | 0.622027 |
| Generic SO(8) exponential | 5/5 | 0/5 | 0.000000 | 0.578429 |

Generic SO(8) had lower homomorphism RMS in three seeds and higher RMS in two.
The paired generic-minus-positive differences were:

```text
-0.0652, +0.0608, -0.1524, -0.1479, +0.0868
```

The mean difference was `-0.0436`; at five seeds with mixed signs this is not a
reliability claim. Both charts exhibit the same qualitative failure: excellent
short endpoint fit and catastrophic raw long-composition drift.

## Interpretation

The exact algebra audit proves equal transition capacity. The SGD/AdamW causal
audit proves that adaptive optimization can separate the charts despite that
equality. This cohort then shows no stable behavioral advantage for either chart
under the frozen AdamW protocol.

Therefore the existing Q8 result should be attributed to learning dense
orthogonal actions and compiling/retracting them—not to a uniquely positive-
spinorial single-stream transition family.

The remaining genuinely Spin(8)-specific hypotheses are:

1. differing global center kernels on center-sensitive tasks;
2. a shared controller acting jointly on vector and both chiral streams;
3. an explicit invariant triality coupling that no isolated SO(8) vector stream
   possesses.

Artifacts:

- `spin8_so8_paired_seeds60_64.json`
- `spin8_so8_paired_seeds60_64_analysis.json`
- `spin8_so8_paired_checkpoints/`
