# Pure Spin(8) adaptive lift-bit calibration results

Status: **PASS**, frozen fresh cohort, no median rescue<br>
Protocol frozen: **2026-08-17T07:05:22.8794763+02:00**<br>
Fresh cohort adjudicated: **2026-08-17T07:32:36.941651+02:00**<br>
Documentation reconciled: **2026-08-17T07:34:13.2082201+02:00**

## Result in one sentence

On the fixed injective synthetic endpoint task, a max-coordinate chart address
plus one lift-odd sign bit repairs every frozen vector-only lift failure for the
shared Pure Spin(8) router through length 128; the result is an exact statement
about the two-element cover fiber and a replicated empirical statement about
this model/task, not recovery from vector data alone.

## Exact fiber result

Let the two lifts over one quotient state be `{y,-y}`, with `y` a unit vector
in the eight-real positive half-spin representation. Define

`j(y) = argmax_k |y_k|`, `b(y) = sign(y_j(y))`,

using deterministic first-index tie-breaking. Then:

1. `j(-y)=j(y)`, because absolute values are unchanged;
2. `|y_j| >= 1/sqrt(8)`, because `sum_k |y_k|^2=1`;
3. the selected coordinate is therefore nonzero and `b(-y)=-b(y)`;
4. one lift-odd bit is necessary to distinguish a two-element fiber and this
   bit is sufficient once the lift-invariant chart address is known.

The calibration word contains **four transmitted bits**: a three-bit address
and one sign bit. Only the sign bit contains lift information. Calling the
whole interface “one total bit” would be false. The max chart also is not a
global continuous section: its address can jump at equal-magnitude chart
boundaries, although deterministic tie-breaking preserves `j(-y)=j(y)` and
the selected-amplitude lower bound.

## Frozen experiment

The protocol in
[`PURE_SPIN8_LIFT_BIT_CALIBRATION_PROTOCOL.md`](PURE_SPIN8_LIFT_BIT_CALIBRATION_PROTOCOL.md)
was frozen before seeds 4--6. It retains the previous endpoint teacher, noisy
seven-coordinate injective observation chart, excluded adjacent center
relation, L16 endpoint-only training, 2,000 updates, and L16/L64/L128 tests.

Every mode receives the same precomputed schedule and candidate-specific
initialization:

| Mode | Final supervision |
|---|---|
| `vector_only` | all eight `8v` scalars |
| `vector_plus_positive_bit` | `8v` plus one fixed-coordinate spinor sign |
| `vector_plus_adaptive_lift_bit` | `8v` plus max-chart address and sign |
| `positive_only` | all eight `8s+` scalars |
| `full_triality` | all 24 triality scalars |

The adaptive loss transfers only the final vector target, integer address, and
one Boolean sign target to the accelerator. A gradient contract verifies that
the loss touches only the final `8v` block and one addressed `8s+` scalar. It
does not transfer a hidden full-spinor target or any intermediate prefix
target. The shared router has 930 parameters; the exactly 24-state independent
`SO(8)^3` control has 957.

## Frozen adjudication

Every global integrity check and every one of the 13 seedwise gates passed.
No median or aggregate statistic rescues an individual row.

| Seed | shared action RMSE, all views | shared L128 MSE, all rows | L128 lift bit | minimum training margin |
|---:|---:|---:|---:|---:|
| 4 | `0.013630--0.013633` | `0.012094--0.026986` | `1.0 / 1.0` | `0.406697` |
| 5 | `0.012825--0.012827` | `0.009198--0.021578` | `1.0 / 1.0` | `0.399978` |
| 6 | `0.013297--0.013300` | `0.011457--0.023796` | `1.0 / 1.0` | `0.411755` |

Both half-spin center classifiers are exactly correct in every early/late L128
row. Every relation pair retains one chart address and opposite sign bits.
All observed training margins exceed the exact `1/sqrt(8)` bound.

The cross-seed aggregate gives:

| Mode / candidate | action RMSE range, all views | L128 MSE range, all views |
|---|---:|---:|
| shared vector-only | `0.049224--0.056667` | `0.046332--0.093163` |
| shared fixed sign probe | `0.064973--0.106815` | `0.058680--0.191420` |
| **shared adaptive calibration** | **`0.012825--0.013633`** | **`0.009198--0.026986`** |
| independent adaptive calibration | `0.065834--0.222239` | `0.068460--0.254302` |
| shared positive-only reference | `0.011750--0.012862` | `0.008841--0.022723` |
| shared full-triality reference | `0.011021--0.011860` | `0.007446--0.019649` |

Relative to identically initialized vector-only training, adaptive calibration
reduces median action RMSE by `73.4%--73.7%` across the three views and median
L128 MSE by `73.3%--78.0%`. It approaches, but does not equal, the eight-scalar
positive-only and 24-scalar full-triality references.

The naïve fixed-coordinate sign is a replicated negative control rather than a
weaker success: it is worse than vector-only in every aggregate action view and
has L128 errors up to `0.191420`. Its development schedules contained probe
magnitudes as small as `2.38e-6`; the adaptive chart removes that amplitude
conditioning failure.

## What the result establishes

- **Proved:** exactly one lift-odd bit is necessary and sufficient to select a
  member of a known two-element cover fiber; the max-coordinate chart gives
  selected amplitude at least `1/sqrt(8)`.
- **Empirical, frozen, replicated:** the address-plus-bit interface closes all
  tested vector-only hidden-lift failures for the shared 930-parameter Spin(8)
  router on seeds 4--6, including excluded relations and 8x length
  extrapolation.
- **Empirical architectural evidence:** shared Spin(8) beats the exactly
  24-state independent family in every frozen adaptive action and L128 view.
- **Negative:** an arbitrary fixed lift-odd probe is not enough for reliable
  optimization; chart conditioning matters.

## What remains open

This experiment does **not** infer the calibration word from the visible
eight-scalar vector endpoint. The address and bit are legal external targets;
physical sensing would need to supply or compute them. It does not construct a
global continuous section, test perturbations near equal-magnitude address
boundaries, cover noninjective or shifted input charts, train all 28 Lie
coordinates, use natural data, or compare fused optimized throughput with a
modern production SSM.

The independent adaptive control ends one seed's final minibatch at bit
accuracy `0.875`, although it passes the pre-frozen absolute vector-capability
gates. The subsequent exact gradient audit shows that its negative-specific
head receives zero data gradient and exactly follows the AdamW decay-only
counterfactual; a longer identical run cannot directly identify that head. See
[`PURE_SPIN8_LIFT_GRADIENT_IDENTIFIABILITY_RESULTS.md`](PURE_SPIN8_LIFT_GRADIENT_IDENTIFIABILITY_RESULTS.md).

The stronger shared-latent scrambled-alignment falsifier was also executed. Its
pre-frozen universal all-view gate failed on two seed-7 vector-L128 cells, so
the earlier broad architectural interpretation is not retained. The narrower
non-rescued result is complete action, spinor-L128, and hidden-negative-L128
dominance for the correct alignment, with full-supervision capability in the
scrambled control. See
[`PURE_SPIN8_SCRAMBLED_ALIGNMENT_RESULTS.md`](PURE_SPIN8_SCRAMBLED_ALIGNMENT_RESULTS.md).

## Integrity and artifacts

The strict validator regenerates all three 1,024,000-observation training
schedules, all observation systems, adaptive addresses and bits, and all 18
evaluation schedules. It rehashes and strictly reloads all 30 checkpoints and
recomputes every reported metric. All sources, schedules, checkpoints, and
metrics pass.

- seed 4 SHA-256:
  `d25e56ac62912ab78c975c35194bae9b089dbfdf4cb013d31dc3389574970663`
- seed 5 SHA-256:
  `fd2e8206124d732708950fcdef5937cbe7409da0f654c122a8fa16f08ad18bad`
- seed 6 SHA-256:
  `07a3cda44449013e5047939f6443a8728006e4ff6cb466a5b46cc635d5b142b3`
- strict aggregate SHA-256:
  `fb89fd75d5aa7c3b16448844225baf838aeb4bdf40cb62ba1276f07ac7503b69`

Reproduce with
[`benchmark_pure_spin8_lift_bit_calibration.py`](../benchmark_pure_spin8_lift_bit_calibration.py)
and adjudicate with
[`validate_pure_spin8_lift_bit_calibration.py`](../validate_pure_spin8_lift_bit_calibration.py).
