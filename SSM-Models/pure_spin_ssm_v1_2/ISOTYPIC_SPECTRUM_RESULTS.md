# Static isotypic-spectrum result

**Closed:** 2026-08-22

## Verdict

The static continuous-time isotypic spectrum lost all three frozen seeds and
is not promoted.

| seed | shared retention bpb | static spectrum bpb | improvement |
|---:|---:|---:|---:|
| 283 | **2.689336** | 2.708272 | -0.018937 |
| 293 | **2.671825** | 2.688723 | -0.016898 |
| 307 | **2.700452** | 2.705528 | -0.005076 |
| mean | **2.687204** | 2.700841 | -0.013637 |

Every common parameter and initial logit was bitwise equal. The candidate
added only 24 static log-rate parameters and remained inside the single-seed
safety bound, but it failed both the win-count and mean criteria. No speed gate
is authorized.

## Consequence

Combining this result with the dynamic isotypic-retention gate closes a useful
branch:

- dynamic sector timescales: 2/3 wins, only `+0.00242` mean bpb;
- static sector spectrum: 0/3 wins, `-0.01364` mean bpb.

The retained state decomposition is mathematically valid and the CUDA support
is reusable, but retention allocation alone is not the missing mechanism. The
next architecture must change what memory can overwrite and retrieve. It
should not add another action algebra, readout invariant, or decay
parameterization.

The next design is specified in
[`SPIN_DELTA_SUCCESSOR_DESIGN.md`](SPIN_DELTA_SUCCESSOR_DESIGN.md).

## Evidence

- [`artifacts/isotypic_spectrum_quality_summary.json`](artifacts/isotypic_spectrum_quality_summary.json)
- `artifacts/isotypic_spectrum_seed_{283,293,307}.json`
- [`ISOTYPIC_SPECTRUM_PREREGISTRATION.md`](ISOTYPIC_SPECTRUM_PREREGISTRATION.md)
