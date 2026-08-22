# Dynamic isotypic-retention result

**Closed:** 2026-08-22  
**Verdict:** failed the frozen mean-improvement threshold; shared retention
remains the maintained default.

## Frozen result

The candidate gave the inequivalent `8v`, `8+`, and `8-` sectors centered,
token-dependent retention offsets while preserving both independent Spin
controllers. Every common parameter and initial logit was bitwise paired.

| seed | shared retention bpb | isotypic retention bpb | improvement |
|---:|---:|---:|---:|
| 271 | 2.812292 | **2.787002** | +0.025290 |
| 277 | 2.704332 | **2.687651** | +0.016681 |
| 281 | **2.701912** | 2.736628 | -0.034716 |
| mean | 2.739512 | **2.737094** | +0.002418 |

The candidate won 2/3 and stayed inside the `-0.0500` single-seed safety
bound, but its mean improvement was below the frozen `+0.0100` requirement.
Quality promotion failed and no systems gate is authorized.

## What this localizes

The two favorable seeds are evidence that the previous shared retention was a
meaningful model restriction. The adverse third seed shows that asking every
token to produce an unconstrained three-sector split is not a stable successor
at this budget. This is not evidence for adding more transport or readout
features.

The next minimal hypothesis is a continuous-time isotypic spectrum. Keep the
token-dependent shared step but learn static positive sector rates:

```text
s_c,r(x) = s_c(x) ** lambda_c,r,
lambda_c,r > 0.
```

This mirrors a selective step size multiplied by learned state decay rates,
uses only a few static parameters, and preserves the same equivariance and
boundedness. It requires a new prospective gate; the current artifacts cannot
be reused as confirmation.

## Evidence

- [`artifacts/isotypic_retention_quality_summary.json`](artifacts/isotypic_retention_quality_summary.json)
- `artifacts/isotypic_retention_seed_{271,277,281}.json`
- [`ISOTYPIC_RETENTION_PREREGISTRATION.md`](ISOTYPIC_RETENTION_PREREGISTRATION.md)
