# Spin-Delta pairing amendment

**Frozen:** 2026-08-22 before natural-data training of replacement seeds
373/379/383.

## Why the first cohort is invalid

The original preregistration froze a maximum initial-logit difference of
`1e-6` from one full-size random-input probe. Seed 353 then completed its pair,
but seed 359 was rejected before training because its deterministic validation
pairing residual was `1.0728836059570312e-06`. Common parameters were still
bitwise identical and the mean absolute residual was only
`8.404941098660856e-08`; this was a one-ulp-scale float32 reduction difference,
not an architectural mismatch.

Because seed 353's quality result was already visible (`+0.012685655305954668`
bpb), changing the tolerance and continuing the same cohort would move a gate
after observing an outcome. Seed 353 is therefore quarantined as commissioning
evidence and cannot count toward promotion. Its preserved artifact is
`artifacts/spin_delta_gate/seed_353.json`, SHA-256
`9d6370404536fbc82603c61bcab95d619b6e45b7b3c7da274add7a80f33b746a`.
Seeds 359 and 367 were never trained.

## Corrected frozen pairing contract

The replacement tolerance is `2e-6`. This is a compatibility guard, not a
quality threshold. It is larger than every characterized residual while still
being tiny relative to initial logits. Before any replacement-seed training,
their deterministic `B=8`, `L=256` validation probes measured:

| Seed | Maximum absolute residual | Mean absolute residual |
|---:|---:|---:|
| 373 | `9.5367431640625e-07` | `8.55644728403604e-08` |
| 379 | `9.834766387939453e-07` | `8.78850698882161e-08` |
| 383 | `9.5367431640625e-07` | `8.671352702549484e-08` |

The replacement quality cohort is seeds 373, 379, and 383. Every other frozen
condition and promotion threshold from `SPIN_DELTA_PREREGISTRATION.md` remains
unchanged:

- fixed variant order and identical batches;
- 300 updates, batch 8, length 256, and 16 validation batches;
- at least two wins;
- mean improvement at least `+0.0100` bpb;
- no regression worse than `-0.0500` bpb;
- bitwise-equal common parameters and finite compatible artifacts.

Only this replacement cohort may decide promotion or authorize a speed gate.
