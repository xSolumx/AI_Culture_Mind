# Spin-Delta temporal observability audit

**Protocol status:** frozen before full-model measurements.

## Question

Can final-only retrieval loss observe the desired tokenwise query grammar, and
how much of that observability is direct versus induced by stacking blocks?

## Frozen design

- fresh seeds `761`, `769`, `773`;
- batch 64, eight writes, one fixed batch per seed;
- otherwise unchanged `d_model=64` Spin-Delta models with either one or two
  blocks;
- bitwise-paired hard fallback, soft query event, and authoritative routed
  query paths;
- no optimizer update;
- final retrieval cross-entropy only;
- per-position gradients with respect to query-event and query-slot logits;
- role aggregates for write marker, write key, write value, query marker, and
  final query key;
- gradient-descent alignment with the desired event-off/nonfinal and
  event-on/final grammar, plus the final correct-slot margin.

## Frozen decisions

1. **One-block structural identity** requires event and slot gradients at every
   non-final position to be at most `1e-12` in all paths and seeds.
2. **Final-path topology** requires hard fallback to retain final event credit
   above `1e-8` but zero final slot credit, while soft and authoritative paths
   restore final slot credit above `1e-8`.
3. **Stack-induced observability** requires the two-block soft path to have
   non-final event and slot gradient above `1e-10` in every seed.
4. **Hard-slot dead path** requires hard fallback query-slot gradient at every
   position and both depths to remain at most `1e-12` while all hard query
   events are forward-zero.
5. **Grammar-aligned credit** requires, in every two-block soft seed, at least
   90% of non-final event-gradient absolute mass to point toward event-off,
   at least 90% of final event-gradient mass to point toward event-on, and a
   positive gradient-descent margin for the correct final query slot.

These decisions are independent. Indirect nonzero derivatives do not by
themselves establish useful grammar supervision.

## Boundary

This is an initialization Jacobian audit on a finite synthetic task. It is not
a training, natural-data, or general identifiability theorem.
