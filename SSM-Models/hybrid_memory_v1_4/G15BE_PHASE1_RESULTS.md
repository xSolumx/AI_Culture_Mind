# G15B-E Phase-1 Result Ledger

**Protocol:**
[`G15BE_EFFECTIVE_EDIT_PROTOCOL_2026-08-26.md`](G15BE_EFFECTIVE_EDIT_PROTOCOL_2026-08-26.md)

**Phase-0 result:**
[`G15BE_PHASE0_QUALIFICATION_RESULTS.md`](G15BE_PHASE0_QUALIFICATION_RESULTS.md)

**Runner:**
[`g15be_effective_edit_cohort.py`](g15be_effective_edit_cohort.py)

**Current status:** runner implemented and semantic smoke contracts pass; exact
clean-SM75 Phase-1 execution smoke and quality cohort pending. No G15B-E
learning result exists yet.

## Frozen comparison

The cohort trains fresh matched product `P` and logit-additive `A` arms on
seeds `2481`, `2483`, and `2489`. Both use the full causal value-position view,
67,033 parameters, 4,864 FP32 state bytes, the same HarmonicMuonAdamW
partition, hash-identical `128 -> 256 -> 512 -> 1024` batches, and the same
retrieval, reverse-binding, and value-position address losses. There is no
timing BCE, parser label, attention, geometry, checkpoint reuse, or optimizer
change.

The sealed runner records:

- MQAR, overwrite, constructed guard, selective copy, and needle at L128,
  L512, L1024, and L2048;
- ordinary-overwrite strata without filling unsupported cells;
- effective erase/write count, mean, standard deviation, minimum, and maximum
  on valid event and non-event positions;
- trained address top-1, state/read maxima, BPQ, speed, memory, schedules,
  fingerprints, source hashes, and external checkpoint hashes;
- A-only memory-zero, valid-event-edit-zero, erase-zero, permuted-binding,
  valid-event-only, and non-event-only causal interventions;
- trained chunk/masked-step numerical parity against the prospectively scaled
  FP32 bounds with exact predictions.

The exact conjunctive thresholds and decisions remain in the protocol. A
smoke artifact cannot promote either arm. Quality output will be documented
here without deleting or relabelling G15B-T's failed result.
