# G15B-E Phase-1 Result Ledger

**Protocol:**
[`G15BE_EFFECTIVE_EDIT_PROTOCOL_2026-08-26.md`](G15BE_EFFECTIVE_EDIT_PROTOCOL_2026-08-26.md)

**Phase-0 result:**
[`G15BE_PHASE0_QUALIFICATION_RESULTS.md`](G15BE_PHASE0_QUALIFICATION_RESULTS.md)

**Runner:**
[`g15be_effective_edit_cohort.py`](g15be_effective_edit_cohort.py)

**Current status:** exact clean-SM75 execution smoke passed; quality cohort
pending. No G15B-E learning result exists yet.

## Exact SM75 execution smoke

The sealed runner executed from clean commit
`475221e20feb8b0f1e5ed04ae4b1a28a935c52be` under WSL2 on the NVIDIA GeForce
RTX 2070 SUPER, exact compute capability `(7,5)`, Python 3.11.16, PyTorch
2.9.0+cu128, and CUDA 12.8. The artifact is
[`artifacts/g15be_phase1_smoke_sm75_2026-08-26.json`](artifacts/g15be_phase1_smoke_sm75_2026-08-26.json),
SHA-256
`ac5829d23625a0bc111ee19b5c1ef3d6f5dcd63b6e9a3324121c3220bd36ebe2`.

Preflight binds the passing Phase-0 hash and unchanged `model.py` /
`transactional_delta.py` source hashes. `P` and `A` have bit-identical initial
tensors, 67,033 parameters, 4,864 FP32 state bytes, complete optimizer
partition, finite nonzero gradients, and exact reconstructed forward logits.
Both completed four deliberately tiny updates, 3,840 tokens, all 20 ordinary
cells, the A intervention cells, external checkpoint hashing, and the trained
boundary audit. Maximum observed chunk/masked-step logit residual is
`3.576e-7`; maximum state residual is `8.941e-8`; predictions are exact.

The smoke adjudicator returns `passed=false` and
`eligible_for_promotion=false` by construction. Its zero boundary accuracy and
untrained task metrics are expected after four updates and are not learning
evidence. Only the unchanged three-seed quality cohort can adjudicate P/A.

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
