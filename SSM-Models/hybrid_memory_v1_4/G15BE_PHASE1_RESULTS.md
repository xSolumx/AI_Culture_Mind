# G15B-E Phase-1 Result Ledger

**Protocol:**
[`G15BE_EFFECTIVE_EDIT_PROTOCOL_2026-08-26.md`](G15BE_EFFECTIVE_EDIT_PROTOCOL_2026-08-26.md)

**Phase-0 result:**
[`G15BE_PHASE0_QUALIFICATION_RESULTS.md`](G15BE_PHASE0_QUALIFICATION_RESULTS.md)

**Runner:**
[`g15be_effective_edit_cohort.py`](g15be_effective_edit_cohort.py)

**Exact quality artifact:**
[`artifacts/g15be_phase1_quality_sm75_2026-08-26.json`](artifacts/g15be_phase1_quality_sm75_2026-08-26.json)

**SHA-256:**
`2a41ed4694c4b2df473e08e6a62e455a309ffa7106616ffdfe420346eee4b469`

**Current status:** Phase-1 quality failed for both `P` and `A`; test a
separately frozen residual-delta write law; geometry remains blocked.

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
smoke artifact cannot promote either arm. The quality output below does not
delete or relabel G15B-T's failed result.

## Exact SM75 quality result

The sealed quality cohort completed from clean commit
`c2f160cf7c9a7824dc9c04de49495ad6922fce64` under WSL2 on the NVIDIA GeForce
RTX 2070 SUPER, exact compute capability `(7,5)`, Python 3.11.16, PyTorch
2.9.0+cu128, and CUDA 12.8. The artifact is evidentiary and contains all six
seed/arm reports:

- fresh seeds `2481`, `2483`, and `2489`;
- matched product `P` and logit-additive `A` arms;
- 3,400 updates and 13,926,400 training tokens per report;
- 20,400 updates and 83,558,400 training tokens in total;
- 67,033 total/active parameters and 4,864 FP32 state bytes per sequence for
  every arm.

Preflight, exact-SM75 provenance, phase-0 source binding, optimizer partition,
finite/nonzero gradients, learned-forward reconstruction, paired schedules,
fingerprint disjointness, complete cells, numerical finiteness, and all six
external checkpoint hashes pass. No G15B-T checkpoint or optimizer state was
loaded.

### Ordinary overwrite

Three-seed mean query accuracy is:

| Length | Product `P` | Additive `A` | `A - P` |
|---:|---:|---:|---:|
| 128 | 0.926595 | 0.918132 | -0.008464 |
| 512 | 0.922689 | 0.928223 | +0.005534 |
| 1,024 | 0.923340 | 0.924967 | +0.001628 |
| 2,048 | 0.921875 | 0.918620 | -0.003255 |

`A` never reaches the prospectively frozen `+0.02` mean margin at L128,
L512, or L1024. It is slightly above `P` at L512/L1024 and below it at
L128/L2048; this is not evidence that either parameterization dominates
generally.

Both absolute-quality conjunctions fail. `P` misses per-seed overwrite and
post-same-key gates. `A` seed 2481 misses overwrite/post-same-key throughout
the promoted lengths, while seed 2489 also misses overwrite/post-same-key and
the L128 MQAR threshold. The additive L128 worst-seed robustness gate fails.
Needle and constructed-guard cells pass, but conjunctive promotion cannot
discard the failed ordinary cells.

Ordinary overwrite retains `null` for unsupported strata. Constructed-guard
strata are not relabelled as ordinary-overwrite evidence.

### Addressing and causal use

Trained address top-1 remains strong, but the declared objective explicitly
contains value-position address supervision. It is a trained-address
diagnostic, not unsupervised generic-association evidence.

The additive arm passes memory-zero, valid-event-edit-zero, binding
permutation, non-event-only failure, and most erase interventions. It still
fails the frozen causal conjunction:

- valid-event-only execution is not within `0.02` of learned execution for
  seeds 2483 and 2489 across MQAR, overwrite, and selective copy at L512 and
  L1024;
- seed 2489 fails erase-zero MQAR preservation at L512 and L1024.

Therefore the learned score cannot be attributed solely to the declared valid
WRITE/SELECT value-position edits. Per the frozen protocol, a causal-use
failure stops promotion.

Every trained boundary audit passes its prospectively scaled FP32 bounds with
exact predictions. The negative decision is a learning/causal-use result, not
a numerical-parity or runtime failure.

## Frozen adjudication and boundary

The artifact records:

```text
product_absolute_quality_passed = false
additive_absolute_and_causal_passed = false
passed = false
eligible_for_promotion = false
decision = both effective-edit arms fail; test a frozen residual-delta write law
```

The logit-additive parameterization did not materially improve the matched
constructed learning problem. The next residual-delta write law is only a
prospective recommendation and requires its own frozen protocol before any
training.

This does not establish that all effective edits, residual-delta memories, or
transactional memories fail. It provides no natural-text, longer-context
pretraining, tokenizer, optimizer, efficiency, scaling-law, attention, torus,
Spin, or model-family promotion evidence. Geometry remains blocked.
