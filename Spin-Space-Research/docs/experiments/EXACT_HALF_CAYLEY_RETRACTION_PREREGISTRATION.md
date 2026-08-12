# Exact-half Cayley retraction: prospective GPU contract

Date fixed: 2026-08-03, after the CPU completion audit and before any
zero-calibration GPU training run.

## Intervention

Repeat the frozen ten-seed partial-Cayley experiment with exactly 120/240
compiler-visible directed edges. The mask retains one orientation of every
reverse-edge pair and no bidirectional calibration edges. The learner receives
neither the inverse-token matching nor the hidden directions.

For each of the three perfect matchings of four tokens, the learner propagates
the candidate inverse relation across the whole observed family. A candidate
is accepted only if it yields four complete permutations without contradiction.
The unique feasible matching supplies the inverse map and all 120 hidden edges.

This changes only the compiler-visible edge mask and inverse-pair inference.
Task training still receives dense prefix-state labels. The architecture,
optimizer, batches, losses, exact joint retraction, evaluation, and untouched
changed-generator selection index 44 remain identical to
`PARTIAL_CAYLEY_RETRACTION_PREREGISTRATION.md`.

## Frozen cohort

- Seeds 0 through 9.
- 2,000 CUDA steps per seed, deterministic algorithms enabled.
- Four-channel pure Cl(3) rotor recurrence.
- Mask seed `910001 + training_seed`.
- `inverse_cover_calibration = 0.0`.
- Untouched macro alphabet index 44.
- Dense lengths 16, 32, ..., 256; L4096 original and untouched; L16384
  untouched.

## Gates

All ten seeds must:

1. infer inverse tokens `(1, 0, 3, 2)`;
2. complete exactly 120 hidden directed edges and replay all transitions;
3. reconstruct a group of order 60 and compile a 3D representation with
   invariance and homomorphism RMS below `1e-10`;
4. trigger joint shared-manifold retraction;
5. maintain at least 90% accuracy at every registered dense and long length on
   both required alphabets.

Secondary mechanistic prediction: because both masks reconstruct the identical
action before any compiler or optimizer decision, each exact-half seed should
match its 122-edge counterpart tensor-for-tensor in the final `state_dict` and
at every logged training trajectory point. This is recorded before the
exact-half run. A mismatch would not by itself fail gates 1--5, but it would
show that supposedly discarded evidence still influenced the pipeline.

The equal-budget control is already fixed: 100 uniform random 120-edge masks
must not be described as comparable evidence coverage; they recover 0/100 and
leave 42--76 edges unknowable even after the true inverse map is granted for
diagnostic purposes.

## Claim boundary

A pass establishes end-to-end preservation of the exact-half structured
completion through optimization and long recurrent execution. It does not
remove dense state labels, infer arbitrary missing edges, or establish the same
sample count for another group or token alphabet.

## Post-registration adversarial finding

Before this GPU cohort was launched, `inverse_cover_adversarial_audit.py` found
2-SAT witnesses under which either wrong token matching is also feasible at
120 edges. The learner refuses both witnesses as ambiguous. Consequently this
registered cohort, if run, measures the stated seeded random-mask distribution;
it cannot establish universal exact-half identifiability. The worst-case-safe
matching protocol adds one calibration edge (121 total) and is registered
separately. This note records a prospective claim reduction, not a changed
threshold after results.
