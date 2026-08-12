# One-calibration Cayley retraction results

Date: 2026-08-03.

Prospective contract:
`ONE_CALIBRATION_CAYLEY_RETRACTION_PREREGISTRATION.md`.
Machine report: `one_calibration_cayley_10seeds.json`.
Validated summary: `one_calibration_cayley_10seeds_summary.json`.
Cross-cohort audit: `one_vs_two_calibration_cohort_equivalence.json`.
Continuous long audit: `one_calibration_long_adversarial_audit.json`.

## Decisive result

The worst-case-safe 121-edge cohort passes every preregistered gate in all ten
seeds.

| Metric | Result |
|---|---:|
| Compiler-visible directed transitions | 121/240 (50.4167%) |
| Inferred hidden transitions | 119/240 |
| Correct inverse-token inference | 10/10 |
| Exact transition replay and order-60 reconstruction | 10/10 |
| Joint representation discovery/retraction | 10/10 |
| Dense L16--L256 minimum, original alphabet | 100% |
| Dense L16--L256 minimum, untouched index 44 | 100% |
| L4096 minimum, both alphabets | 100% |
| L16384 minimum, untouched index 44 | 100% |
| Dense L4096--L16384 sweep (13 lengths), untouched index 44 | 100% |
| Compiler invariance RMS | `3.051e-16` |
| Compiler homomorphism RMS | `5.025e-16` |
| Worst float32 vector homomorphism RMS | `2.607e-7` |

Discovery triggers range from step 400 to 850 (mean 515). Accuracy is 100%,
not merely above the 90% gate, at every registered seed/length/alphabet cell.
The additional long sweep evaluates every 1,024 tokens from L4096 through
L16384. Its population minimum and mean are both 100%; maximum path-to-canonical
state drift is `0.006111`, remaining decoder-safe. Every recovered action is
post-hoc exactly isomorphic to a noncommutative simple order-60 group with
conjugacy-class sizes `[1, 12, 12, 15, 20]`.

## Separation of evidence recovery from optimization

The one-calibration cohort was compared with the earlier two-calibration cohort
seed-for-seed. All ten final `state_dict` objects are tensor-for-tensor bitwise
equal. All ten logged training trajectories are exactly equal. All selected
candidate, mechanism, dense-evaluation, and long-evaluation report fields are
exactly equal.

This confirms the mathematical decomposition implemented by the pipeline:

1. different sufficient partial evidence masks recover the same exact action;
2. once recovered, the compiler and optimizer receive identical inputs;
3. the discarded calibration edge has no hidden downstream influence.

## Why 121, not 120 or 122

- 122 edges pass 10/10 but are conservative: they calibrate both inverse
  families independently.
- 120 edges pass 1,000/1,000 random reverse covers, but exact 2-SAT witnesses
  make either wrong matching feasible; the learner refuses those masks.
- 121 edges reveal one true reverse pair. Its positive identity support selects
  the true matching, both wrong matchings have zero support, and the remaining
  token pair is forced. This is worst-case safe under the assumptions stated in
  `INVERSE_COVER_IDENTIFIABILITY_THEOREM.md`.

The exact 2-SAT algorithm is complete for the existence question over all
`2^120` orientations for each of the only two wrong matchings. It solves that
question in polynomial time rather than enumerating the masks, and returns one
witness per matching rather than all witness shapes. The 240/240 repair audit is exhaustive over the 120
calibration choices for those two witnesses. Universal 121-edge safety follows
from the algebraic identity-support proof, not from conflating those checks with
an enumeration of all ambiguous masks.

## Scope

This is a finite-action identifiability and recurrent-reliability result for
A5 with four distinct tokens forming two inverse pairs. The masking environment
guarantees a reverse cover, and dense exact prefix labels remain available
before masking. It is not an arbitrary missing-table result and does not carry
the number 121 to Q8, D4, S3, other alphabets, noisy observations,
endpoint-only supervision, language modeling, or groups without a real 3D
irrep.

## Validation

- 37/37 PyTorch tests pass, including CUDA forward/backward, full/chunk/token
  streaming equivalence, exact-half completion, adversarial ambiguity refusal,
  and one-calibration repair.
- 12/12 JAX geometric-algebra and streaming tests pass.
- All 153 JSON artifacts parse.
- All 31 checked Markdown research documents have no missing relative links.
- Python byte-compilation passes for `SSM-Models` and `SpinorModel`.
