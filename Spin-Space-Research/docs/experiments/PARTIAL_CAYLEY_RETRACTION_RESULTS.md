# Partial-Cayley retraction results: conservative 122-edge cohort

Date: 2026-08-03.

Prospective contract: `PARTIAL_CAYLEY_RETRACTION_PREREGISTRATION.md`.
Machine report: `partial_cayley_retraction_10seeds.json`.
Validated summary: `partial_cayley_retraction_10seeds_summary.json`.

## Outcome

The preregistered cohort passes all gates in all ten seeds.

| Metric | Result |
|---|---:|
| Observed directed transitions | 122/240 (50.8333%) |
| Inferred hidden transitions | 118/240 |
| Correct inverse-token inference | 10/10 |
| Order-60 reconstruction and exact replay | 10/10 |
| Joint representation discovery/retraction | 10/10 |
| Dense L16--L256 minimum, original alphabet | 100% |
| Dense L16--L256 minimum, untouched index 44 | 100% |
| L4096 minimum, both alphabets | 100% |
| L16384 minimum, untouched index 44 | 100% |
| Compiler invariance RMS | `3.051e-16` |
| Compiler homomorphism RMS | `5.025e-16` |
| Worst float32 vector homomorphism RMS | `2.607e-7` |

Discovery triggers range from step 400 to 850 (mean 515). Every seed infers
the unknown inverse mapping `(1, 0, 3, 2)`, completes the missing directions,
recovers a regular order-60 action, finds one real 3D candidate, and keeps the
anchor token family on a single exact conjugacy orbit during later ambient
updates.

## Interpretation after the sharper audit

This run was prospective and remains a valid pass, but its scientific role
changed while it was running. It was registered with one bidirectional
calibration pair in each inverse family. Subsequent global-matching work showed:

- 120 edges work in 1,000/1,000 random reverse covers, but a 2-SAT adversary
  constructs ambiguous exact-half masks;
- one calibration pair total resolves every tested adversarial choice, making
  121 edges the worst-case-safe threshold for this matching protocol;
- 122 edges are therefore a conservative regression point, not the tightest
  result.

Nothing in the 122-edge report is relabeled as a 121- or 120-edge result. The
separate 121-edge cohort was preregistered after the adversarial finding and is
the decisive follow-up.

## Scope

Dense prefix-state labels remain available to the task and to the source
transition record before masking. The environment guarantees a reverse-edge
cover. This does not establish arbitrary partial-table completion,
endpoint-only discovery, noise robustness, language-model utility, or scaling
beyond A5 and the four-token inverse-closed alphabet.
