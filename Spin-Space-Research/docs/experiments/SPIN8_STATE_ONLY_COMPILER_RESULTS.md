# Spin(8) state-only compiler: results

Date completed: 2026-08-03.

## Frozen result: 7/9, gate failed

The state-only compiler removes decoder predictions from discovery. It sees
only recurrent states reached by uniformly random token strings, the token
identities needed to measure one-step successors, and the supplied cardinality
`k=8`. It receives no target labels, logits, hidden Q8 table, inverse map, or
group-aware sampler until after anonymous algebra recovery has finished.

| Cohort | Passes | Frozen requirement | Outcome |
|---|---:|---:|---:|
| Development seed 20 | 1/1 | excluded | diagnostic |
| Prospective smoke seed 38 | 1/1 | smoke pass | pass |
| Untouched seeds 39--47 | 7/9 | at least 8/9 | **fail** |

All seven accepted checkpoints recover an independently replicated regular
group of order eight, are post-hoc isomorphic to Q8, and reach 100% at every
dense and long checkpoint through L16384 after shared-family Spin(8)
retraction and observer transport. Seeds 43 and 46 are refused solely because
their frozen centroid-separation ratios are `1.896973` and `1.834542`, below
the preregistered floor of `2.0`.

Two implementation failures in the first cohort artifact were genuine code
failures, not scientific failures: one invalid k-means restart aborted the
entire eight-restart search for seeds 44 and 45. The implementation was
corrected to discard an invalid restart and fail only if all restarts are
invalid. Deterministic reruns pass with separation ratios `4.012361` and
`3.935135`. The original 5/9 artifact is preserved; the separate adjudication
artifact records the corrected 7/9 result and does not change any threshold.

## Torsor-origin correction

The development cycle exposed a deeper issue before the prospective runs.
Anonymous state clusters do not identify the group identity, and the raw
initial recurrent state is not a reliable identity anchor because endpoint
curriculum training never supervises the empty word. Both cluster-label zero
and the centroid nearest the initial state produced exact anonymous tables but
0% behavior in seed 20.

The correct origin is recovered without labels by replaying each observed
calibration word through the anonymous transition action from every possible
base cluster. The unique base agreeing with observed endpoint clusters is the
torsor origin. All prospective accepted runs have origin winner fraction and
winner-minus-runner-up gap exactly `1.0`; in multiple runs that origin differs
from the cluster nearest the raw initial state.

## What the failed gate taught us

The failure is not evidence that seeds 43 and 46 lack an eight-state algebra.
The frozen follow-up cardinality audit finds exact transition determinism,
exact regular closure, exact torsor-origin recovery, and exact independent
replication at `k=8` in both. However, each also admits an exact `k=2` action.
Thus decoder-free state geometry contains a lattice of stable finite
quotients; state cardinality is not identified by closure alone.

The next gate must recover the **finest reproducible congruence** and prove
that every coarser viable action is its homomorphic quotient. Picking the
largest `k` or lowering the separation threshold post hoc would not be a
scientifically acceptable repair.

Artifacts:

- `spin8_state_only_validation_seeds39_47_compiled.json` (original 5/9)
- `spin8_state_only_validation_seeds39_47_adjudicated.json` (corrected 7/9)
- `spin8_state_only_seed44_rerun.json`
- `spin8_state_only_seed45_rerun.json`
- `spin8_state_cardinality_audit_seed38.json`
- `spin8_state_cardinality_audit_seed43.json`
- `spin8_state_cardinality_audit_seed46.json`
