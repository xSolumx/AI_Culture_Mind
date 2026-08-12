# Programme 02: Equivariant intertwiner identification and structured priors

## Scope

Identification of shared action families and equivariant intertwiners from
partial observations. The programme compares structured hypothesis classes
with unrestricted fits, independently fitted families, and explicit group
augmentation. Spin(8), SO(3), A5, and Q8 are controlled instances rather than
the title claim.

## Core Questions

1. Which relational constraints remove null directions left by independent
   endpoint fitting?
2. When does a supplied equivariant hypothesis class extrapolate to unseen
   group orbits, views, or action words?
3. What information can data augmentation recover that architecture supplies
   directly, and at what supervision cost?

## Proven / Established Results

- In the frozen teacher-aligned gate, the complete one-dimensional Spin(8)
  intertwiner family recovers its scalar and extrapolates to held-out orbits
  through length 512 at float64 roundoff.
- An unrestricted bilinear tensor fits the same restricted endpoints but fails
  on unseen orbit directions. Four transformed copies make its design full
  rank and restore roundoff-level extrapolation.
- The SO(3) cross-product control reproduces the structured-versus-restricted
  result. The established effect is therefore a general equivariant-prior and
  support-identification effect, not an exceptional triality advantage.
- The prospective paired-action Spin(8) replication on untouched seeds
  `20`--`29` passes the shared-action decision `10/10`; independently fitted
  actions remain incomplete even when direct and delta memories use identical
  hard routes.
- Finite-group shared-family, retraction, curriculum, and compiler studies
  establish controlled relational-completion and optimization results under
  their stated supervision contracts.

## Open Claims

- Discovery of the group, representations, or intertwiner from raw or
  naturalistic data is not established.
- Robustness to noisy labels, representation misspecification, and learned
  rather than supplied action frames remains open.
- A general sample-complexity theorem comparing architectural priors with
  augmentation has not been proved.
- Transfer to end-to-end sequence models remains an empirical question.

## Dependencies

- Programme 01 supplies the recurrence/scan execution used by some gates, but
  the identification result is about the hypothesis class rather than the scan
  algorithm.
- Programme 04 supplies the exceptional Spin(8) and Spin(9) representation
  instances. The matched SO(3) cross-product intertwiner experiment is
  mandatory when interpreting generality; it is a control family, not a
  standalone maintained model.
- Programme 03 supplies retrieval tasks used to expose action-completion
  errors; direct/delta parity is not itself an identification claim.

## Non-claims

- Structured completion is not extra storage capacity or a superior update
  law.
- Supplying the complete intertwiner direction is not discovering it.
- Endpoint interpolation is not orbit identification.
- These controlled studies do not establish natural-language or production
  model superiority.

## Canonical Evidence

- [Equivariant-identification result with SO(3) cross-product intertwiner control](../../Spin-Space-Research/docs/experiments/INTERTWINER_SCHURSCAN_EQUIVARIANT_IDENTIFICATION_RESULTS.md)
- [Prospective paired-action replication](../../Spin-Space-Research/docs/experiments/TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md)
- [Detailed shared-family evidence ledger](EVIDENCE_LEDGER.md)
- [Spin(3) isotypic correction](../../Spin-Space-Research/docs/experiments/SPIN3_ISOTYPIC_SCHUR_SCAN_RESULTS.md)

The paired-action and isotypic reports are intentionally cross-listed. Their
primary identification/prior conclusions belong here; their memory and scan
conclusions remain in Programmes 03 and 01.

## Reproduction and Horizon Expansion

The first HRT control is now reproduced locally as a finite-sampling sanity
check. It encodes the symmetric `(2n+1,2)` configurations from Guan--Okoudjou,
general four-point real-valued controls, an ultimately-positive window control,
and a lattice control. All six recorded cases have full sampled column rank in
the current artifact:
[`hrt_configuration_controls_20260810.json`](artifacts/hrt_configuration_controls_20260810.json).
The same artifact also checks the projective finite Weyl law to machine
precision and records a four-vector orbit-rank control in dimension 32.

Run it with:

```powershell
$env:PYTHONPATH='src'
python src/hrt_reproduction.py --output artifacts/hrt_configuration_controls_20260810.json
python -m unittest discover -s tests -v
```

This does not prove the arbitrary-`L2` HRT theorem. The expansion opportunity
is instead precise: implement finite Heisenberg time-frequency actions as a
new structured-prior control, then compare supplied representation constraints,
generic bilinear fitting, and orbit augmentation using the same identification
protocol already used for Spin(8) and SO(3). Any resulting theorem would be a
new Heisenberg/Gabor result, not evidence for the existing Spin claims.
