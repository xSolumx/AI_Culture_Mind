# Q8 spinor selection cohort: seeds 0--9

Date completed: 2026-08-03. These seeds select the representation-quality gate
and are not its validation cohort.

## Three distinct reliability questions

| Variant | Dense smoke gate | Strict algebra gate |
|---|---:|---:|
| Raw learned spinor | 9/10 | 0/10 whole-model at `1e-5` |
| All-channel joint retraction | 9/10 | 10/10 |
| Joint retraction + frozen quality gate | 10/10 | 10/10 |

Raw seed 5 falls to 44.7% pair-member and 32.5% both-correct accuracy; raw seed
6 narrowly passes at 99.56% and 99.12%. Every seed nevertheless contains two
or three channels within `0.0116` action RMS of the exact Q8 manifold.

All-channel retraction moves every channel onto exact Q8, producing whole-model
homomorphism RMS around `1.6e-7` in all ten seeds. It does not repair seed 5's
decoder: the large action movement on its nuisance channels creates
orientation-valid but decoder-misaligned exact representations, leaving even
lengths near 75%.

## Quality-gate selection

Projection distance is cleanly bimodal across the 40 channels:

- useful representation channels: `0.0004`--`0.0116`;
- nuisance channels: `0.4918`--`0.6977`.

The frozen `0.10` rule retains channel masks:

```text
seed 0  [1,0,1,1]    seed 5  [0,1,1,0]
seed 1  [0,1,1,1]    seed 6  [0,1,0,1]
seed 2  [0,1,1,1]    seed 7  [0,1,0,1]
seed 3  [1,1,0,1]    seed 8  [1,0,1,1]
seed 4  [1,1,0,1]    seed 9  [1,1,0,1]
```

Zeroing only the rejected decoder channel columns restores all ten selection
seeds to 100% pair-member and both-correct accuracy at the smoke dense
checkpoints. The gate uses no endpoint labels, accuracies, or decoder ablation
results, but its threshold was selected after observing seeds 0--9. Therefore
this 10/10 is hypothesis formation, not confirmation.

Fresh seeds 10--19 and the full dense/long protocol are prospectively fixed in
`Q8_SPINOR_QUALITY_GATE_PREREGISTRATION.md`.

That prospective validation subsequently passes 10/10; it is reported
separately in `Q8_SPINOR_QUALITY_GATE_VALIDATION_RESULTS.md` so the selection
and validation evidence remain auditable.

Artifacts:

- `q8_spinor_center_spinor_10seeds.json`
- `q8_spinor_joint_retraction_10seeds.json`
- `q8_spinor_quality_gate_selection_10seeds.json`
