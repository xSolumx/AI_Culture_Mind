# Robust Channel-Gating Results

## Outcome

The prospectively specified frozen-action gate passes the untouched third
changed-generator alphabet in all ten seeds. The minimum learned-gate floor
over every seed and dense L16-L256 length is `99.71%`; nine seeds have floors
from `99.90%` through `100.00%`.

The protocol was fixed in `ROBUST_CHANNEL_GATING_PREREGISTRATION.md` before
index-2 evaluation. Original actions plus macro class 0 fit three bounded
auxiliary scalars, class 1 selected the optimization step, and class 2 was used
once for final testing. Recurrent actions, initial states, output heads, biases,
and logit scales remained frozen.

This split is essential. If fitting, gate selection, and reporting used the
same macro alphabets, the learned gate could be only a soft re-expression of
the labeled oracle subset search. The untouched class-2 evaluation is what
makes the result a generalization test.

The untouched class-2 macro alphabet is `12453`, `12534`, `24531`, `51423`.
It is the third repository-ordered order-3/order-5 equivalence class after
quotienting inverse re-labellings. Like the first two classes, it retains the
first order-3 inverse pair and changes the order-5 inverse pair.

## Per-seed results

| seed | anchor | selected step | learned gates | class-2 anchor | class-2 full | class-2 learned |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 1 | 60 | 0.64, 1.00, 0.87, 0.88 | 91.89% | 99.90% | 100.00% |
| 1 | 0 | 0 | 1.00, 0.50, 0.50, 0.50 | 96.29% | 100.00% | 99.90% |
| 2 | 0 | 10 | 1.00, 0.62, 0.62, 0.38 | 98.05% | 95.80% | 99.90% |
| 3 | 2 | 0 | 0.50, 0.50, 1.00, 0.50 | 98.44% | 99.90% | 100.00% |
| 4 | 1 | 0 | 0.50, 1.00, 0.50, 0.50 | 97.85% | 99.41% | 99.90% |
| 5 | 2 | 0 | 0.50, 0.50, 1.00, 0.50 | 100.00% | 99.02% | 99.90% |
| 6 | 2 | 370 | 0.06, 0.46, 1.00, 0.67 | 94.63% | 80.57% | 99.71% |
| 7 | 2 | 10 | 0.39, 0.61, 1.00, 0.39 | 100.00% | 98.14% | 100.00% |
| 8 | 0 | 10 | 1.00, 0.39, 0.62, 0.38 | 98.63% | 99.12% | 99.90% |
| 9 | 3 | 10 | 0.62, 0.39, 0.39, 1.00 | 98.63% | 97.07% | 99.90% |

The learned gate passes the 90%-at-every-length class-2 criterion `10/10`.
Anchor-only also passes `10/10` on class 2, but its population worst floor is
`91.89%`; all-channels passes `9/10` because seed 6 falls to `80.57%`.

## Regression contract

The preregistered stronger target also passes `10/10`: no learned gate reduces
the disjoint original-alphabet or class-1 dense floor by more than one
percentage point relative to the better of anchor-only and all-channels for
that seed.

Learned-gate dense floors across the four final evaluation alphabets are:

- original: at least `99.41%`;
- class 0: at least `97.27%`;
- class 1: at least `99.12%`;
- untouched class 2: at least `99.71%`.

The gate is therefore not rescuing class 2 by sacrificing the original
language or the model-selection alphabet.

## Interpretation

This is the first prospective intervention derived from the mechanistic audit
that survives an untouched generator distribution. It converts the earlier
oracle-subset observation into a deployable fixed gate learned without final
test labels.

The result favors **anchor-preserving shrinkage**:

`logits = bias + anchor_logits + sum(alpha_c * auxiliary_logits_c)`,

with `alpha_c in [0, 1]`. The dominant approximate irrep remains fully active;
auxiliary channels are retained but their ability to overturn its decoder
boundaries is calibrated by a distributionally robust margin objective.

Four seeds select optimization step 0, where every auxiliary gate is exactly
`0.5`. Thus a substantial part of the gain is simple residual shrinkage rather
than elaborate fitting. Other seeds, especially seed 6, require asymmetric
weights. Seed 6 is the clearest causal case: `[0.06, 0.46, 1.00, 0.67]` raises
the untouched floor from `80.57%` full and `94.63%` anchor-only to `99.71%`.

This does not revive the geometric error-correction claim. The prior audit
shows class-boundary correction without proportional defect tracking or full
defect-vector cancellation. The gate controls a decoder ensemble; it does not
make the learned operators exact.

## Paired hard-oracle comparison

The joint pair0/pair1 hard-oracle subset was fixed before pair-2 exposure. A
post-result paired evaluator then scored that hard subset and the already-fixed
soft gate on exactly the same pair-2 sequences:

| seed | hard subset | hard floor | soft floor | soft - hard |
|---:|---|---:|---:|---:|
| 0 | 0+1+2+3 | 99.90% | 99.90% | 0.00 |
| 1 | 0+1 | 99.90% | 100.00% | +0.10 |
| 2 | 0+1+2 | 100.00% | 100.00% | 0.00 |
| 3 | 0+2 | 100.00% | 99.90% | -0.10 |
| 4 | 0+1+2+3 | 99.12% | 99.71% | +0.59 |
| 5 | 0+1+2 | 100.00% | 100.00% | 0.00 |
| 6 | 2+3 | 98.24% | 99.51% | +1.27 |
| 7 | 1+2+3 | 99.12% | 100.00% | +0.88 |
| 8 | 0+2+3 | 99.90% | 99.80% | -0.10 |
| 9 | 0+2+3 | 100.00% | 100.00% | 0.00 |

Both pass `10/10`. The hard-subset population floor is `98.24%`; the soft-gate
floor is `99.51%`. Soft wins four seeds, ties four, and loses two by only one
example per worst stratum (`0.10` point). Its material gains concentrate in
seeds 4, 6, and 7, where continuous attenuation improves over binary channel
inclusion.

Only one of the 30 auxiliary coefficients is near binary under a `<=0.1` or
`>=0.9` criterion (seed 6 channel 0 at `0.06`). The other 29 are intermediate.
The gate is therefore not generally rediscovering a hard subset; it represents
a genuinely soft ensemble that subset selection cannot express.

## Limits

- This is ten deterministic seeds of one A5 architecture.
- The three changed macro classes share their order-3 inverse pair and vary the
  order-5 pair. Transfer to a changed order-3 class remains untested.
- Class-1 labels select the gate checkpoint, so only class 2 is untouched.
- Evaluation uses 1,024 examples per dense length and stops at L256.
- Raw homomorphism remains far above `1e-3`; no infinite-horizon result follows.

## Changed-order-3 and long-horizon falsifier

The proposed frozen-gate falsifier is complete; see
`JOINT_A5_ROUNDING_RESULTS.md`. The selected soft gates transfer through L256
on the untouched changed-order-3 class in all ten seeds, but the unchanged
learned anchors fail L4096 in all ten. Thus robust decoder calibration does not
remove coherent transition error.

With the same gates frozen, a jointly aligned exact A5 anchor passes L4096 in
all ten seeds with a `96.88%` population floor. Independent angle rounding
still fails three seeds. The gate and the group action solve different
problems: soft auxiliary weighting improves finite-horizon margins; joint
representation closure supplies long-horizon compositional reliability.

## Next falsifier

Remove the oracle from joint representation projection. Learn a shared exact or
retracted token-action family and its global conjugation without using the
untouched alphabet, then retain L4096 as a required test. In parallel, compare
the learned scalar shrinkage against the non-learned `alpha_aux = 0.5` rule to
measure how much credit belongs to robust optimization versus a universal
half-residual heuristic.

Only after that should gating be integrated into recurrence training. Spin(8)
remains deferred: the current breakthrough is calibrated observability around
a small faithful channel, not larger transition dimension.

## Artifact

`robust_channel_gating_a5_10seeds.json`

Paired diagnostic:
`changed_generator_channel_subset_lattice_a5_pair2_paired_gates_10seeds.json`.
