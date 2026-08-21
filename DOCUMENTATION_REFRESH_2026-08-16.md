# Documentation reconciliation — 2026-08-16

Reconciliation opened at **2026-08-16T16:05:27+02:00** and latest
research status reconciliation completed at **2026-08-17T03:47:29+02:00**
(`Africa/Johannesburg`, UTC+02:00).

## Scope and method

This pass inventories the documentation visible in this checkout, rather than
retroactively rewriting historical reports. After integration with the
flattened theorem tree, the latest `rg --files` inventory is 443 Markdown/RST
files, including this timestamped ledger. It covers the
maintained Pure Rotor and Pure Spin(8) sources and tests, frozen comparison
artifacts, exact representation programmes, and the public programme front
doors.

Current-status documents were corrected directly. Dated result reports and
preregistrations retain their recorded measurements; where a later status is
relevant, they now receive a dated pointer instead of an edited result table.

## Current authoritative status

- The canonical trainable implementation remains
  `SSM-Models/pure_rotor_ssm/`, version 2.1.0. `ga_ssm.py` is a training shell
  and `rotor_ssm_torch.py` is an import-compatibility shell.
- `SSM-Models/pure_spin8_ssm/`, version 1.1.0, is a second maintained model
  family with a separate checkpoint schema. It does not replace or silently
  upgrade Pure Rotor v2.1. Version 1.1 adds an opt-in frozen finite-token
  action compiler and Triton recurrence; the continuous maintained layer and
  its older checkpoint family remain separate.
- PyTorch exposes opt-in `scan_mode="schur_parallel"`. It is equivalent to the
  maintained direct recurrence within tested numerical tolerance, including
  forward, first-order gradients, padding, cache continuation, and CUDA. It is
  not a fused kernel, does not alter checkpoints, remains PyTorch-only, and is
  not yet a performance claim.
- `SSM-Models/benchmark_pure_rotor_vs_mamba2.py` is the direct architecture
  runner for maintained Pure Rotor, an identity-transport ablation, and
  Transformers Mamba-2. Its raw parameter counts are 141,796, 141,796, and
  145,016 respectively. The identity ablation disables 6,060
  rotor-controller parameters, so raw equality is not effective-capacity
  equality.
- The artifact
  `SSM-Models/experiments/artifacts/pure_rotor_vs_mamba2_smoke_seed0.json`
  completed a one-step CUDA smoke only (SHA-256
  `d7caef587864e46288c81afb842c99f2ac339a99f0df2e7809f9efcf8a6d658a`). It
  is operational/provenance evidence, not a quality, memory, or systems
  result. Its Mamba-2 backend was Transformers without importable
  `mamba_ssm`, so it is not a native fused-Mamba throughput comparison.
- The 2026-08-06 v2.1 five-seed transport-ablation cohort remains the current
  local evidence for Pure Rotor transport. It supports a qualified prediction
  advantage over its retrained identity row, but not a rotor-specific, memory,
  or compute-efficiency advantage.
- `Spin8-SSM-Benchmark/SpinorDeltaLM`, the Spin(8) research branch, and the
  historical `SpinorModel` implementations remain separate model families.
  No smoke artifact or theory result transfers across those boundaries without
  an explicit bridge experiment or theorem.
- The fresh three-seed latent-token gate removes supplied Lie coordinates for
  one finite eight-token dictionary and passes every frozen unseen-center
  relation gate. The v1.1 compiler then freezes the learned actions and runs a
  register-resident Triton recurrence. These are respectively a synthetic
  every-prefix identification result and a local inference systems result,
  not a natural-task, state/compute-matched, or fused-Mamba claim.

## Documentation changed in this reconciliation

- Root research map: `README.md`.
- Programme status maps:
  `research-programs/01-associative-scan-algebra-and-compilers/`,
  `research-programs/04-triality-clifford-representation-dynamics/`, and
  `research-programs/06-rotor-noncommutative-state-space-models/`, plus
  `research-programs/SUPPORTING_TRACKS.md` for controlled benchmarks.
- Maintained-model contract and foundations: `SSM-Models/README.md`,
  `SSM-Models/FOUNDATIONS.md`, and `SSM-Models/pure_rotor_ssm/CONTRACT.md`.
- Historical/context boundaries: `SSM-Models/FOUNDATIONS_VALIDITY_AUDIT_2026-08-06.md`,
  `SSM-Models/SPIN8_TRIALITY_EXPERIMENT.md`, `Spin8-SSM-Benchmark/README.md`,
  `SpinorModel/README.md`, and `SpinorModel/overhauled/README.md`.
- Result-status pointers: `SSM-Models/experiments/PURE_ROTOR_SSM_V2_RESULTS.md`,
  `SSM-Models/experiments/PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md`, and
  `SSM-Models/experiments/PURE_ROTOR_VS_MAMBA2_BENCHMARK.md`.

All other dated reports were inspected as historical records and left unchanged
unless they were a front door or directly named the maintained Pure Rotor,
Schur, or Mamba comparison. Their original date and result interpretation are
part of the provenance record.

## Initial verification and later topology reconciliation

- `python -m pytest SSM-Models/test_pure_rotor_ssm.py -k schur -q`: passed
  (2 tests).
- `python -m pytest SSM-Models/test_pure_rotor_vs_mamba2_benchmark.py -q`:
  passed (2 tests).
- `ruff check SSM-Models/benchmark_pure_rotor_vs_mamba2.py
  SSM-Models/test_pure_rotor_vs_mamba2_benchmark.py` and Python compilation:
  passed.
- The first local Markdown scan was run before the 2026-08-11 topology commits
  reached this branch and therefore reported 16 missing submodule-relative
  paths plus the LaTeX `x_t` false positive. Integration with current `main`
  vendors the theorem tree as `Spin-Space-Research/`, removes the submodule,
  and updates the canonical programme taxonomy. The initial missing-link count
  is historical, not the final checkout status.

This reconciliation does not establish any new mathematical theorem, trained
model quality, or production-kernel throughput result.

## Post-reconciliation research update

At **2026-08-16T17:02:30+02:00**, a new direct A5 comparison runner and its
protocol, execution log, 200-update pilot, and 1,000-update three-seed screen
were added. The screen is a controlled symbolic result, not a language-model
or systems result: it finds a variable short-pair signal for Pure Rotor and no
long-horizon retention for any candidate. The authoritative record is
[`PURE_ROTOR_A5_MAMBA2_BUDGET1000_RESULTS.md`](SSM-Models/experiments/PURE_ROTOR_A5_MAMBA2_BUDGET1000_RESULTS.md).

At **2026-08-16T17:19:37+02:00**, the repository added a separate
Spin(3)--Spin(12) Clifford/Dirac ladder and A5 binary-lift audit. Its
Gaussian-integer matrix checks pass exactly, while the explicitly labelled
float64 quaternion enumeration finds 120 lifts and 60 projective classes. The
result does not extend triality beyond Spin(8) and is not an ML result. The
authoritative record is
[`SPIN_DIRAC_A5_LADDER_RESULTS.md`](SSM-Models/experiments/SPIN_DIRAC_A5_LADDER_RESULTS.md),
with artifact SHA-256
`00109bcff339cc139045c6bfef0274621ced47fbcc4749b1e44d3f4c0f264c68`.

At **2026-08-16T17:33:40+02:00**, the binary-group and tangent-centralizer
parts of that ladder were promoted from float64 evidence to exact
`Q(sqrt(5))` certificates. The exact closure has 120 binary elements and 60
projective classes. For every listed rung, the complete relation Jacobian has
kernel equal to the infinitesimal conjugacy image, proving `H1=0` for the
fixed embedding. This is first-order rigidity, not global classification or a
derived obstruction theorem. The authoritative record is
[`SPIN_DIRAC_A5_RIGIDITY_RESULTS.md`](SSM-Models/experiments/SPIN_DIRAC_A5_RIGIDITY_RESULTS.md),
with artifact SHA-256
`e5c742c29ad7b9de044df7efc7d3836852d93bcc37d293d8824d61de29a3da66`.

At **2026-08-16T17:41:32+02:00**, an exact averaging contraction on the
complete 120-element `2.A5` table closed the degree-two group-cohomology gate.
The executable checks every associativity triple, every vector-action product,
and all degree-one/degree-two homotopy outputs, proving `H1=H2=0` for every
linear module over `Q(sqrt(5))`. The positive cokernel of the raw
three-relator Jacobian is therefore presentation-syzygy redundancy, not `H2`.
The authoritative record is
[`SPIN_DIRAC_A5_COHOMOLOGY_RESULTS.md`](SSM-Models/experiments/SPIN_DIRAC_A5_COHOMOLOGY_RESULTS.md),
with artifact SHA-256
`c96315dd1338c66da5f58e4e0edf70b60ae4b85e9873dc741d76cad77b24db81`.

At **2026-08-16T17:59:40+02:00**, the next global gate was completed through
dimensions `3, 8, 9, 10, 11, 12`. The exact quaternion table yields nine
orthonormal irreducible characters, five real-type and four quaternionic-type
real modules, the affine-`E8` McKay graph, and all 81 nonnegative integral
tensor decompositions. Exhaustive multiplicity enumeration plus the separately
labelled standard universal-cover theorem for `2.A5` produces
`3, 32, 32, 42, 59, 98` Spin-conjugacy components. Independent generating
functions reproduce both the type counts and the 7/14 orientation splits in
dimensions 8/12. The authoritative record is
[`SPIN_DIRAC_A5_COMPONENT_ATLAS_RESULTS.md`](SSM-Models/experiments/SPIN_DIRAC_A5_COMPONENT_ATLAS_RESULTS.md),
with artifact SHA-256
`ed0097caf2e605b0462a94910567471f12c1d095e57a89a7fcdaabfb170895ef`.

At **2026-08-16T18:12:53+02:00**, the finite-group Dirac branching gate was
completed for all 245 orthogonal representation types. Quaternionic base
spinors are derived from signed SU(2) weight parity, while an independent
Newton/exterior-algebra calculation verifies the Clifford identities for every
component. The 7 dimension-8 and 14 dimension-12 orientation splits all
exchange distinct chiral characters. The fixed Spin(3) ladder is exactly
isotypic in the defining binary spinor and has no invariant spinors. The
authoritative record is
[`SPIN_DIRAC_A5_SPINOR_BRANCHING_RESULTS.md`](SSM-Models/experiments/SPIN_DIRAC_A5_SPINOR_BRANCHING_RESULTS.md),
with artifact SHA-256
`2cfe3573b49bcb2a217bcd7545e4629c7e6fa8cc50be61137c0b9f9f9864f21e`.

At **2026-08-16T18:39:55+02:00**, the frozen center-sensitive `2.A5`
three-seed cohort completed its provenance-preserving replay. All 12 trained
rows carry explicit seed IDs, evaluation schedule hashes, checkpoint paths, and
checkpoint SHA-256 values. Independent validation found no split, oracle, or
checkpoint-hash errors. The exact and float64 quaternion oracles are perfect;
the projective oracle is exactly 50% on binary-center metrics. The Spin
quaternion product scan passes the registered central-margin gate at L16, L64,
and L128 in every seed, with mean exact accuracies 99.76%, 79.60%, and 62.20%.
Pure Rotor v2.1, its identity ablation, and Transformers Mamba-2 approach chance
on the long center distinction. The authoritative report is
[`PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md`](SSM-Models/experiments/PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md),
and the artifact SHA-256 is
`911815d9e104fa08e632161f97f41a966991a9102c70ca65e52a5f07d28d4476`.
This is a parameter-near finite-group mechanism result, not a state-matched,
language-model, fused-kernel, or general Mamba theorem.

At **2026-08-16T18:48:02+02:00**, the successful operation was implemented as
the separate experimental PyTorch module
[`spin_scan.py`](SSM-Models/pure_rotor_ssm/spin_scan.py). It preserves the Spin
representative through Hamilton-product recurrence rather than quotienting by
conjugation. Its recurrent and Hillis--Steele paths expose identity padding,
streaming cache continuation, compact/Cl(3,0) conversion, trainable token
increments, and a minimal decoder. The canonical Pure Rotor remains v2.1.0;
neither its checkpoint layout nor bounded affine recurrence changed.

At **2026-08-16T18:56:42+02:00**, a no-retraining exploratory relation
falsifier completed. A deterministic input-only selector found the shortest
locally reduced identity/center pair absent from all three realized training
schedules at length 11. The frozen Spin checkpoints are 100% exact at L16,
retain 100% central margin through L128 in every seed, and average 59.68% exact
L128 accuracy. The result reduces the local-`a,a`-memorization explanation but
remains post-pilot exploratory. The authoritative report is
[`PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md`](SSM-Models/experiments/PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md),
with artifact SHA-256
`6580fb4a27cf9a77d19b39ba95fa614026903e81558a605d2e8dcd37c85a3b81`.

At **2026-08-16T19:29:51+02:00**, the preregistered multi-relation and
conjugated-coordinate `2.A5` pilot completed. Training excludes `a^2`, `b^3`,
and `(ab)^5` while exact legal-language search and realized schedules still
cover all 120 binary states. The benchmark adds an unfused, equation-faithful
DeltaProduct reference pinned to its reviewed upstream commit and an exact
regular-action PD ceiling. Across three byte-identical inner-conjugate token
schedules, the Spin quaternion scan has at least 99.50% center margin and is
the unique exact-accuracy winner in all 18 registered early-L64/L128 splits.
All other learned candidates fail the center gate. All 216 oracle contracts
and 15 checkpoint hashes pass. The authoritative report is
[`SPIN_2A5_MULTIRELATION_RESULTS.md`](SSM-Models/experiments/SPIN_2A5_MULTIRELATION_RESULTS.md),
with artifact SHA-256
`054527e8c3e30d64df30217c2128616e82e7f2025278c200cdbde611647fe6d4`.
This is one shared initialization across paired coordinates, not multi-seed
replication, official fused-kernel evidence, or a theorem about SSM families.

At **2026-08-16T19:44:30+02:00**, the corrected frozen unit-dual-quaternion
motor audit completed. The eight-scalar state is a sign-sensitive double cover
of `SE(3)` and reduces exactly to the quaternion Spin scan at zero translation.
All numerical gates pass through length 4096: motor versus recurrent
homogeneous-matrix error is at most `5.82e-14`, the Study condition error is at
most `6.66e-16`, chunk/cache and recurrent/parallel parity pass, and central
negation changes the state antipode while leaving the physical matrix exactly
unchanged. The local eager CUDA motor tree is about 8.6 times slower than the
4 by 4 matrix tree, so compact representation is not reported as a kernel win.
The authoritative report is
[`MOTOR_PATH_DEVELOPMENT_RESULTS.md`](SSM-Models/experiments/MOTOR_PATH_DEVELOPMENT_RESULTS.md),
with artifact SHA-256
`3496e374ddb68d48f3105b41ac39c23a40be3edaa872f76dcf6ce9e06ea8f95a`.
This is an implementation/path-development gate, not a learned-model result.

At **2026-08-16T20:05:53+02:00**, the frozen learned rigid `2.A5` pilot
completed. The task combines all three held-out central words with body-frame
translations and compares 22,056--22,132-parameter quaternion, motor,
Transformers Mamba-2, and DeltaProduct candidates. All four 300-step learned
readouts fail the registered long center and joint-pose gates, including the
motor classifier. The artifact SHA-256 is
`49d5d031a6e496e36e8404b666d07d49ff4af6469aa7b4a2af6f3f5c01e2d3e2`.
This negative result is retained rather than overwritten.

At **2026-08-16T20:09:52+02:00**, a separately frozen direct-state follow-up
completed. The 49-parameter `Spin(3) x R3` ablation retains the center sign on
every long split but fails translation, while the randomly initialized direct
motor fails to optimize adequately in 300 updates. The artifact SHA-256 is
`7a364b61ba51666db65f0ced909fc78d81855582fd14e9dd5e598d2d4d3ab1f2`.

At **2026-08-16T20:14:21+02:00**, local motor identification from legal
supervised prefix differences passed all six preregistered gates without
reading evaluation inputs or targets. The resulting 49-parameter/8-state-
scalar tracker reaches 100% joint signed pose and paired double-cover pose on
all 18 splits through L128. The screening artifact SHA-256 is
`df132600d8be86505a4e5156b161a7e2ee33ce84dc4dba1dae26d3207653c62b`.

At **2026-08-16T20:17:33+02:00**, the identifier replicated across 3 generator
coordinates by 3 independent legal schedules. All nine runs and all 162 splits
pass; the minimum joint and paired accuracies are 100%, worst mean translation
error is `7.74e-7`, and worst mean signed rotation error is `0.00433` degrees.
All nine training schedule hashes are unique. The compact replication artifact
SHA-256 is
`97ffc994889278b21da7482ff49a597d9799f4ac38472e43b459465468a00aa5`.
This is a finite deterministic identification result under every-prefix pose
supervision, not end-to-end superiority, continuous-data evidence, or a broad
SSM theorem. The authoritative report is
[`SPIN_MOTOR_RIGID_2A5_RESULTS.md`](SSM-Models/experiments/SPIN_MOTOR_RIGID_2A5_RESULTS.md).

At **2026-08-16T20:31:20+02:00**, the separately frozen signed-pose noise audit
completed 4 tiers by 5 independent noise seeds. All 15 clean, 1-degree/0.01,
and 5-degree/0.05 runs pass every center, joint-pose, and paired-pose gate. At
15 degrees/0.15, all five runs retain the central sign but joint/paired
accuracy reaches a 46.875% worst case. The artifact SHA-256 is
`2adc41d821e110e8b8e05624a0587c2aaa04492c2c47f8eabd7e35251020a6d5`.
Its 20 checkpoints were independently rehashed with zero mismatches. Noise
preserves the signed lift, so this is not recovery from unsigned `SO(3)` data.

At **2026-08-16T20:49:07+02:00**, the nonassociative-octonion barrier was
resolved at the architecture level by an associative multiplication-operator
lift. The implementation retains an explicit norm-2 associator witness,
matches the canonical Fano table exactly, and has an exact rank-28 `so(8)`
coordinate determinant `-2^49`. Its bounded affine layer uses an eight-scalar
streaming cache and passes the length-4096 stability gate. The native-Windows
artifact SHA-256 is
`992ca268f50e366b1a024c9e7ac63814f981312219097c178039b31d2e7f1830`.

At **2026-08-16T20:58:30+02:00**, the optional Ubuntu WSL2/Triton continuation
completed. Its custom forward and reverse kernels pass forward, gradient, and
L4096 unit-norm gates. The L4096 forward median is 1.805 ms versus 8.951 ms for
the work-efficient operator path (4.96x), and the L1024 forward/backward median
is 1.691 ms versus 11.628 ms (6.88x). The WSL artifact SHA-256 is
`3af7bb5d0e8711d96c9ef2e0bca60eef98f1958e1bcb5c78570b8bd4c78a1d2c`.
Both results are documented in
[`OCTONION_OPERATOR_SCAN_RESULTS.md`](SSM-Models/experiments/OCTONION_OPERATOR_SCAN_RESULTS.md)
and remain separate from Pure Rotor v2.1, generic task-quality claims, and the open
unrestricted Dirac--Gram theorem.

At **2026-08-16T21:05:51+02:00**, the frozen continuous associator-tracking
pilot completed. Training at L16 and evaluating through L128, the 72-parameter
algebra-matched operator encoder reaches L128 MSE `1.848e-12`; the intentionally
invalid collapsed-octonion ablation, unfused DeltaProduct reference, and
unfused Transformers Mamba-2 control score `0.21599`, `0.12435`, and `0.12526`.
The complete operator target requires a 64-scalar recurrent cache. This is one
seed, identity-near initialization, and an algebra-matched synthetic target,
so it establishes realizability and length extrapolation only. Artifact
SHA-256:
`c282b21a2050006f69a1c31c42dd28d2fcd9311e7ba52ef310c3c5cde49d802e`.
All three learned checkpoints rehash and reload; the frozen test locks the
schedule hashes and result/claim boundary.

At **2026-08-16T21:17:32+02:00**, the three-Haar-basis successor completed.
Its 28-parameter learned `SO(8)` gauge passes its registered L128 gate in every
basis with MSE `1.54e-9`--`8.74e-8`. The overall cohort correctly records
`all_required_checks_passed: false`: the 512-parameter dense AdamW control
misses its `1e-3` gate in all bases, and one float32 exact oracle produces
`1.0749e-12` against the strict `1e-12` cutoff. Cohort artifact SHA-256:
`b96bed5d0e4c33e229816f6ce2db24d2c42c5b33ae1b978950a2a0bd9960daf5`.

At **2026-08-16T21:19:15+02:00**, a labelled post-protocol diagnostic used only
legal position-1 training prefixes. Float64 least squares identifies the dense
leaf map at L128 MSE `3.76e-13`--`9.38e-12`, proving its frozen miss is
optimization rather than realizability. The learned-to-true basis residuals
satisfy the `G2` left-action automorphism equation to at most `2.17e-4`, which
explains the irreducible gauge ambiguity. All twelve learned checkpoints and
all nine basis/schedule hashes replay. Audit artifact SHA-256:
`bb149467ebc4cfe32b8ded8311d6a0b5ec47a7b606d13973d371ee7ae955d5c6`.

At **2026-08-16T22:00:42+02:00**, the new separately maintained Pure Spin(8)
v1.0 model completed its frozen three-seed comparison. Its faithful
`(8v,8s+,8s-)` triality cache uses 24 recurrent scalars and one shared 28D
bivector controller. All model/checkpoint contracts pass. On the supplied-
coordinate signed-transport task, all three rows reach L128 MSE
`5.81e-5`--`6.68e-5` and 100% central-sign classification; Transformers
Mamba-2 reaches `0.132`--`0.135` and 50%. The approximately 2,000x MSE gap is
specific to this algebra-matched synthetic cohort. The authoritative report is
[`PURE_SPIN8_VS_MAMBA2_RESULTS.md`](SSM-Models/experiments/PURE_SPIN8_VS_MAMBA2_RESULTS.md),
and the artifact SHA-256 is
`d265e28a132c28261ae317958adfa34619c5dd0c58a0859b26e7afd653ad9876`.

At **2026-08-16T22:05:06+02:00**, the terminal-only octonion continuation
resolved its failed unsigned-`G2` audit. Fixed L16 training had recovered 3/9
structured and 0/9 dense laws; the frozen L2/L4/L8/L16 curriculum recovers
9/9 of each, but all its lengths are even. The corrected audit proves the
observed ambiguity is `G2 union -G2`: four checkpoints occupy the positive
coset and five the negative coset, and held-out L17 produces the predicted
sign in all nine. The original failed audit remains in the repository. The
authoritative report is
[`OCTONION_FINAL_ONLY_RESULTS.md`](SSM-Models/experiments/OCTONION_FINAL_ONLY_RESULTS.md),
and the signed-parity audit SHA-256 is
`32545cb8174bfd84cf0cb2d1a84202570af34dce20ff4f0e48dd9573118c8d30`.

## Final post-update verification

Verification completed at **2026-08-16T19:03:06+02:00**:

- `test_pure_rotor_ssm.py`: 11/11 passed, including cross-backend algebra,
  bounded recurrence, Schur gradients/cache, and CUDA;
- `test_pure_rotor_a5_benchmark.py`: 4/4 passed;
- `test_pure_rotor_2a5_benchmark.py`: 6/6 passed, including the frozen pilot
  artifact hash and registered gate;
- `test_pure_rotor_2a5_unseen_relation.py`: 3/3 passed, including deterministic
  word selection, oracle controls, artifact hash, and the exploratory gate;
- `test_spin_scan.py`: 7/7 passed, including product orientation, center-sign
  separation, parallel/recurrent forward and gradient parity, L4096 norm,
  padding/cache continuation, classifier streaming, and CUDA backward;
- Ruff, Python compilation, and `git diff --check` passed for the new runners,
  modules, and tests. Git reported only the checkout's existing LF-to-CRLF
  conversion warnings;
- a repository Markdown scan checked 221 local links. No new-scope link is
  missing. The 17 reported entries remain the 16 known paths into the
  then-uninitialized theorem submodule plus the known LaTeX `x_t` false
  positive in `FOUNDATIONS.md`; this predates the flattening integration.

## Multi-relation and motor verification

The later research additions were verified at
**2026-08-16T19:48:42+02:00**:

- 40/40 focused unit tests passed across the DeltaProduct reference,
  multi-relation `2.A5` runner/artifact, original center and unseen-relation
  cohorts, Spin scan, motor scan, and motor audit/artifact;
- Ruff formatting/checks and Python compilation passed for all new Python
  modules and tests;
- CUDA backward passed for DeltaProduct, Spin, and motor parallel scans;
- `git diff --check` passed, with only the existing LF-to-CRLF warnings;
- both new authoritative artifact hashes replay exactly;
- 170 local links in the changed/current-status documentation were checked.
  The only two unresolved targets are the already-known links into the
  then-uninitialized theorem submodule; this is a pre-flattening snapshot.

## Rigid learned/identification verification

Verification completed at **2026-08-16T20:25:40+02:00**:

- 49/49 focused tests passed across DeltaProduct, Spin scan, motor algebra and
  direct trackers, motor numerical audit, multi-relation `2.A5`, rigid
  translation-plus-center, and the original center/unseen-relation cohorts;
- the rigid tests replay all four authoritative artifact SHA-256 values and
  preserve both the failed gradient-trained gates and successful
  identification gates;
- all 16 newly referenced checkpoints were rehashed from disk with zero
  mismatches;
- Ruff checks and formatting, Python compilation, and `git diff --check`
  passed for the new runner/identifier/replicator, layers, and tests; Git emitted
  only the existing LF-to-CRLF warnings;
- 12 current/front-door documents were link-checked. The only unresolved links
  were two then-unavailable theorem-tree paths and the known LaTeX `x_t` false
  positive in `FOUNDATIONS.md`; this is a pre-flattening snapshot.

## Whole-SSM and final Spin(8) verification

Final integrated verification completed at **2026-08-16T22:37:10+02:00**:

- `python -m pytest -q SSM-Models`: 262 tests and 79 subtests passed in
  274.59 seconds on the release-integration tree; the two warnings are an
  upstream `pytz` deprecation and a
  recorded JAX future dtype-cast warning;
- the Ubuntu WSL custom-CUDA shard passed 5/5 Triton operator and benchmark
  tests against the same release commit in 7.43 seconds;
- Ruff formatting and checks passed across all 59 changed/new Python files;
- Python compilation and `git diff --check` passed; Git emitted only the
  checkout's existing LF-to-CRLF conversion warnings;
- a detached fresh checkout reproduced the frozen WSL Triton, Pure Spin(8)
  comparison, and octonion parity-audit SHA-256 values exactly and passed their
  10/10 focused evidence tests; `.gitattributes` now pins the intended JSON
  line endings and treats PyTorch checkpoints as binary;
- the maintained Pure Spin(8), frozen comparison, final-only, and parity-audit
  tests rehash and reload their committed artifacts/checkpoints; and
- all 169 compact PyTorch checkpoints (approximately 13.6 MB) were inspected
  for machine-local path strings before the explicit distribution decision in
  `PUBLICATION_SCOPE.md`; no such path was found; and
- after integration with current `main`, a repository-wide scan checked 1,168
  local Markdown targets. Its only reports are the intentionally non-public
  `.private/` directory named by `REPOSITORY_MAP.md` and the known LaTeX `x_t`
  false positive in `FOUNDATIONS.md`; no real local link is missing.

## Exact adjacent-octet determinant continuation

At **2026-08-17T00:02:40+02:00**, the maintained Spin-Space theorem tree was
reconciled with three new exact computer-assisted results:

- all ten coordinate faces of the five-variable adjacent endpoint-octet
  determinant are nonnegative; nine reduce to one-mode perfect squares and
  the tenth replays the hash-bound `Z=X^2` endpoint identity;
- strict Walsh diagonal dominance holds on the complete central cube
  `[1/4,3/4]^5`; and
- an adaptive 2,140-leaf exact atlas extends that strict result to
  `[1/8,7/8]^5`, with no unresolved leaf and exact minimum physical margin
  `320281275533252594456507202812099057 /
  5316911983139663491615228241121378304`.

The corresponding artifact SHA-256 values are, respectively,
`9a8988673ce4c5af4e0dca4b822b818b0f14e58364656bcc5e884e6f7edcbbec`,
`26611845fd3f5a5b5e63be145c04ede664e4ed033091fc74284381cb9c308fe2`, and
`cab3e87fb0c0c0ff9abb1c21cf2ea32bec9a92edc6364e225a5a66720486f762`.
The authoritative reports are
[`SPIN8_DIRAC_OCTET_DETERMINANT_BOUNDARY_RESULTS.md`](Spin-Space-Research/docs/experiments/SPIN8_DIRAC_OCTET_DETERMINANT_BOUNDARY_RESULTS.md)
and
[`SPIN8_DIRAC_OCTET_EXTENDED_CORE_DOMINANCE_RESULTS.md`](Spin-Space-Research/docs/experiments/SPIN8_DIRAC_OCTET_EXTENDED_CORE_DOMINANCE_RESULTS.md).

The complete `Spin-Space-Research/tests` suite passed **398 tests, 224
subtests, and one expected skip** in 2,165.47 seconds. Targeted theorem-chain,
artifact, gate-contract, Ruff, compilation, and documentation audits also
passed. These results do not cover the remaining width-`1/8` collars, the
unrestricted seven-variable Dirac--Gram inequality, or global five-query
D-optimality.

## Exact octonion-operator group classification

At **2026-08-17T00:20:13+02:00**, the finite groups implicit in the maintained
associative octonion-operator lift were classified. The seven imaginary-basis
left operators generate the plus-extraspecial group `2_+^(1+6)` of order 128.
All signed Fano-basis automorphisms form the non-split group
`2^3.PSL(2,7)` of order 1,344. Their order-eight intersection yields a split,
perfect, orientation-preserving group `2_+^(1+6):PSL(2,7)` of order 21,504.

The result is an exact identification of the repository's fixed matrix
embedding, not discovery of a new abstract finite group. The authoritative
report is
[`OCTONION_OPERATOR_GROUP_RESULTS.md`](Spin-Space-Research/docs/experiments/OCTONION_OPERATOR_GROUP_RESULTS.md),
and the broader inventory is
[`GROUP_AND_NUMBER_STRUCTURE_CATALOGUE.md`](research-programs/04-triality-clifford-representation-dynamics/GROUP_AND_NUMBER_STRUCTURE_CATALOGUE.md).
The exact replay additionally exhausts all 384 lifts of a fixed quotient
generating pair: 192 have product order seven and 64 also have commutator
order four; at least one of the latter generates an explicit split complement.
The artifact SHA-256 is
`835d0535d7827834ecd6b707984a7f3ae0eeac651f6fec4c3ff6f4ba74796a92`.

## Exact binary-icosahedral triality closure

At **2026-08-17T00:43:30+02:00**, the previously open closure of the vector
and two half-spin views of the fixed `2.A5` embedding was classified exactly.
The six common-carrier matrices preserve two four-dimensional blocks and two
spanning 120-point `H4` root orbits. Their faithful degree-240 permutation
action has order 864,000 and structure
`((2.A5 x 2.A5)/C2_diagonal) x 2.A5`.

The first block image is the order-7,200 orientation-preserving 600-cell
symmetry group; the second is one order-120 binary-icosahedral image. Equality
between the full order and the product of the two projection orders proves the
direct-product conclusion. The group is perfect, and its center is exactly the
Klein four-group of independent signs on the two blocks.

This proves that the fixed triality closure is finite but reducible. It is not
a new abstract finite group, an irreducible eight-dimensional exceptional
group, or evidence of sequence-model advantage. The authoritative report is
[`SPIN8_TRIALITY_2A5_CLOSURE_RESULTS.md`](Spin-Space-Research/docs/experiments/SPIN8_TRIALITY_2A5_CLOSURE_RESULTS.md),
and the artifact SHA-256 is
`ff238d047c94362136d70a9c57ae41c832d63bf15fb6263d0019456a9351cfc9`.

## Exact mixed monomial/golden SO(8) density

At **2026-08-17T01:29:02+02:00**, the next open embedding-generated group
question was resolved negatively. Adjoining the maintained order-21,504
monomial octonion-operator group to the vector, positive-half-spin, or
negative-half-spin golden image produces a topologically dense subgroup of
`SO(8)` in all three cases.

The same shortest witness word `FanoA FanoB b` works in every view. Exact
characteristic polynomials over `Q(sqrt(5))` contain the coefficient
`(1-sqrt(5))/4`, which lies outside `Z[(1+sqrt(5))/2]`; their rational
characteristic norms have coefficient denominators `{1,2,4}` and therefore
cannot be cyclotomic products. An exhaustive exact audit first verifies that
all 187 symmetric mixed words of length two have finite order.

An exact adjoint calculation then decomposes `so(8)` into irreducible
dimensions `7+21`, with commutant-system ranks 48, 440, and 782. The
seven-dimensional summand is not bracket closed; the 21-dimensional summand is
a Lie algebra but is not normalized by golden `b` in any view. Since infinitude
forces a nonzero identity component, the full `so(8)` algebra is the only
remaining possibility. This proves density but no quantitative spectral gap or
mixing rate. The authoritative report is
[`MIXED_MONOMIAL_GOLDEN_CLOSURE_RESULTS.md`](Spin-Space-Research/docs/experiments/MIXED_MONOMIAL_GOLDEN_CLOSURE_RESULTS.md),
and the artifact SHA-256 is
`4d6840bd89a0c58ac086c308e88f823928cc4d3206450f24bcc035f499d6d8c7`.

## Exact low-degree mixing band for the dense mixed groups

At **2026-08-17T01:53:16+02:00**, the first quantitative continuation of the
density theorem was completed. The fixed uniform symmetric labelled measures
have exact strict contraction certificates in the defining `8`, adjoint `28`,
and traceless-symmetric `35` representations. Sturm root counts prove the
defining bounds; exact `Q(sqrt(5))` LDL pivots prove both signed radius forms in
dimensions 28 and 35. The half-spin alphabets retain cross-source label
multiplicity: 21 labels represent 19 distinct matrices.

The result is explicitly finite-band. It does not promote to a spectral gap on
the full mean-zero `L2(SO(8))`, a total-variation mixing theorem, optimized
generator weights, or a model advantage. The authoritative report is
[`MIXED_MONOMIAL_GOLDEN_MIXING_RESULTS.md`](Spin-Space-Research/docs/experiments/MIXED_MONOMIAL_GOLDEN_MIXING_RESULTS.md),
and the artifact SHA-256 is
`0082b9df621dfe4b14c41a26cc914f124cebde37abb38974ed79c19411f7eac9`.

## Higher-weight Cayley bottleneck and compiled sandwich

At **2026-08-17T02:13:30+02:00**, exact spectral coverage was extended through
`Lambda^3` dimension 56 and both 35-dimensional Hodge halves of `Lambda^4`.
The maintained monomial subgroup has a one-dimensional fixed space in the
orientation-labelled Hodge-minus sector: an explicit Cayley-form line proved
by a rank-34 exact system. Sparse Rayleigh witnesses show that this sector, not
the earlier `8+28+35` band, is the first quantitative bottleneck.

A symmetric compiled `N-H-N` distribution now has exact six-band gaps greater
than `3/20` in the vector view and `3/100` in both half-spin views. Against the
exact original-walk witness bounds, these are strict macro-step improvements
above `3x` and `56/25x`. The boundary is explicit: one macro contains three
primitive letters unless a finite dictionary of at most 867 or 1,156 matrices
is precompiled, so no primitive-cost, full `L2(SO(8))`, or model advantage is
claimed. The authoritative report is
[`MIXED_MONOMIAL_GOLDEN_HIGHER_WEIGHT_RESULTS.md`](Spin-Space-Research/docs/experiments/MIXED_MONOMIAL_GOLDEN_HIGHER_WEIGHT_RESULTS.md),
and the artifact SHA-256 is
`d5b3fb35092d2fae603c546530502a95e04eb31ac1566035ef866fc7e033b1d6`.

## Exact macro compiler and local CPU/CUDA benchmark

At **2026-08-17T02:24:00+02:00**, the `N-H-N` sandwich was compiled into exact
finite dictionaries. The 867 vector labels reduce to 530 matrices; each 1,156
half-spin set reduces to 394. Multiplicity-weighted inverse symmetry and exact
equality with `M_N M_H M_N` are replayed, so the implementation does not
silently replace the theorem's labelled measure with uniform distinct-matrix
sampling.

On the reference single-thread CPU and synchronized RTX 2070 SUPER float32
benchmark, the direct labelled table beat online construction in all 24
view/device/batch cells: transition materialization improved by `1.92-8.07x`
and final-state application by `1.60-3.14x`, with maximum sampled error
`4.76837158203125e-7`. This is endpoint-only local evidence. Every-prefix scan,
backward, training, and end-to-end SSM speed remain open. The authoritative
report is
[`MIXED_MONOMIAL_GOLDEN_MACRO_COMPILER_RESULTS.md`](Spin-Space-Research/docs/experiments/MIXED_MONOMIAL_GOLDEN_MACRO_COMPILER_RESULTS.md).
Exact compiler SHA-256:
`9595578918bd46387ce5773607d1ded3d6117b11cb2b2353b586a1c6fc0cd438`.
Empirical benchmark SHA-256:
`93103d6fa5b4c36a43c89d8cac012d22deaa41a020c516d4c2b9ad9fc2bd8add`.

## Exact every-prefix chunk compiler and recurrent benchmark

At **2026-08-17T02:35:43+02:00**, the endpoint-only limitation was removed for
the fixed discrete `N-H-N` alphabet. Each labelled triple has an exact `24x8`
operator with row blocks `R`, `H R`, and `L H R`, so one application emits all
three causal prefix states. The full FP32 tables remain below 0.9 MB per view.

On recurrent sequences of 3–192 primitive steps, compiled endpoint execution
was `1.565-2.688x` faster and compiled every-prefix execution was
`1.007-2.394x` faster across all 72 recorded CPU/CUDA cells. Maximum recurrent
float32 error was `2.1457672119140625e-6`. The `1.007x` minimum is explicitly
near break-even; at that stage this was not a fused parallel-scan, backward,
training, or continuous-transition result. The later custom-kernel sections
below supersede only that implementation frontier. The authoritative report is
[`MIXED_MONOMIAL_GOLDEN_CHUNK_SCAN_RESULTS.md`](Spin-Space-Research/docs/experiments/MIXED_MONOMIAL_GOLDEN_CHUNK_SCAN_RESULTS.md).
Exact compiler SHA-256:
`ed1ae7e8ac98c5e037be4e45d10f22ec3236e7d6f8337fbc2b9f9a499e13e5de`.
Empirical benchmark SHA-256:
`35aef11f6e2577ac5848c800d5afb1dcbd38def2db3dcb5f3deb4dc820793f74`.

## Two-stage parallel chunk scan and initial-state backward

At **2026-08-17T02:49:15+02:00**, the compiled every-prefix path was promoted
from sequential recurrence to the maintained ordered work-efficient scan. The
primitive control scans `3C` matrices; the compiled path scans `C` endpoints
and applies all selected `24x8` local operators in parallel. At 64 chunks this
reduces tree compositions from 766 to 190.

Float64 tests match sequential recurrence, Hillis--Steele, and work-efficient
trees, including gradients with respect to `L`, `H`, `R`, and the initial
state. On the final single-thread CPU/synchronized CUDA artifact, forward wins
are `1.160-3.984x` and forward-plus-initial-state-backward wins are
`1.012-3.392x`; the minimum is explicitly near break-even. Maximum forward
error is `2.384185791015625e-6`, and maximum initial-state gradient error is
`7.450580596923828e-9`. This is eager PyTorch, not a fused kernel or full-model
backward. The authoritative report is
[`MIXED_MONOMIAL_GOLDEN_PARALLEL_CHUNK_SCAN_RESULTS.md`](Spin-Space-Research/docs/experiments/MIXED_MONOMIAL_GOLDEN_PARALLEL_CHUNK_SCAN_RESULTS.md),
and the artifact SHA-256 is
`73816cbcf8733ad9e2a4be8a87376ebfa96e369be64b3aa4093b3f0cafa93900`.

## Fused indexed local-prefix expansion

At **2026-08-17T03:02:51+02:00**, the selected `24x8` table lookup and local
matrix-vector product were fused into one Triton program per `(batch, chunk)`.
A custom transpose-matvec backward covers the incoming state; CPU,
unsupported dtype, and trainable-table inputs retain eager semantics.

Across the 27 synchronized CUDA cells, isolated forward beat realistic
indexed eager in 26 cells (`0.950-3.238x`, median `1.245x`) and isolated state
backward won 25 (`0.958-1.409x`, median `1.051x`). In the complete two-stage
pipeline this fell to 16/27 forward and 14/27 backward wins, locating the
eager endpoint tree as the remaining bottleneck. Maximum errors were
`4.76837158203125e-7` forward and `3.725290298461914e-9` for the measured
state gradient. The authoritative report is
[`MIXED_MONOMIAL_GOLDEN_TRITON_LOCAL_PREFIX_RESULTS.md`](Spin-Space-Research/docs/experiments/MIXED_MONOMIAL_GOLDEN_TRITON_LOCAL_PREFIX_RESULTS.md),
and the artifact SHA-256 is
`b124288e9e68d0513e66f39e5fd20c5c831b49d824be28d73ae76305040c6f17`.

## Register-resident compiled chunk recurrence

At **2026-08-17T03:10:03+02:00**, the bottleneck result was followed by a
one-program-per-sequence Triton recurrence. It retains the eight-state in
registers, walks frozen exact labels, emits all three causal prefixes, and
implements a reverse recurrence for the initial state.

Against the optimistic pre-gathered parallel control, all 27 forward cells won
by `4.709-19.861x` (median `8.344x`) and all 27 forward-plus-initial-backward
cells won by `2.213-6.465x` (median `2.755x`). Maximum errors were
`2.384185791015625e-6` forward and `1.4901161193847656e-8` for the initial
gradient. The kernel is serial in chunk depth, tested through 64 chunks, and
does not train the table or selector; it is not a parallel-prefix theorem or
end-to-end SSM comparison. The authoritative report is
[`MIXED_MONOMIAL_GOLDEN_TRITON_CHUNK_RECURRENCE_RESULTS.md`](Spin-Space-Research/docs/experiments/MIXED_MONOMIAL_GOLDEN_TRITON_CHUNK_RECURRENCE_RESULTS.md),
and the artifact SHA-256 is
`b6897f60a011f2dcb0788fccc32195277a54c600f87199d8d13f75b043f37a7d`.

## Latent Pure Spin(8) identification and compiled model runtime

At **2026-08-17T03:39:19+02:00**, the frozen latent-increment cohort completed
fresh seeds 1--3. Training sees every token and every ordered pair except
`a,a`, while every prefix in all three triality representations is supervised.
All three Pure Spin(8) routers identify the local action table, achieve 100%
center/identity row correctness through L128, and record six L128 post-relation
MSEs in `2.66e-5`--`2.87e-4`. Parameter-near Mamba-2 and GRU references remain
near chance on the central distinction. The aggregate artifact SHA-256 is
`c3d49145fb710c43aa087262212e4005f887995ffb67f399a49afb57e8ae51a2`.
This is finite-dictionary every-prefix identification, not natural-input,
state/compute-matched, or generic language-model superiority.

At **2026-08-17T03:46:46+02:00**, Pure Spin(8) v1.1 compiled the passing seed-1
router into a frozen `[8,3,8,8]` faithful action table. A register-resident
Triton recurrence preserves every held-out relation signature and matches the
dynamic source within `1.32e-5` over the benchmark grid. Synchronized local
forward speedups are `30.7x--67.1x`, with median `42.75x`; at B8/L1024 peak
allocation falls from 509.6 MB to 11.0 MB. Artifact SHA-256:
`aa19ba66e5e2d17967f189c4744a1f1c165e0181b11c999f6cd0e4329dd6fb55`.
The runtime is serial in sequence depth and inference-only; it is not a
parallel-prefix, fused-training, continuous-routing, or fused-Mamba result.

## Noisy continuous Pure Spin(8) identification

At **2026-08-17T04:27:24.147531+02:00**, the frozen continuous-observation
cohort completed fresh seeds 1--3. Unique 12-real noisy observations hide seven
active Spin(8) coordinates behind a different injective nonlinear chart per
seed. Training excludes adjacent half-center pairs and supervises every
24-real triality prefix. All 135 frozen per-seed gates pass, including strict
rehash/reload of 18 checkpoints. Shared Spin(8) action RMSE is
`0.01334--0.01468`, every L128 center/identity row is correct, and its six L128
post-relation MSEs are `0.01216--0.02814`. A capable 957-parameter, exactly
24-state independent `SO(8)^3` control also classifies every relation but has
`3.295x` larger median MSE. Aggregate artifact SHA-256:
`34238a1d98fa467e8f8f38b1f90d1a24bc2495cc510934b4327cba81e09ebc6`.

At **2026-08-17T04:40:35.261975+02:00**, the separately frozen measured-wall
continuation completed. One maximal schedule per seed supplies identical
prefixes, schedule construction is excluded from timing, and all 3,381,248
observations per seed are unique. Frozen candidate allocations from 636 to
6,604 updates land within `1.99--2.97%` of shared update wall on the RTX 2070
SUPER. Shared median L128 MSE remains `0.01860`, versus `0.07476` independent,
`0.13288` Mamba-2, `0.13707` parameter-near GRU, `0.13181` state-matched GRU,
and `0.14099` observation-only. A second independent adjudicator rehashes and
strictly reloads all 18 wall checkpoints. Aggregate artifact SHA-256:
`262be117892cc2b511eaa3edde0ccea6695533dd9a621312d43c42dafaf33a02`.

This upgrades the local evidence from finite-token identification to online
noisy continuous action inference and isolates a shared-action inductive-bias
advantage from parameter count, recurrent-state size, and local measured update
wall. At this adjudication stage it remained an injective seven-coordinate,
every-prefix synthetic task; the endpoint-only successor below removes that
specific supervision dependency. The Mamba row is unfused, and natural data,
unsigned/partial observations, all 28 tangent coordinates, fused training, and
language-model utility remain open. The
authoritative report is
[`PURE_SPIN8_CONTINUOUS_OBSERVATION_RESULTS.md`](SSM-Models/experiments/PURE_SPIN8_CONTINUOUS_OBSERVATION_RESULTS.md).

## Endpoint-only continuous Pure Spin(8) identification

Documentation reconciled: **2026-08-17T05:46:54.0059683+02:00**

At **2026-08-17T05:17:13.461565+02:00**, the frozen endpoint-only cohort
completed fresh seeds 1--3. It retains the earlier noisy observation charts,
hidden seven-coordinate action family, architectures, and excluded adjacent
center relation, but removes every intermediate target. Each length-16 sequence
exposes only its final signed 24-real triality state: 64,000 endpoint targets
for 1,024,000 unique observations per seed. The batch schema has no prefix-
target field, model inputs contain observations only, and the unit contract
requires exactly zero loss gradient at every nonfinal prediction.

All 51 frozen checks pass independently in each seed. Shared Spin(8) records
median L128 post-relation MSE `0.01296` over `0.00840--0.02014`, versus
`0.06268` over `0.04161--0.09505` for the capable 957-parameter, exactly
24-state independent `SO(8)^3` family. Both structured rows achieve 100%
classification and row correctness on every L128 split. Mamba-2,
parameter-near GRU, observation-only, and state-matched GRU medians remain
`0.12813--0.13301`. The independent/shared median ratio is `4.8365x`.
All 18 primary checkpoints strictly rehash and reload. Aggregate SHA-256:
`1cf51a4af05303bc3ca9e781478e2352e8dbb077d1c9b367f46af2f384653880`.

At **2026-08-17T05:40:55.849697+02:00**, the separately pre-frozen measured-
wall continuation completed. Each seed uses one maximal deterministic schedule
with 7,926,784 unique observations, 495,424 endpoint targets, no intermediate
targets, and zero excluded adjacent pairs. Candidate allocations of
1,558--15,482 updates land within `1.33--1.97%` of the shared model-update wall
on the RTX 2070 SUPER. Shared median L128 MSE remains `0.01296`, versus
`0.09080` independent, `0.12753` Mamba-2, `0.12787` observation-only,
`0.15422` parameter-near GRU, and `0.15049` state-matched GRU. Extra GRU
updates reduce some training losses but can worsen L128 extrapolation. All 18
wall checkpoints strictly rehash and reload. Aggregate SHA-256:
`538a3bdbddfd76863a5bef5507a6d0019a114b35021d7b5b9d1223d31983ac64`.

This closes the dense every-prefix-supervision objection for this fixed,
injective, signed synthetic teacher family. It does not establish unsigned or
partial observation recovery, noninjective/chart-shift robustness, all 28
tangent coordinates, natural-task utility, fused modern-SSM parity, FLOP or
energy equality, or language-model superiority. The authoritative report is
[`PURE_SPIN8_ENDPOINT_SUPERVISION_RESULTS.md`](SSM-Models/experiments/PURE_SPIN8_ENDPOINT_SUPERVISION_RESULTS.md).

## Endpoint observability and the double-cover information boundary

Documentation reconciled: **2026-08-17T06:41:37.6166325+02:00**

At **2026-08-17T06:11:01.888654+02:00**, the exact endpoint-observability
certificate completed. Multiplying the maintained triality generators by two
makes them integral; exact SymPy ranks over the rationals on one through eight
basis probes are `7,13,18,22,25,27,28,28` in `8v`, `8s+`, and `8s-`. The
tested center acts as `(+I,-I,-I)`. Plane-0 coordinates `0` and `2*pi` give
identical vector actions and opposite positive-spinor targets, proving balanced
hidden-lift Bayes MSE `1/8` and accuracy `1/2`. Certificate SHA-256:
`fa29a9d74a927993c17328b7dffb5f96c7f42f308b2e30450d4f714a9ce89a53`.

At **2026-08-17T06:39:25.816374+02:00**, the frozen partial-readout cohort was
adjudicated and **failed** without median rescue. Seeds 1--3 pass `37/40`,
`39/40`, and `39/40` gates. Vector-only shared supervision misses at least one
exact hidden-spinor center row in every seed; minimum center accuracy is
`0.984375`. Seed 1 also leaves the independent positive-only supervised control
near chance, failing its capability and center gates.

The failed aggregate contains a strong but narrower replicated stratum. With
only eight final `8s+` or `8s-` scalars in the loss, shared Spin(8) transfers
the action into all three views in all seeds. Positive-only all-view L128 MSE
spans `0.00974--0.02492`; negative-only spans `0.00932--0.02520`; every shared
spinor center row is correct. Paired-spinor and full-readout masks pass every
gate. These strata do not rescue the failed all-mask protocol.

The independent validator regenerates all three 1,024,000-observation training
schedules and all 18 evaluation schedules, reproduces their hashes, and
strictly rehashes/reloads all 30 fresh checkpoints. Every integrity gate passes.
Failed aggregate SHA-256:
`baed378d569391e86c46df731cfc72db4f0c0a24d21883bb17a4604db9e5c987`.
The authoritative report is
[`PURE_SPIN8_ENDPOINT_OBSERVABILITY_RESULTS.md`](SSM-Models/experiments/PURE_SPIN8_ENDPOINT_OBSERVABILITY_RESULTS.md).

## Adaptive lift-bit calibration

Documentation reconciled: **2026-08-17T07:34:13.2082201+02:00**

The successor protocol was frozen at
**2026-08-17T07:05:22.8794763+02:00**, before fresh seeds 4--6. It tests the
minimum lift-odd repair while preserving the endpoint teacher, noisy injective
chart, excluded relation, L16 final-only supervision, optimizer, 2,000 updates,
and L128 tests. For unit half-spin endpoint `y`, the address/sign chart
`j=argmax |y_j|`, `b=sign(y_j)` has lift-invariant address, a sign that flips
on `{y,-y}`, and selected magnitude at least `1/sqrt(8)`. The complete
calibration word has three address bits plus one sign bit; only the sign bit
contains lift information.

At **2026-08-17T07:32:36.941651+02:00**, the strict fresh cohort was
adjudicated and **passed**. Every one of 13 gates passes independently in each
seed, with no median rescue. Shared adaptive action RMSE spans
`0.012825--0.013633`; all-view L128 MSE spans `0.009198--0.026986`; every L128
adaptive bit, half-spin center row, and relation address/opposite-bit check is
exact. Relative to identically initialized vector-only training, median action
RMSE falls by `73.4%--73.7%` and median L128 MSE by `73.3%--78.0%` across
views. The fixed-coordinate sign ablation is worse than vector-only, with L128
MSE as high as `0.191420`.

The validator regenerates all three 1,024,000-observation schedules, the
observation systems, addresses, bits, and all 18 evaluation schedules; it
rehashes/reloads all 30 checkpoints and recomputes every metric. Aggregate
SHA-256:
`fb89fd75d5aa7c3b16448844225baf838aeb4bdf40cb62ba1276f07ac7503b69`.

This closes the frozen empirical failure only with an externally supplied
four-bit calibration interface. It does not infer the address or bit from
`8v`, construct a global continuous section, cover chart-boundary noise,
unknown initial lifts, natural inputs, all 28 trained coordinates, or fused
modern-SSM throughput. The authoritative report is
[`PURE_SPIN8_LIFT_BIT_CALIBRATION_RESULTS.md`](SSM-Models/experiments/PURE_SPIN8_LIFT_BIT_CALIBRATION_RESULTS.md).

## Lift-gradient and scrambled-alignment boundary

Documentation reconciled: **2026-08-17T08:58:51.9207930+02:00**

The exact gradient certificate regenerates the first frozen adaptive batch for
seeds 4--6 and traces the final checkpoints. All independent vector/positive
head weights and all 108 shared-trunk weights receive nonzero gradient; all 252
negative-head weights and 28 biases have bitwise-zero data gradient. After
2,000 AdamW updates, that negative block equals the exact repeated decay-only
counterfactual with residual `0.0` in every seed. All 28 shared Spin(8)
coordinate rows receive nonzero gradient. Certificate SHA-256:
`dee7c22e94bd627704609b7cad58939532e11c583a483e0e73987149f9339ab5`.

The stronger protocol froze at **2026-08-17T08:40:40.5270068+02:00**. Its
scrambled control keeps the shared model's 12-to-22-to-28 router initialization
and 24-scalar state, adds 56 trainable legal Spin(8) alignment parameters, and
is independently capable under full supervision. At
**2026-08-17T08:54:32.136060+02:00**, the strict seeds 7--9 aggregate
**failed**: exactly one of 36 seedwise gates is false because seed 7 scrambled
alignment slightly beats the maintained model in two directly supervised
vector-L128 cells. No median rescue is applied.

The preserved narrower stratum is exact across the cohort: correct alignment
wins `9/9` action, `12/12` spinor-L128, and `6/6` completely hidden negative-
L128 comparisons, but only `4/6` vector-L128 comparisons. Median adaptive
negative action RMSE is `0.013999` shared versus `0.185325` scrambled; full
supervision repairs scrambled negative action to `0.011859`. The adaptive
negative alignment follows decay only, while full supervision updates both
alignments. This supports bounded cross-view spinor transfer and disproves the
frozen universal all-view headline. Aggregate SHA-256:
`ec6802d9c55f318aa85aaacb9ce4030df697f716158a3bdc5752432394f044a7`.

Authoritative reports:
[`PURE_SPIN8_LIFT_GRADIENT_IDENTIFIABILITY_RESULTS.md`](SSM-Models/experiments/PURE_SPIN8_LIFT_GRADIENT_IDENTIFIABILITY_RESULTS.md)
and
[`PURE_SPIN8_SCRAMBLED_ALIGNMENT_RESULTS.md`](SSM-Models/experiments/PURE_SPIN8_SCRAMBLED_ALIGNMENT_RESULTS.md).

## Exact alignment-calibration threshold

Documentation reconciled: **2026-08-17T10:34:43.2866889+02:00**

The negative-only calibration protocol froze at
**2026-08-17T09:52:34.5014289+02:00**, before fresh seeds 10--12. It keeps the
maintained 930-parameter router and 24-scalar state bitwise matched, adds one
28-parameter negative-view alignment, and transfers only external ordered
basis probes to the alignment optimizer. No negative sequence endpoint target
is used.

The exact probe map `T -> (T e_1,...,T e_m)` has fiber `SO(8-m)` and ranks
`0,7,13,18,22,25,27,28,28`. Rational quarter-turn stabilizers now certify
global action non-identifiability for every `m<=6`; seven probes globally
determine the `SO(8)` action and eight are redundant. This does not determine
factorized coordinates or a discrete Spin-cover lift. Exact certificate
SHA-256:
`0a1c6ea0107aa732a0656bbb739b1e1eab650eabf40d64ad6229f42810255124`.

At **2026-08-17T10:34:43.2866889+02:00**, strict fresh adjudication completed.
Every source, schedule, observation, evaluation, checkpoint, reload, replay,
and router-identity check passes. Seeds 11 and 12 pass all 14 frozen gates;
seed 10 passes 12/14. Its nonzero rank-27 frame residual is small enough that
total action error remains near the common router floor, failing the frozen
factor-of-two gate, and it narrowly wins one late-L128 cell. No aggregate
statistic rescues the failed headline.

The surviving empirical stratum is replicated: every nonzero probe frame is
fit, every seven-probe alignment has full-frame RMSE at most `1.255e-21`, and
seven/eight probes reproduce the aligned action and L128 metrics in every
seed. Failed aggregate SHA-256:
`1708d2932cce32f0b1715c1563af35686aa096e7020fe0cbd80ed7f67a2bad2a`.
The authoritative report is
[`PURE_SPIN8_ALIGNMENT_CALIBRATION_RANK_RESULTS.md`](SSM-Models/experiments/PURE_SPIN8_ALIGNMENT_CALIBRATION_RANK_RESULTS.md).
