# Documentation reconciliation — 2026-08-16

Reconciliation opened at **2026-08-16T16:05:27+02:00** and latest
research status reconciliation completed at **2026-08-16T22:37:10+02:00**
(`Africa/Johannesburg`, UTC+02:00).

## Scope and method

This pass inventories the documentation visible in this checkout, rather than
retroactively rewriting historical reports. After integration with the
flattened theorem tree, the latest `rg --files` inventory is 429 Markdown/RST
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
- `SSM-Models/pure_spin8_ssm/`, version 1.0.0, is a second maintained model
  family with a separate checkpoint schema. It does not replace or silently
  upgrade Pure Rotor v2.1.
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
