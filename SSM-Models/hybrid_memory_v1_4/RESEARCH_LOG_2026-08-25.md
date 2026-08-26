# Hybrid Memory research log: 2026-08-25

This is a decision and evidence log, not a manuscript and not a replacement
for [`RESULTS.md`](RESULTS.md). It records negative paths as well as successful
contracts so later work cannot reconstruct a cleaner story than the evidence.

## Learning diagnosis

The present v1.4.5 failure is not ordinary text fitting, global retention, or
missing gradient credit. G11--G13 show that the recurrent layer carries most
ordinary compression and that the longer curriculum improves 4,096-token loss
in every seed. Rare one-shot bindings still disappear because ordinary tokens
write strongly into the same small fast-weight matrix. A retention floor
protects untouched directions; it cannot protect the direction being erased
and rewritten on each content update.

The primary repair is therefore an edit-law and objective intervention:
independent erase/write controls, a genuinely content-addressed association
state, sparse/protected admission where justified, and an explicitly named
delayed-binding objective alongside ordinary next-token loss. Tokenizer and
optimizer changes remain useful controls, but neither creates an identifiable
archive by itself.

## G14

The prospectively frozen G14 mechanism screen passed its narrow theorem-backed
gate. Decoupled GDN2 can accumulate eight value features at one address while
the tied GDN-v1 law has a `49/64` MSE lower bound. Three trained GDN2 seeds
reached 100% bit accuracy; tied GDN-v1 reached 0%. This is an unequal-parameter
constructed state-law result, not language evidence. The maintained semantic
GDN2 layer was integrated into the common shell with scan, chunk, mask,
gradient, auxiliary-loss, state-byte, and initialization contracts.

## SpinDirac redesign

The old structured 24-scalar cache transported values but lacked a credible
address/edit/retrieve law. The replacement stores one $8\times8$ fast-weight
association per head and gives geometry only the transport and Clifford-read
roles. Its two-sided factor scan is exact in float64 and streaming state is
constant-size.

The first draft used channelwise retention/erase/write. The prospective inner-
conjugation audit exposed that this chooses a fixed coordinate basis. Before
any G15 training, the primary law was changed to independent head-scalar
erase/write/retention; the channelwise version is retained as a named
non-equivariant ablation. This repair makes the address edit, transported
state, positive read, and Clifford-coupled negative read transform in one
shared Spin frame.

Implemented fixed transports are identity, `SO(2)^4`, the exact constrained
`SU(3)` rank-two torus, full factorized Spin(8), and a deliberately broken
carrier coupling. No moving (G_2/SU(3)) frame is implemented or claimed.

## Optimizer and tokenizer

Repository evidence rejects a universal optimizer story. The retained
`HarmonicMuonAdamW` composite assigns hidden matrices to Muon, geometric/edit
controllers to a custom scalar-second-moment AdamW update, and embeddings,
norms, biases, and convolution to ordinary AdamW groups. The scalar moment is
orthogonally covariant in the tested mapping and now rejects zero epsilon;
checkpoint continuation is tested against an identical replica.

The train-only lossless ByteLevel BPE remains the selected local tokenizer.
It improved bounded BPRB/compute allocation, but G12/G13 show it is not a
long-range recall cure. Tokenizer comparisons must report raw bytes, token
targets, context in both units, and attention-window exposure.

## Native SM75 work

Source-current Mamba, FLA, causal-conv1d, and the Turing FlashAttention fork
were cloned into the WSL Linux filesystem and installed into dedicated
environments. Passing probes bind the install to the exact source checkout.

- Mamba-3 SISO native training forward/backward passed with complete finite
  parameter and input gradients.
- Mamba-3 MIMO failed because its TileLang path requires an SM80 TF32 MMA
  instruction; it is excluded.
- FLA GDN2 recurrent produced finite outputs but did not propagate gradients
  into the recurrent-core projections; it is excluded as a training baseline.
- FLA GDN2 chunk forward completed, while backward exceeded the bounded
  1,200-second attempt; it is excluded pending a real qualifying run.
- The source-built Turing FlashAttention extension passed causal/noncausal
  FP16 forward/backward numerical gates at head dimensions 64 and 128.
- Actual 187M Mamba-3 weights executed on SM75 with the declared tokenizer
  family. This is checkpoint execution, not a matched quality result.
- Actual 130M Mamba-2 weights also executed after source-building the current
  causal-conv1d dependency; its GPT-NeoX tokenizer and all file/revision hashes
  are bound separately.

Full paths, revisions, packages, and claim boundaries are in
[`SM75_NATIVE_RUNTIME.md`](SM75_NATIVE_RUNTIME.md).

## Documentation and experiment organization

- [`RESULTS.md`](RESULTS.md) remains the complete chronological model ledger.
- [`FRONTIER_REVIEW_2026-08-25.md`](FRONTIER_REVIEW_2026-08-25.md) explains the
  architectural pivot and method map.
- [`SPIN_TORUS_RESEARCH.md`](SPIN_TORUS_RESEARCH.md) owns the fixed/moving
  geometry boundary and transported-kernel interpretation.
- [`G15_SPIN_DIRAC_RESULTS.md`](G15_SPIN_DIRAC_RESULTS.md) owns G15 integrity
  status and will receive the learning adjudication.
- [`SM75_NATIVE_RUNTIME.md`](SM75_NATIVE_RUNTIME.md) owns local implementation
  eligibility. Runtime smoke and model quality are never merged.

## Next decisions

1. The G15 integrity artifact now passes training-dtype 4,096-step growth,
   optimizer/SGD mapped covariance, and delayed scored-position perturb-and-
   descent on the local SM75 runtime.
2. Run G15A. Include I, I+C, C, and S;
   add the exact `SU(3)` torus as a scientific ablation.
3. If S passes, run identity-read and broken-coupling controls before using
   "triality-specific."
4. Hold the winning edit law fixed for generic association and three-seed
   natural-text tests.
5. Resume `256 -> 512 -> 1,024 -> 2,048 -> 4,096` only with parameter, state,
   target-token, raw-byte, and synchronized-compute accounting.
6. Implement conjugate torus banks or moving (G_2) frames only after an oracle
   moving-frame task falsifies the fixed-torus family.

## G15A operational freeze

A second audit found that the structural G15A preregistration still lacked
execution-critical variables: exact seeds, training budget, task support,
dtype, aggregation, retry policy, and a supplied-coordinate pathway. Running a
cohort at that point would have made those choices post hoc.

The prospective
[`G15A execution protocol`](G15A_EXECUTION_PROTOCOL_2026-08-25.md) now freezes
those variables before any runner metric is inspected. The implementation adds
validated externally supplied Spin coordinates through the full model and an
explicit oracle-control semantic path for one-hot, overwrite, collision,
orthogonal-query, and inner-conjugation checks. The quality runner is required
to start from a clean committed worktree. This log entry records protocol
readiness, not a G15A result.

## G15A primary result and attribution freeze

The quality cohort was then executed from clean commit `73df687f` on the
source-bound SM75 WSL runtime. It passed every frozen condition in all three
seeds. S reached 1.00 macro accuracy on the supplied-coordinate symmetry task;
C reached 0.20; I and I+C reached 0.10. All four arms learned the separate
finite no-symmetry delayed-value task to 1.00 accuracy at lengths 64, 256, and
1,024. Parameter shapes, schedules, state bytes, and execution budgets were
matched; numerical semantic and inner-conjugation gates passed.

The learning problem is therefore split cleanly rather than declared solved:

1. The content-addressed edit law and bounded recurrent shell can learn the
   finite delayed-value controller task.
2. Full noncommuting transport is necessary relative to identity and one fixed
   torus on the designed supplied-coordinate action support.
3. Coordinate discovery was not learned on the symmetry task.
4. The result does not yet identify the fixed Clifford read or shared triality
   lift as the cause; marginally rich orthogonal transport may suffice.

Before inspecting either required control, the
[`conditional-controls protocol`](G15A_CONDITIONAL_CONTROLS_PROTOCOL_2026-08-25.md)
froze `S+identity-read` and `S-broken` against the immutable primary artifact.
The per-seed two-point margin is binding, with no seed averaging. This is the
next executable falsifier before G15B or any triality-specific wording.

## G15A conditional attribution outcome

The first conditional execution completed all metrics but exited nonzero
because the runner incorrectly treated S-broken's intentionally failed
inner-conjugation diagnostic as an integrity gate. The already frozen protocol
said explicitly that conditional residuals were diagnostic only. The raw
first-run artifact was retained outside Git, the adjudicator and regression
test were corrected in commit `5fc3d7b`, and the identical cohort was rerun
from that clean pushed commit. No seed, data, threshold, optimizer, or training
setting changed.

The evidentiary rerun passed its execution integrity gate. S+identity-read tied
S at 1.00 in all three seeds, rejecting a necessary contribution from the
fixed Clifford/negative-spin read. S-broken scored 0.30, 0.20, and 0.20 against
S at 1.00, passing the shared-coupling attribution margin by 0.70, 0.80, and
0.80. Both controls learned the finite no-symmetry task to 1.00 through length
1,024. S-broken's `0.071--0.078` covariance residual confirms that the control
breaks the common lift.

The result supports a shared vector/positive Spin lift on this designed,
supplied-coordinate task. It does not support necessity of the negative-spin
Clifford read or usefulness of all three triality carriers. The next learning
problem is to infer transport coordinates from observable action tokens under
end loss; adding more exceptional geometry before that controller passes would
not address the remaining bottleneck.

## G15A-L learned-coordinate freeze

Before running a smoke or quality row, the
[`G15A-L protocol`](G15AL_LEARNED_COORDINATE_PROTOCOL_2026-08-25.md) freezes a
minimal 17-token by 28-coordinate controller. It trains only that table from
delayed positive-read loss with `ScalarSecondMomentAdamW`; edit controls and
the exact transported final query stay oracle-fixed. Training uses short random
primitive compositions, while evaluation uses fresh 8/12/16-action
compositions spaced across lengths 64/256/1,024.

This design admits the hard alternative rather than baking in success:
S-broken learns its own coordinate table. If it inverts the frozen carrier
permutation and catches S, then the supplied-coordinate attribution is not
identifiable under learned control. The per-seed/per-length gate and the
covariant optimizer are fixed before seeing that outcome.

The first non-evidentiary smoke showed that dense evaluation time was dominated
by identity filler rather than learning. Before quality, the prospective
[`exact event-sparse amendment`](G15AL_EXECUTION_AMENDMENT_2026-08-25.md)
replaced those redundant identity actions by the closed-form scalar retention
factor and ordered products over actual events. A float64 regression requires
agreement with dense `forward_controls` to `1e-10`; no scientific setting or
gate changed.

## G15A-L quality failure

The clean quality run at commit `0c49f64` completed all fifteen arm/seed rows
and failed the preregistered gate. S reached high but sub-threshold cosine on
most rows and never beat every comparator by 0.05. S-broken was equal to S to
roughly `6e-8` across all nine evaluation rows.

The retained tables identify the mechanism. The learned broken chart, after
its frozen signed permutation is applied, equals S's effective positive chart
within `4e-7`. Cosine scoring normalizes away the scalar
`<q,Vk>`, so the vector action is invisible while that scalar remains positive.
The controller can invert the positive-carrier chart without respecting a
shared lift. This is exactly the gauge/observability failure the protocol was
designed to expose.

The optimizer did learn nontrivial charts, and the S/S-broken losses agree by
symmetry. More steps, AdamW, or a larger controller cannot by themselves make
an unobserved carrier identifiable. The next task should expose a full-rank
association through multiple queries and raw Frobenius error, so both carrier
actions affect the scored object.

## G15A-F observability repair freeze

The post-hoc diagnostic reproduced the G15A-L checkpoints and separated the
hidden raw error from the scored quotient. Across the three seeds,
S/S-broken effective positive charts agree within `3.94e-7`, and their output
cosine is at least `0.99999988`; nevertheless, individual raw predictions
differ by as much as `0.284`. This verifies that the old metric hid a real
carrier mismatch rather than proving the two transports equal.

An independent design audit then rejected a single full-frame probe as still
gauge-degenerate. Before any G15A-F smoke or quality metric, the
[`four-probe full-frame protocol`](G15AF_FULL_FRAME_PROTOCOL_2026-08-25.md)
was frozen. Each action composition now transports the same four deterministic
orthogonal association frames and is scored by raw Frobenius MSE. The run must
stop before optimization unless the independent vector/positive tangent
Jacobian has rank 56 with adequate conditioning, every target primitive has a
fixed residual outside the broken tied-coordinate image, and an exhaustive
integer-scaled Lie-bracket check proves that the broken coordinate map is not a
Lie-algebra automorphism.

The seeds, 300-update controller budget, scalar-second-moment optimizer, action
support, and long-composition evaluations are otherwise held fixed in kind
from G15A-L. This isolates the observation law. A pass would still be only
multi-probe, oracle-frame controller evidence up to a common discrete center;
it would not be generic memory or language evidence.

## G15A-F quality failure with restored separation

The clean exact-SM75 cohort at commit `503fa82` passed all pretraining
certificates and completed in 64.5 seconds. The four-probe Jacobians have full
rank 56, condition ratios `0.286--0.310`, and minimum target residuals above
`0.598` outside the broken chart. The exhaustive broken-map certificate records
474 nonzero Lie-bracket mismatches out of 784 ordered pairs.

S then beat identity, the fixed torus, and S-broken by the required `0.05` mean
relative-error margin in every seed and length. This is the separation that
G15A-L's cosine quotient could not express. But S itself remained above the
frozen absolute error ceiling in all nine rows: `0.0705--0.0927` at L64,
`0.1036--0.1201` at L256, and `0.1304--0.1433` at L1,024. P95 errors also fail
throughout, and the two-times-broken check passes only two rows.

The learning problem has therefore moved. It is no longer structural blindness
to the vector carrier; it is accurate chart recovery and accumulation under a
stochastic composition-only curriculum. The final S singleton errors are much
smaller (`0.0166--0.0319`) than its long-composition errors. The next read-only
diagnostic will project the learned table onto exact primitive support while
preserving learned amplitudes, separating inactive-coordinate leakage from
active-angle bias. Optimizer or curriculum changes will be frozen only after
that decomposition is recorded.

## G15A-F chart decomposition and G15A-R freeze

The source-bound SM75 diagnostic confirms that inactive-coordinate leakage is
the dominant learning error. Across seeds, active target-axis MAE is only
`0.00189--0.00324`, while inactive RMS is `0.00394--0.00780`. When the learned
tables are projected onto the oracle primitive support but keep their learned
amplitudes, all nine long-composition rows clear the original G15A-F absolute
gates: mean `0.0051--0.0183`, p95 at most `0.0270`, maximum at most `0.0392`.
Exact amplitudes with learned leakage remain near the failed result. This is a
post-hoc decomposition, not a rescued promotion.

Before inspecting a new metric, the
[`G15A-R protocol`](G15AR_FIRST_ORDER_PROTOCOL_2026-08-25.md) freezes a paired
first-order ablation. It tests staged LR decay first, then one scalar second
moment per token row, then a balanced singleton/inverse curriculum, with a
fixed least-intervention selection order. `BlockScalarSecondMomentAdamW`
preserves the 28-coordinate orthogonal covariance that motivated the existing
global scalar optimizer while no longer coupling unrelated token rows through
one denominator. The winner, if any, must be reinitialized and pass the full
I/C/S/S-broken cohort on three untouched seeds. No support labels or post-hoc
projection enter training.

## G15A-R passes composition-only chart learning

The clean SM75 run from commit `eca70f0` completed in 290.6 seconds and passed
the complete frozen programme. All five development recipes qualified. The
longer 600-step fixed-LR/random control reached mean errors
`0.0155--0.0285`; therefore the earlier 300-step miss was not a hard optimizer
or representation failure, and LR decay is not shown necessary.

The predeclared selectable order chose `G-decay/random`. This retains the
original dense 476-scalar table, global rotation-covariant scalar moment,
random two-to-six-action composition data, raw Frobenius loss, and four-probe
observation. Only the 600-update budget and staged learning rate change. It
reached `1.03e-7--1.85e-7` mean error on development. Block-scalar moments and
the primitive/inverse curriculum also solve the task but add no demonstrated
need.

Fresh confirmation reinitialized I, C, S, and S-broken on seeds 2251, 2267,
and 2273. S mean error was `1.21e-7--1.89e-7`, p95 at most `2.68e-7`, and
maximum at most `3.27e-7` across all nine length rows. Its minimum margins were
`0.234` over I, `0.231` over C, and `0.153` over S-broken; every broken-two-
times and pairing check passed.

The present learning problem is now sharply advanced: a token controller can
learn the shared vector/positive Spin coordinate chart from composition-only
end loss when the observation exposes both carriers. Oracle frame probes and
edit timing remain, so G15B must learn address/write/query behavior on generic
association while holding this recipe and the four transports fixed. That is
the next bottleneck; additional exceptional geometry is not.

## G15A-S spans the chart and closes the center-sensitive pre-G15B gate

The complete history was retained rather than flattened: G15A supplied exact
coordinates; G15A-L failed because cosine removed a carrier; G15A-F repaired
observability but exposed leakage; G15A-R repaired precision. Before advancing
to G15B, the
[`G15A-S protocol`](G15AS_SPANNING_CENTER_PROTOCOL_2026-08-25.md) froze the
strongest remaining G15A test: 56 hidden signed tokens covering all 28 planes,
disjoint train/evaluation probe pools, and unseen global-center schedules.

An initial exact execution stopped without an artifact because the
evaluation-only coordinate assay treated a one-dimensional token tensor as if
it already had batch and sequence axes. This was an implementation failure,
not a model result. Commit `4067926` fixed that tensor shape and added a
regression test; no threshold, seed, optimizer, data, or learned gate changed.

The rerun started clean at `4067926` in WSL on the exact RTX 2070 SUPER SM75
runtime and completed in 163.25 seconds. All three fresh seeds passed. S mean
held-out-frame error was `6.89e-7--1.18e-6`; maximum active-coordinate error
was `7.15e-7`; inactive RMS was at most `2.10e-8`; and the worst direct
carrier error over all 132 structured schedules was `2.36e-5`. Identity,
fixed torus, and S-broken each remained at least `0.266` worse in mean random
frame error. The exact oracle matched center signs within `5.11e-15` and the
two `minus_volume` words verified that projective frame scoring loses a center
distinction which direct vector/positive assays retain.

The supported claim is deliberately narrow: composition-only end loss can
learn a complete signed coordinate dictionary compatible with the repository's
hard-coded shared vector/positive Spin lift, unseen frames, and center words
under oracle edit timing. The next learning problem is no longer local chart
precision. It is learning when and where to address, erase, write, and query
content on generic association tasks. That is G15B; negative-spin/Clifford
utility, natural text, scaling, and fused performance remain unestablished.

Evidence:
[`artifacts/g15as_spanning_center_sm75_2026-08-25.json`](artifacts/g15as_spanning_center_sm75_2026-08-25.json),
SHA-256 `96e939fa4411e305637961941a565ac26da5a4212b47de3fc198687693b5dbcc`.

## G15B is now the controller problem, not another geometry assay

The repository history changes the correct experiment. Old Pure Spin work
showed that a final retrieval score can pass while an explicit query-event
router has zero event F1, and that nonzero parameter gradients do not imply a
grammar-aligned temporal path. The current Spin-Dirac query is continuously
live, so no hard event switch was reintroduced. Old generic builders also put
all writes first and all queries last with disjoint key/value/filler ranges;
they cannot support the stronger claim that write/query timing was identified.

The prospectively frozen
[`G15B protocol`](G15B_CONTROL_PROTOCOL_2026-08-25.md) therefore uses one
attention-free causal-conv Spin-Dirac block over a shared payload alphabet.
The complete executable cohort is
[`g15b_interleaved_cohort.py`](g15b_interleaved_cohort.py); its data contract is
[`g15b_interleaved_tasks.py`](g15b_interleaved_tasks.py).
Writes, queries, changed overwrites, selected/unselected writes, and needles are
interleaved. The address target groups all versions of one key: classifying the
latest write occurrence would ask the query to infer information it does not
contain. Balanced write and overwrite-erase objectives prevent length-1,024
filler negatives from dominating. Oracle direct reads through the actual
`8 x 8` state law pass every task and `I/C/S` arm exactly in the committed task
tests, removing the old representation-capacity ambiguity.

This is explicit label-supervised commissioning under the parent G15B
boundary. A fresh external-loss-only lane can follow only after the identified
controller passes. Retrieval without controller metrics and the frozen
no-memory/no-write/no-erase/wrong-query interventions will not be called a
memory result.

The first exact-SM75 quality launch was stopped before any checkpoint, output,
or held-out metric became visible. External monitoring showed the original
L1,024 evaluation batch of four using only about 17% of the GPU and implied a
multi-hour systems bottleneck unrelated to the frozen decision. A prospective
no-gradient qualification then executed the full-Spin six-intervention needle
cell at batch 32 and 16. Batch 32 peaked at 8,255,536,640 allocated bytes and
was rejected as unsafe on the 8 GB card; batch 16 peaked at 4,132,163,072
bytes. The quality evaluation cap is therefore 16. Seeds, 4,096 decisions per
cell, tasks, thresholds, training batches, optimizer, and schedule are
unchanged. The interrupted attempt has no result and cannot be used for model
selection.

## G16 completed and rejected the current local ordinary-text frontier

Before launch, a read-only audit found that initialization evaluation left the
model in eval mode for the first 200 updates. Dropout was zero, but the phase-
dependent mode violated the execution contract. Commit `32e03e5` restored
training mode before update one and added a regression guard. No G16 quality
metric had been seen. The exact runtime qualification and all source/backend
bindings remained unchanged.

The full run then started clean at commit `5796a851df02` on the RTX 2070 SUPER
SM75 runtime. All four arms completed 1,000 updates and 4,096,000 paired target
tokens. Every integrity gate passed. At L4096, v1.4.5 reached `1.58335` BPRB,
local GDN2 `1.60457`, official fused Mamba-2 `1.48571`, and actual OLMo Hybrid
`1.61413`. Mamba-2 won every context and beat v1.4.5 by `0.09764` BPRB. Its
median synchronized update was `0.03073` seconds versus v1.4.5 `0.11349` on
this exact runtime. OLMo was fastest at `0.01675` seconds but did not convert
that speed into quality.

Every arm failed learned factual recall. OLMo's nominally best 8,192-byte mean
gain was only `0.003786` nats; Mamba-2 and v1.4.5 were effectively zero and
GDN2 was negative. Thus ordinary pretraining still does not teach reliable
long-range binding, even in the best compressor.

The decision is uncomfortable but clear: do not promote the local GDN2 edit
law from its constructed G14 win, and do not treat v1.4.5 as competitive with
the official fused Mamba-2 ordinary-compression reference at this scale. G15B
must first show that a content edit memory can be commissioned and causally
used; only then should it return to a Mamba-2-controlled natural-text hybrid.

This remains a one-seed, small repeated-snapshot, optimizer-specific SM75
development cohort, not a model-family promotion or scaling law. Evidence:
[`G16_SM75_FRONTIER_SHOOTOUT_RESULTS.md`](G16_SM75_FRONTIER_SHOOTOUT_RESULTS.md)
and
[`artifacts/g16_frontier_shootout_sm75_2026-08-25.json`](artifacts/g16_frontier_shootout_sm75_2026-08-25.json),
SHA-256 `76323bb4b3a87705ac66e77bdd1c056f2a4cbb6bf7b5597386f4953187f2cac7`.

## G15B completed: address learned, collision erase was unobservable

The exact-SM75 G15B quality cohort completed from clean commit `bd5045a` after
37,800 optimizer updates: three seeds, three primary arms, and 4,200 updates
per seed/arm. All source, schedule, namespace-separation, oracle-capacity,
checkpoint, and finite-value integrity checks passed. Each arm had 57,949
trainable parameters and 1,792 recurrent-state bytes per sequence.

The result is a binding failure, not an absence of learned memory. Identity
reached three-seed mean query accuracy of `0.972--0.973` on MQAR,
`0.972--0.977` on selective copy, and `1.0` on needles. Address top-1 was at
least `0.999756`; no-memory/no-write and wrong-query interventions reduced
mean accuracy by about `0.938` and `0.920`. The recurrent association matrix
was therefore learned, used, and addressable.

The failed component was edit timing. Write recall was one, but the identity
write F1 mean was only about `0.785`, and overwrite-erase recall was about
`0.506`, far below the frozen `0.95` gate. No arm/seed passed every absolute
controller metric. Full Spin was not noninferior: it lost all nine non-needle
task/length mean cells to identity and trailed by as much as 17.21 percentage
points in a paired overwrite seed. The commuting arm was worse still.

Inspection identified the present learning problem. `_controls` derives
address, erase, write, retention, and transport from the current block input.
In the frozen one-block shell that input contains an embedding plus a width-4
causal convolution. The erase target, however, is one exactly when the current
write key occurred anywhere earlier in the episode. Two histories can have
the same local controller input and different collision labels. No optimizer
can recover a target that is absent from its observations.

The next prospective repair is consequently narrow:

1. implement last-write-wins by erasing the addressed key on every valid
   write, so first-write erase is a harmless no-op on an empty address;
2. if independent state-dependent erase is retained, expose a causal pre-write
   occupancy/read signal and test its temporal observability before training;
3. retain identity as the generic association reference; token-wise Spin
   transport is eligible only when keys, queries, and memory share a coherent
   transported frame or the task supplies an observable moving frame;
4. repair the learned-path oracle intervention so it preserves the learned
   address/value gauge instead of replacing only the address with one-hot
   coordinates.

G15C and the external-loss-only lane remain blocked. This does not undo
G15A-S: that result established a learned 28-generator transport dictionary
under oracle edit timing, not a generic controller. Evidence:
[`G15B_INTERLEAVED_CONTROLLER_RESULTS.md`](G15B_INTERLEAVED_CONTROLLER_RESULTS.md)
and
[`artifacts/g15b_interleaved_controller_sm75_2026-08-26.json`](artifacts/g15b_interleaved_controller_sm75_2026-08-26.json),
SHA-256 `f74d860e30ab40ec747521dfcecd74aac2bb75151206c25b7104d334727429eb`.

Before a fresh trained repair, G15B-R0 is frozen as a zero-update checkpoint
intervention. It preserves learned keys, queries, values, retention, and the
decoder, avoiding the old one-hot oracle's gauge change. It compares learned
editing, soft erase-equals-write delta correction, exact collision timing, and
exact delta timing on the retained identity checkpoints. A constructive pair
of histories proves that collision timing is absent from the width-four local
observation, while valid-write timing is exactly decoded by the marker two
positions earlier. See
[`G15BR_CHECKPOINT_REPAIR_PROTOCOL_2026-08-26.md`](G15BR_CHECKPOINT_REPAIR_PROTOCOL_2026-08-26.md).

G15B-R0 then completed from clean commit `f303435` with exact replay of all 36
recorded identity cells. The naive repair fails decisively: soft delta loses
6.6--7.9 points on MQAR, 9.3--11.8 on overwrite, and 10.7--11.1 on selective
copy. Exact collision and exact delta timing fall to roughly 0.39--0.57 on
non-needle tasks. The failure exposes a second temporal fact. Depending on the
seed, one, two, or all four heads fire on essentially every token immediately
after a valid write, while firing on only about 0.76% of filler positions
overall. Those token-level false positives are a structured write continuation
used by the learned code. Atomic exact timing removes it; tying erase to it
makes the continuation destructive.

The next candidate must keep the two roles separate: locally anchored erase on
every valid write event, and a short learned write window. This removes the
unobservable collision target without erasing the useful continuation. Before
fresh training, a paired checkpoint intervention must preserve learned writes
and replace only erase timing. Evidence:
[`G15BR_CHECKPOINT_REPAIR_RESULTS.md`](G15BR_CHECKPOINT_REPAIR_RESULTS.md) and
[`artifacts/g15br_checkpoint_repair_sm75_2026-08-26.json`](artifacts/g15br_checkpoint_repair_sm75_2026-08-26.json),
SHA-256 `4d92d6af2fb062cf2baaa035c4e4eff89d494dfcb56b9b666523bbbdbfe3cf9c`.

G15B-R1 is now frozen before fresh training. It leaves the learned write
microprogram untouched and replaces only erase with either the learned write
amplitude at exact event positions or unit erase at those positions. Learned
prototype off-diagonal cosine is binding diagnostic evidence: a large value
would explain collateral erasure of unrelated keys. See
[`G15BR1_EVENT_ERASE_PROTOCOL_2026-08-26.md`](G15BR1_EVENT_ERASE_PROTOCOL_2026-08-26.md).

G15B-R1 then completed from clean commit `dba3f9a` on exact SM75. All replay,
parent-hash, local-decoder, model-forward, and bitwise preserved-control checks
pass. Both erase-at-every-write arms lose all nine non-needle gates, including
9.7--11.5 points on overwrite; needle remains perfect. Learned key prototypes
are strongly nonorthogonal (mean absolute off-diagonal cosine `0.54822`,
maximum `0.999934`), but the statistic is not monotone with damage across seeds
and is not sole-cause proof. The artifact SHA-256 is
`c015b128846e4b5c63d927778815a87728a7d613369163b1027ed3dd9f0b2912`.
Do not train event-anchored erase. The remaining checkpoint-only factorial is
collision-only erase with the learned write continuation preserved; R0 changed
both timing and write, while R1 changed timing at first writes and overwrites.
See [`G15BR1_EVENT_ERASE_RESULTS.md`](G15BR1_EVENT_ERASE_RESULTS.md).

G15B-R2 is frozen before inspecting any quality metric. It completes the
missing factorial cell by preserving the learned write continuation and
restricting soft/unit erase to true collision events. Overwrite queries are
partitioned into before-any-overwrite, after-unrelated-overwrite-only, and
after-same-key-overwrite strata. This is oracle causal timing and cannot
authorize the current token-local controller. If same-key recall does not
improve, the frozen next interpretation is exact logical-component replacement;
if it improves while a guard stratum fails, test an oblique/separate erase
address. See
[`G15BR2_COLLISION_ERASE_PROTOCOL_2026-08-26.md`](G15BR2_COLLISION_ERASE_PROTOCOL_2026-08-26.md).

G15B-R2 completed from clean commit `5eae963` on exact SM75 in 1,690.2
seconds. Parent hashes, all baseline accuracy/episode/BPQ cells, ordinary
logits, local and collision masks, the observability witness, and bitwise non-
erase controls pass exactly. Collision-only erase raises MQAR by 0.8--1.2
points and pre-overwrite recall by 1.8--2.1, yet lowers post-same-key-overwrite
recall by 10.3--12.1 and aggregate overwrite by 8.7--10.4. The registered
unrelated-overwrite-only stratum is empty in this generator, so no claim is
made for it. Artifact SHA-256:
`90652fe7034e5901b968eb5d139f02eb8bc714b0417c0889e16a2fdd6b7cf924`.
This rules out first-write collateral and erase amplitude as the primary
repair. Do not train a scalar erase controller. The next oracle should reset
the complete value-token-plus-tail component for the overwritten logical key
under the same learned transitions. See
[`G15BR2_COLLISION_ERASE_RESULTS.md`](G15BR2_COLLISION_ERASE_RESULTS.md).

G15B-R3 was frozen before retained-checkpoint intervention metrics. It assigns
value-token plus in-range `t+1` injection to an oracle logical-key component,
disables the rejected symmetric erase, and compares exact component reset
against an erase-free no-reset control. A deterministic guard separately
populates before-any, unrelated-only, and same-key-overwrite histories. The
first attempted quality invocation stopped before evaluating a batch on a
final-token write; the protocol was amended fail-closed to record its value
without inventing a tail, then recommitted before the evidentiary run.

The exact-SM75 quality run completed from clean commit `3e2e5f0` in 3,485.5
seconds. Oracle component reset raises ordinary overwrite 12.2--12.8 points
over learned erase and post-same-key recall 13.6--14.3 points; it reaches 1.0
in every constructed-guard cell and preserves unrelated keys. The registered
promotion still fails. The guard requires a 0.10 gain over an already
0.9997--0.9999 learned arm, while LWW is perfect, and learned decomposition
exceeds the `5e-4` FP32 logit tolerance in seeds 2309/2333 despite identical
query predictions and passing state residuals. Artifact SHA-256:
`0fe54b8ce38868d67a7ecb0cb888f2279d8809c2bbaf3ccbda678326ff808959`.

Record this as strong bounded component-replacement mechanism evidence but a
failed R3 promotion gate. It does not authorize training. The next prospective
diagnostic must separate value-only/tail ownership, following-marker tails,
background contribution, and numerical reconstruction, with an absolute
ceiling-aware guard frozen before quality. See
[`G15BR3_LOGICAL_COMPONENT_RESULTS.md`](G15BR3_LOGICAL_COMPONENT_RESULTS.md).

G15B-R4 is now frozen before inspecting any new intervention metric. It is a
two-by-two retained-checkpoint factorial: value-only versus value-plus-tail
component ownership, crossed with background included versus excluded only at
locally observable query positions. It retains matching no-reset controls,
binds the sealed R3 arm as an ineligible reference, adds a separate FP64
algebraic contract, and replaces R3's ceiling-blocked guard-superiority test
with prospective absolute `0.995` success plus non-regression. Only a passing
value-only arm can authorize a separately frozen explicit-slot training
screen. See
[`G15BR4_OWNERSHIP_BACKGROUND_PROTOCOL_2026-08-26.md`](G15BR4_OWNERSHIP_BACKGROUND_PROTOCOL_2026-08-26.md).

The exact-SM75 R4 quality run subsequently completed from clean commit
`d014259` in 5,031.34 seconds. The sealed R3 reference replays at zero metric
residual, every provenance/runtime gate passes, and the FP64 algebraic
contract has maximum residual `4.44e-15`. Both value-plus-tail arms pass the
prospective ceiling-aware gates. Neither value-only arm passes. With
background included, value-only overwrite is `0.702962`, `0.764486`, and
`0.753418`, below the learned arm at all three lengths, and its guard is only
`0.938965`--`0.941813`. Seed 2311 is the decisive multi-seed collapse.

Removing query-time background leaves value-plus-tail accuracy essentially
unchanged but destroys the value-only arm. This localizes the useful
association to the learned `t+1` continuation: it is not an intrinsic need
for a shared background read. Because those tails include following write and
item markers, the passing law is not a clean semantic slot boundary. Record
the frozen decision exactly: **do not train; passing behavior remains
dependent on ambiguous value-plus-tail ownership**. Artifact SHA-256:
`921d45e3c492e172fae62064120e9e051dca2965bacc44891268b135d8cef26e`.
See
[`G15BR4_OWNERSHIP_BACKGROUND_RESULTS.md`](G15BR4_OWNERSHIP_BACKGROUND_RESULTS.md).

The next bounded question is transaction formation, not another optimizer or
erase-amplitude sweep: can a causal pending-write/commit variable reproduce
the required continuation without assigning the current event marker's own
injection to the preceding key? No fresh explicit-slot training is authorized
until that ownership problem has a prospective falsifier.

G15B-R5 is now frozen before any history-only/current-only metric. At each
write tail it decomposes the exact width-four depthwise-convolution
preactivation into bias, three strict-history taps, and the current-token tap.
The candidate key component receives an injection generated either from
history only or current only; exact `full - source` injection remains in
background so the no-reset sum reconstructs the original recurrence. Only a
passing history-only arm can authorize a separately frozen pending-write/
commit training screen. See
[`G15BR5_CAUSAL_TAIL_SOURCE_PROTOCOL_2026-08-26.md`](G15BR5_CAUSAL_TAIL_SOURCE_PROTOCOL_2026-08-26.md).

Prospective adversarial review, still before any R5 metric, identified two
necessary restrictions. First, `BG+` can read the exact residual containing
current-token and nonlinear interaction information, so it cannot establish
clean history sufficiency by itself. Second, history/current sources both
contain convolution bias. R5 now includes a bias-only arm, requires history to
beat its matched bias control, and authorizes training only from a passing
background-free history arm. It also labels the result as injection-source
sufficiency conditional on the unchanged full-token transition.

The first dirty-tree, 16-decision CPU implementation smoke is explicitly
non-evidentiary. It passed the FP64 algebraic contract at `1.55e-15`, exact
source assignments, finite outputs, control preservation, and query-prediction
parity, but direct FP32 convolution reconstruction reached `9.54e-7` because
the decomposed tap sum changes addition order. Before any quality run, the R5
protocol therefore freezes the FP32 structural bound at `2e-6` while retaining
the independent `1e-10` FP64 bound. No performance or authorization threshold
was changed.

## 2026-08-26: G15B-R5 exact-SM75 quality result

The clean quality cohort completed from commit `e039e49` in 4,611.91 seconds;
artifact SHA-256 is
`ba627fe34e8dd29458fc1321b52c98242838c3b56e2abdc7e44c749f50aaa313`.
It used all three retained identity checkpoints, 4,096 decisions per cell,
lengths 128/512/1,024, and zero updates.

The substantive attribution is positive. `h_lww_bgminus` passes all 132 frozen
performance and bias-separation checks. Mean ordinary overwrite is
0.9424/0.9456/0.9466 and every constructed-guard mean is 1.0. Current-only and
bias-only arms fail. The learned association at the ambiguous R4 tail is
therefore carried by the completed transaction in strict convolution history,
not bias or the new token alone. This remains injection-source sufficiency
conditional on the unchanged full-token transition and oracle logical-key
components.

The frozen formal decision is still a fail. Every discrete R4 replay metric is
exact, but the decomposed no-reset BPQ differs by at most `1.397e-7` from R4
against the frozen `1e-12` bound. FP32 no-reset state and background-read
relations reach `2.384e-6` and `3.576e-6` against `2e-6`. Learned logits are
bit-identical, all reference predictions match, source assignment is exact,
and FP64 algebra passes at `3.997e-15`. Treat these as a prospectively testable
numerical-ratification problem, not permission to waive R5 after the result.
Freeze R5-S before any training; do not erase the formal R5 failure.

## 2026-08-26: G15B-R5-S exact-SM75 numerical result

The prospective R5-S cohort was frozen in commits `d66d6e8` and `534c53b`
before any stability batch. The implementation was sealed and pushed as
`dde868a`. The clean exact-SM75 quality run then completed in 1,155.11 seconds
with artifact SHA-256
`3ac514e16e6fa1c720d5ef4244525f5d0f08c233634648e59181c6acfccc3a00`.

The result is a formal fail and the retained-checkpoint repair route stops.
All 135 source/cell checks exceed the frozen scaled-logit allowance; the ratio
ranges from 1.171875 to 66.078125. No threshold is amended. This is not a
failure of every numerical contract: all predictions remain identical,
maximum BPQ drift is `1.720e-7`, state/read residuals remain below `3e-6`,
source assignment is exact, transitions are bit-identical, and independent
FP64 algebra passes at `5.862e-14`.

The audit reproduced every sealed R5 aggregate digest, regenerated the fresh
cohort twice, proved individual fingerprint disjointness, and matched every
checkpoint and R5 source hash. Worst-batch comparison against a common FP64
reference places both monolithic and decomposed FP32 read error near
`2.4e-6`; the downstream RMS-normalized readout and projections amplify that
into logits separated by as much as `6.387e-4` without categorical change.

Do not waive R5-S or run another tolerance sweep. Preserve R5's independent
performance-positive history attribution, but pivot the next fresh model to
an explicit causal pending transaction, separate commit/edit control, and an
exact monolithic residual/read path. Component reads should protect key
content, not be summed merely to recreate the monolithic forward path.
