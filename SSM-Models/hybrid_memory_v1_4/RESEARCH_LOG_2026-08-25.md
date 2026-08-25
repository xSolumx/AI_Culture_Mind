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
