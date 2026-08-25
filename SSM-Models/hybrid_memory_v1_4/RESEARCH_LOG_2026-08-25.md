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
