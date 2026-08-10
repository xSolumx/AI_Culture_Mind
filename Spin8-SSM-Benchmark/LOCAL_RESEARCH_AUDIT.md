# Local research audit and architecture decisions

This file records which local results were used in the isolated benchmark and
which claims were deliberately rejected. It is a research audit, not a claim
that every historical prototype has been re-derived line by line.

> **2026-08-10 status note.** A later, separate memory-core campaign completed
> hierarchical learned-routing and co-moving FLA transport benchmarks. It did
> not modify this model and found no same-router triality capacity advantage.
> Treat it as a candidate component and control design for a future model-level
> benchmark, not as evidence for the results recorded here.

## 2026-08-06 refresh: newer theorem results and model consequences

The newer archive was read read-only while the benchmark model was changed
only in this isolated folder. The conclusions below are deliberately narrower
than the archive's strongest mathematical statements:

- The balanced Cayley information family now has an exact invariant-coordinate
  split of `8 + 8 + 8 + 4`; the repeated determinants are explained by an
  exact signed-permutation conjugacy. This supports structured state design and
  debugging, but it does not imply a language-model advantage.
- The variable-Cayley one-edge Dirac--Gram inequality is an exact theorem on
  its frozen five-variable family, backed by the two Duffy charts, exact
  Bernstein boundary identities, and 256 direct rational holdouts. The global
  seven-invariant inequality and global D-optimality remain open.
- The two-edge bridge has an exact eight-sector sign quotient, a proved
  `Delta^3` common factor, conservative degree bounds, and a complete
  reconstruction of sector `110101` with exact off-grid holdouts. The other
  sectors' positivity is not proved, so this is a structural guide rather than
  a loss or architecture claim.
- The strongest transferable ML findings are joint-family retraction,
  multiplicity-aware/gauge-aware parameterization, and triangular bilinear
  scans. A single chiral eight-dimensional view is only an SO(8)-type chart;
  triality-specific language superiority remains an explicit falsifier.

These findings motivated retaining the stable isotypic rotor-affine core and
avoiding a speculative Spin(8) claim. The isolated quality iteration promoted
one falsifiable change: unit scaling for the untied vocabulary head. An
identity-initialized causal mixer and a coordinate-sensitive direct decoder
ensemble were implemented as opt-in controls, but short matched screens were
worse than or indistinguishable from the promoted path and therefore remain
off by default.

## Sources reviewed

### Spin8-Triality-Research

- `docs/RESEARCH_AUDIT_AND_NEXT_STRATEGY_2026-08-06.md` and
  `docs/ITERATION_NOTE_2026-08-06.md`;
- `docs/experiments/SPIN8_CAYLEY_BLOCK_THEOREM.md`;
- `docs/experiments/SPIN8_DIRAC_ONE_EDGE_RESULTS.md`;
- `docs/experiments/SPIN8_DIRAC_TWO_EDGE_GATE_RESULTS.md`;
- `docs/experiments/SPIN8_DIRAC_TWO_EDGE_SECTOR_110101_RESULTS.md` and
  `docs/SPIN8_TWO_EDGE_AMPLITUDE_THEOREM.md`;
- `docs/experiments/RESEARCH_REVIEW_2026-08-02.md`
- `docs/experiments/RESEARCH_PHASE_2_RESULTS.md`
- `docs/experiments/RECURRENCE_LADDER_RESULTS.md`
- `docs/experiments/SPIN3_ISOTYPIC_SCHUR_SCAN_RESULTS.md`
- `docs/experiments/MECHANISM_GATE_RESULTS.md`
- `docs/experiments/SELF_COMPILING_RETRACTION_RESULTS.md`
- `docs/experiments/ROBUST_CHANNEL_GATING_RESULTS.md`
- `docs/experiments/Q8_SPINOR_QUALITY_GATE_VALIDATION_RESULTS.md`
- `docs/experiments/SPIN8_SO8_PAIRED_RESULTS.md`
- `docs/experiments/SPIN8_ACTIVE_SENSING_RESULTS.md`
- the variable-Cayley/Dirac theorem results and counterexample documents;
  these remain mathematical work, not language-model evidence.

Relevant source implementations were checked in `src/schur_scan.py`,
`src/rotor_ssm_torch.py`, the recurrence harnesses, and the triality/Q8
audits.

### SSM-Models and SpinorModel

- `SSM-Models/ga_ssm.py` and `GALib.py` for the original stable rotor-affine
  scan and JAX parallel/recurrent parity;
- `SSM-Models/rotor_ssm_torch.py`, `mechanistic_group_actions.py`,
  `robust_channel_gating.py`, and `pdssm_group_actions.py` for controls and
  negative results;
- `SpinorModel/overhauled/model.py` and `algebra.py` for the independent write
  gate, retention bounds, streaming interface, and scan backend split.

## Strong findings retained

1. Rotor transport is norm preserving; bounded affine innovation gives a
   useful stability/BIBO invariant.
2. The recurrence is an associative affine monoid, so an exact differentiable
   logarithmic-depth scan is valid for training and a recurrent step is valid
   for streaming.
3. `GradeLinear` is not the complete Spin(3) equivariant family. The validated
   Schur/isotypic construction mixes scalar/pseudoscalar and
   vector/Hodge-bivector copies and has twice the commutant dimension.
4. Independent write and erase controls are the justified next ablation; tying
   write magnitude to retention is not a theorem and hurt the recorded Q8 run.
5. Exact/retracted group actions and quality-gated decoders can solve the
   controlled A5/Q8 mechanism tests, but raw SGD, long-horizon language
   behavior, and Spin(8)-specific superiority remain unproved.

## Claims rejected or kept as controls

- Spin(8) triality is not assumed to improve WikiText. The paired positive
  half-spin versus generic SO(8) audit found no stable raw advantage.
- Static rotors, hybrid complex+GA direct sums, grade-specific decay, and
  unconstrained raw learned actions are controls, not default improvements.
- Exact reconstruction, interpolation, and positivity in the theorem archive
  are separate proof layers and are not converted into a neural-language
  claim.
- Synthetic mechanism accuracy does not substitute for a matched language
  benchmark.

## Benchmark integrity corrections

The original harness instantiated models before setting the requested seed.
That invalidated its initialization control and any future multi-seed claim.
The current harness accepts model factories, seeds Python/NumPy/PyTorch/CUDA,
and constructs each model only after seeding. Earlier JSON results must not be
used as multi-seed evidence.

The matched run uses a 42-channel state (336 scalars) and a 20-channel
equivariant decoder bottleneck (160 scalars), giving 674,322 Spinor parameters
versus 688,220 for the four-layer Mamba-2 control. The bottleneck is explicit so
the vocabulary projection is compute-comparable rather than silently paying
for a 352-wide tied decoder.

## Open falsifiers

- parameter-matched five-seed WikiText runs with the corrected RNG harness;
- A5/Q8 and copy/MQAR-style synthetic tasks alongside WikiText;
- recurrent streaming parity after checkpoint loading;
- a Linux fused-kernel comparison, since the current Windows run uses pure
  PyTorch Mamba-2 fallback and tensor-only rotor CUDA execution;
- a deeper prior-art review before claiming a novel architecture result.
