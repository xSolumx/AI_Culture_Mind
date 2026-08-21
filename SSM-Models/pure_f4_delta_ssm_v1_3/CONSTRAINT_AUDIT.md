# Architecture constraint audit

**Date:** 2026-08-21
**Question:** Which v1.2 restrictions are mathematics, which are model
hypotheses, and which came only from an old experiment or kernel?

## Verdict

The old research did not justify a permanently 24-scalar, compact,
orthogonal, rank-one, tied-write Spin(8) model. It justified a smaller set of
principles:

1. nonassociative products must be lifted to associative operator or affine
   composition before a prefix scan;
2. group action, address memory, routing, and hardware lowering are separate
   mechanisms;
3. a symmetry prior is useful when the data actually contains the matching
   cross-view action, but it does not manufacture overwrite capacity;
4. complete streaming state and matched controls must be counted honestly;
5. backend decisions are conditional on dtype, shape, device, and reuse.

v1.3 preserves those principles and removes the rest as hard constraints.

## Constraint ledger

| Candidate restriction | Where it entered | Evidence-supported status | v1.3 decision |
|---|---|---|---|
| three `8v/8+/8-` states | faithful Spin(8) triality model | exact representation, but no matched-state overwrite-capacity gain | replaced by full Albert 27D value carrier; Spin(8) remains a tier |
| 26D trace-free Albert state | first compact (F_4) design | correct irreducible (F_4) module, but excludes natural (E_6) structure action | full 27D default; tested 26D restriction retained |
| compact/orthogonal transport | Spin/rotor stability and inverse-backward kernels | useful special case, not required by the co-moving theorem | (E_{6(-26)}) noncompact action implemented; generic invertible scan law |
| exactly 28 plane factors | Spin(8) factor compiler | profitable representation-specific lowering on RTX 2070 SUPER | not a semantic limit; 28/36/52/78/custom banks supported |
| one exponential per token | initial (F_4) proposal | sufficient for compact connected (F_4), not a guaranteed global chart for noncompact (E_6) | configurable ordered products of exponentials |
| one undifferentiated E6 chart | first v1.3 implementation | locally valid but hides compact frame versus noncompact content | direct, polar `K exp(P)`, and rank-two Cartan `KAK` hypotheses are executable |
| monotone group schedule | v1.2 `Spin(3)->Spin(4)->Spin(6)->Spin(8)` benchmark | architectural hypothesis only | removed; schedules may move up, down, repeat, or stay fixed |
| scalar/rank-one update | available affine and fused DeltaRule paths | scan-compatible, not an optimality theorem | configurable rank (r) block update |
| tied erase/write address | classical DeltaRule kernel | valid control; local notes explicitly identify independent write/erase as a candidate | independent bounded default; tied and unconstrained controls |
| contraction | finite-state stability proof | sufficient stability policy, not group algebra | default retention is contractive; unconstrained mode remains executable |
| depthwise convolution | natural-data v1.2 repair | useful local mixer, not part of Spin algebra | optional and its streaming cache is now counted |
| SwiGLU | v1.2 custom-mixer falsification | strongest local baseline among tested mixers, not universally optimal | retained as control; Albert-Jordan and no-mixer alternatives |
| action shared across copies | triality and compiler reuse | legal only when semantics really share the action | no implicit inference; custom banks and per-layer choices are explicit |
| Tensor-Core packing | isotypic-to-silicon audit | won only selected high-occupancy cells | future dispatch policy, never an architecture axiom |
| projective memory state | Riemann-sphere/PGL prompt | quotient removes amplitude required by write/erase memory | rejected for core state; implemented only as a router control |
| RMS-normalized read only | inherited stable readout | deletes magnitude even though overwrite strength is meaningful | direction plus trace, log-energy, and bounded Albert determinant |
| causal ordering | SSM task definition | genuine semantic requirement | retained |
| associative transition summaries | parallel prefix execution | genuine algebraic requirement | retained; raw octonion/Jordan multiplication is never scanned |
| generic two-sided scan for identity transport | reuse of exceptional-action compiler | semantically valid but performs a nonexistent value action | specialized to the exact one-sided affine monoid |
| explicit Albert determinant | exact cubic formula and lower operation count | algebraically equal, but finite-precision training equivalence was not guaranteed | retained as control; rejected as default by prospective five-seed gate |
| sparse Albert product | 531 nonzero structure entries | exact sparse representation, but reduction order and GPU schedule are backend facts | retained as control; dense remains default |

## Local evidence chain

1. `experiments/PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md` showed that structured
   transport could be active yet lose recall and compute gates. It prevents
   “group action alone is memory” from becoming a premise.
2. `experiments/SPIN8_TRIANGULAR_TRIALITY_LIFT_RESULTS.md` proved that triangular
   bilinear dependencies admit a fixed associative lift, while generic cyclic
   feedback causes unbounded polynomial degree. That is a real closure gate.
3. `Spin-Space-Research/docs/experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`
   separated routing, exact addressed memory, optional symmetry, and the
   co-moving compiler. It also proved the compiler formula for general
   invertible transport, not only orthogonal transport.
4. `Spin-Space-Research/docs/experiments/LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md`
   made physical sparse routing—not extra triality capacity—the positive
   memory result.
5. `experiments/ISOTYPIC_TO_SILICON_COMPILER_V211.md` and
   `experiments/SPIN8_TRAINABLE_FACTOR_COMPILER_V212.md` showed that exact
   representation type, shared-action semantics, and profitable hardware
   layout are distinct obligations. Maximal fusion lost to staged controller
   reuse on the local GPU.
6. `pure_spin_ssm_v1_2/FRONTIER_TRAINING_RESULTS.md` retained the honest natural-
   data boundary: Mamba-2 remained ahead in both bits/byte and final controlled
   throughput. The quality gap points at controller/readout/memory design, not
   a license to overfit the next model to one microkernel.
7. Programme 01's reducible/isotypic and algebraic-field work showed that
   coefficient field, real representation category, multiplicity, and hardware
   schedule must not be conflated. v1.3 follows the same separation.

## A limitation found and removed during this audit

The first v1.3 draft repeated v1.2's nondecreasing schedule validation. No
theorem or controlled result requires layer algebras to grow monotonically.
The validation was deleted and a regression test now executes the reverse
(E_6\to F_4\to\mathrm{Spin}(9)\to\mathrm{Spin}(8)) schedule.

The first draft also returned only the exceptional memory matrix as recurrent
state while using a causal convolution. That understated the cache and broke
exact chunk continuation. The state now contains both memory and convolution
history; a full-model chunk equivalence test protects the contract.

The next draft exposed polar and Cartan E6 charts, then tested them rather than
declaring the most elaborate chart superior. Direct and polar E6 were
indistinguishable at the short Shakespeare quality gate while direct completed
more training tokens per second. Direct is therefore the development default;
the other charts remain controls for hypotheses requiring explicit factor
separation.

Finally, the original read path applied RMS normalization to all 27 Albert
coordinates and silently removed read amplitude. The revised readout retains
the normalized direction but adds trace, log-energy, and bounded cubic
determinant channels. A regression test verifies that positive rescaling is no
longer invisible and that gradients reach the full read.

The seed-17 Shakespeare screen then suggested placing E6 only in the first of
two layers. That post-hoc signal was not promoted. A decision rule was committed
before five fresh seeds were run; the candidate won only 2/5 and its mean paired
effect was `-0.0093` bpb. Identity transport is therefore the supported
benchmark reference for generic natural-text development at this scale. This
does not remove the exceptional actions; it prevents a valid algebra from
being mistaken for an empirically justified universal language prior.

The subsequent optimization audit followed the same discipline. Identity
transport was reduced to its exact one-sided affine composition law, yielding
bitwise-equal model outputs and gradients and lower eager cost. An algebraically
equivalent explicit determinant was not assumed training-equivalent: it was
preregistered and run on five fresh seeds. Its 1.134x geometric-mean training
throughput and lower memory did not compensate for a +0.0160 bpb mean quality
regression beyond the +0.0100 limit. The explicit formula was therefore
removed from the default. `torch.compile(reduce-overhead)` is an opt-in
fixed-shape tier because its eager-relative errors are small but nonzero and
its cold compilation cost is material. Full evidence is in
`V1_3_OPTIMIZATION_RESULTS.md`.

## Still deliberate, not accidental

- The language model is causal because it is an autoregressive SSM.
- The exceptional built-in carrier has dimension 27 because that is the Albert
  representation; the scan and custom-bank action APIs accept other sizes.
- The default retention and erase strength are bounded for a sane initial
  training regime, but both can be disabled as explicit falsifiers.
- (E_7) is not silently approximated. Its natural Freudenthal continuation
  needs a 56D symplectic/quartic carrier and a separate correctness gate.
- No CUDA or quality claim is inherited from v1.2. Those require new matched
  artifacts.
