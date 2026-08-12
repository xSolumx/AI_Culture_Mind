# Research audit, understated papers, and next strategy

**Audit date:** 2026-08-06  
**Scope:** frozen `Spin-Space-Research` archive, its extracted SSM lineage,
and the historical `SpinorModel`. The separately active benchmark worktree was
not inspected as a frozen result and was not modified.

> **2026-08-10 roadmap addendum.** The memory-specific matched-baseline and
> fused-transport targets below are now complete at the memory-core level.
> Direct and triality slots tie under a shared router; hierarchical block
> routing improves direct and delta retrieval; and the co-moving recurrence
> runs through official FLA chunk/recurrent kernels with frozen fresh-process
> CUDA measurements. Model-level integration, end-to-end sparse-routing cost,
> and naturalistic task quality remain open. The theorem and Dirac--Gram
> targets in this dated roadmap are unchanged. See
> [`SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`](experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md).

This note has three jobs:

1. preserve results that are large enough to become papers in their own right;
2. give the next researcher a compressed, ordered strategy;
3. record corrections for claims or implementations that did not survive a
   hostile audit.

The vocabulary is deliberate:

- **exact theorem** means an algebraic argument or replayable exact
  certificate;
- **local theorem/evidence** means a Jacobian, tangent-space, or
  machine-precision result near a specified action;
- **empirical result** means a finite cohort, length sweep, or optimizer run;
- **proposal** means the decisive proof or experiment has not been run.

## Executive conclusion

The archive contains more than one possible paper. Its most original common
thread is not merely the use of geometric algebra. It is the following
relational principle:

> When the missing constraint relates several learned objects, optimize local
> coordinates freely but retract the complete family jointly onto one shared
> representation manifold. Independent normalization cannot recover
> information that exists only in relations between family members.

Spin(8) triality then adds two exceptional ingredients: three inequivalent
eight-dimensional views of the same 28-dimensional action, and a unique
equivariant trilinear binding tensor. A single chiral eight-dimensional
recurrence alone is only another chart for an SO(8) recurrence and must not be
used as evidence of a triality-specific advantage.

## Understated paper-scale contributions

### Paper A: Joint family retraction from incomplete observations

**Publication-safe claim.** Matched endpoint observations can leave large
independent action families locally underdetermined even when every observed
endpoint is fit to numerical precision. Requiring all views to arise from one
shared 28-dimensional Spin(8) action eliminates this relational slack and
recovers unobserved transport that composes through long horizons.

**Evidence already present.**

- shared family: 28 coordinates, observed rank 28;
- independent family: 84 coordinates, observed rank 63, leaving 21 directions;
- ten-seed hidden negative-action completion;
- dense composition through length 2,048;
- matched direct-transport control, added after a binding-path bypass was
  detected;
- the same family-level lesson independently appears in finite-group
  retraction and Birkhoff/Sinkhorn address completion.

**What makes it broader than Spin(8).** The result suggests a general method for
learning families of operators with shared algebraic relations: group
representations, inverse-paired actions, permutation families, multiview
transport, and latent address systems.

**Required before publication.** Add non-Spin controls with the same number of
coordinates and observations; state the remaining gauge freedom; distinguish
local rank closure from global uniqueness; and include a theorem describing
when joint-family constraints reduce the nullspace.

**Possible title.** *Relational Completion by Joint Retraction of Learned
Operator Families*.

### Paper B: The five-probe Spin(8) base-size problem

**Publication-safe claim today.** Every four-probe sensor has a
positive-dimensional stabilizer. The principal stabilizer Lie algebra is
`spin(4)` for one view and `su(2)` for every mixed allocation. Every mixed
five-probe allocation has an open dense globally free stratum; five probes in
one view retain `Spin(3)`. Exact invariant Jacobians, action ranks, compact Lie
types, and full-closure points support the compact principal-orbit proof.

**Why this is mathematically interesting.** This is naturally a base-size or
generic-stabilizer problem for Spin(8) acting on a disjoint union of its three
triality representations. The observed Spin(7), G2, and SU(2)-sized stabilizer
chain is representation theory, not merely a neural-network diagnostic.

**What is still missing.** The sharp principal boundary is closed. What remains
is classification of every exceptional nonprincipal five-probe orbit and its
possible finite or positive-dimensional stabilizer, plus the separate
conditioning problem inside the free stratum.

**Decisive upgrade achieved.** The exact continuous orbit theorem now supports
the global generic five-probe claim. The next upgrade is an exceptional-strata
classification, not another generic rank experiment.

**Possible title.** *Generic Stabilizers and Minimal Multiview Sensing for
Spin(8) Triality*.

### Paper C: Cayley-null D-optimal triality sensing

**Exact result already present.** On the complete orthonormal balanced
information family,
the information determinant depends only on the Cayley coordinate `c`:

\[
\det I_c=\frac{(1-c^2)^3(9-c^2)^2}{1024}.
\]

The unique unoriented optimum is Cayley-null (`c=0`) with determinant
`81/1024`; calibrated endpoints have rank 25. The complete degree-28
characteristic polynomial factors into small repeated blocks.

**New exact mechanism.** The information family has constant invariant
coordinate blocks of dimensions `8 + 8 + 8 + 4`. Two eight-dimensional blocks
are exactly conjugate by a signed permutation, and the balanced block
determinants are `1/4`, `9/16`, `9/16`, and `1`. Their product explains
`81/1024` without treating the degree-28 factorization as an opaque CAS result.
Classifying these blocks under the residual stabilizer remains a deeper
representation-theoretic opportunity; the present certificate does not call
them irreducible.

**What is proved separately.** The signed-star theorem, the Cayley-null
four-correlation edge theorem, and the variable-Cayley one-edge extension have
exact positivity certificates. A rational counterexample exactly falsifies
coordinatewise residual removal.

**What remains open.** Two more residual Cholesky correlations, the unrestricted
Gram--Cayley inequality, and global five-query D-optimality across every
allocation and nonorthogonal frame.

**Continuous local upgrade.** The complete 35-dimensional Riemannian Hessian
is now exact. Its kernel is precisely the 28-dimensional shared `Spin(8)`
orbit, and the seven quotient eigenvalues are strictly negative. Thus the
balanced sensor is a strict local maximum modulo symmetry. All 35 one-probe
coordinate great circles are also exact: 15 are determinant-flat and 20
strictly decrease to rank-25 boundaries. Distant coupled deformations remain
the gap between this local theorem and the global claim.

**Possible title.** *Cayley-Null Optimal Experimental Design in Spin(8)
Triality*.

### Paper D: Intertwiner SchurScans

**Exact result already present.** A triangular dependency graph permits an
equivariant bilinear drive to be evaluated using two associative scan stages.
For triality, independently scanned positive- and negative-spinor streams drive
the vector stream through the invariant tensor while streaming only 24 scalars.
A homogeneous 81-dimensional lift closes the two source streams and their
tensor product. The complete single-stage proof lift, including the downstream
vector state, is 89-dimensional. The staged runtime materializes neither lift.

**General theorem now stated and checked.** For representations \(U\), \(V\),
and \(W\) with an equivariant bilinear intertwiner

\[
\beta:U\otimes V\longrightarrow W,
\]

independent affine scans on \(U\) and \(V\), followed by an affine \(W\) scan
driven pointwise by \(\beta(u_t,v_t)\), form a scan-compatible triangular
construction. An explicit homogeneous lift supplies an independent finite
linear closure, and the SO(3) cross product supplies a nonexceptional control.
The relevant object is a representation quiver or triangular semidirect scan;
Spin(8) triality is an exceptional multiplicity-one instance.

**Important obstruction.** If `W` feeds back into both source streams, generic
polynomial degree doubles every step. No fixed-degree lift stays exact. The
triangular boundary is therefore part of the theorem, not an implementation
detail.

**Required experiment.** Compare against a same-width direct slot memory,
generic bilinear fast weights, Householder/DeltaProduct transport, and a generic
non-triality intertwiner. Triality must win on equivariant transport,
extrapolation, or sample/state efficiency--not merely solve retrieval.

**Possible title.** *Intertwiner SchurScans: Exact Parallel Recurrence with
Equivariant Bilinear Drives*.

### Paper E: Neural discovery followed by exact finite-action compilation

**Supported contribution.** Across the finite-group line, gradient training can
find an approximate representation or orbit code; a subsequent compiler can
recover discrete algebraic structure and retract it to an exact action with
zero long-horizon drift. The strongest versions progressively remove supplied
tables and labeled compiler queries.

**Why it matters.** It separates two problems that ordinary end-to-end training
conflates: finding a useful basin and enforcing exact compositional closure.

**Required before a broad claim.** Catalogue exactly which semantic labels,
endpoint classes, group orders, inverse priors, or orbit assumptions each
compiler receives. Demonstrate at least one family of groups and one unknown
presentation, rather than a single fixed A5/Q8 setup. Exact compilation under a
known finite action prior is not unrestricted semantic discovery.

**Possible title.** *From Approximate Neural Orbits to Exact Finite-State
Actions*.

### Paper F: Center fidelity as a representation-selection theorem

**Clean result.** Sandwich/vector actions quotient out central elements that a
spinor left action can distinguish. Q8 makes the obstruction observable:
capacity absent from the representation cannot be recovered by longer
training, a wider chart, or better optimization.

**Best scope.** This is a compact theoretical and empirical paper or a strong
section in Paper E. Its value is the unusually clean separation of
representational kernel, chart width, and learnability.

### Paper G: Binary triality sensor geometry

**New exact result.** The 24 coordinate probes carry five-bit labels in
`F_2^2 + F_2^3`, and triality support contraction is exactly binary addition.
All 52,752 multiview coordinate sensors of sizes four and five were exhausted.
No four-set identifies; exactly 21,504 five-sets identify, precisely the binary
bases. Binary ranks `3,4,5` give exact stabilizer dimensions `8,3,0`, with
representatives certified as `SU(3), SU(2), trivial`. The earlier 14 Hamming
blocks are the affine planes inside this complete binary geometry.

**Why it matters.** This supplies an exact discrete skeleton for the exceptional
`su(2)` stabilizer locus and links the probe-identifiability problem to the
classical octonion--Hamming--`E8` chain. The code itself is classical; its role
as the complete coordinate failure atlas for triality sensing is the candidate
new result.

**Required before a standalone paper.** Compare carefully with binary spinor
encodings already in the literature, then state the new contribution narrowly
as sensor identifiability and stabilizer classification. Extend or explicitly
separate the arbitrary continuous-probe orbit problem.

**Possible title.** *Binary Matroid Geometry of Spin(8) Triality Sensors*.

## Corrections and claims that did not survive audit

These corrections should remain in the record even if later code is fixed.

### C1. Remaining loss aggregation defects

Several legacy evaluators compute a mean loss for each batch and then average
those means without weighting by the number of tokens. Equal-sized historical
batches are unaffected; unequal tail batches or variable sequence lengths are
biased. Replace these with summed loss divided by the exact token count:

- `src/pdssm_group_actions.py`;
- `src/train_rotor_ssm_torch.py`;
- `src/ga_ssm.py`;
- audit any remaining helper that returns `np.mean(losses)`.

The prefix-product target alignment is correct. Shifting those targets would
introduce an off-by-one error.

### C2. Machine-precision witnesses are not exact certificates

Names such as `exact_four_probe_witness` overstate a construction obtained by
floating-point SVD. Use **machine-precision constructive stabilizer witness**
until a symbolic or exact-number proof is supplied. The new canonical
five-versus-four certificate is such an exact proof, but it does not
retroactively make the earlier random SVD witnesses exact. Apply the same
discipline to numerically extracted A5 irreps and numerical rank claims.

### C3. Rank alone was local; the later orbit theorem supplies the global step

Rank 28 removes infinitesimal slack but does not by itself rule out a finite
global stabilizer. That correction still stands for the original cohort. The
later proof adds full-closure points in every mixed allocation and invokes the
compact principal-orbit theorem; that separate argument, not the old rank
experiment, establishes the open dense globally free strata.

### C4. Active sensing is currently static local design

The implemented oracle optimizes a fixed five-query design before observing
responses. Local Fisher information is action-independent, so local
response-adaptive selection cannot improve that objective. Do not describe the
existing result as a global adaptive policy theorem. Adaptivity may still help
resolve remote modes or finite ambiguities.

### C5. Single-view Spin(8) is equivalent to generic SO(8)

The maintained positive-chiral Lie algebra spans an SO(8) action under a basis
change. A one-view result is evidence for a noncommutative orthogonal
recurrence, not for triality. Triality-specific claims require shared transport
across inequivalent views, the trilinear intertwiner, or a matched multiview
separation.

### C6. Mathematical scan depth is not a production throughput result

The Hillis--Steele reference has logarithmic dependency depth but `O(N log N)`
work, and per-token matrix exponentials are expensive. Do not advertise a
hardware result until a fused `O(N)`-work scan and recurrent decoder are
benchmarked against mature kernels on identical hardware.

### C7. Triality binding is not yet a high-capacity VSA or attention replacement

An eight-dimensional octonionic bind/unbind map is exact for one pair but
superposition crosstalk rises quickly. Multiplicity channels give exact
`H`-slot isolation when addresses are orthogonal; they do not create unlimited
capacity. A same-width direct memory already matches the address-routing gate.
The remaining possible advantage is equivariant transport and structured
binding, not raw slot count.

### C8. Auxiliary channels were not shown to perform geometric error correction

The channel subset audit supports a stable decoder ensemble on several
seed/alphabet combinations. Residual contributions did not reliably track the
anchor channel's geometric defect. Use **distribution-dependent redundant
decoder ensemble**, not **error-correcting code**, unless a future directional
compensation test establishes that mechanism.

### C9. Held-out bigrams answer different questions in different architectures

For a write-free model with one fixed operator per token, unseen-bigram
composition is structurally induced once the token operators are learned. It is
not the same shortcut-freedom falsifier used for contextual write-bearing
models. The meaningful result is that SGD found approximately faithful
per-token operators, not that a fixed-operator architecture invented bigram
composition.

### C10. Functional and raw-homomorphism gates must remain separate

The holonomy cohorts passed useful behavioral accuracy/margin gates but did not
approach the original `1e-3` raw-homomorphism gate. Report the complete gate
hierarchy. Do not turn “8/10 functional” into “8/10 mechanism” by shorthand.

### C11. Grade-specific decay did not break grade preservation

Rotor conjugation remains grade-preserving and grade-diagonal damping commutes
with it. The negative result concerns optimization, anisotropic forgetting, and
the inability of isolated grade reservoirs to exchange useful information--not
loss of grade preservation.

### C12. Coordinatewise decorrelation is exactly false

The rational counterexample falsifies the proposed fixed-coordinate
residual-removal map. It does not falsify the constrained star theorem, the
Cayley-null edge theorem, or every possible invariant-preserving deformation.
The unrestricted Gram--Cayley result remains open.

### C13. Floating-point matrix composition is not exactly associative

The algebraic affine operator is associative. Floating-point evaluation only
approximates that operation and depends slightly on reduction order. Report
measured recurrent/parallel parity, not “exact numerical associativity.”

### C14. Frozen artifacts are not independent reruns

Hashes establish integrity, and exact verifiers can replay mathematical
certificates. Most empirical ten-seed artifacts are not retrained by the unit
test suite. Keep artifact integrity, certificate replay, and experimental
replication as separate provenance layers.

### C15. The original SpinorModel is provenance, not the final mechanism

It is a causal attention model whose generation recomputes context. Ordinary
LayerNorm, componentwise GELU, and averaged attention heads do not preserve a
clean Spin-equivariant interpretation. Preserve it unchanged as the historical
prototype; use the recurrent triality archive for scientific claims.

### C16. The three D4 labels are not interchangeable

The `D4` root system/24-cell and Dynkin type `D4=so(8)` share the classical
triality bridge. The order-eight dihedral benchmark called `D4` does not. Do
not use performance on that finite group as evidence about the 24-cell.

### C17. Equal-five and approximate D-optimality are different problems

The equal balanced five-query sensor is not optimal when arbitrary fractional
weights and support sizes are allowed: its normalized maximum sensitivity is
75, above the Kiefer--Wolfowitz threshold 28, and a rational reweighting
improves it. Uniform mass on an eight-probe basis is globally optimal in that
broader approximate-design domain. The exact five equal-cost pure-query
problem remains open and now has a strict local theorem.

### C18. The 24 sensor projectors are tight but not spectrally optimal

The sensor projectors live in `Gr(7,28)`, not on the 24-cell's `S^3`. Their
sum is tight, and cross-view pairs are isoclinic, but same-view pairs intersect.
The full configuration misses the chordal simplex bound and has spectral
coherence one. The latter proves spectral non-optimality; the former does not
alone prove chordal non-optimality without an attainability result. Moreover,
every orthonormal basis in a view gives the same tight projector sum, so
tightness has continuous non-vertex deformations.

## Next best strategum: compressed researcher handoff

### Target 1 -- classify exceptional nonprincipal five-probe strata

The universal four-probe theorem and mixed five-probe principal strata are now
proved. Next classify exceptional nonprincipal strata using the binary/Hamming
closures as exact normal-form anchors, determine possible finite stabilizers,
and connect orbit type to Cayley/conditioning degeneracy.

**Branch condition:** if exceptional stabilizers are not conjugate to closures
generated by the binary anchors, stop treating the Hamming atlas as a complete
normal-form list and introduce the additional continuous invariants explicitly.

### Target 2 -- certify the reconstructed second-residual bridge

The reconstruction, boundary-kernel audit, and radical elimination are now
complete. Two disjoint grids recover the same 6,664 exact coefficients. The
new edge is locally stable along the complete orthonormal equality line, and
the finite problem reduces exactly to four degree-six sign premises and four
degree-twelve squared margins.

1. Factor the \(y=1\) endpoint and its first inward derivative. It must recover
   the proved one-edge theorem and the local boundary-kernel jet.
2. Factor the independent \(y=0\) endpoint and its first inward derivative.
3. Remove all proved boundary factors before attempting an interior
   certificate.
4. Try Bernstein/Duffy subdivision or constrained SOS on the remaining compact
   polynomial gates. A negative native Bernstein coefficient rejects only that
   certificate basis, not the inequality.
5. Activate the final Cholesky residual only after this family is proved or
   falsified.

**Success condition:** an exact nonnegativity certificate or an exact rational
counterexample for every degree-six and degree-twelve gate in the finite
second-residual family.

### Target 2B -- globalize or falsify the equal-five local optimum

Use the exact seven-dimensional quotient Hessian to define rational coupled
probe curves. Audit finite non-vertex paths, all compact rank-deficient strata,
and exceptional equality components. Any numerical challenger must be
rationalized and checked exactly; absence of a challenger is not a proof.

### Target 3 -- benchmark the now-stated Intertwiner SchurScan theorem

The affine theorem, complete 89-dimensional lift, feedback obstruction, and
SO(3) cross-product control are complete. The scan implementation now also has
a linear-work ordered tree, gradient parity, dense CPU/CUDA timing, and a
replicated long-sequence crossover against the earlier Hillis--Steele tensor
program. These are eager-runtime results, not fused-kernel or model-quality
claims. Next compare triality, SO(3), a learned generic bilinear map, and
same-width direct memory on matched tasks. Spin(8) earns credit only for
equivariant transport, sample/state efficiency, or extrapolation beyond the
universal scan construction.

### Target 4 -- run the first language-shaped gate

Use next-token REPL/state-reveal traces before generic language modeling.
Evaluate sparse reveals, partial observability, overwrites, and long
composition. Record dense length sweeps and error-control diagnostics.

Mandatory matched baselines:

- single-view generic SO(8);
- same-width direct slot memory;
- BD-LRU or another block-dense recurrence;
- SLiCE structured transitions;
- faithful PD-SSM;
- DeltaProduct/Householder;
- Gated DeltaNet-2 or Erase-then-Delta;
- Mamba-3 complex/MIMO where implementation permits.

### Target 5 -- measure error control, not only capacity

For every model and length, log:

- within-symbol state spread;
- between-symbol separation;
- decoder margin;
- distinguishability ratio;
- predicted and observed failure length;
- norm/orthogonality drift;
- recurrent/parallel discrepancy.

This converts the earlier length-selective anomalies into a mechanistic test of
when a formally expressive recurrence becomes unreadable.

### Target 6 -- only then test one triality layer at language-model scale

Insert one coupled multiview triality/Intertwiner SchurScan layer into a strong
recurrent or hybrid backbone. Require a true per-layer recurrent state and a
fused scan. Measure wall-clock throughput, memory, retrieval, perplexity, and
state efficiency. Do not replace the whole backbone until one layer shows a
matched benefit.

## Minimal briefing for the active benchmark researcher

The benchmark should not try to prove “Spin(8) beats everything.” It should
isolate three questions:

1. Does noncommutative orthogonal transport beat diagonal/complex recurrence?
2. Does shared multiview triality beat a parameter-matched generic SO(8)
   multiview control?
3. Does the trilinear intertwiner beat a same-width direct or generic bilinear
   memory on state/sample efficiency or extrapolation?

Report kernel/runtime separately from mathematical scan depth. Keep all
training seeds and dense length sweeps visible. A negative triality-specific
result does not invalidate joint-family retraction, the five-probe geometry,
the Cayley theorems, or the general triangular scan construction.

## Literature anchors for the next researcher

- [Expressive limits of diagonal SSMs](https://openreview.net/forum?id=5bg5Ru5OML)
- [Structured Linear CDEs / SLiCE](https://openreview.net/forum?id=HKDyRDzy1E)
- [Structured sparse transition matrices / PD-SSM](https://papers.neurips.cc/paper_files/paper/2025/file/77b830c18836a9b2e1395a4936dd687a-Paper-Conference.pdf)
- [DeltaProduct](https://arxiv.org/abs/2502.10297)
- [Error-control dynamics in state tracking](https://arxiv.org/abs/2605.07755)
- [Provable spectral representation learning](https://arxiv.org/abs/2606.02993)
- [Equivariance by Contrast](https://openreview.net/forum?id=kvI0QTVRQD)
- [Learning state tracking from code](https://openreview.net/forum?id=ZO92hNK7VC)
- [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)
- [Erase-then-Delta](https://arxiv.org/abs/2606.26560)
- [Mamba-3](https://arxiv.org/abs/2603.15569)
- [M2RNN](https://arxiv.org/abs/2603.14360)
- [Generic stabilizers for simple algebraic groups](https://arxiv.org/abs/2105.09486)
- [Explicit Spin(8) triality and G2](https://arxiv.org/abs/2502.14016)
- [Octonion binding capacity](https://arxiv.org/abs/2204.07186)

## Final claim discipline

The strongest defensible umbrella statement is:

> Joint multiview representation retraction turns locally incomplete
> observations into composable operator families, while triangular equivariant
> intertwiners permit exact constant-state bilinear binding without sacrificing
> associative scan structure.

Do not replace that statement with “geometric algebra is a better SSM” until
the matched architecture and throughput gates actually establish it.
