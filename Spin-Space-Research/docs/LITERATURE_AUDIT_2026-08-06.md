# Primary-literature audit and theorem corrections

**Audit date:** 2026-08-06  
**Scope:** local Spin(8) triality sensing, geometric recurrence, associative
scans, finite-group tracking, and recurrent memory  
**Policy:** primary papers and exact local artifacts take precedence over
reviewer analogy or architecture marketing.

> **2026-08-10 literature addendum.** Native Sparse Attention is relevant here
> as a routing decomposition—compressed coarse summaries, query-dependent
> block selection, and a local window—not as evidence for Schur or triality
> algebra. The local campaign therefore tests an NSA-inspired coarse-to-fine
> router under the same learned router and memory budget; it does not implement
> NSA's token KV cache. A separate co-moving derivation compiles invertibly
> transported delta memory into the official FLA kernels. Both gates now have
> frozen artifacts; end-to-end model and sparse-router comparisons remain open.

## Executive verdict

The archive contains genuine exact results, but the strongest defensible story
is narrower than “Spin(8) is a universally superior sequence model.” The
surviving contributions are:

1. shared-family constraints can remove relational nullspaces that independent
   fitting cannot see;
2. mixed triality sensing has a sharp generic five-probe identifiability
   boundary;
3. the balanced five-query sensor has an exact Cayley spectrum and is now
   proved to be a strict local D-optimum modulo `Spin(8)`;
4. triangular intertwiner recurrences preserve a finite associative lift;
5. spinor actions retain central information that sandwich actions quotient
   away.

At the audit date, global equal-five-query D-optimality, a triality-specific
language-model gain, and production-kernel superiority remained open. The
2026-08-10 addendum closes only the transported memory-core kernel gate.

## D4 and the 24-cell: the caution that changed this audit

Cohn, Conway, Elkies, and Kumar proved that the `D4` roots form the regular
24-cell and a spherical 5-design, yet are not universally optimal. Their
counterexample was not found by checking more vertices: it came from a
continuous deformation, then a rational circle parametrization and exact
polynomial/Sturm verification. They also exhibited a three-parameter family of
24-point 5-design deformations containing `D4`.

Applied here, the lesson is methodological, not a transfer theorem:

- symmetry and exact low moments do not imply optimum;
- vertex enumeration does not audit the continuous manifold;
- numerical discovery should be converted to rational curves or exact local
  certificates before promotion.

The repository now follows that standard. The balanced five-query sensor has a
complete exact tangent Hessian and all 35 one-probe coordinate-circle laws.

## The three meanings of D4

The archive previously risked conflating:

- the rank-four `D4` root system and its 24-cell;
- Dynkin type `D4=so(8)`, whose outer automorphisms give triality;
- the order-eight dihedral benchmark historically labelled `D4`.

Only the first two share the representation-theoretic bridge. The benchmark is
unrelated notation.

The three minuscule weight orbits `8v`, `8s+`, and `8s-` do form the classical
24-cell, and an exact order-three map cycles them. The repository's 24 sensor
objects are different: rank-seven projectors in `Gr(7,28)`.

## Fusion-frame correction and non-vertex deformations

The 24 coordinate projectors form a tight three-colour fusion frame, but the
new 2026 version of Fickus--Iverson--Jasper--Mixon gives the standard tests that
prevent an overclaim:

- same-view sensor subspaces intersect, so the full configuration has spectral
  coherence one;
- its minimum squared chordal distance is `21/4`;
- the chordal simplex bound for `(d,r,n)=(28,7,24)` is `126/23`, which is
  strictly larger.

Therefore the full coloured sensor family is not an EITFF and is not
spectrally optimal as an uncoloured Grassmannian code: generic finite
collections of 7-planes in 28 dimensions are pairwise transverse and have
spectral coherence strictly below one. The configuration also does not attain
the chordal simplex bound. That upper-bound gap alone does **not** prove
chordal non-optimality unless attainability for these parameters is separately
established. What is special is its *coloured triality incidence law*, not a
proved ordinary packing optimum.

Tightness is also non-rigid. Since `P_r(x)` is quadratic in `x`, every
orthonormal basis in a view satisfies

\[
\sum_{j=1}^8P_r(q_j)=2I_{28}.
\]

An exact rational `3/5`--`4/5` basis rotation now certifies a non-coordinate
member. Thus “24 coordinate sensors are tight” must not be read as a vertex
uniqueness theorem.

## Exact designs versus approximate designs

Kiefer--Wolfowitz theory optimizes probability measures or fractional weights
on a continuous design space. It is not automatically a theorem about exactly
five equal-cost observations.

For the normalized balanced design `M=I/5`, the exact maximum sensitivity is
75, above the parameter dimension 28. Therefore it fails the approximate
D-optimality equivalence criterion. Even on the same five support points, the
exact rational weights

\[
\alpha=101/100,\qquad\beta=399/400
\]

increase the determinant. Conversely, uniform mass on one complete
eight-probe basis gives `M=I_28/4` and sensitivity exactly 28 everywhere, so it
is globally D-optimal in the approximate-design domain.

The audited reweighting segment has no hidden boundary winner: its two ends
have exact ranks 25 and 7, with determinant vanishing orders three and 21.

This does **not** refute the exact equal-five-query conjecture. It separates two
different feasible sets that were previously called by one name.

## Continuous equal-five-query audit

The exact product-of-spheres Hessian resolves the immediate non-vertex risk:

\[
\operatorname{spec}\operatorname{Hess}(\log\det I)
=\{0^{28},(-22)^4,(-158/9)^2,(-232/9)^1\}.
\]

The kernel is exactly the rank-28 shared `Spin(8)` orbit. The remaining seven
directions are strictly negative. The design is therefore a strict local
maximum modulo symmetry.

The finite one-query atlas finds 15 exactly flat coordinate circles and 20
strictly decreasing circles. All nonflat orthogonal-replacement endpoints have
rank 25. These exact checks cover arbitrary infinitesimal deformations and a
large structured family of finite deformations, but not every distant coupled
five-probe move.

## Triality and exceptional stabilizers

Katz and Shnider prove that the Cayley form has `Spin(7)` stabilizer. McRae gives
explicit triality bases and the intersection picture producing `G2`. These
sources support the archive's stabilizer interpretation, but the local theorem
does not rely on analogy: the maintained harness checks action ranks, closure,
and compact Lie types directly.

The safe claim is a generic-stabilizer/base-size theorem for the displayed
real triality action. Claims about every exceptional five-probe orbit still
require a full orbit-type classification.

Hiratani--Sompolinsky is also directly relevant to the binding line: its
octonion matrices improve unbinding for a small number of superposed pairs,
but lose their advantage over random quadratic binding at large load. This
supports the archive's exact single-pair and multiplicity-slot claims while
arguing against any unqualified “high-capacity 8D VSA” description.

## Modern SSM and recurrent-memory literature

The current primary literature changes the baseline table more than it changes
the exact mathematics:

- Native Sparse Attention separates coarse compression, selected fine blocks,
  and a local window. Its transferable lesson is hierarchical addressing; it
  neither supplies the local memory algebra nor predicts a triality advantage.
- Shakerinava et al. prove finite-precision limitations for single-layer
  input-dependent complex diagonal SSMs on non-Abelian tracking. This motivates
  noncommutative transitions; it does not prove rotors are the unique solution.
- DeltaProduct uses products of generalized Householder maps and is a mandatory
  noncommutative orthogonal baseline.
- Sequential--Parallel Duality formalizes associative prefix scans as one
  important class, while also studying broader prefix-scannable models. The
  local archive should say *algebraically associative* and continue measuring
  floating-point recurrent/parallel discrepancies.
- p-BIM is the closest current scan-compatible bilinear control for the
  Intertwiner SchurScan claim.
- Gated DeltaNet-2 and Erase-then-Delta independently separate erase and write
  control; both are stronger addressing baselines than an older scalar delta
  rule.
- MuonSSM conditions low-rank writes rather than transition geometry, making it
  a relevant optimization control.
- PD-SSM gives a structured sparse permutation-diagonal construction that can
  emulate arbitrary finite automata with near-minimal state size. It is the
  right discrete-state ceiling, even though its state dimension scales with
  automaton cardinality rather than representation dimension.
- Error-control dynamics distinguishes formal expressivity from readable
  long-horizon state separation. The archive's dense length sweeps should add
  within-state spread, between-state separation, and predicted failure length,
  not just accuracy.
- Mamba-3 is a materially stronger complex/MIMO SSM baseline than the old
  single-channel complex recurrence. M2RNN is a complementary nonlinear
  matrix-state baseline and directly challenges any claim that linear scan
  structure is necessary for state tracking.
- Recent gradient-flow theory proves irrep selection in a two-layer group
  composition setting. That supports the plausibility of learned
  representation discovery while narrowing novelty: the local contribution is
  shared multiview retraction and long compositional transport, not the generic
  observation that neural networks can organize by irreducible representations.
- Recent curriculum work on group problems supports the archive's empirical
  observation that training protocol selects which recurrence algorithm is
  found. It does not turn a successful curriculum into a uniqueness theorem.

## Corrections required in any paper draft

1. Say **strict local equal-five-query optimum modulo Spin(8)**, not global
   optimum.
2. Say **globally optimal approximate design with eight support points** for
   the isotropic measure; keep this separate from the five-query problem.
3. Say **three-colour tight fusion frame**, not spectrally optimal
   Grassmannian packing; state only that the chordal simplex bound is missed.
4. State that its tightness has continuous orthonormal-basis deformations.
5. Keep the Euclidean 24-cell and the `Gr(7,28)` projector family distinct.
6. Do not infer universal optimality from spherical 5-design status.
7. Call scan composition algebraically associative; floating-point parity is a
   measured error.
8. Treat diagonal impossibility results as a baseline requirement, not proof of
   a triality advantage.

## Ordered next strategy

1. The frozen `h=0` two-edge gate is now complete. For the unrestricted gate,
   use the exact sixteen-sector reconstruction rather than regenerating
   determinants. The local obstruction has been isolated at the calibrated
   endpoint: its complete weighted leading form is positive definite. The two
   first-order coupled modes also satisfy an exact global Bernstein inequality
   on \(0\le c^2\le2/3\). The next certificate must combine that low-Cayley
   core, a boundary-adapted endpoint blow-up, and a sign bound for the remaining
   thirteen higher-order sectors. Exact reconstruction, local positivity,
   domain-wide positivity, and CUDA non-violation remain separate layers.
2. For the equal-five-query problem, quotient the 35-dimensional probe tangent
   space by the exact 28-dimensional orbit and use the seven negative Hessian
   eigendirections to construct a rational finite-deformation atlas. Search
   coupled paths and compact boundary strata, then rationalize any challenger.
3. Classify the coloured projector configuration as an association scheme or
   coherent configuration before claiming novelty; ordinary fusion-frame
   spectral packing optimality is already ruled out; chordal optimality is
   still unresolved by the current bound comparison.
4. In ML experiments, compare Intertwiner SchurScan against p-BIM,
   DeltaProduct, direct slots, Gated DeltaNet-2/Erase-then-Delta, and a generic
   bilinear intertwiner at matched state and compute.

There is also a serious analytic alternative to a purely polynomial atlas.
The target is a homogeneous determinant ratio, and the operator-scaling /
Brascamp--Lieb literature supplies geodesically convex determinant-capacity
functionals. The local information map is not yet proved to be a completely
positive map of the whitening Gram matrix, because whitening mixes probes that
belong to different triality views. Therefore operator scaling is a research
route, not a cited proof. The precise next question is whether triality enlarges
the state space so that the mixed-view whitening becomes a positive linear map
with the required determinant weight three. A successful lift would replace a
large box certificate by a structural capacity theorem; failure would explain
why the remaining sector interactions are genuinely exceptional.

## Primary sources

- [Cohn, Conway, Elkies, Kumar: The D4 root system is not universally optimal](https://arxiv.org/abs/math/0607447)
- [Katz, Shnider: Cayley form, comass, and triality isomorphisms](https://arxiv.org/abs/0801.0283)
- [McRae: Exploring Triality Explicitly](https://arxiv.org/abs/2502.14016)
- [Fickus, Iverson, Jasper, Mixon: Totally symmetric Grassmannian codes](https://arxiv.org/abs/2406.19542)
- [Dette et al.: Optimal designs for regression with spherical data](https://arxiv.org/abs/1710.10526)
- [Kiefer, Wolfowitz: Optimum Experimental Designs V](https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Fourth-Berkeley-Symposium-on-Mathematical-Statistics-and/chapter/Optimum-Experimental-Designs-V-with-Applications-to-Systematic-and-Rotatable/bsmsp/1200512174)
- [Shakerinava et al.: Expressive Limits of Diagonal SSMs](https://arxiv.org/abs/2603.01959)
- [Siems et al.: DeltaProduct](https://arxiv.org/abs/2502.10297)
- [Yau et al.: Sequential--Parallel Duality](https://arxiv.org/abs/2506.10918)
- [Fujii, Yamakita: Bilinear Input Modulation for Mamba](https://arxiv.org/abs/2604.17221)
- [Hatamizadeh et al.: Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)
- [Li et al.: Erase-then-Delta Attention](https://arxiv.org/abs/2606.26560)
- [Nguyen et al.: MuonSSM](https://arxiv.org/abs/2606.30461)
- [Zhang et al.: When Does Recurrence Become an Algorithm?](https://arxiv.org/abs/2607.20594)
- [Chung et al.: Error Control Dynamics](https://arxiv.org/abs/2605.07755)
- [He et al.: Provable Spectral Representation Learning](https://arxiv.org/abs/2606.02993)
- [Lahoti et al.: Mamba-3](https://arxiv.org/abs/2603.15569)
- [Mishra et al.: M2RNN](https://arxiv.org/abs/2603.14360)
- [Terzic et al.: Structured Sparse Transition Matrices / PD-SSM](https://papers.neurips.cc/paper_files/paper/2025/file/77b830c18836a9b2e1395a4936dd687a-Paper-Conference.pdf)
- [Garibaldi, Guralnick: Generic Stabilizers for Simple Algebraic Groups](https://arxiv.org/abs/2105.09486)
- [Hiratani, Sompolinsky: Optimal Quadratic Binding](https://arxiv.org/abs/2204.07186)
- [DeepSeek-AI: Native Sparse Attention](https://arxiv.org/abs/2502.11089)
- [Farouki: The Bernstein polynomial basis -- a centennial retrospective](https://doi.org/10.1016/j.cagd.2012.03.001)
- [Garg, Gurvits, Oliveira, Wigderson: Brascamp--Lieb inequalities via operator scaling](https://arxiv.org/abs/1607.06711)
- [Vishnoi, Yildiz: Geodesically convex formulations for the Brascamp--Lieb constant](https://arxiv.org/abs/1804.04051)
- [Heijmans-Kuryatnikova, Vera, Zuluaga: Degree bounds for Positivstellensaetze on semialgebraic sets](https://arxiv.org/abs/2605.15821)

The bibliography is a targeted primary-source audit of the local claims, not a
claim to exhaust every paper containing the words `Spin(8)`, `SSM`, or
`D-optimal`.
