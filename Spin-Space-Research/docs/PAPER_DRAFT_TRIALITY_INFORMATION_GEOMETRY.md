# Triality Information Geometry: Identifiability, Multiplicity Gauges, and Exact Dirac--Gram Bounds

**Working manuscript — revised 2026-08-07**

> **Scope note, 2026-08-10.** The later hierarchical-memory and co-moving FLA
> campaign is deliberately outside this information-geometry manuscript. It
> neither proves the unrestricted Dirac--Gram inequality nor supplies evidence
> for global sensing optimality. Cite its empirical report separately when
> discussing memory architecture.

## Abstract

This work studies experimental design for a shared unknown
\(\operatorname{Spin}(8)\) action observed through the vector and two
chiral-spinor representations. Every unit probe contributes a rank-seven
orthogonal projector to a 28-dimensional linearized information operator
(equivalently, Fisher information under isotropic Gaussian observation noise).
Four linked results are established. First, in every allocation using at least
two triality representations, five probes have an open dense free stratum,
whereas every four-probe design retains a positive-dimensional stabilizer. Second, the
balanced orthonormal five-probe information operator has fixed invariant blocks
of dimensions \(8+8+8+4\), explaining its determinant \(81/1024\) and its
complete one-parameter Cayley spectrum. Third, repeated probes in one
representation possess an exact multiplicity-space orthogonal gauge: their
summed information depends only on their covariance. Finally, the strengthened
Dirac--Gram determinant inequality on a complete five-variable nonorthogonal
family is proved. For the next six-variable bridge, the analysis establishes
the eight-sector amplitude form, local boundary-kernel stability, and an exact
reduction to degree-six and degree-twelve polynomial gates. A complete 34-leaf
rational-circle triangular Bernstein atlas proves all eight physical margins
nonnegative on that six-variable `h=0` domain. The computational premises of
every theorem-level claim are replayable through exact arithmetic or declared
outward enclosures; the logical inference and certificate replay are stated
separately. CUDA searches are reported only as falsification evidence.

## Status at a glance

**Proved:** the sharp generic four-versus-five multiview sensing boundary; the
balanced Cayley spectrum and invariant blocks; the repeated-view covariance
gauge; the variable-Cayley one-edge Dirac--Gram inequality; local stability of
the second residual edge; its finite radical-to-polynomial reduction; and
global positivity on the complete frozen `h=0` two-edge family.

**Numerically tested:** dense interior and boundary searches found no global
five-query challenger. The earlier finite two-edge search has been superseded
as theorem evidence by the complete atlas certificate.

**Open:** the final residual edge, the unrestricted seven-invariant
Dirac--Gram inequality, and global five-query D-optimality over all allocations
and frames.

## 1. Problem

Let \(G_p^{(r)}\), \(p=1,\ldots,28\), denote the infinitesimal generators of
one of the three real eight-dimensional triality representations

\[
r\in\{V,S^+,S^-\}.
\]

For a probe \(x\), define

\[
J_r(x)_{kp}=(G_p^{(r)}x)_k,
\qquad
P_r(x)=J_r(x)^{\mathsf T}J_r(x).
\]

Given probes \((r_j,x_j)\), their information operator is

\[
I=\sum_jP_{r_j}(x_j).
\]

The central questions are:

1. How many probes identify an unknown shared \(\operatorname{Spin}(8)\) action?
2. Which five-probe design maximizes \(\det I\)?
3. Which coordinate choices are genuine geometry and which are gauge?
4. Can these structures become learnable, scan-compatible memory mechanisms?

## 2. Exact projector geometry

For every unit probe,

\[
P_r(x)^2=P_r(x),\qquad \operatorname{rank}P_r(x)=7.
\]

The same-view and cross-view overlaps are

\[
\operatorname{tr}(P_r(x)P_r(y))
=1+6\langle x,y\rangle^2,
\]

and

\[
\operatorname{tr}(P_r(x)P_s(y))=\frac74,\qquad r\ne s.
\]

Thus correlations within a representation change information quadratically,
whereas different triality views are exactly isoclinic at this contraction
level.

## 3. The sharp five-probe boundary

For every allocation using at least two triality representations, the set of
five-probe tuples with trivial stabilizer is open and dense. Equivalently, a
generic tuple in each such allocation has information rank 28. Every
four-probe tuple has a positive-dimensional stabilizer; on each mixed
four-probe allocation, the principal stabilizer has dimension three. Five
probes confined to one representation also have rank 25.

The local statement is therefore sharp:

\[
\text{four probes: positive-dimensional ambiguity},
\qquad
\text{generic five-probe multiview tuple: trivial stabilizer}.
\]

An explicit integral five-probe tuple has trivial stabilizer in the common
\(\operatorname{Spin}(8)\) action, while a displayed four-probe subtuple has
stabilizer Lie algebra \(\mathfrak{su}(2)\). Exact invariant Jacobians and the
compact principal-orbit theorem then promote the coordinate witnesses to the
open dense statement above; see [Wallach](https://arxiv.org/abs/1811.07195) for
a modern principal-orbit-type treatment and its reduction to compact-group
actions. This does not classify every exceptional
nonprincipal five-probe tuple.

This is an identifiability theorem, not a conditioning theorem. The balanced
sensor is separately distinguished by its information spectrum.

## 4. Exact Cayley spectrum

For an orthonormal balanced allocation \((1,2,2)\) across
\((V,S^+,S^-)\), the information operator depends on the normalized Cayley
coordinate \(c\). Completeness of this one-parameter family is a hybrid proof:
the global cohomogeneity-one classification of oriented four-planes is a
classical input, while the maintained exact calculation proves that the
four-plane stabilizer acts through the full \(SO(4)\), eliminating any extra
continuous invariant of the internal \(2+2\) split. Pair reflections identify
\(c\) with \(-c\), so the information coordinate is \(z=c^2\). See the
[flag-quotient audit](CAYLEY_FLAG_QUOTIENT_AUDIT_2026-08-06.md). A fixed
bivector-basis permutation gives invariant coordinate blocks

\[
I(c)=I_8^{(0)}(c)\oplus I_8^{(1)}(c)
\oplus I_8^{(2)}(c)\oplus I_4(c).
\]

Their determinants are

\[
\frac{1-c^2}{4},\qquad
\frac{(1-c^2)(9-c^2)}{16},\qquad
\frac{(1-c^2)(9-c^2)}{16},\qquad 1.
\]

Hence

\[
\det I(c)=\frac{(1-c^2)^3(9-c^2)^2}{1024}.
\]

At \(c=0\),

\[
\det I=\frac{81}{1024},\qquad
\operatorname{tr}I=35,\qquad
\operatorname{tr}I^{-1}=43.
\]

The full characteristic law yields the stronger exact identities

\[
\operatorname{tr}(I(c)^2)=67,
\]

\[
\operatorname{tr}(I(c)^{-1})
=\frac{11z^2-206z+387}{(1-z)(9-z)},
\qquad z=c^2,
\]

and

\[
\operatorname{tr}(I(c)^{-2})
=\frac{19z^4-76z^3+786z^2+2676z+8883}
{(1-z)^2(9-z)^2}.
\]

The direct moments \(\operatorname{tr}I\) and
\(\operatorname{tr}(I^2)\) are constant, while both inverse moments are
strictly increasing in \(z\). Thus the Cayley-null orbit simultaneously
maximizes the determinant and minimizes the first two inverse spectral moments
within the orthonormal balanced family.

At \(|c|=1\), three information directions disappear. Cayley calibration and
information optimality are therefore different extremal problems: the
calibrated endpoint is maximally special geometrically but singular for this
estimation objective. The Cayley form's unit comass and
\(\operatorname{Spin}(7)\) stabilizer are classical facts proved through
triality by [Katz and Shnider](https://arxiv.org/abs/0801.0283).

The matrix identities, characteristic polynomial, and invariant values are
replayed in
[the Cayley spectrum result](experiments/SPIN8_CAYLEY_SPECTRUM_RESULTS.md) and
[the block theorem](experiments/SPIN8_CAYLEY_BLOCK_THEOREM.md). The associated
exact artifact is `artifacts/spin8_cayley_blocks_20260806.json`, whose
published hash is recorded in `ARTIFACTS.sha256`.
The complete proof and the additional criterion certificate appear in
[Balanced Cayley Information Spectra](manuscripts/CAYLEY_INFORMATION_SPECTRUM.md).

## 5. The strengthened Dirac--Gram inequality

### 5.1 Notation and domain

Let \(X\in\mathbb R^{4\times8}\) have unit-norm rows. On the full-row-rank
interior, define

\[
G=XX^{\mathsf T},\qquad
\Delta=\det G>0,\qquad
c=\frac{\Phi(X)}{\sqrt\Delta},
\]

where \(\Phi(X)\) is the oriented Cayley four-form evaluated on the four rows
of \(X\). Thus \(c\) is its Gram-normalized value. Let

\[
Q=G^{-1/2}X,
\]

so that \(QQ^{\mathsf T}=I_4\). Thus \(Q\) is the orthonormal frame obtained by
removing the Gram distortion from \(X\). The inverse square root is used only
when \(\Delta>0\); no pseudoinverse or regularized inverse is hidden in the
statement.

On the singular boundary \(\Delta=0\), \(Q\) and \(c\) need not be defined.
The certified object there is the polynomial form below, interpreted by its
continuous extension from the full-rank interior.

### 5.2 Inequality and certified families

On the interior, the proposed inequality is

\[
\det I(X)\le \Delta^3\det I(Q).
\]

Equivalently,

\[
1024\Delta^2\det I(X)
\le(\Delta-\Phi^2)^3(9\Delta-\Phi^2)^2.
\]

The theorem is proved on three nested domains:

1. two exact one-correlation slices;
2. the full signed-star subfamily;
3. the complete variable-Cayley one-edge family with four active Gram
   correlations.

The largest proof uses common triality symmetry to reduce orientations to a
\(4\times4\) group-circulant positivity problem, exact reconstruction on disjoint
rational grids, 256 direct exact holdouts, and two boundary-adapted Duffy
charts containing 1,901,250 exact Bernstein controls. The final difficult
boundary layers factor into separately certified nonnegative polynomials.
On the signed-star subfamily, exact Bernstein-support analysis also classifies
the normalized equality set completely: equality occurs precisely at the
Cayley endpoint or on the orthonormal star.

Coordinatewise removal of the remaining correlations is not monotone. An
exact rational counterexample rules out that attractive shortcut.

## 6. Multiplicity-space gauge theorem

If \(m\) probes use one representation and form the rows of \(X\), then

\[
\sum_jP_r(x_j)_{pq}
=\operatorname{tr}\!\left((G_p^{(r)})^{\mathsf T}
G_q^{(r)}X^{\mathsf T}X\right).
\]

Their information contribution depends only on the covariance
\(C=X^{\mathsf T}X\). For every \(U\in O(m)\), replacing \(X\) by \(UX\)
changes nothing.

For two unit probes with correlation \(\rho\), the canonical gauge gives

\[
u=\frac{x+y}{\sqrt2},\qquad
v=\frac{x-y}{\sqrt2},
\]

with

\[
u\perp v,\qquad \|u\|^2=1+\rho,\qquad \|v\|^2=1-\rho.
\]

Correlation between repeated equal-strength probes is therefore exactly an
orthogonal-mode energy imbalance. This quotient is relevant both to sensor
design and to redundant channels in learned triality memories.

## 7. Exact amplitude theorem for the two-edge bridge

The next family has six circle pairs

\[
(a,A),(d,D),(e,E),(g,G),(i,I),(c,s).
\]

An exact twelve-coordinate sign audit combines common triality actions with
the independent sign gauge of each probe. Its induced group has order 512 and
its annihilator contains exactly eight characters. Every determinant sector
therefore has the global form

\[
S_m=s^6M_mH_m(a^2,d^2,e^2,g^2,i^2,c^2),
\]

where \(M_m\) is one fixed character monomial and \(H_m\) is an ordinary
rational polynomial.

The factor \(s^6=(1-c^2)^3\) is proved independently. At both Cayley boundary
branches the symbolic observation Jacobian has rank 25 and nullity 3. Every
maximal minor is order at least three in \(s\); Cauchy--Binet makes the
information determinant order at least six. A two-branch quotient-ring
argument converts the analytic order into exact divisibility.

The same argument has now been checked on all ten branches of

\[
A,D,E,G,I=0.
\]

Every branch has rank 25/nullity 3. Thus

\[
A^6D^6E^6G^6I^6=\Delta^3
\]

divides the raw determinant globally, including the newly activated residual.
The normalized determinant is genuinely polynomial in the circle quotient.

The rank-seven Cauchy--Binet bound then yields conservative residual
multidegrees at most four per squared coordinate. The exact slice audit used
2,736 direct determinants: 144 polynomial slices and all 576 disjoint-node
checks passed. After forced factors are removed, all 126 nontrivial slices are
exact polynomial squares; their root degrees are at most three.

These results prove the amplitude form and degree ceiling. They do not yet
prove positivity of the eight orientation margins.

### 7.1 First exact sector

The smallest proof-safe sector, (110101), was reconstructed independently on
two disjoint \(4^6\) rational grids. Their complete 243-term rational coefficient
maps agree, and 32 fresh points reproduce the original direct determinants
exactly. Its observed multidegree is \((2,2,2,2,1,1)\), below the structural
ceiling.

More unexpectedly, exact division gives

\[
H_{110101}=(1-a^2)Q_{110101},
\]

with a 162-term quotient of multidegree \((1,2,2,2,1,1)\). This converts a
slice-observed boundary multiplicity into a global identity for one complete
six-variable sector. It supplied the prospective template for the complete
shared-grid reconstruction below. No sign claim is made for an individual
sector.

The quotient also obeys a nested boundary law:

\[
Q=Q|_{i^2=c^2=0}+(1-d^2)(1-e^2)(1-g^2)R,
\]

where the base contains 42 terms and \(R\) is a 28-term multilinear polynomial.
Hence the two late chart coordinates enter only through the product of three
intermediate complement energies. This exact serial dependence is the main
structural lead for replacing the remaining flat tensor reconstructions by a
boundary-ideal recursion.

Two exact face restrictions make the recursion visible:

\[
Q|_{d^2=1}=3(a^2-1)(g^2-1)^2,
\qquad
Q|_{g^2=1}=3(a^2-1)(d^2-1)^2(e^2-1)(3e^2+1).
\]

The first is nonpositive and the second nonnegative on the squared-coordinate
cube. Individual sector positivity is therefore false even in this first
closed sector; the correct positivity objects are the eight assembled
orientation margins.

### 7.2 Complete reconstruction and the one-edge bridge

A common proof-safe \(5^6\) grid recovers all eight sectors from the same eight
direct determinants at each point. Two disjoint grids required 250,000 exact
direct determinants and independently produced the same 6,664 nonzero
rational coefficients. Thirty-two fresh rational points supplied another 256
direct determinants and 256 exact sector equalities. All eight endpoint-factor
patterns previously observed on slices divide the global six-variable maps.

After those endpoint factors are removed, every sector obeys

\[
Q_m=Q_m|_{i^2=0}
   +i^2(1-d^2)(1-e^2)(1-g^2)T_m.
\]

The \(i^2=0\) base is the proved variable-Cayley one-edge frame. The new residual
therefore enters the even polynomial core through an exact four-factor flag
ideal.

The orientation algebra simultaneously splits by \(i\) parity. The four even
and four odd characters use the same exact order-four Hadamard matrix \(W\), so

\[
\lambda_{r,+}=W_r(E+O),\qquad
\lambda_{r,-}=W_r(E-O).
\]

Equivalently, the eight margins are the spectra of two commuting \(4\times4\)
group-circulants. At \(i=0\) the odd amplitudes vanish, the matrices coincide,
and their spectrum is exactly the previously proved one-edge spectrum. This
is the decisive structural reduction for the remaining positivity problem.

On the principal orthonormal equality line \(a=d=e=g=0\), the first exact
transverse audit gives

\[
\text{margin}=\frac{(1-c^2)^3i^2}{2}P(i^2,c^2),
\]

where \(P\) has degree \((2,2)\) and all nine native Bernstein coefficients are
strictly positive; their minimum is 103. Thus the added residual strictly
decreases the determinant on this slice away from \(i=0\) and the common
Cayley-rank boundary. Exceptional one-edge equality strata remain to be
classified before this becomes a global perturbative argument.

### 7.3 Boundary-kernel theorem and finite-edge reduction

The subsequent local audit closes the immediate perturbative obstruction along
the complete orthonormal equality line. Write \(z=c^2\in[0,1]\), and let \(i\)
denote the new physical edge coordinate. Each paired orientation channel has
the local expansion

\[
m_{r,\pm}
=\lambda_r\pm i\mu_r+i^2\nu_r+O(i^3),
\qquad r=1,\ldots,4.
\]

Here \(\lambda_r\) is the one-edge margin, \(\mu_r\) is its odd transverse
derivative, and \(\nu_r\) is the leading even curvature. The only nontrivial
quadratic tangent block has determinant

\[
4(z-9)^3(z-1).
\]

It is positive for \(0\leq z<1\). At the sole degenerate endpoint \(z=1\), the
odd derivative vanishes on the tangent kernel and the surviving margin is

\[
64s^2(2-s^2)>0,
\qquad 0<s\leq1.
\]

The apparent quadratic flat direction is therefore lifted positively at
fourth order. This proves local stability along that equality line; it does
not prove positivity at finite edge size throughout the parameter cube.

The finite dependence nevertheless admits an exact radical elimination. Set

\[
x=i^2,\qquad
y=\sqrt{1-x},\qquad
0\leq y\leq1.
\]

Each paired margin can be written as

\[
m_\pm(y)=L(y)\pm\sqrt{1-y^2}\,R(y).
\]

Because the sign premise \(L(y)\geq0\) is retained, both margins are
nonnegative if and only if

\[
L(y)\geq0,
\qquad
S(y):=L(y)^2-(1-y^2)R(y)^2\geq0.
\]

No spurious solutions are introduced by squaring. In every channel,
\(\deg_yL=6\) and \(\deg_yS=12\). A float64 CUDA search tested 851,968 interior
and boundary samples without finding a violation. The algebraic reduction is
proved; global nonnegativity of the resulting eight polynomial families
remains open.

## 8. Connection to sequence models

Three reusable architectural principles emerge.

### Shared-family retraction

Relational constraints must be imposed on an entire token-action or address
family. Independent normalization leaves gauge slack and collisions. Joint
retraction recovered hidden Spin(8) actions and latent permutations in the
associated experiments.

### Multiplicity factoring

An eight-dimensional triality representation is a geometric carrier, not a
high-capacity superposition memory. Capacity belongs in a multiplicity factor
\(\mathbb R^H\otimes\mathbb R^8\). The gauge theorem shows that this factor
also contains an exact orthogonal redundancy that should be quotiented or
regularized.

### Triangular bilinear scans

Triality binding is bilinear. A feed-forward triangular coupling can be lifted
to a finite linear state and evaluated by staged associative scans, retaining
constant recurrent state. Feedback reversals cause unbounded polynomial degree
and are therefore outside the exact finite-scan class.

These principles are mathematical mechanisms. They are not yet evidence that
a \(\operatorname{Spin}(8)\) language model beats strong delta-rule,
fast-weight, or structured state-space baselines.

## 9. Reproducibility standard

Promoted results use:

- exact rational or integer arithmetic for proof signs;
- separate numerical falsifiers that cannot promote a theorem;
- preregistered gates and explicit negative results;
- staged caches for large determinants and Bernstein charts;
- disjoint reconstruction grids and holdouts;
- SHA-256 manifests over published artifacts;
- full foundational regression tests.

The finite `h=0` two-edge frontier is now closed.  Its degree-six and
degree-twelve gates are certified on a complete 34-leaf triangular atlas by
outward-rounded Bernstein enclosures, with every interval-indeterminate
control replayed in exact integer arithmetic.  This is a global sign
certificate on that six-variable family, not a numerical extrapolation.  The
unrestricted theorem still contains the final Cholesky residual (h), which
the atlas does not parameterize.

## 10. Next theorem and next ML gate

The immediate mathematical task is the final-residual bridge.  In the complete
Cholesky chart, the fourth moving probe contains an additional component

\[
x_4=g e_0+G\bigl(h e_1+H(i e_2+I(c e_3+s e_4))\bigr).
\]

Setting (h=0) gives the proved atlas family.  Removing (h) by a
multiplicity-space rotation is not legitimate under the individual unit-probe
constraints, because such a rotation generally changes row norms.  A proof of
the unrestricted inequality must therefore control this transverse parameter
or replace the Cholesky chart by an invariant covariance-orbit reduction that
preserves the feasible set exactly.

The next ML gate is conditional on a matched implementation. It should impose
the discovered multiplicity gauge and shared-family retraction in the
blind-action memory harness, then compare against direct slots, delta-rule
memory, linear attention, and Householder transport at matched recurrent-state
and compute budgets. No performance advantage is implied before that comparison.

## Claim boundary

Proved:

- every four-probe design has positive-dimensional stabilizer, and every mixed
  five-probe allocation has an open dense free stratum;
- exact balanced Cayley spectrum and block mechanism;
- repeated-view covariance gauge;
- variable-Cayley one-edge Dirac--Gram theorem;
- two-edge global amplitude ansatz, \(\Delta^3\) divisibility, and conservative
  degree ceiling;
- complete exact reconstruction of all eight two-edge sectors, their endpoint
  factors, the universal one-edge bridge, and the paired four-block reduction;
- local second-edge stability along the complete orthonormal equality line;
- exact reduction of the finite second edge to four degree-six and four
  degree-twelve polynomial gates;
- a complete 34-leaf exact positivity atlas for every physical margin on the
  frozen `h=0` two-edge family.
- the adjacent endpoint-octet first block and scalar, quadratic, and cubic
  second-block gates, plus the complete ten-face coordinate boundary of the
  final determinant by nine one-mode perfect squares and the \(Z=X^2\)
  endpoint;
- strict positivity of every unique adjacent-octet physical margin on the
  extended core \([1/8,7/8]^5\) by a 2,140-leaf adaptive exact
  diagonal-dominance certificate.

Numerical evidence:

- before exact promotion, no finite `h=0` two-edge violation in 851,968
  float64 interior and boundary samples;
- no global equal-five-query challenger in the recorded finite GPU campaign.

Open:

- the final residual and unrestricted Dirac--Gram inequality;
- the width-\(1/8\) collars between the adjacent endpoint-octet coordinate
  boundary and its proved extended core;
- nonbalanced allocation upper bounds and global five-query D-optimality;
- superiority of the resulting learned memory over matched modern baselines.

## Dependency map

\[
\text{five-probe orbit theorem}
\longrightarrow
\text{balanced Cayley blocks}
\longrightarrow
\text{variable-Cayley one-edge theorem}
\longrightarrow
\text{finite two-edge polynomial gates}.
\]

The arrows record logical reuse, not inevitability. Each later stage imports
the earlier structure but still carries its own independent proof obligation.

## Selected primary context

- [Katz and Shnider, *Cayley 4-form, comass, and triality
  isomorphisms*](https://arxiv.org/abs/0801.0283).
- [McRae, *Exploring Triality Explicitly*](https://arxiv.org/abs/2502.14016).
- [Wallach, *Principal orbit type theorems for reductive algebraic group
  actions and the Kempf--Ness Theorem*](https://arxiv.org/abs/1811.07195).
- [Cohn, Conway, Elkies, and Kumar, *The \(D_4\) root system is not
  universally optimal*](https://arxiv.org/abs/math/0607447), which motivates
  the archive's explicit audit of continuous, nonvertex deformations.

The broader sequence-model and experimental-design comparison is maintained in
the [primary-literature audit](LITERATURE_AUDIT_2026-08-06.md).
