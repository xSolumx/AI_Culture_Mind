# Unrestricted Spin(9) three-spinor falsification screen

**Numerical result — 2026-08-07**
**Status:** completed counterexample search; not a global proof
**Harness:** [spin9_three_spinor_global_screen.py](../../src/spin9_three_spinor_global_screen.py)

## Question

The exact companion theorem maximizes the determinant on one explicit
orthonormal/equiangular curve at

\[
c_\star=\frac{\sqrt{241}-17}{24}.
\]

This screen asks whether unconstrained optimization over all three unit
spinors can find a larger determinant. It removes the orthogonality,
equiangularity, and coordinate-support assumptions used to derive the exact
curve.

## Frozen protocol

- hardware: NVIDIA GeForce RTX 2070 SUPER;
- arithmetic: float64;
- seeds: \(0,\ldots,9\);
- starts per seed: \(32\);
- total independent starts: \(320\);
- optimizer: Adam;
- steps: \(500\);
- learning rate: \(0.025\);
- initialization: canonical NumPy CPU random stream, then transfer to CUDA;
- objective: the full \(36\times36\) information log-determinant;
- constraints: each of the three raw 16-vectors is normalized independently;
- deterministic PyTorch algorithms and a fixed cuBLAS workspace configuration.

No start is initialized on the algebraic curve.

## Result

No counterexample was found. Every one of the 320 full-dimensional starts
converged to the same information orbit:

\[
\log\det I_\star=-26.292671210967853.
\]

The worst absolute final discrepancy from the exact curve value was

\[
3.20\times10^{-14}.
\]

Across all final states:

- the worst spinor-Gram error was \(4.23\times10^{-12}\);
- the worst Hopf-Gram error was \(4.64\times10^{-12}\);
- the recovered common Hopf correlation was \(c_\star\);
- the best initial log-determinants ranged from approximately \(-29.49\) to
  \(-27.70\), so the runs did not begin in an already-solved basin.

This is strong evidence that the algebraic point is not merely an optimum of
the displayed curve. It is still not a proof of global optimality on the
nine-dimensional ordered-triple quotient. Since the objective factors through
the frame operator, the generic objective quotient is eight-dimensional. On
the orthonormal-projector subproblem, the Spin(9) action on
\(G_3(\mathbb R^{16})\) has cohomogeneity three. These are different domains;
see the [dimension audit](../SPIN9_QUOTIENT_DIMENSION_AUDIT_2026-08-07.md).

## Local quotient Hessian

With the first spinor fixed, the remaining two unit spheres have 30 tangent
coordinates. Automatic differentiation at the algebraic point gives:

- constrained gradient norm \(4.97\times10^{-15}\);
- 21 null modes at tolerance \(10^{-8}\);
- 9 positive modes;
- no negative mode;
- smallest positive eigenvalue \(19.1202856586\).

The 21 null modes agree with the dimension of the residual Spin(7) group orbit.
Rotating the second and third probes into one another preserves their frame
operator exactly. Its Hessian residual is \(9.53\times10^{-15}\), and the
audit confirms that this flat direction already lies inside the residual
group orbit; it is not an additional independent null mode.

This is numerical evidence for a strict local maximum modulo symmetry. It is
not used to claim global optimality. The local claim has since been promoted
independently: the
[exact strict local theorem](../manuscripts/SPIN9_STRICT_LOCAL_D_OPTIMALITY.md)
constructs the complete frame-operator quotient Hessian, includes
nonorthogonal spectrum changes and their coupling to the Grassmann slice, and
proves negative definiteness for the log determinant. The global conclusion
of this screen remains numerical only.

## Boundary sensitivity

The harness also approaches a duplicate-spinor boundary through

\[
s_3(\varepsilon)
=\frac{s_1+\varepsilon v}{\lVert s_1+\varepsilon v\rVert}.
\]

Successive measured log--log determinant slopes are

\[
15.9594,\ 15.9961,\ 15.9996,\ 15.99996,\ 15.999996,\ 15.9999999.
\]

The limiting power sixteen is independently proved in the
[frame-operator theorem](../manuscripts/SPIN9_FRAME_OPERATOR_REDUCTION.md):
a generic pair leaves eight null directions, a transverse third probe lifts
their amplitudes linearly, and Cauchy--Binet squares the resulting order-eight
maximal minors. The numerical profile is therefore a regression check of a
known boundary law.

## Claim boundary

The result falsifies neither the candidate nor the local-maximizer
interpretation. It does not establish:

- a global theorem over all triples;
- uniqueness modulo Spin(9) and probe relabelling;
- robustness under noisy observations;
- any advantage for a recurrent or language model.

The next proof target is an invariant description of the rank-three frame
domain modulo Spin(9), followed by a sign certificate for the determinant gap.

## Artifact

The complete per-seed report is
[spin9_three_spinor_global_screen_20260807.json](../../artifacts/spin9_three_spinor_global_screen_20260807.json).
Its SHA-256 digest is
9a6bc0d528ef79953d2fca7cd0c1093bc8da3bde8eb97922bef77149b7954f45.
