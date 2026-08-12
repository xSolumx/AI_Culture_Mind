# Final-Residual Dirac--Gram Gate

**Frozen before the new final-residual experiments — 2026-08-07**
**Status at freeze:** the complete `h=0` two-edge family is proved; the final
Cholesky residual and unrestricted theorem are open.

## Question

In the complete lower-triangular chart, write

\[
\begin{aligned}
x_1&=e_0,\\
x_2&=a e_0+A e_1,\\
x_3&=d e_0+D(e e_1+E e_2),\\
x_4&=g e_0+G\bigl(h e_1+H(i e_2+I(c e_3+s e_4))\bigr),
\end{aligned}
\]

where every displayed lower-case/upper-case pair lies on the unit circle and
the diagonal complements are nonnegative.  The proved atlas is the section
`h=0`.  This gate asks whether the strengthened inequality

\[
\det I(X)\leq \det(XX^{\mathsf T})^3\det I(Q),
\qquad Q=(XX^{\mathsf T})^{-1/2}X,
\]

holds on the complete feasible chart, including both signs of every partial
correlation and singular boundary limits.

## Corrections frozen before looking at results

1. The repeated-view multiplicity action preserves the summed information,
   Gram determinant, Cayley form, and whitened target.  It does **not** permit
   `h` to be deleted while retaining the individual unit-row chart: the
   orthogonalized repeated-view rows generally have squared norms
   (1\pm\langle x_3,x_4\rangle).
2. Failure to find a numerical challenger is not a proof.
3. A Bernstein basis with negative native controls rejects that basis or chart,
   not the inequality.
4. The previous whitening-flow derivative bound is false as a proof route and
   will not be recycled as if it were a theorem.

## Frozen evidence ladder

### Gate A — implementation and section identity

- Direct full-chart evaluation at `h=0` must agree with the maintained
  two-edge determinant evaluator at exact rational-circle points.
- Frame Gram determinant must equal
  (A^2D^2E^2G^2H^2I^2).
- The normalized Cayley coordinate must equal (c).

Failure means an implementation error and blocks every later interpretation.

### Gate B — local transverse necessity

On the complete equality line

\[
a=d=e=g=h=i=0,
\]

the signed first derivative in the physical coordinate `h` must vanish, and
the exact second derivative of the target margin must be nonnegative for every
(c^2\in[0,1]).  Endpoint null directions require the first nonzero higher
jet to have nonnegative coefficient.  This is a local theorem only.

### Gate C — adversarial falsification

The numerical campaign must include:

- uniform interior samples;
- beta-distributed boundary-biased samples;
- explicit coordinate faces and near-singular layers;
- gradient ascent on the log determinant ratio;
- targeted starts near the `h=0` equality set and near every endpoint of the
  Cayley parameter.

Any apparent positive margin above (10^{-9}) must be preserved with its full
frame, Gram spectrum, and objective decomposition, then rationalized and
replayed exactly before it is called a counterexample.

### Gate D — global theorem promotion

Promotion requires a complete exact route over the seventh invariant.  An
acceptable route may be:

- an exact invariant polynomial plus a domain-wide positivity certificate;
- a complete chart atlas with outward enclosures and exact fallbacks;
- or a rigorously proved covariance-orbit reduction whose map preserves the
  complete feasible set and both sides of the inequality.

The following do **not** pass Gate D: finite samples, optimizer convergence,
one-dimensional slices, local Hessians, stored `passed` flags, or a certificate
that omits one sign or singular face.

## Reporting categories

- **Counterexample:** exact feasible witness with strictly positive log ratio.
- **Local theorem:** Gate B passes, but Gate D remains open.
- **Global theorem:** Gate D passes with replayable certificate.
- **Inconclusive:** no counterexample and no global certificate.

No threshold will be weakened after observing the data.
