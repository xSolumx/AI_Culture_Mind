# Spin(9) three-spinor quotient-dimension audit

**Claim audit — 2026-08-07**
**Status:** exact dimension bookkeeping plus a literature-backed Grassmannian result

## Why this audit is necessary

Several apparently conflicting dimensions occur naturally in the three-spinor
problem. They describe different spaces. Treating them as interchangeable
would misstate the remaining global proof.

## 1. Ordered unit triples: nine generic quotient dimensions

Three ordered unit spinors lie in \((S^{15})^3\), which has dimension \(45\).
A generic triple has trivial Spin(9) stabilizer, and
\(\dim\operatorname{Spin}(9)=36\). The generic ordered-triple quotient
therefore has dimension

\[
45-36=9.
\]

This is the correct dimension when individual probe labels and their chosen
decomposition are retained.

## 2. Frame operators: eight generic objective dimensions

The information objective depends only on

\[
M=\sum_{r=1}^3s_rs_r^{\mathsf T}.
\]

The smooth rank-three positive-semidefinite trace-three stratum has dimension

\[
16\cdot3-\frac{3(3-1)}2-1=44.
\]

Equivalently, a full-rank \(16\times3\) factor has \(48\) parameters, quotient
by \(O(3)\) removes three, and the trace condition removes one. If the generic
frame stabilizer is finite, the frame-objective quotient has dimension
\(44-36=8\).

The one-dimension drop from nine to eight has a concrete source. For a generic
nonorthogonal frame, the set of unit-column decompositions of one fixed frame
operator is generically one-dimensional: \(O(3)\) contributes three
parameters and the unit-diagonal condition imposes two independent equations.

## 3. Orthonormal triples: a three-dimensional orbit space

For an orthonormal triple, the frame operator is a rank-three projector, so the
domain becomes the Grassmannian

\[
G_3(\mathbb R^{16}),
\qquad \dim G_3(\mathbb R^{16})=3(16-3)=39.
\]

Kollross proves that the spin-representation action of Spin(9) on this
Grassmannian has cohomogeneity three and is not polar. Thus the complete
orthonormal-projector subproblem has three quotient dimensions, not one and
not six. The explicit algebraic curve studies a one-dimensional subset of
this three-dimensional quotient.

Reference: A. Kollross, *A classification of hyperpolar and cohomogeneity one
actions*, Trans. Amer. Math. Soc. **354** (2002), 571–612, Section 2.3.3,
[doi:10.1090/S0002-9947-01-02803-3](https://doi.org/10.1090/S0002-9947-01-02803-3).

## 4. The algebraic candidate is singular

An exact symbolic certificate gives a three-dimensional plane stabilizer on
the entire interior symmetric curve, including \(c=c_\star\). The curve
therefore lies in a higher-symmetry orbit type rather than a generic
Grassmann orbit.

The earlier local Hessian calculation used the **negative** log determinant on
two unit spheres. Its nine positive directions therefore meant nine locally
penalized quotient directions, not nine improving directions. The exact
frame-operator calculation now resolves those directions as
\(V_1\oplus V_5\oplus V_5\) and proves the log-determinant Hessian negative
definite there. This count should not be confused with the Grassmannian
cohomogeneity: one \(V_5\) consists of nonorthogonal frame-spectrum changes.

## 5. Correct hierarchy of remaining proof gates

The global problem should now be split rather than described as one vague
high-dimensional search.

1. **Orthonormality gate.** Prove or disprove that a global exact optimum may
   be chosen with \(M^2=M\). This separates Gram-shape variables from plane
   variables.
2. **Grassmannian gate.** If orthonormality holds, prove the determinant bound
   over the full three-dimensional Spin(9) quotient of
   \(G_3(\mathbb R^{16})\), not only the known curve.
3. **Singular-stratum gate.** Determine the invariant slice at the candidate's
   \(\operatorname{SO}(3)\)-stabilized orbit and certify the determinant gap on
   that slice.
4. **Full frame gate.** If orthonormality fails, work on the generic
   eight-dimensional frame-objective quotient and include the rank-two
   boundary, where the determinant has exact order sixteen.

This audit changes the proof strategy, not the proved determinant formula.

Gate 3 is now locally complete: the exact
[slice theorem](manuscripts/SPIN9_GRASSMANN_SLICE_THEOREM.md) identifies the
normal representation as \(V_1\oplus V_5\), with local invariant coordinates
\(c,\operatorname{tr}(A^2),\operatorname{tr}(A^3)\), and the
[strict local theorem](manuscripts/SPIN9_STRICT_LOCAL_D_OPTIMALITY.md) proves
the complete rank-three Hessian negative modulo Spin(9). A finite-neighbourhood
inequality and the global quotient remain open.
